"""
jb — JSON beautifier / minifier with multiprocessing.

Reads JSON files (or directories of them), then either pretty-prints them
or minifies them in place.

Backend priority (auto-detected at import time):
    orjson  >  ujson  >  msgpack  >  stdlib json
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import sys
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Files larger than this are memory-mapped to avoid a redundant userspace copy.
MMAP_THRESHOLD = 1024 * 1024  # 1 MiB

# Extensions that are always treated as JSON even without a leading '{'.
JSON_EXTENSIONS = frozenset({".json"})

# Indentation used for standard pretty-printing (non-custom beautify mode).
PRETTY_INDENT = 2


# ---------------------------------------------------------------------------
# JSON backend selection
# ---------------------------------------------------------------------------


class _Backend:
    """Uniform interface over the available JSON libraries."""

    name: str = "json"

    def loads(self, text: str) -> Any:
        raise NotImplementedError

    def dumps(
        self,
        obj: Any,
        *,
        sort_keys: bool = False,
        compact: bool = False,
        indent: int | None = None,
    ) -> str:
        """
        Serialize ``obj`` to JSON text. Three mutually-exclusive modes:

        * ``indent=N``    — multi-line, N-space indented pretty output.
        * ``compact=True`` — single line, no whitespace.
        * neither         — single line, default separators (``", "`` / ``": "``).
        """
        raise NotImplementedError


class _StdlibBackend(_Backend):
    name = "json (stdlib)"

    def loads(self, text: str) -> Any:
        return json.loads(text)

    def dumps(self, obj, *, sort_keys=False, compact=False, indent=None) -> str:
        if indent is not None:
            return json.dumps(obj, ensure_ascii=False, sort_keys=sort_keys, indent=indent)
        if compact:
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys)
        return json.dumps(obj, ensure_ascii=False, sort_keys=sort_keys)


class _OrjsonBackend(_Backend):
    name = "orjson"

    def __init__(self) -> None:
        import orjson  # noqa: WPS433 (deferred import)

        self._orjson = orjson

    def loads(self, text: str) -> Any:
        return self._orjson.loads(text)

    def dumps(self, obj, *, sort_keys=False, compact=False, indent=None) -> str:
        opt = self._orjson.OPT_SORT_KEYS if sort_keys else 0
        if indent is not None:
            # orjson only supports a 2-space indent; anything else is
            # silently coerced to OPT_INDENT_2.
            opt |= self._orjson.OPT_INDENT_2
        # Without OPT_INDENT_2, orjson is always compact. That means the
        # "neither" mode also produces compact output — a minor cosmetic
        # divergence from stdlib's `", "` / `": "` defaults.
        return self._orjson.dumps(obj, option=opt).decode("utf-8")


class _UjsonBackend(_Backend):
    name = "ujson"

    def __init__(self) -> None:
        import ujson  # noqa: WPS433

        self._ujson = ujson

    def loads(self, text: str) -> Any:
        return self._ujson.loads(text)

    def dumps(self, obj, *, sort_keys=False, compact=False, indent=None) -> str:
        kwargs: dict[str, Any] = {"ensure_ascii": False, "sort_keys": sort_keys}
        if indent is not None:
            kwargs["indent"] = indent
        elif compact:
            kwargs["separators"] = (",", ":")
        return self._ujson.dumps(obj, **kwargs)


class _MsgpackBackend(_Backend):
    """
    msgpack is a *binary* format — it cannot emit JSON text. To keep the
    tool's output valid JSON, serialization delegates to the stdlib json
    module. Deserialization tries msgpack first, then falls back to
    parsing the bytes as JSON text.
    """

    name = "msgpack"

    def __init__(self) -> None:
        import msgpack  # noqa: WPS433

        self._msgpack = msgpack

    def loads(self, text: str) -> Any:
        data = text.encode("utf-8") if isinstance(text, str) else text
        try:
            return self._msgpack.unpackb(data, raw=False)
        except Exception:
            return json.loads(data.decode("utf-8") if isinstance(data, bytes) else data)

    def dumps(self, obj, *, sort_keys=False, compact=False, indent=None) -> str:
        # msgpack can't emit JSON — delegate to stdlib for valid JSON output.
        if indent is not None:
            return json.dumps(obj, ensure_ascii=False, sort_keys=sort_keys, indent=indent)
        if compact:
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys)
        return json.dumps(obj, ensure_ascii=False, sort_keys=sort_keys)


def _detect_backend() -> _Backend:
    for cls in (_OrjsonBackend, _UjsonBackend, _MsgpackBackend):
        try:
            return cls()
        except ImportError:
            continue
    return _StdlibBackend()


_BACKEND: _Backend = _detect_backend()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def custom_format(data: Any, sort_keys: bool = False) -> str:
    """
    Render ``data`` with only the *top-level* container exploded onto
    multiple lines. Nested structures stay on a single line.

    Uses the active backend for serializing each individual element.
    """
    if isinstance(data, dict) and data:
        items = sorted(data.items()) if sort_keys else data.items()
        inner = ",\n  ".join(
            f"{_BACKEND.dumps(k, sort_keys=sort_keys)}: {_BACKEND.dumps(v, sort_keys=sort_keys)}" for k, v in items
        )
        return "{\n  " + inner + "\n}"

    if isinstance(data, list) and data:
        inner = ",\n  ".join(_BACKEND.dumps(v, sort_keys=sort_keys) for v in data)
        return "[\n  " + inner + "\n]"

    # Scalars, empty dicts/lists, or None.
    return _BACKEND.dumps(data, sort_keys=sort_keys)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    """Read a file as UTF-8, memory-mapping very large files."""
    if path.stat().st_size > MMAP_THRESHOLD:
        with path.open("rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return mm.read().decode("utf-8")
    return path.read_text(encoding="utf-8")


def _atomic_write(path: Path, text: str) -> None:
    """
    Write ``text`` to ``path`` atomically.

    Writes to a sibling temp file and then ``os.replace``s it, so a crash
    or power loss mid-write never leaves the original file corrupted. The
    original file's permission bits are preserved.
    """
    original_mode = path.stat().st_mode

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp_path, original_mode)
        os.replace(tmp_path, path)
    except BaseException:
        # Best-effort cleanup if anything goes wrong.
        tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def process_json_file(
    task: tuple[str, bool, bool, bool],
) -> tuple[str, bool, str | None]:
    """
    Worker entrypoint. Returns ``(path, success, error_message_or_None)``.

    ``task`` is ``(path, minify, sort_keys, custom)``.
    """
    file_path_str, minify, sort_keys, custom = task
    file_path = Path(file_path_str)

    try:
        data = _BACKEND.loads(_read_text(file_path))

        if minify:
            output = _BACKEND.dumps(data, sort_keys=sort_keys, compact=True)
        elif custom:
            output = custom_format(data, sort_keys=sort_keys)
        else:
            output = _BACKEND.dumps(data, sort_keys=sort_keys, indent=PRETTY_INDENT)

        # Always terminate with a trailing newline (POSIX convention).
        _atomic_write(file_path, output + "\n")
        return (file_path_str, True, None)

    except (json.JSONDecodeError, ValueError) as e:
        # orjson.JSONDecodeError subclasses json.JSONDecodeError;
        # ujson raises ValueError for malformed input.
        return (file_path_str, False, f"Invalid JSON: {e}")
    except OSError as e:
        return (file_path_str, False, f"I/O error: {e}")
    except Exception as e:  # pragma: no cover — defensive
        return (file_path_str, False, f"Error: {e}")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def is_json_file(path: Path) -> bool:
    """
    True if ``path`` looks like JSON: either the extension matches or the
    first byte is ``{`` (fast path skips the open for ``*.json``).
    """
    if path.suffix.lower() in JSON_EXTENSIONS:
        return True
    try:
        with path.open("rb") as f:
            return f.read(1) == b"{"
    except OSError:
        return False


def collect_json_files(paths: list[str], recursive: bool = True) -> set[Path]:
    """Return the set of JSON-like files reachable from ``paths``."""
    found: set[Path] = set()
    glob_pat = "**/*" if recursive else "*"

    for raw in paths:
        p = Path(raw)
        if p.is_file():
            # Explicitly named files are always processed.
            found.add(p)
        elif p.is_dir():
            for child in p.glob(glob_pat):
                if child.is_file() and is_json_file(child):
                    found.add(child)

    return found


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prettify or minify JSON files with multiprocessing support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  jb
  jb file.json
  jb .prettierrc
  jb -m file.json
  jb -c file.json
  jb -s -c dir/
  jb -m -s file.json
  jb file1.json dir1/
        """,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-b",
        "--beautify",
        action="store_true",
        help="Beautify/prettify JSON (default)",
    )
    mode_group.add_argument(
        "-m",
        "--minify",
        action="store_true",
        help="Minify JSON (remove whitespace)",
    )

    parser.add_argument(
        "-c",
        "--custom",
        action="store_true",
        help="Use the custom formatter: only the top-level container is "
        "exploded across lines; nested values stay inline. "
        "Implies beautify; ignored when --minify is set.",
    )
    parser.add_argument(
        "-s",
        "--sort-keys",
        action="store_true",
        help="Sort object keys alphabetically (default: off)",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=None,
        help="JSON files or directories (default: current directory)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of worker processes (default: CPU count = {cpu_count()})",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    paths = args.paths or ["."]
    minify = args.minify
    # Custom formatting is a beautify-mode modifier; minify takes precedence.
    custom = args.custom and not minify

    if args.custom and minify:
        print(
            "Note: --custom has no effect with --minify; ignoring it.",
            file=sys.stderr,
        )

    json_files = collect_json_files(paths, recursive=True)
    if not json_files:
        print("No JSON files found.", file=sys.stderr)
        return 1

    # Sorting gives deterministic, reproducible output ordering.
    process_args = [(str(f), minify, args.sort_keys, custom) for f in sorted(json_files)]

    # Cap workers so we never spawn more than we have work for.
    workers = min(args.workers or cpu_count(), len(process_args))
    workers = max(workers, 1)

    if minify:
        mode_text = "Minifying"
    elif custom:
        mode_text = "Custom-formatting"
    else:
        mode_text = "Prettifying"
    sort_text = " with sorted keys" if args.sort_keys else ""

    print(f"Using JSON backend: {_BACKEND.name}")
    print(f"Processing {len(json_files)} JSON file(s) using {workers} worker(s)...")
    print(f"{mode_text}{sort_text}...")

    with Pool(processes=workers) as pool:
        results = pool.map(process_json_file, process_args)

    success_count = 0
    error_count = 0
    for file_path, ok, error in results:
        if ok:
            success_count += 1
        else:
            error_count += 1
            print(f"✗ {file_path}: {error}", file=sys.stderr)

    print(f"\n✓ Successfully processed: {success_count} file(s)")
    if error_count:
        print(f"✗ Failed: {error_count} file(s)")
        return 1

    print("All files processed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
