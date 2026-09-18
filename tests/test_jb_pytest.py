"""
Tests for json_prettifier.

Run with:
    pytest test_json_prettifier.py -v
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from unittest import mock

import pytest

import json_prettifier as jp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_json(tmp_path: Path):
    """Factory: write text to a file under tmp_path and return its Path."""

    def _write(name: str, content: str) -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    return _write


@pytest.fixture
def sample_obj():
    return {"b": 2, "a": [1, 2, {"z": "y"}], "n": None, "s": "héllo"}


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class TestBackendDetection:
    def test_backend_is_always_set(self):
        assert jp._BACKEND is not None
        assert isinstance(jp._BACKEND, jp._Backend)
        assert jp._BACKEND.name

    def test_stdlib_backend_roundtrip(self):
        b = jp._StdlibBackend()
        assert b.loads('{"a":1}') == {"a": 1}
        assert b.dumps({"a": 1}) == '{"a": 1}'
        assert b.dumps({"a": 1}, compact=True) == '{"a":1}'
        assert b.dumps({"b": 2, "a": 1}, sort_keys=True) == '{"a": 1, "b": 2}'
        assert b.dumps({"a": [1]}, indent=2) == '{\n  "a": [\n    1\n  ]\n}'

    def test_stdlib_backend_preserves_unicode(self):
        b = jp._StdlibBackend()
        assert b.dumps({"s": "héllo"}) == '{"s": "héllo"}'
        assert b.dumps({"s": "héllo"}, compact=True) == '{"s":"héllo"}'

    def test_msgpack_backend_delegates_dumps_to_stdlib(self):
        # msgpack can't emit JSON, so dumps must still produce valid JSON.
        b = jp._MsgpackBackend()
        out = b.dumps({"a": 1}, compact=True)
        assert json.loads(out) == {"a": 1}

    def test_detect_backend_priority(self):
        """orjson wins over ujson wins over msgpack wins over stdlib."""
        # Simulate all imports failing → stdlib.
        with mock.patch.dict(sys.modules, {"orjson": None, "ujson": None, "msgpack": None}):
            # None in sys.modules makes `import x` raise ImportError.
            backend = jp._detect_backend()
            assert isinstance(backend, jp._StdlibBackend)

    def test_detect_backend_falls_through_to_stdlib(self):
        with mock.patch.object(jp._OrjsonBackend, "__init__", side_effect=ImportError):
            with mock.patch.object(jp._UjsonBackend, "__init__", side_effect=ImportError):
                with mock.patch.object(jp._MsgpackBackend, "__init__", side_effect=ImportError):
                    assert isinstance(jp._detect_backend(), jp._StdlibBackend)


# ---------------------------------------------------------------------------
# custom_format
# ---------------------------------------------------------------------------


class TestCustomFormat:
    def test_top_level_dict_exploded(self):
        out = jp.custom_format({"a": 1, "b": 2})
        lines = out.split("\n")
        assert lines[0] == "{"
        assert lines[-1] == "}"
        assert len(lines) == 4  # {, two items, }

    def test_nested_values_stay_inline(self):
        out = jp.custom_format({"a": {"x": 1, "y": 2}, "b": [1, 2, 3]})
        assert out.count("\n") == 3  # only top-level items on their own lines
        assert '{"x":1,"y":2}' in out or '{"x": 1, "y": 2}' in out

    def test_top_level_list_exploded(self):
        out = jp.custom_format([1, 2, 3])
        assert out == "[\n  1,\n  2,\n  3\n]"

    def test_empty_dict(self):
        assert jp.custom_format({}) == "{}"

    def test_empty_list(self):
        assert jp.custom_format([]) == "[]"

    def test_scalar(self):
        assert jp.custom_format(42) == "42"
        assert jp.custom_format(None) == "null"
        assert jp.custom_format("hi") == '"hi"'

    def test_sort_keys(self):
        out = jp.custom_format({"b": 1, "a": 2}, sort_keys=True)
        assert out.index('"a"') < out.index('"b"')

    def test_unicode_preserved(self):
        out = jp.custom_format({"s": "héllo"})
        assert "héllo" in out
        assert "\\u" not in out


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


class TestReadText:
    def test_reads_small_file(self, tmp_json):
        p = tmp_json("small.json", '{"a":1}')
        assert jp._read_text(p) == '{"a":1}'

    def test_reads_via_mmap_for_large_files(self, tmp_json, monkeypatch):
        # Force the mmap branch with a tiny threshold.
        monkeypatch.setattr(jp, "MMAP_THRESHOLD", 4)
        p = tmp_json("big.json", '{"a":1,"b":2}')
        assert jp._read_text(p) == '{"a":1,"b":2}'

    def test_utf8_decode(self, tmp_json, monkeypatch):
        monkeypatch.setattr(jp, "MMAP_THRESHOLD", 1)
        p = tmp_json("u.json", '{"s":"héllo"}')
        assert jp._read_text(p) == '{"s":"héllo"}'


class TestAtomicWrite:
    def test_writes_content(self, tmp_json):
        p = tmp_json("f.json", "old")
        jp._atomic_write(p, "new")
        assert p.read_text() == "new"

    def test_preserves_permissions(self, tmp_json):
        p = tmp_json("f.json", "old")
        os.chmod(p, 0o600)
        jp._atomic_write(p, "new")
        assert stat.S_IMODE(p.stat().st_mode) == 0o600

    def test_no_leftover_tmp_files(self, tmp_json, tmp_path):
        p = tmp_json("f.json", "old")
        jp._atomic_write(p, "new")
        leftovers = [f for f in tmp_path.iterdir() if f.name.endswith(".tmp")]
        assert leftovers == []

    def test_cleans_up_on_failure(self, tmp_json, tmp_path):
        p = tmp_json("f.json", "old")
        # Make os.replace blow up after the tmp file is created.
        with mock.patch("json_prettifier.os.replace", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                jp._atomic_write(p, "new")
        assert p.read_text() == "old"
        leftovers = [f for f in tmp_path.iterdir() if f.name.endswith(".tmp")]
        assert leftovers == []


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class TestProcessJsonFile:
    def test_minify(self, tmp_json):
        p = tmp_json("f.json", '{\n  "a": 1,\n  "b": 2\n}')
        path, ok, err = jp.process_json_file((str(p), True, False, False))
        assert ok and err is None
        assert p.read_text() == '{"a":1,"b":2}\n'

    def test_prettify_standard(self, tmp_json):
        p = tmp_json("f.json", '{"a":1}')
        path, ok, err = jp.process_json_file((str(p), False, False, False))
        assert ok
        assert p.read_text() == '{\n  "a": 1\n}\n'

    def test_prettify_custom(self, tmp_json):
        p = tmp_json("f.json", '{"a":{"x":1,"y":2},"b":2}')
        path, ok, err = jp.process_json_file((str(p), False, False, True))
        assert ok
        text = p.read_text()
        # Only 3 newlines inside the object (two top-level items + closing).
        assert text.count("\n") == 3
        assert text.endswith("\n")

    def test_sort_keys(self, tmp_json):
        p = tmp_json("f.json", '{"b":2,"a":1}')
        path, ok, err = jp.process_json_file((str(p), True, True, False))
        assert ok
        assert p.read_text() == '{"a":1,"b":2}\n'

    def test_invalid_json(self, tmp_json):
        p = tmp_json("bad.json", "{not json")
        path, ok, err = jp.process_json_file((str(p), False, False, False))
        assert not ok
        assert "Invalid JSON" in err
        assert p.read_text() == "{not json"  # untouched

    def test_missing_file(self, tmp_path):
        p = tmp_path / "nope.json"
        path, ok, err = jp.process_json_file((str(p), True, False, False))
        assert not ok
        assert "I/O error" in err or "Error" in err

    def test_trailing_newline_always(self, tmp_json):
        p = tmp_json("f.json", '{"a":1}')
        jp.process_json_file((str(p), False, False, False))
        assert p.read_text().endswith("\n")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_json_extension(self, tmp_json):
        p = tmp_json("a.json", "{}")
        assert jp.is_json_file(p)

    def test_extension_case_insensitive(self, tmp_json):
        p = tmp_json("a.JSON", "{}")
        assert jp.is_json_file(p)

    def test_no_extension_but_starts_with_brace(self, tmp_json):
        p = tmp_json(".prettierrc", '{"semi":false}')
        assert jp.is_json_file(p)

    def test_no_extension_and_not_json(self, tmp_json):
        p = tmp_json("notes.txt", "hello")
        assert not jp.is_json_file(p)

    def test_collect_from_dir_recursive(self, tmp_path):
        (tmp_path / "a.json").write_text("{}")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.json").write_text("{}")
        (sub / "skip.txt").write_text("nope")

        found = jp.collect_json_files([str(tmp_path)], recursive=True)
        names = {f.name for f in found}
        assert names == {"a.json", "b.json"}

    def test_collect_from_dir_non_recursive(self, tmp_path):
        (tmp_path / "a.json").write_text("{}")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.json").write_text("{}")

        found = jp.collect_json_files([str(tmp_path)], recursive=False)
        assert {f.name for f in found} == {"a.json"}

    def test_explicit_file_always_included(self, tmp_json):
        # Even if it doesn't look like JSON, an explicit path is used.
        p = tmp_json("weird.dat", "[1,2,3]")
        found = jp.collect_json_files([str(p)])
        assert p in found

    def test_dotfile_with_json_content_in_dir(self, tmp_path):
        (tmp_path / ".prettierrc").write_text('{"semi":false}')
        found = jp.collect_json_files([str(tmp_path)])
        assert {f.name for f in found} == {".prettierrc"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_cli(argv, monkeypatch):
    """Run main() with a single-worker pool so we don't fork in tests."""
    # Force multiprocessing to run inline.
    import multiprocessing.pool

    class _InlinePool:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def map(self, fn, iterable): return [fn(x) for x in iterable]

    monkeypatch.setattr(jp, "Pool", _InlinePool)
    return jp.main(argv)


class TestCli:
    def test_minify(self, tmp_json, monkeypatch, capsys):
        p = tmp_json("f.json", '{\n  "a": 1\n}')
        rc = _run_cli(["-m", str(p)], monkeypatch)
        assert rc == 0
        assert p.read_text() == '{"a":1}\n'

    def test_default_is_prettify(self, tmp_json, monkeypatch):
        p = tmp_json("f.json", '{"a":1}')
        rc = _run_cli([str(p)], monkeypatch)
        assert rc == 0
        assert p.read_text() == '{\n  "a": 1\n}\n'

    def test_custom_flag(self, tmp_json, monkeypatch):
        p = tmp_json("f.json", '{"a":{"x":1,"y":2},"b":2}')
        rc = _run_cli(["-c", str(p)], monkeypatch)
        assert rc == 0
        # Custom: top-level items on separate lines only.
        assert p.read_text().count("\n") == 3

    def test_custom_ignored_with_minify(self, tmp_json, monkeypatch, capsys):
        p = tmp_json("f.json", '{"a":1,"b":2}')
        rc = _run_cli(["-m", "-c", str(p)], monkeypatch)
        assert rc == 0
        assert p.read_text() == '{"a":1,"b":2}\n'
        err = capsys.readouterr().err
        assert "custom" in err.lower()

    def test_sort_keys(self, tmp_json, monkeypatch):
        p = tmp_json("f.json", '{"b":2,"a":1}')
        rc = _run_cli(["-m", "-s", str(p)], monkeypatch)
        assert rc == 0
        assert p.read_text() == '{"a":1,"b":2}\n'

    def test_beautify_and_minify_mutually_exclusive(self, tmp_json, monkeypatch):
        p = tmp_json("f.json", "{}")
        with pytest.raises(SystemExit):
            jp.main(["-b", "-m", str(p)])

    def test_no_files_found(self, tmp_path, monkeypatch, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        rc = _run_cli([str(empty)], monkeypatch)
        assert rc == 1
        assert "No JSON files found" in capsys.readouterr().err

    def test_error_returns_nonzero(self, tmp_json, monkeypatch, capsys):
        p = tmp_json("bad.json", "{not json")
        rc = _run_cli([str(p)], monkeypatch)
        assert rc == 1
        assert "Invalid JSON" in capsys.readouterr().err

    def test_multiple_inputs(self, tmp_json, tmp_path, monkeypatch):
        a = tmp_json("a.json", '{"x":1}')
        d = tmp_path / "d"
        d.mkdir()
        b = d / "b.json"
        b.write_text('{"y":2}')
        rc = _run_cli(["-m", str(a), str(d)], monkeypatch)
        assert rc == 0
        assert a.read_text() == '{"x":1}\n'
        assert b.read_text() == '{"y":2}\n'

    def test_prints_backend_name(self, tmp_json, monkeypatch, capsys):
        p = tmp_json("f.json", "{}")
        _run_cli([str(p)], monkeypatch)
        assert "backend" in capsys.readouterr().out.lower()

    def test_workers_capped_to_file_count(self, tmp_json, monkeypatch):
        p = tmp_json("f.json", "{}")
        rc = _run_cli(["--workers", "999", str(p)], monkeypatch)
        assert rc == 0

    def test_workers_never_below_one(self, tmp_json, monkeypatch):
        p = tmp_json("f.json", "{}")
        rc = _run_cli(["--workers", "0", str(p)], monkeypatch)
        assert rc == 0