"""
Tests for jb.py — JSON beautifier / minifier.

Run with:
    python -m unittest test_jb -v
"""

import io
import json
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import json_prettifier as jb


# ---------------------------------------------------------------------------
# Base class: every test gets its own temp directory
# ---------------------------------------------------------------------------

class FileTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, name: str, content: str) -> Path:
        """Create ``name`` inside the temp dir with ``content`` and return it."""
        p = self.tmp / name
        p.write_text(content, encoding="utf-8")
        return p


# ---------------------------------------------------------------------------
# custom_format
# ---------------------------------------------------------------------------

class TestCustomFormat(unittest.TestCase):
    def test_empty_dict(self):
        self.assertEqual(jb.custom_format({}), "{}")

    def test_empty_list(self):
        self.assertEqual(jb.custom_format([]), "[]")

    def test_scalars(self):
        self.assertEqual(jb.custom_format(42), "42")
        self.assertEqual(jb.custom_format(None), "null")
        self.assertEqual(jb.custom_format(True), "true")

    def test_string(self):
        self.assertEqual(jb.custom_format("hi"), '"hi"')

    def test_dict_preserves_insertion_order(self):
        out = jb.custom_format({"b": 1, "a": 2})
        self.assertEqual(out, '{\n  "b": 1,\n  "a": 2\n}')

    def test_dict_sort_keys(self):
        out = jb.custom_format({"b": 1, "a": 2}, sort_keys=True)
        self.assertEqual(out, '{\n  "a": 2,\n  "b": 1\n}')

    def test_nested_dict_stays_compact(self):
        # Only the *top* level is exploded; children are dumped compactly.
        out = jb.custom_format({"a": {"x": 1, "y": 2}})
        self.assertEqual(out, '{\n  "a": {"x": 1, "y": 2}\n}')

    def test_nested_sort_keys_propagates(self):
        out = jb.custom_format({"a": {"y": 1, "x": 2}}, sort_keys=True)
        self.assertEqual(out, '{\n  "a": {"x": 2, "y": 1}\n}')

    def test_list(self):
        self.assertEqual(jb.custom_format([1, 2, 3]), "[\n  1,\n  2,\n  3\n]")

    def test_list_of_dicts(self):
        out = jb.custom_format([{"a": 1}, {"b": 2}])
        self.assertEqual(out, '[\n  {"a": 1},\n  {"b": 2}\n]')

    def test_unicode_not_escaped(self):
        self.assertIn("café", jb.custom_format({"k": "café"}))
        self.assertIn("café", jb.custom_format(["café"]))

    def test_output_is_valid_json(self):
        # Round-trip through json.loads to prove we didn't corrupt anything.
        data = {"b": [1, 2, {"z": "x"}], "a": None}
        self.assertEqual(json.loads(jb.custom_format(data)), data)


# ---------------------------------------------------------------------------
# _read_text
# ---------------------------------------------------------------------------

class TestReadText(FileTestCase):
    def test_small_file(self):
        p = self.write("small.json", '{"a": 1}')
        self.assertEqual(jb._read_text(p), '{"a": 1}')

    def test_large_file_mmap_path(self):
        # Force the mmap branch by lowering the threshold.
        content = "x" * 200
        p = self.write("big.json", content)
        with mock.patch.object(jb, "MMAP_THRESHOLD", 10):
            self.assertEqual(jb._read_text(p), content)

    def test_utf8_decoding(self):
        p = self.write("u.json", '{"k": "café ☕"}')
        self.assertEqual(jb._read_text(p), '{"k": "café ☕"}')


# ---------------------------------------------------------------------------
# _atomic_write
# ---------------------------------------------------------------------------

class TestAtomicWrite(FileTestCase):
    def test_overwrites_content(self):
        p = self.write("a.json", "old")
        jb._atomic_write(p, "new")
        self.assertEqual(p.read_text(), "new")

    def test_preserves_permissions(self):
        p = self.write("a.json", "x")
        os.chmod(p, 0o640)
        jb._atomic_write(p, "y")
        self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o640)

    def test_leaves_no_temp_files(self):
        p = self.write("a.json", "x")
        jb._atomic_write(p, "y")
        leftovers = [f.name for f in self.tmp.iterdir() if f.name != "a.json"]
        self.assertEqual(leftovers, [], f"unexpected leftovers: {leftovers}")

    def test_cleans_up_temp_on_failure(self):
        p = self.write("a.json", "original")
        # Make os.replace blow up so we exercise the cleanup path.
        with mock.patch("jb.os.replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                jb._atomic_write(p, "new")
        # Original untouched, no temp files left behind.
        self.assertEqual(p.read_text(), "original")
        leftovers = [f.name for f in self.tmp.iterdir() if f.name != "a.json"]
        self.assertEqual(leftovers, [])


# ---------------------------------------------------------------------------
# process_json_file
# ---------------------------------------------------------------------------

class TestProcessJsonFile(FileTestCase):
    def test_beautify(self):
        p = self.write("a.json", '{"b": 1, "a": 2}')
        path, ok, err = jb.process_json_file((str(p), False, False))
        self.assertTrue(ok)
        self.assertIsNone(err)
        self.assertEqual(path, str(p))
        self.assertEqual(p.read_text(), '{\n  "b": 1,\n  "a": 2\n}\n')

    def test_minify(self):
        p = self.write("a.json", '{\n  "b": 1,\n  "a": 2\n}\n')
        _, ok, _ = jb.process_json_file((str(p), True, False))
        self.assertTrue(ok)
        self.assertEqual(p.read_text(), '{"b":1,"a":2}\n')

    def test_sort_keys_beautify(self):
        p = self.write("a.json", '{"b": 1, "a": 2}')
        _, ok, _ = jb.process_json_file((str(p), False, True))
        self.assertTrue(ok)
        self.assertEqual(p.read_text(), '{\n  "a": 2,\n  "b": 1\n}\n')

    def test_sort_keys_minify(self):
        p = self.write("a.json", '{"b": 1, "a": 2}')
        _, ok, _ = jb.process_json_file((str(p), True, True))
        self.assertTrue(ok)
        self.assertEqual(p.read_text(), '{"a":2,"b":1}\n')

    def test_invalid_json_does_not_touch_file(self):
        original = '{"a":'  # truncated
        p = self.write("bad.json", original)
        _, ok, err = jb.process_json_file((str(p), False, False))
        self.assertFalse(ok)
        self.assertIn("Invalid JSON", err)
        self.assertEqual(p.read_text(), original)

    def test_missing_file(self):
        missing = self.tmp / "nope.json"
        _, ok, err = jb.process_json_file((str(missing), False, False))
        self.assertFalse(ok)
        self.assertIsNotNone(err)

    def test_large_file_uses_mmap_path(self):
        # Force the mmap branch and verify round-trip still works.
        p = self.write("big.json", json.dumps({"a": list(range(100))}))
        with mock.patch.object(jb, "MMAP_THRESHOLD", 10):
            _, ok, _ = jb.process_json_file((str(p), True, False))
        self.assertTrue(ok)
        self.assertEqual(json.loads(p.read_text()), {"a": list(range(100))})


# ---------------------------------------------------------------------------
# is_json_file
# ---------------------------------------------------------------------------

class TestIsJsonFile(FileTestCase):
    def test_json_extension(self):
        p = self.write("a.json", "anything")
        self.assertTrue(jb.is_json_file(p))

    def test_extension_is_case_insensitive(self):
        p = self.write("A.JSON", "anything")
        self.assertTrue(jb.is_json_file(p))

    def test_leading_brace_without_extension(self):
        p = self.write(".prettierrc", '{"semi": false}')
        self.assertTrue(jb.is_json_file(p))

    def test_non_json(self):
        p = self.write("notes.txt", "hello world")
        self.assertFalse(jb.is_json_file(p))

    def test_empty_file(self):
        p = self.write("empty", "")
        self.assertFalse(jb.is_json_file(p))


# ---------------------------------------------------------------------------
# collect_json_files
# ---------------------------------------------------------------------------

class TestCollectJsonFiles(FileTestCase):
    def test_explicit_file_is_always_included(self):
        # Named files are not filtered by extension / content.
        p = self.write("thing.txt", "not json at all")
        self.assertEqual(jb.collect_json_files([str(p)]), {p})

    def test_recursive_discovery(self):
        self.write("a.json", "{}")
        sub = self.tmp / "sub"
        sub.mkdir()
        (sub / "b.json").write_text("{}", encoding="utf-8")
        result = jb.collect_json_files([str(self.tmp)], recursive=True)
        self.assertEqual({p.name for p in result}, {"a.json", "b.json"})

    def test_non_recursive_discovery(self):
        self.write("a.json", "{}")
        sub = self.tmp / "sub"
        sub.mkdir()
        (sub / "b.json").write_text("{}", encoding="utf-8")
        result = jb.collect_json_files([str(self.tmp)], recursive=False)
        self.assertEqual({p.name for p in result}, {"a.json"})

    def test_deduplicates(self):
        p = self.write("a.json", "{}")
        result = jb.collect_json_files([str(p), str(p), str(self.tmp)])
        self.assertEqual(len(result), 1)

    def test_ignores_non_json_extensions(self):
        self.write("a.json", "{}")
        self.write("b.txt", "hello")
        result = jb.collect_json_files([str(self.tmp)])
        self.assertEqual({p.name for p in result}, {"a.json"})


# ---------------------------------------------------------------------------
# main (end-to-end)
# ---------------------------------------------------------------------------

class TestMain(FileTestCase):
    def _run(self, argv):
        """Run jb.main(argv), capture stdout/stderr, return (rc, out, err)."""
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = jb.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def test_no_json_files(self):
        rc, _, err = self._run([str(self.tmp), "--workers", "1"])
        self.assertEqual(rc, 1)
        self.assertIn("No JSON files found", err)

    def test_beautify_default(self):
        p = self.write("a.json", '{"a": 1}')
        rc, out, _ = self._run([str(p), "--workers", "1"])
        self.assertEqual(rc, 0)
        self.assertEqual(p.read_text(), '{\n  "a": 1\n}\n')
        self.assertIn("Successfully processed: 1", out)

    def test_minify_flag(self):
        p = self.write("a.json", '{\n  "a": 1\n}\n')
        rc, _, _ = self._run(["-m", str(p), "--workers", "1"])
        self.assertEqual(rc, 0)
        self.assertEqual(p.read_text(), '{"a":1}\n')

    def test_sort_keys_flag(self):
        p = self.write("a.json", '{"b": 1, "a": 2}')
        rc, _, _ = self._run(["-m", "-s", str(p), "--workers", "1"])
        self.assertEqual(rc, 0)
        self.assertEqual(p.read_text(), '{"a":2,"b":1}\n')

    def test_beautify_and_minify_are_mutually_exclusive(self):
        p = self.write("a.json", "{}")
        # argparse calls sys.exit(2) on the mutex violation.
        with self.assertRaises(SystemExit) as ctx:
            self._run(["-b", "-m", str(p), "--workers", "1"])
        self.assertEqual(ctx.exception.code, 2)

    def test_invalid_json_reports_error_and_returns_1(self):
        p = self.write("bad.json", "{")
        rc, _, err = self._run([str(p), "--workers", "1"])
        self.assertEqual(rc, 1)
        self.assertIn("Invalid JSON", err)
        self.assertIn("bad.json", err)

    def test_mixed_success_and_failure(self):
        good = self.write("good.json", '{"a": 1}')
        self.write("bad.json", "{")
        rc, out, err = self._run([str(self.tmp), "--workers", "1"])
        self.assertEqual(rc, 1)
        self.assertEqual(good.read_text(), '{\n  "a": 1\n}\n')
        self.assertIn("Successfully processed: 1", out)
        self.assertIn("Failed: 1", err)

    def test_default_path_is_cwd(self):
        self.write("a.json", '{"a": 1}')
        cwd = os.getcwd()
        try:
            os.chdir(self.tmp)
            rc, _, _ = self._run(["--workers", "1"])
        finally:
            os.chdir(cwd)
        self.assertEqual(rc, 0)

    def test_multiple_workers(self):
        # Smoke test: run a handful of files through more than one worker.
        for i in range(5):
            self.write(f"f{i}.json", f'{{"n": {i}}}')
        rc, out, _ = self._run([str(self.tmp), "--workers", "3"])
        self.assertEqual(rc, 0)
        self.assertIn("Successfully processed: 5", out)

    def test_workers_capped_to_file_count(self):
        # Only one file → workers get clamped from 8 to 1.
        self.write("a.json", "{}")
        rc, out, _ = self._run([str(self.tmp), "--workers", "8"])
        self.assertEqual(rc, 0)
        self.assertIn("using 1 worker(s)", out)

    def test_idempotent(self):
        # Beautifying twice should produce identical output.
        p = self.write("a.json", '{"z": 1, "a": [1, 2]}')
        self._run([str(p), "--workers", "1"])
        first = p.read_text()
        self._run([str(p), "--workers", "1"])
        self.assertEqual(p.read_text(), first)


if __name__ == "__main__":
    unittest.main(verbosity=2)
