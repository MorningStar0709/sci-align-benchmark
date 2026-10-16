"""Tests for mineru_core.utils."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from mineru_core.utils import (
    append_jsonl,
    iter_report_files,
    load_json,
    write_json_atomic,
)


class TestUtils:
    def test_write_json_atomic(self):
        path = os.path.join(tempfile.mkdtemp(), "test.json")
        data = {"key": "value", "nested": [1, 2, 3]}
        write_json_atomic(path, data)
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == data
        os.remove(path)

    def test_write_json_atomic_creates_parent(self):
        tmp = Path(tempfile.mkdtemp())
        nested = tmp / "sub" / "nested" / "test.json"
        write_json_atomic(str(nested), {"key": "value"})
        assert nested.exists()
        nested.unlink()
        # Clean up created directories in reverse order
        (tmp / "sub" / "nested").rmdir()
        (tmp / "sub").rmdir()
        tmp.rmdir()

    def test_load_json(self):
        path = os.path.join(tempfile.mkdtemp(), "test.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"a": 1}, f)
        assert load_json(path) == {"a": 1}
        os.remove(path)

    def test_load_json_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_json("/nonexistent/path.json")

    def test_load_json_invalid_json(self):
        path = os.path.join(tempfile.mkdtemp(), "bad.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{invalid json}")
        with pytest.raises(json.JSONDecodeError):
            load_json(path)
        os.remove(path)

    def test_load_json_with_path_object(self):
        path = Path(tempfile.mkdtemp()) / "test.json"
        path.write_text('{"hello": "world"}', encoding="utf-8")
        assert load_json(path) == {"hello": "world"}
        path.unlink()

    def test_append_jsonl(self):
        path = os.path.join(tempfile.mkdtemp(), "test.jsonl")
        append_jsonl(path, {"step": 1})
        append_jsonl(path, {"step": 2})
        lines = [json.loads(l) for l in open(path, encoding="utf-8")]
        assert len(lines) == 2
        assert lines[0]["step"] == 1
        assert lines[1]["step"] == 2
        os.remove(path)

    def test_append_jsonl_creates_parent(self):
        tmp = Path(tempfile.mkdtemp())
        nested = tmp / "logs" / "test.jsonl"
        append_jsonl(str(nested), {"event": "start"})
        assert nested.exists()
        lines = [json.loads(l) for l in nested.open(encoding="utf-8")]
        assert len(lines) == 1
        nested.unlink()


class TestIterReportFiles:
    def test_yields_matching_files(self, tmp_path):
        files = ["a_report.json", "report_b.json", "output_c.json", "eval_d.json"]
        for f in files:
            (tmp_path / f).write_text("{}", encoding="utf-8")
        results = list(iter_report_files(tmp_path))
        assert len(results) == len(files)
        assert all(p.exists() for p in results)

    def test_skips_leaderboard_and_summary(self, tmp_path):
        (tmp_path / "output_leaderboard.json").write_text("{}", encoding="utf-8")
        (tmp_path / "eval_summary.json").write_text("{}", encoding="utf-8")
        (tmp_path / "real_report.json").write_text("{}", encoding="utf-8")
        results = list(iter_report_files(tmp_path))
        assert len(results) == 1
        assert results[0].name == "real_report.json"

    def test_skips_status_and_progress(self, tmp_path):
        (tmp_path / "output_task.status.json").write_text("{}", encoding="utf-8")
        (tmp_path / "eval_task.progress.json").write_text("{}", encoding="utf-8")
        (tmp_path / "eval_report.json").write_text("{}", encoding="utf-8")
        results = list(iter_report_files(tmp_path))
        assert len(results) == 1
        assert results[0].name == "eval_report.json"

    def test_deduplicates_by_path(self, tmp_path):
        """Files matching multiple patterns should only appear once."""
        # eval_report.json matches both *_report.json and eval_*.json
        f = tmp_path / "eval_report.json"
        f.write_text("{}", encoding="utf-8")
        results = list(iter_report_files(tmp_path))
        assert len(results) == 1

    def test_empty_directory(self, tmp_path):
        results = list(iter_report_files(tmp_path))
        assert results == []

    def test_no_matching_files(self, tmp_path):
        (tmp_path / "readme.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "data.csv").write_text("a,b,c", encoding="utf-8")
        results = list(iter_report_files(tmp_path))
        assert results == []

    def test_path_as_string(self, tmp_path):
        (tmp_path / "eval_report.json").write_text("{}", encoding="utf-8")
        results = list(iter_report_files(str(tmp_path)))
        assert len(results) == 1
