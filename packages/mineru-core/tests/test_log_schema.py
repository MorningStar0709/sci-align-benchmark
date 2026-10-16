"""Tests for mineru_core.log_schema."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from mineru_core.log_schema import ExecutionLog, LogStep, load_log


class TestLogSchema:
    def test_default_log_step(self):
        s = LogStep()
        assert s.status == "success"
        assert s.step_id == 0

    def test_custom_log_step(self):
        s = LogStep(step_id=1, step_name="test", status="success")
        assert s.step_id == 1
        assert s.step_name == "test"

    def test_execution_log_default(self):
        log = ExecutionLog()
        assert log.final_status == "running"
        assert len(log.steps) == 0

    def test_add_step(self):
        log = ExecutionLog(task_id="t1")
        log.add_step(LogStep(step_name="s1", status="success"))
        assert len(log.steps) == 1
        assert log.steps[0].step_name == "s1"

    def test_to_dict_and_save(self):
        log = ExecutionLog(task_id="t1", task_name="test")
        log.add_step(LogStep(step_name="s1", status="success"))
        d = log.to_dict()
        assert d["task_id"] == "t1"
        assert len(d["steps"]) == 1

        tmp = Path(tempfile.mkdtemp()) / "log.json"
        log.save(str(tmp))
        assert tmp.exists()
        loaded = json.loads(tmp.read_text(encoding="utf-8"))
        assert loaded["task_id"] == "t1"
        tmp.unlink()

    def test_load_log(self):
        log = ExecutionLog(task_id="t1", task_name="test")
        log.add_step(LogStep(step_name="s1", status="success"))
        tmp = Path(tempfile.mkdtemp()) / "log.json"
        log.save(str(tmp))
        loaded = load_log(str(tmp))
        assert loaded.task_id == "t1"
        assert len(loaded.steps) == 1
        assert loaded.steps[0].step_name == "s1"
        tmp.unlink()

    def test_load_log_invalid_status_fallback(self, tmp_path):
        """Invalid final_status and step status values should fall back to 'failed'."""
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({
            "schema_version": "1.0",
            "task_id": "t-bad",
            "task_name": "test",
            "steps": [
                {"step_id": 0, "status": "UNKNOWN_STATUS"},
                {"step_id": 1, "status": "success"},
            ],
            "total_duration_ms": 0,
            "final_status": "INVALID",
            "created_at": "",
        }), encoding="utf-8")
        log = load_log(str(p))
        assert log.final_status == "failed"
        assert log.steps[0].status == "failed"
        assert log.steps[1].status == "success"  # valid step status preserved

    def test_load_log_missing_fields_default(self, tmp_path):
        """Missing fields should get sensible defaults."""
        p = tmp_path / "minimal.json"
        p.write_text(json.dumps({}), encoding="utf-8")
        log = load_log(str(p))
        assert log.final_status == "running"
        assert log.task_id == ""
        assert log.steps == []
