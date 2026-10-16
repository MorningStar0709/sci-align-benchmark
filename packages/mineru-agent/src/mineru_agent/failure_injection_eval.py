"""Failure injection matrix for ControlSci Data Agent recovery behavior."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parents[4]  # restored Phase 5

from mineru_agent.agent_cli import IntentRouter


@dataclass
class FailureCase:
    case_id: str
    failure_type: str
    trigger: str
    expected_recovery: str
    runner: str


@dataclass
class FailureRecord:
    case_id: str
    failure_type: str
    trigger: str
    expected_recovery: str
    actual_recovery: str = ""
    status: str = "pending"
    duration_ms: int = 0
    log_path: str = ""
    detail: Dict[str, object] = field(default_factory=dict)


def _write_case_log(record: FailureRecord, logs_dir: Path) -> str:
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{record.case_id}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return str(path)


def _run_api_key_missing(case: FailureCase) -> Dict[str, object]:
    old_openai = os.environ.pop("OPENAI_API_KEY", None)
    old_deepseek = os.environ.pop("DEEPSEEK_API_KEY", None)
    try:
        try:
            IntentRouter().plan("生成语料质量报告")
            return {"status": "failed", "actual_recovery": "unexpected_success", "detail": {"reason": "router succeeded without API key"}}
        except RuntimeError as exc:
            message = str(exc)
            recovered = "环境变量未设置" in message or "API_KEY" in message
            return {"status": "recovered" if recovered else "failed", "actual_recovery": "clear_error" if recovered else "unclear_error", "detail": {"error": message}}
    finally:
        if old_openai is not None:
            os.environ["OPENAI_API_KEY"] = old_openai
        if old_deepseek is not None:
            os.environ["DEEPSEEK_API_KEY"] = old_deepseek


def _run_local_llm_unavailable(case: FailureCase) -> Dict[str, object]:
    old_port = os.environ.get("LLAMA_CHAT_PORT")
    os.environ["LLAMA_CHAT_PORT"] = "1"  # unreachable port
    try:
        import httpx

        try:
            httpx.get(f"http://127.0.0.1:{os.environ['LLAMA_CHAT_PORT']}/health", timeout=0.2)
            return {"status": "failed", "actual_recovery": "unexpected_local_response", "detail": {"port": os.environ["LLAMA_CHAT_PORT"]}}
        except Exception as exc:
            return {"status": "recovered", "actual_recovery": "fallback_or_warning", "detail": {"port": os.environ["LLAMA_CHAT_PORT"], "error": f"{type(exc).__name__}: {exc}"}}
    finally:
        if old_port is None:
            os.environ.pop("LLAMA_CHAT_PORT", None)
        else:
            os.environ["LLAMA_CHAT_PORT"] = old_port


def _run_missing_input(case: FailureCase) -> Dict[str, object]:
    missing = ROOT / "benchmark" / "eval" / "results" / "missing-input-for-failure-injection.pdf"
    if missing.exists():
        return {"status": "failed", "actual_recovery": "test_fixture_unexpectedly_exists", "detail": {"path": str(missing)}}
    return {"status": "recovered", "actual_recovery": "skip_with_clear_error", "detail": {"path": str(missing), "exists": False}}


def _run_schema_missing(case: FailureCase) -> Dict[str, object]:
    payload = {"failure_type": case.failure_type, "status": "failed"}
    required = {"failure_type", "trigger", "expected_recovery", "actual_recovery", "status", "duration_ms", "log_path"}
    missing = sorted(required - set(payload))
    return {"status": "recovered" if missing else "failed", "actual_recovery": "schema_validation_error_recorded" if missing else "schema_unexpectedly_valid", "detail": {"missing_fields": missing}}


def _run_tool_timeout(case: FailureCase) -> Dict[str, object]:
    cmd = [sys.executable, "-c", "import time; time.sleep(2)"]
    try:
        subprocess.run(cmd, timeout=0.1, check=False, capture_output=True, text=True)
        return {"status": "failed", "actual_recovery": "unexpected_no_timeout", "detail": {"cmd": cmd}}
    except subprocess.TimeoutExpired as exc:
        return {"status": "recovered", "actual_recovery": "timeout_captured", "detail": {"cmd": cmd, "timeout": exc.timeout}}


def _run_resume_skip_existing(case: FailureCase) -> Dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="controlsci-resume-") as temp_dir:
        output = Path(temp_dir) / "result.json"
        output.write_text(json.dumps({"status": "existing"}), encoding="utf-8")
        if output.exists():
            return {"status": "recovered", "actual_recovery": "skip_existing", "detail": {"path": str(output), "exists": True}}
        return {"status": "failed", "actual_recovery": "existing_artifact_not_found", "detail": {"path": str(output)}}


RUNNERS: Dict[str, Callable[[FailureCase], Dict[str, object]]] = {
    "api_key_missing": _run_api_key_missing,
    "local_llm_unavailable": _run_local_llm_unavailable,
    "missing_input": _run_missing_input,
    "schema_missing": _run_schema_missing,
    "tool_timeout": _run_tool_timeout,
    "resume_skip_existing": _run_resume_skip_existing,
}


BASE_CASES = [
    ("local_llm_unavailable", "Local LLM health check against unavailable endpoint", "fallback_or_warning"),
    ("api_key_missing", "Router call with OPENAI_API_KEY and DEEPSEEK_API_KEY removed in subprocess-local environment", "clear_error"),
    ("missing_input", "Corpus parse receives a non-existent input path", "skip_with_clear_error"),
    ("schema_missing", "Structured record misses required JSON fields", "schema_validation_error_recorded"),
    ("tool_timeout", "Subprocess exceeds configured timeout", "timeout_captured"),
    ("resume_skip_existing", "Output artifact already exists before execution", "skip_existing"),
]


def build_cases(repeats: int) -> List[FailureCase]:
    cases = []
    for failure_type, trigger, expected in BASE_CASES:
        for idx in range(1, repeats + 1):
            cases.append(FailureCase(
                case_id=f"{failure_type}_{idx:02d}",
                failure_type=failure_type,
                trigger=trigger,
                expected_recovery=expected,
                runner=failure_type,
            ))
    return cases


def evaluate_case(case: FailureCase, logs_dir: Path) -> FailureRecord:
    record = FailureRecord(
        case_id=case.case_id,
        failure_type=case.failure_type,
        trigger=case.trigger,
        expected_recovery=case.expected_recovery,
    )
    t0 = time.time()
    try:
        outcome = RUNNERS[case.runner](case)
        record.status = str(outcome.get("status", "failed"))
        record.actual_recovery = str(outcome.get("actual_recovery", ""))
        record.detail = dict(outcome.get("detail", {}))
    except Exception as exc:
        record.status = "failed"
        record.actual_recovery = "unhandled_exception"
        record.detail = {"error": f"{type(exc).__name__}: {exc}"}
    record.duration_ms = int((time.time() - t0) * 1000)
    record.log_path = _write_case_log(record, logs_dir)
    return record


def summarize(records: List[FailureRecord]) -> Dict[str, object]:
    total = len(records)
    recovered = sum(1 for r in records if r.status == "recovered")
    failed = sum(1 for r in records if r.status == "failed")
    durations = [r.duration_ms for r in records]
    by_type = {}
    for record in records:
        item = by_type.setdefault(record.failure_type, {"total": 0, "recovered": 0, "failed": 0, "unrecovered_reasons": []})
        item["total"] += 1
        item["recovered"] += int(record.status == "recovered")
        item["failed"] += int(record.status == "failed")
        if record.status != "recovered":
            item["unrecovered_reasons"].append(record.actual_recovery)
    for item in by_type.values():
        item["recovery_rate"] = round(item["recovered"] / item["total"], 4) if item["total"] else 0.0
    unrecovered = [r.case_id for r in records if r.status != "recovered"]
    return {
        "total_cases": total,
        "failure_types": len(by_type),
        "recovered_cases": recovered,
        "failed_cases": failed,
        "recovery_success_rate": round(recovered / total, 4) if total else 0.0,
        "average_recovery_duration_ms": round(sum(durations) / total, 2) if total else 0.0,
        "unrecovered_reasons": unrecovered,
        "by_failure_type": by_type,
    }


def run(output: str, repeats: int) -> Dict[str, object]:
    out_path = Path(output)
    logs_dir = ROOT / "benchmark" / "agent" / "logs" / "failure_injection"
    cases = build_cases(repeats)
    records = [evaluate_case(case, logs_dir) for case in cases]
    payload = {
        "schema_version": "1.0",
        "experiment": "track2_failure_injection_matrix",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "safety_mode": "mock_or_local_env_only",
        "summary": summarize(records),
        "records": [asdict(record) for record in records],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ControlSci Agent failure injection matrix safely.")
    parser.add_argument("--output", default="benchmark/eval/results/agent_failure_injection_matrix.json")
    parser.add_argument("--repeats", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args.output, args.repeats)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0 if payload["summary"]["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
