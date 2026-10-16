"""Monitor long-running ControlSci model evaluation reports.

The monitor reads per-model status/progress files emitted by evaluate.py and
prints current progress, stale flags, recent throughput, and ETA estimates.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path


__all__ = [
    "iter_statuses",
    "render",
    "summarize",
]


ROOT = Path(__file__).resolve().parents[4]

from mineru_eval.utils import load_json


def load_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_dt(value):
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def apply_recent_rate(status, recent_window):
    path = status.get("path")
    if not path:
        return status
    progress_name = status.get("progress_log")
    progress_path = path.with_name(progress_name) if progress_name else path.with_suffix(".progress.jsonl")
    rows = load_jsonl(progress_path)
    if len(rows) < 2:
        return status

    recent = rows[-max(recent_window, 2):]
    first = recent[0]
    last = recent[-1]
    first_dt = parse_dt(first.get("updated_at"))
    last_dt = parse_dt(last.get("updated_at"))
    if not first_dt or not last_dt:
        return status
    elapsed_min = (last_dt - first_dt).total_seconds() / 60
    completed_delta = int(last.get("completed") or 0) - int(first.get("completed") or 0)
    if elapsed_min <= 0 or completed_delta <= 0:
        return status

    recent_rate = completed_delta / elapsed_min
    remaining = int(status.get("remaining") or 0)
    recent_eta_min = remaining / recent_rate if recent_rate > 0 else None
    status["recent_rate_q_per_min"] = round(recent_rate, 4)
    status["recent_eta_min"] = round(recent_eta_min, 2) if recent_eta_min is not None else None
    if recent_eta_min is not None:
        status["recent_eta_at"] = datetime.fromtimestamp(time.time() + recent_eta_min * 60).strftime("%Y-%m-%d %H:%M:%S")
    return status


def iter_statuses(reports_dir, recent_window):
    reports_dir = Path(reports_dir)
    status_files = sorted(reports_dir.glob("*.status.json"))
    seen = set()

    for status_path in status_files:
        seen.add(status_path.name.removesuffix(".status.json"))
        try:
            status = load_json(status_path)
        except Exception as exc:
            yield {"model": status_path.name, "status": "unreadable", "error": str(exc), "path": status_path}
            continue
        status["path"] = status_path
        yield apply_recent_rate(status, recent_window)

    for report_path in sorted(reports_dir.glob("*_report.json")):
        if report_path.name in seen:
            continue
        try:
            report = load_json(report_path)
        except Exception:
            continue
        records = report.get("records", [])
        total = report.get("total_questions", len(records))
        complete = report.get("complete", bool(total) and len(records) >= total)
        status = {
            "model": report.get("model", report_path.stem),
            "judge_model": report.get("judge_model", ""),
            "status": "completed" if complete else "partial",
            "completed": len(records),
            "total": total,
            "percent": round(len(records) / total * 100, 2) if total else 0.0,
            "remaining": max(total - len(records), 0),
            "updated_at": report.get("updated_at", ""),
            "path": report_path,
        }
        metrics = report.get("run_metrics") or {}
        status.update(metrics)
        yield apply_recent_rate(status, recent_window)


def fmt_num(value, digits=2, empty="-"):
    if value is None or value == "":
        return empty
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_age(updated_at):
    dt = parse_dt(updated_at)
    if not dt:
        return "-"
    seconds = max((datetime.now() - dt).total_seconds(), 0)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def summarize(statuses):
    total_completed = sum(int(s.get("completed") or 0) for s in statuses)
    total_questions = sum(int(s.get("total") or 0) for s in statuses)
    total_remaining = sum(int(s.get("remaining") or 0) for s in statuses)
    total_rate = sum(float(s.get("recent_rate_q_per_min") or s.get("rate_q_per_min") or 0) for s in statuses)
    eta_min = total_remaining / total_rate if total_rate > 0 else None
    return {
        "completed": total_completed,
        "total": total_questions,
        "remaining": total_remaining,
        "rate_q_per_min": total_rate,
        "eta_min": eta_min,
        "percent": round(total_completed / total_questions * 100, 2) if total_questions else 0.0,
    }


def render(statuses, stale_after_sec):
    statuses = list(statuses)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"ControlSci eval monitor | {now}")
    print("-" * 132)
    print(
        f"{'model':32} {'status':10} {'done':>9} {'pct':>7} "
        f"{'rate/min':>9} {'recent':>9} {'eta_min':>9} {'eta_at':19} {'age':>7} {'last_id':14} flags"
    )
    print("-" * 132)
    for status in statuses:
        model = str(status.get("model", "?"))[:32]
        completed = int(status.get("completed") or 0)
        total = int(status.get("total") or 0)
        done = f"{completed}/{total}" if total else f"{completed}/?"
        updated_at = status.get("updated_at", "")
        age = fmt_age(updated_at)
        flags = []
        dt = parse_dt(updated_at)
        if dt and (datetime.now() - dt).total_seconds() > stale_after_sec and status.get("status") == "running":
            flags.append("STALE")
        if int(status.get("consecutive_failures") or 0) > 0:
            flags.append(f"fail={status.get('consecutive_failures')}")
        if status.get("error"):
            flags.append("ERROR")

        eta_min = status.get("recent_eta_min")
        eta_at = status.get("recent_eta_at")
        if eta_min is None:
            eta_min = status.get("eta_min")
            eta_at = status.get("eta_at")
        print(
            f"{model:32} {str(status.get('status', '-'))[:10]:10} {done:>9} "
            f"{fmt_num(status.get('percent'), 2):>7} {fmt_num(status.get('rate_q_per_min'), 3):>9} "
            f"{fmt_num(status.get('recent_rate_q_per_min'), 3):>9} {fmt_num(eta_min, 1):>9} {str(eta_at or '-')[:19]:19} "
            f"{age:>7} {str(status.get('last_id') or '-')[:14]:14} {','.join(flags)}"
        )
    print("-" * 132)
    summary = summarize(statuses)
    eta_at = "-"
    if summary["eta_min"] is not None:
        eta_at = datetime.fromtimestamp(time.time() + summary["eta_min"] * 60).strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"TOTAL completed={summary['completed']}/{summary['total']} "
        f"percent={summary['percent']:.2f}% remaining={summary['remaining']} "
        f"rate={summary['rate_q_per_min']:.3f} q/min eta_min={fmt_num(summary['eta_min'], 1)} eta_at={eta_at}"
    )


def main():
    parser = argparse.ArgumentParser(description="Monitor ControlSci model evaluation progress.")
    parser.add_argument("--reports-dir", default=str(ROOT / "benchmark" / "eval" / "reports"), help="Directory containing *_report.json.status.json files")
    parser.add_argument("--watch", type=int, default=0, help="Refresh interval in seconds; 0 prints once")
    parser.add_argument("--stale-after-sec", type=int, default=300, help="Mark a running model STALE after this many seconds without updates")
    parser.add_argument("--recent-window", type=int, default=20, help="Use the most recent N progress events for recent rate and ETA")
    args = parser.parse_args()

    while True:
        statuses = list(iter_statuses(args.reports_dir, args.recent_window))
        if not statuses:
            print(f"No status/report files found in {args.reports_dir}")
        else:
            render(statuses, args.stale_after_sec)
        if args.watch <= 0:
            break
        time.sleep(args.watch)
        print()


if __name__ == "__main__":
    main()
