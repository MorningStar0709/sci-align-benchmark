"""Task 4: 故障恢复 — 扫描 checkpoint 文件并分析恢复策略"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]  # restored Phase 5

from mineru_core.log_schema import LogStep, ExecutionLog

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_CHECKPOINT_DIR = ROOT / "benchmark" / "eval"
PIPELINE_CHECKPOINT_DIR = ROOT / "benchmark" / "pipeline"


def scan_checkpoints(base_dir, pattern="**/*.checkpoint*"):
    if not base_dir.exists():
        print(f"  [警告] 目录不存在，跳过扫描: {base_dir}", file=sys.stderr)
        return []
    results = []
    for p in sorted(base_dir.glob(pattern)):
        stat = p.stat()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            completed = data.get("completed", []) if isinstance(data, dict) else []
            failed = data.get("failed", []) if isinstance(data, dict) else []
            pending = data.get("pending", []) if isinstance(data, dict) else []
        except (json.JSONDecodeError, UnicodeDecodeError):
            completed = []
            failed = []
            pending = []

        results.append({
            "path": str(p.relative_to(ROOT)),
            "size_bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "completed": len(completed),
            "failed": len(failed),
            "pending": len(pending),
        })
    return results


def main():
    log = ExecutionLog(task_id="controlsci-recovery-001", task_name="故障恢复扫描与策略分析")

    print("=" * 60)
    print("ControlSci Agent — Task 4: 故障恢复扫描")
    print("=" * 60)

    t0 = time.time()

    print("\n--- Step 1: 扫描评测 checkpoint ---")
    eval_checkpoints = scan_checkpoints(EVAL_CHECKPOINT_DIR, "**/*.checkpoint*")
    eval_found = len(eval_checkpoints)

    if eval_checkpoints:
        for cp in eval_checkpoints:
            print(f"  {cp['path']} ({cp['size_bytes']}B, mtime={cp['mtime']})")
            print(f"    completed={cp['completed']}, failed={cp['failed']}, pending={cp['pending']}")
    else:
        print("  (未发现 checkpoint 文件)")

    log.add_step(LogStep(
        step_id=401, step_name="故障恢复(评测checkpoint扫描)", tool="checkpoint_scan",
        input_summary=f"扫描目录 {EVAL_CHECKPOINT_DIR}"[:200],
        output_summary=f"发现 {eval_found} 个 checkpoint 文件"[:500],
        status="success", duration_ms=0,
    ))

    print("\n--- Step 2: 扫描 pipeline checkpoint ---")
    pipeline_checkpoints = scan_checkpoints(PIPELINE_CHECKPOINT_DIR, "**/*.checkpoint*")
    pipeline_found = len(pipeline_checkpoints)

    if pipeline_checkpoints:
        for cp in pipeline_checkpoints:
            print(f"  {cp['path']} ({cp['size_bytes']}B, mtime={cp['mtime']})")
    else:
        print("  (未发现 pipeline checkpoint 文件)")

    log.add_step(LogStep(
        step_id=402, step_name="故障恢复(pipeline checkpoint扫描)", tool="checkpoint_scan",
        input_summary=f"扫描目录 {PIPELINE_CHECKPOINT_DIR}"[:200],
        output_summary=f"发现 {pipeline_found} 个 checkpoint 文件"[:500],
        status="success", duration_ms=0,
    ))

    print("\n--- Step 3: 恢复策略分析 ---")
    total = eval_found + pipeline_found
    if total > 0:
        strategy = "resume"
        detail = f"共 {total} 个 checkpoint，可执行 --resume 恢复，无需重新开始"
    else:
        strategy = "fresh_start"
        detail = "无 checkpoint 遗留，需从头开始执行"

    print(f"  恢复策略: {strategy}")
    print(f"  分析结论: {detail}")

    log.add_step(LogStep(
        step_id=403, step_name="故障恢复(策略分析)", tool="recovery_analysis",
        input_summary=f"checkpoint 总数 {total}"[:200],
        output_summary=f"策略={strategy}, {detail}"[:500],
        status="success", duration_ms=0,
    ))

    log.final_status = "completed"
    log.total_duration_ms = int((time.time() - t0) * 1000)

    log_path = LOGS_DIR / "task_4_recovery.json"
    log.save(str(log_path))

    print(f"\n日志已保存: {log_path}")
    print(f"总步骤数: {len(log.steps)}, 最终状态: {log.final_status}")


if __name__ == "__main__":
    main()
