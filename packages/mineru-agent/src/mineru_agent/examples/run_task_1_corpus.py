"""Task 1: 语料处理与题目生成 — 演示 plan + 前两步（dry_run mock 日志）"""
import random
import time
from pathlib import Path

# ROOT = Path(__file__).resolve().parents[3]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead

from mineru_core.log_schema import LogStep, ExecutionLog

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    log = ExecutionLog(task_id="controlsci-corpus-001", task_name="语料处理与题目生成")

    print("=" * 60)
    print("ControlSci Agent — Task 1: 语料处理与题目生成")
    print("=" * 60)

    plan = [
        {"step": 1, "name": "文献解析", "tool": "MinerU", "description": "解析公开 PDF/Markdown 为结构化语料"},
        {"step": 2, "name": "题目生成", "tool": "build_benchmark.py", "description": "基于公开语料生成控制工程评测题"},
    ]
    print(f"\n任务规划（共 {len(plan)} 步）:")
    for p in plan:
        print(f"  Step {p['step']}: {p['name']} ({p['tool']}) — {p['description']}")

    t0 = time.time()

    print("\n--- Step 1: 文献解析（dry_run）---")
    log.add_step(
        LogStep(
            step_id=1,
            step_name="文献解析",
            tool="MinerU",
            input_summary="公开文献目录",
            output_summary="dry-run: would parse documents into corpus/",
            status="skipped",
            duration_ms=0,
        )
    )

    time.sleep(random.uniform(0.1, 0.3))

    print("\n--- Step 2: 题目生成（dry_run）---")
    log.add_step(
        LogStep(
            step_id=2,
            step_name="题目生成",
            tool="build_benchmark.py",
            input_summary="corpus/",
            output_summary="dry-run: would generate benchmark candidates",
            status="skipped",
            duration_ms=0,
        )
    )

    log.final_status = "completed"
    log.total_duration_ms = int((time.time() - t0) * 1000)

    log_path = LOGS_DIR / "task_1_corpus.json"
    log.save(str(log_path))

    print(f"\n日志已保存: {log_path}")
    print(f"总步骤数: {len(log.steps)}, 最终状态: {log.final_status}")


if __name__ == "__main__":
    main()
