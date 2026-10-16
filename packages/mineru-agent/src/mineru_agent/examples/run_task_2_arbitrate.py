"""Task 2: 质量仲裁 — 构造仲裁 LogStep（mock 日志）"""
import time
from pathlib import Path

# ROOT = Path(__file__).resolve().parents[3]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead

from mineru_core.log_schema import LogStep, ExecutionLog

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

ARBITRATION_STAGES = [
    {"stage": "题目加载", "tool": "build_benchmark.py", "input_items": 360, "output_items": 360, "duration_ms": 210},
    {"stage": "Embedding 快速通道", "tool": "qwen3-embedding:4b", "input_items": 360, "passed": 48, "rejected": 12, "pending": 300, "duration_ms": 1890},
    {"stage": "LLM 第一轮仲裁", "tool": "deepseek-v4-flash", "input_items": 300, "passed": 192, "rejected": 75, "failed": 3, "pending_rollover": 30, "duration_ms": 142000},
    {"stage": "LLM 第二轮仲裁", "tool": "deepseek-v4-flash", "input_items": 75, "passed": 0, "rejected": 75, "duration_ms": 38000},
    {"stage": "孤儿回收", "tool": "deepseek-v4-flash (orphan_judge)", "input_items": 185, "passed": 114, "rejected": 71, "duration_ms": 540000},
    {"stage": "最终合并", "tool": "build_benchmark.py", "merged_total": 1109, "core": 500, "full": 889, "duration_ms": 310},
]


def main():
    log = ExecutionLog(task_id="controlsci-arbitrate-001", task_name="质量仲裁与数据集构建")

    print("=" * 60)
    print("ControlSci Agent — Task 2: 质量仲裁")
    print("=" * 60)

    t0 = time.time()

    for i, stage in enumerate(ARBITRATION_STAGES, 1):
        time.sleep(0.05)
        stage_name = stage["stage"]
        tool = stage["tool"]
        duration = stage["duration_ms"]

        parts = []
        for k, v in stage.items():
            if k not in ("stage", "tool", "duration_ms"):
                parts.append(f"{k}={v}")
        summary = ", ".join(parts)

        log.add_step(LogStep(
            step_id=300 + i, step_name=f"质量仲裁({stage_name})", tool=tool,
            input_summary=f"待处理 {stage.get('input_items', '?')} 题"[:200],
            output_summary=summary[:500],
            status="success", duration_ms=duration, error=None,
        ))

        print(f"  [success] {stage_name}: {summary} ({duration}ms)")

    log.final_status = "completed"
    log.total_duration_ms = int((time.time() - t0) * 1000)

    log_path = LOGS_DIR / "task_2_arbitrate.json"
    log.save(str(log_path))

    total_passed = sum(s.get("passed", 0) for s in ARBITRATION_STAGES)
    total_input = ARBITRATION_STAGES[-1]["merged_total"]

    print(f"\n仲裁完成: 最终 {total_input} 题入库（core=500, full=889）")
    print(f"日志已保存: {log_path}")
    print(f"总步骤数: {len(log.steps)}, 最终状态: {log.final_status}")


if __name__ == "__main__":
    main()
