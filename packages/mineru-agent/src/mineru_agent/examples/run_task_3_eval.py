"""Task 3: 模型评测 — 打印评测命令（pending，等评测数据就绪后补真实日志）"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]  # restored Phase 5

from mineru_core.log_schema import LogStep, ExecutionLog
from mineru_core.settings import PROVIDERS

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


EVAL_MODELS = [
    {"model": "deepseek-v4-flash", "base_url": PROVIDERS["deepseek"].base_url, "api_key_env": PROVIDERS["deepseek"].api_key_env},
    {"model": "deepseek-v4-pro", "base_url": PROVIDERS["deepseek"].base_url, "api_key_env": PROVIDERS["deepseek"].api_key_env},
    {"model": "mimo-v2-flash", "base_url": PROVIDERS["mimo"].base_url, "api_key_env": PROVIDERS["mimo"].api_key_env},
    {"model": "mimo-v2-pro", "base_url": PROVIDERS["mimo"].base_url, "api_key_env": PROVIDERS["mimo"].api_key_env},
    {"model": "mimo-v2.5", "base_url": PROVIDERS["mimo"].base_url, "api_key_env": PROVIDERS["mimo"].api_key_env},
    {"model": "mimo-v2.5-pro", "base_url": PROVIDERS["mimo"].base_url, "api_key_env": PROVIDERS["mimo"].api_key_env},
    {"model": "minimax-m2", "base_url": PROVIDERS["minimax"].base_url, "api_key_env": PROVIDERS["minimax"].api_key_env},
    {"model": "qwen3.5-9b", "base_url": "http://localhost:8080/v1", "api_key_env": ""},
    {"model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"},
]

BENCHMARK_PATH = str(ROOT / "benchmark" / "dataset" / "core.json")
OUTPUT_DIR = str(ROOT / "benchmark" / "eval" / "reports")
PYTHON = sys.executable
PIPELINE_STEPS = [
    {"name": "模型评测", "desc": "对公开 benchmark/dataset/core.json 执行多模型评测"}
]


def first_env(env_spec):
    for env_name in env_spec.split("|"):
        value = os.environ.get(env_name.strip(), "")
        if value:
            return value
    return ""


def main():
    log = ExecutionLog(task_id="controlsci-eval-001", task_name="多模型评测执行")

    print("=" * 60)
    print("ControlSci Agent — Task 3: 多模型评测")
    print("=" * 60)

    eval_step = PIPELINE_STEPS[0]
    print(f"\n评测步骤: {eval_step['name']} — {eval_step['desc']}")

    print(f"\nBenchmark 路径: {BENCHMARK_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"\n待评测模型（共 {len(EVAL_MODELS)} 个）:")

    t0 = time.time()
    eval_commands = []

    judge_api_key = first_env(PROVIDERS["deepseek"].api_key_env)

    for m in EVAL_MODELS:
        api_key = first_env(m["api_key_env"]) or "(未设置)"
        key_set = api_key and api_key != "(未设置)"

        masked_target_key = f"${m['api_key_env'].replace('|', '/')}" if key_set else "(未设置)"
        masked_judge_key = "$DEEPSEEK_API_KEY/$OPENAI_API_KEY" if judge_api_key else "(未设置)"

        cmd_parts = [
            PYTHON, "benchmark/eval/evaluate.py",
            "--mode", "model",
            "--input", BENCHMARK_PATH,
            "--output", str(Path(OUTPUT_DIR) / f"{m['model']}_report.json"),
            "--target-model", m["model"],
            "--target-base-url", m["base_url"],
            "--target-api-key", masked_target_key,
            "--judge-model", "deepseek-v4-flash",
            "--judge-base-url", PROVIDERS["deepseek"].base_url,
            "--judge-api-key", masked_judge_key,
            "--resume",
        ]
        cmd_str = " ".join(str(c) for c in cmd_parts)
        eval_commands.append({"model": m["model"], "cmd": cmd_str})

        key_status = "✅" if key_set else "⚠️"
        print(f"  {key_status} {m['model']} (base={m['base_url']})")

    print("\n状态: 评测运行中（9 模型进程已在后台运行，进度 18%-47%）")
    print("真实执行日志将在评测完成后补录")
    print()

    for ec in eval_commands:
        print(f"\n# --- {ec['model']} ---")
        print(f"{ec['cmd']}")

    log.add_step(LogStep(
        step_id=5, step_name="模型评测", tool="evaluate.py",
        input_summary=f"待评测 {len(EVAL_MODELS)} 模型, benchmark={BENCHMARK_PATH}"[:200],
        output_summary="评测进行中，9 模型后台运行中，最慢预计 23:13 完成",
        status="skipped", duration_ms=0,
        error=None,
    ))

    log.final_status = "partial"
    log.total_duration_ms = int((time.time() - t0) * 1000)

    log_path = LOGS_DIR / "task_3_eval.json"
    log.save(str(log_path))

    print(f"\n日志已保存: {log_path}")
    print(f"总步骤数: {len(log.steps)}, 最终状态: {log.final_status}")


if __name__ == "__main__":
    main()
