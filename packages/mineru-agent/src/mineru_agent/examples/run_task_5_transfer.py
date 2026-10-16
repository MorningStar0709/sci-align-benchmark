"""Task 5: 领域切换配置分析 — 分析将 Agent 迁移到新领域所需的变更"""
import time
from pathlib import Path

# ROOT = Path(__file__).resolve().parents[3]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead

from mineru_core.log_schema import LogStep, ExecutionLog

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TRANSFER_DOMAINS = [
    {
        "domain": "控制工程 → 生物医学",
        "corpus_change": "arXiv q-bio.* 分类 + PubMed 文献 PDF",
        "prompt_change": "替换学科术语映射表（刚度矩阵→基因表达矩阵）",
        "provider_change": "固定三 Provider，无需改动",
        "eval_change": "可复用 evaluate.py，仅需更新 reference 答案",
        "complexity": "中等（~4 处配置变更）",
    },
    {
        "domain": "控制工程 → 法律文本",
        "corpus_change": "裁判文书网 + JEC-QA 数据集",
        "prompt_change": "维度从公式/推导改为法条引用/判例匹配",
        "provider_change": "需新增法律领域专用模型",
        "eval_change": "需引入 NDCG@10 等检索类指标",
        "complexity": "高（~8 处配置变更）",
    },
    {
        "domain": "控制工程 → 金融分析",
        "corpus_change": "年报 PDF + 研报 + 金融新闻语料",
        "prompt_change": "数值计算维度增强（ROE/PE 等指标题）",
        "provider_change": "可复用，但需关注数值精度",
        "eval_change": "可复用，exact_match 改为 tolerance 匹配",
        "complexity": "中低（~3 处配置变更）",
    },
]

AGENT_CONFIG_LAYERS = [
    {"layer": "Agent 核心", "component": "agent.py", "reusable": True, "note": "execute() 命令行自动适配"},
    {"layer": "日志 Schema", "component": "log_schema.py", "reusable": True, "note": "通用结构化日志，无需改动"},
    {"layer": "语料管理", "component": "corpus/ 目录", "reusable": False, "note": "需更换为目标领域 PDF/MD"},
    {"layer": "题目生成", "component": "build_benchmark.py", "reusable": True, "note": "Provider 抽象层无关领域"},
    {"layer": "质量仲裁", "component": "build_benchmark.py --arbitrate", "reusable": True, "note": "LLM-as-Judge 维度无关"},
    {"layer": "评测引擎", "component": "evaluate.py", "reusable": True, "note": "替换 --input 和 --output 即可"},
    {"layer": "报告生成", "component": "summarize_multi.py", "reusable": True, "note": "维度名称通过 template 切换"},
]


def main():
    log = ExecutionLog(task_id="controlsci-transfer-001", task_name="领域切换配置分析")

    print("=" * 60)
    print("ControlSci Agent — Task 5: 领域切换配置分析")
    print("=" * 60)

    t0 = time.time()

    print("\n--- Step 1: 当前架构分层复用性 ---")
    for layer in AGENT_CONFIG_LAYERS:
        reusable = "✅ 可复用" if layer["reusable"] else "🔧 需改造"
        print(f"  {reusable} {layer['layer']} ({layer['component']}): {layer['note']}")

    reusable_count = sum(1 for l in AGENT_CONFIG_LAYERS if l["reusable"])
    log.add_step(LogStep(
        step_id=1, step_name="领域切换(架构复用性)", tool="config_analysis",
        input_summary=f"分析 {len(AGENT_CONFIG_LAYERS)} 层组件"[:200],
        output_summary=f"{reusable_count}/{len(AGENT_CONFIG_LAYERS)} 层可复用，核心抽象层无需改动"[:500],
        status="success", duration_ms=0,
    ))

    print(f"\n  总结: {reusable_count}/{len(AGENT_CONFIG_LAYERS)} 层可复用")

    print("\n--- Step 2: 三个目标领域迁移分析 ---")
    for idx, domain in enumerate(TRANSFER_DOMAINS):
        print(f"\n  [{domain['complexity']}] {domain['domain']}")
        print(f"    语料变更: {domain['corpus_change']}")
        print(f"    Prompt 变更: {domain['prompt_change']}")
        print(f"    Provider: {domain['provider_change']}")
        print(f"    评测: {domain['eval_change']}")

        log.add_step(LogStep(
            step_id=200 + idx, step_name=f"领域切换({domain['domain']})", tool="domain_analysis",
            input_summary=f"分析迁移至 {domain['domain']}"[:200],
            output_summary=f"复杂度={domain['complexity']}, 语料={domain['corpus_change'][:100]}"[:500],
            status="success", duration_ms=0,
        ))

    print("\n--- Step 3: 推荐迁移路径 ---")
    recommendation = (
        "推荐先迁移至金融分析（中低复杂度，~3 处配置变更），验证 Pipeline 抽象层成立；"
        "再扩展至生物医学（需新增学科术语映射）。CLI 层完全复用，"
        "仅需替换 corpus 目录内容 + 领域专属 prompt template。"
    )
    print(f"  {recommendation}")

    log.add_step(LogStep(
        step_id=3, step_name="领域切换(推荐路径)", tool="migration_plan",
        input_summary="评估迁移优先级与路径"[:200],
        output_summary=recommendation[:500],
        status="success", duration_ms=0,
    ))

    log.final_status = "completed"
    log.total_duration_ms = int((time.time() - t0) * 1000)

    log_path = LOGS_DIR / "task_5_transfer.json"
    log.save(str(log_path))

    print(f"\n日志已保存: {log_path}")
    print(f"总步骤数: {len(log.steps)}, 最终状态: {log.final_status}")


if __name__ == "__main__":
    main()
