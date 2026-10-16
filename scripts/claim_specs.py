"""主张规格表：声明式定义所有候选主张"""

from dataclasses import dataclass


@dataclass
class ClaimSpec:
    """单条主张的规格"""

    claim_id: str  # 唯一标识，如 "elasticity_reversal_4b"
    statement: str  # 主张陈述（中文）
    domain: str  # control / medical / other
    section: str  # 建议展示章节（如 "surface", "ablation", "limitations"）
    supports: list[str]  # 支撑产物路径（相对仓库根）
    recompute: str  # 可复算表达式（指向具体脚本/函数）
    caliber_ids: list[str]  # 关联的 CALIBERS.md 条目 ID


def all_claims() -> list[ClaimSpec]:
    """返回所有主张规格"""
    return [
        # === 弹性反转 ===
        ClaimSpec(
            claim_id="elasticity_reversal_4b",
            statement="4B 模型在 MGPO-EG 下出现弹性反转（增益为负）",
            domain="control",
            section="surface",
            supports=["benchmark/eval/n3_regen/mgpo_eg_gen4.json"],
            recompute="papers/paper-a/figures/scripts/surface_stats.py::contrasts",
            caliber_ids=["T1", "T3", "T22"],
        ),
        ClaimSpec(
            claim_id="elasticity_reversal_9b",
            statement="9B 模型在 MGPO-EG 下弹性反转消失（增益为正）",
            domain="control",
            section="surface",
            supports=["benchmark/eval/n3_regen/mgpo_eg_9b_gen4.json"],
            recompute="papers/paper-a/figures/scripts/surface_stats.py::contrasts",
            caliber_ids=["T1", "T3", "T22"],
        ),
        # === MGPO-EG U 形 ===
        ClaimSpec(
            claim_id="mgpo_eg_u_shape",
            # 数值不进主张句（CALIBERS T8 的规矩）：读表时按 claim_id 从登记聚合取，
            # 见 figures/scripts/surface_stats.gains()。旧句内嵌 4B −5.9 / 9B +4.3
            # 系七月批废止读数（value-version-audit V2）。
            statement="MGPO-EG 对 SFT 的增益沿 Qwen 四尺度点非单调（4B 为局部极小、9B 为局部极大；27B 无训练读数，不在此轴），逐点数值见表 tab:cross-family",
            domain="control",
            section="surface-scale",
            supports=[
                "benchmark/eval/n3_regen/mgpo_eg_gen4.json",
                "benchmark/eval/n3_regen/mgpo_eg_9b_gen4.json",
            ],
            recompute="papers/paper-a/figures/scripts/surface_stats.py::gains",
            caliber_ids=["T1", "T3", "T5"],
        ),
        # === B1 加固 ===
        ClaimSpec(
            claim_id="B1_adjacent_granularity",
            # 数值不进主张句（CALIBERS T8 / V12）：旧句写 margin +0.0261，那属仲裁补判值，
            # 其采纳挂在你未签的 S12 上；判据件现读 mean_margin=0.0139 > 阈值 0.0132。
            statement="B1 相邻粒度差异成立（三对 gen 配对全同向、mean margin 超 2σg(4B) 冻结阈值；具体数值读 b1_verdict.json，仲裁补判值待 S12 签署后方可引用）",
            domain="control",
            section="ablation",
            supports=["benchmark/eval/arbitration_patch/patch_summary.json"],
            recompute="scripts/run_arbitration_patch.py::b1_margins",
            caliber_ids=["T16"],
        ),
        # === 9B 支点 ===
        ClaimSpec(
            claim_id="pivot_9b_00244",
            statement="9B 是两条强化式方法排序的支点（登记 margin 0.0244，MGPO-EG 9B vs RLVR-C v2 9B）",
            domain="control",
            section="surface",
            supports=["benchmark/eval/run1_rejudge/mgpo_eg_9b_run1p.json",
                      "benchmark/eval/n3_regen/mgpo_eg_9b_gen2.json",
                      "benchmark/eval/n3_regen/mgpo_eg_9b_gen3.json",
                      "benchmark/eval/run1_rejudge/rlvr_c_v2_9b_run1p.json",
                      "benchmark/eval/n3_regen/rlvr_c_v2_9b_gen2.json",
                      "benchmark/eval/n3_regen/rlvr_c_v2_9b_gen3.json"],
            recompute="benchmark/eval/n3_regen/n3_aggregation_summary.json#pivot/margin",
            caliber_ids=["T3"],
        ),
        # === EG−Fixed 三档 ===
        ClaimSpec(
            claim_id="eg_fixed_three_tiers",
            statement="熵门增益（EG−Fixed）分档：0.8B 显著为正（+5.7pp）、2B 弱（+0.6pp）、9B 低于阈值（+0.0133 < 2σg）；无 27B 训练读数",
            domain="control",
            section="surface-scale",
            supports=[
                "benchmark/eval/n3_regen/mgpo_fixed_9b_gen2.json",
                "benchmark/eval/n3_regen/mgpo_fixed_9b_gen3.json",
            ],
            recompute="scripts/check_figure_ledger.py::recompute_ablation",
            caliber_ids=["T1", "T5"],
        ),
        # === CF 两档 ===
        ClaimSpec(
            claim_id="cf_two_tiers",
            statement="反事实（CF）两档读数已在产物（4B 0.3942 / 9B 0.4595），正文仅作定性论证未入表；无 27B CF 训练读数",
            domain="control",
            section="surface-scale",
            supports=[
                "benchmark/eval/counterfactual/mgpo_cf_9b_gen1.json",
            ],
            recompute="scripts/check_figure_ledger.py::recompute_cf_margins",
            caliber_ids=["T7"],
        ),
        # === 吸引子 ===
        ClaimSpec(
            claim_id="attractor_gemma_triplet",
            statement="Gemma 三元组在医疗域呈现吸引子效应（性能收敛）",
            domain="medical",
            section="surface",
            supports=[
                "benchmark/eval/medical_dual_rejudge/medical_mgpo_gemma4_run1.json",
                "benchmark/eval/medical_dual_rejudge/medical_mgpo_gemma4_run2.json",
            ],
            recompute="scripts/claim_evidence.py::medical_arm_means",
            caliber_ids=["T9"],
        ),
        ClaimSpec(
            claim_id="attractor_medical_nine",
            statement="医疗九行在跨模型下呈现吸引子效应",
            domain="medical",
            section="surface",
            supports=["benchmark/eval/medical_dual_rejudge/medical_mgpo_9b_run1.json"],
            recompute="scripts/claim_evidence.py::medical_arm_means",
            caliber_ids=["T9"],
        ),
        # === RLVR ±5% 带 ===
        ClaimSpec(
            claim_id="rlvr_5pct_band",
            statement="RLVR 增益在 ±5% 带内波动（无显著规模效应）",
            domain="control",
            section="surface-scale",
            supports=[
                "benchmark/eval/n3_regen/rlvr_c_v2_gen4.json",
                "benchmark/eval/n3_regen/rlvr_c_v2_9b_gen2.json",
            ],
            recompute="papers/paper-a/figures/scripts/surface_stats.py::gains",
            caliber_ids=["T1", "T5"],
        ),
        # === xcheck OFFSET ===
        ClaimSpec(
            claim_id="xcheck_offset",
            statement="xcheck OFFSET 与 Raw 分母存在系统性偏差",
            domain="control",
            section="limitations",
            # 原指 cross_judge/（DATA-TRACE:450 判隔离：8/5 n=2 探针失败品）。
            # 真源按 §3.8 改指 xcheck 定案件（value-version-audit V7）。
            supports=["benchmark/eval/a800/xcheck_verdict.json"],
            recompute="benchmark/eval/a800/xcheck_verdict.json#deviation_from_band_edge",
            caliber_ids=["T11"],
        ),
        # === 小端 Raw 分母 ===
        ClaimSpec(
            claim_id="small_end_raw_denominator",
            statement="小端模型（4B）的 Raw 分母口径与双判口径差异显著",
            domain="control",
            section="limitations",
            supports=["benchmark/eval/n3_regen/raw_9b_gen2.json"],
            recompute="scripts/check_product_self_consistency.py::collect_denominator_map",
            caliber_ids=["T12"],
        ),
        # === 大端锚点 ===
        ClaimSpec(
            claim_id="anchor_27b",
            statement="27B 是未训练 raw 锚点（control 0.4242 / medical 0.5175 双判），只支持趋势位，不含任何增益",
            domain="control",
            section="surface-scale",
            supports=["benchmark/eval/probe_scaling/qwen27b_control_gen1.json"],
            recompute="benchmark/eval/probe_scaling/qwen27b_control_gen1.json#overall_score",
            caliber_ids=["T5"],
        ),
        ClaimSpec(
            claim_id="anchor_gemma_12b",
            statement="Gemma-12B 是未训练 raw 锚点（control 0.5600 / medical 0.6125 双判），只支持趋势位",
            domain="control",
            section="surface-scale",
            # 陈述印两个数（control 0.5600 / medical 0.6125），原先 supports 只挂 control 件
            # ⇒ medical 那个数在登记面里**没有源**（27B 同构行有：medical 件挂在
            # `medical_scale_axis.supports`）。本行补上 medical 源，闭合这条空头指针。
            supports=[
                "benchmark/eval/probe_scaling/gemma12b_control_gen1.json",
                "benchmark/eval/probe_medical_rejudge/gemma12b_medical_rejudged.json",
            ],
            recompute="benchmark/eval/probe_scaling/gemma12b_control_gen1.json#overall_score",
            caliber_ids=["T5"],
        ),
        # === 医疗规模轴 ===
        ClaimSpec(
            claim_id="medical_scale_axis",
            statement="医疗域 MGPO-EG 增益 4B（+0.1306）与 9B（+0.1363）均为正；27B 仅未训练 raw，不构成规模单调性主张",
            domain="medical",
            section="surface-scale",
            supports=[
                "benchmark/eval/medical_dual_rejudge/medical_mgpo_9b_run1.json",
                "benchmark/eval/probe_medical_rejudge/qwen27b_medical_rejudged.json",
                # 医疗阈值源：本行 docstring 自陈「真缺口是复算源在 temp/……进不了调用链」。
                # 10-01 该件入仓（逐字节复制）后补挂此处，登记面才收得到它。
                "benchmark/eval/e12_e4_medical/e12_e4_medical_result.json",
            ],
            recompute="scripts/probe_readers.py::read_medical_gain_vs_base",
            caliber_ids=["T9", "T10"],
        ),
        # === PPL↔行为分 ρ ===
        ClaimSpec(
            claim_id="ppl_behavior_correlation",
            statement="PPL 与行为分呈负相关（ρ < 0；样本定义＝13 个控制域评测，含 3 个未训练 base）",
            domain="other",
            section="analysis",
            supports=["benchmark/eval/e11_judge_ppl/e11_judge_ppl_result.json"],
            recompute="scripts/probe_readers.py::read_judge_ppl_correlation",
            caliber_ids=["T13"],
        ),
        # === QLoRA PPL↓分↓ ===
        ClaimSpec(
            claim_id="qlora_ppl_down_score_down",
            # 10-01 订正（D10）：旧句断言「行为分下降」，但那半边**从未被测**——本行仍是 S 级
            # 脚手架，且 PPL 件与行为分产物之间无登记 join（本行 supports 只有 ppl.json）。
            # 正文 A:56/75-76 写的是 behaviour "did not move"，与旧句方向不一致，故改在册陈述、
            # 不改正文。PPL 侧为实测：avg_fbc_delta_pct=-2.24（176 样本，positive_fbc_ratio 0.2045）。
            statement="历史 QLoRA epoch（89 题、单判，DATA-TRACE §5.1）：PPL 8.4→3.9（−53.6%），"
                      "同模型行为分 0.4669→0.4635（Δ=−0.0034，几乎不动）——此即 A 篇开场白"
                      "「拟合发生而能力未动」的实据。⚠ 两把 PPL 尺子不可混：本行原 supports 指的"
                      "standard_lora_ppl.json 是 176 样本、Δ=−2.24%，与故事讲的 −53.6% 不是同一件。"
                      "实物在 `_archive/`（仓外 NFS symlink），故本行仍留 S 级脚手架、不作证据引用",
            domain="other",
            section="analysis",
            # 10-01 记：故事实际依据的是 _archive 那对 89 题产物（见 DATA-TRACE §5.1），
            # 但它们在仓外 symlink、进不了调用链 ⇒ 本行只能继续留 S 级，不为离开 S 而指错实物。
            supports=["models/v2_formal_rebuilt/standard_lora_ppl.json"],
            recompute="scripts/inventory_experiment_assets.py::scan_ppl_assets",
            caliber_ids=["T14"],
        ),
        # === Wilcoxon+Holm（待算）===
        ClaimSpec(
            claim_id="wilcoxon_holm_significance",
            statement="核心主张的 Wilcoxon 检验经 Holm 校正后仍显著",
            domain="control",
            section="analysis",
            supports=["benchmark/eval/n3_regen/mgpo_eg_gen4.json"],
            recompute="papers/paper-a/figures/scripts/surface_stats.py::contrasts",
            caliber_ids=["T15"],
        ),
        # === 开卷 A 维 ===
        ClaimSpec(
            claim_id="openbook_dimension_a",
            statement="开卷控制组在 A 维（准确性）上大幅高于闭卷（四臂配对差 +0.42~+0.47，75 题）；本行为描述量对比、未做题级检验",
            domain="control",
            section="ablation",
            supports=["benchmark/eval/openbook_control/student_sft_baseline_openbook.json",
                      "release/adapter_manifest.json"],
            recompute="scripts/probe_readers.py::read_openbook_a",
            caliber_ids=["T17"],
        ),
        # === 锚点 0/23 与 21/23 ===
        ClaimSpec(
            claim_id="anchor_0_23",
            # 09-30 独立复查订正：0/23 不是「验证通过」，是三档无一题全序（构造效度限制）
            statement="锚点全序 0/23（fully_ordered=0、full_order_rate=0.0）：现行 v2 居中切片下三档无一题排全对，按 CALIBERS T21 归因为构造效度限制，只作披露，不作判官判别力证据",
            domain="control",
            section="surface",
            supports=["benchmark/eval/judge_audit/anchor_report.json"],
            recompute="scripts/check_figure_ledger.py::recompute_anchor",
            caliber_ids=["T21"],
        ),
        ClaimSpec(
            claim_id="anchor_21_23",
            # 09-30 独立复查订正：21 在件里是 ties（good==bad 打平），不是「排对 21 题」
            statement="锚点平局 21/23（91%，good==bad ties=21、inverted=2、faithful_ahead=0）：按 CALIBERS T21 属构造效度缺陷（good 是论文摘取窗口而非该题作答），不是判官量化弱点",
            domain="control",
            section="surface",
            supports=["benchmark/eval/judge_audit/anchor_report.json"],
            recompute="scripts/check_figure_ledger.py::recompute_anchor",
            caliber_ids=["T21"],
        ),
        # === 漂移 +0.0606 ===
        ClaimSpec(
            claim_id="drift_00606",
            statement="Judge 漂移聚合量为 +0.0606（11 模型 + 剔假 0 口径）",
            domain="control",
            section="limitations",
            # 真源按 §3.1：旧 epoch 答案固定重判（cross_judge/ 属隔离件，V7）
            supports=["benchmark/eval/run1_rejudge/raw_9b_run1p.json",
                      "scripts/verify_judge_drift_aggregate.py",
                      "docs/reports/analysis-judge-drift-run1-rejudge.md"],
            recompute="scripts/check_figure_ledger.py::recompute_drift",
            caliber_ids=["T11", "T18"],
        ),
        # === 仲裁退化 114 ===
        ClaimSpec(
            claim_id="arbitration_degradation_114",
            statement="仲裁退化 114 题次已补判，B1 由贴线过关变稳健过关",
            domain="control",
            section="limitations",
            supports=["benchmark/eval/arbitration_patch/patch_summary.json"],
            recompute="scripts/run_arbitration_patch.py::scan_failures",
            caliber_ids=["T16"],
        ),
        # === E10 位置偏差 ===
        ClaimSpec(
            claim_id="e10_position_bias",
            # 旧句「位置偏差显著（>10%）」与 §3.1c 定案方向相反（value-version-audit V5），
            # 现按定案改写；B 篇正文 main.tex:253-256 本来就是这个方向，无需改论文。
            statement="E10 判序交换后 100 题中 94 题终分不变（mean(A−B)=+0.0175, sd 0.1138），6 题变化全可归因单判官采样噪声：聚合机制（平均与序无关仲裁）不引入位置效应",
            domain="control",
            section="limitations",
            supports=["benchmark/eval/e10_position_bias/mgpo_eg_gen3__swapped.json"],
            recompute="scripts/probe_readers.py::read_position_bias",
            caliber_ids=["T19"],
        ),
        # === E12 争议率 ===
        ClaimSpec(
            claim_id="e12_dispute_rate",
            statement="E12 争议率分母口径有两种（12/9 产物 + 5 个第三种）",
            domain="control",
            section="limitations",
            supports=["benchmark/eval/judge_audit/fidelity_report.json"],
            recompute="scripts/check_product_self_consistency.py::dispute_rate_denominator_note",
            caliber_ids=["T12", "T20"],
        ),
        # === σg 阈值族 ===
        ClaimSpec(
            claim_id="sigma_g_threshold_family",
            statement="σg 阈值族仅 4B/9B 两点已标定（T9），27B 无 σg（未训练、无 n=3 重生成）；9B 的轴已按 T16 结案为 between-generation，残留为代子集的历史偶然",
            domain="control",
            section="limitations",
            supports=["benchmark/eval/judge_audit/anchor_state.json"],
            recompute="benchmark/eval/counterfactual/analysis.json#4b/statistics/judgement/sigma",
            caliber_ids=["T16", "S7"],
        ),
    ]


def recompute_all(claims: list[ClaimSpec]) -> list[dict]:
    """复算所有主张（占位实现，Task 7 完善）"""
    if not claims:
        return []
    # TODO: Task 7 实现具体复算逻辑
    return [{"claim_id": c.claim_id, "status": "pending"} for c in claims]
