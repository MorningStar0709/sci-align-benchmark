"""Public medical demo — hardcoded public PMC chunks + retrieval + synthesis.

This module provides a lightweight, self-contained medical RAG for the
pure-cloud demo profile. It uses a small curated set of public PMC excerpts
and requires no local index or Ollama.
"""

from __future__ import annotations

import re


PUBLIC_MEDICAL_CHUNKS: list[dict[str, str]] = [
    {
        "id": "pmc-t1d-closed-loop-48m-01",
        "title": "Adaptive closed-loop insulin delivery after diagnosis in children and adolescents",
        "source": "公开 PMC 糖尿病闭环胰岛素随访研究",
        "topic": "closed_loop_insulin",
        "text": (
            "In children and adolescents with newly diagnosed type 1 diabetes, daily insulin requirements increased "
            "substantially during the first 48 months after diagnosis. The proportion of insulin delivered by the "
            "closed-loop algorithm also increased over time, suggesting that adaptive closed-loop systems can respond "
            "to changing insulin needs during early disease progression."
        ),
        "zh": (
            "新诊断 1 型糖尿病儿童和青少年在诊断后 48 个月内每日胰岛素需求明显上升；闭环算法输送的胰岛素比例也随时间增加，"
            "说明自适应闭环系统的意义在于跟随需求变化调整输送，而不是给出固定剂量。"
        ),
    },
    {
        "id": "pmc-t1d-hypoglycaemia-02",
        "title": "Closed-loop therapy and hypoglycaemia safety endpoints",
        "source": "公开 PMC 闭环胰岛素安全性终点研究",
        "topic": "hypoglycaemia",
        "text": (
            "Trials of hybrid closed-loop therapy commonly report time below range and severe hypoglycaemia as safety "
            "endpoints. Interpretation should distinguish population-level safety outcomes from individual treatment "
            "decisions, because personal dosing requires clinician assessment and device-specific data."
        ),
        "zh": (
            "闭环治疗研究通常把低于目标血糖范围时间和严重低血糖作为安全性终点。文献层面的安全性结论只能说明群体趋势，"
            "不能替代个体剂量调整或设备处方。"
        ),
    },
    {
        "id": "pmc-chemo-dose-modification-03",
        "title": "Chemotherapy dose modification in oncology trials",
        "source": "公开肿瘤治疗试验方案摘要",
        "topic": "chemotherapy_dose",
        "text": (
            "Oncology trial protocols usually define dose modification rules for toxicity, organ function, and treatment "
            "delays. These rules are protocol-level safeguards and should not be used as patient-specific medication advice "
            "without full clinical context."
        ),
        "zh": (
            "肿瘤试验方案中的化疗剂量调整规则通常围绕毒性、器官功能和延迟治疗设定，是研究方案层面的安全控制，"
            "不能直接当作个人用药建议。"
        ),
    },
    {
        "id": "pmc-survival-itt-04",
        "title": "Intention-to-treat population and survival endpoints",
        "source": "公开临床试验统计方法摘要",
        "topic": "itt_survival",
        "text": (
            "Intention-to-treat analysis preserves the original randomized groups and is widely used for primary efficacy "
            "analysis. Survival endpoints require careful censoring rules, follow-up definitions, and transparent reporting "
            "of excluded or missing data."
        ),
        "zh": (
            "ITT 分析保留原始随机分组，常用于主要疗效分析。生存获益判断需要同时说明删失规则、随访定义以及缺失数据处理。"
        ),
    },
    {
        "id": "pmc-inclusion-exclusion-05",
        "title": "Eligibility criteria in clinical trials",
        "source": "公开临床试验纳入排除标准摘要",
        "topic": "eligibility",
        "text": (
            "Eligibility criteria define the population to which a trial result most directly applies. Inclusion and exclusion "
            "rules protect participants and reduce confounding, but they also limit generalization to patients outside the "
            "studied population."
        ),
        "zh": (
            "纳入排除标准决定研究结论最直接适用的人群。它既保护受试者并减少混杂，也会限制结论向真实世界患者外推。"
        ),
    },
]

MEDICAL_QUERY_HINTS: dict[str, list[str]] = {
    "closed_loop_insulin": ["闭环", "胰岛素", "低血糖", "48", "儿童", "青少年", "insulin", "closed", "loop", "hypoglycaemia", "hypoglycemia"],
    "hypoglycaemia": ["闭环", "胰岛素", "低血糖", "安全", "终点", "insulin", "closed", "loop", "hypoglycaemia", "hypoglycemia", "safety"],
    "chemotherapy_dose": ["化疗", "剂量", "毒性", "减量", "chemotherapy", "dose", "toxicity"],
    "itt_survival": ["生存", "获益", "itt", "人群", "survival", "endpoint", "intention"],
    "eligibility": ["纳入", "排除", "标准", "eligibility", "inclusion", "exclusion"],
}


def retrieve_public_medical_chunks(question: str, limit: int = 3) -> list[dict]:
    """Score and retrieve the most relevant public medical chunks."""
    lowered = question.lower()
    scored: list[tuple[int, dict[str, str]]] = []
    for chunk in PUBLIC_MEDICAL_CHUNKS:
        haystack = " ".join([chunk["title"], chunk["topic"], chunk["text"], chunk["zh"]]).lower()
        score = 0
        for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", lowered):
            if token and token in haystack:
                score += 2
        for topic, hints in MEDICAL_QUERY_HINTS.items():
            if topic == chunk["topic"]:
                score += sum(1 for hint in hints if hint.lower() in lowered)
        if score:
            scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "id": chunk["id"],
            "title": chunk["title"],
            "source": chunk["source"],
            "score": score,
            "text": chunk["text"],
            "zh": chunk["zh"],
        }
        for score, chunk in scored[:limit]
    ]


_ANSWER_TEMPLATES: list[tuple[set, str]] = [
    (
        {"闭环", "insulin", "hypogly"},
        (
            "机制解释：公开来源显示，新诊断 1 型糖尿病儿童和青少年在诊断后 48 个月内每日胰岛素需求明显上升，"
            "闭环算法输送的胰岛素比例也随时间增加。因此，这个案例支持的是「自适应系统能跟随需求变化调整输送」的机制，"
            "而不是「给出某个固定剂量」。安全性需要看低于目标血糖范围时间、严重低血糖等终点，并且只能解释群体层面的研究结论。"
            "这不是个人治疗、处方或紧急诊断。"
        ),
    ),
    (
        {"os", "pfs", "rpfs", "总生存", "无进展", "生存获益"},
        (
            "机制解释：这里要先区分「终点定义」和「生存获益结论」。OS 通常是更硬的生存终点；"
            "PFS/rPFS 是进展相关终点，可以说明疾病控制时间，但不能自动等同于已经证明延长总生存。"
            "如果来源只给出从首次给药到死亡或影像学进展的计算规则，那只是该研究的 endpoint definition，"
            "不是治疗获益结论。\n\n"
            "边界：判断生存获益还需要看随机化/ITT 人群、删失规则、随访成熟度、对照组差异和最终 OS 分析。"
            "这个回答只解释公开文献中的统计终点，不推断具体患者预后。"
        ),
    ),
    (
        {"itt", "intention", "获益"},
        (
            "机制解释：ITT 分析保留原始随机分组，能减少因退出、换组或治疗依从性差异带来的选择偏依。"
            "因此，当问题是「生存获益是否成立」时，不能只看完成治疗人群，还要看随机化人群、删失规则、随访定义和缺失数据。"
            "这个回答只说明文献统计方法的适用边界，不推断具体患者预后。"
        ),
    ),
    (
        {"eligibility", "inclusion", "exclusion", "纳入", "排除"},
        (
            "机制解释：纳入排除标准决定研究结论最直接适用的人群。严格标准能降低混杂、保护受试者，"
            "但也会让试验人群和真实世界患者产生差距，所以外推结论时必须说明哪些人群没有被研究覆盖。"
            "这个回答只讨论研究设计边界，不判断个体是否应入组或接受治疗。"
        ),
    ),
    (
        {"chemotherapy", "dose", "toxicity", "化疗", "剂量"},
        (
            "机制解释：化疗剂量调整规则通常来自试验方案，对毒性、器官功能和延迟治疗做预设处理。"
            "它的作用是保证研究执行一致和受试者安全，而不是把方案规则直接变成个人用药建议。"
            "具体剂量必须由临床医生结合完整病情判断。"
        ),
    ),
]


def _match_answer_template(question: str) -> str | None:
    """Match question keywords to a predefined answer template."""
    lowered = question.lower()
    for keywords, template in _ANSWER_TEMPLATES:
        for kw in keywords:
            if kw in lowered or kw in question:
                return template
    return None


def synthesize_medical_answer(question: str, chunks: list[dict]) -> str:
    """Generate a deterministic answer from retrieved public medical chunks.

    This is a rules-based synthesizer for the pure-cloud demo profile.
    It matches question keywords to predefined answer templates.
    """
    if not chunks:
        return (
            "当前公开来源回放集中没有检索到足够来源，不能强行回答。云端 Demo 只展示公开/脱敏医学文献的来源回放；"
            "完整系统会使用更大的医学索引、混合检索和 claim 校验。"
        )
    source_lines = "\n".join([f"- {chunk['id']}：{chunk['zh']}" for chunk in chunks])
    lowered = question.lower()

    template = _match_answer_template(question)
    if template:
        return template + "\n\n" + f"来源摘要：\n{source_lines}"

    return (
        "根据公开来源，当前问题可以从研究设计、适用人群和安全边界三个层面理解。系统先匹配公开来源，"
        "再基于命中来源合成中文回答；如果问题转向个人诊疗或用药，云端 Demo 会保留边界而不直接给建议。\n\n"
        f"来源摘要：\n{source_lines}"
    )
