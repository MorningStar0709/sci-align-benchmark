"""RAG synthesis service — extracted from controlsci.api.medical_rag.

Pure library functions (no FastAPI dependency) for evidence synthesis pipeline:
question profiling → evidence cards → prompt building → local LLM chat → answer normalization.
"""

from __future__ import annotations

import json
import re



# ── Question profiling ─────────────────────────────────────────────


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _dedupe_words(text: str) -> str:
    seen = set()
    words = []
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", text or ""):
        key = word.lower()
        if key not in seen:
            seen.add(key)
            words.append(word)
    return " ".join(words)


def _unique_queries(items: list[str]) -> list[str]:
    seen = set()
    queries = []
    for item in items:
        text = _dedupe_words(item)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            queries.append(text)
    return queries


def _classify_question_profile(query: str) -> dict:
    text = (query or "").lower()
    categories = [
        (
            "endpoint_definition",
            ("终点定义", "如何定义", "怎么定义", "定义为", "endpoint definition", "defined as"),
            ["结论", "终点口径", "不可外推", "局限"],
            "说明研究如何定义终点，同时避免把终点口径解释为疗效结论。",
        ),
        (
            "survival_benefit_judgement",
            ("生存获益", "延长生存", "总生存", "无进展生存", "os", "pfs", "rpfs", "overall survival", "progression free"),
            ["结论", "终点定义", "获益判断", "局限"],
            "区分 OS/PFS 终点定义、统计分析和是否足以支持生存获益结论。",
        ),
        (
            "mechanism_explanation",
            ("为什么", "为何", "原因", "机制", "意义", "说明什么", "支持", "背后", "怎么解释", "如何解释", "why", "mechanism", "rationale"),
            ["结论", "机制解释", "适用边界", "安全声明"],
            "解释文献结论背后的机制或逻辑链条，同时明确证据边界。",
        ),
        (
            "safety_judgement",
            ("安全", "风险", "不良事件", "严重不良事件", "低血糖", "毒性", "adverse", "safety", "hypogly"),
            ["结论", "安全性发现", "适用边界", "安全声明"],
            "围绕安全性结局、不良事件和适用边界回答。",
        ),
        (
            "efficacy_judgement",
            ("主要终点", "次要终点", "疗效", "获益", "有效", "生存", "endpoint", "survival", "efficacy"),
            ["结论", "关键终点", "证据强度", "局限"],
            "围绕终点、疗效方向和证据强度回答。",
        ),
        (
            "population_applicability",
            ("人群", "纳入", "排除", "适用", "itt", "intention", "population", "inclusion", "exclusion"),
            ["结论", "研究人群", "外推边界", "局限"],
            "说明研究对象、纳排标准和结果可外推边界。",
        ),
        (
            "study_design",
            ("研究设计", "随机", "对照", "统计分析", "方法", "design", "randomized", "statistical"),
            ["结论", "设计要点", "统计口径", "局限"],
            "解释研究设计、统计口径和方法学边界。",
        ),
    ]
    for question_type, triggers, strategy, instruction in categories:
        if any(token in text for token in triggers):
            return {
                "question_type": question_type,
                "answer_strategy": strategy,
                "strategy_instruction": instruction,
            }
    return {
        "question_type": "evidence_summary",
        "answer_strategy": ["结论", "关键证据", "适用边界", "局限"],
        "strategy_instruction": "汇总检索证据中的直接结论，并明确不能外推的部分。",
    }


# ── Evidence role classification ───────────────────────────────────


_EVIDENCE_RULES: list[tuple[tuple[str, ...], str]] = [
    (("defined as", "definition", "endpoint was", "endpoint is", "duration from", "time from"), "endpoint_definition"),
    (("overall survival", "progression-free", "progression free", "kaplan", "hazard ratio", "median survival"), "survival_outcome"),
    (("adverse", "safety", "hypogly", "toxicity"), "safety"),
    (("primary", "secondary", "outcome", "endpoint"), "endpoint"),
    (("population", "eligibility", "inclusion", "exclusion", "itt"), "population"),
    (("intervention", "dose", "treatment"), "intervention"),
    (("statistical", "analysis", "method", "random"), "method"),
]


def _evidence_role(item: dict) -> str:
    blob = " ".join(
        str(item.get(key) or "").lower()
        for key in ("medical_label", "medical_parent", "source_section", "text", "text_preview")
    )
    for tokens, role in _EVIDENCE_RULES:
        if any(token in blob for token in tokens):
            return role
    return "context"


def _is_survival_endpoint_question(query: str) -> bool:
    text = (query or "").lower()
    return any(token in text for token in ["os", "pfs", "rpfs", "总生存", "无进展", "生存获益", "延长生存", "overall survival", "progression free"])


def _support_level_for_card(role: str, question_type: str, text: str) -> tuple[str, str]:
    blob = (text or "").lower()
    direct_roles = {
        "endpoint_definition": {"endpoint_definition", "endpoint", "survival_outcome"},
        "survival_benefit_judgement": {"survival_outcome", "endpoint_definition", "method", "population"},
        "safety_judgement": {"safety", "endpoint"},
        "mechanism_explanation": {"safety", "intervention", "survival_outcome", "endpoint_definition", "context"},
        "population_applicability": {"population", "method"},
        "study_design": {"method", "population", "endpoint"},
        "efficacy_judgement": {"survival_outcome", "endpoint", "method", "population"},
    }
    if role in direct_roles.get(question_type, set()):
        return "direct", "与问题类型直接匹配"
    if any(token in blob for token in ("secondary endpoint", "primary endpoint", "statistical analysis", "intention-to-treat")):
        return "supporting", "可支持统计或终点边界"
    return "context", "仅作背景，不能单独支撑结论"


def _build_evidence_cards(evidence_items: list[dict], question_profile: dict | None = None) -> list[dict]:
    cards = []
    question_type = (question_profile or {}).get("question_type", "evidence_summary")
    for idx, item in enumerate(evidence_items, start=1):
        text = item.get("text") or item.get("text_preview") or ""
        role = _evidence_role(item)
        support_level, support_reason = _support_level_for_card(role, question_type, text)
        cards.append(
            {
                "card_id": f"E{idx}",
                "chunk_id": item.get("chunk_id", ""),
                "source_file": item.get("source_file", "unknown"),
                "section": item.get("source_section") or "-",
                "label": item.get("medical_label") or item.get("medical_parent") or "-",
                "role": role,
                "support_level": support_level,
                "support_reason": support_reason,
                "score": round(float(item.get("rrf_score") or item.get("_rrf_score") or 0.0), 6),
                "key_text": (text or "")[:260].replace("\n", " "),
            }
        )
    return cards


# ── Prompt building ────────────────────────────────────────────────


def _is_mechanism_question(query: str) -> bool:
    text = query or ""
    triggers = [
        "为什么",
        "为何",
        "原因",
        "机制",
        "意义",
        "说明什么",
        "支持",
        "背后",
        "怎么解释",
        "如何解释",
        "why",
        "mechanism",
        "rationale",
    ]
    return any(token.lower() in text.lower() for token in triggers)


def _build_evidence_blocks(evidence_items: list[dict]) -> list[str]:
    """Build formatted evidence text blocks from evidence items."""
    blocks = []
    for item in evidence_items:
        text = item.get("text") or item.get("text_preview") or ""
        blocks.append(
            "\n".join(
                [
                    f"[{item.get('chunk_id')}]",
                    f"section: {item.get('source_section') or '-'}",
                    f"label: {item.get('medical_parent') or '-'} / {item.get('medical_label') or '-'}",
                    f"evidence: {text[:1800]}",
                ]
            )
        )
    return blocks


def _build_rag_prompt(
    query: str,
    evidence_items: list[dict],
    search_query: str | None = None,
    question_profile: dict | None = None,
    evidence_cards: list[dict] | None = None,
) -> list[dict]:
    question_profile = question_profile or _classify_question_profile(query)
    evidence_cards = evidence_cards or _build_evidence_cards(evidence_items, question_profile)
    evidence_blocks = _build_evidence_blocks(evidence_items)

    system = (
        "你是一个本地运行的医学文献 RAG 合成器。"
        "只能依据提供的 evidence chunks 作答，不得使用外部知识或编造医学结论。"
        "如果证据不足，必须拒答。"
        "输出必须是 JSON，不要输出 markdown。"
    )
    strategy_text = " -> ".join(question_profile.get("answer_strategy") or [])
    card_lines = [
        f"{card['card_id']} | {card['chunk_id']} | role={card['role']} | label={card['label']} | {card['key_text']}"
        for card in evidence_cards
    ]
    explanation_mode = question_profile.get("question_type") == "mechanism_explanation" or _is_mechanism_question(query)
    explanation_line = (
        "\n本问题属于机制解释/原因解释型问题。回答时必须先给出文献支持的结论，再解释可能机制或逻辑链条；"
        "不得只复述用户问题或证据原句。如果证据只能支持结论、不能支持机制，必须明确写出机制证据不足。\n"
        if explanation_mode
        else ""
    )
    survival_endpoint_line = (
        "\n本问题涉及 OS/PFS/rPFS、生存获益或终点判断。回答时必须区分："
        "1）单篇研究中 OS/PFS 的 endpoint definition；2）该终点是否足以支持治疗带来生存获益的 clinical conclusion。"
        "不得把某一篇研究的起算日期、事件定义或影像学进展规则泛化为所有研究的通用定义；"
        '不得把“报告了 OS/PFS”写成“已经证明延长生存”。如来源只支持终点口径，必须明确说明这只是研究定义，不是获益结论。\n'
        if any(token in (query or "").lower() for token in ["os", "pfs", "rpfs", "总生存", "无进展", "生存获益", "影像学进展"])
        else ""
    )
    search_line = f"\n英文检索词：\n{search_query}\n" if search_query and search_query != query else ""
    user = f"""用户问题：
{query}
{search_line}
{explanation_line}
{survival_endpoint_line}

问题类型：
{question_profile.get('question_type')}

回答策略：
{strategy_text}

结构化证据卡：
{chr(10).join(card_lines)}

本地检索证据：
{chr(10).join(evidence_blocks)}

请输出 JSON，字段如下：
{{
  "answer": "中文回答。必须引用 chunk_id，例如 [PMCxxx_chunk_001]。",
  "question_type": "{question_profile.get('question_type')}",
  "answer_strategy": {json.dumps(question_profile.get("answer_strategy") or [], ensure_ascii=False)},
  "citations": ["chunk_id"],
  "claims": [
    {{"claim": "可由证据支持的中文结论。", "citations": ["chunk_id"], "supported": true}}
  ],
  "reasoning_trace": [
    {{"step": "识别问题类型/匹配证据卡/组织回答", "citations": ["chunk_id"], "status": "supported"}}
  ],
  "confidence": "high|medium|low",
  "limitations": "证据局限。",
  "abstain": false,
  "abstain_reason": ""
}}

规则：
1. 只允许引用上面出现的 chunk_id。
2. 每个事实性结论必须写入 claims，并给出支撑它的 chunk_id。
3. 如果某个结论没有证据支撑，supported=false 且 citations=[]。
4. 如果证据不能支持回答，answer 写"当前本地知识库证据不足，无法回答该问题。"，abstain=true。
5. 不提供诊疗建议，不替代医生判断。
6. 对"为什么/机制/原因/意义/支持使用"类问题，answer 按"结论、机制解释、适用边界、安全声明"组织；机制解释只能来自 evidence，不能用外部常识补全。
7. 对 OS/PFS/rPFS/生存获益类问题，必须先写清"终点定义"和"获益结论"的区别；若只检索到单篇研究定义，只能回答"该研究如何定义终点"，不能推导疗效获益。
8. reasoning_trace 只记录可由 evidence 支撑的步骤，不输出隐藏推理，不得包含未引用的医学断言。
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ── JSON extraction ────────────────────────────────────────────────


def _extract_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


# ── Evidence sufficiency scoring ───────────────────────────────────


def _tokenize_query(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_\-]{3,}", text or "")}


def _score_evidence_sufficiency(query: str, evidence_items: list[dict]) -> dict:
    query_terms = _tokenize_query(query)
    evidence_terms = _tokenize_query(" ".join((item.get("text") or item.get("text_preview") or "") for item in evidence_items))
    matched = sorted(query_terms & evidence_terms)
    max_rrf = max((float(item.get("rrf_score") or 0.0) for item in evidence_items), default=0.0)
    has_evidence = bool(evidence_items)
    sufficient = has_evidence and (not query_terms or len(matched) >= min(2, len(query_terms)) or max_rrf > 0)
    return {
        "sufficient": sufficient,
        "query_terms": sorted(query_terms)[:20],
        "matched_terms": matched[:20],
        "matched_term_count": len(matched),
        "top_rrf_score": max_rrf,
        "evidence_count": len(evidence_items),
    }


# ── Local LLM chat (library-safe, no FastAPI dependency) ──────────────


def _call_local_llm_chat(model: str, messages: list[dict], timeout: float = 180.0) -> tuple[dict, dict]:
    """Call local LLM via llama.cpp OpenAI-compatible API.

    Args:
        model: Model name (informational; llama.cpp uses the loaded model).
        messages: OpenAI-format messages list.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (parsed JSON dict, raw response dict).
    """
    from mineru_tools.llm_client import LocalLLMClient

    client = LocalLLMClient(timeout=timeout)
    try:
        result = client.chat(
            messages=messages,
            model=model,
            temperature=0.1,
            max_tokens=1200,
            response_format="json",
        )
    except Exception as e:
        raise RuntimeError(f"Local LLM generation failed: {e}") from e

    parsed = _extract_json_object(result.content)
    if not parsed:
        parsed = {
            "answer": result.content.strip() or "Local model did not return a valid response.",
            "citations": [],
            "confidence": "low",
            "limitations": "Model did not return structured JSON output.",
            "abstain": False,
            "abstain_reason": "",
        }
    return parsed, result.raw


# ── Answer normalization ───────────────────────────────────────────


def _normalize_claims(raw: dict, evidence_items: list[dict]) -> tuple[list[dict], dict]:
    allowed = {item["chunk_id"] for item in evidence_items if item.get("chunk_id")}
    claims = []
    raw_claims = raw.get("claims")
    if not isinstance(raw_claims, list):
        raw_claims = []

    for entry in raw_claims:
        if not isinstance(entry, dict):
            continue
        claim = str(entry.get("claim") or "").strip()
        if not claim:
            continue
        claim_citations = [c for c in entry.get("citations", []) if c in allowed]
        supported = bool(entry.get("supported", False)) and bool(claim_citations)
        claims.append(
            {
                "claim": claim,
                "citations": claim_citations,
                "supported": supported,
            }
        )

    if not claims:
        answer = str(raw.get("answer") or "").strip()
        raw_citations = [c for c in raw.get("citations", []) if c in allowed]
        if answer:
            claims.append(
                {
                    "claim": answer,
                    "citations": raw_citations,
                    "supported": bool(raw_citations) and not bool(raw.get("abstain", False)),
                }
            )

    supported_count = sum(1 for claim in claims if claim["supported"])
    unsupported_count = len(claims) - supported_count
    coverage = supported_count / len(claims) if claims else 0.0
    return claims, {
        "claim_count": len(claims),
        "supported_claims": supported_count,
        "unsupported_claims": unsupported_count,
        "citation_coverage": round(coverage, 4),
    }


def _normalize_reasoning_trace(raw: dict, evidence_items: list[dict]) -> list[dict]:
    allowed = {item["chunk_id"] for item in evidence_items if item.get("chunk_id")}
    raw_steps = raw.get("reasoning_trace")
    if not isinstance(raw_steps, list):
        raw_steps = []

    trace = []
    for idx, entry in enumerate(raw_steps[:6], start=1):
        if not isinstance(entry, dict):
            continue
        citations = [c for c in entry.get("citations", []) if c in allowed]
        step = str(entry.get("step") or "").strip()
        if not step:
            continue
        status = str(entry.get("status") or ("supported" if citations else "unsupported")).lower()
        if status not in {"supported", "unsupported", "partial"}:
            status = "supported" if citations else "unsupported"
        trace.append({"step": step, "citations": citations, "status": status})

    if not trace and evidence_items:
        trace.append(
            {
                "step": "检索到本地证据并按问题类型组织回答",
                "citations": [item["chunk_id"] for item in evidence_items[:2] if item.get("chunk_id")],
                "status": "supported",
            }
        )
    return trace


def _check_claim_coverage(claims, evidence_items, evidence_cards):
    """Check citation coverage and direct source cards."""
    issues = []
    passed = []

    allowed = {item["chunk_id"] for item in evidence_items if item.get("chunk_id")}
    cited = {c for claim in claims for c in claim.get("citations", [])}
    if cited and cited <= allowed:
        passed.append("所有 claim 引用均来自本轮检索来源")
    else:
        issues.append("存在未引用或引用覆盖不足的 claim")

    direct_cards = [card for card in evidence_cards if card.get("support_level") == "direct"]
    if direct_cards:
        passed.append(f"命中 {len(direct_cards)} 张直接来源卡")
    else:
        issues.append("未命中直接来源卡，回答应降级为依据不足或背景说明")

    unsupported = [claim for claim in claims if not claim.get("supported")]
    if unsupported:
        issues.append(f"{len(unsupported)} 条 claim 未被来源支撑")
    else:
        passed.append("claim-level 支撑检查通过")

    return issues, passed, direct_cards


def _check_survival_boundary(query, question_type, answer, issues, passed):
    """Check OS/PFS survival boundary in the answer."""
    if not (_is_survival_endpoint_question(query) or question_type in {"endpoint_definition", "survival_benefit_judgement"}):
        return

    has_definition_boundary = any(token in answer for token in ("终点定义", "终点口径", "定义"))
    has_conclusion_boundary = any(token in answer for token in (
        "不等于", "不等同", "不能自动", "不是", "不足以",
        "不能推导", "无法直接推导", "不能外推", "不能将",
        "不能仅因", "并非直接",
    ))
    if has_definition_boundary and has_conclusion_boundary:
        passed.append("OS/PFS 终点定义与获益结论边界已写明")
    else:
        issues.append('OS/PFS 类问题缺少"终点定义 vs 获益结论"的边界')
    if (
        re.search(r"(?<!已)(证明|证实|显著).{0,12}(延长|改善).{0,12}(生存|OS|PFS)", answer, flags=re.IGNORECASE)
        and not has_conclusion_boundary
    ):
        issues.append("疑似把终点报告泛化为生存获益结论")


_PERSONAL_ADVICE_PATTERN = re.compile(
    r"(你|您|患者|本人).{0,16}(应该|建议|需要|可以).{0,16}(吃|服用|使用|调整|加量|减量|停药|剂量|治疗)"
    r"|(?:应该|建议).{0,12}(吃什么药|服用|用药|剂量|治疗方案)"
    r"|(?:take|use|increase|decrease|stop).{0,16}(?:medicine|drug|dose|dosage)",
    flags=re.IGNORECASE,
)


def _check_personal_advice(answer, issues, passed):
    """Check for personal medical advice in the answer."""
    if _PERSONAL_ADVICE_PATTERN.search(answer):
        issues.append("回答中疑似包含个人诊疗或用药建议")
    else:
        passed.append("未发现个人诊疗建议")


def _review_rag_answer(
    query: str,
    raw: dict,
    evidence_items: list[dict],
    claims: list[dict],
    question_profile: dict,
    evidence_cards: list[dict],
) -> dict:
    answer = str(raw.get("answer") or "")
    question_type = question_profile.get("question_type", "evidence_summary")
    issues, passed, direct_cards = _check_claim_coverage(claims, evidence_items, evidence_cards)
    _check_survival_boundary(query, question_type, answer, issues, passed)
    _check_personal_advice(answer, issues, passed)

    return {
        "status": "needs_review" if issues else "passed",
        "passed": passed,
        "issues": issues,
        "direct_source_cards": [card["card_id"] for card in direct_cards],
        "question_type": question_type,
    }


def _resolve_abstention(evidence_items, claim_stats, answer_audit, citations, claims, abstain, answer):
    """根据证据充分性、claim 支持和审计结果，决定是否拒答。"""
    if not evidence_items:
        abstain = True
    if claim_stats["claim_count"] and claim_stats["supported_claims"] == 0:
        abstain = True
    if answer_audit["status"] != "passed" and not citations:
        abstain = True
    if abstain and not answer:
        answer = "当前本地知识库证据不足，无法回答该问题。"
    if not citations and claims:
        citations = sorted({c for claim in claims for c in claim.get("citations", [])})
    if not citations and not abstain:
        citations = [item["chunk_id"] for item in evidence_items[:2] if item.get("chunk_id")]
    return abstain, answer, citations



def _resolve_confidence(confidence, abstain, citation_coverage):
    """规范化置信度等级。"""
    confidence = str(confidence or "medium").lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    if abstain or citation_coverage < 0.5:
        confidence = "low"
    return confidence



def _normalize_rag_answer(
    raw: dict,
    evidence_items: list[dict],
    question_profile: dict | None = None,
    evidence_cards: list[dict] | None = None,
    original_query: str | None = None,
) -> dict:
    allowed = {item["chunk_id"] for item in evidence_items if item.get("chunk_id")}
    citations = [c for c in raw.get("citations", []) if c in allowed]
    answer = str(raw.get("answer") or "").strip()
    abstain = bool(raw.get("abstain", False))
    claims, claim_stats = _normalize_claims(raw, evidence_items)
    reasoning_trace = _normalize_reasoning_trace(raw, evidence_items)
    evidence_sufficiency = _score_evidence_sufficiency(raw.get("query", ""), evidence_items)
    question_profile = question_profile or _classify_question_profile(original_query or raw.get("original_query") or raw.get("query") or "")
    evidence_cards = evidence_cards or _build_evidence_cards(evidence_items, question_profile)
    answer_audit = _review_rag_answer(original_query or raw.get("original_query") or raw.get("query") or "", raw, evidence_items, claims, question_profile, evidence_cards)
    abstain, answer, citations = _resolve_abstention(
        evidence_items, claim_stats, answer_audit, citations, claims, abstain, answer
    )
    confidence = _resolve_confidence(raw.get("confidence"), abstain, claim_stats["citation_coverage"])

    return {
        "answer": answer or "当前本地知识库证据不足，无法回答该问题。",
        "citations": citations,
        "claims": claims,
        "reasoning_trace": reasoning_trace,
        "claim_count": claim_stats["claim_count"],
        "supported_claims": claim_stats["supported_claims"],
        "unsupported_claims": claim_stats["unsupported_claims"],
        "citation_coverage": claim_stats["citation_coverage"],
        "evidence_sufficiency": evidence_sufficiency,
        "confidence": confidence,
        "limitations": str(raw.get("limitations") or "仅基于当前本地医学文献知识库检索结果，不构成诊疗建议。"),
        "abstain": abstain,
        "abstain_reason": str(raw.get("abstain_reason") or ("证据不足" if abstain else "")),
        "answer_audit": answer_audit,
    }
