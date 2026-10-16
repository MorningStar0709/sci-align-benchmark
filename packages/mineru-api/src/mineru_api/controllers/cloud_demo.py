"""Cloud demo controller — pure-cloud demo endpoints.

Migrated from ``cloud_demo_standalone.app``.
All routes are under ``/api/cloud-demo`` prefix to avoid conflicts with
existing mineru-api routes.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from mineru_core.paths import PROJECT_ROOT

from mineru_api.schemas.cloud_demo import (
    CloudDemoAgentPlanRequest,
    DeepSeekAskRequest,
    MedicalRagRequest,
    QuizGenerateRequest,
    QuizGradeRequest,
    UrlParseRequest,
)
from mineru_api.services.cloud_demo import (
    cloud_demo_settings,
    is_personal_medical_question,
    public_boundary,
    require_access,
    validate_public_extract_url,
)
from mineru_api.services.deepseek_client import call_deepseek
from mineru_api.services.mineru_official import parse_mineru_official_upload
from mineru_api.services.quota import consume_quota, quota_status

from mineru_medical.public_demo import (
    retrieve_public_medical_chunks,
    synthesize_medical_answer,
)

router = APIRouter(prefix="/api/cloud-demo", tags=["cloud-demo"])


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


@router.get("/")
def cloud_demo_index() -> dict:
    return {
        "name": "MinerU Pure Cloud Demo API",
        "ui": "Serve the Next.js workbench from cloud_demo_standalone/web or docker compose.",
        "health": "/api/cloud-demo/health",
        "runtime": "/api/cloud-demo/runtime",
    }


# ---------------------------------------------------------------------------
# Health / Runtime / Quota / Tracks
# ---------------------------------------------------------------------------


@router.get("/health")
def cloud_demo_health() -> dict:
    cfg = cloud_demo_settings()
    issues = []
    if not cfg["mineru_token"]:
        issues.append("MINERU_API_TOKEN 未配置")
    if not cfg["deepseek_key"]:
        issues.append("DEEPSEEK_API_KEY 未配置")
    return {
        "status": "degraded" if issues else "ready",
        "profile": "pure_cloud_demo",
        "runtime": "mineru_api",
        "components": {
            "mineru_official_api": {"available": bool(cfg["mineru_token"]), "base_url": cfg["mineru_base_url"]},
            "deepseek_api": {"available": bool(cfg["deepseek_key"]), "base_url": cfg["deepseek_base"], "model": cfg["deepseek_model"]},
            "quota": quota_status(),
        },
        "limits": {
            "upload_max_mb": cfg["upload_max_mb"],
            "daily_live_calls": cfg["daily_limit"],
            "access_code_required": bool(cfg["require_access_code"]),
        },
        "boundary": public_boundary(),
        "issues": issues,
    }


@router.get("/runtime")
def cloud_demo_runtime() -> dict:
    return {
        "mode": "pure_cloud_only",
        "profiles": [{"id": "pure_cloud_demo", "label": "公开云端 Demo"}],
        "parser_backends": [{"id": "mineru_official", "label": "MinerU 官方 API"}],
        "generation_backends": [{"id": "deepseek", "label": "DeepSeek API"}],
        "local_models": [],
        "local_indexes": [],
        "defaults": {
            "profile": "pure_cloud_demo",
            "parser_backend": "mineru_official",
            "generation_backend": "deepseek",
            "data_boundary": "public_or_desensitized_only",
        },
    }


@router.get("/quota")
def cloud_demo_quota() -> dict:
    return quota_status()


@router.get("/tracks")
def cloud_demo_tracks() -> dict:
    return {
        "tracks": [
            {
                "id": "track1",
                "title": "公开论文解析与题目生成",
                "flow": ["公开 PDF", "MinerU 官方 API", "Markdown 摘要", "DeepSeek 题目/评分说明"],
            },
            {
                "id": "track2",
                "title": "公开任务规划与产物回放",
                "flow": ["公开任务目标", "意图识别", "云端 DAG", "回放产物索引", "核验摘要"],
            },
            {
                "id": "track3",
                "title": "医学 RAG 来源回放",
                "flow": ["中文问题", "公开来源匹配", "来源卡片", "稳定回答回放", "安全边界"],
            },
        ],
        "boundary": public_boundary(),
    }


# ---------------------------------------------------------------------------
# MinerU Official URL / Upload
# ---------------------------------------------------------------------------


@router.post("/mineru/url")
def cloud_demo_mineru_url(req: UrlParseRequest, x_demo_code: str | None = Header(default=None)) -> dict:
    require_access(req.access_code or x_demo_code)
    cfg = cloud_demo_settings()
    if not cfg["mineru_token"]:
        raise HTTPException(status_code=503, detail={"error": "MINERU_API_TOKEN 未配置"})
    public_url = validate_public_extract_url(req.url)
    quota_after = consume_quota("mineru_url_parse")
    try:
        with httpx.Client(timeout=25, trust_env=False) as client:
            response = client.post(
                f"{cfg['mineru_base_url']}/api/v4/extract/task",
                headers={"Authorization": f"Bearer {cfg['mineru_token']}", "Content-Type": "application/json"},
                json={"url": public_url, "model_version": req.model_version or "vlm"},
            )
            data = response.json() if response.text else {}
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail={"error": "MinerU 官方 API 请求超时"})
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"error": f"MinerU 官方 API 调用失败: {str(exc)[:180]}"})
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail={"error": data})
    return {"status": "submitted", "provider": "mineru_official", "response": data, "quota": quota_after}


@router.post("/mineru/upload")
async def cloud_demo_mineru_upload(
    file: UploadFile = File(...),
    access_code: str = "",
    x_demo_code: str | None = Header(default=None),
) -> dict:
    require_access(access_code or x_demo_code)
    cfg = cloud_demo_settings()
    if not cfg["mineru_token"]:
        raise HTTPException(status_code=503, detail={"error": "MINERU_API_TOKEN 未配置"})
    original_name = file.filename or "upload.pdf"
    if file.content_type not in {"application/pdf", "application/octet-stream"} and not original_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail={"error": "只接受 PDF 文件"})
    content = await file.read()
    max_bytes = cfg["upload_max_mb"] * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail={"error": f"文件超过 {cfg['upload_max_mb']}MB 限制"})

    quota_after = consume_quota("mineru_upload_parse")
    upload_dir = PROJECT_ROOT / "data" / "demo_cloud" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(original_name).name)
    pdf_path = upload_dir / f"{int(time.time())}_{safe_name}"
    pdf_path.write_bytes(content)

    result = parse_mineru_official_upload(
        pdf_path, token=cfg["mineru_token"], base_url=cfg["mineru_base_url"]
    )
    return {
        "status": result.status,
        "parse_ok": result.parse_ok,
        "detail": result.detail,
        "task_id": result.task_id,
        "markdown": result.markdown,
        "artifacts": result.artifacts,
        "quota": quota_after,
    }


# ---------------------------------------------------------------------------
# Quiz Generate / Grade
# ---------------------------------------------------------------------------


def _mock_quiz(source_text: str, topic: str, count: int) -> list[dict]:
    clean = re.sub(r"\s+", " ", source_text or "").strip()
    evidence = clean[:220] if clean else "公开/脱敏解析文本未提供，验收时请传入 MinerU 解析结果或论文摘要。"
    base = [
        ("来源定位题", f"根据材料，概括与「{topic}」最直接相关的研究结论，并指出来源来自哪里。"),
        ("机制理解题", "解释该研究结果可能支持或限制某个科学判断的原因。"),
        ("边界判断题", "指出该材料不能支持哪些过度推断，并说明原因。"),
        ("评分应用题", "给出一个候选回答，判断它是否忠实于材料并列出扣分点。"),
        ("综合题", "用三句话总结研究问题、关键来源和安全边界。"),
    ]
    return [
        {
            "id": f"q{i + 1}",
            "type": base[i % len(base)][0],
            "question": base[i % len(base)][1],
            "expected_answer": f"应基于公开解析材料回答，至少引用一个来源片段：{evidence}",
            "rubric": "满分 10 分：事实准确 4 分，来源引用 3 分，边界表述 2 分，中文表达 1 分。",
        }
        for i in range(max(1, min(count, 8)))
    ]


@router.post("/quiz/generate")
def cloud_demo_quiz_generate(req: QuizGenerateRequest, x_demo_code: str | None = Header(default=None)) -> dict:
    require_access(req.access_code or x_demo_code, allow_unconfigured=req.dry_run)
    source = (req.source_text or "").strip()
    if len(source) < 20:
        raise HTTPException(status_code=400, detail={"error": "source_text 至少需要 20 个字符，可传入 MinerU markdown_preview 或论文摘要"})
    count = max(1, min(int(req.count or 3), 8))
    cfg = cloud_demo_settings()
    if req.dry_run or not cfg["deepseek_key"]:
        return {
            "status": "dry_run" if req.dry_run else "mock",
            "provider": "deepseek",
            "questions": _mock_quiz(source, req.topic, count),
            "boundary": public_boundary(),
            "quota": quota_status(),
        }

    prompt = (
        "请只基于公开/脱敏解析文本生成中文评测题。返回严格 JSON，不要 Markdown 代码块。"
        "JSON schema: {\"questions\":[{\"id\":\"q1\",\"type\":\"...\",\"question\":\"...\","
        "\"expected_answer\":\"...\",\"rubric\":\"...\"}]}\n\n"
        f"主题：{req.topic}\n题目数量：{count}\n解析文本：{source[:8000]}"
    )
    answer, usage, quota_after = call_deepseek(
        [
            {"role": "system", "content": "你是 ControlMind 云端 Demo 的出题 API，只处理公开或脱敏材料。"},
            {"role": "user", "content": prompt},
        ],
        "quiz_generate",
        temperature=0.1,
    )
    try:
        parsed = json.loads(answer)
    except Exception:
        parsed = {"questions": _mock_quiz(source, req.topic, count), "raw_answer": answer}
    questions = parsed.get("questions") if isinstance(parsed, dict) else None
    if not isinstance(questions, list) or not questions:
        questions = _mock_quiz(source, req.topic, count)
    return {"status": "ok", "provider": "deepseek", "questions": questions[:count], "raw_usage": usage, "quota": quota_after}


@router.post("/quiz/grade")
def cloud_demo_quiz_grade(req: QuizGradeRequest, x_demo_code: str | None = Header(default=None)) -> dict:
    require_access(req.access_code or x_demo_code, allow_unconfigured=req.dry_run)
    question = (req.question or "").strip()
    student_answer = (req.student_answer or "").strip()
    if not question or not student_answer:
        raise HTTPException(status_code=400, detail={"error": "question 和 student_answer 不能为空"})
    cfg = cloud_demo_settings()
    if req.dry_run or not cfg["deepseek_key"]:
        score = 7 if len(student_answer) >= 30 else 5
        return {
            "status": "dry_run" if req.dry_run else "mock",
            "provider": "deepseek",
            "score": score,
            "max_score": 10,
            "feedback": "演示评分：回答已进入云端评分 API；正式配置 DeepSeek key 后会根据题目、评分点和公开来源给出细分扣分理由。",
            "rubric_hits": ["边界检查已生效", "密钥未进入前端", "仅公开/脱敏材料"],
            "quota": quota_status(),
        }

    prompt = (
        "请作为严格评分 API，只根据题目、标准答案/评分点、学生回答和公开来源评分。"
        "返回严格 JSON，不要 Markdown 代码块。"
        "JSON schema: {\"score\":0-10,\"max_score\":10,\"feedback\":\"...\",\"rubric_hits\":[\"...\"],\"deductions\":[\"...\"]}\n\n"
        f"题目：{question}\n标准答案：{req.expected_answer}\n评分点：{req.rubric}\n学生回答：{student_answer}\n公开来源：{req.source_text[:6000]}"
    )
    answer, usage, quota_after = call_deepseek(
        [
            {"role": "system", "content": "你是 ControlMind 云端 Demo 的评分 API，只处理公开或脱敏材料，不给个人诊疗建议。"},
            {"role": "user", "content": prompt},
        ],
        "quiz_grade",
        temperature=0,
    )
    try:
        parsed = json.loads(answer)
    except Exception:
        parsed = {"score": None, "max_score": 10, "feedback": answer, "rubric_hits": [], "deductions": []}
    return {"status": "ok", "provider": "deepseek", **parsed, "raw_usage": usage, "quota": quota_after}


# ---------------------------------------------------------------------------
# Agent Plan
# ---------------------------------------------------------------------------


@router.post("/agent/plan")
def cloud_demo_agent_plan(req: CloudDemoAgentPlanRequest, x_demo_code: str | None = Header(default=None)) -> dict:
    require_access(req.access_code or x_demo_code, allow_unconfigured=True)
    goal = (req.goal or "").strip()
    if len(goal) < 4:
        raise HTTPException(status_code=400, detail={"error": "goal 至少需要 4 个字符"})
    lowered = goal.lower()
    if any(token in lowered for token in ["rag", "医学", "问答", "medical"]):
        intent = "medical_rag_public_check"
        dag = ["边界判定", "公开来源检索", "中文合成", "claim 支撑状态", "安全声明"]
        artifacts = ["docs/submissions/track3_medical_rag_report.md", "docs/submissions/data_trace_bundle/07_medical_rag/"]
    elif any(token in lowered for token in ["评测", "出题", "评分", "quiz", "benchmark"]):
        intent = "sci_align_quiz_eval"
        dag = ["公开 PDF 解析", "Markdown 摘要", "ABCD 出题", "Rubric 评分", "来源路径记录"]
        artifacts = ["docs/submissions/track1_sci_align_report.md", "benchmark/dataset/core.json"]
    else:
        intent = "data_agent_public_plan"
        dag = ["目标解析", "输入边界检查", "资源选择", "执行步骤生成", "核验摘要"]
        artifacts = ["docs/submissions/track2_agent_report.md", "docs/submissions/data_trace_bundle/manifest.json"]
    return {
        "status": "ok",
        "provider": "deterministic_cloud_planner",
        "goal": goal,
        "intent": intent,
        "dag": [{"step": idx + 1, "name": name, "mode": "cloud_public_demo"} for idx, name in enumerate(dag)],
        "resource_policy": {
            "allowed": ["MinerU 官方 API", "DeepSeek API", "公开报告路径", "公开/脱敏 demo index"],
            "removed": ["重资产长任务", "私有原文", "私有向量库", "批量实验"],
        },
        "local_artifacts": artifacts,
        "acceptance": ["页面可回放已验证产物", "云端只处理公开轻量任务", "密钥不进入浏览器", "每个结论可回到来源或产物"],
        "boundary": public_boundary(),
    }


# ---------------------------------------------------------------------------
# Medical RAG Ask
# ---------------------------------------------------------------------------


@router.post("/medical-rag/ask")
def cloud_demo_medical_rag_ask(req: MedicalRagRequest, x_demo_code: str | None = Header(default=None)) -> dict:
    require_access(req.access_code or x_demo_code, allow_unconfigured=req.dry_run)
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail={"error": "question 不能为空"})
    if is_personal_medical_question(question):
        return {
            "status": "refused",
            "provider": "public_medical_mini_rag",
            "answer": "该问题看起来涉及个人诊疗、用药或急症判断。云端 Demo 只回答公开医学文献层面的来源问答，不替代医生诊断、处方或急诊处理。",
            "retrieved_sources": [],
            "claim_status": "refused_personal_medical_advice",
            "boundary": public_boundary(),
            "quota": quota_status(),
        }
    chunks = retrieve_public_medical_chunks(question)
    cfg = cloud_demo_settings()
    if req.dry_run or not cfg["deepseek_key"] or not chunks:
        return {
            "status": "dry_run" if req.dry_run else "ok",
            "provider": "public_medical_mini_rag",
            "answer": synthesize_medical_answer(question, chunks),
            "retrieved_sources": chunks,
            "claim_status": "supported" if chunks else "insufficient_sources",
            "boundary": public_boundary(),
            "quota": quota_status(),
        }
    context = "\n\n".join([f"[{chunk['id']}] {chunk['title']}\n{chunk['text']}\n中文摘要：{chunk['zh']}" for chunk in chunks])
    answer, usage, quota_after = call_deepseek(
        [
            {
                "role": "system",
                "content": (
                    "你是 ControlMind 云端公开医学 RAG 合成器。必须只基于给定来源回答，"
                    "用中文解释机制，列出来源 id，并明确不替代个人诊疗或用药建议。"
                ),
            },
            {"role": "user", "content": f"问题：{question}\n\n检索到的公开来源：\n{context}"},
        ],
        "medical_rag_ask",
        temperature=0.1,
    )
    return {
        "status": "ok",
        "provider": "deepseek_with_public_medical_mini_rag",
        "answer": answer,
        "retrieved_sources": chunks,
        "claim_status": "supported",
        "raw_usage": usage,
        "quota": quota_after,
    }


# ---------------------------------------------------------------------------
# DeepSeek Generic Ask
# ---------------------------------------------------------------------------


@router.post("/deepseek/ask")
def cloud_demo_deepseek_ask(req: DeepSeekAskRequest, x_demo_code: str | None = Header(default=None)) -> dict:
    require_access(req.access_code or x_demo_code, allow_unconfigured=req.dry_run)
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail={"error": "question 不能为空"})
    cfg = cloud_demo_settings()
    if req.dry_run or not cfg["deepseek_key"]:
        return {
            "status": "dry_run" if req.dry_run else "mock",
            "provider": "deepseek",
            "answer": f"演示边界已生效：问题「{question}」只能基于公开/脱敏来源回答，不处理个人诊疗或私有材料。",
            "boundary": public_boundary(),
            "quota": quota_status(),
        }
    system_prompt = "你是 ControlMind 公开云端 Demo 助手。只基于公开或脱敏材料回答，不能给个人诊断、处方或处理私有材料。"
    user_prompt = f"问题：{question}\n\n公开/脱敏上下文：{req.context[:4000] if req.context else '未提供额外上下文，请给出边界清晰的演示性回答。'}"
    answer, usage, quota_after = call_deepseek(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "deepseek_ask",
        temperature=0.2,
    )
    return {"status": "ok", "provider": "deepseek", "answer": answer, "raw_usage": usage, "quota": quota_after}


# ---------------------------------------------------------------------------
# Asset serving
# ---------------------------------------------------------------------------


@router.get("/assets/{name:path}")
def cloud_demo_asset(name: str) -> FileResponse:
    safe = Path(name)
    if safe.is_absolute() or ".." in safe.parts:
        raise HTTPException(status_code=400, detail={"error": "非法资源路径"})
    candidate_roots = [
        PROJECT_ROOT / "docs" / "submissions" / "data_trace_bundle",
        PROJECT_ROOT / "docs" / "submissions" / "shared" / "assets",
    ]
    for base in candidate_roots:
        path = (base / safe).resolve()
        if base.resolve() in path.parents and path.is_file():
            return FileResponse(path)
    raise HTTPException(status_code=404, detail={"error": "资源不存在"})
