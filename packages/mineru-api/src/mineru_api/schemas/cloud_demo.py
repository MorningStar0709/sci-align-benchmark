"""Pydantic models for cloud demo endpoints.

Migrated from ``cloud_demo_standalone.app``.
"""

from __future__ import annotations

from pydantic import BaseModel


class UrlParseRequest(BaseModel):
    url: str
    model_version: str = "vlm"
    access_code: str = ""


class QuizGenerateRequest(BaseModel):
    source_text: str
    topic: str = "主要终点、安全性来源与研究局限"
    count: int = 3
    access_code: str = ""
    dry_run: bool = False


class QuizGradeRequest(BaseModel):
    question: str
    expected_answer: str = ""
    rubric: str = ""
    student_answer: str
    source_text: str = ""
    access_code: str = ""
    dry_run: bool = False


class CloudDemoAgentPlanRequest(BaseModel):
    goal: str
    access_code: str = ""


class MedicalRagRequest(BaseModel):
    question: str
    access_code: str = ""
    dry_run: bool = False


class DeepSeekAskRequest(BaseModel):
    question: str
    context: str = ""
    access_code: str = ""
    dry_run: bool = False
