"""Pydantic models for demo and workbench endpoints.

Migrated from ``controlsci.api.demo_endpoints``.
"""

from __future__ import annotations

from pydantic import BaseModel

from mineru_core.runtime import RuntimeConfig


class AgentPlanRequest(BaseModel):
    query: str
    planner: str = ""
    local_model: str = ""
    api_provider: str | None = None
    privacy_policy: str = "local_only"
    runtime_config: RuntimeConfig | None = None


class Track2ActionRequest(AgentPlanRequest):
    live: bool = True
    artifact_kind: str = "acceptance"


class Track1ParseRequest(BaseModel):
    mode: str = "replay"
    paper_id: str = ""
    runtime_config: RuntimeConfig | None = None


class Track1ParseUrlRequest(BaseModel):
    url: str
    model_version: str = "vlm"
    demo_code: str = ""
    runtime_config: RuntimeConfig | None = None


class Track1ValidateChainRequest(BaseModel):
    paper_id: str = ""
    sample: int = 4
    source: str = "core_json"
    uploaded_filename: str = ""
    demo_code: str = ""
    model_choice: str = "replay"
    judge_model: str = "replay"
    runtime_config: RuntimeConfig | None = None
