"""Runtime configuration contract — shared by workbench endpoints and CLI.

Migrated from ``controlsci.api.runtime``.
"""

from __future__ import annotations

from pydantic import BaseModel


class RuntimeConfig(BaseModel):
    profile: str = "demo_replay"
    api_provider: str | None = None
    local_model: str = "qwen3.5:9b"
    parser_backend: str = "replay"
    privacy_policy: str = "local_only"
    data_class: str = "private_enterprise"
    allow_cloud_upload: bool = False
    t1_answer_model: str = "replay"
    t1_judge_model: str = "replay"
    t2_planner: str = "replay"
    t3_synthesis: str = "qwen3.5:9b"
    retrieval_index: str = "bge_m3"


PARSER_BACKEND_ALIASES = {
    "mineru_docker": "local",
    "mineru_api": "official",
}

DATA_CLASS_ALIASES = {
    "public": "public_open",
    "medical": "private_medical",
    "private": "private_enterprise",
}

PUBLIC_CLOUD_DATA_CLASSES = {"public_open", "public_but_regulated"}


def normalize_parser_backend(value: str | None) -> str:
    backend = (value or "replay").strip()
    return PARSER_BACKEND_ALIASES.get(backend, backend)


def normalize_data_class(value: str | None) -> str:
    data_class = (value or "private_enterprise").strip()
    return DATA_CLASS_ALIASES.get(data_class, data_class)


def allows_cloud_document_upload(runtime: RuntimeConfig) -> bool:
    return (
        runtime.allow_cloud_upload
        and runtime.data_class in PUBLIC_CLOUD_DATA_CLASSES
        and runtime.privacy_policy != "local_only"
        and runtime.profile != "local_private"
    )


def normalize_runtime(config: RuntimeConfig | None) -> RuntimeConfig:
    runtime = (config or RuntimeConfig()).model_copy(deep=True)
    runtime.parser_backend = normalize_parser_backend(runtime.parser_backend)
    runtime.data_class = normalize_data_class(runtime.data_class)
    if runtime.profile == "demo_replay":
        runtime.api_provider = None
        runtime.parser_backend = "replay"
        runtime.privacy_policy = "local_only"
        runtime.allow_cloud_upload = False
        runtime.t1_answer_model = "replay"
        runtime.t1_judge_model = "replay"
        runtime.t2_planner = "replay"
        runtime.t3_synthesis = runtime.local_model or "qwen3.5:9b"
        runtime.retrieval_index = runtime.retrieval_index or "bge_m3"
    if runtime.profile == "local_private":
        runtime.privacy_policy = "local_only"
        runtime.api_provider = None
        runtime.parser_backend = "local" if runtime.parser_backend == "official" else runtime.parser_backend
        runtime.allow_cloud_upload = False
        runtime.t1_answer_model = "replay"
        runtime.t1_judge_model = "replay"
        runtime.t2_planner = "local"
        if runtime.t3_synthesis == "replay":
            runtime.t3_synthesis = runtime.local_model or "qwen3.5:9b"
        runtime.retrieval_index = runtime.retrieval_index or "bge_m3"
    return runtime
