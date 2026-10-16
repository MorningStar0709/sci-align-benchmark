"""Resource configuration constants and data classes.

Extracted from resource_scheduler.py to separate configuration data
from scheduling logic.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class ResourceType(str, Enum):
    API = "api"
    LOCAL_API = "local_api"
    LOCAL_GPU = "local_gpu"
    SCRIPT = "script"


class ProviderStatus(str, Enum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class ProviderHealth:
    provider: str
    status: ProviderStatus
    latency_ms: int = 0
    detail: str = ""


@dataclass
class ResourceStatus:
    providers: Dict[str, ProviderHealth] = field(default_factory=dict)
    available_providers: List[str] = field(default_factory=list)
    degraded_providers: List[str] = field(default_factory=list)
    unavailable_providers: List[str] = field(default_factory=list)
    checked_at: str = ""

    @property
    def summary(self) -> str:
        lines = [f"ResourceStatus @ {self.checked_at}"]
        for name, h in self.providers.items():
            emoji = {"available": "✅", "degraded": "🟡",
                     "unavailable": "❌", "unknown": "❓"}.get(h.status, "❓")
            lines.append(f"  {emoji} {name:<18s} {h.status:<12s} {h.detail}")
        return "\n".join(lines)

    def is_available(self, provider: str) -> bool:
        h = self.providers.get(provider)
        return h is not None and h.status in (ProviderStatus.AVAILABLE, ProviderStatus.DEGRADED)


@dataclass
class ResolvedProvider:
    provider: str
    sub_type: str
    model: str
    base_url: str
    api_key_env: str
    client_type: str
    temperature: float
    max_tokens: int
    timeout: int = 120
    extra_config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Health check configuration
# ---------------------------------------------------------------------------

HEALTH_TTL_SECONDS = 30

PROVIDER_TO_HEALTH_KEY = {
    "deepseek": "deepseek",
    "mimo": "mimo_vision",
    "minimax": "minimax",
    "local": "local_chat",
    "mineru": "mineru",
}

# Provider → fallback provider 列表（与 ProviderParameterMap._fallback_order 互补：
# 前者按 provider 维度，后者按 resource_type 维度，key space 不同，各自服务不同调用方）
PROVIDER_FALLBACK = {
    "deepseek": ["script"],
    "mimo": ["script"],
    "minimax": ["script"],
    "local": ["script"],
    "mineru": ["script"],
}


# ---------------------------------------------------------------------------
# Data boundary / intent classification
# ---------------------------------------------------------------------------

DATA_CLASS_DEPENDENT_INTENTS = {
    "mineru_parse",
    "corpus_parse",
    "multi_format_parse",
}

LOCAL_ONLY_INTENTS = {
    "medical_rag",
    "local_finetune",
}

PUBLIC_OR_SANITIZED_INTENTS = {
    "arxiv_search",
    "cross_modal_audit",
    "corpus_quality_report",
    "benchmark_build",
    "quality_arbitrate",
    "model_evaluate",
    "multi_judge_compare",
    "leaderboard_viz",
    "reproduce_all",
    "sciverse_search",
}


# ---------------------------------------------------------------------------
# Intent → provider mapping (auto / local mode)
# ---------------------------------------------------------------------------

AUTO_PROVIDER_MAPPING = {
    "arxiv_search": ("script", "script"),
    "mineru_parse": ("mineru", "chat"),
    "corpus_parse": ("mineru", "chat"),
    "cross_modal_audit": ("mimo", "vision"),
    "corpus_quality_report": ("deepseek", "chat"),
    "benchmark_build": ("deepseek", "chat"),
    "quality_arbitrate": ("deepseek", "chat"),
    "model_evaluate": ("deepseek", "chat"),
    "multi_judge_compare": ("deepseek", "chat"),
    "leaderboard_viz": ("script", "chat"),
    "local_finetune": ("local", "chat"),
    "multi_format_parse": ("mineru", "chat"),
    "medical_rag": ("local", "chat"),
    "sciverse_search": ("api", "chat"),
    "reproduce_all": ("script", "chat"),
}

LOCAL_PROVIDER_MAPPING = {
    "arxiv_search": ("script", "script"),
    "mineru_parse": ("mineru", "chat"),
    "corpus_parse": ("mineru", "chat"),
    "cross_modal_audit": ("local", "chat"),
    "corpus_quality_report": ("local", "chat"),
    "benchmark_build": ("local", "chat"),
    "quality_arbitrate": ("local", "chat"),
    "model_evaluate": ("local", "chat"),
    "multi_judge_compare": ("local", "chat"),
    "leaderboard_viz": ("script", "chat"),
    "local_finetune": ("local", "chat"),
    "multi_format_parse": ("mineru", "chat"),
    "medical_rag": ("local", "chat"),
    "sciverse_search": ("api", "chat"),
    "reproduce_all": ("script", "chat"),
}
