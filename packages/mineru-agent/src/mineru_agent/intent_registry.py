"""Intent data classes and registry — PlanStep, VerifyResult, IntentRegistry.

Migrated from ``benchmark/agent/agent_cli.py``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mineru_agent.utils import _get_capabilities_path

logger = logging.getLogger(__name__)


# ============================================================
# Data Classes
# ============================================================


@dataclass
class PlanStep:
    intent_id: str
    intent_name: str
    description: str
    tool: str
    resource_type: str
    depends_on: List[str] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    rank: int = 0


@dataclass
class VerifyResult:
    passed: bool
    score: float
    reason: str
    issues: List[str] = field(default_factory=list)
    correction_hint: str = ""


# ============================================================
# Intent Registry
# ============================================================


class IntentRegistry:
    """Loads and serves intent definitions from ``agent_capabilities.json``."""

    def __init__(self):
        self._intents: Dict[str, dict] = {}
        self._load()

    def _load(self):
        if not _get_capabilities_path().exists():
            logger.warning("agent_capabilities.json 未找到，使用内置 intent 列表")
            self._load_builtin()
            return
        with open(_get_capabilities_path(), "r", encoding="utf-8") as f:
            raw = json.load(f)
        for intent in raw.get("intents", []):
            self._intents[intent["intent_id"]] = intent

    def _load_builtin(self):
        self._intents = {}
        builtin = [
            ("corpus_parse", 1, "语料解析", "MinerU API + mineru-to-md Skill", "local_api"),
            ("cross_modal_audit", 2, "跨模态对齐审计", "visual_audit.py + MiMo-V2.5 原生视觉", "api"),
            ("corpus_quality_report", 3, "语料质量报告", "DeepSeek v4-flash + 统计脚本", "api"),
            ("benchmark_build", 4, "评测数据集构建", "build_benchmark.py + 三 Provider", "api"),
            ("quality_arbitrate", 5, "质量仲裁", "arbiter.py + DeepSeek + Ollama", "api"),
            ("model_evaluate", 6, "模型评测", "evaluate.py + 自动选 Judge", "api"),
            ("multi_judge_compare", 7, "多 Judge 对比", "evaluate.py (API + Ollama 双路径)", "api"),
            ("leaderboard_viz", 8, "排行榜可视化", "summarize_multi.py + report_viz.py", "script"),
            ("local_finetune", 9, "本地微调", "train_qlora.py + Ollama", "local_gpu"),
            ("reproduce_all", 10, "全量复现", "已停用：Windows 编排时代", "script"),
            ("medical_rag", 11, "医疗RAG全流程", "medical_rag_handler.py", "script"),
            ("sciverse_search", 12, "Sciverse文献检索", "sciverse_search_handler.py", "api"),
        ]
        for pid, rank, name, tool, rt in builtin:
            self._intents[pid] = {
                "intent_id": pid, "rank": rank, "name": name,
                "description": name, "toolchain": [tool], "resource_type": rt,
                "depends_on": [],
            }

    def list_intents(self) -> List[dict]:
        return sorted(self._intents.values(), key=lambda x: x.get("rank", 99))

    def get(self, intent_id: str) -> Optional[dict]:
        return self._intents.get(intent_id)

    def get_batch(self, intent_ids: List[str]) -> Dict[str, Optional[dict]]:
        """Batch query multiple intents at once, reducing loop overhead."""
        return {pid: self._intents.get(pid) for pid in intent_ids}

    def list_ids(self) -> List[str]:
        return list(self._intents.keys())

    def to_compact_description(self) -> str:
        lines = []
        for i in self.list_intents():
            depends = i.get("depends_on", [])
            dep_str = f" (需先执行: {', '.join(depends)})" if depends else ""
            lines.append(
                f"  [{i['intent_id']}] {i.get('name', '')} — "
                f"{i.get('toolchain', [''])[0]}{dep_str}"
            )
        return "\n".join(lines)


class CachedIntentRegistry:
    """TTL-cached proxy for IntentRegistry to reduce repeated .get() calls.

    Usage:
        registry = CachedIntentRegistry(IntentRegistry(), ttl_seconds=300)
        info = registry.get("arxiv_search")  # cached after first call
    """

    def __init__(self, registry: IntentRegistry, ttl_seconds: int = 300):
        self._registry = registry
        self._cache: Dict[str, tuple[float, Optional[dict]]] = {}
        self._ttl = ttl_seconds

    def _is_fresh(self, timestamp: float) -> bool:
        return time.monotonic() - timestamp < self._ttl

    def get(self, intent_id: str) -> Optional[dict]:
        now = time.monotonic()
        cached = self._cache.get(intent_id)
        if cached and self._is_fresh(cached[0]):
            return cached[1]
        value = self._registry.get(intent_id)
        self._cache[intent_id] = (now, value)
        return value

    def get_batch(self, intent_ids: List[str]) -> Dict[str, Optional[dict]]:
        return {pid: self.get(pid) for pid in intent_ids}

    def list_ids(self) -> List[str]:
        return self._registry.list_ids()

    def invalidate(self, intent_id: str | None = None):
        if intent_id:
            self._cache.pop(intent_id, None)
        else:
            self._cache.clear()
