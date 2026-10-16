"""四路径资源调度器 — GPU/API/local/MinerU 资源感知 + 自动降级

架构：
  Layer 2: Resource Scheduler
  ├─ 四路径决策 (优先级: 可复现性 > 隐私 > 成本)
  │  ├─ api → DeepSeek v4-flash API
  │  ├─ mimo_vision → MiMo-V2.5 原生视觉 (raw httpx, 单模型零拼接)
  │  ├─ local → llama.cpp qwen3.5-9b 本地 (OpenAI-compatible /v1)
  │  └─ mineru → MinerU 本地 API
  ├─ HealthCheck — 每 provider 独立探活，零密钥泄露
  ├─ ClientFactory — 每 provider 独立客户端实例（规则 10.1.2）
  ├─ ProviderParameterMap — 从 agent_capabilities.json 加载（规则 10.1.5）
  └─ Scheduler — resolve → schedule → execute → fallback 闭环

环境变量：
  OPENAI_API_KEY / DEEPSEEK_API_KEY — DeepSeek API
  MIMO_API_KEY — MiMo API（视觉 + 文本共享）
  MINIMAX_API_KEY — MiniMax Anthropic API
  LLAMA_CHAT_PORT — llama.cpp chat 端口（默认 8080）
  MINERU_API_BASE — MinerU API 地址

安全红线：
  - HealthCheck 不暴露 GPU 显存原始数据
  - 所有 Key 从 os.environ 读取，零硬编码
  - 日志不包含 API Key / 本地路径 / 个人邮箱
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

import mineru_core  # noqa: F401 — trigger _load_dotenv()
from mineru_core.settings import PROVIDERS
from mineru_agent.resource_config import (
    AUTO_PROVIDER_MAPPING,
    DATA_CLASS_DEPENDENT_INTENTS,
    HEALTH_TTL_SECONDS,
    LOCAL_ONLY_INTENTS,
    LOCAL_PROVIDER_MAPPING,
    PROVIDER_FALLBACK,
    PROVIDER_TO_HEALTH_KEY,
    PUBLIC_OR_SANITIZED_INTENTS,
    ProviderStatus,
    ResolvedProvider,
    ResourceStatus,
)
from mineru_agent.resource_health import HealthCheck

# ROOT = Path(__file__).resolve().parents[2]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead
# NOTE: prefer mining mineru_core.paths.PROJECT_ROOT for new code.
# Kept as local ROOT for backward-compat with capabilities path.
ROOT = Path(__file__).resolve().parents[4]
CAPABILITIES_PATH = ROOT / "benchmark" / "agent" / "agent_capabilities.json"

logger = logging.getLogger("ResourceScheduler")


class ProviderParameterMap:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self):
        if self._loaded:
            return
        self._loaded = True
        self._raw = {}
        self._intent_map: Dict[str, dict] = {}
        self._provider_map: Dict[str, Dict[str, dict]] = {}
        self._fallback_order: Dict[str, list] = {}
        self._resource_types: Dict[str, dict] = {}
        self._load()

    def _load(self):
        if not CAPABILITIES_PATH.exists():
            logger.warning("agent_capabilities.json not found at %s, using built-in defaults", CAPABILITIES_PATH)
            self._load_defaults()
            return
        with open(CAPABILITIES_PATH, "r", encoding="utf-8") as f:
            self._raw = json.load(f)
        for intent in self._raw.get("intents", []):
            self._intent_map[intent["intent_id"]] = intent
        sched = self._raw.get("resource_scheduler_config", {})
        self._provider_map = sched.get("provider_parameter_map", {})
        self._fallback_order = sched.get("fallback_order", {})
        self._resource_types = sched.get("resource_types", {})

    def _load_defaults(self):
        self._provider_map = {
            "deepseek": {
                "chat": {
                    "model": "deepseek-v4-flash",
                    "base_url": PROVIDERS["deepseek"].base_url,
                    "default_temperature": 0.1,
                    "default_max_tokens": 4096,
                    "client_type": "openai",
                    "proxy_required": False,
                }
            },
            "mimo": {
                "vision": {
                    "model": "mimo-v2.5",
                    "base_url": PROVIDERS["mimo"].base_url,
                    "default_temperature": 1.0,
                    "default_max_tokens": 4096,
                    "client_type": "raw_httpx",
                    "auth_type": "api-key header",
                },
                "text": {
                    "model": "mimo-v2-flash",
                    "base_url": PROVIDERS["mimo"].base_url,
                    "default_temperature": 0.7,
                    "default_max_tokens": 4096,
                    "client_type": "openai",
                    "proxy_required": False,
                },
            },
            "minimax": {
                "chat": {
                    "model": "minimax-2.5",
                    "base_url": PROVIDERS["minimax"].base_url,
                    "client_type": "anthropic",
                    "default_temperature": 0.7,
                    "default_max_tokens": 4096,
                    "proxy_required": False,
                }
            },
            "local": {
                "chat": {
                    "model": "qwen3.5-9b",
                    "base_url": "http://localhost:8080",
                    "api_path": "/v1/chat/completions",
                    "default_temperature": 0.1,
                    "default_max_tokens": 4096,
                    "client_type": "openai",
                },
                "completions": {
                    "model": "qwen3.5-9b",
                    "base_url": "http://localhost:8080",
                    "api_path": "/v1/chat/completions",
                    "default_temperature": 0.1,
                    "default_max_tokens": 4096,
                    "client_type": "openai",
                },
            },
        }
        self._fallback_order = {
            "api": ["script"],
            "local_api": ["script"],
            "local_gpu": ["script"],
            "script": [],
        }
        self._resource_types = {
            "api": {"description": "外部 API 调用", "default_timeout": 120},
            "local_api": {"description": "本地 API 服务", "default_timeout": 300},
            "local_gpu": {"description": "本地 GPU 任务", "default_timeout": 3600},
            "script": {"description": "纯本地脚本", "default_timeout": 600},
        }

    def get_intent(self, intent_id: str) -> Optional[dict]:
        # Public API
        return self._intent_map.get(intent_id)

    def get_provider_config(self, provider: str, sub_type: str = "chat") -> Optional[dict]:
        # Public API
        return self._provider_map.get(provider, {}).get(sub_type)

    def get_fallback_types(self, resource_type: str) -> List[str]:
        # Public API
        return self._fallback_order.get(resource_type, [])

    def get_resource_timeout(self, resource_type: str) -> int:
        # Public API
        return self._resource_types.get(resource_type, {}).get("default_timeout", 120)

    def list_providers(self) -> List[str]:
        # Public API
        return list(self._provider_map.keys())

    def list_sub_types(self, provider: str) -> List[str]:
        # Public API
        return list(self._provider_map.get(provider, {}).keys())



class ClientFactory:
    """为每个 Provider 创建独立客户端实例（规则 10.1.2：四路径各自独立，不可共享）。"""

    @staticmethod
    def create_deepseek(timeout: int = 120) -> Tuple:
        # Public API (factory method)
        from openai import OpenAI, DefaultHttpxClient
        from mineru_eval.client_factory import wrap_client
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 或 DEEPSEEK_API_KEY 未设置")
        client = wrap_client(OpenAI(
            api_key=api_key,
            base_url=PROVIDERS["deepseek"].base_url,
            timeout=timeout,
            max_retries=0,
            http_client=DefaultHttpxClient(proxy=None, trust_env=False),
        ))
        return client, PROVIDERS["deepseek"].default_model

    @staticmethod
    def create_deepseek_custom(model: str | None = None, timeout: int = 120) -> tuple:
        # Public API (factory method)
        from openai import OpenAI, DefaultHttpxClient
        from mineru_eval.client_factory import wrap_client
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 或 DEEPSEEK_API_KEY 未设置")
        client = wrap_client(OpenAI(
            api_key=api_key,
            base_url=PROVIDERS["deepseek"].base_url,
            timeout=timeout,
            max_retries=0,
            http_client=DefaultHttpxClient(proxy=None, trust_env=False),
        ))
        return client, model or PROVIDERS["deepseek"].default_model

    @staticmethod
    def create_mimo_vision() -> Tuple:
        # Public API (factory method)
        api_key = os.environ.get("MIMO_API_KEY", "")
        if not api_key:
            raise ValueError("MIMO_API_KEY 未设置")
        return None, "mimo-v2.5"

    @staticmethod
    def create_mimo_text(model: str | None = None, timeout: int = 120) -> tuple:
        # Public API (factory method)
        from openai import OpenAI, DefaultHttpxClient
        from mineru_eval.client_factory import wrap_client
        api_key = os.environ.get("MIMO_API_KEY", "")
        if not api_key:
            raise ValueError("MIMO_API_KEY 未设置")
        client = wrap_client(OpenAI(
            api_key=api_key,
            base_url=PROVIDERS["mimo"].base_url,
            timeout=timeout,
            max_retries=0,
            http_client=DefaultHttpxClient(proxy=None, trust_env=False),
        ))
        return client, model or "mimo-v2-flash"

    @staticmethod
    def create_minimax(timeout: int = 120) -> Tuple:
        # Public API (factory method)
        from anthropic import Anthropic
        api_key = os.environ.get("MINIMAX_API_KEY", "")
        if not api_key:
            raise ValueError("MINIMAX_API_KEY 未设置")
        client = Anthropic(
            api_key=api_key,
            base_url=PROVIDERS["minimax"].base_url,
            timeout=timeout,
            http_client=httpx.Client(proxy=None, trust_env=False),
        )
        return client, "MiniMax-M2.5-highspeed"

    @staticmethod
    def create_local_native(timeout: int = 120) -> Tuple:
        """Create OpenAI client pointing to llama.cpp server."""
        from openai import OpenAI, DefaultHttpxClient
        port = os.environ.get("LLAMA_CHAT_PORT", "8080")
        client = OpenAI(
            api_key="no-key-required",
            base_url=f"http://localhost:{port}/v1",
            timeout=timeout,
            max_retries=0,
            http_client=DefaultHttpxClient(proxy=None, trust_env=False),
        )
        return client, "qwen3.5-9b"  # no wrap — llama.cpp doesn't understand provider-specific extra_body

    @staticmethod
    def create_local_openai(timeout: int = 120) -> Tuple:
        """Alias for create_local_native (both use OpenAI format for llama.cpp)."""
        return ClientFactory.create_local_native(timeout=timeout)

    @staticmethod
    def create_mineru(timeout: int = 300) -> httpx.Client:
        # Public API (factory method)
        return httpx.Client(timeout=timeout, proxy=None)



class ResourceScheduler:
    """四路径资源调度器 — 决策优先级：隐私边界 > 可复现性 > 成本。

    数据边界原则:
      - local_only: 原文、医疗、微调数据、本地解析中间产物不出本机。
      - public_or_sanitized: 公开论文、脱敏统计、评测题与排行榜可使用云端 API 增强。
      - local_mode=True: 全部可替代任务优先走 llama.cpp/MinerU/本地脚本。

    Usage:
        scheduler = ResourceScheduler()
        status = scheduler.health_check()
        print(status.summary)

        resolved = scheduler.resolve("corpus_quality_report", local_mode=False)
        # → ResolvedProvider(provider="deepseek", model="deepseek-v4-flash", ...)

        resolved_local = scheduler.resolve("corpus_quality_report", local_mode=True)
        # → ResolvedProvider(provider="local", model="qwen3.5-9b", ...)
    """

    def __init__(self, local_mode: bool = False):
        self.local_mode = local_mode
        self._ppm = ProviderParameterMap()
        self._resource_status: Optional[ResourceStatus] = None
        self._health_checked_at: float = 0.0

    def health_check(self, force: bool = False) -> ResourceStatus:
        # Public API
        now = time.time()
        if not force and self._resource_status is not None and (now - self._health_checked_at) < HEALTH_TTL_SECONDS:
            return self._resource_status
        self._resource_status = HealthCheck.check_all()
        self._health_checked_at = now
        return self._resource_status

    @staticmethod
    def _infer_sub_type(provider: str, intent_id: str | None = None) -> str:
        if provider == "mimo":
            return "vision" if intent_id == "cross_modal_audit" else "text"
        return "chat"

    @staticmethod
    def _data_policy(intent_id: str) -> dict:
        if intent_id in DATA_CLASS_DEPENDENT_INTENTS:
            return {
                "boundary": "data_class_dependent",
                "cloud_allowed": "conditional",
                "reason": "public_documents_may_use_cloud_only_with_explicit_allow_cloud_upload",
            }
        if intent_id in LOCAL_ONLY_INTENTS:
            return {
                "boundary": "local_only",
                "cloud_allowed": False,
                "reason": "raw_or_sensitive_artifacts_remain_local",
            }
        if intent_id in PUBLIC_OR_SANITIZED_INTENTS:
            return {
                "boundary": "public_or_sanitized",
                "cloud_allowed": True,
                "reason": "public_inputs_or_sanitized_derivatives_only",
            }
        return {
            "boundary": "review_required",
            "cloud_allowed": False,
            "reason": "unknown_intent_defaults_to_local_review",
        }

    def resolve(self, intent_id: str, local_mode: bool = None,
                force_provider: str = None) -> ResolvedProvider:
        # Public API
        if local_mode is None:
            local_mode = self.local_mode

        intent = self._ppm.get_intent(intent_id)
        if intent is None:
            raise KeyError(f"Unknown intent_id: {intent_id}")

        provider_defaults = intent.get("provider_defaults", {})

        if force_provider:
            sub_type = self._infer_sub_type(force_provider, intent_id)
            return self._build_resolved(force_provider, sub_type, intent, provider_defaults)

        data_policy = self._data_policy(intent_id)
        if local_mode or not data_policy["cloud_allowed"]:
            return self._resolve_local(intent_id, intent, provider_defaults)

        return self._resolve_auto(intent_id, intent, provider_defaults)

    def _resolve_auto(self, intent_id, intent, provider_defaults):
        provider, sub_type = AUTO_PROVIDER_MAPPING.get(intent_id, ("deepseek", "chat"))
        return self._build_resolved(provider, sub_type, intent, provider_defaults)

    def _resolve_local(self, intent_id, intent, provider_defaults):
        provider, sub_type = LOCAL_PROVIDER_MAPPING.get(intent_id, ("local", "chat"))
        return self._build_resolved(provider, sub_type, intent, provider_defaults)

    def _build_resolved(self, provider, sub_type, intent, provider_defaults):
        if provider == "script":
            return ResolvedProvider(
                provider="script", sub_type="script", model="local_script",
                base_url="", api_key_env="", client_type="subprocess",
                temperature=0, max_tokens=0, timeout=600,
                extra_config={"data_policy": self._data_policy(intent.get("intent_id", ""))},
            )

        if provider == "mineru":
            base = os.environ.get("MINERU_API_BASE", "http://127.0.0.1:8000")
            timeout = self._ppm.get_resource_timeout("local_api")
            return ResolvedProvider(
                provider="mineru", sub_type="chat", model="mineru",
                base_url=base, api_key_env="", client_type="raw_httpx",
                temperature=0, max_tokens=0, timeout=timeout,
                extra_config={"data_policy": self._data_policy(intent.get("intent_id", ""))},
            )

        config = self._ppm.get_provider_config(provider, sub_type)
        if config is None:
            config = self._ppm.get_provider_config(provider, "chat") or {}

        model = config.get("model", "")
        base_url = config.get("base_url", "")
        client_type = config.get("client_type", "openai")
        pd_temp = provider_defaults.get("temperature")
        temperature = pd_temp if pd_temp is not None else config.get("default_temperature", 0.1)
        pd_tok = provider_defaults.get("max_tokens")
        max_tokens = pd_tok if pd_tok is not None else config.get("default_max_tokens", 4096)
        timeout = self._ppm.get_resource_timeout(intent.get("resource_type", "api"))

        api_key_env_map = {
            "deepseek": PROVIDERS["deepseek"].api_key_env,
            "mimo": PROVIDERS["mimo"].api_key_env,
            "minimax": PROVIDERS["minimax"].api_key_env,
            "local": "LLAMA_CHAT_PORT",  # llama.cpp 不需要 API key，LLAMA_CHAT_PORT 是服务端口
        }
        api_key_env = api_key_env_map.get(provider, "")

        extra_config = {}
        if client_type == "raw_httpx" and "auth_type" in config:
            extra_config["auth_type"] = config["auth_type"]
        if "api_path" in config:
            extra_config["api_path"] = config["api_path"]
        extra_config["data_policy"] = self._data_policy(intent.get("intent_id", ""))

        return ResolvedProvider(
            provider=provider,
            sub_type=sub_type,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            client_type=client_type,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_config=extra_config,
        )

    def get_fallback(self, resolved: ResolvedProvider) -> Optional[ResolvedProvider]:
        # Public API
        fallback_list = PROVIDER_FALLBACK.get(resolved.provider, ["script"])
        if not fallback_list:
            return None
        status = self.health_check()
        for fb_provider in fallback_list:
            if fb_provider == "script":
                return ResolvedProvider(
                    provider="script", sub_type="script", model="local_script",
                    base_url="", api_key_env="", client_type="subprocess",
                    temperature=0, max_tokens=0, timeout=600,
                )
            health_key = PROVIDER_TO_HEALTH_KEY.get(fb_provider)
            if health_key and status.is_available(health_key):
                return self._build_resolved(fb_provider, "chat", {}, {})
        return None

    def get_timeout(self, intent_id: str) -> int:
        # Public API
        intent = self._ppm.get_intent(intent_id)
        if intent is None:
            return 120
        return self._ppm.get_resource_timeout(intent.get("resource_type", "api"))

    def list_available_providers(self) -> List[str]:
        # Public API
        status = self.health_check()
        return [p for p, h in status.providers.items()
                if h.status in (ProviderStatus.AVAILABLE, ProviderStatus.DEGRADED)]

    def is_ready(self, intent_id: str, local_mode: bool = None) -> Tuple[bool, str]:
        # Public API
        """检查指定 intent 的 Provider 是否就绪。

        Returns:
            (ready, message): 是否就绪及原因描述
        """
        try:
            resolved = self.resolve(intent_id, local_mode=local_mode)
        except Exception as e:
            return False, f"解析失败: {e}"

        if resolved.provider == "script":
            return True, "本地脚本无需外部依赖"

        status = self.health_check()
        health_key = PROVIDER_TO_HEALTH_KEY.get(resolved.provider)
        if health_key is None:
            h = status.providers.get(resolved.provider)
        else:
            h = status.providers.get(health_key)
        display_key = health_key or resolved.provider
        if h is None:
            return False, f"Provider {display_key} 不在健康检查列表中"
        if h.status == ProviderStatus.AVAILABLE:
            return True, f"{display_key} 就绪: {h.detail}"
        if h.status == ProviderStatus.DEGRADED:
            return True, f"{display_key} 降级可用: {h.detail}"
        return False, f"{display_key} 不可用: {h.detail}"


_global_scheduler: Optional[ResourceScheduler] = None
_global_scheduler_local_mode: Optional[bool] = None


def get_global_scheduler(local_mode: bool = False) -> ResourceScheduler:
    # Public API
    global _global_scheduler, _global_scheduler_local_mode
    if _global_scheduler is not None and _global_scheduler_local_mode == local_mode:
        return _global_scheduler
    _global_scheduler = ResourceScheduler(local_mode=local_mode)
    _global_scheduler_local_mode = local_mode
    return _global_scheduler
