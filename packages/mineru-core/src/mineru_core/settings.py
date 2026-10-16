"""Shared configuration for MinerU API services.

This module was migrated from ``controlsci.api.settings``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel


class ApiSettings(BaseModel):
    api_host: str = "127.0.0.1"
    api_port: int = 17001
    mineru_base_url: str = "http://localhost:8000"
    mineru_official_base_url: str = "https://mineru.net"
    mineru_official_token: str = ""
    llm_chat_url: str = "http://localhost:8080"
    llm_embed_url: str = "http://localhost:8081"
    upload_max_mb: int = 20
    cloud_demo_profile: str = "cloud_demo"
    cloud_daily_limit: int = 100
    demo_access_code: str = ""
    python_path: str = ""

    @property
    def api_base_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


CONFIG_PATH = Path.home() / ".controlmind" / "config.json"


def _read_cli_config() -> dict:
    try:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ControlMind 配置文件损坏：{CONFIG_PATH} ({exc})") from exc
    except OSError:
        return {}
    return {}


def _config_value(config: dict, key: str, default: str = "") -> str:
    value = config.get(key, default)
    return str(value).strip() if value is not None else default


def _env_or_config(env_names: tuple[str, ...], config: dict, config_key: str, default: str) -> str:
    for name in env_names:
        value = os.environ.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return _config_value(config, config_key, default)


def _int_env_or_config(env_names: tuple[str, ...], config: dict, config_key: str, default: int) -> int:
    for name in env_names:
        value = os.environ.get(name)
        if value is not None and str(value).strip() != "":
            try:
                return int(value)
            except ValueError:
                return default
    try:
        return int(config.get(config_key, default))
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    config = _read_cli_config()
    return ApiSettings(
        api_host=_env_or_config(("CONTROLMIND_API_HOST", "CONTROLSCI_API_HOST"), config, "api.host", "127.0.0.1"),
        api_port=_int_env_or_config(("CONTROLMIND_API_PORT", "CONTROLSCI_API_PORT"), config, "api.port", 17001),
        mineru_base_url=_env_or_config(("CONTROLMIND_MINERU_URL", "CONTROLSCI_MINERU_URL"), config, "mineru.url", "http://localhost:8000").rstrip("/"),
        mineru_official_base_url=_env_or_config(("MINERU_OFFICIAL_BASE_URL",), config, "mineru.official_url", "https://mineru.net").rstrip("/"),
        mineru_official_token=_env_or_config(("MINERU_API_TOKEN", "MINERU_OFFICIAL_TOKEN"), config, "mineru.official_token", ""),
        llm_chat_url=_env_or_config(("CONTROLMIND_LLM_CHAT_URL", "LLAMA_CHAT_URL"), config, "llm.chat_url", "http://localhost:8080").rstrip("/"),
        llm_embed_url=_env_or_config(("CONTROLMIND_LLM_EMBED_URL", "LLAMA_EMBED_URL"), config, "llm.embed_url", "http://localhost:8081").rstrip("/"),
        upload_max_mb=_int_env_or_config(("CONTROLMIND_UPLOAD_MAX_MB", "CONTROLSCI_UPLOAD_MAX_MB"), config, "upload.max_mb", 20),
        cloud_demo_profile=_env_or_config(("CONTROLMIND_PROFILE", "PROFILE"), config, "profile", "cloud_demo"),
        cloud_daily_limit=_int_env_or_config(("CONTROLMIND_DAILY_API_LIMIT", "DEMO_DAILY_API_LIMIT"), config, "daily_api_limit", 100),
        demo_access_code=_env_or_config(("CONTROLMIND_DEMO_ACCESS_CODE", "DEMO_ACCESS_CODE"), config, "demo.access_code", ""),
        python_path=_env_or_config(("CONTROLMIND_PYTHON", "CONTROLSCI_PYTHON"), config, "python.path", ""),
    )


# ---------------------------------------------------------------------------
# LLM Provider Registry — single source of truth for API endpoints
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderConfig:
    """Immutable provider configuration (base_url, key, model registry).

    The ``models`` dict maps short aliases to full model IDs, enabling
    CLI usage like ``--model pro`` instead of ``--model deepseek-v4-pro``.
    """
    name: str
    base_url: str
    api_key_env: str
    default_model: str
    models: dict[str, str] | None = None


PROVIDERS: dict[str, ProviderConfig] = {
    "deepseek": ProviderConfig(
        name="deepseek",
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key_env="DEEPSEEK_API_KEY|OPENAI_API_KEY",
        default_model="deepseek-v4-flash",
        models={"ds-flash": "deepseek-v4-flash", "ds-pro": "deepseek-v4-pro"},
    ),
    "mimo": ProviderConfig(
        name="mimo",
        base_url=os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1"),
        api_key_env="MIMO_API_KEY",
        default_model="mimo-v2.5",
        models={"mm-flash": "mimo-v2-flash", "mm-2.5": "mimo-v2.5", "mm-2.5-pro": "mimo-v2.5-pro"},
    ),
    "qwen": ProviderConfig(
        name="qwen",
        base_url=os.environ.get("QWEN_BASE_URL", "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"),
        api_key_env="QWEN_API_KEY|DASHSCOPE_API_KEY",
        default_model="qwen3.7-plus",
        models={"q-plus": "qwen3.7-plus"},
    ),
    # Backward compat — downstream eval/agent/medical still reference PROVIDERS["minimax"]
    "minimax": ProviderConfig(
        name="minimax",
        base_url=os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic"),
        api_key_env="MINIMAX_API_KEY",
        default_model="MiniMax-M2.5-highspeed",
        models={"m-2.5": "MiniMax-M2.5-highspeed", "m-2.7": "MiniMax-M2.7-highspeed"},
    ),
}


def get_provider(name: str) -> ProviderConfig:
    """Get provider config by name. Raises ValueError if not found."""
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[name]
