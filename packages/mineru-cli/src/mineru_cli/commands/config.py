"""Configuration commands — migrated from controlsci.cli.config."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from mineru_core.settings import get_settings, CONFIG_PATH
from mineru_cli.commands.common import console, fail, mask_secrets, print_json, render_kv_panel, write_json


app = typer.Typer(help="查看和保存 MinerU 运行配置。")

ENV_MAP = {
    "api.host": "CONTROLMIND_API_HOST",
    "api.port": "CONTROLMIND_API_PORT",
    "mineru.url": "CONTROLMIND_MINERU_URL",
    "llm_chat.url": "CONTROLMIND_LLM_CHAT_URL",
    "profile": "CONTROLMIND_PROFILE",
    "upload.max_mb": "CONTROLMIND_UPLOAD_MAX_MB",
    "daily_api_limit": "CONTROLMIND_DAILY_API_LIMIT",
    "python.path": "CONTROLMIND_PYTHON",
    "mineru.official_url": "MINERU_OFFICIAL_BASE_URL",
    "mineru.official_token": "MINERU_API_TOKEN",
    "demo.access_code": "CONTROLMIND_DEMO_ACCESS_CODE",
}


def _saved_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            fail(f"配置文件损坏：{CONFIG_PATH} ({exc})。请修复或删除该文件。")
    return {}


@app.command("show")
def show_config(
    json_output: Annotated[bool, typer.Option("--json", help="输出 JSON。")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="将 JSON 以 UTF-8 写入文件。")] = None,
    show_secrets: Annotated[bool, typer.Option("--show-secrets", help="显示 API token 等敏感字段。默认脱敏。")] = False,
) -> None:
    saved = _saved_config()
    try:
        get_settings.cache_clear()
        settings = get_settings()
    except RuntimeError as exc:
        fail(str(exc))
    data = {
        "effective": settings.model_dump(),
        "saved_config_path": str(CONFIG_PATH),
        "saved": saved,
        "env_keys": ENV_MAP,
    }
    safe_data = data if show_secrets else mask_secrets(data)
    if json_output or output:
        print_json(safe_data, output=output)
        return
    effective = safe_data["effective"]
    render_kv_panel(
        "MinerU 配置",
        {
            "profile": effective["cloud_demo_profile"],
            "api": settings.api_base_url,
            "mineru": effective["mineru_base_url"],
            "llm_chat": effective["llm_chat_url"],
            "llm_embed": effective["llm_embed_url"],
            "upload_max_mb": effective["upload_max_mb"],
            "mineru_official_token": effective["mineru_official_token"],
            "saved_config": CONFIG_PATH if CONFIG_PATH.exists() else "未写入",
        },
    )


@app.command("set")
def set_config(
    key: Annotated[str, typer.Argument(help="配置键，例如 mineru.url。")],
    value: Annotated[str, typer.Argument(help="配置值。")],
) -> None:
    if key not in ENV_MAP:
        fail(f"未知配置键：{key}。可用键：{', '.join(ENV_MAP)}")
    config = _saved_config()
    config[key] = value
    write_json(CONFIG_PATH, config)
    env_key = ENV_MAP[key]
    safe_value = mask_secrets({key: value})[key]
    console.print(f"[green]已保存[/green] {key} = {safe_value}")
    console.print(f"[dim]当前进程如需立即生效，也可以设置环境变量：{env_key}={safe_value}[/dim]")
    console.print("[yellow]提示：[/yellow]已运行的 FastAPI/前端进程不会自动读取新配置；请运行 `mineru demo restart` 或手动重启服务。")


@app.command("env")
def print_env() -> None:
    for key, env_key in ENV_MAP.items():
        value = os.environ.get(env_key, "")
        safe_value = mask_secrets({key: value})[key] if value else "[dim]未设置[/dim]"
        console.print(f"{key:16} {env_key:28} {safe_value}")
