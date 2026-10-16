"""Tiny persistent quota helper for the public cloud demo.

Migrated from ``controlsci.api.cloud_quota``.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
import threading
import tempfile
import warnings
from contextlib import contextmanager

from fastapi import HTTPException

from mineru_core.paths import PROJECT_ROOT
from mineru_core.settings import get_settings

QUOTA_PATH = PROJECT_ROOT / "data" / "demo_cloud" / "quota.json"
QUOTA_LOCK_PATH = PROJECT_ROOT / "data" / "demo_cloud" / "quota.lock"
_LOCK = threading.Lock()


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _default_state() -> dict:
    settings = get_settings()
    return {
        "date": _today(),
        "limit": settings.cloud_daily_limit,
        "used": 0,
    }


def _requires_demo_code() -> bool:
    raw = os.environ.get("CONTROLMIND_REQUIRE_DEMO_CODE", os.environ.get("DEMO_REQUIRE_ACCESS_CODE", "true"))
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def _quota_lock():
    QUOTA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        lock_file = QUOTA_LOCK_PATH.open("a+")
        try:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def _read_state_unlocked() -> dict:
    settings = get_settings()
    if not QUOTA_PATH.exists():
        return _default_state()

    try:
        state = json.loads(QUOTA_PATH.read_text(encoding="utf-8"))
    except Exception:
        state = _default_state()

    if state.get("date") != _today():
        state = _default_state()

    state["limit"] = settings.cloud_daily_limit
    state["used"] = max(0, int(state.get("used", 0)))
    return state


def _write_state(state: dict) -> None:
    QUOTA_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="quota.", suffix=".tmp", dir=str(QUOTA_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, QUOTA_PATH)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def quota_status() -> dict:
    state = _read_state_unlocked()
    remaining = max(0, int(state["limit"]) - int(state["used"]))
    return {
        **state,
        "remaining": remaining,
        "enabled": True,
        "access_code_required": _requires_demo_code(),
    }


def consume_quota(action: str, amount: int = 1) -> dict:
    if amount <= 0:
        return quota_status()

    with _quota_lock():
        state = _read_state_unlocked()
        remaining = max(0, int(state["limit"]) - int(state["used"]))
        if remaining < amount:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "今日云端实时调用额度已用完，请查看回放结果或明日再试。",
                    "quota": {
                        **state,
                        "remaining": remaining,
                        "enabled": True,
                    },
                    "action": action,
                },
            )
        state["used"] = int(state["used"]) + amount
        _write_state(state)
        remaining = max(0, int(state["limit"]) - int(state["used"]))
        return {
            **state,
            "remaining": remaining,
            "enabled": True,
            "last_action": action,
        }


def require_demo_code(provided: str | None) -> None:
    """[DEPRECATED: no known callers] Validate demo access code against settings."""
    warnings.warn("require_demo_code is deprecated and has no known callers", DeprecationWarning, stacklevel=2)
    expected = get_settings().demo_access_code
    if _requires_demo_code() and not expected:
        raise HTTPException(
            status_code=503,
            detail={"error": "DEMO_ACCESS_CODE 未配置，实时云端能力已锁定。公开证据与回放页面仍可直接查看。"},
        )
    if not expected:
        return
    if (provided or "").strip() == expected:
        return
    raise HTTPException(
        status_code=403,
        detail={"error": "实时云端能力需要有效 Demo Code。公开证据与回放页面仍可直接查看。"},
    )
