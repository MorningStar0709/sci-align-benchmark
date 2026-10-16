"""Agent utility functions — subprocess runner, JSON extraction, path helpers.

Migrated from ``benchmark/agent/agent_cli.py``.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[4]
"""Project root (4 levels up from ``mineru_agent/utils.py``)."""

VERSION = "1.1.0"

logger = logging.getLogger(__name__)

_capabilities_path = None
_logs_dir = None


def _configure_stdio():
    """Keep direct Windows runs from failing on UTF-8 console output."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _get_capabilities_path():
    global _capabilities_path
    if _capabilities_path is None:
        _capabilities_path = ROOT / "benchmark" / "agent" / "agent_capabilities.json"
    return _capabilities_path


def _get_logs_dir():
    global _logs_dir
    if _logs_dir is None:
        _logs_dir = ROOT / "benchmark" / "agent" / "logs"
    return _logs_dir


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ============================================================
# JSON Extraction (规则 10.3.4)
# ============================================================


def _extract_json_block(text: str) -> str:
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start) if "```" in text[start:] else len(text)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start) if "```" in text[start:] else len(text)
        return text[start:end].strip()
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        return text[brace_start:brace_end + 1]
    return text.strip()


# ============================================================
# Subprocess Runner
# ============================================================

def _python_cmd(args: List[str]) -> List[str]:
    """构造子进程 Python 命令：一律用当前解释器。

    原实现是 conda run 包装 + 可用性探测缓存；本仓环境由 uv 管理 `.venv`、
    不存在 conda（`command -v conda` 为空），包装恒回落到 sys.executable，
    却每次都要付一次探测与一条告警，故直接删除该层。
    """
    return [sys.executable] + args


_SECRET_FLAGS = {"--api-key", "--target-api-key", "--judge-api-key", "--hf-token"}


def _safe_cmd(cmd_args: List[str]) -> str:
    masked = []
    redact_next = False
    for arg in cmd_args:
        text = str(arg)
        if redact_next:
            masked.append("***")
            redact_next = False
            continue
        if text in _SECRET_FLAGS:
            masked.append(text)
            redact_next = True
            continue
        if any(text.startswith(flag + "=") for flag in _SECRET_FLAGS):
            flag = text.split("=", 1)[0]
            masked.append(f"{flag}=***")
            continue
        masked.append(text)
    return " ".join(masked)


def _run_subprocess(cmd_args: List[str], cwd: str = None, timeout: int = 7200,
                    env_extra: dict = None) -> Tuple[int, str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)

    try:
        result = subprocess.run(
            cmd_args,
            cwd=cwd or str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"进程超时（{timeout}s）"
    except FileNotFoundError as e:
        return -1, "", f"命令或文件未找到: {e}"
    except Exception as e:
        return -1, "", str(e)[:500]
