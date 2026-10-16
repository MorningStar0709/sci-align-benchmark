"""mineru-core — zero-dependency shared primitives for all MinerU packages."""

import os

from mineru_core.paths import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Auto-load project .env into os.environ (runs once at first import)
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Load ``PROJECT_ROOT/.env`` into os.environ as fallback for unset vars.

    Format: one ``KEY=VALUE`` per line, ``#`` for comments, blank lines ignored.
    Called automatically when mineru_core is first imported — no user action needed.

    Existing system environment variables take precedence; ``.env`` only fills in
    keys that are not already set.  I/O errors are silently ignored (``.env`` is
    optional).
    """
    dotenv = PROJECT_ROOT / ".env"
    if not dotenv.is_file():
        return
    try:
        text = dotenv.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return  # unreadable .env is non-fatal
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)  # system env wins; .env is fallback

# Only auto-load when not explicitly disabled (set MINERU_LOAD_DOTENV=0 to opt out)
if os.environ.get("MINERU_LOAD_DOTENV", "1") not in ("0", "false", "no"):
    _load_dotenv()

# ---------------------------------------------------------------------------

from mineru_core.settings import ApiSettings, get_settings, ProviderConfig, PROVIDERS, get_provider
from mineru_core.runtime import (
    PUBLIC_CLOUD_DATA_CLASSES,
    RuntimeConfig,
    allows_cloud_document_upload,
    normalize_data_class,
    normalize_parser_backend,
    normalize_runtime,
)
from mineru_core.utils import (
    append_jsonl,
    iter_report_files,
    load_json,
    write_json_atomic,
)
from mineru_core.log_schema import (
    ExecutionLog,
    FinalStatus,
    LogStep,
    StepStatus,
    load_log,
)
from mineru_core.mineru_client import SciverseClient
from mineru_core.constants import (
    CORE_9,
    FAMILY_COLORS,
    MODEL_FAMILY,
    MODEL_SHORT,
    STANDARD_CATEGORIES,
    VERSION,
)

__all__ = [
    "SciverseClient",
    "PROJECT_ROOT",
    "ApiSettings",
    "get_settings",
    "ProviderConfig",
    "PROVIDERS",
    "get_provider",
    "PUBLIC_CLOUD_DATA_CLASSES",
    "RuntimeConfig",
    "allows_cloud_document_upload",
    "normalize_data_class",
    "normalize_parser_backend",
    "normalize_runtime",
    "append_jsonl",
    "iter_report_files",
    "load_json",
    "write_json_atomic",
    "ExecutionLog",
    "FinalStatus",
    "LogStep",
    "StepStatus",
    "load_log",
    "CORE_9",
    "FAMILY_COLORS",
    "MODEL_FAMILY",
    "MODEL_SHORT",
    "STANDARD_CATEGORIES",
    "VERSION",
]
