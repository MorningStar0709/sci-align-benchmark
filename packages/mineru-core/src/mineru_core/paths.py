"""Repository path helpers — shared by all packages.

PROJECT_ROOT points to the repository root (parent of packages/, data/, etc.).
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_project_root(marker: str = "pyproject.toml", marker_content: str | None = "sci-align-benchmark-workspace") -> Path:
    """Walk up from this file's directory to find the project root.

    Detection strategy (tried in order):
      1. Look for a pyproject.toml containing a [tool.uv] workspace section.
      2. Look for a pyproject.toml containing the string "sci-align-benchmark-workspace".
      3. Fall back to a known parent count heuristic.

    Returns:
        Absolute Path to the repository root directory.
    """
    here = Path(__file__).resolve().parent  # src/mineru_core/

    # Walk up at most 10 levels
    for candidate in [here] + list(here.parents)[:10]:
        candidate_pyproject = candidate / marker
        if candidate_pyproject.exists():
            try:
                text = candidate_pyproject.read_text(encoding="utf-8")
                if "[tool.uv]" in text and "workspace" in text:
                    return candidate
                if marker_content and marker_content in text:
                    return candidate
            except (OSError, UnicodeDecodeError):
                continue

    # Last resort: heuristic parent count for standard package layout
    # packages/mineru-core/src/mineru_core/paths.py → parents[4] = repo root
    return Path(__file__).resolve().parents[4]


PROJECT_ROOT: Path = _find_project_root()
"""Absolute path to the repository root."""


def resolve_corpus_root() -> Path:
    """外部论文语料库根路径（MinerU 解析产物），按「落点分置」原则解析候选根。

    候选顺序：`MINERU_CORPUS_ROOT` > `SCIMONITOR_ASSET_ROOTS`（os.pathsep 分隔的多根）
    下的 `MinerU/corpus/processed` > 仓内 `corpus/processed`。取第一个真实存在的候选；
    都不存在时返回首选候选而不抛错 —— 语料是跨项目共享资产，允许缺席（如 CI 干净环境），
    由真正读取时报错，不能在 import 期把 `--help` 打炸。

    刻意不再内置任何个人机器目录布局：资产在哪由环境说，不由 leaf 包猜。
    """
    direct = os.environ.get("MINERU_CORPUS_ROOT")
    if direct:
        return Path(direct)

    relative = Path("MinerU") / "corpus" / "processed"
    candidates = [
        Path(root) / relative
        for root in (os.environ.get("SCIMONITOR_ASSET_ROOTS") or "").split(os.pathsep)
        if root
    ]
    candidates.append(PROJECT_ROOT / "corpus" / "processed")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
