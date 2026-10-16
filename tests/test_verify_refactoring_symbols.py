"""Linux 化 Task 6：verify_refactoring 的跨包符号可达性守卫。

原脚本只做包级 import 与一张手工维护的 CHECK_IMPORTS 表，从不校验源码里
`from mineru_x import name` 的 name 是否存在 —— medical_rag_eval 那条断链
（`from mineru_api.medical_rag import ...`）正是靠这个盲区活到今天的。
"""

from pathlib import Path

from scripts.verify_refactoring import check_import_symbols


def test_detects_unreachable_symbol(tmp_path: Path):
    (tmp_path / "broken.py").write_text(
        "from mineru_core import nonexistent_symbol_xyz\n", encoding="utf-8"
    )
    problems = check_import_symbols(tmp_path)
    assert len(problems) == 1
    assert "nonexistent_symbol_xyz" in problems[0]
    assert "mineru_core" in problems[0]


def test_detects_missing_module(tmp_path: Path):
    (tmp_path / "broken2.py").write_text(
        "from mineru_api.no_such_module_abc import thing\n", encoding="utf-8"
    )
    problems = check_import_symbols(tmp_path)
    assert len(problems) == 1
    assert "no_such_module_abc" in problems[0]


def test_type_checking_imports_are_ignored(tmp_path: Path):
    (tmp_path / "typing_only.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from mineru_core import only_in_annotations\n",
        encoding="utf-8",
    )
    assert check_import_symbols(tmp_path) == []


def test_live_repo_has_no_broken_cross_imports():
    """本计划的真链判据：源码面零断链。"""
    repo_root = Path(__file__).resolve().parents[1]
    problems = []
    for src in sorted((repo_root / "packages").glob("*/src")):
        problems += check_import_symbols(src)
    assert problems == [], "\n".join(problems)
