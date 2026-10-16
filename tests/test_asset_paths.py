"""Linux 化 Task 18：外部资产落点解析口径唯一化。

此前 `MINERU_CORPUS_ROOT` 的默认值在 run_openbook_control.py 与 run_judge_audit.py
各写了一份字面量副本；解析器收口到 mineru_core.paths，并按「落点分置」原则
允许资产留在仓外（不强制把共享语料搬进仓库）。
"""

from pathlib import Path

from mineru_core.paths import resolve_corpus_root


def test_env_override_wins(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MINERU_CORPUS_ROOT", str(tmp_path / "corpus"))
    assert resolve_corpus_root() == tmp_path / "corpus"


def test_asset_roots_selects_existing_candidate_only(monkeypatch, tmp_path: Path):
    """多根候选只取真实存在者；都不存在时返回首选而不抛错（CI 无语料也要能 import）。"""
    import os

    monkeypatch.delenv("MINERU_CORPUS_ROOT", raising=False)
    missing, present = tmp_path / "nope", tmp_path / "here"
    (present / "MinerU" / "corpus" / "processed").mkdir(parents=True)
    monkeypatch.setenv("SCIMONITOR_ASSET_ROOTS", f"{missing}{os.pathsep}{present}")
    assert resolve_corpus_root() == present / "MinerU" / "corpus" / "processed"
    monkeypatch.setenv("SCIMONITOR_ASSET_ROOTS", str(missing))
    assert resolve_corpus_root() == missing / "MinerU" / "corpus" / "processed"


def test_no_machine_local_layout_in_resolver():
    """leaf 包不得内置个人机器目录布局：盘符或 ~/WorkPlace 这类 home 检出路径都不行。"""
    import inspect

    from mineru_core import paths

    src = inspect.getsource(paths)
    assert "Path.home()" not in src
    assert "WorkPlace" not in src


def test_both_scripts_share_one_resolution():
    """两支脚本不再各自复制默认值 —— 用源码面复算，不只看返回值。"""
    import subprocess

    out = subprocess.run(
        ["grep", "-rn", "_DEFAULT_CORPUS", "scripts/"],
        capture_output=True,
        text=True,
    ).stdout
    assert out.strip() == "", out
