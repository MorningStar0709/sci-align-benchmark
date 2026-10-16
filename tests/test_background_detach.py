"""Linux 化 Task 16：后台脱离语义必须有 Linux 实现。

原代码只有 `if os.name == "nt": creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`，
没有 else 分支 —— 意味着 Linux 下「后台模式」拿不到任何脱离手段，父终端一退出子进程就死，
正是 AGENTS.md 长任务规则明令禁止的裸 `&` 启动形态。
"""

import subprocess

import mineru_eval.cross_domain_test as cdt


def test_run_background_detaches_on_linux(monkeypatch, tmp_path):
    captured = {}

    class FakeProc:
        pid = 4242

    def spy(cmd, **kwargs):
        captured.update(kwargs)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", spy)
    monkeypatch.setattr(cdt, "write_status", lambda *a, **k: None)
    monkeypatch.setattr(cdt, "ROOT", tmp_path)  # 日志落点挪出工作区

    cdt._run_background(
        ["true"], "test-model", "test_model", str(tmp_path / "out.json"),
        {"family": "qwen", "size": "4b"},
    )
    assert captured.get("start_new_session") is True
    assert "creationflags" not in captured


def test_no_windows_only_branch_in_detachers():
    import inspect
    from pathlib import Path

    for mod in (cdt,):
        src = inspect.getsource(mod)
        assert "DETACHED_PROCESS" not in src
    launcher_src = Path("packages/mineru-pipeline/src/mineru_pipeline/launcher.py").read_text(encoding="utf-8")
    assert "DETACHED_PROCESS" not in launcher_src
    assert 'os.name == "nt"' not in launcher_src
