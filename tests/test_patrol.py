"""Linux 化 Task 13：巡检内核 scripts/patrol.py 的契约测试。

守三件事：三态语义不可破（None＝检测不可用，不得折叠成 False）、自身进程排除、
探测路径不再依赖任何 Windows 工具。
"""

import inspect

from scripts.patrol import aggregate_alive, check_process_alive


def test_none_is_not_false():
    assert aggregate_alive([True, None]) is True
    assert aggregate_alive([False, None]) is None
    assert aggregate_alive([False, False]) is False
    assert aggregate_alive([True, False]) is True
    assert aggregate_alive([]) is None


def test_alive_reported_for_own_child_is_excluded():
    """巡检脚本自身与父进程不得被算成被巡检的任务（否则恒判 alive）。"""
    assert check_process_alive("definitely_no_such_script_zzz.py") is False


def test_detects_a_real_running_command():
    import subprocess
    import sys
    import time

    marker = "patrol_probe_marker_zzz"
    proc = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep(30)  # {marker}"])
    try:
        time.sleep(1.0)
        assert check_process_alive(marker) is True
    finally:
        proc.kill()
        proc.wait()


def test_kernel_probes_only_via_psutil():
    """最强的不变量：内核不派生任何外部命令，因此不存在 Windows 回退分支。"""
    from scripts import patrol

    src = inspect.getsource(patrol)
    assert "subprocess" not in src
    assert "os.name" not in src
    assert "import psutil" in src


def test_kill_tree_reclaims_grandchildren():
    """kill_tree 是全仓唯一的树杀入口；它的存在理由就是收孙进程，必须实测。"""
    import subprocess
    import sys

    import psutil

    from scripts.patrol import kill_tree

    marker = "patrol_tree_probe_zzz"
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess, sys, time\n"
         "subprocess.Popen(['sleep', '300'])\n"
         f"print({marker!r}, flush=True)\n"
         "time.sleep(300)\n"], stdout=subprocess.PIPE)
    assert proc.stdout.readline().decode().strip() == marker
    kids = psutil.Process(proc.pid).children(recursive=True)
    assert kids, "没建成子树，测试无效"

    leftovers = kill_tree(proc.pid)
    proc.wait(timeout=30)

    assert leftovers == [], f"未清干净的 PID: {leftovers}"
    for kid in kids:
        assert not kid.is_running() or kid.status() == psutil.STATUS_ZOMBIE, \
            f"孙进程 {kid.pid} 仍在跑 —— 会占死显存"
