"""巡检内核：进程活性、三态聚合、产物时间戳三件泛用能力。

本仓 20 支 `check_*_health.py` 是「一个 campaign 一砖」的产物 —— 各自冻结了该战役的
期望常量（`EVAL_RUNS`、`TOTAL_UNITS` 等），战役收官即失效；可复用的只有下面这三个函数，
它们曾被复制 36 份。各脚本自己的判据常量与 `diagnose()` 留在原地。

刻意不派生任何外部命令：原副本里的 PowerShell CIM / tasklist 回退在 Linux 上只会
返回「检测不可用」，而三态聚合把「不可用」和「确认无进程」区分得很清楚。
"""

from __future__ import annotations

import os
import signal
from pathlib import Path


def check_process_alive(pattern: str, extra_ok=None) -> bool | None:
    """按命令行片段判进程活性。

    返回 True=存活 / False=确认无进程 / None=检测不可用。
    调用方不得把 None 当作「进程已死」—— 那会把探测故障误报成任务 crashed。

    `extra_ok(cmdline) -> bool` 用于区分共享同一脚本的不同任务
    （例如 4B/9B 两段复用同一个 `run_v3_rlvr_c.py`）。
    """
    try:
        import psutil
    except ImportError:
        return None

    try:
        me = os.getpid()
        try:
            parent = psutil.Process(me).ppid()
        except psutil.Error:
            parent = None
        for proc in psutil.process_iter(["cmdline"]):
            if proc.pid in (me, parent):
                continue
            cmdline = proc.info.get("cmdline") or []
            # uv launcher 会派生双进程，两个都命中属正常；路径分隔符统一后再比对
            if any(chunk.replace("\\", "/").endswith(pattern) for chunk in cmdline):
                if extra_ok is None or extra_ok(cmdline):
                    return True
        return False
    except (psutil.Error, OSError):
        return None


def aggregate_alive(results: list[bool | None]) -> bool | None:
    """三态聚合：任一 True → True；全 False → False；含 None → None。

    防 `any()` 把「检测失败」折叠成「确认无进程」，导致 stalled 误判。
    """
    if any(item is True for item in results):
        return True
    if results and all(item is False for item in results):
        return False
    return None


def latest_mtime(dir_path: Path) -> float | None:
    """目录内最新文件 mtime（递归），目录不存在或无文件返回 None。"""
    if not dir_path.is_dir():
        return None
    newest = None
    for path in dir_path.rglob("*"):
        try:
            if path.is_file():
                mtime = path.stat().st_mtime
                if newest is None or mtime > newest:
                    newest = mtime
        except OSError:
            continue
    return newest


def kill_tree(pid: int, term_timeout: float = 10.0) -> list[int]:
    """杀整棵进程树（自底向上），返回未能回收的 PID；空列表代表干净。

    本仓 8 支编排器原先各自 `proc.kill()` —— SIGKILL 只杀直接子进程，
    accelerate/torchrun worker 会留成孤儿继续占显存（单卡单进程下即死锁）。
    `_pipeline_watchdog._kill_tree` 是同一配方，那里额外需要 Popen 收尸语义。
    """
    try:
        import psutil
    except ImportError:
        return [pid]

    try:
        parent = psutil.Process(pid)
        targets = parent.children(recursive=True)
    except psutil.Error:
        return []

    targets.reverse()                      # 孙先于子、子先于父
    targets.append(parent)
    for proc in targets:
        try:
            proc.send_signal(signal.SIGTERM)
        except psutil.Error:
            continue

    _, alive = psutil.wait_procs(targets, timeout=term_timeout)
    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            continue
    _, still = psutil.wait_procs(alive, timeout=5)
    return [proc.pid for proc in still]
