"""盲评隔离审计的消费者测试。

为什么要有：`humaneval/pipeline/` 三件是 10-03 才从"未跟踪"补进版本控制的。零消费者的
在册件是本仓最会漂移的形状（`claim_matrix.json` 那份假账就是这么活的）——所以入库的同一笔
必须给它接一个执行面。`check_blinding.py --self-test` 自带两道注入自测（往渲染视图塞
agent_score、篡改 app.py 的展示白名单），正好当这根桩。

刻意不用相对默认路径：脚本默认 `pool/pool.json` 是 cwd 敏感的，测试显式传绝对路径。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "humaneval" / "pipeline" / "check_blinding.py"
POOL = ROOT / "humaneval" / "pool" / "pool.json"
APP = ROOT / "humaneval" / "server" / "app.py"


def test_blinding_pipeline_inputs_are_present():
    """先证输入在盘——读不到的话"绿"可能只是根本没跑到检查那一步。"""
    for path in (SCRIPT, POOL, APP):
        assert path.is_file(), f"缺输入件，消费者等于空转：{path}"


def test_self_test_passes_and_reports_both_gates():
    """正例：两道门都过，且输出里必须点名是自测而非静默通过。"""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--pool", str(POOL), "--app", str(APP), "--self-test"],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"隔离审计自测失败：\n{proc.stdout}\n{proc.stderr}"
    out = proc.stdout
    assert "自测试 1 通过" in out and "自测试 2 通过" in out, \
        f"自测两道门没都跑到，输出如下：\n{out}"


def test_plain_audit_without_self_test_also_passes():
    """常规面（不注入篡改）也必须过，否则红线是虚设的。"""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--pool", str(POOL), "--app", str(APP)],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"盲评隔离被破：\n{proc.stdout}\n{proc.stderr}"
