"""回归测试：track3 supplemental 的 payload 构建器不得再触碰已清空的注册表。

第 1 轮审查的 C1 是 `_supplemental_summary_path()` 仍下标取被删的实验键，
两个子命令进入即 KeyError —— 而它当时没被测到，因为 CLI 冒烟测试只跑顶层 `--help`。
本文件直接调 payload 构建器，覆盖到真正的崩溃路径。
"""

from mineru_cli.commands.track3 import (
    _supplemental_status_payload,
    _supplemental_summary_payload,
)


def test_status_payload_builds_without_keyerror():
    payload = _supplemental_status_payload()
    assert payload["status"] in {"ok", "partial", "missing", "degraded"}
    assert payload["evidence_file_count"] > 0


def test_summary_payload_builds_without_keyerror():
    payload = _supplemental_summary_payload()
    assert isinstance(payload, dict)


def test_payload_has_no_registered_task_keys():
    """注册表已判死清空，输出里不应再有 registered_* 字段（防半改状态回归）。"""
    payload = _supplemental_status_payload()
    assert "registered_tasks" not in payload
    assert "registered_task_count" not in payload
