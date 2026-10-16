"""闸执行载体 scripts/run_provenance_gates.py 的契约测试。

为什么要它：溯源闸整组此前**全靠人手敲**——曾抓到一例「产物字段相对自己的 records 过期」
在本机静置数十天，没有任何自动检查会撞上它。这道脚本的价值不在"再算一遍"，而在**让闸被跑到**。

三条设计取舍都被测试钉住：
  1. **登记制而非盲目自动发现**：`scripts/` 下 `check|verify|audit_` 前缀的脚本里，若干支是
     活进程巡检、至少一支是会写盘的生成工具——按前缀全收会让提交前闸去查"有没有实验在跑"，
     既慢又会误红。故用显式清单；
  2. **未登记即红**（双向防腐烂）：新增一支匹配前缀的脚本而没进任一清单 → 报
     `unregistered` 并非零退出，逼一次决策；
  3. 判据不以"全绿"为参照，而是与上次红集合做差，只报"新增红"。
"""

import json
import re

from scripts.run_provenance_gates import (
    GATES,
    NOT_GATES,
    candidate_scripts,
    diff_against_previous,
    main,
    repo_root,
    run_gate,
    unregistered_candidates,
)

ROOT = repo_root()


def test_registry_covers_all_provenance_gates():
    for expected in ("check_lexec_dialect.py", "verify_refactoring.py"):
        assert expected in GATES, f"闸未登记：{expected}"
    # 闸数是**有意的摩擦**，不是当日快照：加/减闸必须同笔改这里的计数断言与
    # `.github/workflows/ci.yml`，否则一条只在本地跑过的「检查」会伪装成全仓纪律。
    assert len(GATES) == 2, (f"闸数变了但登记未被显式复核：现 {len(GATES)} 道，{GATES}")


def test_ci_workflow_lists_exactly_the_registered_gates():
    """`.github/workflows/ci.yml` 里出现的闸必须与 GATES 全集相等。

    以前这条只是注释里的"保持一致"，结果是本地 9 道、CI 列 6 道，注释在说谎。
    """
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    in_ci = set(re.findall(r"scripts/([a-z_0-9]+\.py)", text))
    assert in_ci == set(GATES), (f"CI 缺 {sorted(set(GATES) - in_ci)}；"
                                 f"CI 多 {sorted(in_ci - set(GATES))}")


def test_patrol_health_monitors_are_declared_not_gates():
    """`check_*_health.py` 查的是"实验在不在跑"，塞进提交前闸既慢又会误红。"""
    health = [n for n in candidate_scripts(ROOT) if n.name.endswith("_health.py")]
    for h in health:
        assert h.name in NOT_GATES, f"公开仓巡检类脚本未标注非闸理由：{h.name}"
    for h in health:
        assert h.name in NOT_GATES, f"巡检脚本未标注非闸理由：{h.name}"
        assert NOT_GATES[h.name], "理由不得为空串"
    assert not any(n.startswith("check_") and n.endswith("_health.py") for n in GATES)


def test_no_candidate_is_left_unregistered():
    """当前仓必须零未登记项——否则这条测试就是摆设。"""
    assert unregistered_candidates(ROOT) == []


def test_unregistered_candidate_is_reported(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "check_brand_new.py").write_text("import sys\nsys.exit(0)\n",
                                                             encoding="utf-8")
    found = unregistered_candidates(tmp_path)
    assert found == ["check_brand_new.py"]


def test_new_red_is_detected_and_known_red_is_not_relisted():
    previous = {"check_a.py": 0, "check_b.py": 1}
    current = {"check_a.py": 1, "check_b.py.py": 0, "check_c.py": 0, "check_b.py": 1}
    out = diff_against_previous(current, previous)
    assert out["new_red"] == ["check_a.py"]
    assert out["still_red"] == ["check_b.py"]
    assert out["fixed"] == []


def test_first_run_treats_all_red_as_new():
    out = diff_against_previous({"check_a.py": 1}, None)
    assert out["new_red"] == ["check_a.py"]


def test_run_gate_reports_true_exit_code(tmp_path):
    """退出码必须来自子进程本身——不能从管道尾部取（本会话反复踩过的坑）。"""
    script = tmp_path / "check_always_fails.py"
    script.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
    rc, tail = run_gate(script)
    assert rc == 3
    assert isinstance(tail, str)


def test_check_only_writes_nothing():
    log_dir = ROOT / "logs"
    before = set(log_dir.glob("provenance_gates*")) if log_dir.is_dir() else set()
    assert main(["--check-only"]) == 0          # 当前仓应全绿
    after = set(log_dir.glob("provenance_gates*")) if log_dir.is_dir() else set()
    assert after == before, "check-only 模式不得落盘"


def test_state_payload_shape():
    """落盘状态必须含红集合与时间戳，供下次做差；缺字段等于没有防回归。"""
    state = ROOT / "logs" / "provenance_gates_latest.json"
    assert main([]) == 0
    assert state.is_file()
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert {"gates", "red", "new_red", "started_at"} <= set(payload)
    assert payload["red"] == {}   # red 是 {闸名: 退出码}，全绿即空 dict
