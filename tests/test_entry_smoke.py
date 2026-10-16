"""在册入口 CLI 的冒烟件：每个 `--check` 入口跑一遍，断言它**不甩 Python 栈**。

为什么需要它（10-07 实测）：发布仓 390 例全绿的同一棵树里，`emit_provenance_index.py --check`
和 `score_claim_matrix.py --dry-run` 一直在裸崩 `ModuleNotFoundError: holm_wilcoxon`——
套件测的是包与函数，**没有任何一条跑过入口命令本身**。崩溃和"出声的 fail-closed"的差别不是
排版问题：一条给评审执行的命令吐出整段栈，等于当场宣布这仓没人跑过。

判据取「不得出现 Traceback」，不取「必须 rc=0」：本树里合法缺席的工作仓依赖就该判
fail-closed（见 `claim_evidence.WorkbenchOnly`），那是我们要的形态，不是失败。
rc 只允许 0 或 1；2 是 argparse 用法错——参数名猜错或入口根本没实现这条旗标。

范围只收 `--check`，不收 `--dry-run`：训练编排件的 dry-run 仍要权重与 GPU，
不是无网络冒烟。`--check` 面按登记制走——未登记即红，豁免必须带理由。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# 在无网无 GPU 的树里可安全跑的只读入口
CHECK_ENTRIES = {
    "claim_anchors.py",
    "collect_model_identities.py",
    "collect_unattended_walltime.py",
    "emit_provenance_index.py",
}

# 带 --check 但**故意不在套件里跑**的件，理由必须写出来（空理由等于没解释）
CHECK_EXEMPT = {
    "audit_corpus_license.py": "逐篇发 Europe PMC／arXiv 请求取授权元数据 ⇒ 套件不得依赖网络",
    "audit_pool_license.py": "同上（判题池 361 个来源的再分发权）",
    "build_supplementary_sample.py": "在工作仓它是第 13 道闸，由闸套件跑，不在这里跑第二遍",
    "build_preprint_variants.py": "校验对象是 `papers/**/preprint/` 下预印本变体与源稿的同步，"
                                  "属投稿面（不随公开面），且判据挂在 latexmk 产物上 ⇒ 不在此套件里跑",
}

_FLAG = re.compile(r'add_argument\("(--[a-z-]+)"')


def _scripts_with_flag(flag: str) -> set[str]:
    """现扫：本树里哪些 `scripts/*.py` 声明了这条旗标。"""
    out = set()
    for p in sorted(SCRIPTS.glob("*.py")):
        if flag in _FLAG.findall(p.read_text(encoding="utf-8", errors="ignore")):
            out.add(p.name)
    return out


def test_check_flag_roster_is_complete():
    """未登记即红：新增一支带 --check 的入口而没做判断，本件必须响。"""
    found = _scripts_with_flag("--check")
    assert found, "扫不到任何 --check 入口——扫描面为空不得当通过"
    unregistered = found - CHECK_ENTRIES - set(CHECK_EXEMPT)
    assert not unregistered, (
        f"带 --check 却未登记的入口：{sorted(unregistered)}；"
        f"加进 CHECK_ENTRIES（可安全跑）或 CHECK_EXEMPT（带理由）")
    # 只防"件还在、旗标却没了/改名"这一种漂移。名单里的件在本树**不存在**是合法状态
    # （发布面按裁定不随迁若干件），把它算成漂移会让同一份测试在两棵树里给出不同结论。
    present = {p.name for p in SCRIPTS.glob("*.py")}
    stale = {n for n in CHECK_ENTRIES | set(CHECK_EXEMPT) if n in present} - found
    assert not stale, f"名单里的件在本树已不带 --check（旗标改名或入口没实现）：{sorted(stale)}"


def test_check_entries_do_not_crash():
    """每个在册入口：跑 `--check`，不得抛栈，退出码只许 0 或 1。"""
    ran = []
    for name in sorted(CHECK_ENTRIES & _scripts_with_flag("--check")):
        proc = subprocess.run([sys.executable, str(SCRIPTS / name), "--check"],
                              cwd=ROOT, capture_output=True, text=True, timeout=300)
        blob = f"{proc.stdout}\n{proc.stderr}"
        assert "Traceback (most recent call last)" not in blob, (
            f"{name} --check 抛了裸栈（评审会看到 Python 栈而不是结论）：\n{blob[-900:]}")
        assert proc.returncode in (0, 1), (
            f"{name} --check 退出码 {proc.returncode}：2=argparse 用法错，"
            f">1=非预期终止\n{blob[-500:]}")
        ran.append((name, proc.returncode))
    assert len(ran) == len(CHECK_ENTRIES & _scripts_with_flag("--check"))
    print(f"  入口冒烟：{len(ran)} 支跑过，退出码读数 {ran}")
