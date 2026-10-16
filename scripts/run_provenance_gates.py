#!/usr/bin/env python3
"""溯源闸执行载体：把十道闸一次跑完，与上次红集合做差，只报"新增红"。

为什么需要它：闸存在 ≠ 闸在跑。本仓无 remote ⇒ `.github/workflows/test.yml` 里那些
步骤在真实 runner 上永远不会触发；而第八道闸抓到的"产物字段相对自己的 records 过期"
已在盘上存在五十多天，没有任何自动检查会撞上它。本脚本就是那个"自动"：
    .venv/bin/python scripts/run_provenance_gates.py            # 跑 + 落状态
    .venv/bin/python scripts/run_provenance_gates.py --check-only   # 只判，不写任何文件

登记制而非盲目自动发现：`scripts/` 下 `check|verify|audit_` 前缀共 17 支，其中 6 支
`*_health.py` 查的是"实验进程在不在跑"、1 支是会写盘的生成工具 —— 按前缀全收会让
提交前闸去依赖运行时状态，既慢又会误红。所以用显式 GATES，并配 **未登记即红**：
新增一支匹配前缀的脚本而没进任一清单，本脚本直接判失败，逼一次决策（双向防腐烂）。

判据不以"全绿"为参照（AGENTS.md 收官判据）：与 `logs/provenance_gates_latest.json`
里上次的红集合做差，只对 **new_red** 报失败；已知红会如实列出但不重复判失败。

用法：
    .venv/bin/python scripts/run_provenance_gates.py [--check-only] [--quiet]
退出码：0 = 无新增红且无未登记脚本；1 = 有新增红 / 有未登记脚本 / allowlist 类失败。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 溯源/规范闸清单（与 .github/workflows/test.yml 的 Provenance audit 步保持一致）。
# 条数不写进注释：写死就等于把清单登第二遍，加一支就漂一次。汇总行打印的是 len(GATES)。
GATES: tuple[str, ...] = (
    "check_lexec_dialect.py",
    "verify_refactoring.py",
)
# 匹配同一前缀但**故意不作为闸跑**的脚本，理由必须写出来（空理由等于没解释）
NOT_GATES: dict[str, str] = {
    "audit_corpus_license.py": "Licence-ledger generator: fetches per-paper arXiv metadata over the network. An action, not a verdict — it writes a ledger rather than judging one",
    "audit_question_pool.py": "Structural health check for the pool. Re-runnable, and it reports rather than passing or failing",
    "check_figure_ledger.py": "Three-way reconciliation of figure numbers against the manuscript's number ledger and the source products. This tree ships the source products but not the manuscript face, so there is no second side to reconcile against here",
    "check_local9b_health.py": "Liveness probe for a running training host. A public checkout has no such process to watch",
}
PREFIXES = ("check_", "verify_", "audit_")
STATE_REL = "logs/provenance_gates_latest.json"


def repo_root() -> Path:
    for base in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for cand in (base, *base.parents):
            if (cand / "scripts").is_dir() and (cand / "benchmark").is_dir():
                return cand
    raise SystemExit(f"[GATE RUNNER] repo root not found (cwd={Path.cwd()})")


def candidate_scripts(root: Path) -> list[Path]:
    scripts = root / "scripts"
    if not scripts.is_dir():
        return []
    return sorted(p for p in scripts.glob("*.py") if p.name.startswith(PREFIXES))


def unregistered_candidates(root: Path) -> list[str]:
    """匹配前缀但既不在 GATES 也不在 NOT_GATES ⇒ 必须有人来登记一次。"""
    known = set(GATES) | set(NOT_GATES)
    return [p.name for p in candidate_scripts(root) if p.name not in known]


def run_gate(script: Path, timeout: int = 900) -> tuple[int, str]:
    """跑一支闸，返回 (真实退出码, 输出尾部)。

    退出码直接取 CompletedProcess.returncode —— 不经管道，避免"量到的是 head"。"""
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          timeout=timeout)
    tail = "\n".join((proc.stdout or proc.stderr).strip().split("\n")[-6:])
    return proc.returncode, tail


def diff_against_previous(current: dict[str, int], previous: dict[str, int] | None) -> dict:
    # 只把**上次非零**的当已知红；若把绿项也算进去，"上次绿、这次红"就会被
    # 误判成"仍红"，新增红永远报不出来。
    prev_red = {k for k, v in (previous or {}).items() if v != 0}
    cur_red = {k for k, v in current.items() if v != 0}
    return {"new_red": sorted(cur_red - prev_red),
            "still_red": sorted(cur_red & prev_red),
            "fixed": sorted(prev_red - cur_red)}


def load_previous(root: Path) -> dict[str, int] | None:
    path = root / STATE_REL
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return {name: code for name, code in payload.get("red", {}).items()} if payload else None


def write_state(root: Path, results: dict[str, int], delta: dict, started: str) -> None:
    path = root / STATE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"started_at": started,
               "gates": len(results),
               "red": {k: v for k, v in results.items() if v != 0},
               "new_red": delta["new_red"], "still_red": delta["still_red"],
               "fixed": delta["fixed"]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    with (path.parent / "provenance_gates.log").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-only", action="store_true",
                        help="只判定、不写任何文件（供测试与安全的手动预检使用）")
    parser.add_argument("--quiet", action="store_true", help="只打印汇总行")
    args = parser.parse_args(argv)

    root = repo_root()
    started = datetime.now().isoformat(timespec="seconds")
    missing = [g for g in GATES if not (root / "scripts" / g).is_file()]
    if missing:
        print(f"[GATE RUNNER FAIL] 登记的闸在盘上不存在：{missing}")
        return 1

    results: dict[str, int] = {}
    for gate in GATES:
        code, tail = run_gate(root / "scripts" / gate)
        results[gate] = code
        if not args.quiet:
            flag = "✅" if code == 0 else "❌"
            print(f"  {flag} {gate:<38} rc={code}")
            if code != 0:
                print("      " + tail.replace("\n", "\n      "))

    unregistered = unregistered_candidates(root)
    previous = None if args.check_only else load_previous(root)
    delta = diff_against_previous(results, previous)
    if not args.check_only:
        write_state(root, results, delta, started)

    print(f"\n  闸 {len(results)} 道：新增红 {len(delta['new_red'])}、"
          f"仍红 {len(delta['still_red'])}、转绿 {len(delta['fixed'])}；"
          f"未登记脚本 {len(unregistered)}")
    if unregistered:
        print(f"  [MISS] 匹配闸前缀但未登记，请加入 GATES 或 NOT_GATES（带理由）："
              f"{unregistered}")
    failed = bool(delta["new_red"] or unregistered)
    print("  " + ("[GATE RUNNER FAIL] 见上。" if failed else "✅ 无新增红"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
