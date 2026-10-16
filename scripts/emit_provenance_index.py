"""判据件 hash 清单：把「矩阵依赖哪些实物、内容指纹是什么」落成可公开的入仓件。

为什么需要（09-30 J1）：矩阵 26 行依赖的 34 个实物里 **25 个 untracked** ⇒ 我们说"数值可复算"，
外部拿不到件就无法验证（T11 干净克隆试跑已证缺 29/33）。搬实物进公开仓会重开版权与 PII 面
（题级产物含 `model_answer`），所以这一步只做**指纹登记**：路径 + sha256 + 字节数 + 跟踪状态 +
被哪些主张依赖。这样"可复算"从空口断言变成可核验的清单——读者至少能确认我们给的数出自哪个件。

清单里绝不写绝对路径与机器名（本件要进公开仓），且 `tests/test_provenance_index.py` 反向钉住这点。

用法：
  .venv/bin/python scripts/emit_provenance_index.py                 # 裸跑＝只读比对（闸用这个）
  .venv/bin/python scripts/emit_provenance_index.py --write         # 写盘刷新登记件
  .venv/bin/python scripts/emit_provenance_index.py --require-present  # 写盘时缺实物即判红
退出码：0 = 已写出；1 = 有依赖实物缺失且要求全在（--require-present）。
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.claim_specs import all_claims  # noqa: E402

DEFAULT_OUT = "provenance/artifact-index.json"
# 这些前缀下的是题级大产物（含 model_answer/引语），只做指纹登记，不入公开仓
PUBLIC_SAFE_PREFIXES = ("benchmark/eval/n3_regen/n3_aggregation_summary.json",
                        "benchmark/eval/counterfactual/analysis.json",
                        "benchmark/eval/n3_regen/b1_verdict.json",
                        "benchmark/eval/judge_audit/",
                        "benchmark/eval/a800/")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked_paths(root: Path) -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True)
    return set((out.stdout or "").split())


def referenced_artifacts(root: Path) -> dict[str, list[str]]:
    """{实物相对路径: [依赖它的主张 id]}。

    三个来源取并集，**缺一个就会假绿**：
      1. `claim_specs.supports` 声明的支撑件；
      2. `recompute` 指针直取的件；
      3. 装载件跑起来**实际读过**的件（`claim_evidence.READ_LOG`）——
         B1 就是声明写 patch_summary、实读 b1_verdict 的例子，只取前两项会漏。

    **产物缺席在精简检出里是合法状态**（发布面按裁定只随包逐题评测件，其余分析件留在工作仓）。
    装载件读不到产物时不许甩裸栈：转成 `WorkbenchOnly` 出声退出，把"读不到的依赖"说出来。
    工作仓里产物齐备，这条路走不到。
    """
    from scripts.claim_evidence import READ_LOG, WorkbenchOnly
    from scripts.score_claim_matrix import build_matrix

    deps: dict[str, list[str]] = {}
    try:
        build_matrix(root)                      # 跑一遍才有 READ_LOG
    except FileNotFoundError as exc:
        # 消息只带**仓库相对**路径：本件的自我约束是"清单里绝不写绝对路径与机器名"，
        # 而出声消息同样会被读者看到，带绝对路径等于从这条缝里漏出机器目录。
        rel = str(exc.filename or exc)
        try:
            rel = Path(rel).resolve().relative_to(root.resolve()).as_posix()
        except (ValueError, OSError):
            rel = Path(rel).name
        raise WorkbenchOnly(f"主张矩阵所需产物不在本树：{rel}") from None
    for claim_id in READ_LOG:
        deps.setdefault(claim_id, []).append("(装载件实读)")
    for claim in all_claims():
        found = list(claim.supports)
        expr = claim.recompute.split(" ")[0]
        if expr.startswith("UNBOUND:"):
            pass
        elif "::" in expr:
            found.append(expr.split("::", 1)[0])
        elif "#" in expr:
            found.append(expr.split("#", 1)[0])
        for path in found:
            deps.setdefault(path, []).append(claim.claim_id)
    return deps


def build_index(root: Path) -> dict:
    tracked = tracked_paths(root)
    entries = []
    for path, claims in sorted(referenced_artifacts(root).items()):
        file_path = root / path
        exists = file_path.is_file()
        entries.append({
            "path": path,
            "on_disk": exists,
            "tracked": path in tracked,
            "bytes": file_path.stat().st_size if exists else None,
            "sha256": sha256_of(file_path) if exists else None,
            "public_safe": path.startswith(PUBLIC_SAFE_PREFIXES),
            "required_by": sorted(claims),
        })
    return {
        # 同 claim_anchors：与冻结件 provenance/artifact-index.json 内的 schema 字段配套。
        "schema": "scimonitor.artifact-index/v1",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "generated_by": "scripts/emit_provenance_index.py",
        "purpose": ("主张-证据矩阵的实物依赖清单：指纹级可核验，不含实物本体。"
                    "public_safe=true 的条目是 KB 级聚合件，可作为 artifact 仓白名单候选；"
                    "其余为含题级文本的大产物，只登记指纹。"),
        "counts": {
            "artifacts": len(entries),
            "on_disk": sum(1 for e in entries if e["on_disk"]),
            "missing": sum(1 for e in entries if not e["on_disk"]),
            "untracked": sum(1 for e in entries if not e["tracked"]),
            "public_safe": sum(1 for e in entries if e["public_safe"]),
        },
        "artifacts": entries,
    }


def compare(index: dict, stored: dict) -> list[str]:
    """现算清单与入仓清单的漂移点（路径或指纹不同、条目数不等都算）。

    单独成函数是为了让测试能把「改坏一个 sha」真的喂进同一条比对里——
    判据能不能咬，不能靠在测试里另写一遍比较逻辑来证明。
    """
    if len(index["artifacts"]) != len(stored.get("artifacts", [])):
        return [f"条目数不等：现算 {len(index['artifacts'])} vs 入仓 {len(stored.get('artifacts', []))}"]
    return [entry["path"] for entry, past in zip(index["artifacts"], stored["artifacts"])
            if entry["path"] != past.get("path") or entry["sha256"] != past.get("sha256")]


def main() -> int:
    parser = argparse.ArgumentParser(description="生成判据件 hash 清单")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true",
                        help="只比对：重算与入仓清单是否同一代（漂移即红）。裸跑即此模式")
    parser.add_argument("--write", action="store_true",
                        help="写盘刷新登记件。**写盘必须显式**，见下方注释")
    parser.add_argument("--require-present", action="store_true",
                        help="写盘时：有依赖实物缺失即判红")
    args = parser.parse_args()
    if args.check and args.write:
        print("--write 与 --check 不能同时给", file=sys.stderr)
        return 2

    out_path = (ROOT / args.out).resolve()
    index = build_index(ROOT)
    text = json.dumps(index, ensure_ascii=False, indent=1) + "\n"

    if not args.write:
        # 裸跑＝只读比对，好让常驻闸按统一方式（不带参数）调用它；写盘必须显式 --write。
        # 同 build_supplementary_sample.py 的约定（那条注释是 10-03 立的：`裸跑（不带参数）等价于
        # --check…`）。它此前**默认写盘**，于是被登记进 GATES 后成了一支「绿着的空闸」：runner 不带
        # 参数调它，它把登记件重写一遍再返回 0，什么都没判（实测：索引只多了个 generated_at 就过去了）。
        if not out_path.is_file():
            print(f"[GATE FAIL] 清单缺失：{out_path.relative_to(ROOT)}")
            return 1
        stored = json.loads(out_path.read_text(encoding="utf-8"))
        drift = compare(index, stored)
        if drift:
            print(f"[GATE FAIL] 实物与入仓清单不同代（{len(drift)} 处）：{drift[:6]}")
            return 1
        print(f"[GATE OK] {len(index['artifacts'])} 个实物指纹全部与入仓清单同代")
        return 0

    tmp = out_path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out_path)
    counts = index["counts"]
    print(f"已生成 {out_path.relative_to(ROOT)}：{counts['artifacts']} 个实物"
          f"（在盘 {counts['on_disk']}／缺 {counts['missing']}／"
          f"untracked {counts['untracked']}／public_safe {counts['public_safe']}）")
    return 1 if (args.require_present and counts["missing"]) else 0


if __name__ == "__main__":
    from scripts.claim_evidence import WorkbenchOnly  # noqa: E402

    try:
        sys.exit(main())
    except WorkbenchOnly as exc:
        print(f"[WORKBENCH-ONLY] {exc}")
        print("[UNCHECKED] 这一项在本树不可复算 ⇒ 记为未验，不得当通过")
        sys.exit(1)
