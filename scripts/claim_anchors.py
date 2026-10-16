"""正文行级锚点：把主张绑到正文里的**具体句子**（D7）。

为什么需要（10-01）：18 行读者可见强度词里，**10 行落在没有被任何主张映射的章节**
（引言 3／结论 1／Setup 1 等）。引言是读者形成第一印象的地方，而现有链条只能给到
「章节 → 分级集合」，判不了「这一句在为哪条主张背书」。

两条设计取舍，都是本轮教训的直接产物：

1. **锚点存引文，不存行号。** 行号会随任何排版移动悄悄错位——那正是本轮反复清掉的
   「副本与实物不同代」。存引文则句子一改就红，不会假绿。行号只作信息位，漂移不判失败。
2. **引文一律机器派生，我不手写。** 提案试验证明关键词法会把 `elasticity_reversal_4b`
   配到讲 perplexity 的句子、把 `medical_scale_axis` 配到论文标题行。那种锚点入册等于
   往审计链注入新的假登记。所以本件只接受**唯一数值 token 命中**这种可自证的绑定；
   其余主张留在「未覆盖」里如实报数，**不用假完整换覆盖率**。

覆盖度是**报告的数字**，不是初始闸门条件：`--check` 不因 6/26 而红，
但任何锚点找不到、或找到多处、或指向不存在的 claim_id，一律红。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.claim_specs import all_claims  # noqa: E402
from scripts.display_exposure import exposure_rows, visible  # noqa: E402

ANCHORS_PATH = "provenance/claim-anchors.json"
# 人工锚点**单独成件**，绝不写进上面那份生成件：人工内容一旦被 `--write` 盖掉，
# 就重演 display-map 那次「生成器一跑，人工修全丢」；一份文件只允许一个生产者。
MANUAL_PATH = "provenance/claim-anchors.manual.json"
# schema 标识与冻结登记件 provenance/claim-anchors.json 内的同名字段配套，改名即对不上。
SCHEMA = "scimonitor.claim-anchors/v1"


def derive_anchors(root: Path) -> list[dict]:
    """只从「独有 token 在正文唯一命中」派生锚点——每条都能自证。"""
    out = []
    for r in exposure_rows(root):
        if r.locator_kind != "line" or not r.tex_file or not r.tex_line:
            continue
        path = root / r.tex_file
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        quote = visible(lines[r.tex_line - 1]).strip()
        if not quote:
            continue
        out.append({"claim_id": r.claim_id, "file": r.tex_file,
                    "quote": quote, "line_hint": r.tex_line,
                    "method": "unique-token"})
    return sorted(out, key=lambda a: a["claim_id"])


def count_occurrences(quote: str, path: Path) -> int:
    """引文在目标文件里出现几次——按可见面数，注释里的重复不算命中。"""
    if not path.is_file():
        return -1
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines()
               if quote in visible(ln))


def check(anchors: list[dict], root: Path) -> tuple[list[str], dict]:
    """返回（问题清单，统计）。统计里的 coverage 只报不判。"""
    known = {c.claim_id for c in all_claims()}
    problems: list[str] = []
    seen: set[str] = set()
    for a in anchors:
        cid = a["claim_id"]
        if cid not in known:
            problems.append(f"{cid}: claim_id 不在 claim_specs 里（假登记）")
        if cid in seen:
            problems.append(f"{cid}: 重复锚点")
        seen.add(cid)
        if a.get("status") == "not-stated":
            continue
        n = count_occurrences(a["quote"], root / a["file"])
        if n == -1:
            problems.append(f"{cid}: 文件不存在 {a['file']}")
        elif n == 0:
            problems.append(f"{cid}: 引文在 {a['file']} 里已找不到（句子被改过）")
        elif n > 1:
            problems.append(f"{cid}: 引文在 {a['file']} 命中 {n} 处，不唯一")
    absent = manual_absent(root)
    both = sorted(set(absent) & seen)
    for cid in both:
        problems.append(f"{cid}: 既记 not-stated 又有锚点（自相矛盾）")
    for cid in absent:
        if cid not in known:
            problems.append(f"{cid}: not-stated 指向不存在的 claim")
    rest = sorted(known - seen - set(absent))
    stats = {"anchors": len(anchors), "claims": len(known),
             "coverage": f"{len(seen & known)}/{len(known)}",
             "not_stated": absent, "uncovered": rest}
    return problems, stats


def _load_json_list(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("anchors", [])


def load(root: Path) -> list[dict]:
    """生成件 + 人工件合并；同一 claim_id 两边都有时去重，不重复计覆盖。"""
    merged: dict[str, dict] = {}
    for a in _load_json_list(root / MANUAL_PATH):
        merged[a["claim_id"]] = a
    for a in _load_json_list(root / ANCHORS_PATH):
        merged.setdefault(a["claim_id"], a)
    return sorted(merged.values(), key=lambda a: a["claim_id"])


def manual_absent(root: Path) -> list[str]:
    """人工判定「正文未陈述」的主张——这也是**可审计的负记录**，不许只是没写。"""
    out = []
    for a in _load_json_list(root / MANUAL_PATH):
        if a.get("status") == "not-stated":
            out.append(a["claim_id"])
    return sorted(out)


def _blocks(rel: str) -> tuple[dict[int, int], list[str]]:
    """按空行切逻辑块。

    为什么不用行级：LaTeX 把句子折成多行，实测 18 个强度词行里只有 1 行与锚点行重合——
    行级归因等于没有归因。段落块（空行分隔）是当前依赖下能做到的最细粒度。
    """
    ls = (ROOT / rel).read_text(encoding="utf-8").splitlines()
    mark: dict[int, int] = {}
    blk = 0
    opened = True
    for i, ln in enumerate(ls, 1):
        if not ln.strip():
            opened = True
            mark[i] = blk
            continue
        if opened:
            blk += 1
            opened = False
        mark[i] = blk
    return mark, ls


def paragraph_attribution(root: Path) -> list[dict]:
    """每个强度词命中行 → 同块内的主张与其分级（段落粒度，可复跑）。"""
    from scripts.display_exposure import distinct_strength_lines, exposure_rows, visible

    rows = {r.claim_id: r for r in exposure_rows(root)}
    lines = distinct_strength_lines(root)
    anchors = load(root)
    out: list[dict] = []
    for rel in sorted({f for f, _ in lines}):
        mark, ls = _blocks(rel)
        by_block: dict[int, list[str]] = {}
        for a in anchors:
            if a["file"] == rel and a.get("status") != "not-stated":
                by_block.setdefault(mark.get(a["line_hint"], -1), []).append(a["claim_id"])
        for f, ln in sorted(x for x in lines if x[0] == rel):
            ids = sorted(set(by_block.get(mark.get(ln, -1), [])))
            out.append({"file": rel, "line": ln, "claims": ids,
                        "grades": sorted({rows[c].grade for c in ids}),
                        "text": visible(ls[ln - 1]).strip()[:70]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="正文行级锚点的派生与合法性校验")
    ap.add_argument("--write", action="store_true", help="派生并原子写入锚点件")
    ap.add_argument("--check", action="store_true", help="只校验在盘锚点件")
    ap.add_argument("--attribute", action="store_true",
                    help="强度词行 → 同段落块内主张与分级（段落粒度归因）")
    args = ap.parse_args()

    if args.write:
        anchors = derive_anchors(ROOT)
        problems, stats = check(anchors, ROOT)
        for why in problems:
            print(f"[ANCHOR FAIL] {why}", file=sys.stderr)
        if problems:
            return 1
        out = ROOT / ANCHORS_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        body = {"schema": SCHEMA, "derivation": "unique-token only（不手写引文）",
                "anchors": anchors}
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
        os.replace(tmp, out)  # 原子写
        print(f"已生成 {ANCHORS_PATH}：{stats['coverage']}，未覆盖 {len(stats['uncovered'])} 条")
        return 0

    if args.attribute:
        att = paragraph_attribution(ROOT)
        weak = [x for x in att if {"N/A", "S", "C"} & set(x["grades"])]
        print(f"强度词行 {len(att)} 行 / 段落块内可归因 {sum(1 for x in att if x['grades'])} 行"
              f" / backing 含 N-A／S／C 的 {len(weak)} 行")
        for x in weak:
            print(f"  {Path(x['file']).parent.parent.name[-6:]}:{x['line']:4d} "
                  f"分级 {','.join(x['grades']):10s} {','.join(x['claims'])[:32]:34s} {x['text'][:44]}")
        return 0

    anchors = load(ROOT)
    problems, stats = check(anchors, ROOT)
    print(f"锚点 {stats['anchors']} 条 / 覆盖 {stats['coverage']}"
          f" / 人工判未陈述 {len(stats['not_stated'])} 条"
          f" / 既无锚点也未判 {len(stats['uncovered'])} 条）")
    if stats["uncovered"]:
        print(f"  未处置：{stats['uncovered']}")
    for why in problems:
        print(f"[ANCHOR FAIL] {why}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
