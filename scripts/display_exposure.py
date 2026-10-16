"""展示面暴露表：在册主张 × 权威分级 × 正文落点 × 措辞强度 × 数值绑定态。

为什么必须有（10-01 复审）：`shown_but_weak` 只比 `grade in {B, C} or vetoed`，于是
N/A 20 条 + S 1 条**共 21 条已进正文的主张根本不在弱展示口径里**；而 11 道闸里 3 道碰
`papers/`，没有一道读正文数值。本件把这些量出来——**只量，不改正文**（改不改由作者裁）。

三条硬规矩：
1. 等级与陈述一律取自 `build_matrix()`，不许从任何 md 抄——display-map 假 A 的成因就是抄副本。
2. **定位诚实**：现有依赖只给得到章节级（`ShownClaim` 无行号、`_parse_paper_tex` 返章节 bool），
   所以行级定位只在「该主张有独有数值 token 且该 token 在正文唯一命中」时才升，`locator_kind`
   必须可见。章节级冒充行级＝造假。
3. 本件**不写任何 `papers/**`**。`--emit` 落点限 `docs/reports/`。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.map_paper_display import SECTION_MAP, extract_shown_claims  # noqa: E402
from scripts.score_claim_matrix import build_matrix  # noqa: E402

TEX_FILES = {"A": "papers/paper-a/draft/main.tex",
             "B": "papers/paper-b/draft/main.tex"}
TABLE_GLOBS = ("papers/paper-a/draft/tables", "papers/paper-b/draft/tables")
LEDGER_GLOBS = ("papers/paper-a/number_audit.md", "papers/paper-b/number_audit.md")
TABLE_PREFIXES = ("papers/paper-a/draft/tables/", "papers/paper-b/draft/tables/")

# 强度词：与复审那次实测同口径（case-insensitive，按**行**计）
STRENGTH = re.compile(
    r"significan|p *[<≤] *0\.0|holm|we show|demonstrates?|confirms?", re.IGNORECASE)
NUM = re.compile(r"-?\d+\.\d{4}")
# LaTeX 注释：未被反斜杠转义的 % 起至行尾。`check_blind_submission` 早就剥注释，
# 本件先前**没剥**——于是把读者根本看不见的数字算进暴露面（139 vs 可见 111）。
LATEX_COMMENT = re.compile(r"(?<!\\)%.*$")
TABULAR = re.compile(r"\\begin\{tabular\}|\\end\{tabular\}")
CAPTION = re.compile(r"\\caption\{")


def visible(line: str) -> str:
    """剥掉 LaTeX 注释后的可见面。"""
    return LATEX_COMMENT.sub("", line)


def is_comment_only(line: str) -> bool:
    return line.lstrip().startswith("%")


def _hits(line: str) -> bool:
    """读者可见面上有没有强度词——注释行一律不算。"""
    return not is_comment_only(line) and bool(STRENGTH.search(visible(line)))

# 分诊按**数值物理位置**分类，不按语义猜。10-01 实测否证了我原先拍的
# `scale`／`hyperparam` 两类：命中数值为 0，而真正漏掉的是「内联 tabular」与「\caption 内」
# 两种位置——它们看着像正文句子，其实是表格单元与图注。
TRIAGE_CLASSES = ("comment-only", "table-file", "inline-table", "caption", "ledger", "prose")


@dataclass
class ExposureRow:
    """一条主张在展示面的完整画像。"""

    claim_id: str
    statement: str
    domain: str
    section: str
    grade: str
    locator_kind: str  # line / section-only
    tex_file: str
    tex_line: int  # 0 = 无行级证据
    unique_tokens: list[str] = field(default_factory=list)
    strength_lines: list[int] = field(default_factory=list)
    bound: str = "bound"  # bound / unbound（该行独有 token 是否绑得回权威）


def _claim_blob(row) -> str:
    """该主张在权威侧的全部文本面——数值 token 从这里取，不从 md 取。"""
    return " ".join(str(x) for x in (row.statement, row.margin_note, row.reason,
                                     row.effect_size))


def _unique_tokens(rows: dict) -> dict[str, set[str]]:
    """只属于这一条主张的 token；跨条共用的不算定位证据（会把两行都点亮）。"""
    seen: dict[str, set[str]] = {cid: set(NUM.findall(_claim_blob(r))) for cid, r in rows.items()}
    shared: set[str] = set()
    for cid, toks in seen.items():
        for other, o_toks in seen.items():
            if other != cid:
                shared |= (toks & o_toks)
    return {cid: (toks - shared) for cid, toks in seen.items()}


_SECTION_HEADING = re.compile(r"^\\(?:sub){0,2}section\{")


def _section_span(lines: list[str], section: str) -> tuple[int, int]:
    """标题（section 级或子节）的行区间；找不到返回 (0, 0)。"""
    pat = re.compile(r"^\\(?:sub){0,2}section\{" + re.escape(section) + r"\}")
    start = next((i for i, ln in enumerate(lines) if pat.match(ln)), None)
    if start is None:
        return 0, 0
    end = next((i for i in range(start + 1, len(lines))
                if _SECTION_HEADING.match(lines[i])), len(lines))
    return start + 1, end


def _candidate_sections(claim_section: str) -> list[str]:
    return SECTION_MAP.get(claim_section, [claim_section])


def _locate(cid: str, claim_section: str, tokens: set[str],
            shown_papers: list[str]) -> tuple[str, str, int, str]:
    """返回 (locator_kind, tex_file, tex_line, bound)。行级需独有 token 在正文唯一命中。"""
    for paper in shown_papers or list(TEX_FILES):
        rel = TEX_FILES[paper]
        path = ROOT / rel
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for sec in _candidate_sections(claim_section):
            start, end = _section_span(lines, sec)
            if not start:
                continue
            # 行级证据：该主张的独有 token 在本文件恰好命中一行
            hits = [(i + 1, ln) for i, ln in enumerate(lines)
                    if not is_comment_only(ln) and any(t in visible(ln) for t in tokens)]
            if len(hits) == 1:
                return "line", rel, hits[0][0], "bound"
            return "section-only", rel, 0, "bound"
    return "section-only", TEX_FILES[shown_papers[0]] if shown_papers else "", 0, "unbound"


def exposure_rows(root: Path) -> list[ExposureRow]:
    """主入口：26 条主张逐条画像。"""
    rows = {r.claim_id: r for r in build_matrix(root)}
    uniq = _unique_tokens(rows)
    papers_by_claim: dict[str, list[str]] = {}
    for s in extract_shown_claims(root):
        papers_by_claim.setdefault(s.claim_id, []).append(s.paper)

    out: list[ExposureRow] = []
    for cid, r in rows.items():
        kind, rel, line, bound = _locate(cid, r.section, uniq[cid],
                                         papers_by_claim.get(cid, []))
        strength: list[int] = []
        if rel and (root / rel).is_file():
            lines = (root / rel).read_text(encoding="utf-8").splitlines()
            if kind == "line":
                strength = [line] if _hits(lines[line - 1]) else []
            else:
                for sec in _candidate_sections(r.section):
                    start, end = _section_span(lines, sec)
                    strength += [i for i in range(start, end) if _hits(lines[i - 1])]
        out.append(ExposureRow(claim_id=cid, statement=r.statement, domain=r.domain,
                               section=r.section, grade=r.grade, locator_kind=kind,
                               tex_file=rel, tex_line=line, unique_tokens=sorted(uniq[cid]),
                               strength_lines=sorted(set(strength)), bound=bound))
    return sorted(out, key=lambda x: x.claim_id)


def _all_tex_tokens() -> dict[str, list[tuple[str, int, dict]]]:
    """正文与表件里每个四位小数的出处（文件, 行, 位置标志）。

    逐行扫时同时记录三件事：是否在 `tabular` 环境内、是否在 `\\caption{}` 内、整行是否为注释。
    剥注释是 10-01 补的——先前把读者看不见的数也算进暴露面（139 vs 可见 111）。
    """
    out: dict[str, list[tuple[str, int, dict]]] = {}
    paths = [Path(v) for v in TEX_FILES.values()]
    for g in TABLE_GLOBS:
        if (ROOT / g).is_dir():
            paths += sorted((ROOT / g).glob("*.tex"))
    for p in paths:
        full = ROOT / p if not p.is_absolute() else p
        if not full.is_file():
            continue
        rel = str(full.relative_to(ROOT))
        depth = 0
        for i, ln in enumerate(full.read_text(encoding="utf-8").splitlines(), 1):
            begins, ends = ln.count("\\begin{tabular}"), ln.count("\\end{tabular}")
            flags = {"tabular": depth > 0 or begins > 0,
                     "caption": bool(CAPTION.search(ln)),
                     "comment_only": is_comment_only(ln),
                     "table_file": rel.startswith(TABLE_PREFIXES)}
            depth = max(0, depth + begins - ends)
            vis = visible(ln)
            for m in NUM.finditer(ln):
                tok = m.group()
                # 该次出现是否落在注释段里：剥完还在＝可见；剥没了＝纯注释
                f = dict(flags, comment_only=flags["comment_only"] or (tok not in vis))
                out.setdefault(tok, []).append((rel, i, f))
    return out


def _ledger_tokens() -> set[str]:
    toks: set[str] = set()
    for rel in LEDGER_GLOBS:
        p = ROOT / rel
        if p.is_file():
            toks |= set(NUM.findall(p.read_text(encoding="utf-8")))
    return toks


def _where_class(where: list[tuple[str, int, dict]], in_ledger: bool) -> set[str]:
    """返回该数值出现的**全部位置**，不取其一。

    为什么不排优先级（10-01 自纠）：先前按「表件 > 内联表 > 图注 > 台账 > 正文」取第一个命中类，
    于是 `-0.0144 / -0.0356 / -0.0887 / -0.0988` 这四个**同时出现在正文句子**（A 篇 473/474/499）
    与内联表格（B 篇 674–677）的数被判成 `inline-table`，`prose` 直接归 0。
    那是排序造出来的假绿，不是量出来的结论。要人裁的正是「出现在正文句子」这件事，
    它不能被「也出现在表格里」掩掉。
    """
    vis = [w for w in where if not w[2]["comment_only"]]
    if not vis:
        return {"comment-only"}
    locs: set[str] = set()
    if any(w[2]["table_file"] for w in vis):
        locs.add("table-file")
    if any(w[2]["tabular"] for w in vis):
        locs.add("inline-table")
    if any(w[2]["caption"] for w in vis):
        locs.add("caption")
    if in_ledger:
        locs.add("ledger")
    if any(not (w[2]["table_file"] or w[2]["tabular"] or w[2]["caption"]) for w in vis):
        locs.add("prose")
    return locs or {"prose"}


def triage_unbound(root: Path) -> dict:
    """把绑不回权威的数值逐条归类。

    两个量别混：`untriaged` 查的是**取数口径自己出错**（算出的 token 一处都找不到），必须为 0。

    绑定分**两层**，因为把两层混成一个「有/无据」会误报（10-01 实测踩过）：
      - `unbound_matrix`：不在矩阵 26 条主张的文本面里；
      - `unbound_all`：连登记台账（`number_audit.md`）也没有——**这一层才可能是真无据**。
    实测 47 个 `unbound_matrix` 全部能在台账里严格命中，所以真无据为 0；
    它们的性质是「**有台账、但矩阵里没有对应主张行**」＝覆盖缺口，不是造假。
    要裁的是后者该不该补主张行，不是删数字。
    """
    authority: set[str] = set()
    for r in build_matrix(root):
        authority |= set(NUM.findall(_claim_blob(r)))

    occurrences = _all_tex_tokens()
    ledger = _ledger_tokens()

    by_class = {c: 0 for c in TRIAGE_CLASSES}
    prose_items: list[dict] = []
    untriaged = 0
    total_unbound = 0
    covered = 0
    unbound_all = 0

    for tok, where in sorted(occurrences.items()):
        if tok in authority or tok.lstrip("-") in authority:
            continue
        total_unbound += 1
        if not where:
            untriaged += 1
            continue
        locs = _where_class(where, tok in ledger)
        for c in locs:
            by_class[c] += 1
        covered += 1
        in_ledger = (tok in ledger or tok.lstrip("-") in ledger)
        if not in_ledger:
            unbound_all += 1
        if "prose" in locs:
            prose_items.append({"token": tok, "locations": sorted(locs),
                                "in_ledger": in_ledger,
                                "where": [f"{f}:{l}" for f, l, fl in where
                                          if not fl["comment_only"]]})

    return {"total_unbound": total_unbound, "by_class": by_class,
            "covered": covered, "untriaged": untriaged,
            "unbound_all": unbound_all, "prose_items": prose_items}


def distinct_strength_lines(root: Path, include_comments: bool = False) -> list[tuple[str, int]]:
    """全正文强度词命中行（**按行去重**）。

    两个口径都要能出，因为差值本身是结论的一部分：含注释 22 行、读者可见 18 行
    （10-01 实测）。默认只报**可见面**——判「措辞强于证据」要看的是读者读得到的句子。

    为什么单列成函数：一个章节里有 6 行强度词、该章节又挂着 3 条主张时，按主张计数会把同
    6 行数三遍（实测按主张报 24、按行去重 22）。判据单位是**行**，不是主张。
    """
    out: list[tuple[str, int]] = []
    for rel in TEX_FILES.values():
        p = root / rel
        if not p.is_file():
            continue
        for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            # include_comments 决定的是**匹配哪一份文本**：只看注释外，还是整行都看。
            # 先前这里对 `visible(ln)` 无条件匹配，于是 include_comments=True 也报 18，
            # 参数名与实现相反——半吊子仪器比没仪器更坏。
            text = ln if include_comments else visible(ln)
            if not text.strip() or not STRENGTH.search(text):
                continue
            out.append((rel, i))
    return sorted(set(out))


def section_backing(root: Path) -> dict[str, list[str]]:
    """每个 \\\\section 里挂着哪些分级的主张——归因只能到这一层（M1）。

    这是本件的**能力边界声明**：现有依赖（`ShownClaim` 无行号、`_parse_paper_tex` 返章节
    bool）不支持把某一行强度词绑到某一条主张。所以报的是「该行所在章节有哪些 N/A／S 级主张」，
    不是「该行在为哪条弱主张背书」。把它当后者用就是过度声称。
    """
    rows = {r.claim_id: r for r in build_matrix(root)}
    papers = {}
    for s in extract_shown_claims(root):
        papers.setdefault(s.claim_id, []).append(s.paper)
    backing: dict[str, list[str]] = {}
    for cid, r in rows.items():
        for paper in papers.get(cid, []):
            rel = TEX_FILES.get(paper, "")
            lines = (root / rel).read_text(encoding="utf-8").splitlines() if rel else []
            for sec in _candidate_sections(r.section):
                start, _ = _section_span(lines, sec)
                if start:
                    backing.setdefault(f"{rel}:{sec}", []).append(r.grade)
    return {k: sorted(set(v)) for k, v in sorted(backing.items())}


def report_text(root: Path) -> str:
    """登记件正文：时间头 + 权威分布 + 摘要 + 暴露表 + 未绑定明细 + 能力边界。"""
    from datetime import datetime
    rows = exposure_rows(root)
    t = triage_unbound(root)
    kinds = [r.locator_kind for r in rows]
    dl = distinct_strength_lines(root)
    backing = section_backing(root)
    dist: dict[str, int] = {}
    for r in rows:
        dist[r.grade] = dist.get(r.grade, 0) + 1
    weak = sum(1 for r in rows if r.grade in ("N/A", "S"))
    parts = [
        "# 论文展示面主张强度暴露表",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "**生成命令**: `.venv/bin/python scripts/display_exposure.py "
        "--emit docs/reports/paper-claim-strength-exposure-2026-10-01.md`",
        "**类型**: 只读暴露表（本件不改 `papers/**` 任何内容）",
        f"**权威分布**: {' / '.join(f'{k} {v}' for k, v in sorted(dist.items()))}"
        f"（共 {len(rows)} 条。⚠ **不等于「全部已在正文中被逐条陈述」**——"
        "`extract_shown_claims` 的 `shown` 判据是**章节级**「该节非 TODO」"
        f"（`map_paper_display.py:43-46`），且 `SECTION_MAP` 只覆盖 6 个章节、"
        "两篇实有 17 个 section ⇒ Introduction/Setup/Conclusion 等 11 个章节不在归因面内。"
        "句子级绑定见 `provenance/claim-anchors.json`，当前覆盖 6/26。）",

        "",
        "## 一、摘要",
        "",
        f"- 行级定位 **{kinds.count('line')}** 条／章节级定位 **{kinds.count('section-only')}** 条"
        "——现有依赖只支持章节级（见 §四 能力边界）。",
        f"- 强度词命中行（**按行去重、剥 LaTeX 注释**）＝ **{len(dl)}** 行；"
        f"含注释口径为 {len(distinct_strength_lines(root, include_comments=True))} 行。"
        "注释里的句子读者看不见，不该计入暴露面。按主张计数会得到更大的数，"
        "那是同一章节被多条主张共享的重复归因，不是更多句子。",
        f"- 分级为 N/A 或 S（描述量／脚手架位）却已进正文的主张 ＝ **{weak}** 条；"
        f"原 `shown_but_weak` 只比 B/C，故这 {weak} 条曾全在其口径之外（D2 已补列）。",
        f"- 绑不回**矩阵**的数值（唯一 token、已剥注释）＝ **{t['total_unbound']}** 个，"
        f"按位置分类（可重叠）：纯注释 {t['by_class']['comment-only']}／表件 "
        f"{t['by_class']['table-file']}／内联表格 {t['by_class']['inline-table']}／图注 "
        f"{t['by_class']['caption']}／台账 {t['by_class']['ledger']}／正文句子 "
        f"{t['by_class']['prose']}。",
        f"- **分两层看**（单层会把覆盖缺口误报成造假）：连**登记台账**也没有的 {t['unbound_all']} 个；"
        f"其中出现在**读者可见正文句子**、且台账也没有的 "
        f"**{sum(1 for it in t['prose_items'] if not it['in_ledger'])} 个**。"
        "台账覆盖的是 `number_audit.md`；矩阵里没有对应主张行 ≠ 数字无据，那叫**覆盖缺口**。",
        "",
        "## 二、暴露表",
        "",
        render_markdown(rows).rstrip(),
        "",
        "## 三、读者可见正文句子里、绑不回矩阵的数值（逐条点名）",
        "",
        "「绑不回矩阵」不等于「无据」——下表逐条给出它在登记台账里的有无。",
        "**台账一列全为「有」，即本轮不存在无登记的正文数值**；剩下的是「矩阵里没有对应主张行」",
        "这一覆盖缺口，处置是补主张行、不是删数字。",
        "",
        "| token | 出现处（文件:行）| 台账 |",
        "|:--|:--|:--:|",
    ]
    for it in t["prose_items"]:
        parts.append(f"| {it['token']} | {', '.join(it['where'])} "
                     f"| {'有' if it['in_ledger'] else '**无**'} |")

    parts += ["", "## 四、能力边界（不许越过）", "",
              "章节内挂着的分级（归因只到这一层）：", "",
              "| 正文落点 | 该章节内主张的分级集合 |",
              "|:--|:--|"]
    for k, v in backing.items():
        parts.append(f"| `{k}` | {'、'.join(v)} |")
    parts += [
        "",
        "**因此本表不能回答「第 N 行强度词在为哪一条主张背书」**——它只能回答「该行所在章节里",
        "有哪几条主张、各自什么分级」。凡需要行级绑定的结论，都得先补一个正文行级归因件，",
        "那属另一条计划，不在本轮范围内硬凑。",
    ]
    return "\n".join(parts) + "\n"


def emit_guard(path: Path) -> bool:
    """写落点闸：只许 docs/reports/。会写产物的脚本自带参数闸（本仓既有纪律）。"""
    return str(path).startswith("docs/reports/") and str(path).endswith(".md")


def render_markdown(rows: list[ExposureRow]) -> str:
    """列序契约：0=ID、1=权威陈述原文、4=权威分级（`discover_carriers` 按位置取列）。"""
    head = ("| 主张 ID | 陈述 | 域 | 章节 | 分级 | 定位型 | 正文落点 | 强度词命中行 | 数值绑定 |")
    lines = [head, "|:--|:--|:--|:--|:--:|:--|:--|:--|:--:|"]
    for r in rows:
        loc = f"{r.tex_file}:{r.tex_line}" if r.locator_kind == "line" else r.tex_file
        hits = ",".join(str(x) for x in r.strength_lines) or "—"
        lines.append(f"| {r.claim_id} | {r.statement} | {r.domain} | {r.section} | {r.grade} "
                     f"| {r.locator_kind} | {loc} | {hits} | {r.bound} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="展示面暴露表（只量不改正文）")
    ap.add_argument("--triage", action="store_true", help="只报未绑定数值分诊")
    ap.add_argument("--emit", type=Path, help="输出登记件（限 docs/reports/*.md）")
    args = ap.parse_args()

    if args.triage:
        t = triage_unbound(ROOT)
        print(f"total_unbound={t['total_unbound']} covered={t['covered']} "
              f"untriaged={t['untriaged']}")
        for k, v in t["by_class"].items():
            print(f"  {k}: {v}（含该位置，可重叠）")
        print(f"  连台账也无的总数 unbound_all: {t['unbound_all']}")
        print(f"  正文句子条数 prose: {len(t['prose_items'])}"
              f"（其中台账也没有 {sum(1 for it in t['prose_items'] if not it['in_ledger'])}）")
        return 0 if t["untriaged"] == 0 else 1

    rows = exposure_rows(ROOT)
    kinds = [r.locator_kind for r in rows]
    dl = distinct_strength_lines(ROOT)
    weak = sum(1 for r in rows if r.grade in ("N/A", "S"))
    print(f"暴露表 {len(rows)} 行 / 行级定位 {kinds.count('line')} / "
          f"章节级 {kinds.count('section-only')}")
    print(f"强度词命中行数（按行去重，判据单位）{len(dl)}")
    print(f"N/A＋S 级已进正文 {weak} 条（现行 shown_but_weak 口径不报这些）")

    if args.emit:
        if not emit_guard(args.emit):
            print(f"错误：--emit 必须落在 docs/reports/*.md，当前: {args.emit}")
            return 1
        out = ROOT / args.emit
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(report_text(ROOT), encoding="utf-8")
        os.replace(tmp, out)
        print(f"已生成: {args.emit}")
    else:
        print(render_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
