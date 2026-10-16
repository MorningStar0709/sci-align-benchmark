import argparse
from collections import Counter
from pathlib import Path


# ROOT = Path(__file__).resolve().parents[2]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead

from mineru_eval.utils import load_json


def normalize_issues(value):
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        if value.strip().lower() in ("", "none"):
            return []
        return [item.strip() for item in value.split(";") if item.strip()]
    return [str(value).strip()] if str(value).strip() else []


# ── render functions ──────────────────────────────────────────────


def render_header(title, date, target_model, judge_model, scoring_mode):
    """输出 Markdown 标题 + 元数据行."""
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    if date:
        lines.append(f"- **Date**: {date}")
    if target_model:
        lines.append(f"- **Target Model**: {target_model}")
    if judge_model:
        lines.append(f"- **Judge Model**: {judge_model}")
    if scoring_mode:
        lines.append(f"- **Scoring Mode**: {scoring_mode}")
    lines.append("")
    return lines


def render_summary_table(dimension_data):
    """输出 Markdown 表格含 Dimension / Questions / Avg Score / Issues 四列.

    dimension_data: {"A": {"count": 100, "avg_score": 0.85, "issues": 12}, ...}
    """
    lines = []
    lines.append("## Score Summary")
    lines.append("")
    lines.append("| Dimension | Questions | Avg Score | Issues |")
    lines.append("|-----------|-----------|-----------|--------|")
    for dim_key in sorted(dimension_data.keys()):
        d = dimension_data[dim_key]
        count = d.get("count", 0)
        avg = d.get("avg_score", 0.0)
        issues = d.get("issues", 0)
        lines.append(f"| {dim_key} | {count} | {avg:.4f} | {issues} |")
    lines.append("")
    return lines


def render_dimension_detail(dimension, records):
    """对每条 record 输出详情.

    record 字段: id, dimension, difficulty_level/level, score,
                 reason, issues, model_answer/answer, expected_answer
    """
    lines = []
    for record in records:
        q_id = record.get("id", "?")
        level = record.get("difficulty_level") or record.get("level", "?")
        score = record.get("score", 0.0)
        reason = record.get("reason", "")
        issues = normalize_issues(record.get("issues", ""))
        model_answer = record.get("model_answer") or record.get("answer", "")
        expected_answer = record.get("expected_answer", "")

        lines.append(f"### {q_id} (Dimension {dimension}, {level})")
        lines.append("")
        lines.append(f"**Score**: {score}")
        lines.append("")
        if reason:
            lines.append(f"**Reason**: {reason}")
            lines.append("")
        if issues:
            lines.append(f"**Issues**: {'; '.join(issues)}")
            lines.append("")
        if model_answer:
            lines.append(f"**Model answer**: {model_answer}")
            lines.append("")
        if expected_answer:
            lines.append(f"**Expected answer**: {expected_answer}")
            lines.append("")
    return lines


def render_issues_summary(issues_counter):
    """对高频 issue 输出排行列表."""
    lines = []
    lines.append("## Issues Summary")
    lines.append("")
    if not issues_counter:
        lines.append("No issues found.")
        lines.append("")
        return lines

    sorted_issues = sorted(issues_counter.items(), key=lambda x: x[1], reverse=True)
    lines.append("| Issue | Count |")
    lines.append("|-------|-------|")
    for issue, count in sorted_issues:
        lines.append(f"| {issue} | {count} |")
    lines.append("")
    return lines


def render_report(meta, records, issues_counter, format="md"):
    """组装 header + summary + details + issues_summary 输出完整 Markdown."""
    title = meta.get("title", "Benchmark Report")
    date = meta.get("date", "")
    target_model = meta.get("target_model") or meta.get("model", "")
    judge_model = meta.get("judge_model", "")
    scoring_mode = meta.get("scoring_mode", "")

    lines = []
    lines.extend(render_header(title, date, target_model, judge_model, scoring_mode))

    # ── overall score ──
    overall = meta.get("overall_score")
    if overall is not None:
        lines.append(f"**Overall Score**: {overall}")
        lines.append("")

    # ── compute per-dimension stats from records ──
    by_dim = {}
    for r in records:
        dim = r.get("dimension", "?")
        if dim not in by_dim:
            by_dim[dim] = {"count": 0, "total_score": 0.0, "issues_count": 0}
        by_dim[dim]["count"] += 1
        by_dim[dim]["total_score"] += r.get("score", 0.0)
        issues = normalize_issues(r.get("issues", ""))
        if issues:
            by_dim[dim]["issues_count"] += 1

    dimension_data = {}
    for dim in sorted(by_dim.keys()):
        d = by_dim[dim]
        avg = d["total_score"] / d["count"] if d["count"] > 0 else 0.0
        dimension_data[dim] = {
            "count": d["count"],
            "avg_score": round(avg, 4),
            "issues": d["issues_count"],
        }

    lines.extend(render_summary_table(dimension_data))

    # ── dimension details ──
    lines.append("## Dimension Details")
    lines.append("")
    for dim in sorted(by_dim.keys()):
        dim_records = [r for r in records if r.get("dimension") == dim]
        lines.extend(render_dimension_detail(dim, dim_records))

    # ── issues summary ──
    lines.extend(render_issues_summary(issues_counter))

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown report from benchmark evaluation results."
    )
    parser.add_argument("--input", required=True, help="Input JSON report file path.")
    parser.add_argument(
        "--format",
        default="md",
        choices=["md"],
        help="Output format (default: md).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output file path. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--max-detail",
        type=int,
        default=0,
        help="Max detail records per dimension (0 = unlimited).",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    data = load_json(input_path)

    # ── build meta dict from flexible input formats ──
    meta = {}
    if "meta" in data:
        meta.update(data["meta"])

    # Top-level fields from evaluate.py / sample_report.json
    for key in ("model", "overall_score", "total_questions", "dimension_scores"):
        if key in data:
            meta.setdefault(key, data[key])

    # ── extract records ──
    records = data.get("records", data.get("questions", []))

    # ── apply --max-detail ──
    max_detail = args.max_detail
    if max_detail > 0:
        by_dim = {}
        for r in records:
            dim = r.get("dimension", "?")
            by_dim.setdefault(dim, []).append(r)
        records = []
        for dim in sorted(by_dim.keys()):
            records.extend(by_dim[dim][:max_detail])

    # ── build issues_counter from records ──
    issues_counter = Counter()
    for r in records:
        for issue in normalize_issues(r.get("issues", "")):
            issues_counter[issue] += 1

    report = render_report(
        meta, records, issues_counter, format=args.format
    )

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Report saved: {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
