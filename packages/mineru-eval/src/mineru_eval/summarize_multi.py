import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]

from mineru_eval.utils import iter_report_files, load_json, write_json_atomic


_CHUNK_PATTERN = re.compile(r"_\d{8}_\d{6}_chunk_")


def _is_merged_report(filepath):
    name = filepath.name
    if name.startswith("cross_domain_"):
        return False
    if _CHUNK_PATTERN.search(name):
        return False
    return True


def build_multi_report(results_dir, allow_empty=False):
    results_dir = Path(results_dir)
    json_files = list(iter_report_files(results_dir))
    json_files = [jf for jf in json_files if _is_merged_report(jf)]

    if not json_files:
        print(f"No merged report JSON files found in {results_dir}")
        if not allow_empty:
            raise SystemExit(1)
        return None

    reports = []
    for jf in json_files:
        try:
            data = load_json(jf)
            model_name = data.get("model", jf.stem.replace("output_", ""))

            dim_scores = data.get("dimension_scores", {})
            overall = data.get("overall_score", 0)
            total = data.get("total_questions", 0)

            reports.append({
                "model": model_name,
                "overall_score": overall,
                "total_questions": total,
                "dimension_scores": dim_scores,
                "source_file": jf.name,
                "complete": data.get("complete", total > 0),
                "failure_summary": data.get("failure_summary", {}),
            })
        except Exception as e:
            print(f"Skipping {jf.name}: {e}")

    return reports


def render_leaderboard_table(reports):
    all_dims = sorted(set(d for r in reports for d in r.get("dimension_scores", {}).keys()))

    lines = []
    lines.append("# Multi-Model Evaluation Leaderboard")
    lines.append("")
    lines.append("| Rank | Model | Overall | " + " | ".join(all_dims) + " | Questions | Complete |")
    lines.append("|------|-------|---------|" + "|".join("---" for _ in all_dims) + "|-----------|----------|")

    sorted_reports = sorted(reports, key=lambda r: r["overall_score"], reverse=True)
    for rank, r in enumerate(sorted_reports, 1):
        overall = f"{r['overall_score']:.4f}"
        dim_cells = []
        for d in all_dims:
            score = r.get("dimension_scores", {}).get(d, "-")
            if isinstance(score, (int, float)):
                dim_cells.append(f"{score:.4f}")
            else:
                dim_cells.append(str(score))
        complete = "yes" if r.get("complete") else "no"
        lines.append(f"| {rank} | {r['model']} | {overall} | " + " | ".join(dim_cells) + f" | {r['total_questions']} | {complete} |")

    lines.append("")
    lines.append(f"*Generated from {len(reports)} model results*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize multi-model evaluation reports into a leaderboard.")
    parser.add_argument("--input", default=str(ROOT / "benchmark" / "results"), help="Evaluation report JSON directory")
    parser.add_argument("--output", default="", help="Output path; .json writes structured data, other suffixes write Markdown")
    parser.add_argument("--allow-empty", action="store_true",
                        help="Exit 0 when no merged reports are found; intended only for exploratory dry runs.")
    args = parser.parse_args()

    reports = build_multi_report(args.input, allow_empty=args.allow_empty)
    if not reports:
        return 0

    reports = sorted(reports, key=lambda r: r["overall_score"], reverse=True)

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix.lower() == ".json":
            payload = {
                "models": reports,
                "model_count": len(reports),
                "source_dir": str(Path(args.input).resolve()),
            }
            write_json_atomic(out_path, payload)
        else:
            md = render_leaderboard_table(reports)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(md, encoding="utf-8")
        print(f"Leaderboard saved: {out_path}")
    else:
        md = render_leaderboard_table(reports)
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
