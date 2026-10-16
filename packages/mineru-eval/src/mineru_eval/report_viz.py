import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]

from mineru_eval.utils import iter_report_files, load_json


def load_all_results(results_dir):
    input_path = Path(results_dir)
    records_by_model = {}

    if input_path.is_file():
        data = load_json(input_path)
        if "models" in data:
            for item in data.get("models", []):
                records_by_model[item.get("model", item.get("name", "unknown"))] = {
                    "model": item.get("model", item.get("name", "unknown")),
                    "overall_score": item.get("overall_score", 0),
                    "dimension_scores": item.get("dimension_scores", {}),
                    "total_questions": item.get("total_questions", 0),
                }
            return records_by_model
        model = data.get("model", input_path.stem)
        records_by_model[model] = data
        return records_by_model

    json_files = list(iter_report_files(input_path))
    records_by_model = {}
    for jf in json_files:
        try:
            data = load_json(jf)
            model = data.get("model", jf.stem.replace("output_", ""))
            records_by_model[model] = data
        except Exception as e:
            print(f"Warning: skipping {jf.name}: {e}", flush=True)
    return records_by_model


def render_png(records_by_model, output_path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        html_path = Path(output_path).with_suffix(".html")
        print(f"matplotlib unavailable ({exc}); writing HTML fallback: {html_path}", flush=True)
        render_radar_html(records_by_model, html_path)
        return

    dimensions = ["A", "B", "C", "D"]
    model_scores = []
    for model, data in records_by_model.items():
        dim_scores = data.get("dimension_scores", {})
        overall = data.get("overall_score", 0)
        model_scores.append((model, overall, dim_scores))
    model_scores.sort(key=lambda x: x[1], reverse=True)

    labels = [item[0] for item in model_scores]
    overall_scores = [item[1] for item in model_scores]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(labels) * 0.45)))

    axes[0].barh(labels[::-1], overall_scores[::-1], color="#2f6f8f")
    axes[0].set_xlim(0, 1)
    axes[0].set_title("Overall Score")
    axes[0].set_xlabel("Score")

    heatmap = []
    for _, _, dim_scores in model_scores:
        heatmap.append([dim_scores.get(dim, 0) for dim in dimensions])
    im = axes[1].imshow(heatmap, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axes[1].set_xticks(range(len(dimensions)), dimensions)
    axes[1].set_yticks(range(len(labels)), labels)
    axes[1].set_title("Dimension Scores")
    for y, row in enumerate(heatmap):
        for x, value in enumerate(row):
            axes[1].text(x, y, f"{value:.3f}", ha="center", va="center", color="white" if value < 0.55 else "black", fontsize=8)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("ControlSci Leaderboard", fontsize=14)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart generated: {output_path}")


def render_radar_html(records_by_model, output_path):
    """生成四维评分雷达图的 HTML（使用纯 CSS/SVG 或简化的表格代替）。"""
    models = list(records_by_model.keys())
    dimensions = ["A", "B", "C", "D"] if records_by_model else []

    lines = []
    lines.append("""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>ControlSci Leaderboard</title>
<style>
body { font-family: sans-serif; margin: 20px; }
table { border-collapse: collapse; margin: 20px 0; width: 100%; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
th { background-color: #4CAF50; color: white; }
tr:nth-child(even) { background-color: #f2f2f2; }
.radar { margin: 20px 0; }
.bar-container { display: flex; align-items: center; margin: 4px 0; }
.bar-label { width: 120px; font-size: 12px; }
.bar { height: 18px; background: #4CAF50; border-radius: 3px; min-width: 2px; }
.bar-value { margin-left: 8px; font-size: 12px; }
</style>
</head>
<body>
<h1>ControlSci Benchmark Leaderboard</h1>""")

    # Leaderboard table
    lines.append("<h2>Overall Scores</h2>")
    lines.append("<table><tr><th>Rank</th><th>Model</th>")
    for d in dimensions:
        lines.append(f"<th>Dim {d}</th>")
    lines.append("<th>Overall</th><th>Questions</th></tr>")

    model_scores = []
    for model, data in records_by_model.items():
        dim_scores = data.get("dimension_scores", {})
        overall = data.get("overall_score", 0)
        total = data.get("total_questions", 0)
        model_scores.append((model, overall, dim_scores, total))

    model_scores.sort(key=lambda x: x[1], reverse=True)

    for rank, (model, overall, dim_scores, total) in enumerate(model_scores, 1):
        lines.append(f"<tr><td>{rank}</td><td><strong>{model}</strong></td>")
        for d in dimensions:
            score = dim_scores.get(d, "-")
            if isinstance(score, (int, float)):
                lines.append(f"<td>{score:.4f}</td>")
            else:
                lines.append(f"<td>{score}</td>")
        lines.append(f"<td><strong>{overall:.4f}</strong></td><td>{total}</td></tr>")
    lines.append("</table>")

    # Bar chart for each model
    lines.append("<h2>Dimension Scores by Model</h2>")
    for model, data in records_by_model.items():
        dim_scores = data.get("dimension_scores", {})
        overall = data.get("overall_score", 0)
        lines.append(f"<h3>{model} (Overall: {overall:.4f})</h3>")
        for d in dimensions:
            score = dim_scores.get(d, 0)
            pct = score * 100 if isinstance(score, (int, float)) else 0
            color = f"hsl({120 * score:.0f}, 70%, 45%)" if isinstance(score, (int, float)) else "#ccc"
            lines.append(f"""<div class="bar-container"><span class="bar-label">Dim {d}</span>
<div class="bar" style="width:{max(pct, 2)}%;background:{color};"></div>
<span class="bar-value">{score if isinstance(score, (int, float)) else '-'}</span></div>""")

    # dimension comparison heatmap
    lines.append("<h2>Score Heatmap (Model × Dimension)</h2>")
    lines.append("<table><tr><th>Model</th>")
    for d in dimensions:
        lines.append(f"<th>Dim {d}</th>")
    lines.append("</tr>")
    for model, _, dim_scores, _ in model_scores:
        lines.append(f"<tr><td>{model}</td>")
        for d in dimensions:
            score = dim_scores.get(d, 0)
            if isinstance(score, (int, float)):
                intensity = int(255 * (1 - score))
                r = intensity
                g = 255 - intensity
                lines.append(f'<td style="background:rgb({255 - g},{g},100);color:{"white" if score < 0.4 else "black"}">{score:.4f}</td>')
            else:
                lines.append("<td>-</td>")
        lines.append("</tr>")
    lines.append("</table>")

    lines.append("<p><em>Generated by ControlSci Benchmark Report</em></p>")
    lines.append("</body></html>")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation result visualizations.")
    parser.add_argument("--input", default=str(ROOT / "benchmark" / "results"), help="Report directory, single model report, or leaderboard.json")
    parser.add_argument("--output", default=str(ROOT / "benchmark" / "results" / "leaderboard.html"), help="Output path; supports .html or .png")
    parser.add_argument("--allow-empty", action="store_true", help="Exit 0 when no evaluation reports are found.")
    args = parser.parse_args()

    records = load_all_results(args.input)
    if not records:
        print(f"No evaluation reports found in {args.input}")
        raise SystemExit(0 if args.allow_empty else 1)

    print(f"Loaded {len(records)} model result(s): {', '.join(records.keys())}")
    if Path(args.output).suffix.lower() == ".png":
        render_png(records, args.output)
    else:
        render_radar_html(records, args.output)


if __name__ == "__main__":
    main()
