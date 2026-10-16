"""API 8 Judge 评分一致性分析 — Cohen's κ 成对矩阵 + Fleiss' κ 聚合值

读取 8 个 API Judge 的评测报告 JSON（来自 benchmark/eval/reports/），
提取每题 (question_id, score) 构建 8×500 矩阵，计算评分者间信度。

输出:
  - api_8judge_consolidated.json（本目录）
"""
import json
import sys
from pathlib import Path
from collections import Counter

from mineru_core.paths import PROJECT_ROOT as ROOT


REPORTS_DIR = ROOT / "benchmark" / "eval" / "reports"
OUTPUT_DIR = Path(__file__).resolve().parent

API_JUDGE_FILES = [
    "deepseek-v4-flash_report.json",
    "deepseek-v4-pro_report.json",
    "minimax-m2.5-highspeed_report.json",
    "minimax-m2.7-highspeed_report.json",
    "mimo-v2-flash_report.json",
    "mimo-v2-pro_report.json",
    "mimo-v2.5-pro_report.json",
    "mimo-v2.5_report.json",
]


def load_report(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cohens_kappa(a, b, categories):
    n = len(a)
    if n == 0:
        return 0.0

    def discretize(s, cats):
        for i, cat in enumerate(cats):
            if s <= cat:
                return i
        return len(cats)

    da = [discretize(s, categories) for s in a]
    db = [discretize(s, categories) for s in b]

    obs = Counter()
    for va, vb in zip(da, db):
        obs[(va, vb)] += 1

    pa = Counter(da)
    pb = Counter(db)
    n_cat = len(categories) + 1

    po = sum(obs[(k, k)] for k in range(n_cat)) / n
    pe = sum((pa.get(k, 0) / n) * (pb.get(k, 0) / n) for k in range(n_cat))

    if pe >= 1:
        return 1.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings, categories):
    n_subjects = len(ratings[0])
    n_raters = len(ratings)
    n_cat = len(categories) + 1

    def discretize(s, cats):
        for i, cat in enumerate(cats):
            if s <= cat:
                return i
        return len(cats)

    category_counts = []
    for i in range(n_subjects):
        counts = [0] * n_cat
        for rater_scores in ratings:
            cat = discretize(rater_scores[i], categories)
            counts[cat] += 1
        category_counts.append(counts)

    total_assignments = n_subjects * n_raters
    p_j = []
    for j in range(n_cat):
        p_j.append(sum(row[j] for row in category_counts) / total_assignments)

    P_i = []
    for row in category_counts:
        s = sum(row[j] ** 2 for j in range(n_cat))
        p = (s - n_raters) / (n_raters * (n_raters - 1))
        P_i.append(p)

    P_bar = sum(P_i) / n_subjects
    P_e = sum(p_j[j] ** 2 for j in range(n_cat))

    if P_e >= 1:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def _load_all_reports() -> tuple[dict, list, list]:
    """Load all judge reports and extract scores per model."""
    all_scores = {}
    model_names = []
    load_errors = []
    for fname in API_JUDGE_FILES:
        fpath = REPORTS_DIR / fname
        if not fpath.exists():
            load_errors.append(f"{fname} (不存在)")
            continue
        report = load_report(fpath)
        model = report.get("model", fname.replace("_report.json", ""))
        records = report.get("records", [])
        total = report.get("total_questions", 0)
        q_scores = {}
        for r in records:
            qid = r["id"]
            s = r["score"]
            if qid in q_scores:
                q_scores[qid].append(s)
            else:
                q_scores[qid] = [s]
        avg_scores = {qid: sum(v) / len(v) for qid, v in q_scores.items()}
        all_scores[model] = avg_scores
        model_names.append(model)
        mean_s = sum(avg_scores.values()) / len(avg_scores)
        print(f"  [{len(model_names)}] {model:<25s}  n_avg={total:>3d}  "
              f"n_records={len(records):>4d}  n_q={len(avg_scores):>3d}  "
              f"mean={mean_s:.4f}")
    return all_scores, model_names, load_errors


def _build_score_matrix(all_scores, model_names):
    """Build common score matrix across shared question IDs."""
    common_ids = set(all_scores[model_names[0]].keys())
    for m in model_names[1:]:
        common_ids &= set(all_scores[m].keys())
    common_ids = sorted(common_ids)
    score_matrix = {}
    for m in model_names:
        score_matrix[m] = [all_scores[m][qid] for qid in common_ids]
    return score_matrix, common_ids


def _compute_kappa_matrices(score_matrix, model_names, categories):
    """Compute pairwise Cohen's kappa and Fleiss' kappa."""
    n_models = len(model_names)
    kappa_matrix = []
    for i, mi in enumerate(model_names):
        row = []
        for j, mj in enumerate(model_names):
            if i == j:
                row.append(None)
            else:
                k = cohens_kappa(score_matrix[mi], score_matrix[mj], categories)
                row.append(round(k, 4))
        kappa_matrix.append(row)
    ratings_list = [score_matrix[m] for m in model_names]
    fk = fleiss_kappa(ratings_list, categories)
    return kappa_matrix, fk


def _build_model_summaries(all_scores, model_names, score_matrix, n_common):
    """Build per-model full and common summaries."""
    import statistics
    full_summary = {}
    for m in model_names:
        all_vals = list(all_scores[m].values())
        n = len(all_vals)
        sorted_s = sorted(all_vals)
        full_summary[m] = {
            "mean": round(sum(all_vals) / n, 4),
            "std": round(statistics.stdev(all_vals), 4) if n > 1 else 0,
            "median": sorted_s[n // 2],
            "min": round(min(all_vals), 4),
            "max": round(max(all_vals), 4),
            "n_questions": n,
        }
    common_summary = {}
    for m in model_names:
        vals = score_matrix[m]
        common_summary[m] = {"mean": round(sum(vals) / len(vals), 4), "n_common": n_common}
    return full_summary, common_summary


def _print_kappa_table(model_names, n_common, cat_labels, kappa_matrix):
    """Print kappa matrix table."""
    print(f"\n{'=' * 90}")
    print(f"Cohen's κ 成对矩阵 (n={n_common} 共同题目, {len(cat_labels)} 分类)")
    header = f"{'':>24s}"
    for m in model_names:
        header += f"{m[:13]:>14s}" if len(m) > 13 else f"{m:>14s}"
    print(header)
    for i, mi in enumerate(model_names):
        row_str = f"{mi[:24]:>24s}"
        for j, mj in enumerate(model_names):
            if i == j:
                row_str += f"{'—':>14s}"
            else:
                row_str += f"{kappa_matrix[i][j]:>14.4f}"
        print(row_str)


def _print_markdown_table(model_names, full_summary):
    """Print Markdown summary table."""
    print("Markdown 可复制表格 (全量记录均值):\n")
    print(f"| {'Judge':<24s} | {'n':>5s} | {'Mean':>8s} | {'Std':>8s} | {'Median':>8s} | {'Min':>8s} | {'Max':>8s} |")
    print(f"| {'-' * 24} | {'-' * 5} | {'-' * 8} | {'-' * 8} | {'-' * 8} | {'-' * 8} | {'-' * 8} |")
    for m in model_names:
        s = full_summary[m]
        print(f"| {m:<24s} | {s['n_questions']:>5d} | {s['mean']:>8.4f} | "
              f"{s['std']:>8.4f} | {s['median']:>8.4f} | {s['min']:>8.4f} | {s['max']:>8.4f} |")



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_scores, model_names, load_errors = _load_all_reports()
    if load_errors:
        print(f"\n  ⚠️  加载失败: {', '.join(load_errors)}")

    n_models = len(model_names)
    if n_models < 2:
        print(f"\n❌ 需要至少 2 个 Judge，当前 {n_models}")
        sys.exit(1)

    score_matrix, common_ids = _build_score_matrix(all_scores, model_names)
    n_common = len(common_ids)
    print(f"\n共同题目数: {n_common}")

    categories = [0.15, 0.45, 0.80]
    cat_labels = ["0 (0-0.15]", "低 (0.15-0.45]", "中 (0.45-0.80]", "高 (0.80-1.0]"]

    kappa_matrix, fk = _compute_kappa_matrices(score_matrix, model_names, categories)
    full_summary, common_summary = _build_model_summaries(all_scores, model_names, score_matrix, n_common)

    _print_kappa_table(model_names, n_common, cat_labels, kappa_matrix)
    print(f"\nFleiss' κ (8 评者): {fk:.4f}")

    output = {
        "analysis": "API 8 Judge 评分一致性分析",
        "n_models": n_models,
        "models": model_names,
        "score_categories": cat_labels,
        "summary_statistics": full_summary,
        "judge_consistency": {
            "description": "8 Judge × 共同题目评分一致性",
            "n_common_questions": n_common,
            "common_question_summary": common_summary,
            "pairwise_cohens_kappa": {
                model_names[i]: {model_names[j]: kappa_matrix[i][j] for j in range(n_models) if i != j}
                for i in range(n_models)
            },
            "fleiss_kappa": round(fk, 4),
        },
    }

    output_file = OUTPUT_DIR / "api_8judge_consolidated.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ JSON 输出: {output_file}")

    print(f"\n{'=' * 90}")
    print(f"Fleiss' κ (8 评者 × {n_common} 共同题目): {fk:.4f}")
    print(f"\n{'=' * 90}")
    _print_markdown_table(model_names, full_summary)


if __name__ == "__main__":
    main()
