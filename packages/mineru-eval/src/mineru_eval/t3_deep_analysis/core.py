"""T3 Deep Analysis — core constants and data loading utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from mineru_eval.utils import load_json, iter_report_files
from mineru_eval._constants import (
    CORE_9, _MODEL_ALIASES, MODEL_SHORT, MODEL_FAMILY,
    FAMILY_COLORS, DIMENSIONS, DIFFICULTIES,
)

ROOT = Path(__file__).resolve().parents[5]

RNG = np.random.RandomState(42)

OUTPUT_DIR = ROOT / "benchmark" / "eval" / "results" / "analysis"


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_core_9(reports_dir=None):
    if reports_dir is None:
        reports_dir = ROOT / "benchmark" / "eval" / "reports"
    reports_dir = Path(reports_dir)
    data = {}
    for jf in iter_report_files(reports_dir):
        try:
            d = load_json(jf)
            model_raw = d.get("model", "")
            model = _MODEL_ALIASES.get(model_raw, model_raw)
            if model in CORE_9 and d.get("total_questions", 0) >= 400:
                data[model] = d
        except Exception:
            pass
    missing = set(CORE_9) - set(data.keys())
    if missing:
        print(f"  缺失模型报告: {missing}")
    return data


def extract_records(data):
    all_scores = {}
    all_lengths = {}
    all_records = {}
    by_dim = {m: {d: [] for d in DIMENSIONS} for m in data}
    by_diff = {m: {lvl: [] for lvl in DIFFICULTIES} for m in data}
    length_by_dim = {m: {d: [] for d in DIMENSIONS} for m in data}
    dimension_means = {}

    for model, d in data.items():
        scores = []
        lengths = []
        recs_raw = []
        for rec in d.get("records", []):
            dim = rec.get("dimension", "")
            lvl = rec.get("difficulty_level", "")
            sc = rec.get("score", 0.0)
            ma = rec.get("model_answer", "")
            scores.append(sc)
            lengths.append(len(ma))
            recs_raw.append(rec)
            if dim in by_dim[model]:
                by_dim[model][dim].append(sc)
            if lvl in by_diff[model]:
                by_diff[model][lvl].append(sc)
            if dim in length_by_dim[model]:
                length_by_dim[model][dim].append(len(ma))

        all_scores[model] = np.array(scores)
        all_lengths[model] = np.array(lengths)
        all_records[model] = recs_raw
        dimension_means[model] = {
            k: np.mean(v) if v else 0.0 for k, v in by_dim[model].items()
        }

    return all_scores, all_lengths, all_records, by_dim, by_diff, length_by_dim, dimension_means


def _save_fig(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {path}")
    return path


def _sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    if p < 0.10: return "†"
    return "ns"
