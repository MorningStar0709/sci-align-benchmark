"""
T2: 9-Model Deep Analysis — core constants & data loading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[5]

from mineru_eval.utils import load_json, iter_report_files
from mineru_eval._constants import (
    CORE_9, _MODEL_ALIASES, MODEL_SHORT, MODEL_FAMILY,
    FAMILY_COLORS, DIMENSIONS, DIFFICULTIES,
)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = ROOT / "benchmark" / "eval" / "results" / "analysis"

RNG = np.random.RandomState(42)


# ── Data loading ──────────────────────────────────────────────────

def load_core_9(reports_dir: Path | str | None = None):
    """Load the 9 core-model full-evaluation reports."""
    if reports_dir is None:
        reports_dir = ROOT / "benchmark" / "eval" / "reports"

    reports_dir = Path(reports_dir)
    if not reports_dir.is_dir():
        raise FileNotFoundError(f"Reports directory not found: {reports_dir}")

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
        print(f"⚠ 缺失模型报告: {missing}")

    return data


def extract_records(data: dict):
    """data[model] = full report → arrays of scores by dimension / difficulty."""
    by_dim = {m: {d: [] for d in DIMENSIONS} for m in data}
    by_diff = {m: {lvl: [] for lvl in DIFFICULTIES} for m in data}
    all_scores = {}
    dimension_means = {}

    for model, d in data.items():
        scores = []
        for rec in d.get("records", []):
            dim = rec.get("dimension", "")
            lvl = rec.get("difficulty_level", "")
            sc = rec.get("score", 0)
            scores.append(sc)
            if dim in by_dim[model]:
                by_dim[model][dim].append(sc)
            if lvl in by_diff[model]:
                by_diff[model][lvl].append(sc)
        all_scores[model] = np.array(scores)
        dimension_means[model] = {k: np.mean(v) if v else 0.0 for k, v in by_dim[model].items()}

    return all_scores, by_dim, by_diff, dimension_means


# ── Helpers ───────────────────────────────────────────────────────

def _save_fig(fig, name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {path}")
    return path


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "†"
    return "ns"


def _d_interpret(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def _model_rank_desc(rank: int, name: str, overall: float, d_mat, names_all, models_all):
    """Rich model description line."""
    idx = models_all.index(name)
    strengths = d_mat[idx] > 0.3
    weakness = d_mat[idx] < -0.3
    n_beats = int(np.sum(strengths))
    n_beaten = int(np.sum(weakness))
    return f"  beats {n_beats}/8 with d>0.3, loses to {n_beaten}/8"
