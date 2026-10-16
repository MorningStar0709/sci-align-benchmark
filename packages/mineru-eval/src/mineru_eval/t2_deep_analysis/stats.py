"""
T2: 9-Model Deep Analysis — statistical inference.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
import scikit_posthocs as sp

from .core import CORE_9, MODEL_SHORT, DIMENSIONS, DIFFICULTIES


def kruskal_wallis_test(all_scores: dict, label: str = "Overall"):
    """Kruskal-Wallis H test across all models."""
    groups = [all_scores[m] for m in CORE_9 if m in all_scores]
    if len(groups) < 3:
        return None
    H, p = stats.kruskal(*groups)
    return {"test": "Kruskal-Wallis", "label": label, "H": H, "p": p,
            "df": len(groups) - 1, "n_models": len(groups), "n_total": sum(len(g) for g in groups)}


def dunn_posthoc_test(all_scores: dict):
    """Dunn's post-hoc with Bonferroni correction."""
    models = [m for m in CORE_9 if m in all_scores]
    groups = [all_scores[m] for m in models]
    if len(groups) < 3:
        return None
    df = sp.posthoc_dunn(groups, p_adjust="bonferroni")
    df.index = [MODEL_SHORT.get(m, m) for m in models]
    df.columns = [MODEL_SHORT.get(m, m) for m in models]
    return df


def cohens_d_matrix(all_scores: dict):
    """Pairwise Cohen's d effect sizes."""
    models = sorted(all_scores.keys(), key=lambda m: all_scores[m].mean(), reverse=True)
    n = len(models)
    d_mat = np.zeros((n, n))
    names = [MODEL_SHORT.get(m, m) for m in models]

    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            if i == j:
                d_mat[i, j] = 0.0
                continue
            s1, s2 = all_scores[mi], all_scores[mj]
            n1, n2 = len(s1), len(s2)
            mean_diff = s1.mean() - s2.mean()
            pooled_std = np.sqrt(((n1 - 1) * s1.var() + (n2 - 1) * s2.var()) / (n1 + n2 - 2))
            d_mat[i, j] = mean_diff / pooled_std if pooled_std > 1e-12 else 0.0

    return d_mat, names, models


def spearman_correlation(all_scores: dict):
    """Spearman rank correlation matrix of per-question scores."""
    models = sorted(all_scores.keys(), key=lambda m: all_scores[m].mean(), reverse=True)
    n = len(models)
    corr = np.zeros((n, n))
    names = [MODEL_SHORT.get(m, m) for m in models]

    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            if i == j:
                corr[i, j] = 1.0
            elif i < j:
                r, _ = stats.spearmanr(all_scores[mi], all_scores[mj])
                corr[i, j] = corr[j, i] = r

    return corr, names


def per_dimension_kruskal(by_dim: dict):
    """K-W test per dimension."""
    results = {}
    for dim in DIMENSIONS:
        dim_groups = {}
        for m in CORE_9:
            if m in by_dim and by_dim[m].get(dim):
                dim_groups[m] = np.array(by_dim[m][dim])
        if len(dim_groups) >= 3:
            groups = list(dim_groups.values())
            H, p = stats.kruskal(*groups)
            means = {m: s.mean() for m, s in dim_groups.items()}
            best = max(means, key=means.get)
            worst = min(means, key=means.get)
            results[dim] = {
                "H": H, "p": p, "n_models": len(groups),
                "best_model": MODEL_SHORT.get(best, best),
                "best_mean": means[best],
                "worst_model": MODEL_SHORT.get(worst, worst),
                "worst_mean": means[worst],
                "spread": means[best] - means[worst],
            }
    return results


def per_difficulty_kruskal(by_diff: dict):
    """K-W test per difficulty level."""
    results = {}
    for lvl in DIFFICULTIES:
        lvl_groups = {}
        for m in CORE_9:
            if m in by_diff and by_diff[m].get(lvl):
                lvl_groups[m] = np.array(by_diff[m][lvl])
        if len(lvl_groups) >= 3:
            groups = list(lvl_groups.values())
            H, p = stats.kruskal(*groups)
            means = {m: s.mean() for m, s in lvl_groups.items()}
            results[lvl] = {
                "H": H, "p": p,
                "best_model": MODEL_SHORT.get(max(means, key=means.get), ""),
                "worst_model": MODEL_SHORT.get(min(means, key=means.get), ""),
                "spread": max(means.values()) - min(means.values()),
            }
    return results
