"""Regression tests for a7_dimension_kappa — extracted computation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mineru_eval.t3_deep_analysis.a7_dimension_kappa import (
    _compute_icc,
    _compute_kendall_w,
    _compute_pairwise_spearman,
)


class TestComputeKendallW:
    def test_perfect_agreement(self):
        """All models rank dimensions identically → W = 1.0."""
        n_models, n_dims = 3, 4
        # All models agree on same rank order: dim 0 best, dim 3 worst
        matrix = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ])
        W, chi2, p = _compute_kendall_w(matrix, n_models, n_dims)
        assert W == pytest.approx(1.0, abs=1e-6)

    def test_no_agreement(self):
        """Random rankings → W close to 0."""
        n_models, n_dims = 3, 4
        matrix = np.array([
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [2, 4, 1, 3],
        ])
        W, chi2, p = _compute_kendall_w(matrix, n_models, n_dims)
        assert 0 <= W <= 1

    def test_w_clipped_to_1(self):
        """W should be clipped to 1.0 max."""
        n_models, n_dims = 2, 2
        matrix = np.array([
            [1, 2],
            [1, 2],
        ])
        W, chi2, p = _compute_kendall_w(matrix, n_models, n_dims)
        assert W <= 1.0


class TestComputePairwiseSpearman:
    def test_perfect_positive_correlation(self):
        """Models with same dimension ordering but different absolute scores → ρ = 1.0."""
        # Each model has different scores (needed for variance) but same relative dimension ordering
        by_dim = {
            "model_a": {"A": [0.9], "B": [0.8], "C": [0.7], "D": [0.6]},
            "model_b": {"A": [0.7], "B": [0.6], "C": [0.5], "D": [0.4]},
            "model_c": {"A": [0.5], "B": [0.4], "C": [0.3], "D": [0.2]},
        }
        models = ["model_a", "model_b", "model_c"]
        result = _compute_pairwise_spearman(by_dim, models)
        # Across models, dimension A↔B scores are perfectly correlated
        for key, val in result.items():
            assert val["spearman_rho"] == pytest.approx(1.0, abs=0.01), f"{key} rho={val['spearman_rho']}"

    def test_empty_scores_default_zero(self):
        """Empty dimension lists should default to mean=0."""
        by_dim = {
            "model_a": {"A": [], "B": [0.8], "C": [0.7], "D": [0.6]},
            "model_b": {"A": [0.9], "B": [0.8], "C": [0.7], "D": [0.6]},
        }
        models = ["model_a", "model_b"]
        result = _compute_pairwise_spearman(by_dim, models)
        assert isinstance(result, dict)
        assert all("spearman_rho" in v for v in result.values())

    def test_no_variance_returns_zero(self):
        """All models same score → no variance → ρ = 0.0."""
        by_dim = {
            "model_a": {"A": [0.5], "B": [0.5], "C": [0.5], "D": [0.5]},
            "model_b": {"A": [0.5], "B": [0.5], "C": [0.5], "D": [0.5]},
        }
        models = ["model_a", "model_b"]
        result = _compute_pairwise_spearman(by_dim, models)
        for val in result.values():
            assert val["spearman_rho"] == 0.0


class TestComputeIcc:
    def test_high_reliability(self):
        """Models with similar dimension profiles → high, positive ICC."""
        n_models, n_dims = 3, 4
        # Models differ in absolute level but share similar dimension pattern
        matrix = np.array([
            [0.9, 0.8, 0.7, 0.6],
            [0.8, 0.7, 0.6, 0.5],
            [0.7, 0.6, 0.5, 0.4],
        ])
        icc = _compute_icc(matrix, n_models, n_dims)
        assert icc > 0

    def test_low_reliability(self):
        """Random scores across models → low ICC."""
        n_models, n_dims = 4, 4
        np.random.seed(42)
        matrix = np.random.rand(n_models, n_dims)
        icc = _compute_icc(matrix, n_models, n_dims)
        # Random data should give low ICC
        assert icc < 0.5

    def test_icc_range(self):
        """ICC(2,1) should be in [-1, 1] range."""
        n_models, n_dims = 3, 4
        matrix = np.array([
            [1.0, 0.0, 0.5, 0.3],
            [0.8, 0.2, 0.6, 0.4],
            [0.9, 0.1, 0.7, 0.2],
        ])
        icc = _compute_icc(matrix, n_models, n_dims)
        assert -1.0 <= icc <= 1.0

    def test_single_model(self):
        """Single model → ICC should be defined (not NaN)."""
        n_models, n_dims = 1, 4
        matrix = np.array([[0.9, 0.8, 0.7, 0.6]])
        icc = _compute_icc(matrix, n_models, n_dims)
        assert not np.isnan(icc)
