"""Tests for mineru_core.constants."""

from __future__ import annotations

from mineru_core.constants import (
    CORE_9,
    FAMILY_COLORS,
    MODEL_FAMILY,
    MODEL_SHORT,
    STANDARD_CATEGORIES,
    VERSION,
)


class TestConstants:
    def test_version(self):
        assert VERSION == "0.1.0"

    def test_core_9_count(self):
        assert len(CORE_9) == 9

    def test_model_short_coverage(self):
        assert all(m in MODEL_SHORT for m in CORE_9)

    def test_model_family_coverage(self):
        assert all(m in MODEL_FAMILY for m in CORE_9)

    def test_family_colors(self):
        families = set(MODEL_FAMILY.values())
        for family in families:
            assert family in FAMILY_COLORS

    def test_standard_categories(self):
        assert len(STANDARD_CATEGORIES) == 14
        assert "PID控制" in STANDARD_CATEGORIES
