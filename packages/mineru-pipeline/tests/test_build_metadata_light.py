"""Tests for build_metadata_light helper functions."""

from __future__ import annotations

from mineru_pipeline.build_metadata_light import _resolve_meta, _resolve_difficulty


class TestResolveMeta:
    """Cover _resolve_meta: arXiv matched/fallback, textbook matched/fallback."""

    def test_arxiv_matched(self):
        """arXiv ID present in arxiv_lookup → use lookup data."""
        arxiv_lookup = {
            "2103.12345": {
                "title": "PID Control Theory",
                "author": "Author A",
                "source": "arXiv: 2103.12345",
                "copyright": "arXiv Open Access",
                "control_category": "经典控制",
                "year": 2021,
            }
        }
        meta, doc_type, is_arxiv = _resolve_meta(
            "2103.12345_PID_Control",
            txt_by_dirname={},
            txt_fallback={},
            arxiv_lookup=arxiv_lookup,
        )
        assert doc_type == "arxiv"
        assert is_arxiv is True
        assert meta["arxiv_id"] == "2103.12345"
        assert meta["title"] == "PID Control Theory"
        assert meta["author"] == "Author A"
        assert meta["year"] == 2021

    def test_arxiv_fallback(self):
        """arXiv ID NOT in arxiv_lookup → generate fallback metadata."""
        meta, doc_type, is_arxiv = _resolve_meta(
            "2103.99999_UnknownPaper",
            txt_by_dirname={},
            txt_fallback={},
            arxiv_lookup={},
        )
        assert doc_type == "arxiv"
        assert is_arxiv is True
        assert meta["arxiv_id"] == "2103.99999"
        assert meta["author"] == "Unknown / arXiv"
        assert meta["year"] == 2021  # 21 + 2000

    def test_textbook_matched(self):
        """dirname in txt_by_dirname → use source_list metadata."""
        txt_by_dirname = {
            "经典控制": {
                "title": "经典控制理论",
                "author": "某作者",
                "source": "中文教材 / Chinese Textbook",
                "copyright": "Educational",
                "control_category": "经典控制",
            }
        }
        meta, doc_type, is_arxiv = _resolve_meta(
            "经典控制",
            txt_by_dirname=txt_by_dirname,
            txt_fallback={},
            arxiv_lookup={},
        )
        assert doc_type == "textbook"
        assert is_arxiv is False
        assert meta["title"] == "经典控制理论"

    def test_textbook_fallback(self):
        """dirname NOT in txt_by_dirname but IN txt_fallback → use fallback."""
        txt_fallback = {
            "非线性系统_Khalil": {
                "title": "非线性系统（第三版）",
                "author": "Hassan K. Khalil",
                "control_category": "非线性控制",
            }
        }
        meta, doc_type, is_arxiv = _resolve_meta(
            "非线性系统_Khalil",
            txt_by_dirname={},
            txt_fallback=txt_fallback,
            arxiv_lookup={},
        )
        assert doc_type == "textbook"
        assert is_arxiv is False
        assert meta["author"] == "Hassan K. Khalil"

    def test_textbook_generic_fallback(self):
        """dirname in neither txt_by_dirname nor txt_fallback → generic fallback."""
        meta, doc_type, is_arxiv = _resolve_meta(
            "SomeUnknownBook",
            txt_by_dirname={},
            txt_fallback={},
            arxiv_lookup={},
        )
        assert doc_type == "textbook"
        assert is_arxiv is False
        assert meta["author"] == "Unknown / To be verified"
        assert meta["source"] == "中文教材 / Chinese Textbook"

    def test_arxiv_without_underscore(self):
        """Handle arXiv ID as full dirname when no underscore exists."""
        meta, doc_type, is_arxiv = _resolve_meta(
            "2103.12345",
            txt_by_dirname={},
            txt_fallback={},
            arxiv_lookup={},
        )
        assert doc_type == "arxiv"
        assert is_arxiv is True


class TestResolveDifficulty:
    """Cover _resolve_difficulty: single/multi category, unknown, empty, case-insensitive."""

    def test_single_category(self):
        """Single known category → return its difficulty level."""
        d = _resolve_difficulty({"control_category": "经典控制"})
        assert d == "L1"

    def test_multi_category_picks_min(self):
        """Multiple categories → return lowest difficulty."""
        d = _resolve_difficulty({"control_category": "非线性控制, 经典控制"})
        assert d == "L1"  # L1 < L3

    def test_same_level_categories(self):
        """Categories at same level → correct level."""
        d = _resolve_difficulty({"control_category": "经典控制, PID控制"})
        assert d == "L1"

    def test_unknown_category_defaults_l3(self):
        """All categories unknown → default L3."""
        d = _resolve_difficulty({"control_category": "UnknownCategory"})
        assert d == "L3"

    def test_empty_control_category(self):
        """Missing control_category → default '控制理论' which maps to L2."""
        d = _resolve_difficulty({})
        assert d == "L2"

    def test_none_control_category(self):
        """control_category is None → guard returns L3."""
        d = _resolve_difficulty({"control_category": None})
        assert d == "L3"

    def test_case_insensitive_match(self):
        """English category with wrong case → matched via .lower()."""
        d = _resolve_difficulty({"control_category": "CLASSICAL"})
        assert d == "L1"

    def test_mixed_difficulty_picks_min(self):
        """Mixture of L1, L3 categories → pick L1."""
        d = _resolve_difficulty({"control_category": "经典控制, 鲁棒控制, PID控制"})
        assert d == "L1"

    def test_l2_category(self):
        """Category mapped to L2 → correct."""
        d = _resolve_difficulty({"control_category": "现代控制"})
        assert d == "L2"
