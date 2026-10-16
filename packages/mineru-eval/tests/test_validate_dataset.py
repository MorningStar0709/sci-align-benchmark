"""Regression tests for validate_dataset — extracted helpers and validation functions."""

from __future__ import annotations

from collections import Counter

from mineru_eval.validate_dataset import (
    validate_categories,
    validate_cross_references,
    validate_meta,
    validate_question,
    validate_rubric,
)


class TestValidateMeta:
    def test_valid_meta(self):
        meta = {"project": "test", "version": "1.0", "updated": "2024-01-01", "total_questions": 500, "dimensions": ["A", "B", "C", "D"]}
        assert validate_meta(meta) == []

    def test_missing_keys(self):
        errors = validate_meta({})
        assert len(errors) == 5  # all 5 required keys missing

    def test_total_questions_not_int(self):
        meta = {"project": "t", "version": "1", "updated": "d", "total_questions": "500", "dimensions": []}
        errors = validate_meta(meta)
        assert any("not int" in e for e in errors)

    def test_total_questions_wrong_count(self):
        meta = {"project": "t", "version": "1", "updated": "d", "total_questions": 100, "dimensions": []}
        errors = validate_meta(meta)
        assert any("!= 500" in e for e in errors)


class TestValidateCategories:
    def test_valid_single(self):
        errors = []
        validate_categories(errors, 0, "CS-EVO-00001", ["classical"])
        assert errors == []

    def test_valid_multiple(self):
        errors = []
        validate_categories(errors, 0, "CS-EVO-00001", ["classical", "modern"])
        assert errors == []

    def test_not_a_list(self):
        errors = []
        validate_categories(errors, 0, "CS-EVO-00001", "classical")
        assert any("must be array" in e for e in errors)

    def test_too_many(self):
        errors = []
        validate_categories(errors, 0, "CS-EVO-00001", ["a", "b", "c", "d"])
        assert any("must contain 1-3" in e for e in errors)

    def test_duplicates(self):
        errors = []
        validate_categories(errors, 0, "CS-EVO-00001", ["classical", "classical"])
        assert any("duplicates" in e for e in errors)

    def test_unknown_category(self):
        errors = []
        validate_categories(errors, 0, "CS-EVO-00001", ["unknown_cat"])
        assert any("unknown" in e for e in errors)


class TestValidateRubric:
    def test_valid_rubric(self):
        errors = []
        rubric = {
            "feasibility": {"max_score": 5, "weight": 1.0, "description": "desc"},
            "method_choice": {"max_score": 5, "weight": 1.0, "description": "desc"},
            "completeness": {"max_score": 5, "weight": 1.0, "description": "desc"},
            "innovation": {"max_score": 5, "weight": 1.0, "description": "desc"},
            "clarity": {"max_score": 5, "weight": 1.0, "description": "desc"},
        }
        validate_rubric(errors, 0, "CS-EVO-00001", rubric)
        assert errors == []

    def test_not_object(self):
        errors = []
        validate_rubric(errors, 0, "CS-EVO-00001", "not a dict")
        assert any("must be object" in e for e in errors)

    def test_missing_keys(self):
        errors = []
        validate_rubric(errors, 0, "CS-EVO-00001", {})
        assert len(errors) == 1  # consolidated error listing all missing keys

    def test_missing_sub_keys(self):
        errors = []
        validate_rubric(errors, 0, "CS-EVO-00001", {"feasibility": {}})
        assert any("max_score" in e for e in errors)


class TestValidateQuestion:
    def test_valid_basic_question(self):
        q = {
            "id": "CS-EVO-00001",
            "dimension": "A",
            "difficulty_level": "L1",
            "control_category": ["classical"],
            "question": "What is PID?",
            "answer": "Proportional-Integral-Derivative",
            "reasoning_steps": ["Identify"],
            "source_ref": "book",
            "consistency_status": "auto_passed",
        }
        errors = validate_question(q, 0, [], Counter(), Counter(), Counter(), Counter())
        assert errors == []

    def test_missing_required_fields(self):
        errors = validate_question({}, 0, [], Counter(), Counter(), Counter(), Counter())
        # Errors for each missing field + ID format check on empty value
        assert len(errors) > 9

    def test_invalid_id(self):
        q = {
            "id": "BAD-ID",
            "dimension": "A",
            "difficulty_level": "L1",
            "control_category": ["classical"],
            "question": "q",
            "answer": "a",
            "reasoning_steps": [],
            "source_ref": "s",
            "consistency_status": "auto_passed",
        }
        errors = validate_question(q, 0, [], Counter(), Counter(), Counter(), Counter())
        assert any("CS-EVO-NNNNN" in e for e in errors)

    def test_invalid_dimension(self):
        q = {
            "id": "CS-EVO-00001",
            "dimension": "X",
            "difficulty_level": "L1",
            "control_category": ["classical"],
            "question": "q",
            "answer": "a",
            "reasoning_steps": [],
            "source_ref": "s",
            "consistency_status": "auto_passed",
        }
        errors = validate_question(q, 0, [], Counter(), Counter(), Counter(), Counter())
        assert any("illegal" in e.lower() for e in errors)

    def test_invalid_difficulty(self):
        q = {
            "id": "CS-EVO-00001",
            "dimension": "A",
            "difficulty_level": "L5",
            "control_category": ["classical"],
            "question": "q",
            "answer": "a",
            "reasoning_steps": [],
            "source_ref": "s",
            "consistency_status": "auto_passed",
        }
        errors = validate_question(q, 0, [], Counter(), Counter(), Counter(), Counter())
        assert any("illegal" in e.lower() for e in errors)

    def test_d_dimension_rubric(self):
        q = {
            "id": "CS-EVO-00001",
            "dimension": "D",
            "difficulty_level": "L3",
            "control_category": ["classical"],
            "question": "q",
            "answer": "a",
            "reasoning_steps": [],
            "source_ref": "s",
            "consistency_status": "auto_passed",
            "rubric": {
                "feasibility": {"max_score": 5, "weight": 1.0, "description": "d"},
                "method_choice": {"max_score": 5, "weight": 1.0, "description": "d"},
                "completeness": {"max_score": 5, "weight": 1.0, "description": "d"},
                "innovation": {"max_score": 5, "weight": 1.0, "description": "d"},
                "clarity": {"max_score": 5, "weight": 1.0, "description": "d"},
            },
        }
        errors = validate_question(q, 0, [], Counter(), Counter(), Counter(), Counter())
        assert errors == []

    def test_c_dimension_sensitivity(self):
        q = {
            "id": "CS-EVO-00001",
            "dimension": "C",
            "difficulty_level": "L2",
            "control_category": ["classical"],
            "question": "q",
            "answer": "a",
            "reasoning_steps": [],
            "source_ref": "s",
            "consistency_status": "auto_passed",
            "sensitivity_dimension": "parameter",
            "sibling_id": "CS-EVO-00002",
        }
        errors = validate_question(q, 0, [], Counter(), Counter(), Counter(), Counter())
        assert errors == []


class TestValidateCrossReferences:
    def test_no_c_dim_questions(self):
        questions = [
            {"id": "CS-EVO-00001", "dimension": "A"},
            {"id": "CS-EVO-00002", "dimension": "B"},
        ]
        errors = []
        validate_cross_references(questions, errors)
        assert errors == []

    def test_missing_sibling(self):
        questions = [
            {"id": "CS-EVO-00001", "dimension": "C", "sibling_id": "CS-EVO-99999"},
        ]
        errors = []
        validate_cross_references(questions, errors)
        assert any("missing question" in e for e in errors)

    def test_bidirectional_sibling(self):
        questions = [
            {"id": "CS-EVO-00001", "dimension": "C", "sibling_id": "CS-EVO-00002"},
            {"id": "CS-EVO-00002", "dimension": "C", "sibling_id": "CS-EVO-00001"},
        ]
        errors = []
        validate_cross_references(questions, errors)
        assert errors == []
