"""Tests for evaluate.py helper functions."""

from __future__ import annotations

from mineru_eval.evaluate import _score_or_fallback


def _fake_score_ok(*args, **kwargs):
    """Stub score_fn that always returns a perfect score."""
    return {"score": 1.0, "reason": "OK", "issues": []}


def _fake_score_failure(*args, **kwargs):
    """Stub score_fn that simulates a judge failure."""
    return {"score": 0.0, "reason": "Parsing error", "issues": ["judge failure"]}


class TestScoreOrFallback:
    """Cover _score_or_fallback: success/failure/exception branches."""

    BASE_KWARGS = dict(
        question_text="What is PID control?",
        expected="Proportional-Integral-Derivative",
        dimension="A",
        q={},
        judge_client=None,
        judge_model="gpt-4",
        retries=2,
    )

    def test_model_answer_success_resets_failures(self):
        """model_answer provided, scoring succeeds with no issues → reset to 0."""
        result, cf = _score_or_fallback(
            model_answer="PID is...",
            consecutive_failures=3,
            score_fn=_fake_score_ok,
            **self.BASE_KWARGS,
        )
        assert result["score"] == 1.0
        assert cf == 0  # reset

    def test_model_answer_failure_increments_failures(self):
        """model_answer provided, scoring result has 'failure' in issues → inc."""
        result, cf = _score_or_fallback(
            model_answer="PID is...",
            consecutive_failures=2,
            score_fn=_fake_score_failure,
            **self.BASE_KWARGS,
        )
        assert cf == 3  # 2 + 1

    def test_no_model_answer_returns_fallback(self):
        """model_answer is None → fallback score, failures incremented."""
        result, cf = _score_or_fallback(
            model_answer=None,
            consecutive_failures=1,
            **self.BASE_KWARGS,
        )
        assert result["score"] == 0.0
        assert result["reason"] == "Target model failed after retries"
        assert cf == 2  # 1 + 1 (fallback issues also contain "failure")

    def test_empty_model_answer_returns_fallback(self):
        """model_answer is empty string → fallback score."""
        result, cf = _score_or_fallback(
            model_answer="",
            consecutive_failures=0,
            **self.BASE_KWARGS,
        )
        assert result["score"] == 0.0
        assert cf == 1  # 0 + 1 (fallback issues contain "failure")

    def test_exception_during_scoring_increments_failures(self):
        """score_fn raises → error score, failures incremented."""
        def _raising_score(*args, **kwargs):
            raise ValueError("API timeout")

        result, cf = _score_or_fallback(
            model_answer="PID is...",
            consecutive_failures=0,
            score_fn=_raising_score,
            **self.BASE_KWARGS,
        )
        assert result["score"] == 0.0
        assert "Scoring error" in result["reason"]
        assert cf == 1  # incremented
