"""Unit tests for GRPO reward functions.

Tests cover: accuracy_reward, format_reward, reasoning_depth_reward,
and the get_reward_funcs aggregator.
"""

from __future__ import annotations

import pytest

from mineru_eval.finetune.rewards import (
    accuracy_reward,
    accuracy_reward_c,
    constraint_reward,
    format_reward,
    get_reward_funcs,
    get_reward_funcs_c_v2,
    reasoning_depth_reward,
    _extract_content,
    _extract_final_answer,
    _extract_think_content,
    _tokenize_answer,
    _numerical_similarity,
    _count_reasoning_steps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completion(content: str) -> list[list[dict[str, str]]]:
    """Wrap a single content string into GRPO completion format."""
    return [[{"role": "assistant", "content": content}]]


def _make_completions(*contents: str) -> list[list[dict[str, str]]]:
    """Build a list of completions from multiple content strings."""
    return [[{"role": "assistant", "content": c}] for c in contents]


# ---------------------------------------------------------------------------
# _extract_final_answer
# ---------------------------------------------------------------------------


class TestExtractFinalAnswer:
    def test_with_think_tag(self):
        content = "<think>\nreasoning\n</think>\nThe answer is 42."
        assert _extract_final_answer(content) == "The answer is 42."

    def test_without_think_tag(self):
        content = "Just the answer."
        assert _extract_final_answer(content) == "Just the answer."

    def test_empty_after_think(self):
        content = "<think>\nreasoning\n</think>"
        assert _extract_final_answer(content) == ""

    def test_multiline_answer(self):
        content = "<think>\nthink\n</think>\nLine 1\nLine 2"
        assert "Line 1" in _extract_final_answer(content)
        assert "Line 2" in _extract_final_answer(content)


# ---------------------------------------------------------------------------
# _extract_think_content
# ---------------------------------------------------------------------------


class TestExtractThinkContent:
    def test_normal(self):
        content = "<think>\nstep 1\nstep 2\n</think>\nanswer"
        assert "step 1" in _extract_think_content(content)
        assert "step 2" in _extract_think_content(content)

    def test_no_tags(self):
        assert _extract_think_content("no tags here") == ""

    def test_nested_think_extracts_first(self):
        # New regex extracts first think block (non-greedy)
        content = "<think>outer<think>inner</think>more</think>"
        result = _extract_think_content(content)
        # Extracts everything up to the first </think>
        assert "outer" in result
        assert "inner" in result

    def test_multiple_think_blocks_extracts_first(self):
        content = "<think>\nfirst\n</think>\nanswer\n<think>\nsecond\n</think>"
        result = _extract_think_content(content)
        assert "first" in result
        assert "second" not in result

    def test_empty_think(self):
        content = "<think></think>\nanswer"
        assert _extract_think_content(content) == ""


# ---------------------------------------------------------------------------
# _tokenize_answer
# ---------------------------------------------------------------------------


class TestTokenizeAnswer:
    def test_chinese_and_formula(self):
        tokens = _tokenize_answer("传递函数 G(s) = 1/(s+1)")
        assert "传递函数" in tokens

    def test_strips_latex(self):
        tokens = _tokenize_answer("\\frac{12}{34} + \\alpha")
        # LaTeX commands are stripped
        assert "\\frac" not in tokens and "\\alpha" not in tokens
        # Multi-char numbers remain (single-char filtered by len >= 2)
        assert "12" in tokens or "34" in tokens

    def test_empty(self):
        tokens = _tokenize_answer("")
        assert tokens == set()


# ---------------------------------------------------------------------------
# _numerical_similarity
# ---------------------------------------------------------------------------


class TestNumericalSimilarity:
    def test_exact_match(self):
        assert _numerical_similarity("w2 = 0.629", "w2 = 0.629") == 1.0

    def test_within_tolerance(self):
        assert _numerical_similarity("result = 1.04", "result = 1.05") == 1.0

    def test_outside_tolerance(self):
        assert _numerical_similarity("result = 2.0", "result = 1.0") == 0.0

    def test_no_numbers(self):
        assert _numerical_similarity("hello", "world") is None

    def test_mismatched_count(self):
        assert _numerical_similarity("a=1 b=2", "a=1") == 0.0

    def test_order_matters(self):
        # Same numbers but assigned to different variables — should fail
        assert _numerical_similarity("Kp=1, Ki=2", "Kp=2, Ki=1") == 0.0


# ---------------------------------------------------------------------------
# _count_reasoning_steps
# ---------------------------------------------------------------------------


class TestCountReasoningSteps:
    def test_numbered_steps(self):
        text = "1. First step。2. Second step。3. Third step。"
        assert _count_reasoning_steps(text) >= 3

    def test_sentence_based(self):
        text = "首先分析系统稳定性。然后计算传递函数。最后验证收敛条件。"
        assert _count_reasoning_steps(text) >= 3

    def test_empty(self):
        assert _count_reasoning_steps("") == 0

    def test_short_fragment(self):
        assert _count_reasoning_steps("ok") == 0


# ---------------------------------------------------------------------------
# format_reward
# ---------------------------------------------------------------------------


class TestFormatReward:
    def test_perfect_format(self):
        content = "<think>\nThis is a detailed reasoning process with enough content to pass the threshold.\n</think>\nFinal answer."
        rewards = format_reward(_make_completions(content))
        assert rewards == [1.0]

    def test_missing_tags(self):
        content = "No think tags here, just a plain answer."
        rewards = format_reward(_make_completions(content))
        assert rewards == [0.0]

    def test_empty_think(self):
        content = "<think></think>\nAnswer here."
        rewards = format_reward(_make_completions(content))
        # Has tags (0.5), think content <50 chars (0.0), answer present (0.2) = 0.7
        assert rewards[0] == 0.7

    def test_short_think_content(self):
        # Think content is 40 chars (< 50 threshold)
        content = "<think>\n" + "x" * 40 + "\n</think>\nThis is a longer final answer."
        rewards = format_reward(_make_completions(content))
        # Tags present (0.5) + short think (0.0) + answer present (0.2) = 0.7
        assert rewards[0] == 0.7

    def test_no_answer_after_think(self):
        content = "<think>\n" + "detailed reasoning " * 10 + "</think>"
        rewards = format_reward(_make_completions(content))
        # Tags (0.5) + good think (0.3) + no answer (0.0) = 0.8
        assert rewards[0] == 0.8

    def test_multiple_completions(self):
        good = "<think>\n" + "detailed reasoning " * 10 + "</think>\nAnswer."
        bad = "No format at all."
        rewards = format_reward(_make_completions(good, bad))
        assert len(rewards) == 2
        assert rewards[0] > rewards[1]


# ---------------------------------------------------------------------------
# accuracy_reward
# ---------------------------------------------------------------------------


class TestAccuracyReward:
    def test_exact_match(self):
        completion = _make_completion(
            "<think>\nthink\n</think>\n传递函数为 G(s) = 1/(s+1)"
        )
        solution = ["传递函数为 G(s) = 1/(s+1)"]
        rewards = accuracy_reward(completion, solution)
        assert rewards[0] is not None and rewards[0] > 0.5

    def test_keyword_overlap(self):
        completion = _make_completion(
            "<think>\n推导\n</think>\nPID控制器的传递函数为 Kp + Ki/s + Kd*s"
        )
        solution = ["PID控制器传递函数: Kp + Ki/s + Kd*s"]
        rewards = accuracy_reward(completion, solution)
        assert rewards[0] is not None and rewards[0] > 0.3

    def test_wrong_answer(self):
        completion = _make_completion(
            "<think>\nthink\n</think>\n答案是线性系统稳定"
        )
        solution = ["该系统是非线性不稳定的，发散至无穷"]
        rewards = accuracy_reward(completion, solution)
        assert rewards[0] is not None and rewards[0] < 0.5

    def test_empty_solution(self):
        completion = _make_completion("any answer")
        solution = [""]
        rewards = accuracy_reward(completion, solution)
        assert rewards[0] == 0.0  # returns 0.0 for GRPOTrainer compatibility (was None)

    def test_empty_prediction(self):
        completion = _make_completion("<think>\nthink\n</think>")
        solution = ["some reference answer"]
        rewards = accuracy_reward(completion, solution)
        assert rewards[0] == 0.0

    def test_numerical_answer(self):
        completion = _make_completion("<think>\n计算\n</think>\nw2 = 0.629")
        solution = ["w2 = 0.629"]
        rewards = accuracy_reward(completion, solution)
        assert rewards[0] is not None and rewards[0] > 0.5

    def test_multiple_completions(self):
        completions = _make_completions(
            "<think>\nthink\n</think>\nG(s) = 1/(s+1)",
            "<think>\nthink\n</think>\ncompletely wrong answer xyz",
        )
        solutions = ["G(s) = 1/(s+1)", "G(s) = 1/(s+1)"]
        rewards = accuracy_reward(completions, solutions)
        assert len(rewards) == 2
        assert rewards[0] > rewards[1]


# ---------------------------------------------------------------------------
# reasoning_depth_reward
# ---------------------------------------------------------------------------


class TestReasoningDepthReward:
    def test_deep_reasoning(self):
        think = (
            "<think>\n"
            "步骤1：首先分析系统的开环传递函数 G(s) = 1/(s^2 + 2s + 1)。\n"
            "步骤2：因此计算闭环特征方程 s^2 + 2s + 1 + K = 0。\n"
            "步骤3：根据Routh判据，可得系统稳定条件为 K > 0。\n"
            "步骤4：代入 K = 2，验证 s^2 + 2s + 3 = 0 的根均在左半平面。\n"
            "</think>\n系统稳定。"
        )
        rewards = reasoning_depth_reward(_make_completions(think))
        assert rewards[0] >= 0.7  # good depth

    def test_shallow_reasoning(self):
        think = "<think>\n答案是42。\n</think>\n42"
        rewards = reasoning_depth_reward(_make_completions(think))
        assert rewards[0] < 0.5  # too shallow

    def test_no_think_tags(self):
        content = "Just a plain answer without thinking."
        rewards = reasoning_depth_reward(_make_completions(content))
        assert rewards[0] == 0.0

    def test_formula_heavy(self):
        think = (
            "<think>\n"
            "V(x) = x^T P x，其中 P = P^T > 0。\n"
            "\\dot{V}(x) = x^T (A^T P + P A) x。\n"
            "令 A^T P + P A = -Q，其中 Q > 0。\n"
            "因此 \\dot{V}(x) = -x^T Q x < 0，系统渐近稳定。\n"
            "</think>\n系统渐近稳定。"
        )
        rewards = reasoning_depth_reward(_make_completions(think))
        # Has formulas (0.3) + logical connectors (0.2) + steps (0.3)
        assert rewards[0] >= 0.5

    def test_repetitive_content(self):
        # Highly repetitive — should lose the dedup score
        chunk = "这是一段重复的内容。" * 50
        think = f"<think>\n{chunk}\n</think>\n答案"
        rewards = reasoning_depth_reward(_make_completions(think))
        # Repetition penalty reduces score
        assert rewards[0] < 0.8

    def test_multiple_completions(self):
        deep = (
            "<think>\n"
            "1. 首先分析系统结构。\n"
            "2. 然后计算传递函数 G(s) = \\frac{1}{s+1}。\n"
            "3. 因此可以得出系统稳定的结论。\n"
            "</think>\n稳定。"
        )
        shallow = "<think>\n对。\n</think>\n对。"
        rewards = reasoning_depth_reward(_make_completions(deep, shallow))
        assert len(rewards) == 2
        assert rewards[0] > rewards[1]


# ---------------------------------------------------------------------------
# get_reward_funcs
# ---------------------------------------------------------------------------


class TestGetRewardFuncs:
    def test_default_weights(self):
        funcs, weights = get_reward_funcs()
        assert len(funcs) == 3
        assert weights == [1.0, 0.1, 0.3]
        assert funcs[0] is accuracy_reward
        assert funcs[1] is format_reward
        assert funcs[2] is reasoning_depth_reward

    def test_custom_weights(self):
        funcs, weights = get_reward_funcs(weights=(0.5, 0.5, 0.5))
        assert weights == [0.5, 0.5, 0.5]
        assert len(funcs) == 3


# ---------------------------------------------------------------------------
# Integration: full pipeline with sample control science data
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end test simulating GRPO reward computation."""

    def test_full_pipeline(self):
        # Simulate a good completion
        good = (
            "<think>\n"
            "1. 首先，根据Delta学习规则，Δw = η(d-y)x。\n"
            "2. 代入已知参数：η=0.1, w0=0.5, x1=1, d1=0.8。\n"
            "3. 计算 y1 = w0 * x1 = 0.5 * 1 = 0.5。\n"
            "4. 因此 e1 = d1 - y1 = 0.8 - 0.5 = 0.3。\n"
            "5. Δw1 = 0.1 * 0.3 * 1 = 0.03，w1 = 0.53。\n"
            "</think>\n"
            "最终权值 w2 = 0.498。"
        )

        # Simulate a poor completion
        poor = "答案是 w2 = 0.5。"

        completions = _make_completions(good, poor)
        solutions = [
            "最终权值w2 = 0.498。计算过程：y1=0.5, e1=0.3, Δw1=0.03, w1=0.53, y2=1.06, e2=-0.16, Δw2=-0.032, w2=0.498。",
            "最终权值w2 = 0.498。计算过程：y1=0.5, e1=0.3, Δw1=0.03, w1=0.53, y2=1.06, e2=-0.16, Δw2=-0.032, w2=0.498。",
        ]

        acc = accuracy_reward(completions, solutions)
        fmt = format_reward(completions)
        depth = reasoning_depth_reward(completions)

        # Good completion should score higher across all dimensions
        assert acc[0] > acc[1]
        assert fmt[0] > fmt[1]
        assert depth[0] > depth[1]

        # Weighted total
        w = [1.0, 0.1, 0.3]
        total_good = w[0] * acc[0] + w[1] * fmt[0] + w[2] * depth[0]
        total_poor = w[0] * acc[1] + w[1] * fmt[1] + w[2] * depth[1]
        assert total_good > total_poor


# ---------------------------------------------------------------------------
# Defensive programming edge cases
# ---------------------------------------------------------------------------


class TestExtractContentSafety:
    def test_empty_completion(self):
        assert _extract_content([]) == ""

    def test_missing_content_key(self):
        # Should not raise KeyError
        assert _extract_content([{"role": "assistant"}]) == ""

    def test_normal(self):
        assert _extract_content([{"role": "assistant", "content": "hello"}]) == "hello"


class TestLengthMismatch:
    def test_accuracy_reward_raises_on_mismatch(self):
        completions = _make_completions("answer1", "answer2")
        solutions = ["sol1"]  # only 1 solution for 2 completions
        with pytest.raises(ValueError, match="length mismatch"):
            accuracy_reward(completions, solutions)


class TestFormatRewardOrdering:
    def test_answer_before_think_loses_order_score(self):
        # Pathological: answer appears before <think>
        content = (
            "The answer is forty-two and here is more text."
            "<think>" + "detailed reasoning " * 10 + "</think>"
            "Some trailing text that is long enough."
        )
        rewards = format_reward(_make_completions(content))
        # Has tags (0.5) + content (0.3) but think_start > len//2
        # so order score (0.2) should NOT be awarded
        assert rewards[0] <= 0.8


# ---------------------------------------------------------------------------
# accuracy_reward_c
# ---------------------------------------------------------------------------


class TestAccuracyRewardC:
    """Tests for accuracy_reward_c with relaxed numerical matching."""

    def test_no_think_tag_still_scores(self):
        """Completion without <think> tag still gets non-zero accuracy."""
        completion = _make_completion("传递函数为 G(s) = 1/(s+1)")
        solution = ["传递函数为 G(s) = 1/(s+1)"]
        rewards = accuracy_reward_c(completion, solution)
        assert rewards[0] > 0.0

    def test_relaxed_numerical(self):
        """Different number counts still get partial score via coverage."""
        # Solution has 2 numbers, answer has 1 (different count)
        completion = _make_completion("结果是 0.629")
        solution = ["参数 w1=0.629, w2=1.414"]
        rewards = accuracy_reward_c(completion, solution)
        # Should get some score from keyword + formula + number coverage
        assert rewards[0] > 0.0

    def test_empty_solution(self):
        """Empty solution returns 0.0."""
        completion = _make_completion("any answer")
        solution = [""]
        rewards = accuracy_reward_c(completion, solution)
        assert rewards[0] == 0.0

    def test_keyword_overlap(self):
        """Keyword overlap contributes to score."""
        completion = _make_completion(
            "<think>\n推导\n</think>\nPID控制器的传递函数为 Kp + Ki/s + Kd*s"
        )
        solution = ["PID控制器传递函数: Kp + Ki/s + Kd*s"]
        rewards = accuracy_reward_c(completion, solution)
        assert rewards[0] > 0.3

    def test_empty_prediction(self):
        """Empty prediction after </think> returns 0.0."""
        completion = _make_completion("<think>\nthink\n</think>")
        solution = ["some reference answer"]
        rewards = accuracy_reward_c(completion, solution)
        assert rewards[0] == 0.0

    def test_multiple_completions(self):
        """Multiple completions processed independently."""
        completions = _make_completions(
            "<think>\nthink\n</think>\nG(s) = 1/(s+1)",
            "<think>\nthink\n</think>\ncompletely wrong answer xyz",
        )
        solutions = ["G(s) = 1/(s+1)", "G(s) = 1/(s+1)"]
        rewards = accuracy_reward_c(completions, solutions)
        assert len(rewards) == 2
        assert rewards[0] > rewards[1]

    def test_length_mismatch_raises(self):
        """Different completions/solutions lengths raise ValueError."""
        completions = _make_completions("answer1", "answer2")
        solutions = ["sol1"]
        with pytest.raises(ValueError, match="length mismatch"):
            accuracy_reward_c(completions, solutions)


# ---------------------------------------------------------------------------
# get_reward_funcs_c_v2
# ---------------------------------------------------------------------------


class TestGetRewardFuncsCv2:
    """Tests for the v2 C‑dimension reward aggregator."""

    def test_default_weights(self):
        """Returns 2 functions with default weights [0.5, 1.0]."""
        funcs, weights = get_reward_funcs_c_v2()
        assert len(funcs) == 2
        assert weights == [0.5, 1.0]
        assert funcs[0] is accuracy_reward_c
        assert funcs[1] is constraint_reward

    def test_custom_weights(self):
        """Custom weights are reflected in output."""
        funcs, weights = get_reward_funcs_c_v2(weights=(0.3, 0.8))
        assert weights == [0.3, 0.8]
        assert len(funcs) == 2
