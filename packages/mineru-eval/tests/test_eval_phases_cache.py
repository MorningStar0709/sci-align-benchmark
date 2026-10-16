"""Regression tests for answers-cache mode: cache-missing questions must not be silently dropped.

Bug (2026-08-07): in cache mode, `_evaluate_parallel` built `pending` exclusively from cached
answers. Questions whose answer was missing in the cache (e.g. empty model_answer filtered
out at load) were silently discarded — the eval loop could never reach `complete=True`
(sft_gemma4 stuck at 298/300 across 3 re-runs).
"""

from __future__ import annotations

import json
from unittest.mock import patch

from mineru_eval.eval_phases import _evaluate_parallel

Q = [
    {"id": "Q1", "dimension": "A", "difficulty_level": "L1", "question": "q1", "answer": "a1"},
    {"id": "Q2", "dimension": "A", "difficulty_level": "L1", "question": "q2", "answer": "a2"},
    {"id": "Q3", "dimension": "A", "difficulty_level": "L1", "question": "q3", "answer": "a3"},
]


def _make_cache(tmp_path, records):
    p = tmp_path / "cache.json"
    p.write_text(json.dumps({"records": records}), encoding="utf-8")
    return p


def _run(questions, cache_path, output_path):
    with patch("mineru_eval.eval_phases._phase2_score") as mock_p2, \
            patch("mineru_eval.eval_phases._generate_one_answer") as mock_gen:
        mock_p2.return_value = ([], False)
        mock_gen.return_value = ("generated-answer", 0.0)
        _evaluate_parallel(
            questions,
            completed_ids={"Q1"},
            records=[],
            model=object(),
            tokenizer=object(),
            judge_client=object(),
            judge_model="judge",
            adapter_name="adapter",
            output_path=output_path,
            status_path=None,
            total=len(questions),
            started_at="2026-08-07 00:00:00",
            session_start=0.0,
            initial_completed=1,
            retries=3,
            max_consecutive_failures=5,
            parallel=4,
            answers_cache=cache_path,
        )
        return mock_p2, mock_gen


class TestCacheMissingQuestions:
    def test_cache_missing_question_goes_to_gpu(self, tmp_path):
        """Cache 缺失答案的题必须走 GPU 生成，不得被静默丢弃。"""
        cache = _make_cache(tmp_path, [
            {"id": "Q2", "model_answer": "cached-answer"},
            {"id": "Q3", "model_answer": ""},  # 空答案 → 加载时被过滤
        ])
        output = tmp_path / "out.json"

        mock_p2, mock_gen = _run(Q, cache, output)

        # Q3 必须触发 GPU 生成
        assert mock_gen.called, "cache 缺失题必须走 GPU 生成"
        # pending 必须包含 Q2（cache）和 Q3（生成）
        pending = mock_p2.call_args[0][0]
        ids = {q["id"] for q, _ in pending}
        assert ids == {"Q2", "Q3"}, f"pending 应为 {{Q2, Q3}}，实际 {ids}"

    def test_cache_complete_no_gpu(self, tmp_path):
        """cache 完整时不应触发 GPU 生成。"""
        cache = _make_cache(tmp_path, [
            {"id": "Q2", "model_answer": "cached-answer"},
            {"id": "Q3", "model_answer": "cached-answer3"},
        ])
        output = tmp_path / "out.json"

        mock_p2, mock_gen = _run(Q, cache, output)

        assert not mock_gen.called
        pending = mock_p2.call_args[0][0]
        assert {q["id"] for q, _ in pending} == {"Q2", "Q3"}
