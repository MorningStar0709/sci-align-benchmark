"""仲裁重试（根因修复）的契约测试。

根因：`_dual_judge_score` 的仲裁调用是裸 try/except、**零重试**，一次网络抖动就让
final_score 停在两评委均值上、只留一行 reason。当年 174 件争议里 65 件这样入了分，
其中 40 件正好压在 B1 的 raw_9b 臂上（B1 margin 0.0139 对阈值 0.0132 只差 0.0007）。

这里用注入假仲裁器的方式测，不触网。判据三条：
  1. 仲裁器"前两次抛错、第三次成功"时，必须重试到成功并采用仲裁分（旧代码会直接落均值）；
  2. 全部失败时，行为与旧版**逐字一致**：score 保持均值、arbitrated=False、
     reason 里仍带 `arbitration_failed`（不改历史可比性）；
  3. 重试次数有上界，不会无限打。

期望值（0.75、调用次数 3）都是硬编码的独立真值，不由被测代码回推。
"""

from mineru_eval._shared.scoring_base import _dual_judge_score


def _judge_factory(scores):
    """按 judge 身份返回固定分，制造 |s1-s2|>0.5 的争议。"""
    def score_fn(question, expected, model_answer, dimension, client, model,
                 **kwargs):
        return {"score": scores[client], "reason": f"{client} ok", "issues": []}
    return score_fn


class ArbFlaky:
    """前 fail_times 次抛 ConnectionError，之后返回固定仲裁分。"""

    def __init__(self, fail_times, score=0.75):
        self.fail_times = fail_times
        self.score = score
        self.calls = 0

    def __call__(self, client, model, question, expected, model_answer, dimension, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("Connection error.")
        return {"score": self.score, "reason": "arb ok", "issues": []}


def _run(arb, arb_retries, base=0.0):
    return _dual_judge_score(
        "Q?", "expected", "answer", "A",
        "c1", "m1", "c2", "m2",
        arb_client="ac", arb_model="qwen3.7-plus",
        score_fn=_judge_factory({"c1": 0.0, "c2": 0.75}),
        arb_fn=arb,
        arb_retries=arb_retries, arb_backoff_base=base,
    )


def test_retries_until_arbitration_succeeds():
    """旧代码在这里必然失败（第一次抛错就放弃、落均值）——这正是 65 件的成因。"""
    arb = ArbFlaky(fail_times=2, score=0.75)
    result = _run(arb, arb_retries=3)
    assert arb.calls == 3, f"应重试到第 3 次成功，实际调用 {arb.calls} 次"
    assert result["arbitrated"] is True
    assert result["score"] == 0.75
    assert "ARBITRATED" in result["reason"]


def test_first_try_success_is_unchanged_and_calls_once():
    arb = ArbFlaky(fail_times=0, score=1.0)
    result = _run(arb, arb_retries=3)
    assert arb.calls == 1
    assert result["arbitrated"] is True and result["score"] == 1.0


def test_all_attempts_fail_falls_back_to_average_exactly_as_before():
    """失败路径必须与旧版逐字一致：均值入分、arbitrated=False、留 arbitration_failed。"""
    arb = ArbFlaky(fail_times=99)
    result = _run(arb, arb_retries=3)
    assert result["arbitrated"] is False
    assert result["score"] == result["avg_score"]  # 0.0 与 0.75 的均值吸附到 0.5
    assert result["score"] == 0.5
    assert "arbitration_failed" in result["reason"]


def test_retry_count_is_bounded():
    arb = ArbFlaky(fail_times=99)
    _run(arb, arb_retries=2)
    assert arb.calls == 3  # 首次 + 2 次重试，不得多打


def test_no_arb_client_means_no_change():
    """未配置仲裁器时不得进入重试分支（保持单/双判无仲裁的原行为）。"""
    result = _dual_judge_score(
        "Q?", "expected", "answer", "A", "c1", "m1", "c2", "m2",
        arb_client=None, arb_model=None,
        score_fn=_judge_factory({"c1": 0.0, "c2": 0.75}),
        arb_fn=ArbFlaky(fail_times=0),
    )
    assert result["arbitrated"] is False and result["disputed"] is True
    assert result["score"] == 0.5


def test_backoff_sleeps_with_increasing_delay(monkeypatch):
    """退避要真的递增（1s,2s,4s），且最后一次失败后不再多睡。"""
    slept = []
    monkeypatch.setattr("mineru_eval._shared.scoring_base.time.sleep", lambda d: slept.append(d))
    arb = ArbFlaky(fail_times=99)
    _run(arb, arb_retries=3, base=1.0)
    assert slept == [1.0, 2.0, 4.0]
