"""Task 4/5/6 tests — system_prompt 注入与 medical max_tokens 预算参数化。

覆盖:
- Task 4: call_target_model_local 默认 system prompt 不变 / 自定义注入；
          evaluate_adapter 串行+并行路径贯穿 system_prompt
- Task 5: eval_cli 新参数解析 + 现有参数默认值快照（回归锚）
- Task 6: call_target_model 默认 max_tokens=4000 不变 / 覆盖生效；
          _call_medical_target 默认 MEDICAL_TARGET_SYSTEM_PROMPT
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from mineru_eval._shared.call_utils import TARGET_SYSTEM_PROMPT, call_target_model
from mineru_eval.control.eval_adapter import call_target_model_local
from mineru_eval.medical.prompts import MEDICAL_TARGET_SYSTEM_PROMPT
from mineru_eval.medical.scoring import _call_medical_target


# ── Fake 模型/tokenizer（真实 torch，CPU，不加载真实权重）─────────────

class FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        return torch.zeros((1, 20), dtype=torch.long)


class _FakeEncoding(dict):
    """模拟 transformers BatchEncoding：dict 行为 + .to(device)。"""

    def to(self, device):
        return self


class FakeTokenizer:
    def __init__(self):
        self.captured_messages = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.captured_messages = messages
        return "prompt"

    def __call__(self, prompt, return_tensors="pt"):
        # 真实 tokenizer 返回 BatchEncoding（Mapping + .to），支持 **inputs 解包
        return _FakeEncoding(input_ids=torch.zeros((1, 5), dtype=torch.long))

    def decode(self, token_ids, skip_special_tokens=True):
        return "model answer"

    pad_token_id = 0
    eos_token_id = 0


# ── Task 4: call_target_model_local 的 messages 构造 ──────────────────

class TestCallTargetModelLocal:
    def test_default_system_prompt_unchanged(self):
        tok = FakeTokenizer()
        ans = call_target_model_local("Q?", FakeModel(), tok)
        assert tok.captured_messages[0]["role"] == "system"
        assert tok.captured_messages[0]["content"] == TARGET_SYSTEM_PROMPT
        assert ans == "model answer"

    def test_custom_system_prompt_injected(self):
        tok = FakeTokenizer()
        call_target_model_local("Q?", FakeModel(), tok, system_prompt="X")
        assert tok.captured_messages[0]["content"] == "X"

    def test_empty_string_falls_back_to_builtin(self):
        # 空串回退内置（or 语义，与 call_utils.call_target_model 一致）
        tok = FakeTokenizer()
        call_target_model_local("Q?", FakeModel(), tok, system_prompt="")
        assert tok.captured_messages[0]["content"] == TARGET_SYSTEM_PROMPT


# ── Task 4: evaluate_adapter 贯穿 system_prompt ───────────────────────

def _eval_data():
    return {"questions": [{"id": "q1", "dimension": "A", "difficulty_level": "L1",
                           "question": "Q?", "answer": "A"}]}


class TestEvaluateAdapterInjection:
    def test_serial_path_passes_system_prompt(self):
        with patch("mineru_eval.control.eval_adapter.call_target_model_local") as mock_gen, \
             patch("mineru_eval.control.eval_adapter._score_or_fallback",
                   return_value=({"score": 0.5, "issues": []}, 0)):
            from mineru_eval.control.eval_adapter import evaluate_adapter
            evaluate_adapter(
                _eval_data(), object(), object(), object(), "judge", "t",
                parallel=1, system_prompt="X",
            )
            assert mock_gen.call_args.kwargs["system_prompt"] == "X"

    def test_serial_path_default_is_none(self):
        with patch("mineru_eval.control.eval_adapter.call_target_model_local") as mock_gen, \
             patch("mineru_eval.control.eval_adapter._score_or_fallback",
                   return_value=({"score": 0.5, "issues": []}, 0)):
            from mineru_eval.control.eval_adapter import evaluate_adapter
            evaluate_adapter(
                _eval_data(), object(), object(), object(), "judge", "t",
                parallel=1,
            )
            assert mock_gen.call_args.kwargs["system_prompt"] is None

    def test_parallel_path_passes_system_prompt(self):
        with patch("mineru_eval.control.eval_adapter._evaluate_parallel",
                   return_value={"complete": True}) as mock_ep:
            from mineru_eval.control.eval_adapter import evaluate_adapter
            evaluate_adapter(
                _eval_data(), object(), object(), object(), "judge", "t",
                parallel=4, system_prompt="X",
            )
            assert mock_ep.call_args.kwargs["system_prompt"] == "X"


# ── Task 5: eval_cli 参数解析 ─────────────────────────────────────────

class TestCliArgs:
    def _parse(self, extra=None):
        from mineru_eval.eval_cli import parse_args
        argv = ["--adapter-path", "x", "--output", "y"] + (extra or [])
        return parse_args(argv)

    def test_new_args_parsed(self):
        args = self._parse([
            "--target-system-prompt", "X",
            "--medical-max-tokens", "8192",
        ])
        assert args.target_system_prompt == "X"
        assert args.medical_max_tokens == 8192

    def test_new_args_default_none(self):
        args = self._parse()
        assert args.target_system_prompt is None
        assert args.medical_max_tokens is None

    def test_existing_defaults_unchanged(self):
        """现有参数默认值快照（回归锚：默认行为未动证明）。"""
        args = self._parse()
        assert args.adapter_path is None or args.raw is False
        assert args.raw is False
        assert args.judge_model == "deepseek-v4-flash"
        assert args.limit == 0
        assert args.parallel == 1
        assert args.balanced_mini is False
        assert args.resume is False
        assert args.dual_judge is False
        assert args.all_dimensions_dual is False
        assert args.retry_failed is False
        assert args.retries == 3
        assert args.max_consecutive_failures == 5
        assert args.max_answer_tokens == 2048
        assert args.fixed_questions is None
        assert args.answers_cache is None
        assert args.domain == "control"
        assert args.allow_incomplete is False
        assert args.base_model_id is None
        assert args.judge_api_key is None


# ── Task 6: call_target_model max_tokens 参数化 ───────────────────────

class TestCallTargetModel:
    def _client(self):
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="ans", reasoning_content=None))],
        )
        client = Mock()
        client.chat.completions.create.return_value = resp
        return client

    def test_default_max_tokens_4000(self):
        client = self._client()
        call_target_model("q", client, "m", provider_kind="openai")
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["max_tokens"] == 4000
        assert kwargs["messages"][0]["content"] == TARGET_SYSTEM_PROMPT

    def test_override_max_tokens(self):
        client = self._client()
        call_target_model("q", client, "m", provider_kind="openai", max_tokens=8192)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["max_tokens"] == 8192

    def test_local_provider_uses_max_tokens(self):
        client = self._client()
        call_target_model("q", client, "m", provider_kind="local", max_tokens=4096)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["max_tokens"] == 4096

    def test_anthropic_provider_uses_max_tokens(self):
        resp = SimpleNamespace(content=[SimpleNamespace(text="ans")])
        client = Mock()
        client.messages.create.return_value = resp
        call_target_model("q", client, "m", provider_kind="anthropic", max_tokens=8192)
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["max_tokens"] == 8192


class TestMedicalTargetWrapper:
    def test_default_medical_prompt(self):
        client = Mock()
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="ans", reasoning_content=None))],
        )
        client.chat.completions.create.return_value = resp
        _call_medical_target("q", client, "m", "openai")
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["messages"][0]["content"] == MEDICAL_TARGET_SYSTEM_PROMPT
        assert kwargs["max_tokens"] == 4000

    def test_override_prompt_and_tokens(self):
        client = Mock()
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="ans", reasoning_content=None))],
        )
        client.chat.completions.create.return_value = resp
        _call_medical_target("q", client, "m", "openai",
                             system_prompt="Y", max_tokens=8192)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["messages"][0]["content"] == "Y"
        assert kwargs["max_tokens"] == 8192
