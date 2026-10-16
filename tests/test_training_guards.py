"""Tests for mineru_eval.finetune.training_guards — attn 校验 + 缓存诊断回调。"""
from __future__ import annotations

from types import SimpleNamespace

from mineru_eval.finetune.training_guards import (
    CacheDiagnosticCallback,
    validate_attn_implementation,
)


def _model(impl):
    return SimpleNamespace(config=SimpleNamespace(_attn_implementation=impl))


def test_attn_match():
    assert validate_attn_implementation(_model("flash_attention_2"),
                                        "flash_attention_2") is True


def test_attn_mismatch_warns(capsys):
    ok = validate_attn_implementation(_model("eager"), "flash_attention_2")
    assert ok is False
    assert "attn" in capsys.readouterr().out.lower()


def test_diag_prints_key_fields(capsys):
    cb = CacheDiagnosticCallback()
    model = SimpleNamespace(training=True, is_gradient_checkpointing=True,
                            is_fast_path_available=False,
                            config=SimpleNamespace(use_cache=False,
                                                   _attn_implementation="flash_attention_2"))
    cb.on_train_begin(None, None, None, model=model)
    out = capsys.readouterr().out
    for key in ["training", "use_cache", "gradient_checkpointing",
                "attn_implementation", "is_fast_path_available"]:
        assert key in out


def test_diag_first_step_only_once(capsys):
    cb = CacheDiagnosticCallback()
    model = SimpleNamespace(training=True, is_gradient_checkpointing=True,
                            config=SimpleNamespace(use_cache=False,
                                                   _attn_implementation="eager"))
    cb.on_step_end(None, None, None, model=model)
    cb.on_step_end(None, None, None, model=model)
    # 仅首步打印一次，避免刷屏
    assert out_count(capsys.readouterr().out) == 1


def out_count(text: str) -> int:
    return text.count("[diag][first_step_end]")
