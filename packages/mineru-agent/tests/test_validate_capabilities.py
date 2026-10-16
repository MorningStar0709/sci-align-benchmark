"""Regression tests for _validate_capabilities — extracted helper functions."""

from __future__ import annotations

from unittest.mock import patch

from mineru_agent._validate_capabilities import (
    _parameters,
    _print_intents,
    _validate_intent,
    _validate_intents,
    _validate_meta,
    _validate_scheduler,
)


class TestParameters:
    def test_with_parameters(self):
        intent = {
            "parameters": {
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        }
        props, required = _parameters(intent)
        assert props == {"query": {"type": "string"}}
        assert required == ["query"]

    def test_no_parameters(self):
        intent = {"parameters": None}
        props, required = _parameters(intent)
        assert props == {}
        assert required == []


class TestValidateMeta:
    def test_total_intents_match(self):
        meta = {"total_intents": 3, "privacy_policy": {"principle": "cloud_for_public_or_sanitized_local_for_private", "cloud_allowed": ["search"], "local_only": ["parse"]}}
        errors = _validate_meta(meta, [1, 2, 3])
        assert errors == []

    def test_total_intents_mismatch(self):
        meta = {"total_intents": 5, "privacy_policy": {"principle": "cloud_for_public_or_sanitized_local_for_private", "cloud_allowed": ["search"], "local_only": ["parse"]}}
        errors = _validate_meta(meta, [1, 2, 3])
        assert any("total_intents mismatch" in e for e in errors)

    def test_missing_privacy_policy(self):
        meta = {}
        errors = _validate_meta(meta, [])
        assert any("principle" in e for e in errors)
        assert any("cloud_allowed" in e for e in errors)
        assert any("local_only" in e for e in errors)


class TestValidateIntent:
    def test_valid_intent(self):
        intent = {
            "intent_id": "arxiv_search",
            "rank": 1,
            "parameters": {
                "additionalProperties": False,
                "properties": {"query": {"type": "string"}},
            },
            "output": {"artifacts": ["result"]},
        }
        errors = _validate_intent(intent, set())
        assert errors == []

    def test_missing_intent_id(self):
        errors = _validate_intent({}, set())
        assert any("intent_id" in e for e in errors)

    def test_duplicate_intent_id(self):
        seen = {"test_id"}
        intent = {"intent_id": "test_id"}
        errors = _validate_intent(intent, seen)
        assert any("duplicate" in e for e in errors)

    def test_missing_additional_properties(self):
        intent = {
            "intent_id": "search",
            "parameters": {"properties": {}},
            "provider_defaults": {},
            "output": {"artifacts": []},
        }
        errors = _validate_intent(intent, set())
        assert any("additionalProperties" in e for e in errors)

    def test_special_no_provider_defaults(self):
        """intents in _SPECIAL_NO_PROVIDER_DEFAULTS should NOT have provider_defaults."""
        intent = {
            "intent_id": "arxiv_search",
            "provider_defaults": {},  # should NOT have this
            "output": {"artifacts": []},
        }
        errors = _validate_intent(intent, set())
        assert any("should NOT have provider_defaults" in e for e in errors)

    def test_missing_provider_defaults(self):
        intent = {
            "intent_id": "some_intent",
            "output": {"artifacts": []},
        }
        errors = _validate_intent(intent, set())
        assert any("missing provider_defaults" in e for e in errors)

    def test_missing_output_artifacts(self):
        intent = {
            "intent_id": "test",
            "output": {},
            "provider_defaults": {},
        }
        errors = _validate_intent(intent, set())
        assert any("artifacts" in e for e in errors)


class TestValidateIntents:
    def test_empty_intents(self):
        errors = _validate_intents({})
        assert errors == []

    def test_valid_intents(self):
        d = {
            "intents": [
                {
                    "intent_id": "search",
                    "rank": 1,
                    "parameters": {"additionalProperties": False, "properties": {}},
                    "provider_defaults": {},
                    "output": {"artifacts": []},
                }
            ]
        }
        errors = _validate_intents(d)
        assert errors == []

    def test_multiple_intents(self):
        d = {
            "intents": [
                {"intent_id": "a", "parameters": {"additionalProperties": False, "properties": {}}, "provider_defaults": {}, "output": {"artifacts": []}},
                {"intent_id": "b", "parameters": {"additionalProperties": False, "properties": {}}, "provider_defaults": {}, "output": {"artifacts": []}},
            ]
        }
        errors = _validate_intents(d)
        assert errors == []


class TestValidateScheduler:
    def test_valid_scheduler(self):
        scheduler = {
            "fallback_order": {"api": [], "local_api": [], "local_gpu": [], "script": []},
            "provider_parameter_map": {
                "openai": {
                    "default": {
                        "model": "gpt-4",
                        "default_temperature": 0.1,
                        "default_max_tokens": 4096,
                    }
                }
            },
        }
        errors = _validate_scheduler(scheduler)
        assert errors == []

    def test_missing_fallback_order_keys(self):
        errors = _validate_scheduler({})
        assert len(errors) == 4  # 4 missing resource_types

    def test_missing_model_in_provider(self):
        scheduler = {
            "fallback_order": {"api": [], "local_api": [], "local_gpu": [], "script": []},
            "provider_parameter_map": {
                "openai": {
                    "default": {
                        "default_temperature": 0.1,
                        "default_max_tokens": 4096,
                    }
                }
            },
        }
        errors = _validate_scheduler(scheduler)
        assert any("missing model" in e for e in errors)

    def test_missing_temperature(self):
        scheduler = {
            "fallback_order": {"api": [], "local_api": [], "local_gpu": [], "script": []},
            "provider_parameter_map": {
                "openai": {
                    "default": {
                        "model": "gpt-4",
                        "default_max_tokens": 4096,
                    }
                }
            },
        }
        errors = _validate_scheduler(scheduler)
        assert any("missing default_temperature" in e for e in errors)


class TestPrintIntents:
    def test_print_intents(self):
        intents = [
            {"rank": 1, "intent_id": "search", "resource_type": "api", "parameters": {"properties": {}}},
        ]
        with patch("builtins.print") as mock_print:
            _print_intents(intents)
            mock_print.assert_called_once()
