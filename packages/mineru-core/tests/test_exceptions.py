"""Tests for mineru_core.exceptions — hierarchy, __all__, and custom messages."""

from __future__ import annotations

import pytest

from mineru_core.exceptions import (
    ApiClientError,
    ConfigError,
    DataIntegrityError,
    EvaluationError,
    MedicalRAGError,
    PipelineError,
    SciAlignError,
)


class TestExceptionHierarchy:
    """All custom exceptions must inherit from SciAlignError."""

    def test_sci_monitor_error_is_base(self):
        assert issubclass(ConfigError, SciAlignError)
        assert issubclass(PipelineError, SciAlignError)
        assert issubclass(EvaluationError, SciAlignError)
        assert issubclass(MedicalRAGError, SciAlignError)
        assert issubclass(DataIntegrityError, SciAlignError)
        assert issubclass(ApiClientError, SciAlignError)

    def test_sci_monitor_error_not_subclass_of_standard(self):
        """SciAlignError is NOT a subclass of ValueError / RuntimeError."""
        assert not issubclass(SciAlignError, ValueError)
        assert not issubclass(SciAlignError, RuntimeError)


class TestExceptionMessages:
    def test_config_error_message(self):
        exc = ConfigError("missing env var SCIVERSE_API_KEY")
        assert str(exc) == "missing env var SCIVERSE_API_KEY"

    def test_pipeline_error_message(self):
        exc = PipelineError("corpus build failed")
        assert str(exc) == "corpus build failed"

    def test_evaluation_error_message(self):
        exc = EvaluationError("judge returned invalid score")
        assert str(exc) == "judge returned invalid score"

    def test_medical_rag_error_message(self):
        exc = MedicalRAGError("retrieval returned empty")
        assert str(exc) == "retrieval returned empty"

    def test_data_integrity_error_message(self):
        exc = DataIntegrityError("schema mismatch in core.json")
        assert str(exc) == "schema mismatch in core.json"

    def test_api_client_error_message(self):
        exc = ApiClientError("HTTP 403 Forbidden")
        assert str(exc) == "HTTP 403 Forbidden"

    def test_isinstance_check(self):
        """Catching SciAlignError should catch all subclasses."""
        try:
            raise ConfigError("test")
        except SciAlignError:
            pass
        else:
            pytest.fail("ConfigError not caught by SciAlignError")  # noqa: F821

    def test_raise_and_catch_specific(self):
        """Specific exception types should be catchable independently."""
        try:
            raise PipelineError("pipeline error")
        except PipelineError:
            pass
        else:
            pytest.fail("PipelineError not caught")  # noqa: F821
