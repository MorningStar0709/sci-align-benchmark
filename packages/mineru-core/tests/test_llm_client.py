"""Tests for llm_client.call_openai_direct — mock httpx to test retry, timeout, success."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from mineru_core.llm_client import call_openai_direct


class TestCallOpenaiDirect:
    """Cover call_openai_direct: success, empty, retry, exhaustion, invalid JSON."""

    KWARGS = dict(
        api_key="test-key",
        system_prompt="You are helpful",
        user_prompt="Hello",
    )

    def _mock_response(self, content: str) -> MagicMock:
        resp = MagicMock(spec=httpx.Response)
        resp.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        return resp

    def test_success_returns_parsed_json(self):
        resp = self._mock_response('{"result": "ok"}')
        with patch("httpx.Client") as cls:
            cls.return_value.__enter__.return_value.post.return_value = resp
            result = call_openai_direct(**self.KWARGS)
        assert result == {"result": "ok"}

    def test_empty_content_returns_none(self):
        resp = self._mock_response("")
        with patch("httpx.Client") as cls:
            cls.return_value.__enter__.return_value.post.return_value = resp
            result = call_openai_direct(**self.KWARGS)
        assert result is None

    def test_invalid_json_returns_none(self):
        """Non-JSON content triggers retry exhaustion, returning None."""
        resp = self._mock_response("not json")
        with patch("httpx.Client") as cls:
            cls.return_value.__enter__.return_value.post.return_value = resp
            with patch("mineru_core.llm_client.time.sleep"):
                result = call_openai_direct(**self.KWARGS)
        assert result is None

    def test_retry_on_429_then_succeeds(self):
        """429 rate limit triggers retry with backoff, then succeeds."""
        responses = [
            httpx.HTTPStatusError(
                "429",
                request=MagicMock(),
                response=MagicMock(status_code=429),
            ),
            self._mock_response('{"ok": true}'),
        ]
        with patch("httpx.Client") as cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = responses
            cls.return_value.__enter__.return_value = mock_client
            with patch("mineru_core.llm_client.time.sleep") as sleep:
                result = call_openai_direct(**self.KWARGS, max_retries=3)
        assert result == {"ok": True}
        assert sleep.call_count == 1

    def test_all_retries_fail_returns_none(self):
        """All retries exhausted → return None."""
        with patch("httpx.Client") as cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = httpx.TransportError("Connection failed")
            cls.return_value.__enter__.return_value = mock_client
            with patch("mineru_core.llm_client.time.sleep"):
                result = call_openai_direct(**self.KWARGS, max_retries=2)
        assert result is None

    def test_custom_model_and_base_url(self):
        """Custom model/base_url passed through to the API call."""
        resp = self._mock_response('{"ok": true}')
        with patch("httpx.Client") as cls:
            mock_client = MagicMock()
            mock_client.post.return_value = resp
            cls.return_value.__enter__.return_value = mock_client
            result = call_openai_direct(
                **self.KWARGS,
                model="gpt-4",
                base_url="https://api.openai.com",
            )
        assert result == {"ok": True}
        # Verify the URL used in the call
        call_kwargs = mock_client.post.call_args[1]
        posted_url = mock_client.post.call_args[0][0]
        assert "api.openai.com" in posted_url

    def test_httpx_error_different_from_429(self):
        """Non-429 HTTPStatusError (e.g. 403) should still trigger retry."""
        error_403 = httpx.HTTPStatusError(
            "403", request=MagicMock(), response=MagicMock(status_code=403)
        )
        with patch("httpx.Client") as cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = [error_403, self._mock_response('{"ok": true}')]
            cls.return_value.__enter__.return_value = mock_client
            with patch("mineru_core.llm_client.time.sleep"):
                result = call_openai_direct(**self.KWARGS, max_retries=3)
        assert result == {"ok": True}

    def test_default_parameters(self):
        """Default parameters (model, timeout, etc.) should not cause errors."""
        resp = self._mock_response('{"ok": true}')
        with patch("httpx.Client") as cls:
            cls.return_value.__enter__.return_value.post.return_value = resp
            result = call_openai_direct(**self.KWARGS)
        assert result == {"ok": True}
