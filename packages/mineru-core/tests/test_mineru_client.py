"""Tests for SciverseClient — mock urllib to test retry, timeout, error paths."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest
import urllib.error

from mineru_core.mineru_client import SciverseClient, MAX_RETRIES


@pytest.fixture
def client():
    with patch.dict("os.environ", {"SCIVERSE_API_KEY": "test-key-123"}):
        return SciverseClient()


class TestSciverseClientInit:
    def test_init_from_env(self):
        with patch.dict("os.environ", {"SCIVERSE_API_KEY": "env-key"}):
            c = SciverseClient()
            assert c.api_key == "env-key"

    def test_init_explicit_key(self):
        c = SciverseClient(api_key="explicit-key")
        assert c.api_key == "explicit-key"

    def test_init_no_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="SCIVERSE_API_KEY"):
                SciverseClient()


class TestExecuteWithRetry:
    @patch("mineru_core.mineru_client.time.sleep")
    @patch("mineru_core.mineru_client.urllib.request.urlopen")
    def test_success_first_try(self, mock_urlopen, mock_sleep, client):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"ok": true}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        req = MagicMock()
        resp = client._execute_with_retry(req)
        assert resp is mock_resp
        mock_sleep.assert_not_called()

    @patch("mineru_core.mineru_client.time.sleep")
    @patch("mineru_core.mineru_client.urllib.request.urlopen")
    def test_retry_on_429(self, mock_urlopen, mock_sleep, client):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"ok": true}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            urllib.error.HTTPError("url", 429, "Too Many Requests", {}, io.BytesIO()),
            mock_resp,
        ]

        req = MagicMock()
        resp = client._execute_with_retry(req)
        assert resp is mock_resp
        assert mock_sleep.call_count == 1

    @patch("mineru_core.mineru_client.time.sleep")
    @patch("mineru_core.mineru_client.urllib.request.urlopen")
    def test_non_retryable_http_error(self, mock_urlopen, mock_sleep, client):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 403, "Forbidden", {}, io.BytesIO()
        )

        req = MagicMock()
        with pytest.raises(RuntimeError, match="HTTP 403"):
            client._execute_with_retry(req)
        mock_sleep.assert_not_called()

    @patch("mineru_core.mineru_client.time.sleep")
    @patch("mineru_core.mineru_client.urllib.request.urlopen")
    def test_url_error_retries_exhausted(self, mock_urlopen, mock_sleep, client):
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        req = MagicMock()
        with pytest.raises(RuntimeError, match="Connection refused"):
            client._execute_with_retry(req)
        assert mock_sleep.call_count == MAX_RETRIES


class TestRequest:
    @patch("mineru_core.mineru_client.time.sleep")
    @patch("mineru_core.mineru_client.urllib.request.urlopen")
    def test_request_json_response(self, mock_urlopen, mock_sleep, client):
        body = json.dumps({"results": [1, 2, 3]}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = client._request("POST", "/test", body={"q": "hello"})
        assert result == {"results": [1, 2, 3]}

    @patch("mineru_core.mineru_client.time.sleep")
    @patch("mineru_core.mineru_client.urllib.request.urlopen")
    def test_request_with_query_params(self, mock_urlopen, mock_sleep, client):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"data": []}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = client._request("GET", "/content", query_params={"doc_id": "abc"})
        assert result == {"data": []}
        # Verify URL contains query params
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "doc_id=abc" in req.full_url


class TestGetResource:
    @patch("mineru_core.mineru_client.time.sleep")
    @patch("mineru_core.mineru_client.urllib.request.urlopen")
    def test_get_resource_bytes(self, mock_urlopen, mock_sleep, client):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"PDF-content-bytes"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        data = client.get_resource("paper.pdf")
        assert data == b"PDF-content-bytes"
