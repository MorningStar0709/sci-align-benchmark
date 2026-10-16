"""Tests for LocalLLMClient — unit tests with HTTP mocking."""

from unittest.mock import MagicMock, patch


from mineru_tools.llm_client import ChatResponse, EmbedResponse, LocalLLMClient


class TestLocalLLMClient:
    """Test the unified local LLM client."""

    def test_chat_returns_content(self) -> None:
        """Chat should extract content from OpenAI-format response."""
        client = LocalLLMClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}],
        }
        with patch.object(client._http, "post", return_value=mock_resp):
            result = client.chat([{"role": "user", "content": "Hi"}])
        assert isinstance(result, ChatResponse)
        assert result.content == "Hello!"

    def test_chat_json_format(self) -> None:
        """Chat with response_format='json' should include json_object in payload."""
        client = LocalLLMClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"answer": 42}'}}],
        }
        with patch.object(client._http, "post", return_value=mock_resp) as mock_post:
            client.chat(
                [{"role": "user", "content": "test"}],
                response_format="json",
            )
            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload.get("response_format") == {"type": "json_object"}

    def test_embed_returns_embeddings(self) -> None:
        """Embed should extract embeddings from OpenAI-format response."""
        client = LocalLLMClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ],
        }
        with patch.object(client._http, "post", return_value=mock_resp):
            result = client.embed(["hello", "world"])
        assert isinstance(result, EmbedResponse)
        assert len(result.embeddings) == 2
        assert result.embeddings[0] == [0.1, 0.2, 0.3]

    def test_health_check_chat_success(self) -> None:
        """Health check should return True when server responds 200."""
        client = LocalLLMClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch.object(client._http, "get", return_value=mock_resp):
            assert client.is_chat_healthy() is True

    def test_health_check_chat_failure(self) -> None:
        """Health check should return False when server is unreachable."""
        client = LocalLLMClient()
        with patch.object(client._http, "get", side_effect=Exception("timeout")):
            assert client.is_chat_healthy() is False
