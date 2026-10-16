"""Integration smoke tests for llama.cpp migration.

Requires running llama.cpp servers on ports 8080 (chat) and 8081 (embedding).
Skipped automatically when services are unavailable (e.g. in CI).
"""

import pytest

from mineru_tools.llm_client import LocalLLMClient


def _llama_services_available() -> bool:
    """Check llama.cpp chat (8080) and embed (8081) servers are up."""
    try:
        c = LocalLLMClient()
        ok = c.is_chat_healthy() and c.is_embed_healthy()
        c.close()
        return ok
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _llama_services_available(),
    reason="llama.cpp servers not running (ports 8080/8081); start via: docker compose up -d",
)


class TestLLMIntegration:
    """End-to-end tests against live llama.cpp servers."""

    @pytest.fixture
    def client(self):
        c = LocalLLMClient()
        yield c
        c.close()

    def test_chat_server_healthy(self, client):
        """Chat server should be reachable on port 8080."""
        assert client.is_chat_healthy(), "Chat server not reachable on port 8080"

    def test_embed_server_healthy(self, client):
        """Embed server should be reachable on port 8081."""
        assert client.is_embed_healthy(), "Embed server not reachable on port 8081"

    def test_chat_returns_response(self, client):
        """Chat should return non-empty content (thinking mode disabled by default)."""
        result = client.chat(
            [{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10,
        )
        assert result.content.strip() != "", "Content should not be empty when thinking mode is disabled"

    def test_embed_returns_vectors(self, client):
        """Embed should return 2560-dimensional vectors."""
        result = client.embed(["test sentence"])
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 2560  # Qwen3-Embedding dim
