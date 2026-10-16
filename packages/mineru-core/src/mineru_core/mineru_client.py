"""Sciverse API client — low-level HTTP client for sciverse.space APIs."""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

SCIVERSE_BASE = "https://api.sciverse.space"
MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]
REQUEST_TIMEOUT = 30
_RETRYABLE_HTTP_CODES = (429, 500, 502, 503)


class SciverseClient:
    """HTTP client for Sciverse scientific literature search API.

    Provides agentic-search, meta-search, content retrieval, and resource download.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("SCIVERSE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SCIVERSE_API_KEY not set. Set env var or pass api_key=..."
            )

    @staticmethod
    def _retry_sleep(attempt: int) -> None:
        delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
        time.sleep(delay)

    def _execute_with_retry(self, req: urllib.request.Request) -> urllib.request.addinfourl:
        """Shared retry loop for HTTP requests."""
        last_error: str = ""
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                return urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.reason}"
                if e.code in _RETRYABLE_HTTP_CODES and attempt <= MAX_RETRIES:
                    self._retry_sleep(attempt)
                    continue
                raise RuntimeError(last_error) from e
            except urllib.error.URLError as e:
                last_error = str(e.reason)
                if attempt <= MAX_RETRIES:
                    self._retry_sleep(attempt)
                    continue
                raise RuntimeError(last_error) from e
        raise RuntimeError(
            f"Request failed after {MAX_RETRIES} retries: {last_error}"
        )

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        query_params: dict | None = None,
    ) -> dict:
        url = f"{SCIVERSE_BASE}{path}"
        if query_params:
            parts = []
            for k, v in query_params.items():
                parts.append(f"{k}={urllib.parse.quote(str(v))}")
            url += "?" + "&".join(parts)

        data = json.dumps(body).encode() if body else None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        req = urllib.request.Request(
            url, data=data, headers=headers, method=method
        )
        with self._execute_with_retry(req) as resp:
            raw = resp.read().decode()
            return json.loads(raw)

    def agentic_search(self, query: str, top_k: int = 10) -> dict:
        return self._request(
            "POST",
            "/agentic-search",
            body={"query": query, "top_k": top_k},
        )

    def meta_search(
        self,
        query: str = "",
        filters: list[dict[str, Any]] | None = None,
        fields: list[str] | None = None,
        page: int = 1,
        page_size: int = 25,
    ) -> dict:
        body: dict = {"query": query, "page": page, "page_size": page_size}
        if filters:
            body["filters"] = filters
        if fields:
            body["fields"] = fields
        return self._request("POST", "/meta-search", body=body)

    def get_content(self, doc_id: str, offset: int = 0, limit: int = 700) -> dict:
        return self._request(
            "GET",
            "/content",
            query_params={"doc_id": doc_id, "offset": offset, "limit": limit},
        )

    def get_resource(self, file_name: str) -> bytes:
        url = (
            f"{SCIVERSE_BASE}/resource"
            f"?file_name={urllib.parse.quote(file_name)}"
        )
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {self.api_key}")
        with self._execute_with_retry(req) as resp:
            return resp.read()
