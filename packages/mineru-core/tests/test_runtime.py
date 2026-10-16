"""Tests for mineru_core.runtime."""

from __future__ import annotations

from mineru_core.runtime import (
    RuntimeConfig,
    allows_cloud_document_upload,
    normalize_data_class,
    normalize_parser_backend,
    normalize_runtime,
)


class TestRuntimeConfig:
    def test_default_profile(self):
        r = RuntimeConfig()
        assert r.profile == "demo_replay"

    def test_normalize_replay(self):
        r = normalize_runtime(None)
        assert r.parser_backend == "replay"
        assert r.privacy_policy == "local_only"

    def test_normalize_parser_backend(self):
        assert normalize_parser_backend("mineru_docker") == "local"
        assert normalize_parser_backend("mineru_api") == "official"
        assert normalize_parser_backend("replay") == "replay"

    def test_normalize_data_class(self):
        assert normalize_data_class("public") == "public_open"
        assert normalize_data_class("medical") == "private_medical"
        assert normalize_data_class("unknown") == "unknown"

    def test_allows_cloud_upload(self):
        r = RuntimeConfig(
            allow_cloud_upload=True,
            data_class="public_open",
            privacy_policy="permissive",
            profile="cloud_demo",
        )
        assert allows_cloud_document_upload(r)

    def test_disallows_cloud_upload_local_only(self):
        r = RuntimeConfig(allow_cloud_upload=True, data_class="public_open")
        assert not allows_cloud_document_upload(r)  # privacy_policy default is local_only
