"""Tests for mineru_core.settings."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mineru_core.settings import (
    ApiSettings,
    _env_or_config,
    _int_env,
    _int_env_or_config,
    _read_cli_config,
    get_settings,
)


class TestApiSettings:
    def test_default_api_host(self):
        s = ApiSettings()
        assert s.api_host == "127.0.0.1"

    def test_default_api_port(self):
        s = ApiSettings()
        assert s.api_port == 17001

    def test_api_base_url(self):
        s = ApiSettings()
        assert s.api_base_url == "http://127.0.0.1:17001"

    def test_custom_settings(self):
        s = ApiSettings(api_host="0.0.0.0", api_port=8080)
        assert s.api_base_url == "http://0.0.0.0:8080"

    def test_get_settings_cached(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2  # lru_cache guarantees same object


class TestIntEnv:
    def test_valid_int(self):
        with patch.dict(os.environ, {"MY_PORT": "8080"}):
            assert _int_env("MY_PORT", 17001) == 8080

    def test_invalid_int_returns_default(self):
        with patch.dict(os.environ, {"MY_PORT": "not-a-number"}):
            assert _int_env("MY_PORT", 17001) == 17001

    def test_missing_env_returns_default(self):
        assert _int_env("NONEXISTENT_VAR", 42) == 42


class TestReadCliConfig:
    def test_no_config_file_returns_empty_dict(self):
        with patch.object(Path, "exists", return_value=False):
            assert _read_cli_config() == {}

    def test_valid_config(self, tmp_path):
        config_dir = tmp_path / ".controlmind"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text('{"api.host": "0.0.0.0"}', encoding="utf-8")
        with patch("mineru_core.settings.CONFIG_PATH", config_file):
            result = _read_cli_config()
        assert result == {"api.host": "0.0.0.0"}

    def test_corrupted_config_raises(self, tmp_path):
        config_dir = tmp_path / ".controlmind"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text("{invalid json}", encoding="utf-8")
        with patch("mineru_core.settings.CONFIG_PATH", config_file):
            with pytest.raises(RuntimeError, match="配置文件损坏"):
                _read_cli_config()


class TestEnvOrConfig:
    def test_env_takes_priority(self):
        with patch.dict(os.environ, {"MY_HOST": "from-env"}):
            result = _env_or_config(("MY_HOST",), {}, "host", "default")
        assert result == "from-env"

    def test_empty_env_skips_to_config(self):
        with patch.dict(os.environ, {"MY_HOST": ""}):
            result = _env_or_config(("MY_HOST",), {"host": "from-config"}, "host", "default")
        assert result == "from-config"

    def test_config_fallback(self):
        result = _env_or_config(("MISSING",), {"host": "cfg"}, "host", "default")
        assert result == "cfg"

    def test_default_fallback(self):
        result = _env_or_config(("MISSING",), {}, "host", "fallback")
        assert result == "fallback"


class TestIntEnvOrConfig:
    def test_env_takes_priority(self):
        with patch.dict(os.environ, {"MY_PORT": "8080"}):
            result = _int_env_or_config(("MY_PORT",), {}, "port", 17001)
        assert result == 8080

    def test_invalid_env_returns_default(self):
        with patch.dict(os.environ, {"MY_PORT": "invalid"}):
            result = _int_env_or_config(("MY_PORT",), {}, "port", 17001)
        assert result == 17001

    def test_config_fallback(self):
        result = _int_env_or_config(("MISSING",), {"port": "9090"}, "port", 17001)
        assert result == 9090

    def test_config_not_int_returns_default(self):
        result = _int_env_or_config(("MISSING",), {"port": "not-int"}, "port", 17001)
        assert result == 17001

    def test_config_missing_key_returns_default(self):
        result = _int_env_or_config(("MISSING",), {}, "port", 17001)
        assert result == 17001
