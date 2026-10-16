"""Tests for mineru_core.protocols — verify runtime_checkable Protocol classes."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from mineru_core.protocols import (
    ApiClient,
    DataStore,
    LogEntry,
    SettingsProvider,
)


class TestDataStoreProtocol:
    """DataStore is runtime_checkable — valid/invalid implementations."""

    def test_valid_implementation(self):
        class MyStore:
            def load_json(self, path):
                return {}

            def write_json_atomic(self, path, data):
                pass

            def append_jsonl(self, path, item):
                pass

        assert isinstance(MyStore(), DataStore)

    def test_invalid_implementation_missing_method(self):
        class BadStore:
            pass

        assert not isinstance(BadStore(), DataStore)


class TestFileStoreProtocol:
    def test_valid_implementation(self):
        class MyFileStore:
            @property
            def root(self) -> Path:
                return Path("/tmp")

        store = MyFileStore()
        assert store.root == Path("/tmp")


class TestSettingsProviderProtocol:
    def test_valid_implementation(self):
        class MySettings:
            VERSION: ClassVar[str] = "1.0"

            def get(self, key, default=None):
                return default

            @property
            def cloud_demo_profile(self) -> str:
                return "test"

        assert isinstance(MySettings(), SettingsProvider)

    def test_invalid_implementation_missing_get(self):
        class BadSettings:
            VERSION: ClassVar[str] = "1.0"

            @property
            def cloud_demo_profile(self) -> str:
                return "test"

        assert not isinstance(BadSettings(), SettingsProvider)


class TestApiClientProtocol:
    def test_valid_implementation(self):
        class MyClient:
            BASE_URL: ClassVar[str] = "https://test.com"

            def is_available(self) -> bool:
                return True

            async def parse_pdf(self, pdf_path, **kwargs):
                return {}

        client = MyClient()
        assert isinstance(client, ApiClient)

    def test_is_available_returns_false(self):
        class OfflineClient:
            BASE_URL: ClassVar[str] = "https://offline.com"

            def is_available(self) -> bool:
                return False

            async def parse_pdf(self, pdf_path, **kwargs):
                return {}

        client = OfflineClient()
        assert isinstance(client, ApiClient)
        assert client.is_available() is False


class TestLogEntryProtocol:
    def test_valid_implementation(self):
        class MyLog:
            def to_dict(self) -> dict:
                return {"key": "value"}

        assert isinstance(MyLog(), LogEntry)

    def test_to_dict_content(self):
        class MyLog:
            def __init__(self, data):
                self._data = data

            def to_dict(self) -> dict:
                return self._data

        log = MyLog({"event": "test", "level": "info"})
        assert isinstance(log, LogEntry)
        assert log.to_dict() == {"event": "test", "level": "info"}


class TestLockManagerProtocol:
    def test_valid_implementation(self):
        class MyLock:
            def acquire(self, name, timeout=10.0):
                return True

            def release(self, name):
                pass

        lock = MyLock()
        assert lock.acquire("resource") is True

    def test_acquire_failure(self):
        class BusyLock:
            def acquire(self, name, timeout=10.0):
                return False

            def release(self, name):
                pass

        lock = BusyLock()
        assert lock.acquire("resource") is False


class TestProtocolAll:
    """Verify __all__ exports the expected protocol names."""

    def test_all_exports(self):
        expected = {
            "ApiClient",
            "DataStore",
            "FileStore",
            "LockManager",
            "LogEntry",
            "SettingsProvider",
        }
        from mineru_core.protocols import __all__ as protocol_all

        assert set(protocol_all) == expected
