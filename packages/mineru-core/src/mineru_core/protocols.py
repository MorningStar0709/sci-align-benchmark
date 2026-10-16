"""mineru-core — 包间接口协议定义。

此模块定义了 mineru-core 与其他包之间的正式接口契约（Protocol）。
所有跨包依赖应通过此模块定义的接口进行解耦。
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable


# ── 数据持久化接口 ───────────────────────────────────────────────


@runtime_checkable
class DataStore(Protocol):
    """数据读写接口。

    实现类: mineru_core.utils (load_json, write_json_atomic, append_jsonl)
    消费者: mineru-benchmark, mineru-pipeline, mineru-cli
    """

    def load_json(self, path: str | Path) -> dict | list:
        """从 JSON 文件加载数据。"""
        ...

    def write_json_atomic(self, path: str | Path, data: dict | list) -> None:
        """原子写入 JSON 文件（先写临时文件再 rename）。"""
        ...

    def append_jsonl(self, path: str | Path, item: dict) -> None:
        """追加一行 JSONL。"""
        ...


# ── 文件/路径接口 ────────────────────────────────────────────────


@runtime_checkable
class FileStore(Protocol):
    """项目文件路径提供者。

    提供者: mineru_core.paths.PROJECT_ROOT
    """

    @property
    def root(self) -> Path:
        """项目根目录路径。"""
        ...


# ── 配置接口 ──────────────────────────────────────────────────────


@runtime_checkable
class SettingsProvider(Protocol):
    """配置提供者接口。

    实现类: mineru_core.settings.get_settings()
    消费者: mineru-api, mineru-medical, mineru-cli
    """

    VERSION: ClassVar[str]

    def get(self, key: str, default=None):
        """获取配置值。"""
        ...

    @property
    def cloud_demo_profile(self) -> str:
        """当前部署 profile。"""
        ...


# ── 日志接口 ──────────────────────────────────────────────────────


@runtime_checkable
class LogEntry(Protocol):
    """日志条目接口。

    实现类: mineru_core.log_schema.ExecutionLog, LogStep
    """

    def to_dict(self) -> dict:
        """序列化为字典。"""
        ...


# ── 外部 API 客户端接口 ──────────────────────────────────────────


@runtime_checkable
class ApiClient(Protocol):
    """外部 API 客户端接口。

    实现类: mineru_core.mineru_client.SciverseClient
    消费者: mineru-api, mineru-benchmark
    """

    BASE_URL: ClassVar[str]

    def is_available(self) -> bool:
        """检查服务是否可用。"""
        ...

    async def parse_pdf(self, pdf_path: str | Path, **kwargs) -> dict:
        """解析 PDF 文件。"""
        ...


# ── 数据库锁接口 ──────────────────────────────────────────────────


@runtime_checkable
class LockManager(Protocol):
    """分布式/文件锁接口。

    实现类: mineru_core.utils 中的各类锁实现
    """

    def acquire(self, name: str, timeout: float = 10.0) -> bool:
        """获取锁。"""
        ...

    def release(self, name: str) -> None:
        """释放锁。"""
        ...


__all__ = [
    "ApiClient",
    "DataStore",
    "FileStore",
    "LockManager",
    "LogEntry",
    "SettingsProvider",
]
