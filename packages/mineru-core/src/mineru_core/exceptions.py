"""MinerU-core — 统一异常层次结构。

所有 mineru-* 包应通过此层次抛出标准化的异常，
而不是直接使用 ValueError / RuntimeError / bare except:。
"""


class SciAlignError(Exception):
    """全部自定义异常的基类。"""


class ConfigError(SciAlignError):
    """配置错误 — 环境变量缺失、配置无效、profile 不匹配。"""


class PipelineError(SciAlignError):
    """管线处理错误 — 数据管线（pipeline）中的处理失败。"""


class EvaluationError(SciAlignError):
    """评测错误 — 评测流程中的失败（judge、scorer、评估脚本）。"""


class MedicalRAGError(SciAlignError):
    """医学 RAG 错误 — 检索、合成中的失败。"""


class DataIntegrityError(SciAlignError):
    """数据完整性错误 — JSON schema 不匹配、数据缺失、校验失败。"""


class ApiClientError(SciAlignError):
    """外部 API 客户端错误 — SciVerse / DeepSeek / Ollama 调用失败。"""


__all__ = [
    "ApiClientError",
    "ConfigError",
    "DataIntegrityError",
    "EvaluationError",
    "MedicalRAGError",
    "PipelineError",
    "SciAlignError",
]
