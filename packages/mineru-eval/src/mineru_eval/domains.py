"""Domain registry — parameterized domain discovery for evaluation pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DomainConfig:
    """Configuration for an evaluation domain."""
    name: str
    dimensions: list[str]
    default_difficulty_levels: list[str] = field(default_factory=lambda: ["L1", "L2", "L3", "L4"])
    scorer_module: str = ""       # e.g. "mineru_eval.control.scorers"
    prompt_module: str = ""       # e.g. "mineru_eval.control.prompts"
    target_system_prompt: str = ""
    default_input: str = ""       # relative to ROOT
    default_output: str = ""


class DomainRegistry:
    """Global registry for evaluation domains.

    Domains register themselves via their __init__.py (e.g. control/__init__.py).
    """
    _domains: dict[str, DomainConfig] = {}

    @classmethod
    def register(cls, config: DomainConfig) -> None:
        cls._domains[config.name] = config

    @classmethod
    def get(cls, name: str) -> DomainConfig:
        if name not in cls._domains:
            available = ", ".join(sorted(cls._domains)) or "(none)"
            raise KeyError(f"Unknown domain '{name}'. Registered: {available}")
        return cls._domains[name]

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._domains)

    @classmethod
    def all(cls) -> dict[str, DomainConfig]:
        return dict(cls._domains)
