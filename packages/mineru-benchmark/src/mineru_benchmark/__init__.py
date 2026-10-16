"""MinerU Benchmark — thin compatibility layer after Phase 3拆包.

Re-exports from:
  - mineru-eval  → mineru_eval
  - mineru-agent → mineru_agent
  - mineru-generator → mineru_generator

Pipeline tools and root-level modules remain in this package.
"""

# Re-export agent sub-package — symbols available as mineru_benchmark.<symbol>
from mineru_agent import (
    IntentRegistry,
    IntentRouter,
    Executor,
    Verifier,
    PlanStep,
    VerifyResult,
    ControlSciAgentCLI,
    build_parser,
    main,
)

# Re-export eval sub-package — enables ``from mineru_benchmark import mineru_eval``
import mineru_eval  # noqa: F401

# Re-export generator sub-package — enables ``from mineru_benchmark import mineru_generator``
import mineru_generator  # noqa: F401

__all__ = [
    "IntentRegistry",
    "IntentRouter",
    "Executor",
    "Verifier",
    "PlanStep",
    "VerifyResult",
    "ControlSciAgentCLI",
    "build_parser",
    "main",
]
