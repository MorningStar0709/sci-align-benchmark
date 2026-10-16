"""ControlSci Data Agent CLI — thin re-export layer (Phase 4.8).

All implementation logic has been split into ``mineru_benchmark.agent.*``:
  - ``utils.py`` — subprocess runner, JSON extraction, path helpers
  - ``intent_registry.py`` — PlanStep, VerifyResult, IntentRegistry
  - ``intent_router.py`` — IntentRouter (LLM few-shot → intent plan)
  - ``verifier.py`` — Verifier (LLM-based step verification)
  - ``executor.py`` — Executor + 15 dispatch handlers
  - ``cli.py`` — ControlSciAgentCLI + argparse entry point

This file is kept for backward compatibility. New code should import directly
from ``mineru_benchmark.agent``.
"""

from mineru_agent import (
    ControlSciAgentCLI,
    IntentRegistry,
    IntentRouter,
    Executor,
    Verifier,
    PlanStep,
    VerifyResult,
)

from mineru_agent.cli import build_parser, main
from mineru_agent.utils import (
    VERSION,
    _configure_stdio,
    _python_cmd,
    _extract_json_block,
    _get_capabilities_path,
    _get_logs_dir,
    _run_subprocess,
    _safe_cmd,
    _setup_logging,
)

if __name__ == "__main__":
    main()
