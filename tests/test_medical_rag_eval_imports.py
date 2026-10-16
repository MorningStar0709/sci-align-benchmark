"""Linux 化 Task 5：medical_rag_eval 的跨包 import 断链守卫。

`mineru_api.medical_rag` 已随 API 层 controllers/services 重构消失，评测侧残留引用会在
`--use-api` 分支触发 ModuleNotFoundError；其中 5 个符号可 repath 到 rag_synthesis，
2 个（查询改写与多路融合）无后继实现，对应分支判死。
"""

import subprocess

import pytest


def test_api_search_case_fails_explicitly_not_import_error():
    from mineru_eval.medical_rag_eval import _api_search_case

    with pytest.raises(RuntimeError, match="停用"):
        _api_search_case(None, "idx", {}, [], 5)


def test_no_dead_module_import_in_eval_package():
    """守卫的是 import 语句本身；停用文案里提到该模块名是允许且有信息量的。"""
    out = subprocess.run(
        ["grep", "-rnE", r"(from|import) +mineru_api\.medical_rag", "packages/mineru-eval/src"],
        capture_output=True,
        text=True,
    ).stdout
    assert out.strip() == "", out


def test_repath_targets_are_importable():
    from mineru_api.services.rag_synthesis import (  # noqa: F401
        _build_evidence_cards,
        _build_rag_prompt,
        _call_local_llm_chat,
        _classify_question_profile,
        _normalize_rag_answer,
    )
