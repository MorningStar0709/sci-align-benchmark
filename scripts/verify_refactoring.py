"""快速验证架构重构后的 10 包导入链。

Usage:
    .venv/bin/python scripts/verify_refactoring.py

Expected output: "ALL 10 PACKAGES VERIFIED SUCCESSFULLY"
"""

import ast
import importlib
import sys
from pathlib import Path


PACKAGES = ["mineru_core", "mineru_api", "mineru_cli", "mineru_pipeline", "mineru_benchmark", "mineru_medical", "mineru_tools", "mineru_eval", "mineru_agent", "mineru_generator"]


def _typing_only_nodes(tree: ast.AST) -> set[int]:
    """收集 `if TYPE_CHECKING:` 块内的导入节点 id —— 它们不产生运行期依赖。"""
    nodes: set[int] = set()
    for stmt in ast.walk(tree):
        if not isinstance(stmt, ast.If):
            continue
        test = stmt.test
        is_guard = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )
        if is_guard:
            for child in ast.walk(stmt):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    nodes.add(id(child))
    return nodes


def check_import_symbols(src_dir: Path) -> list[str]:
    """校验 src_dir 下源码里所有绝对 `mineru_*` from-import 的符号可达性。

    CHECK_IMPORTS 那张手工表只覆盖被点名的子模块；本函数覆盖全部源码引用面，
    专抓 `from mineru_api.medical_rag import ...` 这类模块已随重构消失的断链。
    """
    problems: list[str] = []
    for py in sorted(Path(src_dir).rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        typing_only = _typing_only_nodes(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                continue
            if not node.module.startswith("mineru") or id(node) in typing_only:
                continue
            where = f"{py.relative_to(src_dir)}:{node.lineno}"
            try:
                mod = importlib.import_module(node.module)
            except Exception as exc:
                problems.append(f"[MISS] {where} import {node.module}: {exc}")
                continue
            for alias in node.names:
                if alias.name == "*" or hasattr(mod, alias.name):
                    continue
                # from pkg import submodule 不会被 hasattr 认出，补一次按模块名解析
                try:
                    importlib.import_module(f"{node.module}.{alias.name}")
                except ImportError:
                    problems.append(f"[MISS] {where} {node.module}.{alias.name} 不可达")
    return problems

CHECK_IMPORTS = {
    "mineru_core": [
        ("paths", ["PROJECT_ROOT"]),
        ("settings", ["get_settings"]),
        ("runtime", ["RuntimeConfig"]),
        ("utils", ["load_json", "write_json_atomic", "append_jsonl"]),
        ("log_schema", ["ExecutionLog", "LogStep"]),
        ("constants", ["VERSION", "CORE_9"]),
    ],
    "mineru_api": [
        ("app", ["app"]),
        ("schemas.demo", ["AgentPlanRequest", "Track1ParseRequest"]),
        ("schemas.medical", ["SynthesizeRequest", "AskRequest"]),
        ("schemas.common", ["StatusResponse"]),
        ("schemas.cloud_demo", ["UrlParseRequest", "QuizGenerateRequest"]),
        ("services.demo_utils", ["file_ok", "PROJECT_ROOT"]),
        ("services.mineru_client", ["MinerUClient"]),
        ("services.mineru_official", ["MinerUOfficialResult"]),
        ("services.cloud_demo", ["validate_public_extract_url", "public_boundary"]),
        ("services.deepseek_client", ["call_deepseek"]),
        ("services.quota", ["quota_status"]),
        ("controllers.health", ["router"]),
        ("controllers.demo", ["router"]),
        ("controllers.medical_rag", ["router"]),
        ("controllers.cloud_demo", ["router"]),
    ],
    "mineru_cli": [
        ("app", ["app", "main"]),
        ("commands.common", ["console", "PROJECT_ROOT", "fail", "print_json"]),
        ("commands.config", ["app"]),
        ("commands.demo", ["app"]),
        ("commands.health", ["app"]),
        ("commands.track1", ["app"]),
        ("commands.track2", ["app"]),
        ("commands.track3", ["app"]),
        ("commands.run", ["app"]),
        ("commands.trace", ["app"]),
    ],
    "mineru_medical": [
        ("indexing", ["load_manifest", "search_hybrid"]),
        ("synthesis", ["synthesize_evidence"]),
        ("embedding_providers", ["read_index_metadata"]),
        ("terminology", ["get_terminology_manager"]),
        ("public_demo", ["retrieve_public_medical_chunks", "synthesize_medical_answer"]),
    ],
    "mineru_pipeline": [
        (None, []),  # package-level import
        ("build_corpus", []),
        ("chunk_corpus", []),
        ("complete_metadata", []),
        ("clean_formulas", []),
        ("clean_tables", []),
        ("check_contamination", []),
        ("derive_formula_set", []),
        ("derive_long_reasoning_set", []),
        ("extract_stats", []),
        ("stats_corpus", []),
        ("trace_source_ref", []),
        ("verify_order", []),
        ("retry_arxiv", []),
        ("markdown_io", []),
        ("merge_t1_checkpoints", []),
        ("run_all", []),
        # launcher is script-style, excluded from import check
        ("build_metadata_light", []),
    ],
    "mineru_benchmark": [
        (None, []),  # package-level import
        ("pipeline", []),
        ("pipeline.build_benchmark", []),
        ("pipeline.arbiter", []),
        ("pipeline.export_dataset", []),
        ("pipeline.validate_benchmark", []),
        ("pipeline.merge_benchmarks", []),
        ("pipeline.split_benchmark", []),
        # root-level modules (thin compat layer)
        ("content_audit", []),
        ("leaderboard", []),
        ("coverage", []),
        ("cross_source_eval", []),
        ("medical_closure", []),
        ("medical_expansion", []),
        # qlora_probe/train and medbench_wrapper depend on external 'sciverse'/
        # 'mineru_medical.kb_quality' packages — skipped in env-agnostic check
        # visual_download depends on external 'sciverse' package — skipped
    ],
    "mineru_eval": [
        (None, []),
        ("evaluate", []),
        ("judge", []),
        ("cross_judge", []),
        ("report", []),
        ("client_factory", []),
    ],
    "mineru_agent": [
        (None, []),
        ("intent_registry", []),
        ("intent_router", []),
        ("verifier", []),
        ("executor", []),
    ],
    "mineru_generator": [
        (None, []),
        ("api", []),
        ("prompts", []),
        ("distribution", []),
        ("quality", []),
    ],
    "mineru_tools": [
        (None, []),
    ],
}


def main() -> int:
    errors = 0
    src_root = Path(__file__).resolve().parents[1] / "packages"
    for pkg in PACKAGES:
        try:
            importlib.import_module(pkg)
            print(f"[OK] {pkg}")
        except ImportError as e:
            # 只 catch ModuleNotFoundError 会让一个缺依赖的包中止整轮检查；
            # 这里记下并继续查其余包，另用 [ENV] 前缀区分"环境不全"与"架构断链"
            print(f"[ENV] {pkg}: {e}")
            errors += 1
            continue

        for submodule, symbols in CHECK_IMPORTS.get(pkg, []):
            if submodule is None:
                continue
            full_name = f"{pkg}.{submodule}"
            try:
                mod = importlib.import_module(full_name)
                for sym in symbols:
                    if not hasattr(mod, sym):
                        print(f"  [MISS] {full_name}.{sym}")
                        errors += 1
                if symbols:
                    print(f"  [OK] {full_name}")
            except Exception as e:
                print(f"  [FAIL] {full_name}: {e}")
                errors += 1

    for src in sorted(src_root.glob("*/src")):
        problems = check_import_symbols(src)
        if problems:
            print(f"[CROSS-IMPORT FAIL] {src.parent.name}: {len(problems)} 处")
            for problem in problems:
                print(f"  {problem}")
            errors += len(problems)
        else:
            print(f"[CROSS-IMPORT OK] {src.parent.name}")

    if errors:
        print(f"\n{'='*50}")
        print(f"[FAILURE] {errors} errors found")
        return 1
    else:
        print(f"\n{'='*50}")
        print("[SUCCESS] ALL 10 PACKAGES VERIFIED SUCCESSFULLY")
        return 0


if __name__ == "__main__":
    sys.exit(main())
