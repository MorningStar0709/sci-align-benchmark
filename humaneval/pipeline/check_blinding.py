#!/usr/bin/env python3
"""盲评隔离审计：验证渲染层真的拿不到 agent 分。

两道独立的门：
1. pool.json 不含 _DISPLAY_FORBIDDEN 字段（文件级检查）
2. app.py 的 _RENDER_KEYS / _RENDER_HTML 白名单不含 agent 分（代码级检查）

--self-test：注入 agent_score 并篡改白名单，确认两道门都会报。
"""

import argparse
import ast
import json
import sys
from pathlib import Path

# 与 server/app.py 的 _DISPLAY_FORBIDDEN 故意各写一份：审计侧与被审计侧共用
# 一个常量的话，一次改错就能同时拆掉两道门。
_DISPLAY_FORBIDDEN = frozenset({
    "agent_score", "score", "reason", "issues", "origin_id", "difficulty_level",
    "judge1_score", "judge2_score", "avg_score", "model", "judge_model",
    "model_source", "source_file", "source_ref",
})

# 期望的 _RENDER_KEYS / _RENDER_HTML（与 app.py 对齐）
_EXPECTED_RENDER_KEYS = ("item_id", "dimension")
_EXPECTED_RENDER_HTML = {"question": "question_html", "expected_answer": "expected_html",
                         "model_answer": "answer_html"}


def _check_pool_file(pool_path: Path) -> list[str]:
    """检查 pool.json 是否含禁止字段。"""
    errors = []
    if not pool_path.is_file():
        return [f"pool.json 不存在: {pool_path}"]

    doc = json.loads(pool_path.read_text(encoding="utf-8"))
    items = doc.get("items", [])
    for i, it in enumerate(items):
        hit = _DISPLAY_FORBIDDEN & set(it.keys())
        if hit:
            errors.append(f"第 {i} 项含禁止字段: {sorted(hit)}")
    return errors


def _check_app_code(app_path: Path) -> list[str]:
    """检查 app.py 的 _RENDER_KEYS / _RENDER_HTML 是否被篡改。"""
    errors = []
    if not app_path.is_file():
        return [f"app.py 不存在: {app_path}"]

    tree = ast.parse(app_path.read_text(encoding="utf-8"))

    # 提取 _RENDER_KEYS
    render_keys = None
    render_html = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "_RENDER_KEYS":
                        render_keys = ast.literal_eval(node.value)
                    elif target.id == "_RENDER_HTML":
                        render_html = ast.literal_eval(node.value)

    if render_keys is None:
        errors.append("未找到 _RENDER_KEYS 定义")
    elif tuple(render_keys) != _EXPECTED_RENDER_KEYS:
        errors.append(f"_RENDER_KEYS 被篡改: {render_keys} != {_EXPECTED_RENDER_KEYS}")

    if render_html is None:
        errors.append("未找到 _RENDER_HTML 定义")
    elif dict(render_html) != _EXPECTED_RENDER_HTML:
        errors.append(f"_RENDER_HTML 被篡改: {render_html} != {_EXPECTED_RENDER_HTML}")

    return errors


def _self_test(pool_path: Path, app_path: Path) -> list[str]:
    """自测试：注入 agent_score 并篡改白名单，确认两道门都会报。"""
    errors = []

    # 测试 1：pool.json 注入 agent_score 后应被检出
    doc = json.loads(pool_path.read_text(encoding="utf-8"))
    if doc.get("items"):
        doc["items"][0]["agent_score"] = 0.85
        injected = _check_pool_file_from_dict(doc)
        if not injected:
            errors.append("自测试失败：注入 agent_score 后 pool.json 检查未报")
        else:
            print(f"自测试 1 通过：注入 agent_score 被检出 ({len(injected)} 处)")

    # 测试 2：篡改 _RENDER_KEYS 后应被检出
    # 模拟：把 _RENDER_KEYS 改成包含 "agent_score"
    fake_app = app_path.read_text(encoding="utf-8").replace(
        '_RENDER_KEYS = ("item_id", "dimension")',
        '_RENDER_KEYS = ("item_id", "dimension", "agent_score")'
    )
    if fake_app == app_path.read_text(encoding="utf-8"):
        errors.append("自测试失败：无法模拟 _RENDER_KEYS 篡改")
    else:
        # 用临时文件测试
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(fake_app)
            fake_path = Path(f.name)
        try:
            tampered = _check_app_code(fake_path)
            if not tampered:
                errors.append("自测试失败：篡改 _RENDER_KEYS 后 app.py 检查未报")
            else:
                print(f"自测试 2 通过：篡改 _RENDER_KEYS 被检出 ({len(tampered)} 处)")
        finally:
            fake_path.unlink()

    return errors


def _check_pool_file_from_dict(doc: dict) -> list[str]:
    """从 dict 检查（供自测试用）。"""
    errors = []
    items = doc.get("items", [])
    for i, it in enumerate(items):
        hit = _DISPLAY_FORBIDDEN & set(it.keys())
        if hit:
            errors.append(f"第 {i} 项含禁止字段: {sorted(hit)}")
    return errors


def main():
    p = argparse.ArgumentParser(description="盲评隔离审计")
    p.add_argument("--pool", type=Path, default=Path("pool/pool.json"),
                   help="pool.json 路径")
    p.add_argument("--app", type=Path, default=Path("server/app.py"),
                   help="app.py 路径")
    p.add_argument("--self-test", action="store_true",
                   help="运行自测试：注入 agent_score 并篡改白名单")
    args = p.parse_args()

    all_errors = []

    # 门 1：pool.json 文件级检查
    print("门 1：检查 pool.json 是否含禁止字段...")
    errors = _check_pool_file(args.pool)
    if errors:
        all_errors.extend(errors)
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print(f"  ✓ pool.json 干净（{len(json.loads(args.pool.read_text()))} 题）")

    # 门 2：app.py 代码级检查
    print("门 2：检查 app.py 白名单是否被篡改...")
    errors = _check_app_code(args.app)
    if errors:
        all_errors.extend(errors)
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("  ✓ _RENDER_KEYS / _RENDER_HTML 与预期一致")

    # 自测试
    if args.self_test:
        print("自测试：注入 agent_score 并篡改白名单...")
        errors = _self_test(args.pool, args.app)
        if errors:
            all_errors.extend(errors)
            for e in errors:
                print(f"  ✗ {e}")
        else:
            print("  ✓ 两道门都能检出篡改")

    if all_errors:
        print(f"\n审计失败：{len(all_errors)} 处问题", file=sys.stderr)
        sys.exit(1)
    print("\n审计通过：盲评隔离完整")


if __name__ == "__main__":
    main()
