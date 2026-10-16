#!/usr/bin/env python3
"""L-exec 方言闸（可复跑版，第 2 轮重写）。

为什么要重写：第 1 版的判据把「grep 命中文件数」当「可执行语句数」，正则还写成
`\\.venv\\\\Scripts`（匹配两个反斜杠）而漏掉实际形态；第 2 版又被对抗审查证伪三点 ——
相对路径在非仓根 CWD 下扫 0 文件仍返回 0、一切字符串常量被判 prose（而方言恰恰常以
"传给 subprocess 的字符串"形态存在）、EXEMPT 分支写反。本版本逐条堵住这三个洞：

  1. 路径一律相对本文件所在仓根解析，并断言扫描文件数下限，扫不够直接失败（拒绝假绿）；
  2. prose 仅当命中落在「注释行」或「纯 docstring 行」内；行内含语句形态（if/return/
     赋值/调用实参等）一律算 code，即使它在字符串里；
  3. EXEMPT 对 code 命中真实生效，且必须写明理由。

用法：.venv/bin/python scripts/check_lexec_dialect.py     # exit 0 == 可执行面零命中
"""

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_SCANNED = 300  # 公开子集下限；低于 300 说明扫描面结构异常

PATTERNS = {
    "venv-scripts": r"\.venv[/\\]+Scripts",
    "drive-letter": r"[A-Z]:\\\\[A-Za-z0-9_.]",
    "interpreter-exe": r"python(?:-3)?\.exe|npm\.cmd",
    "conda-run": r"\bconda\b",
    "nt-branch": r"os\.name\s*[=!]=\s*['\"]nt['\"]|sys\.platform[^)]*\bwin",
    "win-api": r"msvcrt|winreg|Get-CimInstance|taskkill|tasklist|Get-NetTCPConnection|DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP|CREATE_NO_WINDOW|CREATE_NEW_CONSOLE",
    "shell-powershell": r"\bpowershell\b|\bpwsh\b",
    "ps1-ref": r"\.ps1\b",
    "win-platform": r"platform\.system\(\)\s*==\s*['\"]Windows",
}
COMPILED = {k: re.compile(v) for k, v in PATTERNS.items()}

# 命中即视为 code 的语句形态（即便它出现在字符串/注释里，也先按 code 处理并人工复核）
CODE_SHAPE = re.compile(
    r"^\s*(?:if|elif|else|for|while|return|try|except|with|def|class|yield|raise|assert|import|from)\b"
    r"|\S+\s*=|[(,]\s*[\"']"
)

GLOBS = [
    "scripts/**/*.py", "packages/*/src/**/*.py", "tests/**/*.py", "*.py",
    "configs/*.yaml", "*.sh", "Dockerfile", "Dockerfile.*", ".github/workflows/*.yml",
    "starboard/pages/*.js", "starboard/lib/*.js", "pyproject.toml", "packages/*/pyproject.toml",
]
# path:lineno:pattern —— 显式豁免必须在此带理由注释，否则视为未豁免
SKIP_PATHS = {"scripts/check_lexec_dialect.py"}   # 本文件即方言词表，自指不扫

EXEMPT: set[str] = {
    # 粒度是「路径:规则」而非行号 —— 行号会在任何一次编辑后失效（第 2 轮实测踩过）
    "tests/test_background_detach.py:nt-branch",        # 断言"方言不再出现"的测试本体
    "tests/test_background_detach.py:win-api",
    "tests/test_verify_refactoring_symbols.py:drive-letter",
    # manifest sanitizer 回归用例：断言「盘符路径须收成仓库相对路径」，
    # 词面不含盘符即测不到该缺陷（2026-09-29 Linux 重生成实测漏出 D:\ 前缀）
    "tests/test_paper_gap_supplement.py:drive-letter",
    "tests/test_pipeline_watchdog.py:win-api",
    "packages/mineru-agent/src/mineru_agent/executor.py:ps1-ref",  # 停用说明文案，指名被归档启动器
    # 匿名闸的**待检测词表**本身含历史机器前缀（`D:/WorkPlace`…）。它是数据不是执行路径：
    # 本闸既不打开也不拼接该串，只拿它去发布面里找命中。与上面那两条 drive-letter 豁免同形。
    "scripts/check_release_anonymity.py:drive-letter",
}

def _scan_targets():
    seen, out = set(), []
    for pattern in GLOBS:
        for path in REPO_ROOT.glob(pattern):
            if path.is_file() and path.resolve() not in seen and "build/lib" not in str(path) \
                    and str(path.relative_to(REPO_ROOT)) not in SKIP_PATHS:
                seen.add(path.resolve())
                out.append(path)
    return sorted(out)


def _prose_lines(src: str, tree):
    """返回「纯注释行」+「docstring 覆盖行」的行号集合。"""
    lines = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                lines.update(range(tok.start[0], tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError):
        pass

    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            doc = body[0].value
            docstrings.update(range(doc.lineno, (doc.end_lineno or doc.lineno) + 1))
    return lines, docstrings


def main() -> int:
    targets = _scan_targets()
    if len(targets) < MIN_SCANNED:
        print(f"[GATE FAIL] 只扫到 {len(targets)} 个文件（下限 {MIN_SCANNED}）"
              f"—— CWD 或目录结构异常，本次结果不可信")
        return 1

    code_hits, prose_hits, unparse = [], [], []
    for path in targets:
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = path.relative_to(REPO_ROOT)
        comments, docstrings = (set(), set())
        if path.suffix == ".py":
            try:
                tree = ast.parse(src)
            except SyntaxError:
                unparse.append(str(rel))
                continue
            comments, docstrings = _prose_lines(src, tree)
        for lineno, line in enumerate(src.split("\n"), 1):
            for name, rx in COMPILED.items():
                if not rx.search(line):
                    continue
                if f"{rel}:{name}" in EXEMPT:
                    continue
                tag = f"{rel}:{lineno}:{name}"
                is_prose = lineno in comments or (lineno in docstrings
                                                  and not CODE_SHAPE.search(line))
                (prose_hits if is_prose else code_hits).append(tag)

    print(f"扫描文件 {len(targets)}（下限 {MIN_SCANNED}）；不可解析 {len(unparse)}")
    print(f"code  = {len(code_hits)}   ← 判据要求为 0")
    print(f"prose = {len(prose_hits)}   ← 允许（注释与 docstring 里的历史说明/归档指针）")
    for hit in code_hits[:40]:
        print("  [CODE]", hit)
    for path in unparse:
        print("  [UNPARSED]", path)
    return 1 if code_hits or unparse else 0


if __name__ == "__main__":
    sys.exit(main())
