#!/usr/bin/env python3
"""生成评分者访问 code 清单 raters.json。

code 是访问凭证：不烘进镜像、不写日志、运行时只读挂载注入。
格式：{"items": [{"code": "..."}], "n": N}
"""

import argparse
import json
import secrets
import string
import sys
from pathlib import Path

# code 字符集：去掉易混淆的 0/O, 1/l/I
_ALPHABET = string.ascii_letters + string.digits
_ALPHABET = _ALPHABET.replace("0", "").replace("O", "").replace("1", "").replace("l", "").replace("I", "")

# code 长度：16 字符，约 95 bits 熵（52^16 ≈ 4.5e27）
_CODE_LENGTH = 16


def generate_code() -> str:
    """生成一个随机访问 code。"""
    return "".join(secrets.choice(_ALPHABET) for _ in range(_CODE_LENGTH))


def make_raters(n: int, output: Path, *, force: bool = False) -> int:
    """生成 n 个评分者 code，写入 raters.json。

    Returns:
        0 on success, 1 on failure.
    """
    if n < 1:
        print(f"错误：n 必须 >= 1，收到 {n}", file=sys.stderr)
        return 1

    if output.exists() and not force:
        print(f"错误：{output} 已存在，使用 --force 覆盖", file=sys.stderr)
        return 1

    codes = set()
    items = []
    while len(items) < n:
        code = generate_code()
        if code not in codes:
            codes.add(code)
            items.append({"code": code})

    doc = {"items": items, "n": n}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    print(f"已生成 {output}: {n} 个 code")
    print(f"  字符集: {len(_ALPHABET)} 字符（去掉 0/O/1/l/I）")
    print(f"  长度: {_CODE_LENGTH} 字符")
    print(f"  熵: ~{len(_ALPHABET) ** _CODE_LENGTH:.2e} 种可能")
    return 0


def main():
    p = argparse.ArgumentParser(description="生成评分者访问 code 清单")
    p.add_argument("n", type=int, help="生成 code 数量")
    p.add_argument("-o", "--output", type=Path, default=Path("secrets/raters.json"),
                   help="输出路径（默认 secrets/raters.json）")
    p.add_argument("--force", action="store_true",
                   help="强制覆盖已存在的文件")
    args = p.parse_args()
    sys.exit(make_raters(args.n, args.output, force=args.force))


if __name__ == "__main__":
    main()
