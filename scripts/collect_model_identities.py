"""模型身份清点生成器：把「被评测的模型身份」从本机 eval 树提成仓内可核的清单。

为什么需要（10-01）：宣传卡 #7「81 个模型身份」的复算命令 glob 整棵
`benchmark/eval/**/*.json`，而该树**只有 193/1086 个文件入库** ⇒ 干净检出复算只得 **22**，
**59 个身份只出现在未跟踪件里**。与 194 h 同类：数据都在、都真，断的是「记录→别人能核」。
本件只提**身份名**（已剥机器前缀），不提任何题级内容。

口径（与宣传卡 #7 一致，勿另立）：
- 取每份 JSON **顶层** `model` 字段（不是 `records[].model`）；
- 剥掉机器前缀（`/root/scimonitor/`、`D:/WorkPlace/AI/SciMonitor/`、盘符），反斜杠归一；
- 去重后按字典序。**同一模型的多份产物只算一个身份**。

用法：
  .venv/bin/python scripts/collect_model_identities.py [--tree benchmark/eval] [--check]
退出码：0 = 写出/同代；1 = 不同代、树不在盘、或落点越界。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 同 surface_direction：匹配冻结数据里的历史绝对路径（含旧项目名），改名即失效。
MACHINE_PREFIX = re.compile(r"^(/root/scimonitor/|D:/WorkPlace/AI/SciMonitor/|[A-Za-z]:[\\/])")
DEFAULT_TREE = "benchmark/eval"
DEFAULT_OUT = "provenance/model_identities.json"


def _normalize(raw: str) -> str:
    return MACHINE_PREFIX.sub("", raw).replace("\\", "/").rstrip("/")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def collect(tree: Path, tracked_only: set[str] | None = None) -> list[str]:
    """`tree` 下所有 JSON 顶层 `model` 去重后的身份表；`tracked_only` 非空则只收其中之一。"""
    found: set[str] = set()
    for path in sorted(tree.rglob("*.json")):
        if tracked_only is not None and _rel(path) not in tracked_only:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("model"), str):
            found.add(_normalize(data["model"]))
    return sorted(found)


def tracked_eval_files(tree_rel: str) -> set[str]:
    proc = subprocess.run(["git", "ls-files", tree_rel], cwd=ROOT,
                          capture_output=True, text=True)
    return set((proc.stdout or "").split())


def main() -> int:
    parser = argparse.ArgumentParser(description="模型身份清点生成器")
    parser.add_argument("--tree", default=DEFAULT_TREE, help="评测树（默认 benchmark/eval）")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--check", action="store_true", help="只比对在盘件与现扫是否同代")
    args = parser.parse_args()

    out_path = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
    if not str(out_path).startswith(str(ROOT / "provenance")):
        print(f"错误：--out 必须落在 provenance/ 下，当前: {out_path}")
        return 1
    tree = ROOT / args.tree
    if not tree.is_dir():
        print(f"[FAIL] 评测树不存在：{tree}")
        return 1

    fresh = collect(tree)
    if args.check:
        if not out_path.is_file():
            print(f"[FAIL] 入仓件缺失：{out_path.relative_to(ROOT)}")
            return 1
        stored = json.loads(out_path.read_text(encoding="utf-8"))["identities"]
        if len(fresh) < len(stored):
            print(f"[SKIP] 本机树不完整（{len(fresh)} < {len(stored)}），无法判定同代")
            return 0
        if fresh != stored:
            print("[FAIL] 入仓件与本机现扫不同代（重跑不带 --check 即可刷新）")
            return 1
        print(f"[OK] 入仓件与现扫同代：{len(fresh)} 个身份")
        return 0

    tracked = tracked_eval_files(args.tree)
    body = {
        "source": f"{args.tree}/**/*.json 顶层 model 字段（剥机器前缀后去重）",
        "scanned_files": sum(1 for _ in tree.rglob("*.json")),
        "tracked_only_count": len(collect(tree, tracked_only=tracked)) if tracked else None,
        "count": len(fresh),
        "identities": fresh,
    }
    tmp = out_path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    print(f"已生成 {out_path.relative_to(ROOT)}：扫 {body['scanned_files']} 份 → "
          f"{body['count']} 个身份（仅已跟踪件只得 {body['tracked_only_count']}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
