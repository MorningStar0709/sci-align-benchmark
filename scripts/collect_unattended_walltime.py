"""无人值守墙钟入仓件生成器：把 `logs/*.log` 的 `[SUMMARY] 总耗时` 摘成可在仓内核的清单。

为什么需要（10-01）：对外那句「无人值守管线累计墙钟 194 h」原先只能靠
`logs/*.log`（0 个文件入库）＋ `models/**/*.log`（NFS symlink、仓外）复算 ⇒
本机能算、干净检出算不出。本件只提**摘要值**，不提整份日志，做法同 E11／e12_e4。

两条硬约束：
1. **只扫调用方给的目录**。原口头复算命令 `rglob("*.log")` 同时扫 `models/`——
   那是 NFS、GB 级，本仓禁用无过滤递归搜索（会把盘读爆、进程卡 `D` 态）。
   实测 `logs/` 单独就给出**同样**的 12 段 / 11637.1 min，`models/` 一趟没贡献任何值。
2. **按值去重**。同一次运行有 launch 与会话两份日志、`总耗时` 相同，直加会虚报 74%
   （实测 337.7 h vs 194.0 h）。

用法：
  .venv/bin/python scripts/collect_unattended_walltime.py            # 写 provenance/unattended_walltime.json
  .venv/bin/python scripts/collect_unattended_walltime.py --check    # 只比对，不写
退出码：0 = 写出/同代；1 = 不同代、无日志、或落点越界。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DURATION = re.compile(r"总耗时\s*([0-9][0-9.]*)\s*min")
DEFAULT_OUT = "provenance/unattended_walltime.json"


def scan(logs_dir: Path) -> dict:
    """扫该目录下的 `*.log`，取每个文件收官一次的 `总耗时`，再**按值去重**。"""
    per_file: dict[str, float] = {}
    for path in sorted(logs_dir.rglob("*.log")):
        values = DURATION.findall(path.read_text(encoding="utf-8", errors="ignore"))
        if values:
            per_file[path.name] = max(float(v) for v in values)
    by_value: dict[float, list[str]] = {}
    for name, minutes in per_file.items():
        by_value.setdefault(minutes, []).append(name)
    segments = [{"minutes": round(v, 1), "logs": sorted(names)}
                for v, names in sorted(by_value.items(), reverse=True)]
    total = round(sum(s["minutes"] for s in segments), 1)
    return {"source_dir": logs_dir.name,
            "scanned_files": len(per_file),
            "segments": segments,
            "total_minutes": total,
            "total_hours": round(total / 60, 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description="无人值守墙钟入仓件生成器")
    parser.add_argument("--logs", type=Path, default=ROOT / "logs",
                        help="日志目录（默认仓库根的 logs/；本件不碰 models/）")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--check", action="store_true", help="只比对在盘件与现扫是否同代")
    args = parser.parse_args()

    out_path = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
    if not str(out_path).startswith(str(ROOT / "provenance")):
        print(f"错误：--out 必须落在 provenance/ 下，当前: {out_path}")
        return 1
    if not args.logs.is_dir():
        print(f"[FAIL] 日志目录不存在：{args.logs}（干净检出没有 logs/，无法重扫）")
        return 1

    fresh = scan(args.logs)
    if args.check:
        if not out_path.is_file():
            print(f"[FAIL] 入仓件缺失：{out_path.relative_to(ROOT)}")
            return 1
        stored = json.loads(out_path.read_text(encoding="utf-8"))
        if [s["minutes"] for s in stored.get("segments", [])] != \
                [s["minutes"] for s in fresh["segments"]]:
            print("[FAIL] 入仓件与本机 logs 现扫不同代（重跑不带 --check 即可刷新）")
            return 1
        print(f"[OK] 入仓件与现扫同代：{len(fresh['segments'])} 段 / "
              f"{fresh['total_minutes']} min = {fresh['total_hours']} h")
        return 0

    tmp = out_path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(fresh, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    print(f"已生成 {out_path.relative_to(ROOT)}：{fresh['scanned_files']} 份日志 → "
          f"{len(fresh['segments'])} 段 / {fresh['total_minutes']} min = {fresh['total_hours']} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
