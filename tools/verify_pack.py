#!/usr/bin/env python3
r"""Pack verifier (shipped inside the artifact as tools/verify_pack.py) — stdlib-only.

Verification scope (deliberate, see ARTIFACT.md): integrity plus *internal
consistency of the pack*. It does not claim end-to-end reproduction — the eval
pipeline and the judge API transcripts are not in the pack.

Chain, in order:
1. manifest.json exists and is self-consistent (on-disk files subset of manifest
   plus itself; every manifest entry present on disk);
2. per-file sha256 matches the manifest ledger;
3. every .json under benchmark/eval/ parses (reported count, reported failures);
4. the CHECK table (below).

Two check kinds:
- kind=self: recompute the mean of records[].score (4 dp) and compare against the
  file's own overall_score — the aggregate a reader would cite is re-derivable
  from per-item evidence.
- kind=projection: hash-match the raw file against the shipped projection's
  source_sha256 and compare overall_score — the aggregate projection really came
  from this raw file.

Usage:
    python3 tools/verify_pack.py --zip <pack.zip>
    python3 tools/verify_pack.py --dir <unpacked dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

CHECKS = (
    # A 面：v3 主线自洽（v3_phase4_300 的 run 件无投影，重算 ↔ 自报）
    {"kind": "self", "pack": "a",
     "raw": "benchmark/eval/v3_phase4_300/raw_qwen_run1.json"},
    {"kind": "self", "pack": "a",
     "raw": "benchmark/eval/v3_phase4_300/sft_baseline_run1.json"},
    {"kind": "self", "pack": "a",
     "raw": "benchmark/eval/v3_9b_300q/raw_9b_run1.json"},
    {"kind": "self", "pack": "a",
     "raw": "benchmark/eval/gemma4_300q/raw_gemma4_run1.json"},
    # B 面：n3 / run1_rejudge 投影对撞（sha + overall_score）
    {"kind": "projection", "pack": "b",
     "raw": "benchmark/eval/n3_regen/mgpo_eg_9b_gen2.json",
     "projection": "release/evidence/mgpo_eg_9b_gen2__n3_regen.json"},
    {"kind": "projection", "pack": "b",
     "raw": "benchmark/eval/n3_regen/raw_9b_gen2.json",
     "projection": "release/evidence/raw_9b_gen2__n3_regen.json"},
    {"kind": "projection", "pack": "b",
     "raw": "benchmark/eval/run1_rejudge/raw_9b_run1p.json",
     "projection": "release/evidence/raw_9b_run1p__run1_rejudge.json"},
)


def load_checks() -> tuple[dict, ...]:
    """CHECKS 表：默认内置；env PACK_VERIFY_CHECKS 指向 JSON 列表时替换（合成包测试用）。"""
    import os
    override = os.environ.get("PACK_VERIFY_CHECKS")
    if override:
        return tuple(json.loads(Path(override).read_text(encoding="utf-8")))
    return CHECKS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fail(msgs: list[str], text: str) -> None:
    msgs.append(text)
    print(f"[FAIL] {text}")


def verify(root: Path, checks: tuple[dict, ...] | None = None) -> list[str]:
    """返回失败消息列表；空 = 通过。"""
    msgs: list[str] = []
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return ["missing: manifest.json at pack root"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = manifest.get("files", {})

    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    for extra in sorted(on_disk - set(listed) - {"manifest.json"}):
        _fail(msgs, f"unlisted file (absent from manifest): {extra}")
    for rel in sorted(listed):
        target = root / rel
        if not target.is_file():
            _fail(msgs, f"missing: {rel} is listed in manifest but absent on disk")
            continue
        if _sha256(target) != listed[rel]:
            _fail(msgs, f"sha mismatch: {rel}")
    if msgs:
        return msgs

    parsed, bad = 0, []
    for p in sorted((root / "benchmark" / "eval").rglob("*.json")):
        try:
            json.loads(p.read_text(encoding="utf-8"))
            parsed += 1
        except (json.JSONDecodeError, UnicodeDecodeError):
            bad.append(p.relative_to(root).as_posix())
    for rel in bad:
        _fail(msgs, f"unparsable JSON: {rel}")
    print(f"[OK] manifest consistent: {len(listed)} entries; "
          f"benchmark/eval parsed {parsed} file(s)")

    pack_name = manifest.get("pack")
    for check in (checks if checks is not None else load_checks()):
        if check.get("pack") and check["pack"] != pack_name:
            print(f"[SKIP] {check['raw']} belongs to pack {check['pack']}, this is pack {pack_name}")
            continue
        raw_path = root / check["raw"]
        if not raw_path.is_file():
            _fail(msgs, f"missing: check raw {check['raw']} not in pack")
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        label = check["raw"]
        if check["kind"] == "self":
            records = raw.get("records", [])
            if not records:
                _fail(msgs, f"empty records, cannot recompute: {label}")
                continue
            recomputed = round(sum(r["score"] for r in records) / len(records), 4)
            claimed = raw.get("overall_score")
            if abs(recomputed - claimed) > 1e-9:
                _fail(msgs, f"MISMATCH {label}: recomputed {recomputed} != reported {claimed}")
            else:
                print(f"[OK] {label}: recomputed {recomputed} == reported {claimed}")
            continue
        proj_path = root / check["projection"]
        if not proj_path.is_file():
            _fail(msgs, f"missing: check projection {check['projection']} not in pack")
            continue
        proj = json.loads(proj_path.read_text(encoding="utf-8"))
        if _sha256(raw_path) != proj.get("source_sha256"):
            _fail(msgs, f"MISMATCH {label}: raw sha256 != projection source_sha256")
            continue
        p_score = proj.get("data", {}).get("overall_score")
        if abs(raw["overall_score"] - p_score) > 1e-9:
            _fail(msgs, f"MISMATCH {label}: raw {raw['overall_score']} != projection {p_score}")
        else:
            print(f"[OK] {label}: same-source sha, overall_score {p_score} matches")
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", help="解包后的目录")
    src.add_argument("--zip", help="ZIP 包（解到临时目录后验证）")
    args = ap.parse_args()

    if args.dir:
        msgs = verify(Path(args.dir))
    else:
        tmp = Path(tempfile.mkdtemp(prefix="pack_verify_"))
        try:
            with zipfile.ZipFile(args.zip) as zf:
                unsafe = [n for n in zf.namelist()
                          if Path(n).is_absolute() or ".." in Path(n).parts]
                if unsafe:
                    print(f"[FAIL] unsafe zip member(s) escape the extraction root: {unsafe[:3]}",
                          file=sys.stderr)
                    print("\n1 check(s) FAILED", file=sys.stderr)
                    return 1
                zf.extractall(tmp)
            msgs = verify(tmp)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    if msgs:
        for m in msgs:
            print(f"[FAIL] {m}", file=sys.stderr)
        print(f"\n{len(msgs)} check(s) FAILED", file=sys.stderr)
        return 1
    print("\nall checks passed — integrity and internal-consistency only "
          "(not an end-to-end reproduction claim)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
