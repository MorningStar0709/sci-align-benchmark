"""反事实变体评测编排器 — 无熵门控 MGPO（gate≡1）× n=3（2 变体 × gen1-3 = 6 run）。

协议依据（docs/reports/prereg-counterfactual-mgpo.md 冻结判据）:
- 变体注册表: mgpo_cf（4B, models/v3_cot_sft/mgpo/final）、mgpo_cf_9b（9B, models/v3_9b_mgpo_cf/final）
- 评测命令与 run_n3_regen._build_cmd 冻结口径**逐字一致**（control 域分支），不得增删 flag
- 每变体独立生成 gen1-3（n=3），不复用 run1、不用 --answers-cache
- 对照组（mgpo_eg）不在此评测: 9B 用 n3_regen gen2-5 既有产物，4B 用 A800 全表段回流产物

机制层（复用 run_n3_regen 语义）:
- state 原子写 + gen 级 DoneMarker（断点续传，resume 幂等）
- 失败判定精确匹配（issues ∈ {"judge failure","target failure"}，非子串）
- 启动前 GPU 空闲断言（nvidia-smi VRAM 占用 > 2GB 则拒绝启动，防双任务争卡）
- 产物复核: complete + records 非空 + 无失败记录

产物: benchmark/eval/counterfactual/{key}_gen{N}.json
state: benchmark/eval/counterfactual/cf_state.json
日志: logs/cf_eval_{timestamp}.log

用法:
    .venv/bin/python scripts/run_cf_eval.py --dry-run
    .venv/bin/python scripts/run_cf_eval.py --smoke       # 每变体前 10 题单 gen
    .venv/bin/python scripts/run_cf_eval.py               # 全量 6 run（后台长任务）
    .venv/bin/python scripts/run_cf_eval.py --model mgpo_cf_9b
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_n3_regen import (  # noqa: E402
    CONTROL_DATASET,
    CONTROL_FIXED,
    MAX_CONSECUTIVE_FAILURES,
    PARALLEL,
    PYTHON,
    RETRIES,
)

# 冻结口径守护: 评测 flags 为预注册冻结值（不得随 n3_regen 常量漂移，
# 若上游调整此处直接报错，提示先审查预注册协议再同步）
_FROZEN = {"PARALLEL": 4, "RETRIES": 3, "MAX_CONSECUTIVE_FAILURES": 5}
for _name, _expect in _FROZEN.items():
    if globals()[_name] != _expect:
        raise RuntimeError(f"冻结口径漂移: run_n3_regen.{_name}={globals()[_name]} "
                           f"!= 预注册值 {_expect}——禁止评测，先审查预注册协议")

OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "eval" / "counterfactual"
STATE_FILE = OUTPUT_DIR / "cf_state.json"
LOG_DIR = PROJECT_ROOT / "logs"
# 单次生成超时（含 judge）: 4B ~3.9h、9B ~4.3h，300min 缓冲（同 run_n3_regen）
RUN_TIMEOUT_MIN = 300
# smoke 题数: 前 10 题（计划 Task 3 Step 2 冻结口径）
SMOKE_LIMIT = 10
# GPU 空闲断言阈值: VRAM 占用 > 2GB 拒绝启动（防与 run1 重判/云端回流冲突）
GPU_IDLE_MB = 2048

# ── 变体注册表（2 条）──
VARIANT_MODELS: list[dict] = [
    {"key": "mgpo_cf", "label": "MGPO-CF 4B", "adapter": "models/v3_cot_sft/mgpo/final",
     "base_model_id": "principled-intelligence/Qwen3.5-4B-text-only"},
    {"key": "mgpo_cf_9b", "label": "MGPO-CF 9B", "adapter": "models/v3_9b_mgpo_cf/final",
     "base_model_id": "Qwen/Qwen3.5-9B"},
]
MODEL_BY_KEY = {m["key"]: m for m in VARIANT_MODELS}
GEN_PLAN = [1, 2, 3]

try:
    from patrol import kill_tree
except ImportError:  # pytest 以 scripts.* 包式导入时
    from scripts.patrol import kill_tree



def _gen_path(model_key: str, gen: int, smoke: bool) -> Path:
    """产物路径（smoke 走独立 _smoke_ 前缀，不污染正式产物）。"""
    if smoke:
        return OUTPUT_DIR / f"_smoke_{model_key}_gen{gen}.json"
    return OUTPUT_DIR / f"{model_key}_gen{gen}.json"


def _smoke_fixed_path(fixed: Path, model_key: str, write: bool = True) -> Path:
    """从题集取前 SMOKE_LIMIT 题构造临时子集路径（smoke 专用）。

    write=False（dry-run）时只返回路径不落盘——dry-run 零副作用。
    """
    out = OUTPUT_DIR / f"_smoke_{model_key}.json"
    if not write:
        return out
    data = json.loads(fixed.read_text(encoding="utf-8"))
    questions = data if isinstance(data, list) else data.get("questions", [])
    subset = questions[:SMOKE_LIMIT]
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = subset if isinstance(data, list) else {**data, "questions": subset}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _build_cmd(m: dict, output: Path, smoke: bool, resume: bool = True,
               dry_run: bool = False) -> list[str]:
    """构建 eval_cli 命令 — 与 run_n3_regen._build_cmd 的 control 域分支逐字一致。

    冻结 flags: --dual-judge --all-dimensions-dual --parallel 4 --retries 3
    --max-consecutive-failures 5 --resume --retry-failed（不得增删）。
    """
    fixed = CONTROL_FIXED
    if smoke:
        fixed = _smoke_fixed_path(CONTROL_FIXED, m["key"], write=not dry_run)
    cmd = [
        str(PYTHON), "-u", "-m", "mineru_eval.eval_cli",
        "--input", str(CONTROL_DATASET),
        "--output", str(output),
        "--fixed-questions", str(fixed),
        "--dual-judge",
        "--all-dimensions-dual",
        "--parallel", str(PARALLEL),
        "--retries", str(RETRIES),
        "--max-consecutive-failures", str(MAX_CONSECUTIVE_FAILURES),
    ]
    if resume:
        cmd.append("--resume")
    cmd += ["--retry-failed"]
    cmd += ["--adapter-path", str(PROJECT_ROOT / m["adapter"]),
            "--base-model-id", m["base_model_id"]]
    return cmd


def _load_state() -> dict:
    """加载编排 state（损坏时按空处理并告警）。"""
    if STATE_FILE.is_file():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [WARNING] state 文件损坏（{e}），按空 state 处理", flush=True)
    return {}


def _save_state(state: dict) -> None:
    """原子持久化 state（崩溃不残留半截 JSON）。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_name(f"{STATE_FILE.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_FILE)


def _gpu_used_mb() -> int | None:
    """查询首卡显存占用（MB）；查询失败返回 None（调用方决定降级策略）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 — nvidia-smi 不可用时不阻断（eval_cli 自身会失败）
        return None


def _assert_gpu_idle() -> bool:
    """GPU 空闲断言: VRAM 占用 > 2GB 拒绝启动（防双任务争卡）。"""
    used = _gpu_used_mb()
    if used is None:
        print("  [WARN] nvidia-smi 查询失败，跳过 GPU 空闲断言", flush=True)
        return True
    if used > GPU_IDLE_MB:
        print(f"  [GPU-BUSY] VRAM 占用 {used} MB > {GPU_IDLE_MB} MB，拒绝启动"
              f"（GPU 被其他任务占用，等待释放后重试）", file=sys.stderr, flush=True)
        return False
    return True


def _has_failed_records(out: Path) -> bool:
    """产物中是否存在真实失败记录（精确匹配 issues ∈ {"judge failure","target failure"}）。

    非子串匹配: 'LLM judge response parse failure' 是重试成功后的批注痕迹，
    记录仍有有效分，不得误判为失败（2026-08-06 实测 n3_regen gen2 11/300 条此类记录）。
    """
    if not out.is_file():
        return False
    try:
        data = json.loads(out.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return True  # 产物无法解析视为未完成
    for r in data.get("records", []):
        for iss in r.get("issues", []):
            s = str(iss).strip().lower()
            if s in ("judge failure", "target failure"):
                return True
    return False


def _product_ok(out: Path) -> bool:
    """磁盘产物复核: 文件存在、可解析、complete=True、records=300、judge 双份字段齐全。"""
    if not out.is_file():
        return False
    try:
        data = json.loads(out.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(data, dict):
        return False  # 子集题集等非 eval 产物结构（eval 产物必为 dict）
    recs = data.get("records")
    if not (isinstance(recs, list) and len(recs) == 300):
        print(f"  [WARN] {out.name}: records={len(recs) if isinstance(recs, list) else 'n/a'}"
              f" != 300，产物不完整", flush=True)
        return False
    if not bool(data.get("complete")):
        return False
    # judge 双份字段抽查（dual-judge 生效性）
    first = recs[0]
    if not (isinstance(first.get("judge1_score"), (int, float))
            and isinstance(first.get("judge2_score"), (int, float))):
        print(f"  [WARN] {out.name}: 首条记录缺 judge1_score/judge2_score（dual-judge 未生效）",
              flush=True)
        return False
    print(f"  [CHECK] {out.name}: records=300, complete=True, dual-judge 双份字段 ✓", flush=True)
    return True


def _gen_done(ms: dict, gen: int) -> bool:
    """gen 是否已完成（state done 标记 + 磁盘产物双重确认）。"""
    st = ms.get(f"gen{gen}")
    if not (isinstance(st, dict) and bool(st.get("done"))):
        return False
    path = st.get("path")
    if not path:
        return False
    return _product_ok(Path(path))


def _run_eval(cmd: list[str], desc: str) -> bool:
    """执行 eval 子进程（超时 kill）；返回成功与否。"""
    print(f"\n{'=' * 60}", flush=True)
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {desc}", flush=True)
    print(f"  CMD: {' '.join(str(c) for c in cmd[:8])}…", flush=True)
    print(f"{'=' * 60}", flush=True)
    t0 = time.time()
    proc = None
    try:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
        proc.wait(timeout=RUN_TIMEOUT_MIN * 60)
        ok = proc.returncode == 0
        print(f"  {'[OK]' if ok else '[FAIL]'} {desc}: "
              f"{'成功' if ok else f'退出码 {proc.returncode}'} ({(time.time() - t0) / 60:.1f} min)",
              flush=True)
        return ok
    except subprocess.TimeoutExpired:
        if proc:
            leftovers = kill_tree(proc.pid)   # 只 kill 直接子进程会留孤儿占显存
            if leftovers:
                print(f"  [ERROR] 进程树未清干净，显存可能仍被占用: {leftovers}", flush=True)
            try:
                proc.wait(timeout=30)
            except (subprocess.TimeoutExpired, OSError):
                pass
        print(f"  [TIMEOUT] {desc}: 超过 {RUN_TIMEOUT_MIN} min 被终止（产物已增量落盘，可 resume）",
              flush=True)
        return False
    except KeyboardInterrupt:
        if proc:
            leftovers = kill_tree(proc.pid)   # 只 kill 直接子进程会留孤儿占显存
            if leftovers:
                print(f"  [ERROR] 进程树未清干净，显存可能仍被占用: {leftovers}", flush=True)
            try:
                proc.wait(timeout=30)
            except (subprocess.TimeoutExpired, OSError):
                pass
        print("\n  [ABORT] 用户中断 — 停止剩余生成", flush=True)
        raise
    except Exception as e:  # noqa: BLE001 — 子进程启动异常统一处理
        if proc:
            leftovers = kill_tree(proc.pid)   # 只 kill 直接子进程会留孤儿占显存
            if leftovers:
                print(f"  [ERROR] 进程树未清干净，显存可能仍被占用: {leftovers}", flush=True)
            try:
                proc.wait(timeout=30)
            except (subprocess.TimeoutExpired, OSError):
                pass
        print(f"  [ERROR] {desc}: {e}", flush=True)
        return False


def _run_model(m: dict, state: dict, smoke: bool, resume_eval: bool = True) -> bool:
    """执行单变体 gen1-3（gen 级 DoneMarker，逐 gen 持久化 state）。"""
    ms = state.setdefault(m["key"], {})
    label = m["label"]
    ok_all = True
    skipped_all = True
    for gen in GEN_PLAN:
        if _gen_done(ms, gen):
            print(f"  [SKIP] {label} gen{gen}: 已完成", flush=True)
            continue
        skipped_all = False
        out = _gen_path(m["key"], gen, smoke)
        # GPU 空闲复查（每 gen 启动前）: 24h+ 串行窗口内其他任务可能抢占 GPU
        if not _assert_gpu_idle():
            print(f"  [GPU-BUSY] {label} gen{gen}: GPU 被其他任务占用，本次运行中止"
                  f"（state 已保留，可 resume）", flush=True)
            return False
        cmd = _build_cmd(m, out, smoke, resume=resume_eval)
        ok = _run_eval(cmd, f"{label} — gen{gen}")
        if ok and smoke:
            # smoke 验证通过: 不标 done、不污染正式 state（正式运行会重新完整执行）
            print(f"  [SMOKE-OK] {label} gen{gen}: {SMOKE_LIMIT} 题验证通过（未写入正式 state）",
                  flush=True)
            continue
        if ok:
            if _has_failed_records(out):
                print(f"  [WARN] {label} gen{gen}: exit 0 但产物含失败记录，不标记完成"
                      f"（下次 resume 将重试）", file=sys.stderr, flush=True)
                ms[f"gen{gen}"] = {"failed": True, "path": str(out),
                                   "completed_at": datetime.now().isoformat()}
                ok_all = False
            else:
                ms[f"gen{gen}"] = {"done": True, "path": str(out),
                                   "completed_at": datetime.now().isoformat()}
        else:
            ms[f"gen{gen}"] = {"failed": True, "path": str(out),
                               "completed_at": datetime.now().isoformat()}
            ok_all = False
        _save_state(state)
    if smoke and skipped_all:
        # 正式产物已存在导致 smoke 空跑——显式提示，避免误判"smoke 通过"
        print(f"  [SMOKE-NOOP] {label}: 全部 gen 已被正式 state 标记完成，smoke 未验证任何题。"
              f"如需强制 smoke 请用 --no-resume（将覆盖正式产物）", flush=True)
    return ok_all


def main() -> int:
    parser = argparse.ArgumentParser(description="反事实变体评测编排器（2 变体 × gen1-3）")
    parser.add_argument("--model", default="", help="逗号分隔变体 key（默认全部）")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令清单，零副作用")
    parser.add_argument("--smoke", action="store_true", help="每变体仅前 10 题（临时子集题集）")
    parser.add_argument("--no-resume", action="store_true",
                        help="忽略 state 中已完成标记强制重跑（慎用，会覆盖产物）")
    args = parser.parse_args()

    # 终端显示规范: 行缓冲，后台/管道场景进度实时可见
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    # 终端双写: 日志文件 + 终端（长任务规范，后台运行日志可追溯）
    from _common import TeeLogger  # noqa: E402

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"cf_eval_{datetime.now():%Y%m%d_%H%M%S}.log"
    tee = TeeLogger(log_file)
    sys.stdout = tee
    print(f"[INFO] 日志: {log_file}", flush=True)

    # Python 解释器 pre-flight（缺失时提前报错，避免子进程启动失败语义不清）
    if not PYTHON.is_file():
        print(f"[ERROR] Python 解释器不存在: {PYTHON}", file=sys.stderr, flush=True)
        return 1

    keys = [k.strip() for k in args.model.split(",") if k.strip()] or list(MODEL_BY_KEY)
    unknown = [k for k in keys if k not in MODEL_BY_KEY]
    if unknown:
        parser.error(f"未知变体 key: {unknown}（可选: {', '.join(MODEL_BY_KEY)}）")

    # adapter 路径预检（真实执行缺失阻断——评测无法进行；dry-run 仅 WARN 不阻断，
    # 便于预览命令清单，训练产物可在 dry-run 之后补齐）
    for key in keys:
        m = MODEL_BY_KEY[key]
        if not (PROJECT_ROOT / m["adapter"]).is_dir():
            msg = f"{key} adapter 路径缺失: {m['adapter']}"
            if args.dry_run:
                print(f"  [WARN] {msg}", file=sys.stderr, flush=True)
            else:
                print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
                return 1

    print(f"\n[INFO] 反事实评测编排 — 变体数={len(keys)}, gen={GEN_PLAN}, "
          f"dry_run={args.dry_run}, smoke={args.smoke}")
    for key in keys:
        m = MODEL_BY_KEY[key]
        print(f"  {key:<12} {m['label']} adapter={m['adapter']}")

    # GPU 空闲断言（仅真实执行; smoke 也需 GPU 生成）
    if not args.dry_run and not _assert_gpu_idle():
        return 1

    if args.dry_run:
        print("\n[DRY-RUN] 执行清单:")
        for key in keys:
            m = MODEL_BY_KEY[key]
            for gen in GEN_PLAN:
                out = _gen_path(key, gen, args.smoke)
                cmd = _build_cmd(m, out, args.smoke, dry_run=True)
                print(f"\n  {m['label']} gen{gen}:")
                print(f"    {' '.join(str(c) for c in cmd)}")
        print("\n[DRY-RUN] 结束 — 未执行任何任务")
        return 0

    state = {} if args.no_resume else _load_state()
    failed: list[str] = []
    t_start = time.time()
    for key in keys:
        m = MODEL_BY_KEY[key]
        ok = _run_model(m, state, smoke=args.smoke, resume_eval=not args.no_resume)
        if not ok:
            failed.append(key)

    total_min = (time.time() - t_start) / 60
    print(f"\n  [SUMMARY] 总耗时 {total_min:.1f} min")
    if failed:
        print(f"  [FAIL] {len(failed)} 变体未全部完成: {', '.join(failed)}", flush=True)
        return 1
    print(f"  [DONE] 全部变体评测完成（state: {STATE_FILE}）")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[ABORT] 用户中断", file=sys.stderr, flush=True)
        raise SystemExit(130)
    except Exception as e:  # noqa: BLE001 — 顶层兜底，防裸栈崩溃
        print(f"\n[FATAL] 未预期异常: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise SystemExit(1)
