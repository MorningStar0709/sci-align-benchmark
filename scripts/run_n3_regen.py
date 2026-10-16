"""n=3 独立重生成编排器 — σg 标定段（2 模型 × 补 4 次至 n=5）+ 全表段（25 模型 × 补 2 次至 n=3）。

协议依据（docs/reports/strategy-topconf-assessment-n3-regen-protocol.md §3.5 定稿）:
- 每次生成 = 完整 run1 协议（GPU 生成 + dual judge + 仲裁），**不使用 --answers-cache**
- 旧 run1 直接复用为第 1 次生成，每模型仅新增 2 次（σg 标定模型新增 4 次）
- 执行顺序硬约束: 4B → 9B → Gemma/医疗（注册表定义顺序即执行顺序）
- 执行时机: teacher 矩阵收官后统一启动（本脚本不执行 GPU，仅编排 eval_cli 子进程）

用法:
    # σg 标定段 dry-run（2 模型 × 4 次）
    .venv/bin/python scripts/run_n3_regen.py --phase sigma --dry-run
    # 全表段 dry-run（23 模型 = 25 − 2 σg 标定模型 × 2 次）
    .venv/bin/python scripts/run_n3_regen.py --phase full --dry-run
    # σg 标定段真实执行（默认 resume）
    .venv/bin/python scripts/run_n3_regen.py --phase sigma
    # 单模型执行
    .venv/bin/python scripts/run_n3_regen.py --phase full --model mgpo_eg_9b
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
PYTHON = Path(sys.executable)
OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "eval" / "n3_regen"
STATE_FILE = OUTPUT_DIR / "n3_state.json"
LOG_DIR = PROJECT_ROOT / "logs"

from _common import TeeLogger  # noqa: E402

# ── 题集 ──
CONTROL_DATASET = PROJECT_ROOT / "benchmark" / "dataset" / "core_v3.json"
CONTROL_FIXED = PROJECT_ROOT / "benchmark" / "eval" / "rebuilt_quick_judge" / "fixed_300.json"
MEDICAL_DATASET = PROJECT_ROOT / "benchmark" / "dataset" / "medical_benchmark.json"
MEDICAL_FIXED = PROJECT_ROOT / "benchmark" / "eval" / "fixed_100_medical.json"

PARALLEL = 4
RETRIES = 3
MAX_CONSECUTIVE_FAILURES = 5
# 单次生成超时: 9B 实测 ~4.3-4.5h（含 judge），300min 余量不足会误杀慢 run；
# 对齐云端快照 run_timeout_min.full=900（2026-08-08 提档决策，kill 虽可 resume 但白损一轮）
RUN_TIMEOUT_MIN = 900
SMOKE_LIMIT = 2

# 旧 run1 产物目录（按 domain × batch，run1 复用检测）
RUN1_DIRS: dict[str, dict[str, Path]] = {
    "control": {
        "4b": PROJECT_ROOT / "benchmark" / "eval" / "v3_phase4_300",
        "9b": PROJECT_ROOT / "benchmark" / "eval" / "v3_9b_300q",
        "gemma": PROJECT_ROOT / "benchmark" / "eval" / "gemma4_300q",
    },
    "medical": {
        "4b": PROJECT_ROOT / "benchmark" / "eval" / "medical_100",
        "9b": PROJECT_ROOT / "benchmark" / "eval" / "medical_100_9b",
        "gemma": PROJECT_ROOT / "benchmark" / "eval" / "medical_100_gemma4",
    },
}

# run1' 重判产物目录（新 judge epoch；2026-08-07 起为 gen1 数据源首选）
REJUDGE_DIR = PROJECT_ROOT / "benchmark" / "eval" / "run1_rejudge"

# ── 25 模型注册表（key, label, adapter, domain, base_model_id, batch）──
# adapter 路径从既有评测 run1 产物 model 字段反推（2026-08-05 磁盘核实）
MODELS: list[dict] = [
    # ── control 4B（batch 1）──
    {"key": "raw_qwen", "label": "Raw Qwen3.5-4B", "adapter": None,
     "domain": "control", "base_model_id": "principled-intelligence/Qwen3.5-4B-text-only", "batch": "4b"},
    {"key": "sft_baseline", "label": "SFT Baseline 4B", "adapter": "models/v3_cot_sft/baseline/final",
     "domain": "control", "base_model_id": "principled-intelligence/Qwen3.5-4B-text-only", "batch": "4b"},
    {"key": "mgpo_eg", "label": "MGPO-EG 4B", "adapter": "models/v3_cot_sft/mgpo_eg/final",
     "domain": "control", "base_model_id": "principled-intelligence/Qwen3.5-4B-text-only", "batch": "4b"},
    {"key": "rlvr_c_v2", "label": "RLVR-C v2 4B", "adapter": "models/v3_cot_sft/rlvr_c_v2_4b_restored/final",
     "domain": "control", "base_model_id": "principled-intelligence/Qwen3.5-4B-text-only", "batch": "4b"},
    # ── control 9B（batch 2）──
    {"key": "raw_9b", "label": "Raw Qwen3.5-9B", "adapter": None,
     "domain": "control", "base_model_id": "Qwen/Qwen3.5-9B", "batch": "9b"},
    {"key": "sft_9b", "label": "SFT Baseline 9B", "adapter": "models/v3_9b_sft/final",
     "domain": "control", "base_model_id": "Qwen/Qwen3.5-9B", "batch": "9b"},
    {"key": "mgpo_eg_9b", "label": "MGPO-EG 9B", "adapter": "models/v3_9b_mgpo_eg/final",
     "domain": "control", "base_model_id": "Qwen/Qwen3.5-9B", "batch": "9b"},
    {"key": "mgpo_fixed_9b", "label": "MGPO-fixed 9B", "adapter": "models/v3_9b_mgpo_fixed/final",
     "domain": "control", "base_model_id": "Qwen/Qwen3.5-9B", "batch": "9b"},
    {"key": "rlvr_c_v2_9b", "label": "RLVR-C v2 9B", "adapter": "models/v3_9b_rlvr_c_v2/final",
     "domain": "control", "base_model_id": "Qwen/Qwen3.5-9B", "batch": "9b"},
    # ── control Gemma（batch 3）──
    {"key": "raw_gemma4", "label": "Raw Gemma-4-E4B", "adapter": None,
     "domain": "control", "base_model_id": "google/gemma-4-E4B-it", "batch": "gemma"},
    {"key": "sft_gemma4", "label": "SFT Gemma-4-E4B", "adapter": "models/gemma4_sft/final",
     "domain": "control", "base_model_id": "google/gemma-4-E4B-it", "batch": "gemma"},
    {"key": "mgpo_eg_gemma4", "label": "MGPO-EG Gemma-4-E4B", "adapter": "models/gemma4_mgpo_eg/final",
     "domain": "control", "base_model_id": "google/gemma-4-E4B-it", "batch": "gemma"},
    {"key": "rlvr_c_v2_gemma4", "label": "RLVR-C v2 Gemma-4-E4B", "adapter": "models/gemma4_rlvr_c_v2/final",
     "domain": "control", "base_model_id": "google/gemma-4-E4B-it", "batch": "gemma"},
    # ── medical 4B（batch 4）──
    {"key": "raw_qwen_med", "label": "Raw Qwen3.5-4B Med", "adapter": None,
     "domain": "medical", "base_model_id": "principled-intelligence/Qwen3.5-4B-text-only", "batch": "4b"},
    {"key": "medical_sft", "label": "Medical SFT 4B", "adapter": "models/v3_cot_sft/medical/baseline/final",
     "domain": "medical", "base_model_id": "principled-intelligence/Qwen3.5-4B-text-only", "batch": "4b"},
    {"key": "medical_mgpo_eg", "label": "Medical MGPO-EG 4B", "adapter": "models/v3_cot_sft/medical/mgpo_eg/final",
     "domain": "medical", "base_model_id": "principled-intelligence/Qwen3.5-4B-text-only", "batch": "4b"},
    {"key": "medical_rlvr_c_v2", "label": "Medical RLVR-C v2 4B", "adapter": "models/v3_cot_sft/medical/rlvr_c_v2/final",
     "domain": "medical", "base_model_id": "principled-intelligence/Qwen3.5-4B-text-only", "batch": "4b"},
    # ── medical 9B（batch 5）──
    {"key": "raw_9b_med", "label": "Raw Qwen3.5-9B Med", "adapter": None,
     "domain": "medical", "base_model_id": "Qwen/Qwen3.5-9B", "batch": "9b"},
    {"key": "medical_sft_9b", "label": "Medical SFT 9B", "adapter": "models/v3_9b_sft/medical/baseline/final",
     "domain": "medical", "base_model_id": "Qwen/Qwen3.5-9B", "batch": "9b"},
    {"key": "medical_mgpo_9b", "label": "Medical MGPO-EG 9B", "adapter": "models/v3_9b_sft/medical/mgpo_eg/final",
     "domain": "medical", "base_model_id": "Qwen/Qwen3.5-9B", "batch": "9b"},
    {"key": "medical_rl_9b", "label": "Medical RLVR-C 9B", "adapter": "models/v3_9b_sft/medical/rlvr_c_v2/final",
     "domain": "medical", "base_model_id": "Qwen/Qwen3.5-9B", "batch": "9b"},
    # ── medical Gemma（batch 6）──
    {"key": "raw_gemma4_med", "label": "Raw Gemma-4-E4B Med", "adapter": None,
     "domain": "medical", "base_model_id": "google/gemma-4-E4B-it", "batch": "gemma"},
    {"key": "medical_sft_gemma4", "label": "Medical SFT Gemma-4-E4B", "adapter": "models/gemma4_medical/sft/final",
     "domain": "medical", "base_model_id": "google/gemma-4-E4B-it", "batch": "gemma"},
    {"key": "medical_mgpo_gemma4", "label": "Medical MGPO-EG Gemma-4-E4B", "adapter": "models/gemma4_medical/mgpo_eg/final",
     "domain": "medical", "base_model_id": "google/gemma-4-E4B-it", "batch": "gemma"},
    {"key": "medical_rl_gemma4", "label": "Medical RLVR-C Gemma-4-E4B", "adapter": "models/gemma4_medical/rlvr_c_v2/final",
     "domain": "medical", "base_model_id": "google/gemma-4-E4B-it", "batch": "gemma"},
]

MODEL_BY_KEY = {m["key"]: m for m in MODELS}

# σg 标定段: 4B RLVR-C v2（控制）+ 9B MGPO-EG 各补 4 次生成至 n=5
SIGMA_MODEL_KEYS = ["rlvr_c_v2", "mgpo_eg_9b"]
# 全表段: 每模型补 2 次生成至 n=3；σg 标定模型在全表段跳过（已 n=5）
FULL_GENS = [2, 3]
SIGMA_GENS = [2, 3, 4, 5]

try:
    from patrol import kill_tree
except ImportError:  # pytest 以 scripts.* 包式导入时
    from scripts.patrol import kill_tree



def _run1_path(m: dict) -> Path:
    """旧 run1 产物路径（复用检测）。"""
    name = {
        "raw_qwen": "raw_qwen", "sft_baseline": "sft_baseline",
        "mgpo_eg": "mgpo_eg", "rlvr_c_v2": "rlvr_c_v2",
        "raw_9b": "raw_9b", "sft_9b": "sft_9b", "mgpo_eg_9b": "mgpo_eg_9b",
        "mgpo_fixed_9b": "mgpo_fixed_9b", "rlvr_c_v2_9b": "rlvr_c_v2_9b",
        "raw_gemma4": "raw_gemma4", "sft_gemma4": "sft_gemma4",
        "mgpo_eg_gemma4": "mgpo_eg_gemma4", "rlvr_c_v2_gemma4": "rlvr_c_v2_gemma4",
        "raw_qwen_med": "raw_qwen", "medical_sft": "medical_sft",
        "medical_mgpo_eg": "medical_mgpo_eg", "medical_rlvr_c_v2": "medical_rlvr_c_v2",
        "raw_9b_med": "raw_9b_med", "medical_sft_9b": "medical_sft_9b",
        "medical_mgpo_9b": "medical_mgpo_9b", "medical_rl_9b": "medical_rl_9b",
        "raw_gemma4_med": "raw_gemma4_med", "medical_sft_gemma4": "medical_sft_gemma4",
        "medical_mgpo_gemma4": "medical_mgpo_gemma4", "medical_rl_gemma4": "medical_rl_gemma4",
    }[m["key"]]
    return RUN1_DIRS[m["domain"]][m["batch"]] / f"{name}_run1.json"


def _run1p_path(m: dict) -> Path:
    """run1' 重判产物路径（新 judge epoch gen1 数据源，key 直命名）。"""
    return REJUDGE_DIR / f"{m['key']}_run1p.json"


def _gen_path(model_key: str, gen: int) -> Path:
    """新生成产物路径。"""
    return OUTPUT_DIR / f"{model_key}_gen{gen}.json"


def _out_path(model_key: str, gen: int, smoke: bool) -> Path:
    """产物路径（smoke 走独立 _smoke_ 前缀，不污染正式产物）。"""
    if smoke:
        return OUTPUT_DIR / f"_smoke_{model_key}_gen{gen}.json"
    return _gen_path(model_key, gen)


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


def _build_cmd(m: dict, output: Path, smoke: bool, resume: bool = True,
               dry_run: bool = False) -> list[str]:
    """构建 eval_cli 命令（不带 --answers-cache，n=3 与旧 run2/3 的本质区别）。

    smoke 时用临时子集题集（前 SMOKE_LIMIT 题），因为 --fixed-questions 会忽略 --limit。
    --retry-failed: resume 时重试失败题，防止 judge 瞬时故障以 score=0.0 静默混入。
    dry_run=True 时 smoke 子集不落盘（只展示路径，零副作用）。
    """
    if m["domain"] == "medical":
        dataset, fixed = MEDICAL_DATASET, MEDICAL_FIXED
        extra = ["--domain", "medical"]
    else:
        dataset, fixed = CONTROL_DATASET, CONTROL_FIXED
        extra = []
    if smoke:
        fixed = _smoke_fixed_path(fixed, m["key"], write=not dry_run)

    cmd = [
        str(PYTHON), "-u", "-m", "mineru_eval.eval_cli",
        "--input", str(dataset),
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
    cmd += ["--retry-failed"] + extra
    if m["adapter"] is None:
        cmd += ["--raw", "--base-model-id", m["base_model_id"]]
    else:
        cmd += ["--adapter-path", str(PROJECT_ROOT / m["adapter"]),
                "--base-model-id", m["base_model_id"]]
    return cmd


def _smoke_fixed_path(fixed: Path, model_key: str, write: bool = True) -> Path:
    """从题集取前 SMOKE_LIMIT 题构造临时子集路径（smoke 专用）。

    write=False（dry-run）时只返回路径不落盘——dry-run 零副作用，
    子集文件只在真实执行前生成。
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


def _has_failed_records(out: Path) -> bool:
    """产物中是否存在真实失败记录（judge failure / target failure）。

    eval_cli 在 --retry-failed 重试耗尽后仍会以 exit 0 落盘失败记录，
    编排层必须二次确认，否则 score=0.0 会静默混入 n=3 聚合。

    精确匹配（非子串）：issues 中的 'LLM judge response parse failure'
    是 judge 首轮响应解析失败、重试成功后保留的批注痕迹，记录仍有有效分，
    不得误判为失败（2026-08-06 实测 gen2 11/300 条此类记录，均带有效分）。
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


def _run_eval(cmd: list[str], desc: str, dry_run: bool) -> bool:
    """执行 eval 子进程（流式输出 + 超时 kill）；返回成功与否。"""
    print(f"\n{'=' * 60}", flush=True)
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {desc}", flush=True)
    print(f"  CMD: {' '.join(str(c) for c in cmd[:8])}…", flush=True)
    print(f"{'=' * 60}", flush=True)
    if dry_run:
        print("  [DRY-RUN] 跳过执行", flush=True)
        return True
    t0 = time.time()
    proc = None
    try:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
        proc.wait(timeout=RUN_TIMEOUT_MIN * 60)
        ok = proc.returncode == 0
        print(f"  {'[OK]' if ok else '[FAIL]'} {desc}: "
              f"{'成功' if ok else f'退出码 {proc.returncode}'} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        return ok
    except subprocess.TimeoutExpired:
        if proc:
            try:
                leftovers = kill_tree(proc.pid)   # 树杀：SIGKILL 只打直接子进程会留孤儿占显存
                if leftovers:
                    print(f"  [ERROR] 进程树未清干净，显存可能仍被占用: {leftovers}", flush=True)
                proc.wait()
            except OSError:
                pass
        print(f"  [TIMEOUT] {desc}: 超过 {RUN_TIMEOUT_MIN} min 被终止（eval_cli 产物已增量落盘，可 resume）", flush=True)
        return False
    except KeyboardInterrupt:
        if proc:
            try:
                leftovers = kill_tree(proc.pid)   # 树杀：SIGKILL 只打直接子进程会留孤儿占显存
                if leftovers:
                    print(f"  [ERROR] 进程树未清干净，显存可能仍被占用: {leftovers}", flush=True)
                proc.wait()
            except OSError:
                pass
        print("\n  [ABORT] 用户中断 — 停止剩余生成", flush=True)
        raise
    except Exception as e:  # noqa: BLE001 — 子进程启动异常统一处理
        if proc:
            try:
                leftovers = kill_tree(proc.pid)   # 树杀：SIGKILL 只打直接子进程会留孤儿占显存
                if leftovers:
                    print(f"  [ERROR] 进程树未清干净，显存可能仍被占用: {leftovers}", flush=True)
                proc.wait()
            except OSError:
                pass
        print(f"  [ERROR] {desc}: {e}", flush=True)
        return False


def _detect_run1(m: dict) -> dict | None:
    """检测 gen1 数据源（complete=True 且 records 非空）。

    优先级（2026-08-07 切换）：run1'（新 judge epoch 重判产物）> 旧 run1。
    """
    for path, tag in ((_run1p_path(m), "run1p"), (_run1_path(m), "run1")):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("complete") and data.get("records"):
            return {"reused": True, "source": tag, "path": str(path),
                    "records": len(data["records"])}
    return None


def _verify_adapter(m: dict) -> bool:
    """验证 adapter 路径存在（dry-run 输出 WARN 不阻断，对应风险表第 1 条）。"""
    if m["adapter"] is None:
        return True
    ok = (PROJECT_ROOT / m["adapter"]).is_dir()
    if not ok:
        print(f"  [WARN] {m['key']} adapter 路径缺失: {m['adapter']}（收官后确认，不阻断）",
              file=sys.stderr, flush=True)
    return ok


def _gen_done(ms: dict, gen: int) -> bool:
    """gen 是否已完成（state done 标记 + 磁盘产物双重确认）。

    只信 state 标记不够——产物被删/损坏时 resume 必须重新生成，
    否则后续聚合会静默缺数据。
    """
    st = ms.get(f"gen{gen}")
    if not (isinstance(st, dict) and bool(st.get("done"))):
        return False
    path = st.get("path")
    if not path:
        return False
    return _product_ok(Path(path))


def _product_ok(out: Path) -> bool:
    """磁盘产物复核: 文件存在、可解析且 records 非空。"""
    if not out.is_file():
        return False
    try:
        data = json.loads(out.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if isinstance(data, list):
        return False  # 子集题集等非 eval 产物结构（eval 产物必为 dict）
    return bool(data.get("records"))


def _plan_generations(phase: str, model_keys: list[str]) -> dict[str, list[int]]:
    """编排计划: 模型 → 需补的 generation 编号列表（σg 模型在全表段跳过）。

    run1 缺失时把 gen1 纳入计划（补足 n=3/n=5 协议），不静默产出 n=2。
    """
    if phase == "sigma":
        gens = SIGMA_GENS
    else:
        gens = FULL_GENS
    plan: dict[str, list[int]] = {}
    for key in model_keys:
        if phase == "full" and key in SIGMA_MODEL_KEYS:
            continue  # σg 标定模型已 n=5，全表段跳过
        missing_run1 = _detect_run1(MODEL_BY_KEY[key]) is None
        plan[key] = ([1] if missing_run1 else []) + list(gens)
    return plan


def _run_model(m: dict, gens: list[int], state: dict, dry_run: bool, smoke: bool,
                resume_eval: bool = True) -> bool:
    """执行单模型补生成（gen 级 DoneMarker，逐 gen 持久化 state）。

    resume_eval=False（--no-resume）时 eval_cli 不带 --resume，强制全量重新生成。
    """
    ms = state.setdefault(m["key"], {})
    label = m["label"]

    # 旧 run1 复用检测（每次运行重新探测，防 state 与磁盘不一致）
    if "run1" not in ms:
        run1 = _detect_run1(m)
        if run1:
            ms["run1"] = run1
            src = run1.get("source", "run1")
            print(f"\n  [RUN1-{src.upper()}-REUSE] {label}: 命中既有 gen1 数据源"
                  f"（{src}，{run1['records']} records），记为第 1 次生成", flush=True)
        else:
            ms["run1"] = {"reused": False}
            print(f"\n  [RUN1-MISS] {label}: 未发现完整 run1 产物（{_run1_path(m).name}），"
                  f"需收官后确认（协议要求 run1 复用，缺失则全表需补 3 次）", flush=True)
        _save_state(state)

    ok_all = True
    for gen in gens:
        if _gen_done(ms, gen):
            print(f"  [SKIP] {label} gen{gen}: 已完成", flush=True)
            continue
        out = _out_path(m["key"], gen, smoke)
        if dry_run:
            cmd = _build_cmd(m, out, smoke, dry_run=True)
            print(f"  {label:<22} gen{gen} → {out.relative_to(PROJECT_ROOT)}", flush=True)
            print(f"      {' '.join(str(c) for c in cmd)}", flush=True)
            continue
        cmd = _build_cmd(m, out, smoke, resume=resume_eval)
        ok = _run_eval(cmd, f"{label} — gen{gen}", dry_run=False)
        if ok and smoke:
            # smoke 验证通过：不标 done、不污染正式 state（正式运行会重新完整执行）
            print(f"  [SMOKE-OK] {label} gen{gen}: 2 题验证通过（未写入正式 state）", flush=True)
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
    return ok_all


def _print_plan(phase: str, keys: list[str], gens_by_key: dict[str, list[int]],
                dry_run: bool) -> None:
    """打印编排总览（dry-run 与真实执行共用）。"""
    print(f"\n[INFO] n=3 重生成编排 — phase={phase}, 模型数={len(keys)}, "
          f"dry_run={dry_run}")
    batch_order = {"4b": 0, "9b": 1, "gemma": 2}
    for key in keys:
        m = MODEL_BY_KEY[key]
        gens = gens_by_key.get(key, [])
        n = len(gens) + (1 if _detect_run1(m) else 0)
        adapter = m["adapter"] or "raw"
        print(f"  {key:<22} {m['batch']:<6} {m['domain']:<8} n={n} "
              f"(+{len(gens)} 次新生成) adapter={adapter}")


def main() -> int:
    parser = argparse.ArgumentParser(description="n=3 独立重生成编排器（σg 标定段 + 全表段）")
    parser.add_argument("--phase", choices=["sigma", "full"], default="full",
                        help="sigma=σg 标定段（2 模型 × 补 4 次至 n=5）；full=全表段（25 模型 × 补 2 次至 n=3）")
    parser.add_argument("--model", default="", help="逗号分隔模型 key（默认 phase 全部）")
    parser.add_argument("--dry-run", action="store_true", help="仅打印执行清单，不执行")
    parser.add_argument("--smoke", action="store_true", help="每模型仅前 2 题（临时子集题集）")
    parser.add_argument("--no-resume", action="store_true",
                        help="忽略 state 中已完成标记强制重跑（慎用，会覆盖产物）")
    args = parser.parse_args()

    # 终端显示规范：行缓冲，后台/管道场景进度实时可见
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    # 终端双写: 日志文件 + 终端（长任务规范，后台运行日志可追溯）
    log_file = LOG_DIR / f"n3_regen_{datetime.now():%Y%m%d_%H%M%S}.log"
    tee = TeeLogger(log_file)
    sys.stdout = tee
    print(f"[INFO] 日志: {log_file}", flush=True)

    # Python 解释器 pre-flight（缺失时提前报错，避免子进程启动失败语义不清）
    if not PYTHON.is_file():
        print(f"[ERROR] Python 解释器不存在: {PYTHON}", file=sys.stderr, flush=True)
        return 1

    model_keys = [k.strip() for k in args.model.split(",") if k.strip()]
    if model_keys:
        unknown = [k for k in model_keys if k not in MODEL_BY_KEY]
        if unknown:
            parser.error(f"未知模型 key: {unknown}（可选: {', '.join(MODEL_BY_KEY)}）")
    else:
        model_keys = SIGMA_MODEL_KEYS if args.phase == "sigma" else [m["key"] for m in MODELS]

    gens_by_key = _plan_generations(args.phase, model_keys)
    keys = [k for k in model_keys if k in gens_by_key]
    if args.model:
        skipped_sigma = [k for k in model_keys if k in SIGMA_MODEL_KEYS
                         and k not in gens_by_key]
        if skipped_sigma:
            print(f"  [INFO] 以下 σg 标定模型已在 σg 段补至 n=5，全表段跳过: "
                  f"{skipped_sigma}", flush=True)

    # adapter 路径预检（缺失仅 WARN，不阻断）
    for key in keys:
        _verify_adapter(MODEL_BY_KEY[key])

    _print_plan(args.phase, keys, gens_by_key, args.dry_run)

    if args.dry_run:
        print("\n[DRY-RUN] 执行清单:")
        for key in keys:
            m = MODEL_BY_KEY[key]
            for gen in gens_by_key[key]:
                out = _out_path(key, gen, args.smoke)
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
        ok = _run_model(m, gens_by_key[key], state, dry_run=False, smoke=args.smoke,
                        resume_eval=not args.no_resume)
        if not ok:
            failed.append(f"{key}(gen{'/'.join(map(str, gens_by_key[key]))})")
        # ETA: 按全部 gen 已完成的模型数外推（仅真实执行）
        done_cnt = sum(1 for k in keys
                       if all(_gen_done(state.get(k, {}), g)
                              for g in gens_by_key[k]))
        total_cnt = len(keys)
        if done_cnt > 0 and done_cnt < total_cnt:
            elapsed = time.time() - t_start
            eta_min = elapsed / done_cnt * (total_cnt - done_cnt) / 60
            print(f"\n  [ETA] {eta_min:.0f} min remaining ({done_cnt}/{total_cnt} 模型完成)", flush=True)

    total_min = (time.time() - t_start) / 60
    print(f"\n  [SUMMARY] 总耗时 {total_min:.1f} min")
    if failed:
        print(f"  [FAIL] {len(failed)} 模型未全部完成: {', '.join(failed)}", flush=True)
        return 1
    print("  [DONE] 全部模型补生成完成（state: %s）" % STATE_FILE)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[ABORT] 用户中断", file=sys.stderr, flush=True)
        raise SystemExit(130)
    except Exception as e:  # noqa: BLE001 — 顶层兜底，防裸栈崩溃
        print(f"\n[FATAL] 未预期异常: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        raise SystemExit(1)
