"""共享工具层 — 所有微调编排脚本的公共基础设施。

提取自 run_v2_ablation / run_v2_control_sci_ablation / run_v2_formal /
run_v3_cot_sft / eval_ppl_v2 等脚本中的重复代码。
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import yaml
from typing import Any

# ── 环境变量统一设置（import 时自动执行） ───────────────────────────────
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("BNB_CUDA_VERSION", "130")

# ── 项目常量 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


# ══════════════════════════════════════════════════════════════════════════
# GPU 工具
# ══════════════════════════════════════════════════════════════════════════

try:
    from patrol import kill_tree
except ImportError:  # pytest 以 scripts.* 包式导入时
    from scripts.patrol import kill_tree



def check_gpu() -> None:
    """打印 GPU 可用性信息（不阻断流程）。"""
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[GPU] {gpu_name} ({mem_gb:.1f} GB)")
    else:
        print("[WARN] CUDA not available — GPU steps will fail!")


def release_gpu() -> None:
    """释放所有 GPU 显存（训练步骤之间调用）。"""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _has_cuda() -> bool:
    """惰性检测 CUDA 可用性（不 import torch 除非需要）。"""
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, RuntimeError):
        return False


def gpu_health_snapshot() -> str | None:
    """Quick GPU utilization snapshot via nvidia-smi. Returns summary or None."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return None


@contextmanager
def training_session(model, tokenizer):
    """统一训练/评估会话：finally 块保证 gc.collect() + CUDA 缓存释放。

    用法：
        model, tokenizer = load_peft_model(...)
        with training_session(model, tokenizer):
            final_path = train_fr_lora_v2(model, tokenizer, ...)

    注意：Python 的 del 只删内部引用，调用方的 model/tokenizer 变量仍存活
    （外部引用计数 ≥ 1），CUDA tensor 不会被 Python GC 析构。本函数主要
    保证 finally 路径执行 gc.collect() + empty_cache()，为后续子进程
    腾出 GPU 缓存空间。调用方仍需在 with 块外置 model = None 以彻底释放。
    """
    _model = model
    _tok = tokenizer
    try:
        yield
    finally:
        del _model, _tok
        gc.collect()
        if _has_cuda():
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ══════════════════════════════════════════════════════════════════════════
# 原子 I/O
# ══════════════════════════════════════════════════════════════════════════


def atomic_write_json(path: Path, data: dict) -> None:
    """原子写入 JSON：先写临时文件，再 rename，防止写入中途崩溃导致损坏。

    在 Windows 上 os.replace 可能因 Defender/杀软实时扫描锁住 .tmp 文件
    而抛出 PermissionError。加入重试机制（最多 3 次，延迟 1s/2s）。
    其他 OSError（磁盘满等）立即清理临时文件并抛出。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        suffix=".tmp", dir=str(path.parent), prefix=".atomic_",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    for attempt in range(3):
        try:
            os.replace(tmp, str(path))
            return
        except PermissionError:
            if attempt < 2:
                time.sleep(attempt + 1)
                continue
            # Final attempt failed → cleanup and raise
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        except OSError:
            # Disk full, filesystem error, etc. → cleanup immediately
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def read_json(path: Path) -> dict | None:
    """读取 JSON 文件，文件不存在或损坏时返回 None。"""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ══════════════════════════════════════════════════════════════════════════
# 日志
# ══════════════════════════════════════════════════════════════════════════


class TeeLogger:
    """同时输出到 stdout 和日志文件的 Tee logger。

    Proxies all unknown attributes (buffer, encoding, isatty, etc.)
    to the underlying stdout to maintain compatibility with libraries
    that access stream metadata (tqdm, rich, torch distributed, etc.).
    """

    def __init__(self, log_path: Path, stdout: Any = None):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(str(log_path), "a", encoding="utf-8")
        self._stdout = stdout or sys.stdout
        self._closed = False
        self._lock = threading.Lock()

    def write(self, msg: str) -> None:
        with self._lock:
            self._stdout.write(msg)
            if not self._closed:
                self._file.write(msg)
            # Auto-flush on newline (line-buffered) — prevents output stalling
            if "\n" in msg:
                self._stdout.flush()
                if not self._closed:
                    self._file.flush()

    def flush(self) -> None:
        with self._lock:
            self._stdout.flush()
            if not self._closed:
                self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._file.close()
        sys.stdout = self._stdout

    def __getattr__(self, name: str) -> Any:
        # Proxy all unknown attributes to stdout (buffer, encoding, isatty, etc.)
        return getattr(self._stdout, name)


# ══════════════════════════════════════════════════════════════════════════
# 断点续跑
# ══════════════════════════════════════════════════════════════════════════


class DoneMarker:
    """统一的 .done 标记管理器，用于步骤级断点续跑。

    每个步骤生成独立的 .done 文件（JSON），记录 step_id 和完成时间。
    """

    def __init__(self, output_dir: Path):
        self._dir = output_dir

    @staticmethod
    def _sanitize(run_id: str) -> str:
        """Windows 文件名不允许 : < > \" / \\ | ? * — 替换为安全字符。

        step_id 常用 'cell:phase:run' 形式（冒号分层），需 sanitize 后才能
        作为 .done 文件名。is_done/mark_done 同一函数处理，保证一致性。
        """
        for ch in ':<>"/\\|?*':
            run_id = run_id.replace(ch, '_')
        return run_id

    def _marker_path(self, run_id: str) -> Path:
        return self._dir / f".{self._sanitize(run_id)}.done"

    def is_done(self, run_id: str) -> bool:
        return self._marker_path(run_id).exists()

    def mark_done(self, run_id: str) -> None:
        marker = self._marker_path(run_id)
        marker.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(marker, {
            "step_id": run_id,
            "completed_at": datetime.now().isoformat(),
        })

    def clear(self, run_id: str) -> None:
        """删除指定步骤的 DoneMarker（用于作废无效的完成标记）。"""
        marker = self._marker_path(run_id)
        if marker.exists():
            marker.unlink()


# ══════════════════════════════════════════════════════════════════════════
# 子进程
# ══════════════════════════════════════════════════════════════════════════


def run_subprocess(
    cmd: list[str],
    desc: str,
    timeout_min: int = 30,
    cwd: Path | None = None,
) -> int:
    """运行子进程，流式输出；返回 exit code。"""
    print(f"\n{'─' * 60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    print(f"{'─' * 60}")
    sys.stdout.flush()

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd or PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # ── 后台线程逐行读取 stdout，主线程负责超时控制 ──
    def _stream():
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
        except Exception:
            pass  # daemon 线程静默退出

    reader = threading.Thread(target=_stream, daemon=True)
    reader.start()

    timeout_sec = timeout_min * 60
    deadline = time.time() + timeout_sec
    try:
        # 轮询等待替代 proc.wait()：每 0.5s 检查一次进程状态，
        # 给 Python 信号处理器检查窗口，确保 Windows 上 Ctrl+C 可即时中断
        while proc.poll() is None:
            if time.time() > deadline:
                raise subprocess.TimeoutExpired(cmd, timeout_sec)
            time.sleep(0.5)
        proc.stdout.close()  # 强制关闭管道，确保 reader 线程退出
        reader.join(timeout=5)
        elapsed = time.time() - t0
        print(f"\n  [{desc}] exit={proc.returncode}  ({elapsed / 60:.1f}min)")
        return proc.returncode
    except subprocess.TimeoutExpired:
        try:
            leftovers = kill_tree(proc.pid)   # 树杀：SIGKILL 只打直接子进程会留孤儿占显存
            if leftovers:
                print(f"  [ERROR] 进程树未清干净，显存可能仍被占用: {leftovers}", flush=True)
            proc.wait(timeout=10)
        except OSError:
            pass
        reader.join(timeout=5)
        print(f"\n  [TIMEOUT] {desc} exceeded {timeout_min}min")
        return -1


# ══════════════════════════════════════════════════════════════════════════
# YAML 配置加载
# ══════════════════════════════════════════════════════════════════════════


def load_experiment_config(
    config_path: Path,
) -> dict:
    """加载 YAML 实验配置，并应用 CLI 覆盖。

    YAML 提供默认值；CLI 参数（非 None）覆盖 YAML 值。
    """
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# ══════════════════════════════════════════════════════════════════════════
# ETA 估算 — 基于 RTX 5090 + Qwen3.5-4B + LoRA 实测数据
# ══════════════════════════════════════════════════════════════════════════

# 实际测得的每步训练耗时（1408 条 control_sci, batch=1, ga_steps=2）
SEC_PER_TRAIN_STEP = 4.0      # ~4s/step on RTX 5090
SEC_ADAPTER_LOAD = 15.0       # ~10-15s 加载 base adapter
SEC_PPL_PER_SAMPLE = 0.7      # ~0.7s per val sample for PPL eval
SEC_GPU_COOLDOWN = 30         # 训练间 GPU 休息


def est_train_seconds(num_steps: int, n_val: int = 0) -> float:
    """估算一次训练的 wall-clock 耗时（含 adapter 加载 + 训练 + PPL + 冷却）。

    基于 RTX 5090 + Qwen3.5-4B + LoRA (r=8, α=16) 实测。
    """
    est = SEC_ADAPTER_LOAD + num_steps * SEC_PER_TRAIN_STEP
    if n_val > 0:
        est += n_val * SEC_PPL_PER_SAMPLE
    est += SEC_GPU_COOLDOWN
    return est


def format_eta_str(total_seconds: float) -> str:
    """格式化 ETA 为 'X.Yh' / 'Ymin' / 'Xs' 可读字符串。"""
    if total_seconds < 120:
        return f"{total_seconds:.0f}s"
    mins = total_seconds / 60
    if mins < 60:
        return f"{mins:.0f}min"
    return f"{mins / 60:.1f}h"
