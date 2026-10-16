"""训练脚本共享防护：缓存诊断回调 + GPU 遥测采样 + attn 实现校验。sft/mgpo/rlvr 复用。

只读诊断（不改变训练数值行为）；用于为「首轮慢」根因诊断收集证据
（缓存路径 vs 生成负载 vs 限频/环境）。
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from transformers import TrainerCallback


def validate_attn_implementation(model, requested: str | None) -> bool:
    """加载后校验实际 attn 实现与请求一致；不一致仅告警不中止。

    Returns True 表示一致（或无请求可比对），False 表示检出可能的静默 fallback。
    """
    actual = getattr(model.config, "_attn_implementation", None)
    if requested is None or actual == requested:
        return True
    print(f"[guard][WARN] attn_implementation mismatch: "
          f"requested={requested} actual={actual} — possible silent fallback", flush=True)
    return False


class CacheDiagnosticCallback(TrainerCallback):
    """只读：训练开始与首步打印缓存/注意力关键状态，供根因诊断。零行为改变。

    仅在 `on_train_begin` 与首个 `on_step_end` 各打印一次，避免刷屏。
    所有字段用 `getattr(..., None)` 软探测，缺失即打印 None，不抛异常。
    """

    def __init__(self):
        self._logged_step = False

    def _dump(self, tag: str, model) -> None:
        cfg = getattr(model, "config", None)
        print(f"[diag][{tag}] training={getattr(model, 'training', None)} "
              f"use_cache={getattr(cfg, 'use_cache', None)} "
              f"gradient_checkpointing={getattr(model, 'is_gradient_checkpointing', None)} "
              f"attn_implementation={getattr(cfg, '_attn_implementation', None)} "
              f"is_fast_path_available={getattr(model, 'is_fast_path_available', None)}", flush=True)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._dump("train_begin", model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not self._logged_step and model is not None:
            self._dump("first_step_end", model)
            self._logged_step = True


class GpuTelemetryCallback(TrainerCallback):
    """诊断专用：训练全程后台采样 GPU 遥测（SM 时钟/温度/功耗/util/显存），
    并记录每步结束的墙钟时刻，供离线将 step_time 与「限频/热」证据对齐。

    不改变训练数值行为。采样经 ``nvidia-smi dmon`` 子进程落盘到
    ``output_dir/gpu_dmon.log``，步边界墙钟写入 ``output_dir/step_marks.csv``。
    best-effort：采样启动失败仅告警不中止训练；但因「缺遥测」正是要修的洞，
    启动失败会打印显著 WARN，提示限频根因将无证据可判。
    """

    def __init__(self, sample_interval_s: int = 2):
        self._proc = None
        self._log_fh = None
        self._interval = sample_interval_s
        self._marks: list[tuple[int, float]] = []
        self._t0 = None
        self._out = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._out = Path(args.output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        dmon_log = self._out / "gpu_dmon.log"
        self._t0 = time.time()
        try:
            self._log_fh = open(dmon_log, "w", encoding="utf-8")
            self._proc = subprocess.Popen(
                ["nvidia-smi", "dmon", "-s", "pucm",
                 "-d", str(self._interval), "-o", "DT"],
                stdout=self._log_fh, stderr=subprocess.STDOUT,
            )
            print(f"[telemetry] GPU dmon sampling every {self._interval}s -> {dmon_log}",
                  flush=True)
        except (OSError, ValueError) as exc:
            print(f"[telemetry][WARN] failed to start nvidia-smi dmon: {exc} — "
                  f"限频/环境根因将无证据可判！", flush=True)
            self._proc = None
            if self._log_fh is not None:
                self._log_fh.close()
                self._log_fh = None

    def on_step_end(self, args, state, control, **kwargs):
        if self._t0 is not None:
            self._marks.append((state.global_step, round(time.time() - self._t0, 2)))

    def on_train_end(self, args, state, control, **kwargs):
        self._stop()

    def _stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        if self._marks and self._out is not None:
            marks_csv = self._out / "step_marks.csv"
            lines = ["step,elapsed_s"] + [f"{s},{t}" for s, t in self._marks]
            marks_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"[telemetry] {len(self._marks)} step marks -> {marks_csv}", flush=True)
