"""FR-LoRA warm-start probe: end-to-end FBC verification.

Orchestrates:
1. Load Standard LoRA adapter (qlora-qwen4b-cross-final)
2. Measure baseline FBC (ChatML vs Raw PPL)
3. Fine-tune 1 epoch with FBC penalty
4. Measure post-training FBC
5. Report delta and verdict
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from mineru_eval.finetune.adapter_loader import load_peft_model
from mineru_eval.finetune.ppl_probe import (
    FBCSummary,
    compute_fbc_batch,
    load_jsonl_samples,
)
from mineru_eval.finetune.trainer import train_fr_lora


# ---------------------------------------------------------------------------
# Report data structures
# ---------------------------------------------------------------------------


@dataclass
class ProbeReport:
    """Full FR-LoRA warm-start probe results."""

    adapter_path: str
    train_data_path: str
    lambda_fbc: float
    num_train_samples: int
    num_eval_samples: int
    num_epochs: int

    before: FBCSummary | None = None
    after: FBCSummary | None = None

    raw_ppl_delta_pct: float = 0.0
    chatml_ppl_delta_pct: float = 0.0
    fbc_delta_change_pct: float = 0.0

    verdict: str = "未执行"
    checkpoint_path: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_train_eval(
    train_data_path: Path,
    eval_data_path: str | Path | None,
    num_train_samples: int,
    num_eval_samples: int,
) -> tuple[list[dict], list[dict]]:
    """Load and split train/eval samples, ensuring no overlap.

    Returns (train_samples, eval_samples).
    When same_file=True: eval from tail, train from head.
    """
    eval_path = Path(eval_data_path) if eval_data_path else train_data_path
    same_file = (eval_path.resolve() == train_data_path.resolve())

    if same_file:
        total_needed = num_eval_samples + num_train_samples
        all_samples = load_jsonl_samples(train_data_path, max_samples=total_needed)
        enough_for_split = len(all_samples) >= total_needed
        eval_samples = (
            all_samples[-num_eval_samples:]
            if len(all_samples) >= num_eval_samples
            else all_samples
        )
        train_samples = (
            all_samples[:num_train_samples]
            if enough_for_split
            else []  # insufficient data for clean split → skip training
        )
    else:
        eval_samples = load_jsonl_samples(eval_path, max_samples=num_eval_samples)
        train_samples = load_jsonl_samples(train_data_path, max_samples=num_train_samples)

    return train_samples, eval_samples


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_warm_start_probe(
    adapter_path: str | Path,
    train_data_path: str | Path,
    eval_data_path: str | Path | None = None,
    lambda_fbc: float = 0.5,
    num_train_samples: int = 500,
    num_eval_samples: int = 100,
    epochs: int = 1,
    output_dir: str | Path | None = None,
    base_model_id: str | None = None,
    max_seq_length: int = 2048,
) -> ProbeReport:
    """Run a complete FR-LoRA warm-start probe.

    Args:
        adapter_path: Path to Standard LoRA adapter directory.
        train_data_path: Path to jsonl training data.
        eval_data_path: Path to jsonl eval data. If None, uses train_data_path
                        (separate sample subset).
        lambda_fbc: FBC penalty weight (0 = standard, 0.5 = moderate).
        num_train_samples: Number of training samples to use.
        num_eval_samples: Number of eval samples for FBC measurement.
        epochs: Number of fine-tuning epochs.
        output_dir: Directory for checkpoint output.
        base_model_id: HF model ID. Defaults to env FR_LORA_BASE_MODEL.
        max_seq_length: Max token length for PPL computation.
    """
    adapter_path = Path(adapter_path)
    train_data_path = Path(train_data_path)

    if output_dir is None:
        output_dir = Path("models/fr_lora_probe_output")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = ProbeReport(
        adapter_path=str(adapter_path),
        train_data_path=str(train_data_path),
        lambda_fbc=lambda_fbc,
        num_train_samples=num_train_samples,
        num_eval_samples=num_eval_samples,
        num_epochs=epochs,
    )

    # ---- Phase 1: Load model + adapter ----
    print("=" * 55)
    print("Phase 1: Loading Standard LoRA adapter")
    print("=" * 55)
    model, tokenizer = load_peft_model(
        base_model_id=base_model_id,
        adapter_path=adapter_path,
        is_trainable=True,
    )
    model.eval()

    # ---- Split train/eval data (Phase 2 prep) ----
    train_samples, eval_samples = _split_train_eval(
        train_data_path, eval_data_path, num_train_samples, num_eval_samples
    )
    if len(eval_samples) < 5:
        print(f"[ERROR] Need at least 5 eval samples, got {len(eval_samples)}")
        report.verdict = "失败：评估样本不足"
        return report

    # ---- Phase 2: Baseline FBC measurement ----
    print("\n" + "=" * 55)
    print("Phase 2: Baseline FBC measurement (Standard LoRA)")
    print("=" * 55)
    print(f"[probe] Evaluating {len(eval_samples)} samples for baseline FBC...")
    report.before = compute_fbc_batch(
        model, tokenizer, eval_samples, max_length=max_seq_length
    )
    _print_fbc("BEFORE (Standard LoRA)", report.before)

    # ---- Phase 3: FR-LoRA fine-tuning ----
    print("\n" + "=" * 55)
    print(f"Phase 3: FR-LoRA fine-tuning (λ={lambda_fbc}, {epochs} epoch)")
    print("=" * 55)
    print(f"[probe] Training on {len(train_samples)} samples (eval={len(eval_samples)}, overlap=0)...")

    checkpoint_path = train_fr_lora(
        model=model,
        tokenizer=tokenizer,
        train_data=train_samples,
        output_dir=output_dir,
        lambda_fbc=lambda_fbc,
        num_epochs=epochs,
        batch_size=1,
    )
    report.checkpoint_path = str(checkpoint_path)

    # ---- Phase 4: Post-training FBC measurement ----
    print("\n" + "=" * 55)
    print("Phase 4: Post-training FBC measurement (FR-LoRA)")
    print("=" * 55)
    model.eval()
    report.after = compute_fbc_batch(
        model, tokenizer, eval_samples, max_length=max_seq_length
    )
    _print_fbc("AFTER (FR-LoRA)", report.after)

    # ---- Phase 5: Compute delta & verdict ----
    _compute_verdict(report)

    # ---- Save report ----
    report_path = output_dir / "probe_report.json"
    report_path.write_text(
        json.dumps(_report_to_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[probe] Report saved to {report_path}")

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_verdict(report: ProbeReport) -> None:
    """Compute delta metrics and set verdict on report (mutates in-place)."""
    if not (report.before and report.after):
        return

    report.raw_ppl_delta_pct = _delta_pct(
        report.before.avg_raw_ppl, report.after.avg_raw_ppl
    )
    report.chatml_ppl_delta_pct = _delta_pct(
        report.before.avg_chatml_ppl, report.after.avg_chatml_ppl
    )
    report.fbc_delta_change_pct = (
        report.after.avg_fbc_delta_pct - report.before.avg_fbc_delta_pct
    )

    raw_drop = -report.raw_ppl_delta_pct  # positive = improvement

    if raw_drop >= 20:
        report.verdict = "🟢 绿灯 — Raw PPL 下降 ≥20%，FR-LoRA 有效"
    elif raw_drop >= 5:
        report.verdict = "🟡 黄灯 — Raw PPL 下降 5-20%，尝试调大 λ"
    elif raw_drop >= -5:
        report.verdict = "🔴 红灯 — Raw PPL 无明显变化，FR-LoRA 在当前 λ 下无效"
    else:
        report.verdict = "🔴 红灯 — Raw PPL 上升，方向错误"


def _delta_pct(before: float, after: float) -> float:
    """Percentage change: ((after - before) / before) × 100."""
    if before == 0:
        return 0.0
    return (after - before) / before * 100.0


def _print_fbc(label: str, summary: FBCSummary) -> None:
    print(f"\n  [{label}]")
    print(f"    Samples:       {summary.n_samples}")
    print(f"    ChatML PPL:    {summary.avg_chatml_ppl:.1f}")
    print(f"    Raw PPL:       {summary.avg_raw_ppl:.1f}")
    print(f"    FBC Δ%:        {summary.avg_fbc_delta_pct:+.1f}%")


def _report_to_dict(report: ProbeReport) -> dict:
    d = asdict(report)
    # Convert nested FBCSummary to dict (asdict handles dataclasses recursively)
    return d


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FR-LoRA warm-start probe — FBC verification",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Path to Standard LoRA adapter directory",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="Path to jsonl training data",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=None,
        help="Path to jsonl eval data (default: same as --train-data)",
    )
    parser.add_argument(
        "--lambda-fbc",
        type=float,
        default=0.5,
        help="FBC penalty weight (default: 0.5)",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=500,
        help="Number of training samples (default: 500)",
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        default=100,
        help="Number of eval samples for FBC measurement (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of fine-tuning epochs (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/fr_lora_probe_output"),
        help="Output directory for checkpoint + report",
    )
    args = parser.parse_args()

    report = run_warm_start_probe(
        adapter_path=args.adapter,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        lambda_fbc=args.lambda_fbc,
        num_train_samples=args.num_train,
        num_eval_samples=args.num_eval,
        epochs=args.epochs,
        output_dir=args.output_dir,
    )

    print(f"\n{'=' * 55}")
    print(f"VERDICT: {report.verdict}")
    print(f"{'=' * 55}")
    if report.raw_ppl_delta_pct != 0:
        print(f"  Raw PPL Δ:     {report.raw_ppl_delta_pct:+.1f}%")
        print(f"  ChatML PPL Δ:  {report.chatml_ppl_delta_pct:+.1f}%")
        print(f"  FBC Δ change:  {report.fbc_delta_change_pct:+.1f}pp")
    print(f"  Checkpoint:    {report.checkpoint_path}")
