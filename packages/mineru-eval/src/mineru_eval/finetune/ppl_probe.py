"""ChatML / Raw dual-mode perplexity probe for FBC quantification.

Format Binding Cost (FBC) = (Raw_PPL - ChatML_PPL) / ChatML_PPL × 100%.
A large positive FBC indicates the model over-depends on ChatML formatting.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PPLResult:
    """Per-sample PPL measurement in both modes."""

    idx: int
    instruction: str
    output: str
    chatml_ppl: float
    raw_ppl: float
    fbc_delta_pct: float  # (raw - chatml) / chatml × 100

    @property
    def has_positive_fbc(self) -> bool:
        return self.fbc_delta_pct > 0


@dataclass
class FBCSummary:
    """Aggregated FBC statistics across samples."""

    n_samples: int
    avg_chatml_ppl: float
    avg_raw_ppl: float
    avg_fbc_delta_pct: float
    median_fbc_delta_pct: float
    results: list[PPLResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ChatML formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "你是控制科学专家。请准确、专业、完整地回答以下问题。涉及公式推导时逐步写出。"


def build_chatml_text(
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    output: str,
) -> str:
    """Build ChatML-formatted text via tokenizer's chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def build_raw_text(instruction: str, output: str) -> str:
    """Build plain text (no formatting) for raw PPL measurement."""
    return f"{instruction}\n\n{output}"


# ---------------------------------------------------------------------------
# Core PPL computation
# ---------------------------------------------------------------------------


def compute_ppl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_length: int = 2048,
) -> float:
    """Compute perplexity for a single text via cross-entropy loss."""
    if not text or not text.strip():
        return float("inf")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    loss = outputs.loss.item()
    return math.exp(loss)


def compute_fbc_sample(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    output: str,
    idx: int = 0,
    max_length: int = 2048,
) -> PPLResult:
    """Compute FBC for a single instruction-output pair."""
    chatml_text = build_chatml_text(tokenizer, instruction, output)
    raw_text = build_raw_text(instruction, output)

    chatml_ppl = compute_ppl(model, tokenizer, chatml_text, max_length)
    raw_ppl = compute_ppl(model, tokenizer, raw_text, max_length)

    if chatml_ppl > 0 and chatml_ppl != float("inf") and not math.isnan(chatml_ppl):
        fbc_delta = (raw_ppl - chatml_ppl) / chatml_ppl * 100.0
    else:
        fbc_delta = 0.0

    return PPLResult(
        idx=idx,
        instruction=instruction,
        output=output,
        chatml_ppl=chatml_ppl,
        raw_ppl=raw_ppl,
        fbc_delta_pct=fbc_delta,
    )


def _iter_fbc_samples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: list[dict],
    max_length: int = 2048,
    verbose: bool = True,
) -> list[PPLResult]:
    """Iterate samples, compute PPL, return results (no aggregation)."""
    results: list[PPLResult] = []
    for i, sample in enumerate(samples):
        instruction = sample.get("instruction") or ""
        output = sample.get("output") or ""
        if not output:
            continue

        result = compute_fbc_sample(
            model, tokenizer, instruction, output, idx=i, max_length=max_length
        )
        results.append(result)

        if verbose and (i + 1) % 10 == 0:
            valid_chatml = [r.chatml_ppl for r in results
                            if not math.isnan(r.chatml_ppl) and r.chatml_ppl != float("inf")]
            valid_raw = [r.raw_ppl for r in results
                         if not math.isnan(r.raw_ppl) and r.raw_ppl != float("inf")]
            chatml_avg = sum(valid_chatml) / len(valid_chatml) if valid_chatml else float("nan")
            raw_avg = sum(valid_raw) / len(valid_raw) if valid_raw else float("nan")
            print(
                f"  [{i + 1}/{len(samples)}] "
                f"chatml_ppl={chatml_avg:.1f}  raw_ppl={raw_avg:.1f}",
                flush=True,
            )

    return results


def _build_fbc_summary(results: list[PPLResult]) -> FBCSummary:
    """Filter NaN/Inf then aggregate statistics into FBCSummary."""
    if not results:
        return FBCSummary(n_samples=0, avg_chatml_ppl=0, avg_raw_ppl=0,
                          avg_fbc_delta_pct=0, median_fbc_delta_pct=0)

    chatml_ppls = [r.chatml_ppl for r in results
                   if not math.isnan(r.chatml_ppl) and r.chatml_ppl != float("inf")]
    raw_ppls = [r.raw_ppl for r in results
                if not math.isnan(r.raw_ppl) and r.raw_ppl != float("inf")]
    fbc_deltas = [r.fbc_delta_pct for r in results
                  if not math.isnan(r.fbc_delta_pct)
                  and r.fbc_delta_pct != float("inf")
                  and r.fbc_delta_pct != float("-inf")]

    if not chatml_ppls:
        return FBCSummary(n_samples=0, avg_chatml_ppl=float('nan'),
                          avg_raw_ppl=float('nan'), avg_fbc_delta_pct=float('nan'),
                          median_fbc_delta_pct=float('nan'))

    if not fbc_deltas:
        return FBCSummary(
            n_samples=len(chatml_ppls),
            avg_chatml_ppl=sum(chatml_ppls) / len(chatml_ppls),
            avg_raw_ppl=sum(raw_ppls) / len(raw_ppls) if raw_ppls else 0,
            avg_fbc_delta_pct=0, median_fbc_delta_pct=0,
            results=results,
        )

    sorted_fbc = sorted(fbc_deltas)
    mid = len(sorted_fbc) // 2
    median_fbc = (
        sorted_fbc[mid]
        if len(sorted_fbc) % 2 == 1
        else (sorted_fbc[mid - 1] + sorted_fbc[mid]) / 2
    )

    return FBCSummary(
        n_samples=len(chatml_ppls),
        avg_chatml_ppl=sum(chatml_ppls) / len(chatml_ppls),
        avg_raw_ppl=sum(raw_ppls) / len(raw_ppls),
        avg_fbc_delta_pct=sum(fbc_deltas) / len(fbc_deltas),
        median_fbc_delta_pct=median_fbc,
        results=results,
    )


def compute_fbc_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: list[dict],
    max_length: int = 2048,
    verbose: bool = True,
) -> FBCSummary:
    """Compute FBC across a batch of samples.

    Args:
        samples: List of dicts with 'instruction' and 'output' keys.
    """
    results = _iter_fbc_samples(model, tokenizer, samples, max_length, verbose)
    return _build_fbc_summary(results)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl_samples(path: str | Path, max_samples: int | None = None) -> list[dict]:
    """Load instruction-output pairs from jsonl file.

    Expected format per line:
      {"instruction": "...", "output": "..."}
    or ChatML-style:
      {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    path = Path(path)
    samples: list[dict] = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        item = json.loads(line)
        # ChatML messages format → extract instruction / output
        if "messages" in item:
            user_msg = next(
                (m["content"] for m in item["messages"] if m["role"] == "user"), ""
            )
            asst_msg = next(
                (m["content"] for m in item["messages"] if m["role"] == "assistant"), ""
            )
            samples.append({"instruction": user_msg, "output": asst_msg})
        elif "instruction" in item:
            samples.append(item)
        if max_samples and len(samples) >= max_samples:
            break
    return samples


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from mineru_eval.finetune.adapter_loader import load_peft_model

    parser = argparse.ArgumentParser(description="Smoke-test PPL probe")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("models/qlora-qwen4b-cross-final"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/finetune/medical/val.jsonl"),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
    )
    args = parser.parse_args()

    if not args.adapter.is_dir():
        print(f"[SKIP] Adapter not found: {args.adapter}")
        exit(0)
    if not args.data.exists():
        print(f"[SKIP] Data not found: {args.data}")
        exit(0)

    print(f"[ppl_probe] Loading adapter: {args.adapter}")
    model, tokenizer = load_peft_model(adapter_path=args.adapter, is_trainable=False)
    model.eval()

    samples = load_jsonl_samples(args.data, max_samples=args.max_samples)
    print(f"[ppl_probe] Loaded {len(samples)} samples from {args.data}")

    summary = compute_fbc_batch(model, tokenizer, samples)

    print(f"\n{'=' * 55}")
    print("FBC Summary (Standard LoRA)")
    print(f"{'=' * 55}")
    print(f"  Samples:            {summary.n_samples}")
    print(f"  Avg ChatML PPL:     {summary.avg_chatml_ppl:.1f}")
    print(f"  Avg Raw PPL:        {summary.avg_raw_ppl:.1f}")
    print(f"  Avg FBC Δ%:         {summary.avg_fbc_delta_pct:+.1f}%")
    print(f"  Median FBC Δ%:      {summary.median_fbc_delta_pct:+.1f}%")

    if summary.n_samples > 0:
        positive_count = sum(1 for r in summary.results if r.has_positive_fbc)
        print(f"  Positive FBC:       {positive_count}/{summary.n_samples} "
              f"({positive_count / summary.n_samples * 100:.0f}%)")
    else:
        print("  Positive FBC:       N/A (no valid samples)")

    # Quick sanity check against historical data (-47% ChatML delta is domain
    # adaptation, not FBC. FBC should be strongly positive if the format
    # binding phenomenon is present.)
    if summary.avg_fbc_delta_pct > 100:
        verdict = "✅ FBC phenomenon confirmed (strong positive delta)"
    elif summary.avg_fbc_delta_pct > 10:
        verdict = "⚠️  FBC weak — may need larger sample or different domain"
    else:
        verdict = "❌ Unexpected — FBC near zero or negative"
    print(f"\n  Verdict: {verdict}")
