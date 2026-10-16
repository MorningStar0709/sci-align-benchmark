"""FR-LoRA training with optional FBC penalty via SFTTrainer.

Provides:
- train_standard_lora: Standard SFT fine-tuning (baseline)
- train_fr_lora: SFT fine-tuning with FBC penalty via custom compute_loss
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from transformers import TrainerCallback, ProgressCallback
from trl import SFTConfig, SFTTrainer
from peft import PeftModel
from datasets import Dataset


# ---------------------------------------------------------------------------
# Compact training log callback — replaces verbose raw-dict per-step output
# with a single-line aligned format suitable for TeeLogger (stdout capture).
# ---------------------------------------------------------------------------


class CompactLogCallback(TrainerCallback):
    """Compact per-step log format with ETA and speed.

    Per-step: loss gn lr | speed ETA
    Eval logs: prints eval loss line.
    Final summary: train_loss, runtime, speed, total_flos.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        import time
        self._train_start = time.monotonic()

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        """Format seconds into human-readable HH:MM:SS or MM:SS."""
        if seconds < 0:
            return "?"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return

        step = state.global_step
        _max = state.max_steps if state.max_steps is not None else state.epoch
        max_steps = _max if _max is not None else "?"

        # ── Final summary (train_runtime present) ──
        if "train_runtime" in logs:
            rt = float(logs["train_runtime"])
            sps = float(logs.get("train_samples_per_second", 0))
            train_loss = logs.get("train_loss", "?")
            flos = logs.get("total_flos", 0)
            flos_str = f"{flos / 1e12:.1f} TFLOPS" if flos else ""
            print(
                f"  [DONE] loss={train_loss} "
                f"runtime={self._fmt_duration(rt)} "
                f"speed={sps:.1f} samp/s"
                + (f"  {flos_str}" if flos_str else "")
            )
            return

        # ── Per-step log ──
        loss_k = "eval_loss" if "eval_loss" in logs else "loss"
        loss = float(logs.get(loss_k, 0))
        gn = float(logs.get("grad_norm", 0))
        lr = float(logs.get("learning_rate", 0))
        tag = "[eval]" if "eval_loss" in logs else ""

        # Speed & ETA
        speed_str = ""
        if (
            hasattr(self, "_train_start")
            and step > 0
            and isinstance(max_steps, (int, float))
            and max_steps > 0
        ):
            import time
            elapsed = time.monotonic() - self._train_start
            sps = step / elapsed if elapsed > 0 else 0
            remaining = (max_steps - step) / sps if sps > 0 else 0
            speed_str = f"  {sps:.1f} step/s  ETA {self._fmt_duration(remaining)}"

        print(
            f"  [{step:3d}/{str(max_steps):>3s}] {tag}"
            f"loss={loss:6.1f}  gn={gn:6.0f}  lr={lr:.2e}"
            f"{speed_str}"
        )


# ---------------------------------------------------------------------------
# FR-LoRA loss with FBC penalty
# ---------------------------------------------------------------------------


def make_fbc_compute_loss_func(
    lambda_fbc: float,
    chatml_start_id: int,
    chatml_end_id: int,
) -> Callable:
    """Create a compute_loss_func that adds FBC penalty.

    FBC penalty = λ × max(0, raw_loss - chatml_loss), where:
    - chatml_loss is the standard CE loss on ChatML-formatted text
    - raw_loss is CE loss with ChatML special tokens masked out

    This penalizes the model when it over-depends on ChatML formatting tokens
    to reduce perplexity, incentivizing format-robust representations.
    """
    from torch.nn import CrossEntropyLoss
    loss_fn = CrossEntropyLoss(ignore_index=-100, reduction='sum')

    def _fbc_compute_loss(outputs, labels, num_items_in_batch=None):
        # SFTTrainer does NOT compute loss in forward — outputs.loss is None.
        # We compute both ChatML and Raw CE loss from logits + labels manually.
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Standard ChatML loss (all labels contribute)
        chatml_sum = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        n_tokens = (shift_labels != -100).sum()
        chatml_loss = chatml_sum / n_tokens if n_tokens > 0 else chatml_sum

        # Raw loss: mask out ChatML special tokens, same denominator as ChatML
        if chatml_start_id is not None and chatml_end_id is not None:
            raw_labels = labels.clone()
            special_mask = (labels == chatml_start_id) | (labels == chatml_end_id)
            raw_labels[special_mask] = -100
            shift_raw_labels = raw_labels[..., 1:].contiguous()
            raw_sum = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_raw_labels.view(-1),
            )
            raw_loss = raw_sum / n_tokens if n_tokens > 0 else raw_sum
            fbc_penalty = torch.clamp(raw_loss - chatml_loss, min=0)
            return chatml_loss + lambda_fbc * fbc_penalty

        return chatml_loss

    return _fbc_compute_loss


def make_plw_l2_loss_func(
    alpha: float = 0.3,
    lambda_l2: float = 0.01,
    format_token_ids: set[int] | None = None,
    sample_ratio: float = 1.0,
) -> Callable:
    """Create v2 loss: PLW (Progressive Loss Weighting) + Logit L2 penalty.

    L = Σ w_i · CE(logits_i, labels_i) + λ · mean(∥logits[fmt_pos]∥²)

    where w_i = 1 − α if token_i is a format token, else 1.

    Unlike v1 mask-based FBC, this:
    - Uses continuous weights (not binary mask) → preserves gradient signal
    - Computes L2 only at ~20 format token positions (not full vocab) → no TDR risk
    - Single CE call vs v1's double CE call → ~2× memory reduction over v1

    Args:
        alpha: PLW decay factor (0=no penalty, 1=full mask like v1)
        lambda_l2: Logit L2 regularization strength
        format_token_ids: Set of token IDs to treat as format tokens
        sample_ratio: Fraction of valid tokens for CE (1.0=full, 0.2=v2-sample)
    """
    if format_token_ids is None or len(format_token_ids) == 0:
        raise ValueError("format_token_ids must be a non-empty set")
    fmt_ids = torch.tensor(list(format_token_ids))

    def _plw_l2_loss(outputs, labels, num_items_in_batch=None):
        # outputs.loss is None when compute_loss_func is used (SFTTrainer
        # doesn't pass labels to model.forward). Use manual CE instead.
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        fmt_device = shift_logits.device
        fmt_ids_device = fmt_ids.to(fmt_device)

        # Token subsampling: randomly drop valid tokens before CE
        if sample_ratio < 1.0:
            flat = shift_labels.view(-1)
            valid = (flat != -100)
            valid_idx = torch.where(valid)[0]
            n_keep = max(1, int(valid_idx.numel() * sample_ratio))
            perm = torch.randperm(valid_idx.numel(), device=fmt_device)
            keep_idx = valid_idx[perm[:n_keep]]
            keep_mask = torch.zeros(flat.numel(), dtype=torch.bool, device=fmt_device)
            keep_mask[keep_idx] = True
            ce_labels = shift_labels.clone()
            ce_labels.view(-1)[valid & ~keep_mask] = -100
        else:
            ce_labels = shift_labels

        # CE: fused kernel when alpha==0 (no per-token weighting needed)
        if alpha == 0.0:
            ce_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                ce_labels.view(-1),
                reduction='mean',
                ignore_index=-100,
            )
        else:
            # PLW: per-token weights (1-α for format tokens, 1 for others)
            ce_per_token = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                ce_labels.view(-1),
                reduction='none',
                ignore_index=-100,
            )
            weights = torch.ones(ce_labels.numel(), dtype=shift_logits.dtype, device=fmt_device)
            is_format = torch.isin(ce_labels.view(-1), fmt_ids_device)
            weights[is_format] = 1.0 - alpha
            valid = (ce_labels.view(-1) != -100)
            w_sum = weights[valid].sum()
            ce_loss = (ce_per_token[valid] * weights[valid]).sum() / w_sum if w_sum > 0 else torch.tensor(0.0, device=fmt_device)

        # L2 penalty: mean over format token positions (~20 tokens, O(1))
        # Normalized per position to match CE's per-sample reduction scale,
        # so PLW and L2 components are commensurate in gradient magnitude.
        #
        # ⚠️ BREAKING CHANGE (v2): L2 was torch.sum(...) before commit a89e82b;
        # now uses mean (sum / N). All existing λ values are ~20× too small
        # in the mean semantics and must be re-tuned. See run_v2_ablation.py
        # for the recommended λ sweep.
        # NOTE: uses original shift_labels (all tokens), not ce_labels (subsampled)
        l2_loss = torch.tensor(0.0, device=fmt_device)
        if lambda_l2 > 0:
            valid_mask = (shift_labels != -100)
            fmt_mask = valid_mask & torch.isin(shift_labels, fmt_ids_device)
            if fmt_mask.any():
                format_logits = shift_logits[fmt_mask]
                l2_loss = lambda_l2 * torch.sum(format_logits ** 2) / format_logits.size(0)

        return ce_loss + l2_loss

    return _plw_l2_loss


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------


def _load_or_build_dataset(
    train_data: list[dict],
    tokenizer,
    format_fn: callable,
    cache_dir: str | Path | None = None,
    num_proc: int = 4,
) -> Dataset:
    """Load cached dataset or build from scratch, caching the result."""
    if cache_dir:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            print(f"[trainer] Loading cached dataset from {cache_path}")
            return Dataset.load_from_disk(str(cache_path))
        print(f"[trainer] Building dataset (cache miss: {cache_path})")
        ds = Dataset.from_list(train_data).map(format_fn, num_proc=num_proc)
        ds.save_to_disk(str(cache_path))
        print(f"[trainer] Dataset cached to {cache_path}")
        return ds
    return Dataset.from_list(train_data).map(format_fn, num_proc=num_proc)


class _ProbeSFTTrainer(SFTTrainer):
    """SFTTrainer subclass that supports post-creation optimizer wrapping.

    Used by probe modules (e.g., MGPO) that need to modify the optimizer
    after it is constructed but before training begins.
    """

    def __init__(
        self,
        *args,
        optimizer_wrapper: Callable | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._optimizer_wrapper = optimizer_wrapper

    def create_optimizer(self):
        self.optimizer = super().create_optimizer()
        if self._optimizer_wrapper is not None:
            self.optimizer = self._optimizer_wrapper(self.optimizer)
            print("[trainer] Optimizer wrapped via probe (e.g., MGPO)")
        return self.optimizer


def train_standard_lora(
    model: PeftModel,
    tokenizer,
    train_data: list[dict],
    output_dir: str | Path,
    num_epochs: int = 1,
    batch_size: int = 2,
    lr: float = 2e-5,
    max_steps: int | None = None,
    max_seq_length: int = 2048,
    gradient_accumulation_steps: int = 4,
    save_steps: int = 5,
    save_total_limit: int = 3,  # -1 = keep all checkpoints
    eval_data: list[dict] | None = None,
    packing: bool = False,  # requires Flash Attention to avoid cross-contamination
    dataset_num_proc: int = 1,  # Windows spawn forks reimport modules; use 1
    dataset_cache_dir: str | Path | None = None,
    use_liger: bool = False,  # Windows: manual apply_liger; Linux: use_liger_kernel
    optim: str = "adamw_torch",  # "adamw_8bit" for 9B VRAM savings
    warmup_ratio: float = 0.0,  # 0.03-0.05 for SSM/hybrid architectures
    resume_from_checkpoint: str | bool | None = False,  # resume path or False
    gradient_checkpointing: bool = True,  # extra defense layer: SFTConfig + model-level
    seed: int = 42,
) -> Path:
    """Standard SFT fine-tuning (no FBC penalty)."""
    return _train_sft(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        max_steps=max_steps,
        max_seq_length=max_seq_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        eval_data=eval_data,
        packing=packing,
        dataset_num_proc=dataset_num_proc,
        dataset_cache_dir=dataset_cache_dir,
        use_liger=use_liger,
        lambda_fbc=None,
        optim=optim,
        warmup_ratio=warmup_ratio,
        resume_from_checkpoint=resume_from_checkpoint,
        gradient_checkpointing=gradient_checkpointing,
        seed=seed,
    )


def train_fr_lora(
    model: PeftModel,
    tokenizer,
    train_data: list[dict],
    output_dir: str | Path,
    lambda_fbc: float = 0.5,
    num_epochs: int = 1,
    batch_size: int = 2,
    lr: float = 2e-5,
    max_steps: int | None = None,
    max_seq_length: int = 2048,
    gradient_accumulation_steps: int = 4,
    save_steps: int = 5,
    eval_data: list[dict] | None = None,
    packing: bool = False,  # requires Flash Attention to avoid cross-contamination
    dataset_num_proc: int = 1,  # Windows spawn forks reimport modules; use 1
    dataset_cache_dir: str | Path | None = None,
    use_liger: bool = False,  # Windows: manual apply_liger; Linux: use_liger_kernel
    seed: int = 42,
) -> Path:
    """FR-LoRA fine-tuning with FBC penalty in loss."""
    return _train_sft(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        max_steps=max_steps,
        max_seq_length=max_seq_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        eval_data=eval_data,
        packing=packing,
        dataset_num_proc=dataset_num_proc,
        dataset_cache_dir=dataset_cache_dir,
        use_liger=use_liger,
        lambda_fbc=lambda_fbc,
        seed=seed,
    )


def train_fr_lora_v2(
    model: PeftModel,
    tokenizer,
    train_data: list[dict],
    output_dir: str | Path,
    alpha: float = 0.3,
    lambda_l2: float = 0.01,
    sample_ratio: float = 1.0,
    num_epochs: int = 1,
    batch_size: int = 2,
    lr: float = 2e-5,
    max_steps: int | None = None,
    max_seq_length: int = 2048,
    gradient_accumulation_steps: int = 4,
    save_steps: int = 5,
    save_total_limit: int = 3,  # -1 = keep all checkpoints
    eval_data: list[dict] | None = None,
    packing: bool = False,
    dataset_num_proc: int = 1,  # Windows spawn forks reimport modules; use 1
    dataset_cache_dir: str | Path | None = None,
    use_liger: bool = False,
    resume_from_checkpoint: str | bool = False,  # False = no resume; str = explicit checkpoint path
    seed: int = 42,
) -> Path:
    """FR-LoRA v2: PLW (Progressive Loss Weighting) + Logit L2 penalty.

    resume_from_checkpoint defaults to False (explicit "do not resume")
    rather than None, to prevent HF Trainer from auto-detecting stale
    checkpoints in output_dir. Pass an explicit path (str) to resume.
    """
    # Auto-detect format tokens across architectures (ChatML / Gemma / Llama)
    _FORMAT_TOKEN_CANDIDATES = [
        ("<|im_start|>", "<|im_end|>"),      # ChatML (Qwen)
        ("<|turn>", "<turn|>"),  # Gemma
        ("<|begin_of_text|>", "<|end_of_text|>"),  # Llama 3
    ]
    format_ids: set[int] = set()
    unk = tokenizer.unk_token_id
    for start_tok, end_tok in _FORMAT_TOKEN_CANDIDATES:
        s = tokenizer.convert_tokens_to_ids(start_tok)
        e = tokenizer.convert_tokens_to_ids(end_tok)
        # convert_tokens_to_ids 对未知 token 返回 unk_token_id（非 None），需排除
        if s is not None and s != unk:
            format_ids.add(s)
        if e is not None and e != unk:
            format_ids.add(e)
        if format_ids:
            break  # Found a matching set

    if len(format_ids) == 0:
        raise RuntimeError(
            f"No valid format token IDs found in tokenizer "
            f"(tried: {[c[0] for c in _FORMAT_TOKEN_CANDIDATES]})"
        )

    compute_loss_func = make_plw_l2_loss_func(
        alpha=alpha, lambda_l2=lambda_l2, format_token_ids=format_ids,
        sample_ratio=sample_ratio,
    )
    return _train_sft(
        model=model, tokenizer=tokenizer, train_data=train_data,
        output_dir=output_dir, num_epochs=num_epochs,
        batch_size=batch_size, lr=lr, max_steps=max_steps,
        max_seq_length=max_seq_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps, save_total_limit=save_total_limit,
        eval_data=eval_data,
        packing=packing, dataset_num_proc=dataset_num_proc,
        dataset_cache_dir=dataset_cache_dir,
        use_liger=use_liger,
        compute_loss_func=compute_loss_func,
        resume_from_checkpoint=resume_from_checkpoint,
        seed=seed,
    )


def _apply_liger_kernel(model: PeftModel) -> None:
    """Apply liger kernel matching the model's architecture.

    Dispatches via liger's own model_type mapping (keeps a single source of
    truth across liger upgrades). Raises on unsupported model_type — a loud
    failure at startup is preferable to silently patching the wrong kernel.
    """
    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN

    model_type = getattr(model.config, "model_type", None)
    apply_fn = MODEL_TYPE_TO_APPLY_LIGER_FN.get(model_type)
    if apply_fn is None:
        raise ValueError(
            f"[trainer] use_liger=True but model_type '{model_type}' has no "
            f"liger kernel mapping; supported: {sorted(MODEL_TYPE_TO_APPLY_LIGER_FN)}"
        )
    apply_fn()
    print(f"[trainer] liger kernel applied: model_type='{model_type}' -> {apply_fn.__name__}")


def _train_sft(
    model: PeftModel,
    tokenizer,
    train_data: list[dict],
    output_dir: str | Path,
    num_epochs: int = 1,
    batch_size: int = 2,
    lr: float = 2e-5,
    max_steps: int | None = None,
    max_seq_length: int = 2048,
    gradient_accumulation_steps: int = 4,
    packing: bool = False,  # requires Flash Attention to avoid cross-contamination
    dataset_num_proc: int = 1,  # Windows spawn forks reimport modules; use 1
    dataset_cache_dir: str | Path | None = None,
    use_liger: bool = False,  # Windows: manual apply_liger; Linux: use_liger_kernel
    save_steps: int = 5,
    save_total_limit: int = 3,  # -1 = keep all checkpoints
    eval_data: list[dict] | None = None,
    lambda_fbc: float | None = None,
    compute_loss_func: Callable | None = None,  # externally provided loss (v2 PLW+L2)
    optimizer_wrapper: Callable | None = None,  # wrap optimizer post-creation (MGPO etc.)
    format_fn: Callable | None = None,  # override default ChatML format (augmentation)
    resume_from_checkpoint: str | bool | None = False,  # False = no resume; str = explicit path; None = no resume
    optim: str = "adamw_torch",  # "adamw_torch" | "adamw_8bit" (saves ~200MB on 9B)
    warmup_ratio: float = 0.0,  # 0.03-0.05 recommended for SSM/hybrid architectures
    gradient_checkpointing: bool = False,  # 9B model MUST set True (VRAM >90%)
    seed: int = 42,
) -> Path:
    """Shared SFT training logic.

    resume_from_checkpoint defaults to False (explicit "do not resume")
    rather than None, to clearly signal that auto-resume is disabled.
    Pass an explicit checkpoint path (str) to resume from a specific checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gradient checkpointing: mandatory on Windows to prevent CUDA allocator
    # fragmentation slowdown (75s→3s at >90% GPU memory). Safe practice everywhere.
    model.train()
    model.gradient_checkpointing_enable()

    # Build ChatML-formatted dataset
    SYSTEM_PROMPT = "你是控制科学专家。请准确、专业、完整地回答以下问题。"

    def _format_chatml(example: dict) -> dict:
        system_prompt = example.get("_system_prompt", SYSTEM_PROMPT)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    # Allow probe-driven format override (e.g., augmentation variants)
    _fmt_fn = format_fn if format_fn is not None else _format_chatml

    # Build or load cached ChatML-formatted dataset
    dataset = _load_or_build_dataset(
        train_data, tokenizer, _fmt_fn,
        cache_dir=dataset_cache_dir, num_proc=dataset_num_proc,
    )
    eval_dataset = None
    if eval_data:
        eval_dataset = Dataset.from_list(eval_data).map(
            _fmt_fn, num_proc=1
        )

    effective_batch = batch_size * gradient_accumulation_steps
    total_steps = max_steps or max(1, len(train_data) // effective_batch * num_epochs)

    # liger_kernel: manual injection (safe on Windows, avoids TRL integration crash)
    if use_liger:
        _apply_liger_kernel(model)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        max_steps=max_steps if max_steps else -1,
        max_length=max_seq_length,
        packing=packing,
        loss_type="nll",
        tf32=True,
        dataloader_pin_memory=True,
        learning_rate=lr,
        logging_steps=5,
        save_steps=save_steps,
        save_total_limit=save_total_limit if save_total_limit > 0 else None,
        warmup_steps=0,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        optim=optim,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=save_steps,
        load_best_model_at_end=False,
        report_to="none",
        seed=seed,
    )

    # Build compute_loss_func if not externally provided
    if compute_loss_func is None and lambda_fbc is not None and lambda_fbc > 0:
        # Auto-detect format tokens (ChatML / Gemma / Llama)
        _fmt_candidates = [
            ("<|im_start|>", "<|im_end|>"),
            ("<|turn>", "<turn|>"),
            ("<|begin_of_text|>", "<|end_of_text|>"),
        ]
        _found_start, _found_end = None, None
        for _s, _e in _fmt_candidates:
            _sid = tokenizer.convert_tokens_to_ids(_s)
            _eid = tokenizer.convert_tokens_to_ids(_e)
            # convert_tokens_to_ids 对未知 token 返回 unk（非 None），需排除
            if (_sid is not None and _eid is not None
                    and _sid != tokenizer.unk_token_id
                    and _eid != tokenizer.unk_token_id):
                _found_start, _found_end = _sid, _eid
                break
        unk_id = tokenizer.unk_token_id
        if (_found_start is not None and _found_end is not None
                and _found_start != unk_id and _found_end != unk_id):
            compute_loss_func = make_fbc_compute_loss_func(
                lambda_fbc=lambda_fbc,
                chatml_start_id=_found_start,
                chatml_end_id=_found_end,
            )
            print(f"[trainer] FBC penalty enabled: λ={lambda_fbc}")
        else:
            print("[trainer] WARNING: format tokens not found, FBC disabled")

    # ── Log hygiene: suppress HF+TRL internal INFO noise before trainer creation ──
    import logging
    logging.getLogger("transformers.trainer").setLevel(logging.WARNING)
    logging.getLogger("trl").setLevel(logging.WARNING)

    # Create trainer — use wrapping subclass when optimizer_wrapper is set
    # (SFTTrainer creates optimizer lazily in train(), so we must subclass
    # to intercept create_optimizer(). See _ProbeSFTTrainer above.)
    if optimizer_wrapper is not None:
        trainer = _ProbeSFTTrainer(
            optimizer_wrapper=optimizer_wrapper,
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            compute_loss_func=compute_loss_func,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            compute_loss_func=compute_loss_func,
        )

    print(f"[trainer] Starting training: {total_steps} steps, {num_epochs} epoch(s)")
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(CompactLogCallback())
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final adapter
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[trainer] Final adapter saved to {final_dir}")

    return final_dir


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from mineru_eval.finetune.adapter_loader import load_peft_model
    from mineru_eval.finetune.ppl_probe import load_jsonl_samples

    parser = argparse.ArgumentParser(description="Smoke-test FR-LoRA training (10 steps)")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("models/qlora-qwen4b-cross-final"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/finetune/medical/train.jsonl"),
    )
    parser.add_argument(
        "--max-samples", type=int, default=100,
    )
    parser.add_argument(
        "--max-steps", type=int, default=10,
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048,
        help="Max token length for training sequences",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=1,
        help="Gradient accumulation steps (default 1 for smoke test)",
    )
    parser.add_argument(
        "--no-packing", action="store_true",
        help="Disable sequence packing",
    )
    parser.add_argument(
        "--lambda-fbc", type=float, default=0.0,
        help="FBC penalty weight (0=standard, 0.5=moderate)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("models/fr_lora_smoke_test"),
    )
    args = parser.parse_args()

    if not args.adapter.is_dir():
        print(f"[SKIP] Adapter not found: {args.adapter}")
        exit(0)
    if not args.data.exists():
        print(f"[SKIP] Data not found: {args.data}")
        exit(0)

    print("[trainer] Loading model + adapter...")
    model, tokenizer = load_peft_model(adapter_path=args.adapter, is_trainable=True)
    model.train()

    samples = load_jsonl_samples(args.data, max_samples=args.max_samples)
    print(f"[trainer] Loaded {len(samples)} training samples")

    use_fbc = args.lambda_fbc > 0
    mode = "FR-LoRA" if use_fbc else "Standard SFT"
    print(f"\n[trainer] {mode} smoke test: {args.max_steps} steps" +
          (f", λ={args.lambda_fbc}" if use_fbc else ""))

    if use_fbc:
        final_path = train_fr_lora(
            model=model, tokenizer=tokenizer, train_data=samples,
            output_dir=args.output_dir, lambda_fbc=args.lambda_fbc,
            max_steps=args.max_steps, batch_size=1,
            max_seq_length=args.max_seq_length,
            gradient_accumulation_steps=args.grad_accum,
            packing=not args.no_packing,
            dataset_num_proc=1,
        )
    else:
        final_path = train_standard_lora(
            model=model, tokenizer=tokenizer, train_data=samples,
            output_dir=args.output_dir, max_steps=args.max_steps, batch_size=1,
            max_seq_length=args.max_seq_length,
            gradient_accumulation_steps=args.grad_accum,
            packing=not args.no_packing,
            dataset_num_proc=1,
        )

    print(f"\n✅ Trainer smoke test passed — adapter saved to {final_path}")
