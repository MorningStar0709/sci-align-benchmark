"""LoRA format-robustness probe modules.

Each probe is a stateless function registered via @register decorator.
The orchestrator dispatches probes by name from the YAML configuration.

Probe types and their action targets:
  - dora:     modifies LoraConfig kwargs before PeftModel.from_pretrained()
  - ftd:      returns a compute_loss_func with Format Token Dropout
  - augment:  returns a format_fn wrapper for ChatML variant augmentation
  - mgpo:     wraps torch.optim.Optimizer with momentum-based perturbation
  - mgpo_ar:  mgpo variant — adaptive rho scaled by avg sequence length
  - mgpo_eg:  mgpo variant — entropy-gated rho via gradient direction CV
  - mgpo_fa:  mgpo variant — format-aware, perturb only last-layer LoRA params
"""

from __future__ import annotations

import math
import random
import re
from typing import Any, Callable

import torch

# ── Probe Registry ────────────────────────────────────────────────────────

PROBE_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    """Register a probe function for configuration-driven dispatch."""

    def deco(fn: Callable) -> Callable:
        PROBE_REGISTRY[name] = fn
        return fn

    return deco


# ── DoRA ──────────────────────────────────────────────────────────────────

@register("dora")
def probe_dora(config_kwargs: dict, cfg: dict | None = None) -> dict:
    """Enable DoRA (Weight-Decomposed Low-Rank Adaptation).

    Must be applied BEFORE PeftModel.from_pretrained() — DoRA's magnitude
    vector is initialized during adapter loading and cannot be toggled
    on an already-loaded model.

    Args:
        config_kwargs: LoraConfig keyword arguments dict (modified in-place).
        cfg: Unused (kept for interface consistency).
    """
    config_kwargs["use_dora"] = True
    return config_kwargs


# ── FTD (Format Token Dropout) ────────────────────────────────────────────

@register("ftd")
def probe_ftd(tokenizer, cfg: dict | None = None) -> Callable:
    """Create a compute_loss_func with Format Token Dropout.

    Randomly masks ChatML format tokens (<|im_start|>, <|im_end|>) in
    labels with probability *drop_rate*, forcing the model to rely less
    on formatting cues for perplexity reduction.

    This is the simplified replacement for EntroDrop — no per-token
    entropy computation required.

    Args:
        tokenizer: Tokenizer with ChatML special tokens.
        cfg: Optional config dict with 'drop_rate' (default 0.3).

    Returns:
        A compute_loss_func compatible with SFTTrainer.
    """
    drop_rate = (cfg or {}).get("drop_rate", 0.3)
    # Auto-detect format tokens across architectures (ChatML / Gemma / Llama)
    _fmt_candidates = [
        ("<|im_start|>", "<|im_end|>"),
        ("<|turn>", "<turn|>"),
        ("<|begin_of_text|>", "<|end_of_text|>"),
    ]
    unk = tokenizer.unk_token_id
    fmt_ids: list[int] = []
    for _s, _e in _fmt_candidates:
        _sid = tokenizer.convert_tokens_to_ids(_s)
        _eid = tokenizer.convert_tokens_to_ids(_e)
        _cand = [tid for tid in (_sid, _eid) if tid is not None and tid != unk]
        if _cand:
            fmt_ids = _cand
            break

    if not fmt_ids:
        raise RuntimeError(
            "[probes:ftd] format tokens not found in tokenizer — "
            "FTD cannot identify format positions"
        )

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def _ftd_loss(outputs, labels, num_items_in_batch=None):
        # Randomly mask format token positions in labels
        modified = labels.clone()
        for tid in fmt_ids:
            fmt_mask = (labels == tid)
            drop = torch.rand(labels.shape, device=labels.device) < drop_rate
            modified[fmt_mask & drop] = -100

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = modified[..., 1:].contiguous()
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        # Guard against all-masked batch (→ NaN with reduction='mean')
        if (flat_labels != -100).sum() == 0:
            return torch.tensor(0.0, device=flat_logits.device, requires_grad=True)
        return loss_fn(flat_logits, flat_labels)

    return _ftd_loss


# ── Augmentation (ChatML Format Variant) ──────────────────────────────────

@register("augment")
def probe_augment(tokenizer, cfg: dict | None = None) -> Callable:
    """Create a format_fn with random ChatML format variant augmentation.

    Randomly varies the system prompt phrasing to create format diversity
    without changing semantic content. This is the simplified replacement
    for Contrastive — no dual forward pass or InfoNCE loss required.

    Args:
        tokenizer: Tokenizer with apply_chat_template support.
        cfg: Optional config dict with 'aug_rate' (default 0.5).

    Returns:
        A complete format_fn (example -> {"text": str}) for _train_sft.
    """
    aug_rate = (cfg or {}).get("aug_rate", 0.5)

    SYSTEM_VARIANTS = [
        "你是控制科学专家。请准确、专业、完整地回答以下问题。",
        "你是控制理论领域的专家。请用专业、准确的方式回答下列问题。",
        "你是一位控制科学与工程专家。请完整准确地回答以下问题。",
        "作为控制科学专家，请用准确专业的语言回答以下问题。",
        "你是控制系统领域的专家。请准确、专业、完整地回答以下问题。",
    ]
    DEFAULT_SYSTEM = SYSTEM_VARIANTS[0]

    def _aug_format(example: dict) -> dict:
        # Use pre-seeded system prompt if provided (for reproducibility),
        # otherwise randomly select variant based on aug_rate.
        system = example.get("_system_prompt")
        if system is None:
            system = (
                random.choice(SYSTEM_VARIANTS)
                if random.random() < aug_rate
                else DEFAULT_SYSTEM
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    return _aug_format


# ── MGPO (Momentum-based Gradient Perturbation Optimizer) ─────────────────

@register("mgpo")
def probe_mgpo(optimizer: torch.optim.Optimizer, cfg: dict | None = None):
    """Wrap optimizer with momentum-based gradient perturbation.

    Adds a small perturbation ε = ρ · m/‖m‖ to each gradient before the
    optimizer step, where m is an EMA of historical gradients. This
    simulates exploratory behavior along consistent gradient directions
    without requiring a second forward pass (unlike full SAM).

    Args:
        optimizer: torch.optim.Optimizer instance.
        cfg: Optional config dict with 'rho' (default 0.05)
             and 'ema_decay' (default 0.9).

    Returns:
        MGPOWrapper instance (delegates all methods to underlying optimizer).
    """
    rho = (cfg or {}).get("rho", 0.05)
    ema_decay = (cfg or {}).get("ema_decay", 0.9)
    return _MGPOWrapper(optimizer, rho=rho, ema_decay=ema_decay)


class _MGPOWrapper(torch.optim.Optimizer):
    """Momentum-based Gradient Perturbation as torch.optim.Optimizer subclass.

    Inheriting torch.optim.Optimizer ensures compatibility with HF Accelerate's
    isinstance(optimizer, torch.optim.Optimizer) check for mixed-precision
    gradient scaling and AcceleratedOptimizer wrapping.

    Maintains an EMA of gradients per parameter and adds a scaled,
    normalized perturbation in the EMA direction before each step.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, rho: float = 0.05,
                 ema_decay: float = 0.9):
        # Bypass Optimizer.__init__ required-params check — delegate to inner
        self._optimizer = optimizer
        self.rho = rho
        self.ema_decay = ema_decay
        self._ema: dict[tuple, torch.Tensor] = {}
        # Proxy param_groups, state, defaults to inner optimizer (Accelerator compat)
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self.defaults = optimizer.defaults

    def zero_grad(self, *args: Any, **kwargs: Any) -> Any:
        return self._optimizer.zero_grad(*args, **kwargs)

    def step(self, closure: Callable | None = None) -> Any:
        # Accumulate EMA and perturb gradients before step
        for group_idx, group in enumerate(self._optimizer.param_groups):
            for p_idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                key = (group_idx, p_idx)  # stable across param lifecycle
                # Update EMA of gradients
                if key not in self._ema:
                    self._ema[key] = torch.zeros_like(p.grad)
                self._ema[key].mul_(self.ema_decay).add_(
                    p.grad, alpha=1.0 - self.ema_decay
                )
                # Add perturbation in EMA direction
                g_norm = self._ema[key].norm()
                if g_norm > 1e-8:
                    p.grad.add_(self._ema[key], alpha=self.rho / g_norm)

        return self._optimizer.step(closure)

    def state_dict(self) -> dict:
        """Return inner optimizer state_dict augmented with MGPO EMA."""
        state = self._optimizer.state_dict()
        state["mgpo_ema"] = {
            f"{g}_{p}": v.detach().cpu().clone()
            for (g, p), v in self._ema.items()
        }
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore inner optimizer state and MGPO EMA."""
        ema_data = state_dict.pop("mgpo_ema", {})
        params = [
            p for g in self._optimizer.param_groups for p in g["params"]
        ]
        ref_device = params[0].device if params else torch.device("cpu")
        for key_str, tensor in ema_data.items():
            g_str, p_str = key_str.split("_", 1)
            self._ema[(int(g_str), int(p_str))] = tensor.to(ref_device)
        self._optimizer.load_state_dict(state_dict)

    def __getattr__(self, name: str) -> Any:
        # Delegate all other attributes to the underlying optimizer
        return getattr(self._optimizer, name)


# ── MGPO Variants ─────────────────────────────────────────────────────────
# Three orthogonal improvements to MGPO for cross-domain robustness.
# Each inherits _MGPOWrapper; existing mgpo probe is untouched.


# ── 1A: Adaptive Rho (mgpo_ar) ─────────────────────────────────────────────


class _MGPOAdaptiveRhoWrapper(_MGPOWrapper):
    """Adaptive Rho MGPO: scale rho by domain average sequence length.

    rho_effective = base_rho * min(1.0, avg_seq_len / ref_len)
    - medical (4339 chars): rho = full (1.0)
    - control_sci (373 chars): rho = reduced

    NOTE: avg_seq_len is a fixed value set at initialization (per-domain),
    NOT computed dynamically per batch. Configure it per dataset.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, rho: float = 0.05,
                 ema_decay: float = 0.9, ref_len: int = 500,
                 avg_seq_len: int = 500):
        super().__init__(optimizer, rho=rho, ema_decay=ema_decay)
        if ref_len <= 0:
            raise ValueError(f"[mgpo_ar] ref_len must be > 0, got {ref_len}")
        self.ref_len = ref_len
        self.avg_seq_len = avg_seq_len

    def step(self, closure=None):
        scale = min(1.0, self.avg_seq_len / self.ref_len)
        effective_rho = self.rho * scale
        orig_rho = self.rho
        self.rho = effective_rho
        try:
            return super().step(closure)
        finally:
            self.rho = orig_rho


@register("mgpo_ar")
def probe_mgpo_ar(optimizer: torch.optim.Optimizer,
                  cfg: dict | None = None):
    """Adaptive Rho MGPO: scale rho by avg sequence length."""
    cfg = cfg or {}
    return _MGPOAdaptiveRhoWrapper(
        optimizer,
        rho=cfg.get("rho", 0.05),
        ema_decay=cfg.get("ema_decay", 0.9),
        ref_len=cfg.get("ref_len", 500),
        avg_seq_len=cfg.get("avg_seq_len", 500),
    )


# ── 1B: Entropy-Gated (mgpo_eg) ────────────────────────────────────────────


class _MGPOEntropyGatedWrapper(_MGPOWrapper):
    """Entropy-Gated MGPO: perturb only when gradient directions diverge.

    When gradient norms within a batch are highly consistent (low entropy)
    -> rho->0, don't perturb. When divergent (high entropy) ->
    rho->full, perturb to explore.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, rho: float = 0.05,
                 ema_decay: float = 0.9, threshold: float = 0.5,
                 temperature: float = 1.0):
        super().__init__(optimizer, rho=rho, ema_decay=ema_decay)
        if temperature <= 0:
            raise ValueError(
                f"[mgpo_eg] temperature must be > 0, got {temperature}")
        self.threshold = threshold
        self.temperature = temperature

    def _compute_grad_entropy(self) -> float:
        """Compute gradient direction entropy across params.

        Uses coefficient of variation of per-param gradient norms as a
        proxy for directional consistency (cheaper than pairwise cosine).
        """
        norms = []
        for group in self._optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.norm().item())
        if len(norms) < 2:
            return 1.0  # default to full perturbation
        t = torch.tensor(norms)
        if t.mean() < 1e-8:
            return 0.0  # all-zero grads -> no perturbation
        return min(1.0, (t.std() / t.mean()).item())

    def step(self, closure=None):
        entropy = self._compute_grad_entropy()
        gate = 1.0 / (1.0 + math.exp(-(entropy - self.threshold) / self.temperature))
        orig_rho = self.rho
        self.rho = self.rho * gate
        try:
            return super().step(closure)
        finally:
            self.rho = orig_rho


@register("mgpo_eg")
def probe_mgpo_eg(optimizer: torch.optim.Optimizer,
                  cfg: dict | None = None):
    """Entropy-Gated MGPO: perturb only when gradient directions diverge."""
    cfg = cfg or {}
    return _MGPOEntropyGatedWrapper(
        optimizer,
        rho=cfg.get("rho", 0.05),
        ema_decay=cfg.get("ema_decay", 0.9),
        threshold=cfg.get("entropy_threshold", 0.5),
        temperature=cfg.get("entropy_temperature", 1.0),
    )


# ── 1C: Format-Aware (mgpo_fa) ─────────────────────────────────────────────


def _identify_last_layer_keys(
    optimizer: torch.optim.Optimizer,
    param_names: list[str],
    n_last: int = 2,
) -> set[tuple]:
    """Find (group_idx, param_idx) keys for the last N LoRA layers."""
    layer_re = re.compile(r"layers\.(\d+)\.")
    param_to_layer: dict[tuple, int] = {}

    flat_idx = 0
    for gi, group in enumerate(optimizer.param_groups):
        for pi, p in enumerate(group["params"]):
            if flat_idx < len(param_names):
                m = layer_re.search(param_names[flat_idx])
                if m:
                    param_to_layer[(gi, pi)] = int(m.group(1))
            flat_idx += 1

    if not param_to_layer:
        # Fallback: use last n_last*2 params (lora_A + lora_B pairs)
        total = sum(len(g["params"]) for g in optimizer.param_groups)
        keys: set[tuple] = set()
        flat_idx = 0
        for gi, group in enumerate(optimizer.param_groups):
            for pi, p in enumerate(group["params"]):
                if flat_idx >= total - n_last * 2:
                    keys.add((gi, pi))
                flat_idx += 1
        return keys

    max_layer = max(param_to_layer.values())
    return {k for k, v in param_to_layer.items()
            if v >= max_layer - n_last + 1}


class _MGPOFormatAwareWrapper(_MGPOWrapper):
    """Format-Aware MGPO: perturb only last-layer LoRA parameters.

    Hypothesis: last-layer LoRA params carry the strongest gradient signal
    for format tokens. Perturbing only those layers preserves format
    robustness training while avoiding noise on content/formula layers.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, rho: float = 0.05,
                 ema_decay: float = 0.9,
                 format_layer_indices: set[tuple] | None = None):
        super().__init__(optimizer, rho=rho, ema_decay=ema_decay)
        self._format_keys: set[tuple] = format_layer_indices or set()

    def step(self, closure=None):
        for gi, group in enumerate(self._optimizer.param_groups):
            for pi, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                key = (gi, pi)
                # EMA update (always, for state consistency)
                if key not in self._ema:
                    self._ema[key] = torch.zeros_like(p.grad)
                self._ema[key].mul_(self.ema_decay).add_(
                    p.grad, alpha=1.0 - self.ema_decay)
                # Perturbation: skip non-format layers
                if key not in self._format_keys:
                    continue
                g_norm = self._ema[key].norm()
                if g_norm > 1e-8:
                    p.grad.add_(self._ema[key], alpha=self.rho / g_norm)
        return self._optimizer.step(closure)


@register("mgpo_fa")
def probe_mgpo_fa(optimizer: torch.optim.Optimizer,
                  cfg: dict | None = None):
    """Format-Aware MGPO — perturb only last-layer LoRA parameters.

    Accepts `param_names` (list[str]) in cfg to identify last layer.
    Falls back to last N param groups if names unavailable.
    """
    cfg = cfg or {}
    param_names = cfg.get("param_names", [])
    n_last_layers = cfg.get("n_last_layers", 2)

    format_keys = _identify_last_layer_keys(
        optimizer, param_names, n_last_layers)

    return _MGPOFormatAwareWrapper(
        optimizer,
        rho=cfg.get("rho", 0.05),
        ema_decay=cfg.get("ema_decay", 0.9),
        format_layer_indices=format_keys,
    )


# ── Smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Probe Registry ===")
    for name, fn in PROBE_REGISTRY.items():
        print(f"  {name}: {fn.__name__}")

    expected = ("dora", "ftd", "augment", "mgpo",
                "mgpo_ar", "mgpo_eg", "mgpo_fa")
    for n in expected:
        assert n in PROBE_REGISTRY, f"{n} probe not registered"
    print(f"\n✅ All {len(expected)} probes registered successfully")

    # DoRA smoke
    cfg = {"r": 8, "lora_alpha": 16}
    result = probe_dora(cfg)
    assert result["use_dora"] is True, "DoRA not enabled"
    print("✅ probe_dora: use_dora=True set")

    # MGPO variant construction smoke tests
    dummy_params = [torch.nn.Parameter(torch.randn(3))]
    dummy_opt = torch.optim.SGD(dummy_params, lr=0.01)
    dummy_g = torch.randn(3)
    dummy_params[0].grad = dummy_g
    dummy_opt.step()  # populate state

    result_ar = probe_mgpo_ar(dummy_opt, {"avg_seq_len": 300, "ref_len": 500})
    assert isinstance(result_ar, _MGPOAdaptiveRhoWrapper)
    assert result_ar.ref_len == 500
    assert result_ar.avg_seq_len == 300
    print("✅ probe_mgpo_ar: construction OK")

    result_eg = probe_mgpo_eg(dummy_opt, {"entropy_threshold": 0.3})
    assert isinstance(result_eg, _MGPOEntropyGatedWrapper)
    assert result_eg.threshold == 0.3
    print("✅ probe_mgpo_eg: construction OK")

    result_fa = probe_mgpo_fa(dummy_opt, {"n_last_layers": 2})
    assert isinstance(result_fa, _MGPOFormatAwareWrapper)
    assert len(result_fa._format_keys) > 0
    print("✅ probe_mgpo_fa: construction OK")

    # Validate error handling
    try:
        probe_mgpo_ar(dummy_opt, {"ref_len": 0})
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✅ probe_mgpo_ar: ref_len=0 validation OK")

    try:
        probe_mgpo_eg(dummy_opt, {"entropy_temperature": 0})
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✅ probe_mgpo_eg: temperature=0 validation OK")

    print("\n=== probes.py smoke test PASSED ===")
