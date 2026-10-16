"""Load base model + LoRA adapter via PEFT."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from mineru_eval.finetune.gemma4_constants import GEMMA4_KEY_MAPPING


def _load_base_model(base_model_id: str, dtype, device_map):
    """按 model_type 分发基座加载。gemma4 → text-only CausalLM + key_mapping。

    AutoConfig 失败即 fail-fast：静默降级到 AutoModelForCausalLM 会让 gemma4
    多模态 checkpoint 绕过 key_mapping 分支（审查修复 07-26）。
    """
    try:
        cfg = AutoConfig.from_pretrained(
            base_model_id, trust_remote_code=True, local_files_only=True,
        )
    except OSError as e:
        raise RuntimeError(
            f"Failed to load config for {base_model_id} (local cache broken/missing): {e}"
        ) from e
    model_type = getattr(cfg, "model_type", "")

    if model_type == "gemma4":
        from transformers import Gemma4ForCausalLM
        print("[adapter_loader] gemma4 detected → Gemma4ForCausalLM + key_mapping")
        return Gemma4ForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map=device_map,
            local_files_only=True,
            key_mapping=dict(GEMMA4_KEY_MAPPING),
        )
    return AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True,
    )


def load_peft_model(
    adapter_path: str | Path | None = None,
    base_model_id: str | None = None,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    is_trainable: bool = False,
    use_dora: bool = False,
) -> tuple[PeftModel | AutoModelForCausalLM, AutoTokenizer]:
    """Load base model with optional LoRA adapter, return (model, tokenizer).

    Args:
        base_model_id: HF model ID. Defaults to env FR_LORA_BASE_MODEL
                       or 'principled-intelligence/Qwen3.5-4B-text-only'.
        adapter_path: Path to adapter directory (contains adapter_config.json
                      + adapter_model.safetensors). Pass None to load the
                      bare base model without any LoRA wrapper.
        torch_dtype: Data type for base model loading.
        device_map: Device map strategy.
        is_trainable: Whether to load adapter in training mode (requires_grad=True).
                      Set False for inference-only (PPL measurement).
        use_dora: Enable DoRA (Weight-Decomposed Low-Rank Adaptation).
                  Modifies the LoraConfig before PeftModel.from_pretrained()
                  so the magnitude vector is initialized correctly.
                  Ignored when adapter_path is None.
    """
    if base_model_id is None:
        base_model_id = os.environ.get(
            "FR_LORA_BASE_MODEL",
            "principled-intelligence/Qwen3.5-4B-text-only",
        )

    dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else "auto"

    print(f"[adapter_loader] Loading base model: {base_model_id}")
    # Force offline mode to prevent transformers from trying to download
    # custom_generate.py from HF Hub (unstable in China). All model files
    # are already cached locally.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    base_model = _load_base_model(base_model_id, dtype, device_map)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        local_files_only=True,  # 与 _load_base_model 同款离线防护：防 HF Hub 网络挂起（2026-08-13 Clash 代理事故）
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Bare model mode — return base model without PEFT wrapper.
    # ⚠️ API contract change: when adapter_path=None, returns AutoModelForCausalLM
    # (not PeftModel). Callers MUST check return type before calling PeftModel
    # methods like .merge_and_unload(), .save_pretrained(), etc.
    if adapter_path is None:
        if is_trainable:
            raise ValueError(
                "Training mode requires an adapter path; "
                "bare model cannot be used for LoRA training"
            )
        print("[adapter_loader] Bare model mode (no adapter)")
        return base_model, tokenizer

    adapter = Path(adapter_path)
    if not adapter.is_dir():
        raise ValueError(f"adapter_path must be an existing directory: {adapter_path}")

    print(f"[adapter_loader] Loading adapter: {adapter}")
    # DoRA: modify config before PeftModel construction so the magnitude
    # vector is initialized from the loaded weights.
    peft_config = None
    if use_dora:
        from peft import PeftConfig, LoraConfig
        peft_config = PeftConfig.from_pretrained(str(adapter))
        if not isinstance(peft_config, LoraConfig):
            raise TypeError(
                f"DoRA requires a LoRA adapter, but got "
                f"{type(peft_config).__name__} from {adapter}"
            )
        peft_config.use_dora = True
        peft_config.lora_dropout = 0.0  # enable DoRA optimization path
        print("[adapter_loader] DoRA enabled via config override")
    with warnings.catch_warnings():
        # peft tries to resolve base-model config by filesystem path;
        # when using HF model IDs (not local paths) it emits a harmless
        # "Could not find a config file" warning.  We never modify the
        # vocab, so this is safe to suppress.
        # Also suppresses "Found missing adapter keys" when enabling DoRA
        # on a standard LoRA checkpoint — magnitude vectors are correctly
        # initialized from the base model weights.
        warnings.filterwarnings(
            "ignore",
            message=r"Could not find a config file",
            module=r"peft",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Found missing adapter keys",
            module=r"peft",
        )
        model = PeftModel.from_pretrained(
            base_model, str(adapter),
            config=peft_config,
            is_trainable=is_trainable,
        )
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    dora_tag = " (DoRA)" if use_dora else ""
    print(f"[adapter_loader] Adapter loaded{dora_tag}: {len(model.peft_config)} module(s), {trainable} trainable params")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test adapter loading")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("models/qlora-qwen4b-cross-final"),
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Base model ID (default: env FR_LORA_BASE_MODEL or Qwen3.5-4B)",
    )
    args = parser.parse_args()

    if not args.adapter.is_dir():
        print(f"[SKIP] Adapter not found: {args.adapter}")
        print("  Run this after data migration or provide --adapter path.")
        exit(0)

    model, tokenizer = load_peft_model(
        adapter_path=args.adapter,
        base_model_id=args.model,
    )

    # Quick inference check
    prompt = "Explain the concept of a PID controller in one sentence."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n[smoke] Prompt: {prompt}")
    print(f"[smoke] Response: {response[:200]}...")
    print("\n✅ adapter_loader smoke test passed")
