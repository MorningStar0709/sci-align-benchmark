"""Summarize Track 1 training utility evidence from existing QLoRA and PPL artifacts."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ROOT = Path(__file__).resolve().parents[2]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def round_float(value: float) -> float:
    return round(value, 4)


def numeric(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def count_jsonl(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def extract_ppl_overall(data: Dict[str, object]) -> Dict[str, object]:
    overall = data.get("overall")
    if isinstance(overall, dict):
        return {
            "base_avg_ppl": numeric(overall.get("base_avg_ppl")),
            "qlora_avg_ppl": numeric(overall.get("qlora_avg_ppl")),
            "delta_ppl": numeric(overall.get("delta_ppl")),
            "delta_percent": numeric(overall.get("delta_percent")),
        }
    return {
        "base_avg_ppl": numeric(data.get("base_avg_ppl")),
        "qlora_avg_ppl": numeric(data.get("qlora_avg_ppl")),
        "delta_ppl": numeric(data.get("delta_ppl")),
        "delta_percent": numeric(data.get("delta_percent")),
    }


def improvement_label(delta_percent: Optional[float]) -> str:
    if delta_percent is None:
        return "missing_delta"
    if delta_percent < 0:
        return "ppl_improved"
    if delta_percent > 0:
        return "ppl_regressed"
    return "ppl_unchanged"


def ppl_group(name: str, source: Path, data: Dict[str, object], role: str) -> Dict[str, object]:
    overall = extract_ppl_overall(data)
    sample_count = data.get("n_questions", data.get("total_fragments", data.get("train_items")))
    paper_count = data.get("n_papers")
    return {
        "name": name,
        "role": role,
        "status": "available",
        "source": path_text(source),
        "analysis": data.get("analysis", "unknown"),
        "base_model": data.get("base_model", "unknown"),
        "qlora_adapter": data.get("qlora_adapter", "unknown"),
        "sample_count": sample_count,
        "paper_count": paper_count,
        "base_avg_ppl": overall["base_avg_ppl"],
        "qlora_avg_ppl": overall["qlora_avg_ppl"],
        "delta_ppl": overall["delta_ppl"],
        "delta_percent": overall["delta_percent"],
        "ppl_improvement_status": improvement_label(overall["delta_percent"]),
    }


def missing_group(name: str, source: Path, reason: str, role: str) -> Dict[str, object]:
    return {
        "name": name,
        "role": role,
        "status": "missing_group",
        "source": path_text(source),
        "reason": reason,
    }


def load_optional_group(name: str, source: str, role: str) -> Dict[str, object]:
    path = Path(source)
    if not path.exists():
        return missing_group(name, path, "source file not found", role)
    return ppl_group(name, path, load_json(path), role)


def score_delta_if_available(finetuned: str, baseline: str) -> Dict[str, object]:
    finetuned_path = Path(finetuned)
    baseline_path = Path(baseline)
    if not finetuned_path.exists() or not baseline_path.exists():
        return {
            "status": "missing_group",
            "finetuned_source": path_text(finetuned_path),
            "baseline_source": path_text(baseline_path),
            "reason": "one or more score report files not found",
        }
    finetuned_data = load_json(finetuned_path)
    baseline_data = load_json(baseline_path)
    finetuned_score = numeric(finetuned_data.get("overall_score"))
    baseline_score = numeric(baseline_data.get("overall_score"))
    if finetuned_score is None or baseline_score is None:
        return {
            "status": "missing_group",
            "finetuned_source": path_text(finetuned_path),
            "baseline_source": path_text(baseline_path),
            "reason": "overall_score is missing or non-numeric",
        }
    dimension_delta = {}
    finetuned_dims = finetuned_data.get("dimension_scores", {})
    baseline_dims = baseline_data.get("dimension_scores", {})
    if isinstance(finetuned_dims, dict) and isinstance(baseline_dims, dict):
        for dimension in sorted(set(finetuned_dims) & set(baseline_dims)):
            left = numeric(finetuned_dims.get(dimension))
            right = numeric(baseline_dims.get(dimension))
            if left is not None and right is not None:
                dimension_delta[dimension] = round_float(left - right)
    return {
        "status": "available",
        "comparison": "4b_finetuned_vs_4b_baseline",
        "finetuned_source": path_text(finetuned_path),
        "baseline_source": path_text(baseline_path),
        "sample_count": finetuned_data.get("total_questions"),
        "finetuned_score": finetuned_score,
        "baseline_score": baseline_score,
        "delta": round_float(finetuned_score - baseline_score),
        "dimension_delta": dimension_delta,
        "boundary": "Reasoning score delta is reported separately from PPL delta and must not be inferred from perplexity alone.",
    }


def training_split(config: str, train_file: str, val_file: str) -> Dict[str, object]:
    config_path = Path(config)
    train_path = Path(train_file)
    val_path = Path(val_file)
    config_data = load_json(config_path) if config_path.exists() else {}
    training = config_data.get("training", {}) if isinstance(config_data.get("training"), dict) else {}
    data = config_data.get("data", {}) if isinstance(config_data.get("data"), dict) else {}
    return {
        "status": "available" if config_path.exists() else "missing_group",
        "config_source": path_text(config_path),
        "train_file": path_text(train_path),
        "val_file": path_text(val_path),
        "train_items": count_jsonl(train_path),
        "val_items": count_jsonl(val_path),
        "model_name": config_data.get("model", {}).get("name") if isinstance(config_data.get("model"), dict) else None,
        "lora_r": config_data.get("lora", {}).get("r") if isinstance(config_data.get("lora"), dict) else None,
        "num_train_epochs": training.get("num_train_epochs"),
        "max_seq_length": data.get("max_seq_length"),
    }


def ppl_delta(groups: List[Dict[str, object]]) -> Dict[str, object]:
    available = [group for group in groups if group.get("status") == "available"]
    improved = [group for group in available if group.get("ppl_improvement_status") == "ppl_improved"]
    regressed = [group for group in available if group.get("ppl_improvement_status") == "ppl_regressed"]
    strongest = sorted(available, key=lambda item: (item.get("delta_percent") if item.get("delta_percent") is not None else 9999, item.get("name", "")))
    return {
        "available_groups": len(available),
        "missing_groups": len(groups) - len(available),
        "ppl_improved_groups": [group["name"] for group in improved],
        "ppl_regressed_groups": [group["name"] for group in regressed],
        "strongest_ppl_improvement": strongest[0] if strongest else None,
    }


def run(
    ppl: str,
    train_ppl: str,
    output: str,
    controlsci_ppl: str,
    controlsci_9b_ppl: str,
    finetuned_report: str,
    baseline_report: str,
    config: str,
    train_file: str,
    val_file: str,
) -> Dict[str, object]:
    groups = [
        load_optional_group("sciverse_trained_4b", train_ppl, "in_domain_sciverse_train_then_ppl_probe"),
        load_optional_group("sciverse_transfer_9b_existing_adapter", ppl, "cross_source_existing_adapter_ppl_probe"),
        load_optional_group("controlsci_4b_existing_adapter", controlsci_ppl, "controlsci_existing_adapter_ppl_probe"),
        load_optional_group("controlsci_9b_existing_adapter", controlsci_9b_ppl, "controlsci_9b_existing_adapter_ppl_probe"),
    ]
    payload = {
        "schema_version": "1.0",
        "experiment": "track1_training_utility_ablation",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ok" if any(group.get("status") == "available" for group in groups) else "degraded_no_training_artifacts",
        "evaluation_mode": "existing_qlora_ppl_and_score_artifact_summary_no_new_training",
        "inputs": {
            "ppl": path_text(Path(ppl)),
            "train_ppl": path_text(Path(train_ppl)),
            "controlsci_ppl": path_text(Path(controlsci_ppl)),
            "controlsci_9b_ppl": path_text(Path(controlsci_9b_ppl)),
            "finetuned_report": path_text(Path(finetuned_report)),
            "baseline_report": path_text(Path(baseline_report)),
            "config": path_text(Path(config)),
            "train_file": path_text(Path(train_file)),
            "val_file": path_text(Path(val_file)),
        },
        "training_split": training_split(config, train_file, val_file),
        "groups": groups,
        "ppl_delta": ppl_delta(groups),
        "score_delta_if_available": score_delta_if_available(finetuned_report, baseline_report),
        "interpretation": {
            "ppl_improvement": "Negative delta_percent means the adapter assigns lower perplexity to the evaluated text fragments.",
            "reasoning_score_improvement": "Reasoning score must be read from judge reports when available; it is not inferred from PPL.",
            "task5_claim": "The available artifacts support training utility evidence, especially in-domain Sciverse QLoRA PPL reduction, but do not support a blanket reasoning ability improvement claim.",
        },
        "boundary_notes": [
            "This Task 5 summary reuses existing QLoRA, PPL, and judge artifacts and does not run new training or inference.",
            "PPL reduction is evidence of language-model fit to the evaluated corpus, not a direct proof of better control-science reasoning.",
            "Missing groups are recorded as missing_group and no replacement numbers are fabricated.",
            "Adapter checkpoint availability is not assumed; this summary audits JSON artifacts and data/config files present on disk.",
        ],
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Track 1 training utility ablation evidence from existing artifacts.")
    parser.add_argument("--ppl", default="benchmark/eval/results/sciverse_qlora_ppl.json")
    parser.add_argument("--train-ppl", default="benchmark/eval/results/sciverse_qlora_train_ppl.json")
    parser.add_argument("--controlsci-ppl", default="finetune/output/perplexity_delta.json")
    parser.add_argument("--controlsci-9b-ppl", default="finetune/output/perplexity_delta_9b.json")
    parser.add_argument("--finetuned-report", default="finetune/output/eval_finetuned_report.json")
    parser.add_argument("--baseline-report", default="finetune/output/eval_baseline_4b_report.json")
    parser.add_argument("--config", default="finetune/config/qlora_sciverse.yaml")
    parser.add_argument("--train-file", default="finetune/data/sciverse/train.jsonl")
    parser.add_argument("--val-file", default="finetune/data/sciverse/val.jsonl")
    parser.add_argument("--output", default="benchmark/eval/results/track1_training_utility_ablation.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(
        args.ppl,
        args.train_ppl,
        args.output,
        args.controlsci_ppl,
        args.controlsci_9b_ppl,
        args.finetuned_report,
        args.baseline_report,
        args.config,
        args.train_file,
        args.val_file,
    )
    summary = {
        "status": payload["status"],
        "available_groups": payload["ppl_delta"]["available_groups"],
        "missing_groups": payload["ppl_delta"]["missing_groups"],
        "train_items": payload["training_split"].get("train_items"),
        "val_items": payload["training_split"].get("val_items"),
        "strongest_ppl_improvement": payload["ppl_delta"]["strongest_ppl_improvement"].get("name") if payload["ppl_delta"]["strongest_ppl_improvement"] else None,
        "score_delta_status": payload["score_delta_if_available"]["status"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
