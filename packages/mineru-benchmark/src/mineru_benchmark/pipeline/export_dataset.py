import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]  # restored Phase 5
DEFAULT_INPUT = ROOT / "benchmark" / "dataset" / "core.json"
DEFAULT_OUTPUT = ROOT / "huggingface" / "control-sci-corpus"


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(questions, output_path):
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + "\n")


BENCHMARK_DIR = "benchmark"
BENCHMARK_SPLIT = "core"


def format_feature(value):
    if isinstance(value, list):
        return f"list[{value[0]}]" if value else "list[string]"
    return type(value).__name__


def infer_features(questions):
    if not questions:
        return {}
    features = {}
    for key in questions[0]:
        vals = [q.get(key) for q in questions if q.get(key) is not None]
        if not vals:
            features[key] = "string"
            continue
        sample = vals[0]
        if isinstance(sample, list):
            inner = type(sample[0]).__name__ if sample else "string"
            features[key] = f"list[{inner}]"
        elif isinstance(sample, dict):
            features[key] = "object"
        else:
            features[key] = type(sample).__name__
    return features


def write_dataset_info(benchmark_data, output_dir):
    questions = benchmark_data.get("questions", [])
    features = infer_features(questions)
    info = {
        "dataset_name": "control-sci-corpus",
        "version": benchmark_data.get("meta", {}).get("version", "unknown"),
        "description": "ControlSci corpus: Sci-Align benchmark + Sciverse SFT instruction data.",
        "configs": [
            {
                "config_name": "sciverse_sft",
                "description": "Sciverse SFT instruction pairs in ChatML format.",
                "features": {
                    "messages": "list",
                    "lengths": "int64",
                    "_meta": "object",
                },
                "splits": {
                    "train": 785,
                    "validation": 139,
                },
            },
            {
                "config_name": BENCHMARK_DIR,
                "description": "ControlSci Sci-Align benchmark: 500 questions across 4 dimensions.",
                "features": features,
                "splits": {BENCHMARK_SPLIT: len(questions)},
            },
        ],
    }
    (output_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(data, output_dir):
    meta = data.get("meta", {})
    n_q = len(data.get("questions", []))
    text = f"""---
license: cc-by-4.0
language:
- en
- zh
task_categories:
- question-answering
- text-generation
tags:
- control-science
- benchmark
- scialign
- llm-evaluation
- structured-corpus
- instruction-tuning
pretty_name: ControlSci Sci-Align Benchmark & Sciverse SFT
configs:
- config_name: sciverse_sft
  data_files:
  - sciverse_sft/train.jsonl
  - sciverse_sft/val.jsonl
- config_name: benchmark
   data_files:
   - benchmark/core/data-00000-of-00001.jsonl
---

# ControlSci Corpus

Control science structured corpus with two configs: Sci-Align benchmark ({n_q} questions) and Sciverse SFT instruction pairs (924 ChatML entries).

**License:** CC-BY-4.0
**Project:** [MorningStar0709/control-sci](https://github.com/MorningStar0709/control-sci)

---

## Configs

### `benchmark` — Sci-Align Benchmark ({n_q} questions)

4-dimension control science evaluation benchmark generated from the ControlSci structured corpus.

**Split:** `core` ({n_q} questions)
**Load:**
```python
from datasets import load_dataset
ds = load_dataset("MorningStar0709/control-sci-corpus", "benchmark", split="core")
```

### `sciverse_sft` — Sciverse SFT Pairs (924 entries)

Instruction-tuning data in ChatML format, generated from 14 control-science Sciverse papers.

| Split | Count |
|-------|:-----:|
| train | 785 |
| validation | 139 |

**Load:**
```python
ds = load_dataset("MorningStar0709/control-sci-corpus", "sciverse_sft", split="train")
```

---

## Dataset Info

See [`dataset_info.json`](./dataset_info.json) for detailed schema per config.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8", newline="\n")


def try_write_parquet(questions, output_path):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return False

    try:
        rubric_criterion = pa.struct([
            pa.field("max_score", pa.int32()),
            pa.field("weight", pa.float64()),
            pa.field("description", pa.string()),
        ])
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("dimension", pa.string()),
            pa.field("difficulty_level", pa.string()),
            pa.field("control_category", pa.list_(pa.string())),
            pa.field("question", pa.string()),
            pa.field("answer", pa.string()),
            pa.field("reasoning_steps", pa.list_(pa.string())),
            pa.field("source_ref", pa.string()),
            pa.field("model_source", pa.string()),
            pa.field("sensitivity_dimension", pa.string(), nullable=True),
            pa.field("sibling_id", pa.string(), nullable=True),
            pa.field("rubric", pa.struct([
                pa.field("feasibility", rubric_criterion),
                pa.field("method_choice", rubric_criterion),
                pa.field("completeness", rubric_criterion),
                pa.field("innovation", rubric_criterion),
                pa.field("clarity", rubric_criterion),
            ]), nullable=True),
            pa.field("consistency_status", pa.string()),
        ])
        rows = []
        for question in questions:
            row = {k: question.get(k) for k in schema.names}
            rows.append(row)
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, output_path)
        return True
    except (TypeError, ValueError, pa.ArrowException) as e:
        print(f"Warning: Parquet write failed ({e}), falling back to JSONL-only.", flush=True)
        return False


def export_dataset(input_path, output_dir):
    data = load_json(input_path)
    questions = data.get("questions", [])
    data_dir = output_dir / BENCHMARK_DIR / BENCHMARK_SPLIT
    data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(questions, data_dir / "data-00000-of-00001.jsonl")
    parquet_written = try_write_parquet(questions, data_dir / "data-00000-of-00001.parquet")
    write_dataset_info(data, output_dir)
    write_readme(data, output_dir)
    return len(questions), parquet_written


def main():
    parser = argparse.ArgumentParser(description="Export ControlSci benchmark to a HuggingFace-friendly directory.")
    parser.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="Benchmark JSON path.")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="Output dataset directory.")
    parser.add_argument("--push-to-hub", default=None, help="HF dataset repo name (e.g. 'username/control-sci-benchmark'). Uploads output dir.")
    parser.add_argument("--hf-token", default=None, help="HF authentication token. Falls back to HF_TOKEN env var.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    total, parquet_written = export_dataset(input_path, output_dir)
    print(f"Exported {total} question(s) to {output_dir}")
    print(f"Parquet: {'written' if parquet_written else 'skipped (pyarrow not installed)'}")

    if args.push_to_hub:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("huggingface_hub not installed. Run: pip install huggingface_hub")
            return
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            print("HF token required. Pass --hf-token or set HF_TOKEN env var.")
            return
        api = HfApi()
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=args.push_to_hub,
            token=token,
            repo_type="dataset",
            ignore_patterns=[".gitkeep"],
        )
        print(f"Uploaded to https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
