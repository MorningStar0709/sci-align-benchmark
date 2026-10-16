"""
路径 D 辅助脚本: 从已扫描共现集中做分层抽样，输出样本清单。
然后人工逐调用 MiniMax MCP understand_image 做视觉判断。
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

COOCCUR_PATH = ROOT / "benchmark" / "eval" / "results" / "cross_modal_cooccurrence.json"
SAMPLE_PATH = ROOT / "benchmark" / "eval" / "results" / "d_audit_manifest.json"
CHUNKS_DIR = ROOT / "corpus" / "chunks"
SAMPLE_SIZE = 30

FM_BLOCK = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
FM_INLINE = re.compile(r'(?<!\$)\$(?!\$)([^$]+?)(?<!\$)\$(?!\$)')

def strip_code_fences(text):
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)

def extract_formulas(text):
    scan = strip_code_fences(text)
    formulas = []
    for m in FM_BLOCK.findall(scan):
        formulas.append(("block", m.strip()))
    for m in FM_INLINE.findall(scan):
        formulas.append(("inline", m.strip()))
    return formulas

def get_chunk_text(chunk_filepath):
    path = CHUNKS_DIR / chunk_filepath.replace("corpus/chunks/", "")
    return path.read_text(encoding="utf-8")

with open(COOCCUR_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

chunks = data["chunks"]
print(f"共现总chunks: {len(chunks)}")

subdomain_groups = defaultdict(list)
for c in chunks:
    for sd in c["subdomains"]:
        subdomain_groups[sd].append(c)

sorted_sds = sorted(subdomain_groups.items(), key=lambda x: -len(x[1]))
random.seed(42)

samples = []
remaining = SAMPLE_SIZE
target_per_sd = max(1, SAMPLE_SIZE // len(sorted_sds))

for sd, group in sorted_sds:
    if remaining <= 0:
        break
    take = min(target_per_sd, len(group), remaining)
    selected = random.sample(group, take)
    samples.extend(selected)
    remaining -= take

if remaining > 0:
    extra_pool = [c for c in chunks if c not in samples]
    random.shuffle(extra_pool)
    samples.extend(extra_pool[:remaining])

random.shuffle(samples)

manifest = []
for s in samples:
    chunk_text = get_chunk_text(s["filepath"])
    formulas = extract_formulas(chunk_text)
    formulas_text = "\n".join([f"{typ}: ${fm}$" for typ, fm in formulas[:8]])
    text_preview = chunk_text[:500].strip()

    manifest.append({
        "index": len(manifest),
        "chunk_id": s["chunk_id"],
        "filename": s["filename"],
        "subdomains": s["subdomains"],
        "image_path": s["image_paths"][0],
        "image_hash": s["images"][0],
        "total_images": s["image_count"],
        "formula_count": s["formula_count"],
        "formulas_preview": formulas_text[:300],
        "text_preview": text_preview,
    })

output = {
    "generated_at": "2026-05-08",
    "total_cooccurrence": len(chunks),
    "sample_size": len(manifest),
    "samples": manifest,
}

SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

sd_dist = Counter()
for s in samples:
    for sd in s["subdomains"]:
        sd_dist[sd] += 1
print(f"抽样: {len(manifest)} 个")
print(f"子领域分布: {dict(sd_dist.most_common())}")
print(f"清单已保存: {SAMPLE_PATH}")
