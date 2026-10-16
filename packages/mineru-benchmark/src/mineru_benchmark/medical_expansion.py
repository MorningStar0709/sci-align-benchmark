"""D13: Sciverse Medical Literature Expansion for Track 3.

Searches Sciverse for control-medicine cross-domain papers via meta_search,
and Chinese medical papers via language filter. Expands literature base
from 97 PMC papers to 300-500+ papers.

Usage:
    python -m benchmark.sciverse_medical_expansion
"""

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # restored Phase 5

from mineru_core.mineru_client import SciverseClient

OUTPUT_DIR = ROOT / "benchmark" / "eval" / "results" / "medical"
OUTPUT_PATH = OUTPUT_DIR / "sciverse_medical_expansion.json"

CONTROL_QUERIES = [
    "model predictive control clinical trial",
    "adaptive control biomedical application",
    "robust control medical device",
    "optimal control drug delivery",
    "nonlinear control physiological model",
    "PID control clinical",
    "Kalman filter biomedical signal",
    "reinforcement learning control healthcare",
    "fuzzy control medical diagnosis",
    "neural network control surgical robotics",
    "feedback control anesthesia",
    "sliding mode control prosthetics",
    "Lyapunov stability biological system",
    "control barrier function medical safety",
    "linear system identification clinical",
    "distributed control telemedicine",
    "stochastic control epidemiology",
    "event-triggered control wearable device",
    "hierarchical control hospital management",
    "game theory control healthcare resource",
    "optimal estimation medical imaging",
    "system identification pharmacokinetics",
    "adaptive filtering ECG processing",
    "control theory insulin pump closed loop",
    "predictive control ICU ventilation",
]

MEDICAL_BROAD = [
    "clinical trial control group randomized",
    "evidence-based medicine systematic review",
    "diagnostic accuracy sensitivity specificity",
    "therapy effectiveness meta-analysis",
    "prognostic model survival analysis",
    "surgical intervention outcomes",
    "intensive care monitoring",
    "chronic disease management",
    "personalized medicine treatment response",
    "biomarker discovery validation",
    "epidemiology population health",
    "cardiovascular disease prevention",
    "oncology treatment regimen",
    "neurology brain computer interface",
    "radiology image guided intervention",
]

CHINESE_QUERIES = [
    "模型预测控制 临床",
    "自适应控制 医学",
    "鲁棒控制 医疗",
    "最优控制 给药",
    "非线性控制 生理",
    "卡尔曼滤波 生物医学",
    "强化学习 控制 健康",
    "神经网络 控制 手术",
    "糖尿病 闭环控制 胰岛素",
    "临床对照试验 系统评价",
    "诊断准确率 敏感度 特异度",
    "慢性病 管理 干预效果",
    "中医 针灸 随机对照",
    "高血压 药物 疗效 对照",
    "肿瘤 化疗 疗效 评估",
    "脑机接口 神经康复",
    "医学影像 人工智能 诊断",
    "重症监护 预后 模型",
]

MEDICAL_KW = [
    "clinical", "medical", "patient", "therapy", "diagnos", "surgical",
    "disease", "drug", "treatment", "health", "hospital", "physician",
    "biomedical", "pharmac", "cancer", "cardiac", "neural", "brain",
    "surgery", "anesthesia", "ventilation", "insulin", "glucose",
    "prosthetic", "rehabilitation", "imaging", "ECG", "EEG", "MRI",
    "biomarker", "epidemiology", "oncology", "neurology", "radiology",
    "dose", "infusion", "implant", "wearable", "telemedicine",
    "intensive care", "ICU", "chronic", "acute", "infection",
    "tumor", "lesion", "patholog", "gene", "cell", "protein",
    "symptom", "mortality", "survival", "prognosis", "screening",
    "prevention", "intervention", "randomized", "controlled trial",
    "cohort", "systematic review", "meta-analysis", "evidence-based",
]

CHINESE_KW = [
    "临床", "治疗", "诊断", "手术", "患者", "药物", "疾病", "医学",
    "健康", "医院", "康复", "肿瘤", "心脏", "神经", "脑",
    "中医", "针灸", "中药", "血糖", "血压", "麻醉", "重症",
    "感染", "炎症", "免疫", "基因", "细胞", "蛋白",
    "随机对照", "系统评价", "队列", "预后", "筛查",
    "流行病", "慢性病", "急性", "症状", "死亡率",
]


def _is_medical(text, keywords):
    t = text.lower() if text else ""
    return any(kw.lower() in t for kw in keywords)


SCIVERSE_API_KEY = os.environ.get("SCIVERSE_API_KEY", "")


def meta_search_batch(client, queries, page_size=25, max_per_page=3):
    papers = {}
    for qi, query in enumerate(queries):
        print(f"  [{qi+1}/{len(queries)}] {query[:60]}...", end=" ", flush=True)
        total_from_query = 0
        try:
            for page in range(1, max_per_page + 1):
                resp = client.meta_search(
                    query=query, page_size=page_size, page=page)
                for item in resp.get("results", []):
                    doc_id = item.get("doc_id", "")
                    if not doc_id or doc_id in papers:
                        continue
                    title = item.get("title", "") or ""
                    abstract = item.get("abstract", "") or ""
                    combined = f"{title} {abstract}"
                    if not _is_medical(combined, MEDICAL_KW):
                        continue
                    papers[doc_id] = {
                        "title": title,
                        "author": str(item.get("author", ""))[:200],
                        "year": item.get("publication_published_year", ""),
                        "doi": item.get("doi", ""),
                        "doc_id": doc_id,
                        "abstract": abstract[:500],
                        "_query": query,
                    }
                    total_from_query += 1
                if len(resp.get("results", [])) < page_size:
                    break
                time.sleep(0.5)
            print(f"+{total_from_query} ({len(papers)} total)")
        except Exception as e:
            print(f"ERROR: {str(e)[:80]}")
        time.sleep(1.0)
    return papers


def meta_search_chinese(client, queries, page_size=25, max_per_page=3):
    papers = {}
    for qi, query in enumerate(queries):
        print(f"  [ZH {qi+1}/{len(queries)}] {query[:50]}...", end=" ", flush=True)
        total_from_query = 0
        try:
            for page in range(1, max_per_page + 1):
                resp = client.meta_search(
                    query=query, page_size=page_size, page=page)
                for item in resp.get("results", []):
                    doc_id = item.get("doc_id", "")
                    if not doc_id or doc_id in papers:
                        continue
                    title = item.get("title", "") or ""
                    abstract = item.get("abstract", "") or ""
                    combined = f"{title} {abstract}"
                    if not _is_medical(combined, CHINESE_KW):
                        continue
                    papers[doc_id] = {
                        "title": title,
                        "author": str(item.get("author", ""))[:200],
                        "year": item.get("publication_published_year", ""),
                        "doi": item.get("doi", ""),
                        "doc_id": doc_id,
                        "abstract": abstract[:500],
                        "_query": query,
                        "_language": "zh",
                    }
                    total_from_query += 1
                if len(resp.get("results", [])) < page_size:
                    break
                time.sleep(0.5)
            print(f"+{total_from_query} ({len(papers)} total)")
        except Exception as e:
            print(f"ERROR: {str(e)[:80]}")
        time.sleep(1.0)
    return papers


def main():
    print("=" * 60)
    print("D13: Sciverse Medical Literature Expansion")
    print(f"  Control queries:  {len(CONTROL_QUERIES)}")
    print(f"  Medical queries:   {len(MEDICAL_BROAD)}")
    print(f"  Chinese queries:   {len(CHINESE_QUERIES)}")
    print(f"  Total:             {len(CONTROL_QUERIES) + len(MEDICAL_BROAD) + len(CHINESE_QUERIES)}")
    print("=" * 60)

    client = SciverseClient(api_key=SCIVERSE_API_KEY)

    # Phase 1: English control-medical cross-domain
    print("\n[Phase 1] Control × Medical English papers...")
    all_queries = CONTROL_QUERIES + MEDICAL_BROAD
    en_papers = meta_search_batch(client, all_queries, max_per_page=2)

    # Phase 2: Chinese medical papers
    print("\n[Phase 2] Chinese medical papers...")
    zh_papers = meta_search_chinese(client, CHINESE_QUERIES, max_per_page=2)

    # Deduplicate across phases
    en_ids = set(en_papers.keys())
    zh_unique = {k: v for k, v in zh_papers.items() if k not in en_ids}

    # Build report
    report = {
        "analysis": "sciverse_medical_expansion",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "control_queries": len(CONTROL_QUERIES),
            "medical_queries": len(MEDICAL_BROAD),
            "chinese_queries": len(CHINESE_QUERIES),
        },
        "summary": {
            "english_medical_papers": len(en_papers),
            "chinese_medical_papers": len(zh_unique),
            "total_unique_papers": len(en_papers) + len(zh_unique),
            "baseline_pmc_papers": 97,
            "expansion_ratio": round((len(en_papers) + len(zh_unique)) / 97, 1),
            "full_automation_achieved": len(en_papers) + len(zh_unique) >= 200,
        },
        "english_details": {
            "count": len(en_papers),
            "sample_titles": [p["title"] for p in list(en_papers.values())[:10]],
            "year_range": f"{min((str(p.get('year','')) or '?') for p in en_papers.values())} - {max((str(p.get('year','')) or '?') for p in en_papers.values())}",
        },
        "chinese_details": {
            "count": len(zh_unique),
            "sample_titles": [p["title"] for p in list(zh_unique.values())[:10]],
        },
        "english_papers": [{"doc_id": k, **v} for k, v in en_papers.items()],
        "chinese_papers": [{"doc_id": k, **v} for k, v in zh_unique.items()],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("D13 COMPLETE")
    print(f"  English: {len(en_papers)} papers")
    print(f"  Chinese: {len(zh_unique)} papers (unique)")
    print(f"  Total:   {len(en_papers) + len(zh_unique)} new papers")
    print(f"  Baseline: 97 PMC → {97 + len(en_papers) + len(zh_unique)} total")
    print(f"  Ratio:   {(len(en_papers) + len(zh_unique)) / 97:.1f}x expansion")
    print(f"  Report:  {OUTPUT_PATH}")
    print(f"{'=' * 60}")

    if len(en_papers) >= 200:
        print("  ✅ Target met: ≥200 English papers")
    else:
        print(f"  ⚠️ Target not met: {len(en_papers)}/200 English papers (consider increasing max_per_page)")
    if len(zh_unique) >= 50:
        print("  ✅ Target met: ≥50 Chinese papers")
    else:
        print(f"  ⚠️ Target not met: {len(zh_unique)}/50 Chinese papers")


if __name__ == "__main__":
    main()
