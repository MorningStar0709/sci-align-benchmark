import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]  # restored Phase 5
_LATEX_PATTERN = re.compile(r'\\(?:[a-zA-Z]+|\(|\)|\[|\])')


def tokenize(text):
    tokens = []
    for match in re.finditer(r'\\(?:[a-zA-Z]+|\(|\)|\[|\])|[\w\u4e00-\u9fff]+|[^\w\s]', text):
        tokens.append(match.group())
    return tokens


def extract_ngrams(tokens, n):
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def check_overlap(questions, corpus_chunks, threshold=0.3, n=13):
    results = []
    corpus_ngrams = set()
    for chunk in corpus_chunks:
        tokens = tokenize(chunk)
        chunk_ngrams = extract_ngrams(tokens, n)
        corpus_ngrams |= chunk_ngrams
        if len(corpus_ngrams) > 1_000_000:
            break
    if not corpus_ngrams:
        return results

    for q in questions:
        qid = q.get("id", "?")
        q_text = q.get("question", "") + " " + q.get("answer", "")
        q_tokens = tokenize(q_text)
        q_ngrams = extract_ngrams(q_tokens, n)
        if not q_ngrams:
            low_tokens = tokenize(q_text)
            low_ngrams = extract_ngrams(low_tokens, 8) if len(low_tokens) >= 8 else set()
            if not low_ngrams:
                results.append({"id": qid, "overlap_ratio": 0.0, "status": "no_ngrams", "question_sample": q_text[:100]})
                continue
            q_ngrams = low_ngrams
        overlap = len(q_ngrams & corpus_ngrams) / len(q_ngrams)
        results.append({
            "id": qid,
            "overlap_ratio": round(overlap, 4),
            "status": "contaminated" if overlap > threshold else "clean",
            "question_sample": q_text[:100],
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="检查评测题与语料库的 n-gram 数据污染")
    parser.add_argument("--questions", required=True, help="评测集 JSON 路径")
    parser.add_argument("--corpus", default=str(ROOT / "corpus"), help="语料库根目录")
    parser.add_argument("--n", type=int, default=13, help="n-gram 长度（检测短文本时自动回退到 8）")
    parser.add_argument("--threshold", type=float, default=0.3, help="污染阈值")
    parser.add_argument("--output", default="", help="输出路径（默认 stdout）")
    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        sys.exit(f"评测集文件不存在: {questions_path}")

    with questions_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", [])
    print(f"加载评测集: {len(questions)} 题")

    corpus_root = Path(args.corpus)
    manifest_path = corpus_root / "chunks" / "chunks_manifest.json"
    if not manifest_path.exists():
        sys.exit(f"chunks manifest 不存在: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    chunk_texts = []
    loaded = 0
    missing = 0
    for chunk in manifest.get("chunks", []):
        filepath = Path(chunk["filepath"])
        path = filepath if filepath.is_absolute() else ROOT / filepath
        if path.exists():
            chunk_texts.append(path.read_text(encoding="utf-8", errors="ignore"))
            loaded += 1
        else:
            alt = corpus_root / filepath
            if alt.exists():
                chunk_texts.append(alt.read_text(encoding="utf-8", errors="ignore"))
                loaded += 1
            else:
                missing += 1
    print(f"加载语料库: {loaded} chunks 已读, {missing} 缺失, 共 {len(chunk_texts)} 有效")

    results = check_overlap(questions, chunk_texts, args.threshold, args.n)

    contaminated = [r for r in results if r["status"] == "contaminated"]
    clean = [r for r in results if r["status"] == "clean"]
    print(f"\n结果: 共 {len(results)} 题 | 污染: {len(contaminated)} | 安全: {len(clean)}")

    if contaminated:
        print("\n污染题目:")
        for r in contaminated:
            print(f"  {r['id']}: overlap={r['overlap_ratio']:.2%}")

    output = {"n": args.n, "threshold": args.threshold, "total": len(results), "contaminated_count": len(contaminated), "results": results}

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已保存: {out_path}")
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
