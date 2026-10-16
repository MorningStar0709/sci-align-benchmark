"""找出采样中最长的论文。"""

if __name__ == "__main__":
    from pypdf import PdfReader
    from pathlib import Path

    d = Path(__file__).resolve().parents[4] / "data" / "pdf" / "arXiv"
    pdfs = sorted(d.glob("*.pdf"))
    sample = pdfs[:50]

    pages = []
    for p in sample:
        try:
            r = PdfReader(str(p))
            pages.append((len(r.pages), p.name))
        except Exception as exc:
            print(f"  [skip] {p.name}: {exc}")

    pages.sort(key=lambda x: -x[0])
    print("Top 3 longest papers in sample:")
    for n, name in pages[:3]:
        print(f"  {n:3d} pages  {name}")
