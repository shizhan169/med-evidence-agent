"""PDF 解析 -> 切分 -> Chroma 入库。用法：python -m app.ingest --papers data/papers"""
import argparse
import os
from pathlib import Path
from pypdf import PdfReader

# 切分参数：先简单按字符切，后面再调
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def parse_pdf(path: Path):
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": i, "text": text, "source": path.name})
    return pages

def chunk_pages(pages):
    chunks = []
    for p in pages:
        text = p["text"]
        for start in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            piece = text[start:start + CHUNK_SIZE].strip()
            if len(piece) < 50:
                continue
            chunks.append({**p, "text": piece, "chunk_start": start})
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", default="data/papers")
    args = parser.parse_args()
    papers_dir = Path(args.papers)
    pdfs = sorted(papers_dir.glob("*.pdf"))
    print(f"找到 {len(pdfs)} 篇 PDF：{[p.name for p in pdfs]}")
    all_chunks = []
    for pdf in pdfs:
        pages = parse_pdf(pdf)
        chunks = chunk_pages(pages)
        print(f"{pdf.name}：{len(pages)} 页 -> {len(chunks)} 个片段")
        all_chunks.extend(chunks)
    print(f"总片段：{len(all_chunks)}")
    # 接 Chroma 入库；无 PDF 时跳过，ingest 不会凭空造数据
    if all_chunks:
        from app.rag import add_chunks

        n = add_chunks(all_chunks)
        print(f"已入库 {n} 个片段到 Chroma（medevidence）")
    # 先把片段存成 JSON，方便检查切分质量
    import json

    out = Path("data/chunks_preview.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_chunks[:20], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"前 20 个片段已存到 {out}，先检查切分有没有断句、乱码")

if __name__ == "__main__":
    main()
