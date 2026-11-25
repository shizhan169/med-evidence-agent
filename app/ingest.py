"""PDF 解析 -> 切分 -> Chroma 入库。用法：python -m app.ingest --papers data/papers"""
import argparse
import os
import re
from pathlib import Path
from pypdf import PdfReader

# 切分参数：按句子切，一个片段最多 800 字，相邻片段重叠约 200 字（按整句重叠）。
# 为什么按句子切：之前是固定 800 字网格切，经常把一句话从中间切断（比如
# 切出 "pre- dictions" 这种半个词），还把“结论句”和“数字句”切到两个片段里——
# 检索时模型只拿到结论片段、看不到数字，就只能拒答或只答一半（2026-11-25
# 第三轮评测的 Q04/Q06 就是这么丢的）。按句子切能保证每句话完整，
# 结论和它要的数字待在同一个片段里，检索更容易一次命中。
CHUNK_SIZE = 800
CHUNK_OVERLAP = 300  # 约 1~2 句。实测选的：重叠 300 时 BM25 前 5 命中 6/11（和旧切分持平），
# 目标题同框片段 Q04/Q06/Q09 各 2 个；重叠 200 时 BM25 掉到 4/11，Q04 同框只剩 1 个。

# 分句时不能切的缩写：句号后面跟大写字母本来要分句，但这些缩写后面的
# 大写不是新句子（比如 "e.g. The"、"Fig. 2A"），分错了会把句子切碎。
_ABBREV = ("Fig.", "e.g.", "i.e.", "vs.", "et al.", "Dr.", "No.", "Vol.", "pp.", "Eq.", "Ref.", "St.")

# 分句点：句号/问号/感叹号 + 空格 + 大写字母（新句子一般大写开头）。
_SENT_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
# 行尾连字符断词：只有小写字母-换行-小写字母才是被行尾切断的单词
# （比如 "pre-\ndictions"），接回去；其他连字符不动。
_HYPHEN_BREAK_RE = re.compile(r"([a-z])-\n\s*([a-z])")


def parse_pdf(path: Path):
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": i, "text": text, "source": path.name})
    return pages


def _clean_page_text(text: str) -> str:
    """把 PDF 按行断开的文字接回连续文本。

    PDF 提文字时每行末尾都有换行，行尾连字符断词要先接回去，
    再把其余换行换成空格（它们只是行尾，不断句），多余空白压成一个。
    """
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list:
    """按句号分句，返回 [(句子, 在页内起始位置)]。

    分完再把以缩写结尾的碎片接回去（比如 "e.g." 后面跟大写不是新句子）。
    起始位置用来生成稳定的 chunk_start（入库去重的 ID 一部分）。
    """
    bounds = [0]
    for m in _SENT_END_RE.finditer(text):
        bounds.append(m.end())
    bounds.append(len(text))
    out = []
    for i in range(len(bounds) - 1):
        sent = text[bounds[i]:bounds[i + 1]].strip()
        if not sent:
            continue
        if out and out[-1][0].endswith(_ABBREV):
            # 上一片以缩写结尾，说明切错了，和这一片合并，位置取第一片的
            prev_sent, prev_start = out[-1]
            out[-1] = (prev_sent + " " + sent, prev_start)
        else:
            out.append((sent, bounds[i]))
    return out


def _split_long_sentence(sent: str, start: int) -> list:
    """超长句子（> CHUNK_SIZE）才按词切：不断词，在空格处断开。
    论文里句子一般 150–300 字，这条很少触发，只是兜底。"""
    words = sent.split(" ")
    pieces, cur = [], ""
    pos = start
    for w in words:
        if cur and len(cur) + 1 + len(w) > CHUNK_SIZE:
            pieces.append((cur, pos))
            pos += len(cur) + 1
            cur = w
        else:
            cur = cur + " " + w if cur else w
    if cur:
        pieces.append((cur, pos))
    return pieces


def chunk_pages(pages):
    """按句子打包切分：句子是最小单位，不拆句子；相邻片段重叠 1~2 句。

    打包规则：顺着句子往片段里装，装到再加一句就超 800 字时收尾；
    新片段开头带上上个片段末尾的句子（攒够约 CHUNK_OVERLAP 字为止），
    这样跨片段的结论和数字不断线。返回的每个片段仍是
    {page, text, source, chunk_start}，和原来一样的形状，下游不用改。
    """
    chunks = []
    for p in pages:
        text = _clean_page_text(p["text"])
        # 句子列表：(句子, 起始位置)；超长句子先按词切成几段
        units = []
        for sent, start in _split_sentences(text):
            if len(sent) > CHUNK_SIZE:
                units.extend(_split_long_sentence(sent, start))
            else:
                units.append((sent, start))
        cur, cur_len, cur_start = [], 0, None  # cur 存 (句子, 起始位置)
        for sent, start in units:
            if cur and cur_len + 1 + len(sent) > CHUNK_SIZE:
                chunks.append({**p, "text": " ".join(s for s, _ in cur), "chunk_start": cur_start})
                # 重叠：新片段带上上个片段末尾的句子，攒够约 CHUNK_OVERLAP 字
                overlap, overlap_len = [], 0
                for s, s_start in reversed(cur):
                    if overlap and overlap_len + len(s) > CHUNK_OVERLAP:
                        break
                    overlap.insert(0, (s, s_start))
                    overlap_len += len(s) + 1
                cur, cur_len = overlap, overlap_len
                cur_start = overlap[0][1] if overlap else None
            if cur_start is None:
                cur_start = start
            cur.append((sent, start))
            cur_len += len(sent) + 1
        if cur:
            chunks.append({**p, "text": " ".join(s for s, _ in cur), "chunk_start": cur_start})
    # 过短片段（<50 字）跳过，和原来一样
    return [c for c in chunks if len(c["text"]) >= 50]


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
