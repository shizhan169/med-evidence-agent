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


# 章节标题识别（Contextual Retrieval 便宜版用）：只认单独成行的短标题，
# 如 "Abstract"、"1. Introduction"、"Methods"；正文里顺带提到 results
# 的长句子不会误判。标签是中文，因为问题是中文，中文标签能帮向量检索
# 在中文问题和英文正文之间搭桥；BM25 的分词器只认英文数字，中文标签
# 对它不可见，也不捣乱。
_SECTION_HEADERS = [
    (r"abstract", "摘要"),
    (r"summary", "摘要"),
    (r"introduction", "引言"),
    (r"background", "背景"),
    (r"methods?", "方法"),
    (r"methodology", "方法"),
    (r"materials?\s+and\s+methods?", "方法"),
    (r"results?", "结果"),
    (r"discussion", "讨论"),
    (r"conclusions?", "结论"),
    (r"references?", "参考文献"),
    (r"acknowledg(e)?ments?", "致谢"),
]
_HEADER_LINE_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?([A-Za-z][A-Za-z ]*?)\s*$")


def _detect_section_headers(raw_text: str) -> list:
    """从一页原文里找出章节标题，返回 [(行首偏移, 中文标签)]（按出现顺序）。
    只认单独成行、且整行就是标题的短行；期刊名、作者行、图注都不会命中。
    偏移用来判断标题在页首还是页中——见 chunk_pages 里的用法。"""
    found = []
    offset = 0
    for line in (raw_text or "").splitlines():
        m = _HEADER_LINE_RE.match(line)
        if m:
            head = m.group(1).strip().lower()
            for pat, label in _SECTION_HEADERS:
                if re.fullmatch(pat, head):
                    if not found or found[-1][1] != label:
                        found.append((offset, label))
                    break
        offset += len(line) + 1  # +1 是换行符，近似偏移即可
    return found


def _paper_title(source: str) -> str:
    """从文件名拼一个可读的论文标题，如 2024-ge-centilebrain.pdf -> Ge Centilebrain。
    只用来做检索上下文，不是引用，不用很精确。"""
    name = (source or "").removesuffix(".pdf")
    parts = [t for t in name.split("-") if t]
    if parts and re.fullmatch(r"\d{4}", parts[0]):
        parts = parts[1:]
    return " ".join(t[:1].upper() + t[1:] for t in parts) or (source or "未知论文")


def _chunk_context(source: str, page: int, section: str) -> str:
    """检索用上下文：贴在片段前面一起算 embedding，让检索分得清
    这是哪篇论文、哪一页、哪个部分（摘要还是方法）。只给检索用，
    回答时由 retrieve() 剥掉，模型看不到，引用也不变。"""
    return f"【文献：{_paper_title(source)}｜第 {page} 页｜{section}】"


# 图注识别（图注入库用）：图注行以 Fig./Figure/Table + 数字开头。
# 真图注后面跟描述性文字；正文里的引用句（Figure 2 shows...）编号后
# 直接跟动词，不是图注；只有编号、后面没描述的残行也不要。
# 实测（2026-11-25，26 篇）：这套规则提到 147 条图注、28 条表注；
# 只有 Kia 2022 一条都没有。方案稿里的 154/30 是另一套较松的口径
# （把正文引用句也算了进去），以这里实测的 147/28 为准。
_CAPTION_LABEL_RE = re.compile(
    r"^\s*(Fig\.|Figure|Table)\s+(\d+[A-Za-z]?)\s*[.:|]?\s*(.*)$", re.IGNORECASE
)
_CAPTION_VERBS = (
    "shows", "showed", "show", "presents", "presented", "illustrates", "illustrate",
    "demonstrates", "demonstrate", "depicts", "depict", "summarizes", "summarise",
    "compares", "compare", "plots", "plot", "displays", "display", "reports", "report",
    "describes", "describe", "lists", "list", "gives", "give", "provides", "provide",
)
_CAPTION_VERB_RE = re.compile(r"^(" + "|".join(_CAPTION_VERBS) + r")\b", re.IGNORECASE)


def _looks_like_sentence(line: str) -> bool:
    """续行像不像正经句子：够长、字母占多数、有 4 字母以上的词。
    图里的坐标轴残行（"40100Age group, y60"、"50"、"-1.5 -0.5 0.5"）
    字母少或太短，过不了这关。"""
    s = (line or "").strip()
    if len(s) < 20:
        return False
    alpha = sum(c.isalpha() for c in s)
    if alpha / len(s) < 0.7:
        return False
    return bool(re.search(r"[A-Za-z]{4,}", s))


def _extract_captions(raw_text: str) -> list:
    """从一页原文里提图注/表注，返回 [(kind, number, caption_text, 行首偏移)]，
    kind 是 "Figure" 或 "Table"。图注可能跨行：标签行后面的描述行
    最多再接 3 行、总共不超过 600 字，续行必须像句子（见上）。
    正文引用句和只有编号的残行不要。"""
    lines = (raw_text or "").splitlines()
    out = []
    offset = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        line_start = offset
        offset += len(line) + 1  # +1 是换行符，近似偏移即可
        m = _CAPTION_LABEL_RE.match(line)
        i += 1
        if not m:
            continue
        rest = (m.group(3) or "").strip()
        if rest and _CAPTION_VERB_RE.match(rest):
            continue  # 正文引用句，如 Figure 2 shows...
        parts = [rest] if rest else []
        n_cont = 0
        while i < len(lines) and n_cont < 3 and sum(len(p) for p in parts) < 600:
            nxt = lines[i].strip()
            if not nxt or _CAPTION_LABEL_RE.match(lines[i]):
                break
            if not _looks_like_sentence(nxt):
                break
            parts.append(nxt)
            n_cont += 1
            offset += len(lines[i]) + 1  # 续行也要算进偏移，不然之后图注的 chunk_start 会偏小
            i += 1
        text = " ".join(parts).strip()
        if not text:
            continue  # 只有编号的残行
        kind = "Table" if m.group(1).lower().startswith("table") else "Figure"
        out.append((kind, m.group(2), f"{kind} {m.group(2)}. {text}", line_start))
    return out


_LABEL_PREFIX_RE = re.compile(r"^(Figure|Table) \d+[A-Za-z]?\. ")


def _pack_caption(text: str, page: dict, context: str, start_offset: int) -> list:
    """图注打包成片段：一般图注只有 100~400 字，走单片段；超长（>800 字）
    才按句子打包，和正文切分同一套规则。每个片段都带 type=caption，
    chunk_start 是图注在页内原文里的偏移（ID 唯一性靠页内序号兜底）。
    长图注打包时标签（"Figure 9."）只留在第一片：分句会把 "Figure 9."
    切成独立短句，不先剥掉标签、它自己成一片会被长度过滤丢掉。"""
    base = {**page, "context": context, "type": "caption"}
    if len(text) <= CHUNK_SIZE:
        return [{**base, "text": text, "chunk_start": start_offset}]
    m = _LABEL_PREFIX_RE.match(text)
    label = m.group(0) if m else ""
    body = text[len(label):]
    units = []
    for sent, start in _split_sentences(body):
        if len(sent) > CHUNK_SIZE:
            units.extend(_split_long_sentence(sent, start))
        else:
            units.append((sent, start))
    out, cur, cur_len, cur_start = [], [], 0, None
    for sent, start in units:
        if cur and cur_len + 1 + len(sent) > CHUNK_SIZE:
            piece = " ".join(cur)
            out.append({**base,
                        "text": (label + piece) if not out else piece,
                        "chunk_start": start_offset + len(label) + (cur_start or 0)})
            cur, cur_len, cur_start = [], 0, None
        if cur_start is None:
            cur_start = start
        cur.append(sent)
        cur_len += len(sent) + 1
    if cur:
        piece = " ".join(cur)
        out.append({**base,
                    "text": (label + piece) if not out else piece,
                    "chunk_start": start_offset + len(label) + (cur_start or 0)})
    return out


def chunk_pages(pages):
    """按句子打包切分：句子是最小单位，不拆句子；相邻片段重叠 1~2 句。

    打包规则：顺着句子往片段里装，装到再加一句就超 800 字时收尾；
    新片段开头带上上个片段末尾的句子（攒够约 CHUNK_OVERLAP 字为止），
    这样跨片段的结论和数字不断线。返回的每个片段仍是
    {page, text, source, chunk_start}，和原来一样的形状，下游不用改。

    另外每个片段多带一个 "context" 字段（Contextual Retrieval 便宜版）：
    论文标题 + 页码 + 章节，只给检索用，不进回答。章节是启发式识别的：
    一页的章节取该页开头附近（前 1500 字内）第一个标题——标题行一般在
    页首几百字内，摘要比 1500 字长得多，所以 Ge p.1 那种“摘要很长、引言
    在 2500 字后”的页面不会被后半页的标题带跑；没有合条件的标题就沿用
    上一页的章节；第 1 页默认“标题与摘要”（多数论文第 1 页是标题 + 摘要，
    没有 Abstract 标题行）。单页单标签是近似的：跨章节的页面后半部分
    会被标成前半部分的章节，对检索是够用的提示，不是精确的章节标注。

    图注/表注也在这里提（见 _extract_captions）：每条图注单独成一个
    type=caption 的片段，和正文片段一起返回，一起入库。图注文本在正文
    片段里也出现过，这里是多存一份带类型标签的版本，问“图 3 讲什么”
    时检索能直接命中图注。
    """
    chunks = []
    section = "标题与摘要"  # 跨页沿用：第 1 页默认标题与摘要
    for p in pages:
        # 这一页的章节：页首附近第一个标题；没有就沿用上一页的
        headers = _detect_section_headers(p["text"])
        near_top = [label for off, label in headers if off < 1500]
        page_section = near_top[0] if near_top else section
        context = _chunk_context(p["source"], p["page"], page_section)
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
                chunks.append({**p, "text": " ".join(s for s, _ in cur), "chunk_start": cur_start, "context": context})
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
            chunks.append({**p, "text": " ".join(s for s, _ in cur), "chunk_start": cur_start, "context": context})
        # 图注/表注：和正文片段一起入库，metadata 多标 type=caption；
        # 引用格式不变，还是按页码引。图注文本也在正文片段里出现过一次，
        # 这里是单独再存一份带类型标签的版本，检索时能直接命中图注。
        for kind, num, cap_text, cap_off in _extract_captions(p["text"]):
            chunks.extend(_pack_caption(cap_text, p, context, cap_off))
        section = page_section
    # 过短片段跳过：正文 <50 字不要；图注是已验证的描述性文字，>=20 字就留
    return [c for c in chunks
            if len(c["text"]) >= (20 if c.get("type") == "caption" else 50)]


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
