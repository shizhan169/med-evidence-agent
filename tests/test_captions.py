"""图注入库回归测试：图注/表注识别、打包、入库。

图注是“多模态”的轻量版：不做 OCR、不读图，只把 PDF 文本层里
的图注/表注文字提出来当片段入库，问“图 3 讲的是什么”能答出来。
"""
import hashlib

import pytest

from app.ingest import _extract_captions, _pack_caption, chunk_pages
from app import rag


def test_extract_captions_keeps_real_and_filters_refs():
    raw = (
        "Introduction text about the study and what it aims to show in this paper.\n"
        "Figure 1. Age distribution of the participants in the study sample.\n"
        "Figure 2 shows the relative rate of change across the lifespan.\n"
        "Figure 3.\n"
        "Table 1. Participant characteristics by age group and sex.\n"
        "Figure 4.\n"
        "50\n"
    )
    caps = _extract_captions(raw)
    # 真图注 2 条：Figure 1、Table 1；Figure 2 是正文引用句，
    # Figure 3 后面是另一条图注、Figure 4 后面是坐标轴数字，都不要
    assert [(k, n) for k, n, _, _ in caps] == [("Figure", "1"), ("Table", "1")]
    assert caps[0][2].startswith("Figure 1. Age distribution")
    assert caps[1][2].startswith("Table 1. Participant characteristics")


def test_extract_captions_multiline_and_axis_noise():
    raw = (
        "Figure 1.\n"
        "Longitudinal magnetic resonance brain imaging studies measuring whole brain volume change with age in humans.\n"
        "Figure 3. Ridgeline Plots of the Distribution of Brain Structure Volume Change Rates for Each Decade\n"
        "40100Age group, y60\n"
        "50\n"
    )
    caps = _extract_captions(raw)
    assert [(k, n) for k, n, _, _ in caps] == [("Figure", "1"), ("Figure", "3")]
    # 标签单独成行时，下一行描述性文字是图注的一部分
    assert "Longitudinal magnetic resonance" in caps[0][2]
    # 图注本身只有一行，下一行的坐标轴残行不要吸进来
    assert caps[1][2] == ("Figure 3. Ridgeline Plots of the Distribution "
                           "of Brain Structure Volume Change Rates for Each Decade")
    assert "40100Age" not in caps[1][2]


def test_chunk_pages_includes_caption_chunks():
    page_text = (
        "Introduction sentence about the study design and methods used in this work here. "
        "Another background sentence about brain development and aging across the lifespan.\n"
        "Figure 1. Age distribution of the participants in the study sample.\n"
        "The participants were recruited from multiple sites across the country and scanned twice. "
        "A third sentence with more detail about the imaging protocol and the analysis plan. "
        "A fourth sentence to add more content about the results and discussion section."
    )
    chunks = chunk_pages([{"page": 1, "text": page_text, "source": "test.pdf"}])
    caps = [c for c in chunks if c.get("type") == "caption"]
    texts = [c for c in chunks if c.get("type") != "caption"]
    assert len(caps) == 1
    assert caps[0]["text"].startswith("Figure 1. Age distribution")
    assert caps[0]["page"] == 1 and caps[0]["source"] == "test.pdf"
    assert caps[0]["context"].startswith("【文献：")
    assert len(texts) >= 1  # 正文片段还在，图注是多出来的


def test_long_caption_is_packed_with_label_first():
    long_desc = ("Word " * 200)[:999] + "."
    raw = f"Figure 9. {long_desc}\n"
    caps = _extract_captions(raw)
    assert len(caps) == 1
    chunks = _pack_caption(caps[0][2], {"page": 1, "text": raw, "source": "test.pdf"},
                           "【文献：Test｜第 1 页｜摘要】", caps[0][3])
    assert len(chunks) >= 2
    assert all(c["type"] == "caption" for c in chunks)
    assert all(len(c["text"]) <= 820 for c in chunks)  # 标签占 11 字，第一片可略超 800
    assert chunks[0]["text"].startswith("Figure 9. ")


class _FakeEmbeddingFunction:
    """不用下载模型的假 embedding：按文本 md5 生成 8 维向量。
    只测“入库能不能写进去、查库能不能查回来”，不测语义好坏。"""

    def __call__(self, input):
        out = []
        for text in input:
            h = hashlib.md5(text.encode("utf-8")).digest()
            out.append([b / 255.0 for b in h[:8]])
        return out

    def name(self):
        return "fake-test-embedding"


@pytest.fixture
def temp_chroma(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    monkeypatch.setattr(rag, "CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(rag, "get_embedding_function", lambda: _FakeEmbeddingFunction())
    monkeypatch.setattr(rag, "_BM25_CACHE", {"count": -1, "index": None, "chunks": None})
    return tmp_path


def _fujita_p8_text():
    """真实 Fujita p.8 原文（pypdf）：图 3 标签在第 2 行、正文在第 49–50 行；
    图 4 标签在第 51 行、正文在第 85–86 行，中间全是坐标轴文字。"""
    from pathlib import Path

    from app.ingest import parse_pdf

    pdf = Path(__file__).resolve().parents[1] / "data" / "papers" / "2023-fujita-brain-volume-aging.pdf"
    if not pdf.exists():
        pytest.skip("data/papers 里没有 Fujita 2023，跳过真实图注检查")
    pages = parse_pdf(pdf)
    return pages[7]["text"]


def _caption_by_number(caps, kind, number):
    for k, n, text, _ in caps:
        if k == kind and n == number:
            return text
    raise AssertionError(f"没找到 {kind} {number} 的图注")


def test_extract_captions_fujita_p8_recovers_caption_bodies():
    # pypdf 提的原文里词和词之间常没空格（"Solidblacklines..."），
    # 所以断言用去空格后的文本，只验图注正文的内容在不在。
    caps = _extract_captions(_fujita_p8_text())
    cap3 = _caption_by_number(caps, "Figure", "3")
    cap4 = _caption_by_number(caps, "Figure", "4")
    c3, c4 = cap3.replace(" ", ""), cap4.replace(" ", "")
    assert "Ridgeline" in c3
    assert "medianvalues" in c3
    assert "generalizedadditivemodel" in c4
    assert "95%CIs" in c4
    # 图 4 的图注不是只有标题（标题约 80 字）
    assert len(cap4) > 200


def test_real_papers_caption_counts_unchanged():
    # 改动只变图注正文的长度，不变标签识别：26 篇仍是 147 条图注 + 28 条表注。
    # 跑得慢（约 1 分钟，pypdf 要逐页解析），只在改图注提取后跑。
    from pathlib import Path

    from app.ingest import parse_pdf

    papers_dir = Path(__file__).resolve().parents[1] / "data" / "papers"
    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        pytest.skip("data/papers 里没有 PDF，跳过真实文献检查")
    fig = tab = 0
    for pdf in pdfs:
        for p in parse_pdf(pdf):
            for k, _, _, _ in _extract_captions(p["text"]):
                if k == "Figure":
                    fig += 1
                else:
                    tab += 1
    assert (fig, tab) == (147, 28)


def test_pack_caption_appends_figure_token_to_context():
    ctx = "【文献：Test｜第 1 页｜摘要】"
    page = {"page": 1, "text": "", "source": "test.pdf"}
    fig = _pack_caption("Figure 4. Some caption text here for testing.", page, ctx, 0)
    assert fig[0]["context"] == ctx + " figure4"
    assert fig[0]["context"].startswith("【文献：")
    tab = _pack_caption("Table 2. Some table caption text here.", page, ctx, 0)
    assert tab[0]["context"] == ctx + " table2"
    # 没有标签的文本不加 token
    plain = _pack_caption("Some caption text without a label.", page, ctx, 0)
    assert plain[0]["context"] == ctx


def test_body_chunks_get_no_figure_token():
    # 正文里提到 "Figure 4 shows..." 是正文引用句（编号后跟动词），
    # 不提图注；正文片段的 context 不能带 figure4，不然分不清图 3 正文和图 4 图注。
    text = (
        "The study followed a large sample of adults across the lifespan to "
        "characterize age-dependent changes in brain structure volumes and "
        "volume change rates over time in this longitudinal analysis. "
        "Figure 4 shows the volume change rate for each tissue type across age. "
        "A third sentence adds more detail about the imaging protocol used "
        "in the study and the analysis plan for the follow-up visits."
    )
    chunks = chunk_pages([{"page": 8, "text": text, "source": "2023-fujita-brain-volume-aging.pdf"}])
    assert chunks
    assert not any(c.get("type") == "caption" for c in chunks)
    assert all("figure4" not in c["context"] for c in chunks)
    assert all("figure" not in c["context"] for c in chunks)


def test_strip_context_removes_figure_token():
    # token 在 context 行上，回答员收到的片段里不能有它
    ctx = "【文献：Test｜第 1 页｜摘要】 figure4"
    assert rag._strip_context(ctx + "\n" + "Figure 4. body.") == "Figure 4. body."


def test_caption_chunks_roundtrip(temp_chroma):
    page_text = (
        "Introduction sentence about the study design and methods used in this work here. "
        "Another background sentence about brain development and aging across the lifespan.\n"
        "Figure 1. Age distribution of the participants in the study sample.\n"
        "The participants were recruited from multiple sites across the country and scanned twice. "
        "A third sentence with more detail about the imaging protocol and the analysis plan."
    )
    chunks = chunk_pages([{"page": 1, "text": page_text, "source": "test.pdf"}])
    caps = [c for c in chunks if c.get("type") == "caption"]
    assert caps
    n = rag.add_chunks(chunks)
    assert n == len(chunks)
    assert rag.get_collection().count() == len(chunks)
    stored = rag.get_collection().get(include=["documents", "metadatas"])
    assert "caption" in [m.get("type") for m in stored["metadatas"]]
    assert any("Figure 1. Age distribution" in d for d in stored["documents"])
    # BM25 能命中图注里的词
    found = rag.retrieve("Figure Age distribution participants", top_k=5)
    assert any("Age distribution" in c["text"] for c in found)
