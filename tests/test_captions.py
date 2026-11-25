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
