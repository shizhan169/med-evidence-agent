"""Contextual Retrieval 便宜版回归测试：入库前给每个片段贴一行上下文
（论文标题 + 页码 + 章节），embedding 一起算；检索时剥掉，回答员只看原文。

2026-11-25 第四轮评测的教训：Q01/Q09/Q11 三道没答出，全是检索排名问题——
要的片段存在，但检索分不清“同篇论文的不同部分”（摘要 vs 方法段落），
也没法把中文问题和英文片段对上。上下文就是给检索的区分信号。
"""
import hashlib

import pytest

from app.ingest import (
    _chunk_context,
    _detect_section_headers,
    _paper_title,
    chunk_pages,
)
from app import rag


def _sent(n_chars: int) -> str:
    """造一个指定长度的单句：句中没有句号，不会被分句切开；
    以大写字母开头，句号后接大写才算分句点。"""
    return ("Word " * ((n_chars - 1) // 5 + 1))[:n_chars - 1] + "."


def test_section_headers_detected_in_order():
    raw = "Title line\nAbstract\nSome intro text here.\nMethods\nSome method text.\nResults\nSome result text.\n"
    assert [label for _, label in _detect_section_headers(raw)] == ["摘要", "方法", "结果"]
    # 正文里顺带提到 results 的长句子不是标题
    assert _detect_section_headers("We present results of the analysis in detail.\n") == []
    # 带编号的标题也能认
    assert [label for _, label in _detect_section_headers("1. Introduction\nSome text.\n2. Methods\nMore text.\n")] == ["引言", "方法"]


def test_late_header_does_not_relabel_page():
    # 标题出现在页中后部（比如摘要很长、引言在 2000 字后），这一页仍沿用上一页的章节
    intro = _sent(700) + " " + _sent(700) + " " + _sent(700)  # 约 2100 字
    pages = [{"page": 1, "text": intro + "\nIntroduction\n" + _sent(700), "source": "test.pdf"}]
    chunks = chunk_pages(pages)
    assert chunks
    assert all("标题与摘要" in c["context"] for c in chunks)


def test_paper_title_from_filename():
    assert _paper_title("2024-ge-centilebrain.pdf") == "Ge Centilebrain"
    assert _paper_title("2022-bethlehem-brain-charts.pdf") == "Bethlehem Brain Charts"


def test_chunks_carry_context_and_clean_text():
    pages = [
        {"page": 1, "text": "Nature Article Brain charts for the human lifespan\n" + _sent(700) + " " + _sent(150), "source": "2022-bethlehem-brain-charts.pdf"},
        {"page": 2, "text": "Methods\n" + _sent(700) + " " + _sent(150), "source": "2022-bethlehem-brain-charts.pdf"},
        {"page": 3, "text": "Results\n" + _sent(700), "source": "2022-bethlehem-brain-charts.pdf"},
    ]
    chunks = chunk_pages(pages)
    assert chunks
    by_page = {}
    for c in chunks:
        by_page.setdefault(c["page"], []).append(c)
        # 上下文在单独的字段里，正文里没有
        assert c["context"].startswith("【文献：")
        assert "【文献" not in c["text"]
        assert "Bethlehem Brain Charts" in c["context"]
    assert all("标题与摘要" in c["context"] for c in by_page[1])
    assert all("方法" in c["context"] for c in by_page[2])
    assert all("结果" in c["context"] for c in by_page[3])


def test_strip_context_roundtrip():
    ctx = _chunk_context("2024-ge-centilebrain.pdf", 1, "摘要")
    body = "Some chunk text here."
    assert rag._strip_context(ctx + "\n" + body) == body
    assert rag._strip_context(body) == body  # 没有上下文行的旧片段原样返回
    assert rag._strip_context("") == ""


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


def test_retrieve_returns_clean_text(temp_chroma):
    # 入库的文档带上下文，查回来的片段没有上下文行，回答员只看到原文
    text = _sent(700) + " " + _sent(150) + " " + _sent(150)
    chunks = chunk_pages([{"page": 1, "text": text, "source": "test.pdf"}])
    assert chunks and all(c.get("context") for c in chunks)
    n = rag.add_chunks(chunks)
    assert n == len(chunks)
    stored = rag.get_collection().get(include=["documents"])
    assert all(d.startswith("【文献：") for d in stored["documents"])
    found = rag.retrieve("Word", top_k=5)
    assert found
    assert all("【文献" not in c["text"] for c in found)
