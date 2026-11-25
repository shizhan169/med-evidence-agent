"""重排回归测试：cross-encoder 打分排序 + 不可用时退回旧行为。

重排是检索的第三段（向量 + BM25 混合检索 -> cross-encoder 重排 -> 回答）：
bi-encoder 只看问题和片段各自的向量，重排是让模型同时看问题和片段
全文再打分，所以更准；代价是慢，只能对几十个候选片段用。
真模型测试太重（要下载 bge-reranker-base），不在 pytest 里做；
真模型的效果用 /tmp 的探针脚本验证 Q01（见工程记录第 14 条）。
"""
import hashlib

import pytest

from app import rag


class _FakeReranker:
    """假重排模型：含 TARGET 的片段打 1 分，其余 0 分。"""

    def __init__(self):
        self.calls = []

    def predict(self, pairs):
        self.calls.append(len(pairs))
        return [1.0 if "TARGET" in doc else 0.0 for _, doc in pairs]


def _chunks(n):
    return [{"text": f"chunk {i} about brain volume aging"
                     + (" TARGET" if i == n - 1 else ""),
             "source": "t.pdf", "page": 1}
            for i in range(n)]


def test_rerank_promotes_target_to_top(monkeypatch):
    monkeypatch.setattr(rag, "_get_reranker", lambda: _FakeReranker())
    out = rag._rerank("query", _chunks(8), top_k=5)
    assert len(out) == 5
    assert out[0]["text"].endswith("TARGET")
    assert all("TARGET" not in c["text"] for c in out[1:])


def test_rerank_returns_all_when_fewer_than_top_k(monkeypatch):
    monkeypatch.setattr(rag, "_get_reranker", lambda: _FakeReranker())
    assert len(rag._rerank("query", _chunks(3), top_k=5)) == 3


def test_rerank_empty_and_no_model(monkeypatch):
    assert rag._rerank("query", []) == []
    # 模型不可用时原样返回（不排序、不丢片段）
    monkeypatch.setattr(rag, "_reranker", False)
    chunks = _chunks(4)
    assert rag._rerank("query", chunks, top_k=5) == chunks


class _FakeEmbeddingFunction:
    """不用下载模型的假 embedding：按文本 md5 生成 8 维向量。"""

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


def _seed_chunks():
    # 12 个片段：4 个查询词（target/brain/volume/aging）各在 3 个片段里，
    # 互不重叠，只有最后一个含 TARGET。
    # 这样写是有原因的：rank-bm25 里一个词出现在全部（或大多数）片段里时，
    # IDF 会变成负数、分数 <=0，_keyword_chunks 遇到 <=0 就停——
    # 最初版本 12 个片段写得太像（都含 brain volume aging），BM25 只返回
    # 1 个，fallback 测试拿不到 10 个。词分散开、每个词只在少数片段里，
    # IDF 才是正的。
    topics = [
        ("brain networks connectivity", 0), ("brain growth charts", 1), ("brain morphology", 2),
        ("volume change rates", 3), ("volume loss", 4), ("volume trajectories", 5),
        ("aging population", 6), ("aging and cognition", 7), ("aging biomarkers", 8),
        ("cortical thickness development", 9), ("hippocampal atrophy", 10),
        ("Brain volume aging study TARGET marker here", 11),
    ]
    return [{"page": 1, "chunk_start": i * 100, "source": "t.pdf",
             "text": f"{topic} study number {i} about cortical thickness.",
             "context": "【文献：T｜第 1 页｜摘要】"}
            for topic, i in topics]


def test_retrieve_uses_reranker_when_available(temp_chroma, monkeypatch):
    rag.add_chunks(_seed_chunks())
    monkeypatch.setattr(rag, "_get_reranker", lambda: _FakeReranker())
    found = rag.retrieve("TARGET brain volume aging", top_k=10)
    assert len(found) == 5  # 重排后只给回答员 5 个
    assert "TARGET" in found[0]["text"]


def test_retrieve_falls_back_without_reranker(temp_chroma, monkeypatch):
    rag.add_chunks(_seed_chunks())
    monkeypatch.setattr(rag, "_get_reranker", lambda: None)
    found = rag.retrieve("TARGET brain volume aging", top_k=10)
    assert len(found) == 10  # 旧行为：向量 5 + BM25 5，不重排
    assert any("TARGET" in c["text"] for c in found)
