"""入库烟雾测试：切分 -> 入库 -> 查库，全流程在临时 Chroma 上跑一遍。

2026-11-25 的 DuplicateIDError 就是因为没有这层测试：入库在 upsert 那步
崩了、4220 个片段一个都没写进去，库是空的，直到用户跑评测（20 道全拒答）
才发现。以后改切分、改 ID、改入库逻辑，都要先过这个测试。
"""
import hashlib

import pytest

from app.ingest import chunk_pages
from app import rag


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


def test_ingest_roundtrip(temp_chroma):
    # 故意用会撞 ID 的句子组合：第一句 700 字独占一个片段，
    # 重叠让第二个片段也从 0 开始（见 test_ingest_ids.py）。
    text = (
        ("Word " * 140)[:699] + "."
        + " " + ("Word " * 30)[:149] + "."
        + " " + ("Word " * 30)[:149] + "."
    )
    chunks = chunk_pages([{"page": 1, "text": text, "source": "test.pdf"}])
    assert len(chunks) >= 2
    n = rag.add_chunks(chunks)
    assert n == len(chunks)
    collection = rag.get_collection()
    assert collection.count() == len(chunks)
    # 查库能查回东西：向量检索返回非空，说明入库的数据真的可读
    found = rag.retrieve("Word", top_k=5)
    assert len(found) > 0
    assert all("text" in c and "source" in c for c in found)
