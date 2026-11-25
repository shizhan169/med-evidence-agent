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


def test_query_tokens_extract_figure_and_table_refs():
    # "图 4" 拼成 figure4，"图 3" 拼成 figure3，两者不能混
    q4 = rag._query_tokens("Fujita 2023 的图 4 展示的是什么？")
    assert "figure4" in q4
    assert "figure3" not in q4
    assert "figure3" in rag._query_tokens("Fujita 2023 的图 3 是什么图？")
    assert "table2" in rag._query_tokens("Table 2 里列了什么？")
    assert "table2" in rag._query_tokens("表 2 里列了什么？")
    # 没有图/表引用的问题不带 figure/table token
    plain = rag._query_tokens("总皮层灰质体积在几岁达到峰值？")
    assert not [t for t in plain if t.startswith("figure") or t.startswith("table")]
    # 英文写法也认
    assert "figure4" in rag._query_tokens("What does Figure 4 show?")
    assert "figure4" in rag._query_tokens("What does Fig. 4 show?")
    # _tokenize 本身没动：一位数字仍不进通用分词
    assert "4" not in rag._tokenize("Fujita 2023 的图 4 展示的是什么？")


def test_bm25_prefers_figure4_caption_for_f02():
    # 不用 Chroma、不用模型：直接用 rank_bm25 给 F02 的查询 token 打分。
    # 语料按生产 _get_bm25_index 的做法建：文件名 token + 入库文档 token。
    pytest.importorskip("rank_bm25")
    from pathlib import Path

    from rank_bm25 import BM25Okapi

    from app.ingest import _extract_captions, _pack_caption, parse_pdf

    pdf = Path(__file__).resolve().parents[1] / "data" / "papers" / "2023-fujita-brain-volume-aging.pdf"
    if not pdf.exists():
        pytest.skip("data/papers 里没有 Fujita 2023，跳过 BM25 排序检查")
    p8 = parse_pdf(pdf)[7]
    caps = _extract_captions(p8["text"])
    cap3 = next(t for k, n, t, _ in caps if (k, n) == ("Figure", "3"))
    cap4 = next(t for k, n, t, _ in caps if (k, n) == ("Figure", "4"))
    ctx = "【文献：Fujita Brain Volume Aging｜第 8 页｜方法】"
    page = {"page": 8, "text": p8["text"], "source": "2023-fujita-brain-volume-aging.pdf"}
    c3 = _pack_caption(cap3, page, ctx, 0)
    c4 = _pack_caption(cap4, page, ctx, 0)
    assert len(c3) == 1 and len(c4) == 1  # 两幅图的图注都是单片段
    # D 片段：图 3 图注正文（正文切分后的样子），不带 figure token
    d_text = (
        "Solid black lines indicate the median values. Different colors indicate "
        "different age groups. The x-axis shows the normalized volume change per "
        "year as a percentage of the absolute volume of the given structure. Figure 4."
    )
    docs = [
        ("2023-fujita-brain-volume-aging.pdf", c3[0]["context"] + "\n" + c3[0]["text"]),
        ("2023-fujita-brain-volume-aging.pdf", c4[0]["context"] + "\n" + c4[0]["text"]),
        ("2023-fujita-brain-volume-aging.pdf", ctx + "\n" + d_text),
        ("2022-bethlehem-brain-charts.pdf",
         "【文献：Bethlehem Brain Charts｜第 1 页｜标题与摘要】\n"
         "The study included 123,984 magnetic resonance imaging scans from 101,457 participants."),
        ("2024-ge-centilebrain.pdf",
         "【文献：Ge Centilebrain｜第 1 页｜标题与摘要】\n"
         "CentileBrain included 37,407 healthy individuals from 87 datasets aged 3–90 years."),
        ("2026-seidel-baseline-cortical-thickness.pdf",
         "【文献：Seidel Baseline Cortical Thickness｜第 1 页｜标题与摘要】\n"
         "B-Norm uses baseline cortical thickness to predict follow-up thickness."),
    ]
    corpus = [
        rag._tokenize(fname.replace("-", " ").replace("_", " ")) + rag._tokenize(doc)
        for fname, doc in docs
    ]
    tokens = rag._query_tokens("Fujita 2023 的图 4 展示的是什么？")
    assert tokens == ["2023", "figure4", "fujita"]
    scores = BM25Okapi(corpus).get_scores(tokens)
    # 图 4 图注是唯一带 figure4 的片段，必须排在图 3 图注和 D 之前
    assert scores[1] > scores[0] and scores[1] > scores[2]


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
