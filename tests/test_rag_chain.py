"""2.1 多阶段 RAG 检索链路测试：Query Rewrite、RRF 融合、上下文组装。

不需要 LLM、不需要真实模型、不联网：LLM 调用都用假的（传 llm_call），
Chroma 用 test_rerank.py 同款的假 embedding + 临时目录。
真的问答效果（28 道题数字、Q09 召回）要等用户本地跑完整评测。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.agent as agent
from app import rag

HISTORY = [
    {"role": "user", "content": "总皮层灰质体积（GMV）在几岁达到峰值？"},
    {"role": "assistant", "content": "总皮层灰质体积在 5.9 岁达到峰值。"},
]

FOLLOWUP = "那白质呢？"
PREPENDED = "总皮层灰质体积（GMV）在几岁达到峰值？ 那白质呢？"


# ---------- Query Rewrite：多轮追问改写 ----------

def test_rewrite_standalone_unchanged():
    # 没有指代词的完整问题、没历史、空问题，都原样返回
    assert rag.rewrite_query_for_retrieval("总白质体积在几岁达到峰值？", HISTORY) == "总白质体积在几岁达到峰值？"
    assert rag.rewrite_query_for_retrieval(FOLLOWUP, []) == FOLLOWUP
    assert rag.rewrite_query_for_retrieval(FOLLOWUP, None) == FOLLOWUP
    assert rag.rewrite_query_for_retrieval("", HISTORY) == ""


def test_rewrite_falls_back_to_prepend_without_llm():
    # 不传 llm_call 时只用拼上文的退回做法
    assert rag.rewrite_query_for_retrieval(FOLLOWUP, HISTORY) == PREPENDED


def test_rewrite_uses_llm_when_valid():
    # LLM 改出完整独立的问题（指代消解：那白质 → 总白质体积）就用它
    q = rag.rewrite_query_for_retrieval(
        FOLLOWUP, HISTORY, llm_call=lambda p: "总白质体积在几岁达到峰值？")
    assert q == "总白质体积在几岁达到峰值？"


def test_rewrite_rejects_answer_like_output():
    # 假 LLM 对任何 prompt 都返回同一段答案（陈述句、带引用），不能当改写结果
    fake = lambda p: "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]"
    q = rag.rewrite_query_for_retrieval(FOLLOWUP, HISTORY, llm_call=fake)
    assert q == PREPENDED


def test_rewrite_rejects_empty_and_exception():
    q = rag.rewrite_query_for_retrieval(FOLLOWUP, HISTORY, llm_call=lambda p: "")
    assert q == PREPENDED

    def boom(p):
        raise RuntimeError("no key")

    q2 = rag.rewrite_query_for_retrieval(FOLLOWUP, HISTORY, llm_call=boom)
    assert q2 == PREPENDED


def test_rewrite_uses_last_user_question():
    history = HISTORY + [
        {"role": "user", "content": "白质体积峰值是多少？"},
        {"role": "assistant", "content": "28.7 岁。"},
    ]
    q = rag.rewrite_query_for_retrieval("那它呢？", history)
    assert q.startswith("白质体积峰值是多少？")


def test_agent_build_retrieval_query_uses_llm_rewrite(monkeypatch):
    # agent.py 的 _build_retrieval_query 走 rag 的改写：LLM 改出好问题就用它
    monkeypatch.setattr("app.agent._call_llm", lambda p: "总白质体积在几岁达到峰值？")
    assert agent._build_retrieval_query(FOLLOWUP, HISTORY) == "总白质体积在几岁达到峰值？"


def test_agent_build_retrieval_query_rejects_fake_llm_answer(monkeypatch):
    # 假 LLM 返回答案文本时退回拼上文（test_multiturn.py 钉住的行为）
    monkeypatch.setattr(
        "app.agent._call_llm",
        lambda p: "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]")
    q = agent._build_retrieval_query(FOLLOWUP, HISTORY)
    assert "总皮层灰质体积" in q and "那白质呢" in q


# ---------- Query Rewrite：中文翻英文 ----------

def test_translate_pure_english_unchanged():
    called = []
    q = rag.translate_query_to_english(
        "What does Figure 4 show?", llm_call=lambda p: called.append(p) or "x")
    assert q == "What does Figure 4 show?"
    assert called == []  # 纯英文不调 LLM


def test_translate_chinese_with_fake_llm():
    q = rag.translate_query_to_english(
        "总白质体积在几岁达到峰值？",
        llm_call=lambda p: "At what age does total white matter volume peak?")
    assert q == "At what age does total white matter volume peak?"


def test_translate_falls_back_on_chinese_output():
    # 翻出来还是中文 → 原样返回
    q = rag.translate_query_to_english(
        "总白质体积在几岁达到峰值？", llm_call=lambda p: "总白质体积在 28.7 岁达到峰值")
    assert q == "总白质体积在几岁达到峰值？"


def test_translate_falls_back_without_llm_and_on_failure():
    assert rag.translate_query_to_english("总白质体积在几岁达到峰值？") == "总白质体积在几岁达到峰值？"
    assert rag.translate_query_to_english(
        "总白质体积在几岁达到峰值？", llm_call=lambda p: "") == "总白质体积在几岁达到峰值？"

    def boom(p):
        raise RuntimeError("no key")

    assert rag.translate_query_to_english(
        "总白质体积在几岁达到峰值？", llm_call=boom) == "总白质体积在几岁达到峰值？"  # noqa: E501


# ---------- RRF 融合 ----------

def _c(cid, text=None):
    return {"id": cid, "text": text if text is not None else cid,
            "source": "t.pdf", "page": 1}


def test_rrf_fuse_both_lists_win():
    # a 在两路都排第 1（分 2/61），b 只在向量第 1（1/61），c 只在 BM25 第 2（1/62）
    a, b, c = _c("a"), _c("b"), _c("c")
    fused = rag._rrf_fuse([[a, b], [a, c]])
    assert [x["id"] for x in fused] == ["a", "b", "c"]


def test_rrf_fuse_formula_and_order():
    # 精确算分（k=60）：x = 1/61+1/63，y = 1/62+1/61，z = 1/63+1/62 → y > x > z
    lists = [[_c("x"), _c("y"), _c("z")], [_c("y"), _c("z"), _c("x")]]
    fused = rag._rrf_fuse(lists, top_k=3)
    assert [x["id"] for x in fused] == ["y", "x", "z"]


def test_rrf_fuse_respects_top_k_and_dedupes():
    a, b = _c("a"), _c("b")
    fused = rag._rrf_fuse([[a, b], [b, a]], top_k=1)
    assert len(fused) == 1
    # a 和 b 分数相同（1/61+1/62），并列时按最好排名再按首次出现 → a
    assert fused[0]["id"] == "a"
    fused2 = rag._rrf_fuse([[a, b], [b, a]], top_k=10)
    assert [x["id"] for x in fused2] == ["a", "b"]  # 同一片段只出现一次


def test_rrf_fuse_empty_and_missing_id():
    assert rag._rrf_fuse([]) == []
    assert rag._rrf_fuse([[], []]) == []
    # 没 id 的片段融不进去（按 id 去重的前提）
    assert rag._rrf_fuse([[{"text": "no id", "source": "t", "page": 1}]]) == []


# ---------- 上下文组装 ----------

def test_assemble_dedupes_same_content():
    c1 = {"text": "same", "source": "t.pdf", "page": 1}
    c2 = {"text": "same", "source": "t.pdf", "page": 1}
    c3 = {"text": "other", "source": "t.pdf", "page": 2}
    assert rag._assemble_context([c1, c2, c3]) == [c1, c3]


def test_assemble_budget_truncates():
    chunks = [{"text": "x" * 100, "source": "t", "page": i} for i in range(5)]
    out = rag._assemble_context(chunks, max_chars=250)
    assert out == chunks[:2]  # 100+100=200，再加就超 250，排在前面的优先保留


def test_assemble_keeps_first_even_if_over_budget():
    chunks = [{"text": "x" * 500, "source": "t", "page": 1}]
    assert rag._assemble_context(chunks, max_chars=100) == chunks


def test_assemble_empty():
    assert rag._assemble_context([]) == []


# ---------- retrieve() 接线：RRF + 翻译 + 组装 ----------

def test_retrieve_fuses_vector_and_bm25(monkeypatch):
    seen = {}

    def fake_vector(collection, query, limit):
        seen["vector_query"] = query
        return [_c("a", "alpha"), _c("b", "beta"), _c("c", "gamma")]

    def fake_keyword(collection, tokens, limit=5, exclude_ids=None):
        seen["tokens"] = tokens
        return [_c("b", "beta"), _c("d", "delta")]

    monkeypatch.setattr(rag, "get_collection", lambda: object())
    monkeypatch.setattr(rag, "_vector_chunks", fake_vector)
    monkeypatch.setattr(rag, "_keyword_chunks", fake_keyword)
    monkeypatch.setattr(rag, "_get_reranker", lambda: None)
    out = rag.retrieve("Fujita 2023 的图 4 展示的是什么？", top_k=10)
    # RRF：b 两路都在（1/62+1/61）第 1；a 向量第 1（1/61）；d BM25 第 2（1/62）；c 向量第 3（1/63）
    assert [x["text"] for x in out] == ["beta", "alpha", "delta", "gamma"]
    assert all("id" not in x for x in out)  # 对外去掉 RRF 用的内部 id
    assert seen["vector_query"] == "Fujita 2023 的图 4 展示的是什么？"  # 不传 llm_call 不翻译
    assert seen["tokens"] == ["2023", "figure4", "fujita"]  # BM25 用原问题 token


def test_retrieve_translates_vector_query_only(monkeypatch):
    seen = {}

    def fake_vector(collection, query, limit):
        seen["vector_query"] = query
        return []

    def fake_keyword(collection, tokens, limit=5, exclude_ids=None):
        seen["tokens"] = tokens
        return []

    monkeypatch.setattr(rag, "get_collection", lambda: object())
    monkeypatch.setattr(rag, "_vector_chunks", fake_vector)
    monkeypatch.setattr(rag, "_keyword_chunks", fake_keyword)
    monkeypatch.setattr(rag, "_get_reranker", lambda: None)
    fake_llm = lambda p: "What does Figure 4 of Fujita 2023 show?"
    out = rag.retrieve("Fujita 2023 的图 4 展示的是什么？", top_k=5, llm_call=fake_llm)
    assert out == []
    assert seen["vector_query"] == "What does Figure 4 of Fujita 2023 show?"  # 向量用英文
    assert seen["tokens"] == ["2023", "figure4", "fujita"]  # BM25 仍用原问题 token


def test_retrieve_reranks_fused_pool(monkeypatch):
    # 重排器可用时：RRF 融合后的池子再重排取前 5（生产现状是关闭的，这里只保分支）
    class FakeReranker:
        def predict(self, pairs):
            return [1.0 if "TARGET" in doc else 0.0 for _, doc in pairs]

    def fake_vector(collection, query, limit):
        return [_c("a", "alpha"), _c("t", "TARGET here")]

    def fake_keyword(collection, tokens, limit=5, exclude_ids=None):
        return [_c("t", "TARGET here"), _c("b", "beta")]

    monkeypatch.setattr(rag, "get_collection", lambda: object())
    monkeypatch.setattr(rag, "_vector_chunks", fake_vector)
    monkeypatch.setattr(rag, "_keyword_chunks", fake_keyword)
    monkeypatch.setattr(rag, "_get_reranker", lambda: FakeReranker())
    out = rag.retrieve("TARGET", top_k=10)
    assert len(out) == 3  # 池子只有 3 个，重排取前 5 → 全给
    assert out[0]["text"] == "TARGET here"


def test_retrieve_empty_collection(monkeypatch):
    monkeypatch.setattr(
        rag, "get_collection",
        lambda: (_ for _ in ()).throw(RuntimeError("no chroma")))
    assert rag.retrieve("anything") == []
