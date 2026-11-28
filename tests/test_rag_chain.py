"""2.1 多阶段 RAG 检索链路测试：Query Rewrite、RRF 融合、上下文组装。

不需要 LLM、不需要真实模型、不联网：LLM 调用都用假的（传 llm_call），
Chroma 用 test_rerank.py 同款的假 embedding + 临时目录。
真的问答效果（28 道题数字、Q09 召回）要等用户本地跑完整评测。
"""
import sys
from pathlib import Path

import pytest

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


# ---------- 屏蔽 ChromaDB 遥测报错（2026-11-28） ----------

def test_chroma_telemetry_error_suppressed(caplog):
    # ChromaDB 0.5.0 + PostHog 8.x 不兼容，每次建 client / 取集合都会报两条
    # “Failed to send telemetry event …”，评测输出里刷屏。import app.rag 时
    # 就给 chromadb.telemetry.product.posthog 这个 logger 装上过滤器，
    # 只屏蔽这条噪音，其他日志照常。
    import logging

    lg = logging.getLogger("chromadb.telemetry.product.posthog")
    with caplog.at_level(logging.ERROR, logger="chromadb.telemetry.product.posthog"):
        lg.error("Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given")
        lg.error("这条不是遥测报错，应该正常显示")
    assert "Failed to send telemetry event" not in caplog.text
    assert "这条不是遥测报错，应该正常显示" in caplog.text


# ---------- 多篇点名时按篇取（2026-11-28，B 方案） ----------

_SOURCES = [
    "2022-bethlehem-brain-charts.pdf",
    "2023-dibiase-mapping-brain-charts.pdf",
    "2024-ge-centilebrain.pdf",
    "2026-seidel-baseline-cortical-thickness.pdf",
    "2022-rutherford-brain-growth-aging-precision.pdf",
    "2022-rutherford-normative-modeling-framework-psychiatry.pdf",
    "2025-rehak-buckova-normative-models-longitudinal-changes.pdf",
    "2017-coupe-unified-analysis-brain-maturation-aging.pdf",
]


def test_named_papers_single():
    assert rag._named_papers(
        "Bethlehem 2022 的脑图谱研究共汇集了多少 MRI 扫描？", _SOURCES
    ) == ["2022-bethlehem-brain-charts.pdf"]


def test_named_papers_five_papers():
    out = rag._named_papers(
        "比较五篇纵向常模论文（Bethlehem 2022、Di Biase 2023、Ge 2024、"
        "Seidel 2026、Rutherford 2022）的数据集、建模变量、模型、结果和局限性",
        _SOURCES,
    )
    # Rutherford 2022 有两篇（脑生长 + 精神病学框架），都算上，回答员按内容区分
    assert out == [
        "2022-bethlehem-brain-charts.pdf",
        "2023-dibiase-mapping-brain-charts.pdf",
        "2024-ge-centilebrain.pdf",
        "2026-seidel-baseline-cortical-thickness.pdf",
        "2022-rutherford-brain-growth-aging-precision.pdf",
        "2022-rutherford-normative-modeling-framework-psychiatry.pdf",
    ]


def test_named_papers_short_author_needs_year():
    # "ge" 太短，不带年份不算点名，避免误匹配 general/genetic
    assert rag._named_papers(
        "Ge 的 CentileBrain 研究纳入多少健康人？", _SOURCES
    ) == []
    assert rag._named_papers(
        "Ge 2024 的 CentileBrain 研究纳入多少健康人？", _SOURCES
    ) == ["2024-ge-centilebrain.pdf"]


def test_named_papers_compound_surname_and_accent():
    assert rag._named_papers(
        "Rehak-Buckova 2025 的纵向常模综述说了什么？", _SOURCES
    ) == ["2025-rehak-buckova-normative-models-longitudinal-changes.pdf"]
    assert rag._named_papers(
        "Coupé 2017 图 2 的体积轨迹基于多少名受试者？", _SOURCES
    ) == ["2017-coupe-unified-analysis-brain-maturation-aging.pdf"]
    assert rag._named_papers("Di Biase 2023 研究用了哪两个队列？", _SOURCES) == [
        "2023-dibiase-mapping-brain-charts.pdf"
    ]


def test_named_papers_none():
    assert rag._named_papers("今天上海天气怎么样？", _SOURCES) == []
    assert rag._named_papers(
        "用横断面脑图谱推算的年龄相关脑变化率，与纵向实测相比会怎样？", _SOURCES
    ) == []


def test_retrieve_per_paper_each_paper_contributes(monkeypatch):
    # 五篇点名时，每篇都要有片段进池子，综述不能把原文挤掉。
    # 假向量/BM25 按 where/source 参数返回对应论文的片段。
    def fake_vector(collection, query, limit, where=None):
        src = (where or {}).get("source", "")
        return [
            {"id": f"v-{src}-1", "text": f"vec {src} 1", "source": src, "page": 1},
            {"id": f"v-{src}-2", "text": f"vec {src} 2", "source": src, "page": 2},
        ]

    def fake_keyword(collection, tokens, limit=5, exclude_ids=None, source=None):
        return [
            {"id": f"k-{source}-1", "text": f"kw {source} 1", "source": source, "page": 1},
        ]

    monkeypatch.setattr(rag, "get_collection", lambda: object())
    monkeypatch.setattr(rag, "_distinct_sources", lambda c: _SOURCES)
    monkeypatch.setattr(rag, "_vector_chunks", fake_vector)
    monkeypatch.setattr(rag, "_keyword_chunks", fake_keyword)
    monkeypatch.setattr(rag, "_get_reranker", lambda: None)
    out = rag.retrieve(
        "比较五篇纵向常模论文（Bethlehem 2022、Di Biase 2023、Ge 2024、"
        "Seidel 2026、Rutherford 2022）的数据集",
        top_k=10,
    )
    got = {x["source"] for x in out}
    for src in [
        "2022-bethlehem-brain-charts.pdf",
        "2023-dibiase-mapping-brain-charts.pdf",
        "2024-ge-centilebrain.pdf",
        "2026-seidel-baseline-cortical-thickness.pdf",
        "2022-rutherford-brain-growth-aging-precision.pdf",
    ]:
        assert src in got, f"{src} 没进检索池"


def test_retrieve_single_paper_uses_global_path(monkeypatch):
    # 只点名一篇时走原有全局逻辑，不按篇取（_vector/_keyword 不带 where/source）。
    seen = {}

    def fake_vector(collection, query, limit, where=None):
        seen["where"] = where
        return [_c("a", "alpha")]

    def fake_keyword(collection, tokens, limit=5, exclude_ids=None, source=None):
        seen["source"] = source
        return [_c("b", "beta")]

    monkeypatch.setattr(rag, "get_collection", lambda: object())
    monkeypatch.setattr(rag, "_distinct_sources", lambda c: _SOURCES)
    monkeypatch.setattr(rag, "_vector_chunks", fake_vector)
    monkeypatch.setattr(rag, "_keyword_chunks", fake_keyword)
    monkeypatch.setattr(rag, "_get_reranker", lambda: None)
    out = rag.retrieve("Bethlehem 2022 的脑图谱研究共汇集了多少 MRI 扫描？", top_k=10)
    assert seen["where"] is None
    assert seen["source"] is None
    assert [x["text"] for x in out] == ["alpha", "beta"]


@pytest.mark.parametrize(("question", "expected_terms"), [
    ("各使用了什么数据集、样本、人群和纵向设计？", {"datasets", "participants"}),
    ("在建模中分别纳入哪些变量、协变量、暴露或结局定义？", {"predictors", "covariates"}),
    ("纵向常模论文建模中使用的结果变量、预测变量、协变量和时间变量", {"predictors", "covariates"}),
    ("各自采用的统计模型或纵向常模构建方法", {"algorithms", "regression"}),
    ("的主要结果、常模发现或关键估计是什么？", {"findings", "performance"}),
    ("分别报告了哪些局限性、偏倚或外部效度限制？", {"limitations", "generalisability"}),
])
def test_named_comparison_ranks_by_dimension(monkeypatch, question, expected_terms):
    query = "Bethlehem 2022、Di Biase 2023、Ge 2024、Seidel 2026 和 Rutherford 2022 " + question
    monkeypatch.setattr(rag, "get_collection", lambda: object())
    monkeypatch.setattr(rag, "_distinct_sources", lambda c: _SOURCES)
    monkeypatch.setattr(rag, "_get_reranker", lambda: None)

    def check_inputs(collection, vector_query, tokens, named, top_k):
        assert named == _SOURCES[:6]
        assert top_k == 10
        assert expected_terms <= set(tokens)
        assert tokens == rag._query_tokens(vector_query)
        assert set(tokens).isdisjoint(rag._query_tokens(query))
        return []

    monkeypatch.setattr(rag, "_retrieve_per_paper", check_inputs)
    rag.retrieve(query, llm_call=lambda _: pytest.fail("通用维度无需调用翻译模型"))


@pytest.mark.parametrize(("question", "qualifier"), [
    ("五篇纵向常模论文分别使用了哪些数据集、样本量、人群特征和随访时长？", "随访时长"),
    ("在四年随访时的主要结果", "四年随访"),
    ("图 4 的结果", "图 4"),
    ("使用的 MFPR 模型", "MFPR"),
    ("在 ABCD 中的主要结果", "ABCD"),
])
def test_comparison_keeps_qualifiers(question, qualifier):
    content = rag._comparison_content_query("Ge 2024 和 Seidel 2026 " + question, _SOURCES[2:4])
    assert qualifier in content
    assert "Ge 2024" not in content and "Seidel 2026" not in content


def test_english_comparison_strips_only_named_citations():
    query = "Compare ABCD in Ge 2024 and Seidel 2026: datasets, sample size and population characteristics"
    content = rag._comparison_content_query(query, _SOURCES[2:4])
    assert {"abcd", "participants", "datasets"} <= set(rag._query_tokens(content))
    assert {"ge", "seidel", "2024", "2026"}.isdisjoint(rag._query_tokens(content))


@pytest.mark.parametrize("question", ["的数据集、模型和结果", "图 4 中的蓝线表示什么？"])
def test_mixed_or_unknown_comparison_uses_existing_retrieval(question):
    assert not rag._comparison_content_query("Ge 2024 和 Seidel 2026 " + question, _SOURCES[2:4])
