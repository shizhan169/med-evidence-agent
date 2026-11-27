"""PubMed MCP 搜索测试（第 5 步，5a）：不需要真 LLM、不需要 Chroma、不联网。
跑法（仓库根目录，先激活环境）：
    pytest tests/test_pubmed.py -v

思路：把调 LLM、本地检索、MCP 搜索都换成假的（monkeypatch），只验证
触发规则（本地没证据 / 明确问新文献 / use_pubmed=False）、PMID 核查
（假 PMID 被拦、真 PMID 通过）、MCP 挂了退回拒答、以及解析器能读
服务端 2.12.4 实测返回的 Markdown 格式（fixture 就是真跑回来的文本）。
真的 PubMed 端到端（npx + NCBI）要等用户本地跑一次 Streamlit 提问验收。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.agent as agent
import app.mcp_tools as mcp_tools
from app.rag import build_pubmed_prompt

CHUNKS = [{"text": "总白质体积在 28.7 岁达到峰值。", "source": "2022-bethlehem-brain-charts.pdf", "page": 3}]

PUBMED_RESULTS = [
    {
        "pmid": "38395541",
        "title": "Normative modelling of brain morphometry across the lifespan with CentileBrain",
        "authors": "Ruiyang Ge、Yuetong Yu、Yi Xuan Qi 等",
        "journal": "The Lancet. Digital health, (Lancet Digit Health), 2024 Mar, 6(3), e211-e221",
        "year": "2024",
        "abstract": "We developed this framework with regional morphometric data from 37,407 healthy individuals aged 3-90 years.",
    },
    {
        "pmid": "37024597",
        "title": "Heterogeneous aging across multiple organ systems",
        "authors": "Jane Doe 等",
        "journal": "Nature medicine, (Nat Med), 2023 May, 29(5), 1221-1231",
        "year": "2023",
        "abstract": "",
    },
]

# 真跑回来的 pubmed_search_articles 返回（Markdown，只含 PMID 行）
SEARCH_FIXTURE = """## PubMed Search Results
**Query:** `longitudinal brain normative model`
**Returned:** 3 of 347 | **Offset:** 0
**Search URL:** <https://pubmed.ncbi.nlm.nih.gov/?term=longitudinal%20brain%20normative%20model>

**PMIDs:** 38395541, 37024597, 39647778
"""

# 真跑回来的 pubmed_fetch_articles 返回（Markdown，截取两篇；第二篇没有摘要）
FETCH_FIXTURE = """## PubMed Articles
**Articles Returned:** 2

### Normative modelling of brain morphometry across the lifespan with CentileBrain: algorithm benchmarking and model optimisation.

**Authors (31):**
- Ruiyang Ge (R) [aff 0]
- Yuetong Yu (Y) [aff 0]
- Yi Xuan Qi (YX) [aff 0]

**Affiliations:**
- [0] Djavad Mowafaghian Centre for Brain Health, University of British Columbia, Vancouver, BC, Canada.

**Journal:** The Lancet. Digital health, (Lancet Digit Health), 2024 Mar, **6**(3), e211-e221, pii: S2589-7500(23)00250-9, eISSN 2589-7500
**Record Type:** journal-article
**PMID:** 38395541
**DOI:** 10.1016/S2589-7500(23)00250-9
**PMCID:** PMC10929064
**PubMed:** <https://pubmed.ncbi.nlm.nih.gov/38395541/>

#### Abstract
The value of normative models in research and clinical practice relies on their robustness and a systematic comparison of different modelling algorithms and parameters; however, this has not been done to date.

#### MeSH Terms
- Humans [D006801]

### Heterogeneous aging across multiple organ systems and prediction of chronic disease and mortality.

**Authors (6):**
- Jane Doe (J) [aff 0]

**Journal:** Nature medicine, (Nat Med), 2023 May, **29**(5), 1221-1231, eISSN 1546-170X
**PMID:** 37024597
**DOI:** 10.1038/s41591-023-02296-6
**PubMed:** <https://pubmed.ncbi.nlm.nih.gov/37024597/>
"""


def _run_with_mocks(monkeypatch, query, chunks, pubmed_results, drafts,
                    translated="longitudinal brain normative model", use_pubmed=True):
    """跑一次 agent.run（假检索 + 假 MCP + 假 LLM）。
    drafts 是假 LLM 回答员每次调用返回的答案（按顺序取）；翻译调用另外算，
    返回翻译词 translated。返回 (result, search_calls, llm)，
    search_calls 记每次搜 PubMed 用的 query 和 sort，llm 记翻译/回答调用数。"""
    search_calls = []
    llm = {"translate": 0, "answer": 0, "prompts": []}

    def fake_search(q, top_k=5, sort="relevance"):
        search_calls.append({"query": q, "top_k": top_k, "sort": sort})
        return list(pubmed_results)

    def fake_llm(prompt):
        llm["prompts"].append(prompt)
        if prompt.startswith("把下面的中文问题翻译成"):
            llm["translate"] += 1
            return translated
        llm["answer"] += 1
        return drafts[min(llm["answer"] - 1, len(drafts) - 1)]

    monkeypatch.setattr("app.rag.retrieve", lambda q, top_k=10, llm_call=None: list(chunks))
    monkeypatch.setattr("app.mcp_tools.search_pubmed", fake_search)
    monkeypatch.setattr("app.agent._call_llm", fake_llm)
    return agent.run(query, use_pubmed=use_pubmed), search_calls, llm


# ---- 解析器：读服务端 2.12.4 实测的 Markdown ----

def test_parse_search_result_pmids():
    assert mcp_tools._parse_search_result(SEARCH_FIXTURE) == ["38395541", "37024597", "39647778"]


def test_parse_search_result_no_hits():
    assert mcp_tools._parse_search_result("## PubMed Search Results\n**Returned:** 0 of 0\n") == []
    assert mcp_tools._parse_search_result("") == []


def test_parse_fetch_result_fields():
    articles = mcp_tools._parse_fetch_result(FETCH_FIXTURE)
    assert len(articles) == 2
    first, second = articles
    assert first["pmid"] == "38395541"
    assert first["title"].startswith("Normative modelling of brain morphometry")
    assert first["authors"] == "Ruiyang Ge、Yuetong Yu、Yi Xuan Qi 等"
    assert "Lancet" in first["journal"] and first["year"] == "2024"
    assert first["abstract"].startswith("The value of normative models")
    assert second["pmid"] == "37024597"
    assert second["year"] == "2023"
    assert second["abstract"] == ""  # 这篇 fixture 里没有摘要段


def test_parse_fetch_result_skips_block_without_pmid():
    text = "## PubMed Articles\n\n### 一篇没有 PMID 的块\n\n**Journal:** xxx, 2024\n"
    assert mcp_tools._parse_fetch_result(text) == []


def test_parse_authors_short_list():
    block = "**Authors (2):**\n- Alice Smith (A) [aff 0]\n- Bob Jones (B) [aff 1]\n"
    assert mcp_tools._parse_authors(block) == "Alice Smith、Bob Jones"


# ---- 触发规则 ----

def test_trigger_no_local_evidence(monkeypatch):
    # 本地没证据 → 调 PubMed；答案带真 PMID，核查通过；引用列表是 PubMed 的
    result, search_calls, llm = _run_with_mocks(
        monkeypatch,
        "latest literature on longitudinal brain normative models",
        chunks=[],
        pubmed_results=PUBMED_RESULTS,
        drafts=["CentileBrain 用 37,407 人建了常模 PubMed PMID:38395541"],
    )
    assert len(search_calls) == 1
    assert search_calls[0]["query"] == "latest literature on longitudinal brain normative models"
    assert result["verified"] is True
    assert "38395541" in result["answer"]
    assert result["citations"][0]["source"] == "PubMed"
    assert result["citations"][0]["pmid"] == "38395541"
    assert len(result["citations"]) == 2


def test_trigger_new_literature_even_with_local_chunks(monkeypatch):
    # 本地有证据，但用户明确问新文献 → 还是调 PubMed，答案用 PubMed 的
    result, search_calls, llm = _run_with_mocks(
        monkeypatch,
        "最近关于纵向脑常模的新文献有哪些",
        chunks=CHUNKS,
        pubmed_results=PUBMED_RESULTS,
        drafts=["最新的一篇是 CentileBrain 常模 PubMed PMID:38395541"],
    )
    assert len(search_calls) == 1
    # 2026-11-26 改：候选池按相关性排（sort=relevance），再按题目相关性重排
    # （命中几个同义表达、再按年份），不再按发表日期倒序直接取前 10
    assert search_calls[0]["sort"] == "relevance"
    assert search_calls[0]["top_k"] == 30
    assert result["verified"] is True
    assert "38395541" in result["answer"]


def test_no_pubmed_when_local_has_evidence(monkeypatch):
    # 本地有证据、普通问题 → 不调 PubMed，走老路
    result, search_calls, llm = _run_with_mocks(
        monkeypatch,
        "总白质体积在几岁达到峰值？",
        chunks=CHUNKS,
        pubmed_results=PUBMED_RESULTS,
        drafts=["总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]"],
    )
    assert search_calls == []
    assert result["verified"] is True
    assert result["citations"][0]["source"] == "2022-bethlehem-brain-charts.pdf"


def test_no_chunks_with_pubmed_on_searches_then_refuses(monkeypatch):
    # 第 5 步的新行为：PubMed 开着、本地没证据 → 先调 PubMed（中文问题先翻
    # 一次英文）；PubMed 也没结果 → 拒答，回答员一次 LLM 都没调
    result, search_calls, llm = _run_with_mocks(
        monkeypatch,
        "总白质体积在几岁达到峰值？",
        chunks=[],
        pubmed_results=[],
        drafts=["不该被调用"],
    )
    assert len(search_calls) == 1
    assert search_calls[0]["query"] == "longitudinal brain normative model"
    assert llm["translate"] == 1
    assert llm["answer"] == 0
    assert result["answer"] == agent.REFUSAL
    assert result["verified"] is False


def test_use_pubmed_false_disables(monkeypatch):
    # 评测传 use_pubmed=False：本地没证据也不调 PubMed，直接拒答
    result, search_calls, llm = _run_with_mocks(
        monkeypatch,
        "latest literature on longitudinal brain normative models",
        chunks=[],
        pubmed_results=PUBMED_RESULTS,
        drafts=["不该被调用"],
        use_pubmed=False,
    )
    assert search_calls == []
    assert llm["answer"] == 0
    assert result["answer"] == agent.REFUSAL
    assert result["verified"] is False


# ---- PMID 核查：假 PMID 必须被拦 ----

def test_fake_pmid_blocked_then_retry_pass(monkeypatch):
    # 第一次编了个工具没返回过的 PMID，核查失败；重答后引用真 PMID，通过
    result, search_calls, llm = _run_with_mocks(
        monkeypatch,
        "latest literature on longitudinal brain normative models",
        chunks=[],
        pubmed_results=PUBMED_RESULTS,
        drafts=[
            "有一篇讲常模 PubMed PMID:99999999",
            "CentileBrain 常模 PubMed PMID:38395541",
        ],
    )
    assert llm["answer"] == 2
    assert result["verified"] is True
    assert "38395541" in result["answer"]
    # 重答的 prompt 里有 PubMed 重试提醒，不是本地那条
    assert "不要自己编 PMID" in llm["prompts"][1]
    assert "不要自己编页码" not in llm["prompts"][1]


def test_fake_pmid_all_retries_refuse(monkeypatch):
    # 三次都编假 PMID：1 次原答 + 2 次重试，之后拒答
    result, search_calls, llm = _run_with_mocks(
        monkeypatch,
        "latest literature on longitudinal brain normative models",
        chunks=[],
        pubmed_results=PUBMED_RESULTS,
        drafts=["有一篇讲常模 PubMed PMID:99999999"],
    )
    assert llm["answer"] == 3
    assert result["answer"] == agent.REFUSAL
    assert result["verified"] is False


def test_pubmed_answer_without_pmid_retries(monkeypatch):
    # PubMed 路径的答案没给 PMID：重答后给了，通过
    result, search_calls, llm = _run_with_mocks(
        monkeypatch,
        "latest literature on longitudinal brain normative models",
        chunks=[],
        pubmed_results=PUBMED_RESULTS,
        drafts=[
            "CentileBrain 用 37,407 人建了常模。",
            "CentileBrain 用 37,407 人建了常模 PubMed PMID:38395541",
        ],
    )
    assert llm["answer"] == 2
    assert result["verified"] is True


def test_pubmed_answer_with_local_style_citation_blocked(monkeypatch):
    # PubMed 答案里混了本地式引用 [来源：x.pdf p.N]：核查不通过，重答后改对
    result, search_calls, llm = _run_with_mocks(
        monkeypatch,
        "latest literature on longitudinal brain normative models",
        chunks=[],
        pubmed_results=PUBMED_RESULTS,
        drafts=[
            "CentileBrain 常模 PubMed PMID:38395541 [来源：2022-bethlehem-brain-charts.pdf p.3]",
            "CentileBrain 常模 PubMed PMID:38395541",
        ],
    )
    assert llm["answer"] == 2
    assert result["verified"] is True


# ---- PubMed 挂了：退回本地-only / 拒答，不崩 ----

def test_pubmed_down_falls_back_to_refusal(monkeypatch):
    # MCP 挂了（search 返回 []）：本地也没证据 → 拒答，LLM 一次没调，不崩
    result, search_calls, llm = _run_with_mocks(
        monkeypatch,
        "latest literature on longitudinal brain normative models",
        chunks=[],
        pubmed_results=[],
        drafts=["不该被调用"],
    )
    assert len(search_calls) == 1
    assert llm["answer"] == 0
    assert result["answer"] == agent.REFUSAL
    assert result["verified"] is False


def test_pubmed_down_falls_back_to_local(monkeypatch):
    # MCP 挂了，但本地有证据 → 用本地回答，不崩
    result, search_calls, llm = _run_with_mocks(
        monkeypatch,
        "总白质体积在几岁达到峰值？",
        chunks=CHUNKS,
        pubmed_results=[],
        drafts=["总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]"],
    )
    assert result["verified"] is True
    assert "28.7" in result["answer"]


def test_search_pubmed_returns_empty_when_disabled(monkeypatch):
    # MCP_PUBMED_ENABLED=false：search_pubmed 直接返回 []，不拉子进程
    monkeypatch.setattr("app.mcp_tools.MCP_PUBMED_ENABLED", False)
    assert mcp_tools.search_pubmed("brain") == []


# ---- 中文问题要翻成英文再搜（PubMed 只认英文） ----

def test_chinese_query_translated_before_search(monkeypatch):
    # 中文问题：先调一次 LLM 翻译，再拿翻译后的英文词去搜 PubMed
    result, search_calls, llm = _run_with_mocks(
        monkeypatch,
        "最近关于纵向脑常模的新文献有哪些",
        chunks=[],
        pubmed_results=PUBMED_RESULTS,
        drafts=["最新的一篇是 CentileBrain 常模 PubMed PMID:38395541"],
    )
    assert llm["translate"] == 1
    assert search_calls[0]["query"] == "longitudinal brain normative model"
    assert result["verified"] is True


def test_chinese_query_translated_to_or_synonyms(monkeypatch):
    # 2026-11-26 改：翻译输出核心概念的 2-3 个同义表达，用 OR 连接，扩大召回
    or_query = '"longitudinal normative modeling" OR "brain charts" OR "normative brain model"'
    result, search_calls, llm = _run_with_mocks(
        monkeypatch,
        "最近关于纵向脑常模的新文献有哪些",
        chunks=[],
        pubmed_results=PUBMED_RESULTS,
        drafts=["最新的一篇是 CentileBrain 常模 PubMed PMID:38395541"],
        translated=or_query,
    )
    assert llm["translate"] == 1
    assert search_calls[0]["query"] == or_query
    assert result["verified"] is True


def test_malformed_translation_falls_back_to_keywords(monkeypatch):
    # 翻译格式不对（引号不成对、没有 OR）就退回空格分隔的关键词，不直接拿坏检索式去搜
    result, search_calls, llm = _run_with_mocks(
        monkeypatch,
        "最近关于纵向脑常模的新文献有哪些",
        chunks=[],
        pubmed_results=PUBMED_RESULTS,
        drafts=["最新的一篇是 CentileBrain 常模 PubMed PMID:38395541"],
        translated='"longitudinal normative modeling',
    )
    assert search_calls[0]["query"] == "longitudinal normative modeling"
    assert '"' not in search_calls[0]["query"]
    assert result["verified"] is True


# ---- 相关性重排：先取 30 候选，再按题目相关性排、取前 10 ----

OR_QUERY = '"longitudinal normative modeling" OR "brain charts" OR "normative brain model"'

_RANK_PAPERS = [
    # 命中 0 个表达、2027 年：最新但不相关，排最后
    {"pmid": "1", "title": "Longitudinal brain ageing", "abstract": "longitudinal changes in brain structure", "year": "2027"},
    # 命中 1 个表达（brain charts）、2026 年
    {"pmid": "2", "title": "Brain charts across development", "abstract": "brain charts show developmental trajectories", "year": "2026"},
    # 命中 2 个表达、2024 年（modelling 归一成 model）
    {"pmid": "3", "title": "Longitudinal normative modelling in schizophrenia", "abstract": "longitudinal normative modelling of brain morphometry", "year": "2024"},
    # 命中 1 个表达（normative brain model）、2027 年：和 2 同命中数，年份新的在前
    {"pmid": "4", "title": "Normative modeling of brain morphometry", "abstract": "normative modeling across the lifespan", "year": "2027"},
    # 命中 3 个表达、2025 年：命中最多，排第一（不是最新）
    {"pmid": "5", "title": "Normative modelling of brain charts", "abstract": "longitudinal brain charts and normative modelling", "year": "2025"},
]


def test_rank_pubmed_results_by_phrase_hits_then_year():
    # 主键 = 命中几个同义表达，次键 = 年份倒序
    ranked = agent._rank_pubmed_results(_RANK_PAPERS, OR_QUERY, top_k=10)
    assert [r["pmid"] for r in ranked] == ["5", "3", "4", "2", "1"]


def test_rank_pubmed_results_trims_to_top_k():
    ranked = agent._rank_pubmed_results(_RANK_PAPERS, OR_QUERY, top_k=3)
    assert [r["pmid"] for r in ranked] == ["5", "3", "4"]


def test_rank_pubmed_results_non_or_query_keeps_pubmed_order():
    # 不是 OR 式（纯英文问题没翻译、或翻译退回关键词）时按 PubMed 原顺序取前 top_k，不硬排
    ranked = agent._rank_pubmed_results(_RANK_PAPERS, "longitudinal brain normative model", top_k=2)
    assert [r["pmid"] for r in ranked] == ["1", "2"]


def test_pubmed_node_fetches_candidates_and_ranks(monkeypatch):
    # 候选池取 30 篇（按相关性排），再重排取前 10 给回答员；这里只有 2 篇候选，全给
    result, search_calls, llm = _run_with_mocks(
        monkeypatch,
        "最近关于纵向脑常模的新文献有哪些",
        chunks=[],
        pubmed_results=PUBMED_RESULTS,
        drafts=["最新的一篇是 CentileBrain 常模 PubMed PMID:38395541"],
        translated=OR_QUERY,
    )
    assert len(search_calls) == 1
    assert search_calls[0]["top_k"] == 30
    assert search_calls[0]["sort"] == "relevance"
    assert result["verified"] is True
    assert len(result["citations"]) == 2


def test_english_query_skips_translation(monkeypatch):
    # 纯英文问题：不翻译，直接搜原问题，LLM 只调一次（回答）
    result, search_calls, llm = _run_with_mocks(
        monkeypatch,
        "latest literature on longitudinal brain normative models",
        chunks=[],
        pubmed_results=PUBMED_RESULTS,
        drafts=["CentileBrain 常模 PubMed PMID:38395541"],
    )
    assert llm["translate"] == 0
    assert llm["answer"] == 1
    assert search_calls[0]["query"] == "latest literature on longitudinal brain normative models"


# ---- 兜底路径、配置、prompt ----

def test_fallback_path_pubmed(monkeypatch):
    """LangGraph 不可用时走顺序执行兜底，PubMed 分支要和图里一致。"""
    def broken_graph():
        raise RuntimeError("LangGraph 不可用")

    monkeypatch.setattr("app.agent._build_graph", broken_graph)
    result, search_calls, llm = _run_with_mocks(
        monkeypatch,
        "latest literature on longitudinal brain normative models",
        chunks=[],
        pubmed_results=PUBMED_RESULTS,
        drafts=["CentileBrain 常模 PubMed PMID:38395541"],
    )
    assert len(search_calls) == 1
    assert result["verified"] is True


def test_config_defaults():
    import app.config as config

    assert config.MCP_PUBMED_COMMAND == "npx"
    assert "@cyanheads/pubmed-mcp-server@2.12.4" in config.MCP_PUBMED_ARGS
    assert config.MCP_PUBMED_TOOL_SEARCH == "pubmed_search_articles"
    assert config.MCP_PUBMED_TOOL_FETCH == "pubmed_fetch_articles"
    assert config.PUBMED_TOP_K == 10
    assert config.PUBMED_CANDIDATE_K == 30


def test_pubmed_prompt_uses_pmid_format():
    prompt = build_pubmed_prompt("最近关于纵向脑常模的新文献有哪些", PUBMED_RESULTS)
    assert "PubMed PMID:38395541" in prompt
    assert "Normative modelling of brain morphometry" in prompt
    assert "不要自己编 PMID" in prompt
    assert "[来源：" not in prompt  # PubMed 模板里不出现本地引用格式


def test_pubmed_prompt_v2_lists_literature():
    """v2 分两种问法：列文献时把相关论文都列出来、相关性分档直说，
    不能因为没用原词就悄悄扔掉（2026-11-26 用户本地实测：v1 只剩 1 篇）。"""
    from app.rag import PUBMED_ANSWERER_TEMPLATE

    assert "问有哪些文献" in PUBMED_ANSWERER_TEMPLATE
    assert "直接相关" in PUBMED_ANSWERER_TEMPLATE
    assert "悄悄扔掉" in PUBMED_ANSWERER_TEMPLATE
    assert "不要自己编 PMID" in PUBMED_ANSWERER_TEMPLATE
    assert "{history_text}" in PUBMED_ANSWERER_TEMPLATE
    assert "{context}" in PUBMED_ANSWERER_TEMPLATE
    assert "{query}" in PUBMED_ANSWERER_TEMPLATE
    prompt = build_pubmed_prompt("最近关于纵向脑常模的新文献有哪些", PUBMED_RESULTS)
    assert "问有哪些文献" in prompt
    assert "[来源：" not in prompt
