"""Router 测试（框架 v2，第一阶段 1.1）：不需要 LLM、不需要 Chroma、不联网。
规则分类用真规则；LLM 分类用假 LLM（monkeypatch）。
跑法：pytest tests/test_router.py -v

思路：路由是结构化输出 {task_type, requires_planning,
requires_external_search, requires_vision}，三个 flag 可组合，不是 4 选 1。
规则明确时直接分类；规则覆盖不了的必须走 LLM——没有“其余 → single_doc”
兜底，否则 LLM 兜底永远触发不了。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.agent as agent

CHUNKS = [{"text": "总白质体积在 28.7 岁达到峰值。", "source": "2022-bethlehem-brain-charts.pdf", "page": 3}]
ANSWER = "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]"
SIMPLE_ROUTE = {"task_type": "simple", "requires_planning": False,
                "requires_external_search": False, "requires_vision": False}


# ---------- 规则分类：5 类路由 ----------

def test_factoid_routes_to_retriever(monkeypatch):
    # 单篇事实问答：规则直接分类，不调 LLM
    calls = {"n": 0}

    def fake_llm(prompt):
        calls["n"] += 1
        return "{}"

    monkeypatch.setattr("app.agent._call_llm", fake_llm)
    route = agent.router_node({"query": "总白质体积在几岁达到峰值？"})["route"]
    assert route == SIMPLE_ROUTE
    assert calls["n"] == 0  # 规则明确，不调 LLM


def test_comparison_routes_to_planner():
    # 跨文献对比：需要拆解 → requires_planning，task_type 是 complex
    route = agent.router_node({"query": "比较五篇纵向常模论文的数据集有什么不同"})["route"]
    assert route["requires_planning"] is True
    assert route["task_type"] == "complex"
    assert route["requires_external_search"] is False
    assert route["requires_vision"] is False


def test_latest_routes_to_pubmed():
    # 问最新文献：需要外部搜索
    route = agent.router_node({"query": "纵向脑常模模型最新的文献有哪些？"})["route"]
    assert route["requires_external_search"] is True
    assert route["requires_planning"] is False
    assert route["requires_vision"] is False


def test_figure_routes_to_chart_read():
    # 问图：需要读图
    route = agent.router_node({"query": "Fujita 2023 的图 4 是什么图？"})["route"]
    assert route["requires_vision"] is True
    assert route["requires_planning"] is False
    assert route["requires_external_search"] is False


def test_combined_flags_all_true():
    # “对比最近三篇论文中 Figure 3”：规划 + 外部搜索 + 读图，三个 flag 全为 True
    route = agent.router_node({"query": "对比最近三篇论文中 Figure 3 的实验结果"})["route"]
    assert route["requires_planning"] is True
    assert route["requires_external_search"] is True
    assert route["requires_vision"] is True
    assert route["task_type"] == "complex"


# ---------- 分流优先级 ----------

def test_route_after_router_priority():
    # 优先级：planning → vision → 外部搜索 → 本地检索
    assert agent._route_after_router(
        {"route": {"requires_planning": True, "requires_vision": True,
                   "requires_external_search": True}}) == "planner"
    assert agent._route_after_router(
        {"route": {"requires_planning": False, "requires_vision": True,
                   "requires_external_search": True}}) == "chart_read"
    assert agent._route_after_router(
        {"route": {"requires_planning": False, "requires_vision": False,
                   "requires_external_search": True},
         "use_pubmed": True}) == "pubmed"
    # use_pubmed=False（评测）时不走 PubMed——28 道题测本地 RAG
    assert agent._route_after_router(
        {"route": {"requires_planning": False, "requires_vision": False,
                   "requires_external_search": True},
         "use_pubmed": False}) == "retriever"
    assert agent._route_after_router({"route": {}}) == "retriever"
    assert agent._route_after_router({}) == "retriever"


# ---------- LLM 兜底：规则覆盖不了时必须走 LLM ----------

def test_ambiguous_question_goes_to_llm(monkeypatch):
    # “这个结果可靠吗？”：没有规则标志，又不是明显事实问答（有“可靠”推理词）
    # → 规则返回 None，必须走 LLM 分类，不能直接掉进 simple
    calls = {"n": 0, "prompts": []}

    def fake_llm(prompt):
        calls["n"] += 1
        calls["prompts"].append(prompt)
        return ('{"task_type": "complex", "requires_planning": true, '
                '"requires_external_search": false, "requires_vision": false}')

    monkeypatch.setattr("app.agent._call_llm", fake_llm)
    route = agent.router_node({"query": "这个结果可靠吗？"})["route"]
    assert calls["n"] == 1
    assert route["requires_planning"] is True
    assert route["task_type"] == "complex"


def test_llm_garbage_falls_back_to_simple(monkeypatch):
    # LLM 返回的不是 JSON：解析失败，退回简单问答（不断问答）
    monkeypatch.setattr("app.agent._call_llm", lambda prompt: "我不知道怎么分类")
    route = agent.router_node({"query": "这个结果可靠吗？"})["route"]
    assert route == SIMPLE_ROUTE


def test_llm_failure_falls_back_to_simple(monkeypatch):
    # LLM 调用抛异常：退回简单问答
    def boom(prompt):
        raise RuntimeError("LLM 挂了")

    monkeypatch.setattr("app.agent._call_llm", boom)
    route = agent.router_node({"query": "这个结果可靠吗？"})["route"]
    assert route == SIMPLE_ROUTE


# ---------- answer_status 派生：verified 是兼容字段 ----------

def test_answer_status_derivation():
    # _to_response：verified 只有 answer_status == "verified" 时为 True
    base = {"final": "答案", "chunks": [], "pubmed_results": []}
    for status, verified in [("verified", True), ("partial", False),
                             ("refused", False), ("error", False)]:
        result = agent._to_response({**base, "answer_status": status})
        assert result["answer_status"] == status
        assert result["verified"] is verified
    # 还没定论（空）按 refused 处理
    result = agent._to_response({**base, "answer_status": ""})
    assert result["answer_status"] == "refused"
    assert result["verified"] is False


# ---------- 端到端：路由不破坏正常问答 ----------

def _run_with_fake_llm(monkeypatch, drafts):
    """假检索 + 假 LLM（按顺序取答案），跑一次完整问答。"""
    calls = {"n": 0}

    def fake_llm(prompt):
        calls["n"] += 1
        return drafts[min(calls["n"] - 1, len(drafts) - 1)]

    monkeypatch.setattr("app.rag.retrieve", lambda query, top_k=10: CHUNKS)
    monkeypatch.setattr("app.agent._call_llm", fake_llm)
    return agent.run("总白质体积在几岁达到峰值？"), calls


def test_run_verified_status(monkeypatch):
    result, calls = _run_with_fake_llm(monkeypatch, [ANSWER])
    assert result["answer_status"] == "verified"
    assert result["verified"] is True


def test_run_refused_status(monkeypatch):
    result, calls = _run_with_fake_llm(monkeypatch, [agent.REFUSAL])
    assert result["answer"] == agent.REFUSAL
    assert result["answer_status"] == "refused"
    assert result["verified"] is False


def test_run_error_status_when_llm_unavailable(monkeypatch):
    # 假 LLM 返回空串（和没 key 时 _call_llm 的行为一致）→ [待接 LLM] → error
    result, calls = _run_with_fake_llm(monkeypatch, [""])
    assert result["answer_status"] == "error"
    assert result["verified"] is False
    assert result["answer"].startswith("[待接 LLM]")


def test_planning_branch_temporarily_falls_through_to_retriever(monkeypatch):
    # 1.2 之前：requires_planning 的问题暂时走 retriever，不报错、能出答案。
    # 这是临时行为，1.2 接入 Planner 后要改成真正走 planner 节点。
    calls = {"n": 0}

    def fake_llm(prompt):
        calls["n"] += 1
        return ANSWER

    monkeypatch.setattr("app.rag.retrieve", lambda query, top_k=10: CHUNKS)
    monkeypatch.setattr("app.agent._call_llm", fake_llm)
    result = agent.run("比较五篇纵向常模论文的数据集有什么不同")
    assert result["answer_status"] == "verified"
    assert result["verified"] is True
    assert calls["n"] == 1  # 只有回答员调了一次 LLM，Router 是规则分类、没调
