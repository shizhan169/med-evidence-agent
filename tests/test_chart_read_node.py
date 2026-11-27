"""chart_read 节点接线测试（第二阶段 2.3 收尾）：VLM 读图节点 + 图节点接线。
不需要真实 VLM、不需要 API key、不联网；chart_read 工具和 retrieve 都换成假的。
跑法：pytest tests/test_chart_read_node.py -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import agent


def _base_state(query):
    return {
        "query": query, "chunks": [], "draft": "", "final": "", "answer_status": "",
        "route": {}, "plan": {}, "subtasks": [], "tool_calls": [], "figure": {}, "error": "",
        "history": [], "history_summary": "", "retrieval_query": query,
        "retries": 0, "retry": False, "verify_failures": [], "retry_target": "",
        "pubmed_results": [], "use_pubmed": False,
    }


def test_extract_fujita():
    paper, fig = agent._extract_paper_and_figure("Fujita 2023 的图 3 是什么图？")
    assert paper == "fujita"
    assert fig == "3"


def test_extract_dibiase_space_removed():
    paper, fig = agent._extract_paper_and_figure("Di Biase 2023 图 2 展示的是什么？")
    assert paper == "dibiase"
    assert fig == "2"


def test_extract_hedman_no_year():
    paper, fig = agent._extract_paper_and_figure("Hedman 图 3 的全脑体积寿命曲线覆盖几岁到几岁？")
    assert paper == "hedman"
    assert fig == "3"


def test_extract_coupe_accent_stripped():
    paper, fig = agent._extract_paper_and_figure("Coupé 2017 图 2 的体积轨迹基于多少名受试者？")
    assert paper == "coupe"
    assert fig == "2"


def test_extract_no_figure():
    paper, fig = agent._extract_paper_and_figure("Bethlehem 2022 的脑图谱研究共汇集了多少 MRI 扫描？")
    assert paper is None
    assert fig is None


def _fake_chart_read_ok(paper="", figure_no="", question="", image_path=""):
    return {
        "available": True, "source": "2023-fujita-brain-volume-aging.pdf", "page": 8,
        "figure_no": "4", "caption": "Figure 4. Visualization of Age-Dependent Volume Change Rate.",
        "reading": "三幅面板展示体积变化率，阴影表示 95% 置信区间。",
        "verification": {"verified": True}, "downgraded": False, "cropped": True, "reason": "",
    }


def test_chart_read_node_vlm_ok(monkeypatch):
    monkeypatch.setattr("app.agent.call_tool",
                        lambda name, **kw: {"success": True, "data": _fake_chart_read_ok(**kw), "error": None})
    out = agent.chart_read_node(_base_state("Fujita 2023 的图 4 展示的是什么？"))
    assert len(out["chunks"]) == 1
    chunk = out["chunks"][0]
    assert chunk["source"] == "2023-fujita-brain-volume-aging.pdf"
    assert chunk["page"] == 8
    assert "95%" in chunk["text"]
    assert "图注" in chunk["text"]
    assert out["figure"]["figure_no"] == "4"
    assert out["tool_calls"][0]["tool"] == "chart_read"


def test_chart_read_node_downgraded_note(monkeypatch):
    def fake(paper="", figure_no="", question="", image_path=""):
        d = _fake_chart_read_ok(paper, figure_no, question, image_path)
        d["downgraded"] = True
        d["verification"] = {"verified": False}
        return d
    monkeypatch.setattr("app.agent.call_tool",
                        lambda name, **kw: {"success": True, "data": fake(**kw), "error": None})
    out = agent.chart_read_node(_base_state("Fujita 2023 的图 4 展示的是什么？"))
    assert "可能不准" in out["chunks"][0]["text"]


def test_chart_read_node_fallback_to_retriever(monkeypatch):
    monkeypatch.setattr("app.agent.call_tool",
                        lambda name, **kw: {"success": True, "data": {"available": False, "reason": "本地没有 PDF"}, "error": None})
    fake_chunks = [{"source": "2023-fujita-brain-volume-aging.pdf", "page": 8, "text": "图注片段"}]
    monkeypatch.setattr("app.rag.retrieve", lambda q, top_k=10, llm_call=None: fake_chunks)
    out = agent.chart_read_node(_base_state("Fujita 2023 的图 4 展示的是什么？"))
    assert out["chunks"] == fake_chunks
    assert out["tool_calls"][0]["tool"] == "chart_read"


def test_chart_read_node_no_figure_fallback(monkeypatch):
    fake_chunks = [{"source": "2022-bethlehem-brain-charts.pdf", "page": 1, "text": "片段"}]
    monkeypatch.setattr("app.rag.retrieve", lambda q, top_k=10, llm_call=None: fake_chunks)
    out = agent.chart_read_node(_base_state("Bethlehem 2022 的脑图谱研究共汇集了多少 MRI 扫描？"))
    assert out["chunks"] == fake_chunks
    assert out["tool_calls"] == []


def test_route_after_router_vision():
    state = _base_state("Fujita 2023 的图 4 展示的是什么？")
    state["route"] = {"task_type": "simple", "requires_planning": False,
                      "requires_external_search": False, "requires_vision": True}
    assert agent._route_after_router(state) == "chart_read"


def test_router_sends_figure_to_chart_read(monkeypatch):
    monkeypatch.setattr("app.agent._call_llm", lambda prompt: "")
    state = _base_state("Fujita 2023 的图 4 展示的是什么？")
    state.update(agent.router_node(state))
    assert state["route"]["requires_vision"] is True
    assert agent._route_after_router(state) == "chart_read"
    monkeypatch.setattr("app.agent.call_tool",
                        lambda name, **kw: {"success": True, "data": _fake_chart_read_ok(**kw), "error": None})
    state.update(agent.chart_read_node(state))
    assert len(state["chunks"]) == 1
    assert state["chunks"][0]["source"] == "2023-fujita-brain-volume-aging.pdf"
