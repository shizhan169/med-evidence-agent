"""Executor 测试（框架 v2，第一阶段 1.2）：不需要 LLM、不需要 Chroma、不联网。
选工具用假 LLM（monkeypatch），工具执行用假 call_tool（monkeypatch），
只测 Executor 本身。
跑法：pytest tests/test_executor.py -v

思路：Executor 按子任务顺序执行，每个子任务：LLM 按子任务 + 工具描述
选工具 → 校验参数并执行 → 结果写回 State（子任务的 result、
evidence_ids、status）。每个工具调用都记进 tool_calls。
步骤上限 8 步，超了就停，返回部分结果；失败只重跑失败的子任务。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.agent as agent

CHUNKS = [{"text": "总白质体积在 28.7 岁达到峰值。", "source": "2022-bethlehem-brain-charts.pdf", "page": 3}]
PUBMED_RESULTS = [{"pmid": "38395541", "title": "CentileBrain", "authors": "Ge",
                   "journal": "Nature", "year": "2024", "abstract": "normative modelling"}]


def _subtask(task_id, query):
    return {"task_id": task_id, "query": query, "status": "pending",
            "evidence_ids": [], "result": "", "retry_count": 0}


def _state(subtasks):
    return {"query": "比较五篇论文", "subtasks": subtasks,
            "tool_calls": [], "chunks": [], "pubmed_results": []}


def _run(monkeypatch, subtasks, llm_raw, tool_impl):
    """假 LLM 选工具 + 假 call_tool 执行，跑一次 executor_node。"""
    monkeypatch.setattr("app.agent._call_llm", lambda prompt: llm_raw(prompt) if callable(llm_raw) else llm_raw)
    monkeypatch.setattr("app.agent.call_tool", tool_impl)
    return agent.executor_node(_state(subtasks))


def _local_search_json(query="五篇论文的数据集"):
    return f'{{"tool": "local_search", "params": {{"query": "{query}"}}}}'


def _ok_local_search(name, **kwargs):
    return {"success": True, "data": {"chunks": CHUNKS}, "error": None}


def test_executor_selects_llm_chosen_tool(monkeypatch):
    # LLM 选 local_search：Executor 真去调它（不是写死的），结果写回 State
    calls = []

    def fake_tool(name, **kwargs):
        calls.append((name, kwargs))
        return _ok_local_search(name, **kwargs)

    result = _run(monkeypatch, [_subtask("S1", "五篇论文的数据集")],
                  _local_search_json(), fake_tool)
    assert [c[0] for c in calls] == ["local_search"]
    assert calls[0][1]["query"] == "五篇论文的数据集"
    st = result["subtasks"][0]
    assert st["status"] == "completed"
    assert st["evidence_ids"] == ["2022-bethlehem-brain-charts.pdf p.3"]
    assert st["result"] == "检索到 1 个片段"
    assert result["chunks"] == CHUNKS
    assert len(result["tool_calls"]) == 1
    tc = result["tool_calls"][0]
    assert tc["tool"] == "local_search"
    assert tc["success"] is True
    assert tc["error"] is None


def test_executor_tool_choice_changes_with_subtask(monkeypatch):
    # 换一个子任务，选的工具会变——证明是 LLM 动态选的，不是 if-else 写死的
    calls = []

    def fake_llm(prompt):
        # 按子任务内容分发（不能按整个 prompt 查“最新”——prompt 模板的
        # 工具说明里本身就有“最新文献”四个字）
        subtask = prompt.split("子任务：")[-1]
        if "最新" in subtask:
            return '{"tool": "pubmed_search", "params": {"query": "longitudinal normative modeling"}}'
        return _local_search_json()

    def fake_tool(name, **kwargs):
        calls.append(name)
        if name == "pubmed_search":
            return {"success": True, "data": {"results": PUBMED_RESULTS}, "error": None}
        return _ok_local_search(name, **kwargs)

    result = _run(monkeypatch,
                  [_subtask("S1", "五篇论文的数据集"),
                   _subtask("S2", "纵向脑常模模型最新的文献有哪些")],
                  fake_llm, fake_tool)
    assert calls == ["local_search", "pubmed_search"]
    assert result["subtasks"][0]["status"] == "completed"
    assert result["subtasks"][1]["status"] == "completed"
    assert result["subtasks"][1]["evidence_ids"] == ["38395541"]
    assert result["pubmed_results"] == PUBMED_RESULTS


def test_executor_step_cap(monkeypatch):
    # 10 个子任务，步骤上限 8 步：前 8 个完成，后 2 个保持 pending，返回部分结果
    result = _run(monkeypatch, [_subtask(f"S{i}", f"子任务 {i}") for i in range(1, 11)],
                  _local_search_json(), _ok_local_search)
    done = [s for s in result["subtasks"] if s["status"] == "completed"]
    pending = [s for s in result["subtasks"] if s["status"] == "pending"]
    assert len(done) == 8
    assert len(pending) == 2
    assert len(result["tool_calls"]) == 8
    assert len(result["chunks"]) == 1  # 去重：10 次检索都是同一片段，只留 1 个


def test_executor_chart_read_placeholder(monkeypatch):
    # chart_read 是占位（available=False）：选到它时子任务记 insufficient_evidence，不重试
    def fake_tool(name, **kwargs):
        return {"success": True, "data": {"available": False, "reason": "VLM 未实现，第二阶段接入"}, "error": None}

    result = _run(monkeypatch, [_subtask("S1", "Fujita 图 4 的曲线趋势")],
                  '{"tool": "chart_read", "params": {"image_path": "", "question": "曲线趋势"}}',
                  fake_tool)
    st = result["subtasks"][0]
    assert st["status"] == "insufficient_evidence"
    assert st["retry_count"] == 0
    assert "VLM 未实现" in st["result"]
    assert len(result["tool_calls"]) == 1


def test_executor_tool_failure_retries_once_then_completes(monkeypatch):
    # 工具第一次失败、第二次成功：只重跑这个子任务，retry_count 记 1，最终完成
    calls = {"n": 0}

    def fake_tool(name, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"success": False, "data": None, "error": "Chroma 抖了一下"}
        return _ok_local_search(name, **kwargs)

    result = _run(monkeypatch, [_subtask("S1", "五篇论文的数据集")],
                  _local_search_json(), fake_tool)
    st = result["subtasks"][0]
    assert st["status"] == "completed"
    assert st["retry_count"] == 1
    assert len(result["tool_calls"]) == 2
    assert result["tool_calls"][0]["success"] is False
    assert result["tool_calls"][1]["success"] is True


def test_executor_tool_failure_twice_marks_failed(monkeypatch):
    # 工具两次都失败：子任务记 failed，不抛异常，问答不断
    def fake_tool(name, **kwargs):
        return {"success": False, "data": None, "error": "库不可用"}

    result = _run(monkeypatch, [_subtask("S1", "五篇论文的数据集")],
                  _local_search_json(), fake_tool)
    st = result["subtasks"][0]
    assert st["status"] == "failed"
    assert st["retry_count"] == 1
    assert "库不可用" in st["result"]
    assert len(result["tool_calls"]) == 2


def test_executor_bad_tool_json_defaults_to_local_search(monkeypatch):
    # LLM 选工具返回的不是 JSON：退回默认 local_search
    calls = []

    def fake_tool(name, **kwargs):
        calls.append(name)
        return _ok_local_search(name, **kwargs)

    result = _run(monkeypatch, [_subtask("S1", "五篇论文的数据集")],
                  "我不知道选哪个", fake_tool)
    assert calls == ["local_search"]
    assert result["subtasks"][0]["status"] == "completed"


def test_executor_chart_locate_not_found(monkeypatch):
    # chart_locate 没找到图：子任务记 insufficient_evidence
    def fake_tool(name, **kwargs):
        return {"success": True, "data": {"found": False}, "error": None}

    result = _run(monkeypatch, [_subtask("S1", "Fujita 图 9 在哪一页")],
                  '{"tool": "chart_locate", "params": {"paper": "Fujita", "figure_no": 9}}',
                  fake_tool)
    st = result["subtasks"][0]
    assert st["status"] == "insufficient_evidence"
    assert st["result"] == "未找到该图表"


def test_executor_no_subtasks(monkeypatch):
    # 没有子任务：不调 LLM、不调工具，直接返回空
    calls = {"llm": 0, "tool": 0}

    def fake_llm(prompt):
        calls["llm"] += 1
        return _local_search_json()

    def fake_tool(name, **kwargs):
        calls["tool"] += 1
        return _ok_local_search(name, **kwargs)

    result = _run(monkeypatch, [], fake_llm, fake_tool)
    assert result["subtasks"] == []
    assert result["tool_calls"] == []
    assert calls == {"llm": 0, "tool": 0}
