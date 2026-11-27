"""Planner 测试（框架 v2，第一阶段 1.2）：不需要 LLM、不需要 Chroma、不联网。
拆解用假 LLM（monkeypatch），只测 Planner 本身。
跑法：pytest tests/test_planner.py -v

思路：Planner 只负责拆解，不执行、不选工具。输入问题 + route，
输出结构化子任务列表（State.subtasks），每个子任务
{task_id, query, status, evidence_ids, result, retry_count}，
status 初始 pending。触发条件是“需要拆解”，不是“跨文献”。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.agent as agent

PLANNER_JSON = (
    '{"goal": "比较五篇论文的数据集、建模变量和结果", "subtasks": ['
    '{"query": "五篇纵向常模论文的数据集各是什么"}, '
    '{"query": "五篇纵向常模论文的建模变量有什么不同"}, '
    '{"query": "五篇纵向常模论文的结果和局限性是什么"}]}'
)


def _plan(monkeypatch, raw, query="比较五篇纵向常模论文的数据集、建模变量和结果"):
    monkeypatch.setattr("app.agent._call_llm", lambda prompt: raw)
    return agent.planner_node({"query": query})


def _check_subtask_shape(st, task_id, query):
    assert st["task_id"] == task_id
    assert st["query"] == query
    assert st["status"] == "pending"
    assert st["evidence_ids"] == []
    assert st["result"] == ""
    assert st["retry_count"] == 0


def test_planner_decomposes_comparison(monkeypatch):
    # 跨文献对比拆成多个子任务
    result = _plan(monkeypatch, PLANNER_JSON)
    subtasks = result["subtasks"]
    assert len(subtasks) == 3
    assert result["plan"]["goal"] == "比较五篇论文的数据集、建模变量和结果"
    assert result["plan"]["total"] == 3
    _check_subtask_shape(subtasks[0], "S1", "五篇纵向常模论文的数据集各是什么")
    _check_subtask_shape(subtasks[1], "S2", "五篇纵向常模论文的建模变量有什么不同")
    _check_subtask_shape(subtasks[2], "S3", "五篇纵向常模论文的结果和局限性是什么")


def test_planner_single_doc_complex_can_decompose(monkeypatch):
    # 触发条件是“需要拆解”不是“跨文献”：单篇内需要多步推理的复杂问题也能拆
    raw = (
        '{"goal": "讲清 Fujita 的建模方法", "subtasks": ['
        '{"query": "Fujita 2023 里脑体积变化和年龄的关系是怎么建模的"}, '
        '{"query": "为什么用 GAM 而不用线性模型"}]}'
    )
    result = _plan(monkeypatch, raw, query="Fujita 2023 里脑体积变化和年龄的关系是怎么建模的，为什么用 GAM？")
    assert len(result["subtasks"]) == 2
    assert result["subtasks"][0]["query"] == "Fujita 2023 里脑体积变化和年龄的关系是怎么建模的"
    assert result["subtasks"][1]["query"] == "为什么用 GAM 而不用线性模型"


def test_planner_garbage_falls_back_to_single_subtask(monkeypatch):
    # LLM 返回的不是 JSON：退回单个子任务（原问题本身），不断问答
    result = _plan(monkeypatch, "我不知道怎么拆解")
    assert len(result["subtasks"]) == 1
    _check_subtask_shape(result["subtasks"][0], "S1", "比较五篇纵向常模论文的数据集、建模变量和结果")


def test_planner_empty_subtasks_falls_back(monkeypatch):
    # LLM 返回合法 JSON 但子任务为空：同样退回单个子任务
    result = _plan(monkeypatch, '{"goal": "比较", "subtasks": []}')
    assert len(result["subtasks"]) == 1
    assert result["subtasks"][0]["query"] == "比较五篇纵向常模论文的数据集、建模变量和结果"


def test_planner_llm_failure_falls_back(monkeypatch):
    # LLM 调用抛异常：退回单个子任务
    def boom(prompt):
        raise RuntimeError("LLM 挂了")

    monkeypatch.setattr("app.agent._call_llm", boom)
    result = agent.planner_node({"query": "比较五篇纵向常模论文的数据集有什么不同"})
    assert len(result["subtasks"]) == 1
    assert result["subtasks"][0]["status"] == "pending"


def test_planner_caps_at_five(monkeypatch):
    # LLM 拆出 7 个子任务：最多只取 5 个
    subs = ", ".join(f'{{"query": "子任务 {i}"}}' for i in range(1, 8))
    result = _plan(monkeypatch, f'{{"goal": "很多", "subtasks": [{subs}]}}')
    assert len(result["subtasks"]) == 5
    assert result["subtasks"][4]["task_id"] == "S5"


def test_planner_prompt_file_loads():
    # prompt 文件能加载，含 {query} 占位符（.format 用）
    assert "拆解" in agent.PLANNER_PROMPT
    assert "{query}" in agent.PLANNER_PROMPT
    filled = agent.PLANNER_PROMPT.format(query="测试问题")
    assert "测试问题" in filled
    assert "{query}" not in filled
