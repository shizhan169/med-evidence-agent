"""核查重试测试：不需要 LLM、不需要 Chroma、不联网。
跑法（仓库根目录，先激活环境）：
    pytest tests/test_verify_retry.py -v

思路：把调 LLM 和检索都换成假的（monkeypatch），只验证核查失败后
有没有按规则重试、重试几次、最后怎么收尾；真的问答效果要等用户本地
跑完整评测。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.agent as agent

CHUNKS = [{"text": "总白质体积在 28.7 岁达到峰值。", "source": "2022-bethlehem-brain-charts.pdf", "page": 3}]


def _fake_retrieve(query, top_k=10):
    return CHUNKS


def _run_with_llm(monkeypatch, drafts):
    """drafts 是假 LLM 每次调用返回的答案（按顺序取，用完就重复最后一个）；
    返回 (result, calls)，calls 记调用次数和每次的 prompt。"""
    calls = {"n": 0, "prompts": []}

    def fake_llm(prompt):
        calls["n"] += 1
        calls["prompts"].append(prompt)
        return drafts[min(calls["n"] - 1, len(drafts) - 1)]

    monkeypatch.setattr("app.rag.retrieve", _fake_retrieve)
    monkeypatch.setattr("app.agent._call_llm", fake_llm)
    return agent.run("总白质体积在几岁达到峰值？"), calls


def test_retry_once_then_pass(monkeypatch):
    # 第一次引用了不存在的 p.99，核查失败；重答后引用对了 p.3，核查通过
    result, calls = _run_with_llm(monkeypatch, [
        "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.99]",
        "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]",
    ])
    assert calls["n"] == 2
    assert result["verified"] is True
    assert result["answer"].startswith("总白质体积在 28.7 岁达到峰值")
    assert "p.3" in result["answer"]
    # 重答的 prompt 里有提醒，第一轮没有
    assert "不要自己编页码" in calls["prompts"][1]
    assert "不要自己编页码" not in calls["prompts"][0]


def test_retry_exhausted_refuses(monkeypatch):
    # 三次都引用错页码：1 次原答 + 2 次重试，之后拒答，不再试
    result, calls = _run_with_llm(monkeypatch, [
        "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.99]",
    ])
    assert calls["n"] == 3
    assert result["answer"] == agent.REFUSAL
    assert result["verified"] is False


def test_no_citation_retries_then_pass(monkeypatch):
    # 第一次没给引用，重答后给了正确的引用，核查通过
    result, calls = _run_with_llm(monkeypatch, [
        "总白质体积在 28.7 岁达到峰值。",
        "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]",
    ])
    assert calls["n"] == 2
    assert result["verified"] is True


def test_bare_refusal_no_retry(monkeypatch):
    # 回答员本来就拒答：不重试，只调一次 LLM
    result, calls = _run_with_llm(monkeypatch, [agent.REFUSAL])
    assert calls["n"] == 1
    assert result["answer"] == agent.REFUSAL
    assert result["verified"] is False


def test_no_chunks_no_retry(monkeypatch):
    # 没有检索片段：回答员直接拒答，LLM 一次都没调
    calls = {"n": 0}

    def fake_llm(prompt):
        calls["n"] += 1
        return "不该被调用"

    monkeypatch.setattr("app.rag.retrieve", lambda query, top_k=10: [])
    monkeypatch.setattr("app.agent._call_llm", fake_llm)
    result = agent.run("总白质体积在几岁达到峰值？")
    assert calls["n"] == 0
    assert result["answer"] == agent.REFUSAL
    assert result["verified"] is False


def test_fallback_path_retries(monkeypatch):
    """LangGraph 不可用时走顺序执行兜底，重试逻辑要和图里一致。"""
    def broken_graph():
        raise RuntimeError("LangGraph 不可用")

    monkeypatch.setattr("app.agent._build_graph", broken_graph)
    result, calls = _run_with_llm(monkeypatch, [
        "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.99]",
        "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]",
    ])
    assert calls["n"] == 2
    assert result["verified"] is True
