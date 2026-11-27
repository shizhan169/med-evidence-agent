"""多轮对话测试：不需要 LLM、不需要 Chroma、不联网。
跑法（仓库根目录，先激活环境）：
    pytest tests/test_multiturn.py -v

思路：把调 LLM 和检索都换成假的（monkeypatch），只验证历史有没有
正确透传、截断和拼进检索问题；真的问答效果要等用户本地跑完整评测。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.agent as agent
from app.rag import build_prompt


def _make_history(n_turns):
    """造 n 轮假对话历史（每轮一问一答）。"""
    history = []
    for i in range(n_turns):
        history.append({"role": "user", "content": f"第{i + 1}个问题"})
        history.append({"role": "assistant", "content": f"第{i + 1}个回答"})
    return history


def test_normalize_history_keeps_last_6_turns():
    history = agent._normalize_history(_make_history(10))
    assert len(history) == 12  # 6 轮 = 12 条
    assert history[0]["content"] == "第5个问题"  # 只留最后 6 轮
    assert history[-1]["content"] == "第10个回答"


def test_normalize_history_drops_bad_messages():
    raw = [
        {"role": "user", "content": "好问题"},
        {"role": "system", "content": "系统提示不该进历史"},
        {"role": "assistant", "content": "  "},  # 空内容丢掉
        {"role": "assistant"},  # 没 content 丢掉
        "不是 dict 也丢掉",
        {"role": "assistant", "content": "好回答"},
    ]
    history = agent._normalize_history(raw)
    assert history == [
        {"role": "user", "content": "好问题"},
        {"role": "assistant", "content": "好回答"},
    ]


def test_normalize_history_caps_total_chars():
    # 每轮回答都很长，总字数一定超 4000，要从最早的一轮开始丢
    history = []
    for i in range(4):
        history.append({"role": "user", "content": f"问题{i}"})
        history.append({"role": "assistant", "content": "答" * 1500})
    kept = agent._normalize_history(history)
    assert sum(len(m["content"]) for m in kept) <= agent.MAX_HISTORY_CHARS
    assert len(kept) % 2 == 0  # 丢也是整轮丢，不留半轮


def test_retrieval_query_rewrites_followup():
    history = [
        {"role": "user", "content": "总皮层灰质体积（GMV）在几岁达到峰值？"},
        {"role": "assistant", "content": "总皮层灰质体积在 5.9 岁达到峰值。"},
    ]
    q = agent._build_retrieval_query("那白质呢？", history)
    assert "总皮层灰质体积" in q and "那白质呢" in q


def test_retrieval_query_leaves_standalone_alone():
    history = [
        {"role": "user", "content": "总皮层灰质体积（GMV）在几岁达到峰值？"},
        {"role": "assistant", "content": "总皮层灰质体积在 5.9 岁达到峰值。"},
    ]
    # 没有指代词的完整问题，原样返回，不加东西
    assert agent._build_retrieval_query("总白质体积在几岁达到峰值？", history) == "总白质体积在几岁达到峰值？"
    # 没历史时也不改
    assert agent._build_retrieval_query("那白质呢？", []) == "那白质呢？"


def test_prompt_includes_history():
    chunks = [{"text": "总白质体积在 28.7 岁达到峰值。", "source": "2022-bethlehem-brain-charts.pdf", "page": 3}]
    history = [
        {"role": "user", "content": "总皮层灰质体积（GMV）在几岁达到峰值？"},
        {"role": "assistant", "content": "总皮层灰质体积在 5.9 岁达到峰值。"},
    ]
    prompt = build_prompt("那白质呢？", chunks, history=history)
    assert "总皮层灰质体积" in prompt  # 历史进了 prompt
    assert "不要引用历史里的页码" in prompt  # 引用只能来自本次片段
    assert "总白质体积在 28.7 岁达到峰值" in prompt
    # 不传历史时，prompt 里没有对话历史这一段
    assert "对话历史" not in build_prompt("那白质呢？", chunks)


def test_run_passes_history_and_rewrites_retrieval(monkeypatch):
    """端到端（假 LLM + 假检索）：追问要带着上文去检索，prompt 里要有历史。"""
    seen = {}

    def fake_retrieve(query, top_k=10, llm_call=None):
        seen["retrieval_query"] = query
        return [{"text": "总白质体积在 28.7 岁达到峰值。",
                 "source": "2022-bethlehem-brain-charts.pdf", "page": 3}]

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]"

    monkeypatch.setattr("app.rag.retrieve", fake_retrieve)
    monkeypatch.setattr("app.agent._call_llm", fake_llm)

    history = [
        {"role": "user", "content": "总皮层灰质体积（GMV）在几岁达到峰值？"},
        {"role": "assistant", "content": "总皮层灰质体积在 5.9 岁达到峰值。"},
    ]
    result = agent.run("那白质呢？", history=history)

    assert "总皮层灰质体积" in seen["retrieval_query"]  # 检索问题拼上了上文
    assert "那白质呢" in seen["retrieval_query"]
    assert "总皮层灰质体积" in seen["prompt"]  # prompt 里有历史
    assert result["answer"].startswith("总白质体积在 28.7 岁达到峰值")
    assert result["verified"] is True  # 引用在本次检索片段里，核查通过


def test_run_without_history_unchanged(monkeypatch):
    """不传历史时，和原来单轮行为一致：检索用原问题，prompt 里没历史。"""
    seen = {}

    def fake_retrieve(query, top_k=10, llm_call=None):
        seen["retrieval_query"] = query
        return [{"text": "总白质体积在 28.7 岁达到峰值。",
                 "source": "2022-bethlehem-brain-charts.pdf", "page": 3}]

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]"

    monkeypatch.setattr("app.rag.retrieve", fake_retrieve)
    monkeypatch.setattr("app.agent._call_llm", fake_llm)

    result = agent.run("总白质体积在几岁达到峰值？")
    assert seen["retrieval_query"] == "总白质体积在几岁达到峰值？"
    assert "对话历史" not in seen["prompt"]
    assert result["verified"] is True
