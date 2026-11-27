"""Memory 压缩测试（框架 v2，第一阶段 1.4）：不需要真 LLM、不需要 Chroma、不联网。
跑法（仓库根目录，先激活环境）：
    pytest tests/test_memory.py -v

思路：历史管理逻辑在 app/memory.py，agent.py 只调它。假 LLM + 假检索，
验证：短历史不压缩；超限时摘要生成、最近 2 轮完整保留、较早轮次被摘要替代；
LLM 摘要失败退回规则压缩；摘要写进 State；checkpointer 里仍是完整历史；
多轮追问的指代仍能解析。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.agent as agent
from app import memory

CHUNKS = [{"text": "总白质体积在 28.7 岁达到峰值。", "source": "2022-bethlehem-brain-charts.pdf", "page": 3}]
ANSWER = "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]"


def _make_history(n_turns, prefix="第"):
    """造 n 轮假对话历史（每轮一问一答，问题各不相同，好核对摘要）。"""
    history = []
    for i in range(n_turns):
        history.append({"role": "user", "content": f"{prefix}{i + 1}个问题：总白质体积在几岁达到峰值？"})
        history.append({"role": "assistant", "content": f"{prefix}{i + 1}个回答：总白质体积在 28.7 岁达到峰值。"})
    return history


# ---------- 压缩触发：短历史不压，超限才压 ----------

def test_compress_short_history_unchanged():
    history = _make_history(3)
    context, summary = memory.compress_history(history)
    assert context == history  # 原样通过
    assert summary == ""


def test_compress_long_history_keeps_recent_two():
    history = _make_history(8)
    context, summary = memory.compress_history(history)
    # 只留最近 2 轮完整内容
    assert len(context) == 4  # 2 轮 = 4 条
    assert context[0]["content"].startswith("第7个问题")
    assert context[-1]["content"].startswith("第8个回答")
    # 较早轮次被摘要替代：摘要里有第 1–6 轮的问题
    assert summary
    assert "第1个问题" in summary
    assert "第6个问题" in summary
    # 最近 2 轮不在摘要里（它们在 context 里完整保留）
    assert "第7个问题" not in summary
    assert "第8个问题" not in summary


def test_compress_history_caps_chars():
    # 每轮回答都很长，总字数一定超 4000，要压缩
    history = []
    for i in range(4):
        history.append({"role": "user", "content": f"问题{i}"})
        history.append({"role": "assistant", "content": "答" * 1500})
    context, summary = memory.compress_history(history)
    assert len(context) == 4  # 最近 2 轮
    assert summary  # 较早轮次被摘要替代


# ---------- 摘要生成：LLM 优先，失败退回规则 ----------

def test_summarize_history_uses_llm():
    def fake_llm(prompt):
        return "LLM 摘要：用户问了 GMV 峰值，助手答 5.9 岁。"

    summary = memory.summarize_history(_make_history(2), llm_call=fake_llm)
    assert summary == "LLM 摘要：用户问了 GMV 峰值，助手答 5.9 岁。"


def test_summarize_history_falls_back_to_rule_on_llm_failure(monkeypatch):
    def boom(prompt):
        raise RuntimeError("LLM 挂了")

    summary = memory.summarize_history(_make_history(2), llm_call=boom)
    # 规则式：每轮保留用户问题 + 回答的第一句
    assert "用户问：第1个问题" in summary
    assert "助手答：第1个回答" in summary
    assert "用户问：第2个问题" in summary


def test_summarize_history_falls_back_when_llm_returns_empty():
    summary = memory.summarize_history(_make_history(2), llm_call=lambda prompt: "")
    assert "用户问：第1个问题" in summary


def test_summarize_history_falls_back_when_llm_returns_answer():
    # 假 LLM 对任何 prompt 都返回同一段答案文本（带引用）——那不是摘要，拒绝它
    summary = memory.summarize_history(_make_history(2), llm_call=lambda prompt: ANSWER)
    assert "用户问：第1个问题" in summary
    assert "[来源：" not in summary


def test_summarize_history_empty_returns_empty():
    assert memory.summarize_history([]) == ""
    assert memory.summarize_history([], llm_call=lambda prompt: "x") == ""


def test_summarize_history_without_llm_uses_rule():
    # 不传 llm_call：纯规则式压缩
    summary = memory.summarize_history(_make_history(2))
    assert "用户问：第1个问题" in summary
    assert "助手答：第1个回答" in summary


# ---------- 摘要写进 State：追加轮次时被截掉的轮次先摘要 ----------

def test_append_turn_summarizes_dropped_turns(monkeypatch):
    # 6 轮历史 + 1 个新问题：追加后超限，第 1 轮被截掉，先摘要再丢
    monkeypatch.setattr("app.agent._call_llm", lambda prompt: "LLM 摘要：第1轮问了 GMV 峰值。")
    state = {"query": "第7个问题：总白质体积在几岁达到峰值？",
             "history": _make_history(6), "history_summary": ""}
    out = agent._append_turn_to_history(state, "第7个回答：总白质体积在 28.7 岁达到峰值。")
    assert len(out["history"]) == 12  # 仍是 6 轮（新轮次进来，最早一轮出去）
    assert out["history"][0]["content"].startswith("第2个问题")  # 第 1 轮被截掉了
    assert out["history_summary"] == "LLM 摘要：第1轮问了 GMV 峰值。"  # 但摘要留下了


def test_append_turn_falls_back_to_rule_when_llm_fails(monkeypatch):
    def boom(prompt):
        raise RuntimeError("LLM 挂了")

    monkeypatch.setattr("app.agent._call_llm", boom)
    state = {"query": "第7个问题：总白质体积在几岁达到峰值？",
             "history": _make_history(6), "history_summary": ""}
    out = agent._append_turn_to_history(state, "第7个回答。")
    assert "第1个问题" in out["history_summary"]  # 规则式摘要兜底
    assert len(out["history"]) == 12


def test_append_turn_short_history_no_summary_call(monkeypatch):
    # 短历史：不截断、不调 LLM（calls 记数验证）
    calls = {"n": 0}

    def counting_llm(prompt):
        calls["n"] += 1
        return "不该被调到"

    monkeypatch.setattr("app.agent._call_llm", counting_llm)
    state = {"query": "第2个问题", "history": _make_history(1), "history_summary": ""}
    out = agent._append_turn_to_history(state, "第2个回答。")
    assert calls["n"] == 0  # 没截掉轮次，就不调 LLM 摘要
    assert out["history_summary"] == ""
    assert len(out["history"]) == 4


# ---------- prompt 历史：有摘要时只拼最近 2 轮 + 摘要 ----------

def test_prompt_history_short_unchanged():
    history = _make_history(3)
    assert memory.prompt_history(history, "") == history


def test_prompt_history_with_summary_keeps_recent_two():
    history = _make_history(6)
    out = memory.prompt_history(history, "持久化摘要：第1–2轮问了 GMV。")
    # 第一条是摘要入口，其余是最近 2 轮完整内容
    assert out[0]["content"].startswith("[此前对话摘要]")
    assert "持久化摘要：第1–2轮问了 GMV。" in out[0]["content"]
    assert len(out) == 5  # 摘要入口 + 2 轮（4 条）
    assert out[1]["content"].startswith("第5个问题")
    assert out[-1]["content"].startswith("第6个回答")
    # 中间轮次（第 3–4 轮）的内容在摘要里（规则式补进），不在完整内容里
    assert "第3个问题" in out[0]["content"]
    assert "第4个问题" in out[0]["content"]
    assert not any("第3个问题" in m["content"] for m in out[1:])
    assert not any("第4个问题" in m["content"] for m in out[1:])


def test_prompt_history_empty():
    assert memory.prompt_history([], "") == []
    assert memory.prompt_history([], "摘要") == []


# ---------- 端到端：压缩不破坏多轮问答 ----------

def _setup_fake(monkeypatch, tmp_path=None):
    if tmp_path is not None:
        monkeypatch.setattr("app.agent.CHECKPOINTS_DB", str(tmp_path / "ckpt.db"))
    monkeypatch.setattr("app.rag.retrieve", lambda query, top_k=10, llm_call=None: CHUNKS)
    monkeypatch.setattr("app.agent._call_llm", lambda prompt: ANSWER)


def test_followup_resolves_after_compression(monkeypatch):
    """长历史 + 追问：指代仍能解析（最近轮次的上下文还在）。"""
    _setup_fake(monkeypatch)
    seen = {"prompts": []}

    def fake_retrieve(query, top_k=10, llm_call=None):
        seen["retrieval_query"] = query
        return CHUNKS

    def fake_llm(prompt):
        # 假 LLM 会被改写、回答、摘要三次调到，全部记下来；
        # 回答 prompt 是唯一带检索片段引用（[来源：...]）的那一个
        seen["prompts"].append(prompt)
        return ANSWER

    monkeypatch.setattr("app.rag.retrieve", fake_retrieve)
    monkeypatch.setattr("app.agent._call_llm", fake_llm)

    # 8 轮历史，最近一轮问的是 GMV 峰值；追问“那白质呢？”要指代它
    history = _make_history(7)
    history.append({"role": "user", "content": "总皮层灰质体积（GMV）在几岁达到峰值？"})
    history.append({"role": "assistant", "content": "总皮层灰质体积在 5.9 岁达到峰值。"})
    result = agent.run("那白质呢？", history=history)
    assert "总皮层灰质体积" in seen["retrieval_query"]  # 检索问题拼上了上文
    assert "那白质呢" in seen["retrieval_query"]
    answer_prompts = [p for p in seen["prompts"] if "[来源：2022-bethlehem-brain-charts.pdf p.3]" in p]
    assert len(answer_prompts) == 1  # 回答 prompt 只有一次
    assert "总皮层灰质体积" in answer_prompts[0]  # prompt 里有最近轮次的上下文
    assert result["verified"] is True


def test_checkpointer_keeps_full_history_and_summary(monkeypatch, tmp_path):
    """断点续聊：checkpointer 里仍是完整历史（压缩不影响持久化），
    摘要也一起存下来了。"""
    _setup_fake(monkeypatch, tmp_path)
    # 连问 8 轮，同一 thread_id
    for i in range(8):
        r = agent.run(f"第{i + 1}个问题：总白质体积在几岁达到峰值？", thread_id="t1")
        assert r["verified"] is True
    # 从 checkpointer 恢复：history 仍是最近 6 轮完整内容
    cm = agent._get_checkpointer_cm()
    with cm as checkpointer:
        graph = agent._build_graph(checkpointer)
        snap = graph.get_state({"configurable": {"thread_id": "t1"}})
    assert len(snap.values["history"]) == 12  # 6 轮 = 12 条
    assert snap.values["history"][0]["content"].startswith("第3个问题")  # 最早两轮被截掉了
    # 但摘要留下了：被截掉的第 1–2 轮在 history_summary 里
    summary = snap.values.get("history_summary") or ""
    assert summary
    assert "第1个问题" in summary
    assert "第2个问题" in summary
    # 第 9 轮追问：上一轮恢复 + 摘要都在，问答照常
    r9 = agent.run("那白质呢？", thread_id="t1")
    assert r9["verified"] is True
