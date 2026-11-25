"""Checkpointer 测试：不需要真 LLM、不需要 Chroma、不联网。
跑法（仓库根目录，先激活环境）：
    pytest tests/test_checkpointer.py -v

思路：把调 LLM 和检索都换成假的（monkeypatch），checkpointer 指到
临时文件（tmp_path），验证：① 同一个 thread_id 的第二轮能从库里恢复
第一轮的历史（断点续聊）；② 换一个新图实例（模拟重启）后 get_state
还能读到上一轮的最终状态；③ 不传 thread_id 就是原来的无状态行为，
不建库；④ checkpointer 不可用时退回无持久化，问答照常；⑤ 传了
history 就用传来的，不用库里恢复的。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

import app.agent as agent
from app.main import app

CHUNKS = [{"text": "总白质体积在 28.7 岁达到峰值。", "source": "2022-bethlehem-brain-charts.pdf", "page": 3}]
ANSWER = "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]"


def _setup(monkeypatch, tmp_path):
    """假检索 + 假 LLM + checkpointer 指到临时文件。"""
    monkeypatch.setattr("app.agent.CHECKPOINTS_DB", str(tmp_path / "ckpt.db"))
    monkeypatch.setattr("app.rag.retrieve", lambda query, top_k=10: CHUNKS)
    monkeypatch.setattr("app.agent._call_llm", lambda prompt: ANSWER)


def test_thread_resumes_after_restart(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    # 第一轮：thread_id=t1，答对
    r1 = agent.run("总白质体积在几岁达到峰值？", thread_id="t1")
    assert r1["answer"] == ANSWER and r1["verified"] is True
    # 模拟重启：新图实例 + 新 checkpointer（同一文件），get_state 还能读到上一轮
    cm = agent._get_checkpointer_cm()
    assert cm is not None
    with cm as checkpointer:
        graph = agent._build_graph(checkpointer)
        snap = graph.get_state({"configurable": {"thread_id": "t1"}})
    assert snap.values["final"] == ANSWER
    assert snap.values["history"][0] == {"role": "user", "content": "总白质体积在几岁达到峰值？"}
    assert snap.values["history"][1] == {"role": "assistant", "content": ANSWER}
    # 第二轮：只传 query + thread_id，不传 history——上一轮要能从库里恢复，
    # 否则追问“那白质呢”拼不上上文、prompt 里也没有历史
    seen = {}

    def fake_llm2(prompt):
        seen["prompt"] = prompt
        return ANSWER

    monkeypatch.setattr("app.agent._call_llm", fake_llm2)
    r2 = agent.run("那白质呢？", thread_id="t1")
    assert r2["verified"] is True
    assert "总白质体积在几岁达到峰值" in seen["prompt"]  # 上一轮的问题恢复了
    # 恢复后 history 里有两轮（第一轮 + 第二轮）
    cm2 = agent._get_checkpointer_cm()
    with cm2 as checkpointer:
        graph2 = agent._build_graph(checkpointer)
        snap2 = graph2.get_state({"configurable": {"thread_id": "t1"}})
    assert len(snap2.values["history"]) == 4  # 两轮 = 4 条


def test_no_thread_id_is_stateless(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    r = agent.run("总白质体积在几岁达到峰值？")
    assert r["verified"] is True
    assert not (tmp_path / "ckpt.db").exists()  # 不传 thread_id 就不建库


def test_checkpointer_unavailable_falls_back(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setattr("app.agent._get_checkpointer_cm", lambda: None)
    r = agent.run("总白质体积在几岁达到峰值？", thread_id="t1")
    assert r["answer"] == ANSWER and r["verified"] is True


def test_passed_history_takes_precedence(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    agent.run("总白质体积在几岁达到峰值？", thread_id="t1")
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return ANSWER

    monkeypatch.setattr("app.agent._call_llm", fake_llm)
    # 传了 history：用传来的，不用库里恢复的
    history = [{"role": "user", "content": "总皮层灰质体积在几岁达到峰值？"},
               {"role": "assistant", "content": "总皮层灰质体积在 5.9 岁达到峰值。"}]
    r = agent.run("那白质呢？", history=history, thread_id="t1")
    assert r["verified"] is True
    assert "总皮层灰质体积" in seen["prompt"]


def test_ask_accepts_thread_id(monkeypatch, tmp_path):
    """接口也接受 thread_id：第二轮只传 thread_id 不传 history，也能接着聊。"""
    _setup(monkeypatch, tmp_path)
    client = TestClient(app)
    resp = client.post("/ask", json={"query": "总白质体积在几岁达到峰值？", "thread_id": "t1"})
    assert resp.status_code == 200
    assert resp.json()["verified"] is True
    resp2 = client.post("/ask", json={"query": "那白质呢？", "thread_id": "t1"})
    assert resp2.status_code == 200
    assert resp2.json()["verified"] is True
    # 流式接口也接受 thread_id
    resp3 = client.post("/ask/stream", json={"query": "那白质呢？", "thread_id": "t1"})
    assert resp3.status_code == 200
    assert "text/event-stream" in resp3.headers["content-type"]
