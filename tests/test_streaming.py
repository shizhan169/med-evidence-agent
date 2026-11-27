"""流式问答测试：不需要真 LLM、不需要 Chroma、不联网。
跑法（仓库根目录，先激活环境）：
    pytest tests/test_streaming.py -v

思路：把调 LLM 和检索都换成假的（monkeypatch），用 FastAPI TestClient
调 POST /ask/stream，验证 SSE 事件能拼回完整答案、done 事件带核查结果
和引用；再验证 /ask 的返回格式没变。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

import app.agent as agent
from app.main import app, _split_stream_chunks

CHUNKS = [{"text": "总白质体积在 28.7 岁达到峰值。", "source": "2022-bethlehem-brain-charts.pdf", "page": 3}]
ANSWER = "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]"


def _client_with_fake_llm(monkeypatch, answer):
    monkeypatch.setattr("app.rag.retrieve", lambda query, top_k=10, llm_call=None: CHUNKS)
    monkeypatch.setattr("app.agent._call_llm", lambda prompt: answer)
    return TestClient(app)


def _read_sse(resp):
    """把 SSE 响应读成事件列表（每行 `data: {json}`）。"""
    events = []
    for line in resp.iter_lines():
        if line.startswith("data:"):
            events.append(json.loads(line[5:].strip()))
    return events


def test_stream_reconstructs_answer(monkeypatch):
    client = _client_with_fake_llm(monkeypatch, ANSWER)
    resp = client.post("/ask/stream", json={"query": "总白质体积在几岁达到峰值？"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    events = _read_sse(resp)
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert tokens, "至少要有一个 token 事件"
    assert "".join(tokens) == ANSWER  # 拼回去和原答案一模一样
    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1
    assert done[0]["verified"] is True
    assert done[0]["citations"][0]["source"] == "2022-bethlehem-brain-charts.pdf"
    assert done[0]["citations"][0]["page"] == 3


def test_stream_refusal(monkeypatch):
    """拒答也一样流式输出，done 事件里 verified 是 False。"""
    client = _client_with_fake_llm(monkeypatch, agent.REFUSAL)
    resp = client.post("/ask/stream", json={"query": "总白质体积在几岁达到峰值？"})
    assert resp.status_code == 200
    events = _read_sse(resp)
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert "".join(tokens) == agent.REFUSAL
    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1
    assert done[0]["verified"] is False


def test_stream_error_event(monkeypatch):
    """run 抛异常时，流里只有一个 error 事件、没有 token。"""
    monkeypatch.setattr("app.main.run", lambda query, history=None, **kw: (_ for _ in ()).throw(RuntimeError("检索挂了")))
    client = TestClient(app)
    resp = client.post("/ask/stream", json={"query": "总白质体积在几岁达到峰值？"})
    assert resp.status_code == 200
    events = _read_sse(resp)
    assert [e["type"] for e in events] == ["error"]
    assert "检索挂了" in events[0]["message"]


def test_split_chunks_lossless():
    text = "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]，峰值后下降。"
    assert "".join(_split_stream_chunks(text)) == text  # 无损
    assert _split_stream_chunks("") == []
    # 每块都不超过 2*max_chars（硬切上限）
    assert all(len(c) <= 48 for c in _split_stream_chunks(text))
    # 纯英文长词也不断词到一半就丢字
    en = "generalizedadditivemodel " * 5
    assert "".join(_split_stream_chunks(en)) == en


def test_ask_unchanged(monkeypatch):
    """POST /ask 的返回格式：{answer, verified, citations} 没变，
    另新增 answer_status 字段（框架 v2 第一阶段 1.1：verified 改名后的
    四种最终状态；verified 是从它派生的兼容字段）。"""
    client = _client_with_fake_llm(monkeypatch, ANSWER)
    resp = client.post("/ask", json={"query": "总白质体积在几岁达到峰值？"})
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"answer", "verified", "answer_status", "citations"}
    assert data["answer"] == ANSWER
    assert data["verified"] is True
    assert data["answer_status"] == "verified"
    assert data["citations"][0]["source"] == "2022-bethlehem-brain-charts.pdf"
