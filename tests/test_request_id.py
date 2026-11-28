"""3.2 request_id 追踪 + 结构化异常：不需要真 LLM、不需要 Chroma、不联网。
跑法（仓库根目录，先激活环境）：
    pytest tests/test_request_id.py -v

思路：把调 LLM 和检索都换成假的（monkeypatch），用 FastAPI TestClient
调 /ask、/ask/stream、/health，验证每个响应都带 X-Request-ID、
流事件里带同一个 ID、run 抛异常时返回结构化错误而不是笼统 500、
校验错误（422）格式保持 FastAPI 默认不变。
"""
import json
import logging
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

import app.agent as agent
from app.main import REQUEST_ID_HEADER, app

CHUNKS = [{"text": "总白质体积在 28.7 岁达到峰值。", "source": "2022-bethlehem-brain-charts.pdf", "page": 3}]
ANSWER = "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]"


def _client_with_fake_llm(monkeypatch, answer=ANSWER):
    monkeypatch.setattr("app.rag.retrieve", lambda query, top_k=10, llm_call=None: CHUNKS)
    monkeypatch.setattr("app.agent._call_llm", lambda prompt: answer)
    return TestClient(app)


def _read_sse(resp):
    events = []
    for line in resp.iter_lines():
        if line.startswith("data:"):
            events.append(json.loads(line[5:].strip()))
    return events


def test_ask_returns_request_id_header(monkeypatch):
    client = _client_with_fake_llm(monkeypatch)
    resp = client.post("/ask", json={"query": "总白质体积在几岁达到峰值？"})
    assert resp.status_code == 200
    rid = resp.headers.get(REQUEST_ID_HEADER)
    assert rid, "响应头里要有 X-Request-ID"
    uuid.UUID(rid)  # 是合法 UUID，不是随便拼的字符串


def test_request_id_passthrough(monkeypatch):
    """客户端传了 X-Request-ID 就沿用，方便跨服务追踪。"""
    client = _client_with_fake_llm(monkeypatch)
    sent = str(uuid.uuid4())
    resp = client.post("/ask", json={"query": "总白质体积在几岁达到峰值？"},
                       headers={REQUEST_ID_HEADER: sent})
    assert resp.status_code == 200
    assert resp.headers.get(REQUEST_ID_HEADER) == sent


def test_health_has_request_id(monkeypatch):
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.headers.get(REQUEST_ID_HEADER)


def test_stream_done_event_carries_request_id(monkeypatch):
    client = _client_with_fake_llm(monkeypatch)
    resp = client.post("/ask/stream", json={"query": "总白质体积在几岁达到峰值？"})
    assert resp.status_code == 200
    events = _read_sse(resp)
    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1
    assert done[0]["request_id"] == resp.headers.get(REQUEST_ID_HEADER)


def test_stream_error_event_carries_request_id(monkeypatch):
    monkeypatch.setattr("app.main.run", lambda query, history=None, **kw: (_ for _ in ()).throw(RuntimeError("检索挂了")))
    client = TestClient(app)
    resp = client.post("/ask/stream", json={"query": "总白质体积在几岁达到峰值？"})
    assert resp.status_code == 200
    events = _read_sse(resp)
    assert [e["type"] for e in events] == ["error"]
    assert "检索挂了" in events[0]["message"]
    assert events[0]["request_id"] == resp.headers.get(REQUEST_ID_HEADER)


def test_ask_structured_error(monkeypatch):
    """run 抛异常时 /ask 返回结构化错误，不是笼统的 500。
    注意：注册给 Exception 的 handler 在 Starlette 里是 ServerErrorMiddleware
    的 500 处理器，它发完响应还会继续把异常抛出来，TestClient 默认
    raise_server_exceptions=True 会把它再抛回测试里，所以这里要关掉；
    真实部署（uvicorn）下客户端收到的是结构化 500 响应。"""
    monkeypatch.setattr("app.main.run", lambda query, history=None, **kw: (_ for _ in ()).throw(RuntimeError("检索挂了")))
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/ask", json={"query": "总白质体积在几岁达到峰值？"})
    assert resp.status_code == 500
    data = resp.json()
    assert set(data.keys()) == {"error"}
    err = data["error"]
    assert err["code"] == "internal_error"
    assert "检索挂了" in err["message"]
    assert err["request_id"] == resp.headers.get(REQUEST_ID_HEADER)


def test_validation_error_unchanged(monkeypatch):
    """校验错误（缺 query 字段）保持 FastAPI 默认的 422 + detail 格式。"""
    client = TestClient(app)
    resp = client.post("/ask", json={})
    assert resp.status_code == 422
    assert "detail" in resp.json()


def test_logs_carry_same_request_id(monkeypatch, caplog):
    """一次问答的全链路日志都带同一个 request_id。"""
    client = _client_with_fake_llm(monkeypatch)
    with caplog.at_level(logging.INFO, logger="app.main"):
        resp = client.post("/ask", json={"query": "总白质体积在几岁达到峰值？"})
    assert resp.status_code == 200
    rid = resp.headers.get(REQUEST_ID_HEADER)
    lines = [r.getMessage() for r in caplog.records if rid in r.getMessage()]
    assert len(lines) >= 3, "开始、agent 调用、结束三处日志都要带这个 ID"
    assert any("agent run start" in line for line in lines)
    assert any("agent run done" in line for line in lines)
