"""LangSmith 追踪测试：不需要真 key、不需要真 LangSmith 账号、不联网
（trace 发不出去时客户端只记 warning，问答照常跑）。
跑法（仓库根目录，先激活环境）：
    pytest tests/test_tracing.py -v

思路：① 追踪默认关闭；② 开了追踪（LANGSMITH_TRACING=true + 假 key）
时 run() 照常跑、不崩；③ _call_llm 在追踪开着时包一层 wrap_openai、
关着时不包。真的 trace 要等用户本地填 key 后在 LangSmith 后台看。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.agent as agent
from app.config import tracing_enabled

CHUNKS = [{"text": "总白质体积在 28.7 岁达到峰值。", "source": "2022-bethlehem-brain-charts.pdf", "page": 3}]
ANSWER = "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]"


def test_tracing_disabled_by_default():
    # 没设环境变量时默认关闭（config 读不到 LANGSMITH_TRACING）
    assert tracing_enabled() is False


def test_run_works_with_tracing_vars_set(monkeypatch):
    """开了追踪、key 是假的：run() 照常跑、不崩（trace 发不出去只记 warning）。"""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("LANGSMITH_PROJECT", "test")
    # config 的模块级变量是 import 时读的，测试里直接改它
    monkeypatch.setattr("app.config.LANGSMITH_TRACING", "true")
    monkeypatch.setattr("app.rag.retrieve", lambda query, top_k=10, llm_call=None: CHUNKS)
    monkeypatch.setattr("app.agent._call_llm", lambda prompt: ANSWER)
    result = agent.run("总白质体积在几岁达到峰值？")
    assert result["answer"] == ANSWER
    assert result["verified"] is True


# ---- 下面测 _call_llm 包不包 wrap_openai：用假 OpenAI 客户端 ----

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, client):
        self.client = client

    def create(self, **kwargs):
        self.client.calls.append(kwargs)
        return _FakeResp("假答案")


class _FakeChat:
    def __init__(self, client):
        self.completions = _FakeCompletions(client)


class _FakeOpenAI:
    # 签名跟真 OpenAI 客户端对齐：_call_llm 会传 timeout（Harness 3.4），
    # fake 不收就会 TypeError。
    def __init__(self, api_key=None, base_url=None, timeout=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.calls = []
        self.chat = _FakeChat(self)


def _setup_fake_openai(monkeypatch, tracing_on):
    """把 OpenAI 客户端和 wrap_openai 都换成假的，返回记录 wrap 调用次数的 dict。"""
    monkeypatch.setattr("app.config.LANGSMITH_TRACING", "true" if tracing_on else "")
    monkeypatch.setattr("app.agent.get_api_key", lambda: "fake-key")
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    wrapped = {"n": 0}

    def fake_wrap(client):
        wrapped["n"] += 1
        return client

    monkeypatch.setattr("langsmith.wrappers.wrap_openai", fake_wrap)
    return wrapped


def test_call_llm_wraps_client_when_tracing_on(monkeypatch):
    wrapped = _setup_fake_openai(monkeypatch, tracing_on=True)
    assert agent._call_llm("你好") == "假答案"
    assert wrapped["n"] == 1  # 追踪开着，包了一层


def test_call_llm_no_wrap_when_tracing_off(monkeypatch):
    wrapped = _setup_fake_openai(monkeypatch, tracing_on=False)
    assert agent._call_llm("你好") == "假答案"
    assert wrapped["n"] == 0  # 追踪关着，不包，原样调用


def test_call_llm_no_key_returns_empty(monkeypatch):
    """没 key 时直接返回空（走占位分支），不调 OpenAI、也不包 wrap。"""
    monkeypatch.setattr("app.config.LANGSMITH_TRACING", "true")
    monkeypatch.setattr("app.agent.get_api_key", lambda: "")
    wrapped = {"n": 0}
    monkeypatch.setattr("langsmith.wrappers.wrap_openai", lambda c: (wrapped.__setitem__("n", 1), c)[1])
    assert agent._call_llm("你好") == ""
    assert wrapped["n"] == 0
