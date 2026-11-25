"""LangGraph 三角色：检索员 -> 回答员 -> 核查员。轻量版多智能体。"""
import re
from typing import TypedDict, List

from .config import LLM_BASE_URL, LLM_MODEL, get_api_key

REFUSAL = "根据现有文献无法回答。"

# 匹配所有 “论文名.pdf p.页码” 形式的引用，包括用分号并在一个括号里的多个引用；
# “来源：” 前缀不计入文件名，否则和检索片段的 source 对不上
CITE_RE = re.compile(r"(?:\[?来源：)?([^\s\[\]；,：]+\.pdf)\s+p\.(\d+)")


class State(TypedDict):
    query: str
    chunks: List[dict]
    draft: str
    final: str
    verified: bool


def retriever_node(state: State):
    from .rag import retrieve

    chunks = retrieve(state["query"], top_k=10)
    return {"chunks": chunks}


def _call_llm(prompt: str) -> str:
    """有 key 才调 Qwen/DeepSeek（OpenAI 兼容接口），无 key 返回空，由调用方走占位分支。"""
    api_key = get_api_key()
    if not api_key:
        return ""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


def answerer_node(state: State):
    from .rag import build_prompt

    if not state["chunks"]:
        return {"draft": REFUSAL}
    prompt = build_prompt(state["query"], state["chunks"])
    try:
        draft = _call_llm(prompt)
    except Exception as e:
        print(f"[answerer] LLM 调用失败：{e}")
        draft = ""
    if not draft:
        # 无 key 或调用失败：不编造答案，明确告诉用户只生成了 prompt
        return {"draft": f"[待接 LLM] 已生成 Prompt（{len(prompt)} 字符），请在 .env 填入 QWEN_API_KEY 或 DEEPSEEK_API_KEY 后重试。"}
    return {"draft": draft}


def verifier_node(state: State):
    # 核查规则：draft 里的每个 [来源：... p.X] 必须对应一个真实检索片段，否则拒答
    draft = state.get("draft", "")
    if not state.get("chunks") or draft == REFUSAL or draft.startswith("[待接 LLM]"):
        final = REFUSAL if not state.get("chunks") else draft
        return {"final": final, "verified": False}
    cited = set((m.group(1), m.group(2)) for m in CITE_RE.finditer(draft))
    real = set(
        (str(c.get("source", "未知")), str(c.get("page", "?"))) for c in state["chunks"]
    )
    if not cited:
        return {"final": REFUSAL, "verified": False}
    if not cited.issubset(real):
        bad = sorted(cited - real)
        print(f"[verifier] 引用不在检索结果中：{bad}")
        return {"final": REFUSAL, "verified": False}
    return {"final": draft, "verified": True}


def _build_graph():
    from langgraph.graph import StateGraph, END

    graph = StateGraph(State)
    graph.add_node("retriever", retriever_node)
    graph.add_node("answerer", answerer_node)
    graph.add_node("verifier", verifier_node)
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "answerer")
    graph.add_edge("answerer", "verifier")
    graph.add_edge("verifier", END)
    return graph.compile()


def _dedupe_citations(chunks: List[dict]) -> List[dict]:
    """同一论文同一页只显示一次：多个片段可能来自同一页，引用列表按 (source, page) 去重。"""
    seen = set()
    out = []
    for c in chunks:
        key = (str(c.get("source", "未知")), str(c.get("page", "?")))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def run(query: str) -> dict:
    try:
        result = _build_graph().invoke(
            {"query": query, "chunks": [], "draft": "", "final": "", "verified": False}
        )
    except Exception as e:
        # LangGraph 不可用时退回顺序执行，流程本身不变
        print(f"[run] LangGraph 不可用，顺序执行：{e}")
        state: State = {"query": query, "chunks": [], "draft": "", "final": "", "verified": False}
        state.update(retriever_node(state))
        state.update(answerer_node(state))
        state.update(verifier_node(state))
        result = state
    return {
        "answer": result.get("final", REFUSAL),
        "verified": result.get("verified", False),
        "citations": _dedupe_citations(result.get("chunks", [])),
    }
