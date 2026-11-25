"""LangGraph 三角色：检索员 -> 回答员 -> 核查员。轻量版多智能体。"""
import os
import re
from typing import TypedDict, List

from .config import CHECKPOINTS_DB, LLM_BASE_URL, LLM_MODEL, get_api_key, tracing_enabled
from .prompts import load_prompt

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
    # 多轮对话：history 是最近几轮 [{"role": "user"/"assistant", "content": ...}]，
    # retrieval_query 是真正拿去检索的问题（追问会被拼上上文，见 _build_retrieval_query）
    history: List[dict]
    retrieval_query: str
    # 核查重试：retries 是已重试次数，retry 是核查员要求再答一次的标志
    retries: int
    retry: bool


# 多轮对话：历史保留最近 6 轮（一轮 = 一问一答），总字数不超过 4000 字，
# 太长就从最早的一轮开始丢；prompt 太长既费 token、模型也容易抓不住重点。
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 4000

# 追问标志词：问题里有这些词、又有历史时，说明它在指代上文（比如“那白质呢？”），
# 检索时要带上上一轮的问题，否则光凭“那白质呢”几个字检索不到白质体积的片段。
FOLLOWUP_MARKERS = ("那", "它", "这个", "那个", "呢")

# 核查重试：核查失败（引用不在检索结果中、或没给引用）后，回答员最多重答
# MAX_RETRIES 次（总共 1 + MAX_RETRIES 次机会），再不过就拒答；只重试回答，
# 不重新检索——一次只改一个变量。
MAX_RETRIES = 2

# 重答时的提醒：只准引用本次提供的片段里的页码。
# 文本在 prompts/retry/v1.txt，启动时读进来；改提醒就改那个文件。
RETRY_REMINDER = load_prompt("retry", "v1")


def _normalize_history(history) -> List[dict]:
    """把调用方传来的 history 洗成干净的 [{"role", "content"}] 列表。
    只要 user/assistant 两种角色，空内容丢掉；只保留最近 MAX_HISTORY_TURNS 轮；
    再按总字数截断，超了就从最早的一轮开始丢。返回的一定是整轮。"""
    if not history:
        return []
    cleaned = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        cleaned.append({"role": role, "content": content})
    # 只留最近 N 轮；开头如果是 assistant（上一轮的回答没配上问题），丢掉它，保证整轮
    keep = cleaned[-MAX_HISTORY_TURNS * 2:]
    if keep and keep[0]["role"] == "assistant":
        keep = keep[1:]
    # 总字数超了就从最早的一轮开始丢
    while len(keep) > 2 and sum(len(m["content"]) for m in keep) > MAX_HISTORY_CHARS:
        keep = keep[2:]
    return keep


def _build_retrieval_query(query: str, history: List[dict]) -> str:
    """给检索用的问题。追问（带指代词）就把上一轮用户的问题拼在前面，
    让检索能看到完整意思；普通问题原样返回，不加东西。"""
    if history and any(m in query for m in FOLLOWUP_MARKERS):
        for msg in reversed(history):
            if msg["role"] == "user":
                return f"{msg['content']} {query}"
    return query


def retriever_node(state: State):
    from .rag import retrieve

    # 追问已经由 _build_retrieval_query 拼好了完整意思，检索用它；
    # 没历史时 retrieval_query 就是原问题
    chunks = retrieve(state.get("retrieval_query") or state["query"], top_k=10)
    return {"chunks": chunks}


def _call_llm(prompt: str) -> str:
    """有 key 才调 Qwen/DeepSeek（OpenAI 兼容接口），无 key 返回空，由调用方走占位分支。"""
    api_key = get_api_key()
    if not api_key:
        return ""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL)
    if tracing_enabled():
        # LangSmith 追踪开着时，把这次 LLM 调用也记进 trace（能看到 prompt、
        # 返回、耗时）；langsmith 没装或包不上就用原 client，问答不受影响。
        # 图本身的执行过程（检索→回答→核查）由 LangChain 的追踪自动记，
        # 不用这里操心——只要 LANGSMITH_TRACING=true 且装了 langsmith。
        try:
            from langsmith.wrappers import wrap_openai
            client = wrap_openai(client)
        except Exception:
            pass
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
    prompt = build_prompt(state["query"], state["chunks"], history=state.get("history") or [])
    if state.get("retries", 0) > 0:
        # 重答：提醒模型只引用本次提供的片段，上一轮就是引用错了才回来的
        prompt = prompt + RETRY_REMINDER
    try:
        draft = _call_llm(prompt)
    except Exception as e:
        print(f"[answerer] LLM 调用失败：{e}")
        draft = ""
    if not draft:
        # 无 key 或调用失败：不编造答案，明确告诉用户只生成了 prompt
        return {"draft": f"[待接 LLM] 已生成 Prompt（{len(prompt)} 字符），请在 .env 填入 QWEN_API_KEY 或 DEEPSEEK_API_KEY 后重试。"}
    return {"draft": draft}


def _append_turn_to_history(state: State, final: str) -> List[dict]:
    """把本轮（一问一答）追加到历史里，供 checkpointer 持久化。
    只在核查员给出最终答案时调（重试时不调，那轮还没定论）。
    历史照样只留最近 6 轮、4000 字（_normalize_history 的上限）。"""
    history = list(state.get("history") or [])
    history.append({"role": "user", "content": state.get("query", "")})
    history.append({"role": "assistant", "content": final})
    return _normalize_history(history)


def verifier_node(state: State):
    # 核查规则：draft 里的每个 [来源：... p.X] 必须对应一个真实检索片段，否则拒答
    draft = state.get("draft", "")
    if not state.get("chunks") or draft == REFUSAL or draft.startswith("[待接 LLM]"):
        # 没片段、回答员本来就拒答、LLM 没调通：重试也没用，直接结束
        final = REFUSAL if not state.get("chunks") else draft
        return {"final": final, "verified": False, "retry": False,
                "history": _append_turn_to_history(state, final)}
    cited = set((m.group(1), m.group(2)) for m in CITE_RE.finditer(draft))
    real = set(
        (str(c.get("source", "未知")), str(c.get("page", "?"))) for c in state["chunks"]
    )
    if not cited or not cited.issubset(real):
        # 引用错了（编了页码、或没给引用）：还有重试次数就回回答员重答，
        # 用完就拒答——绝不编造
        if not cited:
            print("[verifier] 回答没有给出引用")
        else:
            bad = sorted(cited - real)
            print(f"[verifier] 引用不在检索结果中：{bad}")
        retries = state.get("retries", 0)
        if retries < MAX_RETRIES:
            return {"final": "", "verified": False, "retry": True, "retries": retries + 1}
        return {"final": REFUSAL, "verified": False, "retry": False,
                "history": _append_turn_to_history(state, REFUSAL)}
    return {"final": draft, "verified": True, "retry": False,
            "history": _append_turn_to_history(state, draft)}


def _route_after_verifier(state: State) -> str:
    """核查后的走向：通过就结束；要求重试就回回答员；否则结束（拒答）。"""
    if state.get("verified"):
        return "end"
    if state.get("retry"):
        return "answerer"
    return "end"


def _get_checkpointer_cm():
    """SQLite checkpointer（LangGraph 会话持久化）的上下文管理器，
    用 with 语句进出（开关数据库连接）。
    没有 langgraph-checkpoint-sqlite 包时返回 None，图按原来无
    checkpointer 的方式编译，问答照常、只是不跨重启持久化。
    from_conn_string 在 0.2 和 1.x 里都是 @contextmanager，用法一样；
    连接用 check_same_thread=False，服务进程多线程调 run() 也安全。"""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except Exception as e:
        print(f"[checkpointer] SQLite checkpointer 不可用（{e}），会话不跨重启持久化")
        return None
    try:
        os.makedirs(os.path.dirname(CHECKPOINTS_DB) or ".", exist_ok=True)
        return SqliteSaver.from_conn_string(CHECKPOINTS_DB)
    except Exception as e:
        print(f"[checkpointer] 打开 {CHECKPOINTS_DB} 失败（{e}），会话不跨重启持久化")
        return None


def _restore_history(graph, config) -> List[dict]:
    """从 checkpointer 里恢复上次保存的对话历史（断点续聊）。
    没存过、读失败都返回 []，不影响本次问答。"""
    try:
        snapshot = graph.get_state(config)
    except Exception as e:
        print(f"[checkpointer] 恢复历史失败（{e}），按无历史处理")
        return []
    if not snapshot or not snapshot.values:
        return []
    return snapshot.values.get("history") or []


def _build_graph(checkpointer=None):
    from langgraph.graph import StateGraph, END

    graph = StateGraph(State)
    graph.add_node("retriever", retriever_node)
    graph.add_node("answerer", answerer_node)
    graph.add_node("verifier", verifier_node)
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "answerer")
    graph.add_edge("answerer", "verifier")
    graph.add_conditional_edges("verifier", _route_after_verifier, {"answerer": "answerer", "end": END})
    return graph.compile(checkpointer=checkpointer)


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


def _to_response(result: dict) -> dict:
    return {
        "answer": result.get("final", REFUSAL),
        "verified": result.get("verified", False),
        "citations": _dedupe_citations(result.get("chunks", [])),
    }


def _run_sequential(initial: dict) -> dict:
    """LangGraph 不可用时的顺序执行兜底，流程和图里一致（含重试循环）。"""
    state: State = dict(initial)
    state.update(retriever_node(state))
    while True:
        state.update(answerer_node(state))
        state.update(verifier_node(state))
        if state.get("verified") or not state.get("retry"):
            break
    return state


def _run_with_graph(query: str, history=None) -> dict:
    """无持久化的问答路径（原来的 run 逻辑）：建图、invoke，
    图不可用时退回顺序执行。"""
    history = _normalize_history(history)
    retrieval_query = _build_retrieval_query(query, history)
    initial = {
        "query": query, "chunks": [], "draft": "", "final": "", "verified": False,
        "history": history, "retrieval_query": retrieval_query,
        "retries": 0, "retry": False,
    }
    try:
        result = _build_graph().invoke(initial)
    except Exception as e:
        print(f"[run] LangGraph 不可用，顺序执行：{e}")
        result = _run_sequential(initial)
    return _to_response(result)


def run(query: str, history=None, thread_id=None) -> dict:
    """问答入口。history 可选，不传就是原来的单轮问答；
    传了就是多轮：最近 6 轮进 prompt，追问拼上文去检索。
    thread_id 可选：传了就用 SQLite checkpointer 把这轮对话存下来，
    下次只传 query + thread_id（history 不传）就能接着聊——history
    会从 checkpointer 里恢复；不传就是原来的无状态问答。
    checkpointer 出问题时退回无持久化的图（会话不保存，但问答照常）。"""
    if thread_id:
        cm = _get_checkpointer_cm()
        if cm is not None:
            try:
                with cm as checkpointer:
                    graph = _build_graph(checkpointer)
                    config = {"configurable": {"thread_id": thread_id}}
                    if history is None:
                        # 没传 history：从 checkpointer 里恢复上一轮的
                        history = _restore_history(graph, config)
                    history = _normalize_history(history)
                    retrieval_query = _build_retrieval_query(query, history)
                    initial = {
                        "query": query, "chunks": [], "draft": "", "final": "", "verified": False,
                        "history": history, "retrieval_query": retrieval_query,
                        "retries": 0, "retry": False,
                    }
                    result = graph.invoke(initial, config=config)
                    return _to_response(result)
            except Exception as e:
                print(f"[run] checkpointer 不可用，退回无持久化：{e}")
    return _run_with_graph(query, history)
