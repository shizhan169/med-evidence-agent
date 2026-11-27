"""LangGraph 三角色：检索员 -> 回答员 -> 核查员。轻量版多智能体。
框架 v2 第一阶段 1.1：入口加了 Router 节点（结构化路由），State 加了
route / plan / subtasks / tool_calls / figure / error，verified 改名
answer_status（四种最终状态）。planner / chart_read 节点还没实现，
router 的那两个分支这一步先临时直通 retriever（见 _build_graph 的 TODO）。"""
import json
import os
import re
from typing import TypedDict, List

from .config import CHECKPOINTS_DB, LLM_BASE_URL, LLM_MODEL, PUBMED_CANDIDATE_K, PUBMED_TOP_K, get_api_key, tracing_enabled
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
    # 最终状态（原 verified 布尔字段改名）：verified / partial / refused / error。
    # verified 字段保留在 _to_response 的返回里（从 answer_status 派生），
    # 旧调用方（eval、Streamlit、/ask）读 verified 不受影响。
    # 初始值是 ""（还没定论）；verifier 会在定论时填上四种之一。
    answer_status: str
    # 路由（1.1）：Router 的结构化输出 {task_type, requires_planning,
    # requires_external_search, requires_vision}
    route: dict
    # 规划（1.2 预留）：Planner 的总体计划；subtasks 是结构化子任务列表，
    # 这一步先留空，Planner 在下一步填
    plan: dict
    subtasks: List[dict]
    # 工具调用记录（1.3）：[{tool, params, result, success, error}]
    tool_calls: List[dict]
    # 图表信息（2.3 预留）：{paper, page, figure_no, image_path, caption}，
    # 这一步先留空
    figure: dict
    # 系统异常信息（answer_status=error 时填）
    error: str
    # 多轮对话：history 是最近几轮 [{"role": "user"/"assistant", "content": ...}]，
    # retrieval_query 是真正拿去检索的问题（追问会被拼上上文，见 _build_retrieval_query）
    history: List[dict]
    retrieval_query: str
    # 核查重试：retries 是已重试次数，retry 是核查员要求再答一次的标志
    retries: int
    retry: bool
    # PubMed（第 5 步）：pubmed_results 是本次 PubMed 返回的文章列表
    # [{pmid, title, authors, journal, year, abstract}]，没调 PubMed 时为空；
    # use_pubmed 是本轮是否允许调 PubMed（评测传 False，保证 28 道题数字不变）
    pubmed_results: List[dict]
    use_pubmed: bool


# PubMed 引用的格式：PubMed PMID:38395541（冒号后面可带空格，中文冒号也认）
PMID_RE = re.compile(r"PubMed\s*PMID[:：]?\s*(\d+)")


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

# PubMed 路径重答时的提醒：只准引用本次 PubMed 返回的 PMID。
# 文本在 prompts/pubmed_retry/v1.txt，启动时读进来。
PUBMED_RETRY_REMINDER = load_prompt("pubmed_retry", "v1")

# PubMed 触发：用户明确问新文献的标志词（中文直接匹配，英文小写后匹配）。
# 问的是“最新/最近/新文献”，要的是库外的新东西，哪怕本地库里有相关片段
# 也走 PubMed——用户要的新文献不在本地 26 篇里。
NEW_LITERATURE_MARKERS = ("最新", "最近", "新文献", "新论文", "新研究", "新文章")
NEW_LITERATURE_MARKERS_EN = ("latest", "recent", "new literature", "new paper", "new study", "new article")


def _asks_new_literature(query: str) -> bool:
    """用户是不是在明确问新文献（“最新 / 最近 / 新文献……”）。"""
    q = query or ""
    if any(m in q for m in NEW_LITERATURE_MARKERS):
        return True
    ql = q.lower()
    return any(m in ql for m in NEW_LITERATURE_MARKERS_EN)


# Router（框架 v2，第一阶段 1.1）：输出结构化路由，不是 4 选 1。
# {task_type: "simple"|"complex", requires_planning, requires_external_search, requires_vision}
# 三个布尔 flag 可组合。规则明确时直接分类；规则覆盖不了时调 LLM 分类；
# 没有“其余 → single_doc”兜底——规则覆盖不了的必须走 LLM，否则 LLM
# 兜底永远触发不了。LLM 也调不通时（没 key），退回简单事实问答（retriever）。
ROUTER_PROMPT = load_prompt("router", "v1")

# 规则分类的标志词：
# - 规划：比较/对比/多篇——需要拆解或综合（英文 whole-word 用小写匹配）
# - 读图：图 N / Figure N / Fig. N / 表 N / Table N
# - 外部搜索：复用 _asks_new_literature（最新/最近/新文献……）
PLANNING_MARKERS = ("比较", "对比", "五篇", "多篇", "几篇", "分别", "异同", "区别")
PLANNING_MARKERS_EN = ("compare", "contrast", "multiple papers", "several papers", "across papers", "across studies")
FIGURE_RE = re.compile(r"(?:图|Figure|Fig\.|表|Table)\s*(\d+)", re.IGNORECASE)

# “明显是简单事实问答”的判断：问的是一个具体事实（什么/几岁/多少/哪个……），
# 且没有推理词（为什么/怎么/如何/是否……）。只有这种才直接走 simple；
# 不满足的、规则又覆盖不了的，一律走 LLM 分类。
FACTOID_MARKERS = ("什么", "几岁", "多少", "哪个", "哪篇", "是谁", "哪里", "何时", "几年", "多大")
REASONING_MARKERS = ("为什么", "怎么", "如何", "是否", "能否", "可靠", "评价", "分析", "原因", "机制", "影响", "优缺点")


def _empty_route() -> dict:
    """默认路由：简单事实问答（单步可答，不规划、不搜外部、不读图）。"""
    return {"task_type": "simple", "requires_planning": False,
            "requires_external_search": False, "requires_vision": False}


def _classify_by_rules(query: str) -> dict | None:
    """规则分类：规则明确时返回路由；规则覆盖不了时返回 None（走 LLM）。
    三个 flag 各自独立判断，可组合；task_type 由 requires_planning 决定。"""
    q = query or ""
    ql = q.lower()
    requires_planning = any(m in q for m in PLANNING_MARKERS) or any(m in ql for m in PLANNING_MARKERS_EN)
    requires_external = _asks_new_literature(q)
    requires_vision = bool(FIGURE_RE.search(q))
    if requires_planning or requires_external or requires_vision:
        return {
            "task_type": "complex" if requires_planning else "simple",
            "requires_planning": requires_planning,
            "requires_external_search": requires_external,
            "requires_vision": requires_vision,
        }
    # 没有任何标志：是“明显简单事实问答”就直接 simple，否则交给 LLM 判断
    is_factoid = any(m in q for m in FACTOID_MARKERS)
    has_reasoning = any(m in q for m in REASONING_MARKERS)
    if is_factoid and not has_reasoning:
        return _empty_route()
    return None


def _classify_by_llm(query: str) -> dict:
    """LLM 分类：规则覆盖不了时调 LLM。输出解析失败、LLM 没调通时，
    退回简单事实问答（retriever）——分类失败不断问答。"""
    try:
        raw = _call_llm(ROUTER_PROMPT.format(query=query))
    except Exception as e:
        print(f"[router] LLM 分类失败（{e}），退回简单问答")
        return _empty_route()
    if not raw:
        return _empty_route()
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
    except Exception as e:
        print(f"[router] 分类结果解析失败（{e}），退回简单问答")
        return _empty_route()
    route = _empty_route()
    for key in ("requires_planning", "requires_external_search", "requires_vision"):
        if isinstance(data.get(key), bool):
            route[key] = data[key]
    if data.get("task_type") in ("simple", "complex"):
        route["task_type"] = data["task_type"]
    elif route["requires_planning"]:
        route["task_type"] = "complex"
    return route


def router_node(state: State):
    """Router 节点：先规则分类，规则覆盖不了再调 LLM 分类；
    结果写进 State["route"]，供条件边分流。"""
    query = state.get("query", "")
    route = _classify_by_rules(query)
    if route is None:
        route = _classify_by_llm(query)
    print(f"[router] {query} → {route}")
    return {"route": route}


def _route_after_router(state: State) -> str:
    """router 之后的分流，优先级：planning → vision → 外部搜索 → 本地检索。
    返回的是逻辑目的地（"planner"/"chart_read"/"pubmed"/"retriever"），
    图里的映射把还没实现的节点临时直通 retriever（见 _build_graph 的 TODO）。
    use_pubmed=False（评测）时不走 PubMed——评测的 28 道题测本地 RAG，
    外部搜索开关关了，路由也不触发。"""
    route = state.get("route") or {}
    if route.get("requires_planning"):
        return "planner"
    if route.get("requires_vision"):
        return "chart_read"
    if route.get("requires_external_search") and state.get("use_pubmed", True):
        return "pubmed"
    return "retriever"


def _to_pubmed_query(query: str) -> str:
    """把问题翻成 PubMed 能用的英文检索式。PubMed 只认英文，中文问题直接
    送过去一个都搜不到；只在 PubMed 路径上调一次 LLM 翻译（temperature 0）。
    2026-11-26 改：让 LLM 给核心概念的 2-3 个英文同义表达，用 OR 连接，
    扩大召回（之前是字面词组直译，搜到的太少——“纵向脑常模模型”只搜到 3 篇）。
    翻译失败、翻出来还是中文、没调通就原样返回——search_pubmed 自己会返回 []。
    纯英文问题不翻译，不多花一次调用。注意这只是给 PubMed 搜的，本地检索
    的中文问题翻英文（Q09 那个）还是以后再说，不是一回事。"""
    q = (query or "").strip()
    if not q:
        return q
    if not re.search(r"[\u4e00-\u9fff]", q):
        return q
    try:
        translated = _call_llm(
            "把下面的中文问题翻译成 PubMed 英文检索式：找出核心概念，"
            "给出 2-3 个英文同义表达，每个加英文双引号，用 OR 连接。"
            "只输出检索式，不要解释：\n" + q
        )
    except Exception as e:
        print(f"[pubmed] 翻译查询失败（{e}），用原问题搜索")
        return q
    translated = (translated or "").strip().rstrip("。")
    if not translated or re.search(r"[\u4e00-\u9fff]", translated):
        return q
    # 整个输出被一对引号包起来时才去掉外层引号；OR 式里多个引号短语不动
    if len(translated) >= 2 and translated.startswith('"') and translated.endswith('"') and translated.count('"') == 2:
        translated = translated[1:-1]
    # 校验 OR 检索式：要有 OR、双引号成对；不合规就把词拼成空格分隔的关键词
    if "OR" not in translated.upper() or translated.count('"') % 2 != 0:
        words = re.findall(r"[A-Za-z][A-Za-z\- ]+", translated)
        fallback = " ".join(w.strip() for w in words if w.strip())
        print(f"[pubmed] 翻译格式不对，用关键词搜索：{fallback}")
        return fallback or q
    print(f"[pubmed] 查询翻译：{q} → {translated}")
    return translated


def _norm_pubmed_word(w: str) -> str:
    """PubMed 相关性打分用的词归一：小写；model/modeling/modelling/models
    归一成 model；复数去尾 s（charts→chart，centiles→centile），ss/us/is
    结尾的不去（analysis 这类）。"""
    w = (w or "").lower()
    if w in ("modelling", "modeling", "models", "modelled", "modeled"):
        return "model"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _phrase_hit(phrase: str, text_words: set) -> bool:
    """一个同义表达（引号里的短语）是否命中：表达里的词（归一后）都出现在
    标题+摘要的词集合里才算命中，不要求相邻。"""
    words = [_norm_pubmed_word(t) for t in re.findall(r"[A-Za-z]+", phrase)]
    return bool(words) and all(w in text_words for w in words)


def _pubmed_year(r: dict) -> int:
    """取一篇 PubMed 文章的年份（字符串或整数都行，取不到算 0，即排最后）。"""
    try:
        return int(r.get("year") or 0)
    except (TypeError, ValueError):
        return 0


def _rank_pubmed_results(results: list, en_query: str, top_k: int = PUBMED_TOP_K) -> list:
    """按题目相关性给 PubMed 候选排个序再取前 top_k。主键 = 命中几个 OR 检索
    式的同义表达（表达里的词都出现在标题+摘要里才算命中）；次键 = 年份倒序
    （同等相关时新的在前，问“最新”时这就是要的）。方向（研究领域对不对）
    不单独打分，靠回答员 v2 的相关度分档（直接相关/相关）再把关。不是 OR 式
    （纯英文问题没翻译、或翻译退回关键词）时按 PubMed 原顺序取前 top_k，不硬排。"""
    phrases = re.findall(r'"([^"]+)"', en_query or "")
    if not phrases:
        return results[:top_k]
    def key(r):
        text = f"{r.get('title', '')} {r.get('abstract', '')}"
        text_words = {_norm_pubmed_word(t) for t in re.findall(r"[A-Za-z]+", text)}
        hits = sum(1 for p in phrases if _phrase_hit(p, text_words))
        return (hits, _pubmed_year(r))
    return sorted(results, key=key, reverse=True)[:top_k]


def pubmed_node(state: State):
    """PubMed 工具节点：搜 PubMed（搜 + 取元数据，两次工具调用）。
    调不通/没结果都返回 []，不抛异常——调用方按本地-only/拒答走，问答不断。
    2026-11-26 改：先多取候选（PUBMED_CANDIDATE_K=30，按相关性排的池子），
    再按题目相关性重排取前 10——之前问“最新文献”时按发表日期倒序取前 10，
    最新但相关性差的把真正相关的顶下去了（2024 bioRxiv 常模建模、2023 海
    马体那两篇直接相关，反而不在前 10 里）。"""
    from .mcp_tools import search_pubmed

    query = state.get("retrieval_query") or state["query"]
    en_query = _to_pubmed_query(query)
    candidates = search_pubmed(en_query, top_k=PUBMED_CANDIDATE_K, sort="relevance")
    results = _rank_pubmed_results(candidates, en_query)
    if results:
        print(f"[pubmed] 候选 {len(candidates)} 篇，取前 {len(results)} 篇：{', '.join(r['pmid'] for r in results)}")
    return {"pubmed_results": results}


def _route_after_retriever(state: State) -> str:
    """检索员之后：PubMed 关了（评测）→ 直接回答（没证据就拒答，和原来一样）；
    本地有证据且用户没问新文献 → 直接回答；本地没证据、或用户明确问新文献
    → 走 PubMed 工具节点（有结果就用 PubMed 回答，没结果再按老路走）。"""
    if not state.get("use_pubmed", True):
        return "answerer"
    if state.get("chunks") and not _asks_new_literature(state.get("query", "")):
        return "answerer"
    return "pubmed"


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
    from .rag import build_prompt, build_pubmed_prompt

    pubmed_results = state.get("pubmed_results") or []
    if pubmed_results:
        # PubMed 路径：只用 PubMed 返回的文章回答，本地片段不用——两种引用
        # 体系（PubMed PMID / 本地论文 p.页码）不混在一道答案里，核查也只认 PMID
        prompt = build_pubmed_prompt(state["query"], pubmed_results, history=state.get("history") or [])
        reminder = PUBMED_RETRY_REMINDER
    elif state["chunks"]:
        prompt = build_prompt(state["query"], state["chunks"], history=state.get("history") or [])
        reminder = RETRY_REMINDER
    else:
        return {"draft": REFUSAL}
    if state.get("retries", 0) > 0:
        # 重答：提醒模型只引用本次提供的依据，上一轮就是引用错了才回来的
        prompt = prompt + reminder
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


def _refusal_status(draft: str) -> str:
    """拒答类收尾的状态：LLM 没调通（[待接 LLM]）是系统异常 error，
    其余（没证据、模型主动拒答）是 refused。"""
    return "error" if draft.startswith("[待接 LLM]") else "refused"


def _verification_failed(state: State, reason: str) -> dict:
    """核查失败的统一收尾：打印原因；还有重试次数就回回答员重答，
    用完就拒答——绝不编造。重试时 answer_status 还没定论（不写），
    最终定论时由 verifier 填 verified / refused。"""
    print(f"[verifier] {reason}")
    retries = state.get("retries", 0)
    if retries < MAX_RETRIES:
        return {"final": "", "retry": True, "retries": retries + 1}
    return {"final": REFUSAL, "answer_status": "refused", "retry": False,
            "history": _append_turn_to_history(state, REFUSAL)}


def _verify_pubmed(state: State, draft: str, pubmed_results: list) -> dict:
    """PubMed 路径的核查：答案里的每个 PMID 必须在本次 PubMed 返回的列表里，
    且不许出现本地式引用（[来源：论文名 p.页码]）——两种引用体系不混用。
    假 PMID（工具没返回过）按引用编造处理，走重试/拒答。"""
    cited_pmids = set(PMID_RE.findall(draft))
    real_pmids = {str(r.get("pmid", "")) for r in pubmed_results}
    local_cites = list(CITE_RE.finditer(draft))
    if not cited_pmids:
        return _verification_failed(state, "回答没有给出 PMID 引用")
    if not cited_pmids.issubset(real_pmids):
        bad = sorted(cited_pmids - real_pmids)
        return _verification_failed(state, f"引用的 PMID 不在本次 PubMed 返回结果中：{bad}")
    if local_cites:
        return _verification_failed(state, "PubMed 回答里混入了本地式引用")
    return {"final": draft, "answer_status": "verified", "retry": False,
            "history": _append_turn_to_history(state, draft)}


def verifier_node(state: State):
    # 核查规则：draft 里的每个 [来源：... p.X] 必须对应一个真实检索片段，否则拒答
    draft = state.get("draft", "")
    pubmed_results = state.get("pubmed_results") or []
    if pubmed_results:
        # PubMed 路径：只核查 PMID；回答员本来就拒答、LLM 没调通时不重试
        if draft == REFUSAL or draft.startswith("[待接 LLM]"):
            final = draft
            return {"final": final, "answer_status": _refusal_status(draft), "retry": False,
                    "history": _append_turn_to_history(state, final)}
        return _verify_pubmed(state, draft, pubmed_results)
    if not state.get("chunks") or draft == REFUSAL or draft.startswith("[待接 LLM]"):
        # 没片段、回答员本来就拒答、LLM 没调通：重试也没用，直接结束
        final = REFUSAL if not state.get("chunks") else draft
        return {"final": final, "answer_status": _refusal_status(final), "retry": False,
                "history": _append_turn_to_history(state, final)}
    cited = set((m.group(1), m.group(2)) for m in CITE_RE.finditer(draft))
    real = set(
        (str(c.get("source", "未知")), str(c.get("page", "?"))) for c in state["chunks"]
    )
    if not cited or not cited.issubset(real):
        # 引用错了（编了页码、或没给引用）：还有重试次数就回回答员重答，
        # 用完就拒答——绝不编造
        if not cited:
            return _verification_failed(state, "回答没有给出引用")
        bad = sorted(cited - real)
        return _verification_failed(state, f"引用不在检索结果中：{bad}")
    return {"final": draft, "answer_status": "verified", "retry": False,
            "history": _append_turn_to_history(state, draft)}


def _route_after_verifier(state: State) -> str:
    """核查后的走向：通过（verified/partial）就结束；要求重试就回回答员；
    否则结束（拒答/异常）。partial 这一步还不会出现（1.5 再做），先写上。"""
    if state.get("answer_status") in ("verified", "partial"):
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
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("pubmed", pubmed_node)
    graph.add_node("answerer", answerer_node)
    graph.add_node("verifier", verifier_node)
    graph.set_entry_point("router")
    # router 之后的分流。planner / chart_read 节点还没实现（1.2 / 2.3），
    # 这一步先把这两个分支临时直通 retriever，保证图完整可跑；
    # 节点实现后把映射改成真正的节点名。
    graph.add_conditional_edges("router", _route_after_router, {
        "planner": "retriever",    # TODO(1.2): 改成 planner 节点
        "chart_read": "retriever",  # TODO(2.3): 改成 chart_read 节点
        "pubmed": "pubmed",
        "retriever": "retriever",
    })
    graph.add_conditional_edges("retriever", _route_after_retriever, {"answerer": "answerer", "pubmed": "pubmed"})
    graph.add_edge("pubmed", "answerer")
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


def _dedupe_pubmed_citations(results: List[dict]) -> List[dict]:
    """PubMed 引用列表：按 PMID 去重；每篇只留展示用的字段（来源固定写 PubMed，
    页码没有——PubMed 的引用就是 PMID 本身）。"""
    seen = set()
    out = []
    for r in results:
        pmid = str(r.get("pmid", ""))
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        out.append({
            "source": "PubMed",
            "pmid": pmid,
            "title": r.get("title", ""),
            "journal": r.get("journal", ""),
            "year": r.get("year", ""),
        })
    return out


def _to_response(result: dict) -> dict:
    # answer_status 是最终状态；verified 是兼容字段（旧调用方读它），
    # 只有 answer_status == "verified" 时为 True，其余一律 False。
    # answer_status 为空（还没定论就不该到这一步）时按 refused 处理。
    answer_status = result.get("answer_status") or "refused"
    verified = answer_status == "verified"
    pubmed_results = result.get("pubmed_results") or []
    if pubmed_results:
        return {
            "answer": result.get("final", REFUSAL),
            "verified": verified,
            "answer_status": answer_status,
            "citations": _dedupe_pubmed_citations(pubmed_results),
        }
    return {
        "answer": result.get("final", REFUSAL),
        "verified": verified,
        "answer_status": answer_status,
        "citations": _dedupe_citations(result.get("chunks", [])),
    }


def _run_sequential(initial: dict) -> dict:
    """LangGraph 不可用时的顺序执行兜底，流程和图里一致（含 Router、PubMed 分支和重试循环）。"""
    state: State = dict(initial)
    state.update(router_node(state))
    dest = _route_after_router(state)
    if dest == "pubmed":
        state.update(pubmed_node(state))
    else:
        # planner / chart_read / retriever 目前都临时走 retriever
        state.update(retriever_node(state))
        if _route_after_retriever(state) == "pubmed":
            state.update(pubmed_node(state))
    while True:
        state.update(answerer_node(state))
        state.update(verifier_node(state))
        if state.get("answer_status") in ("verified", "partial") or not state.get("retry"):
            break
    return state


def _run_with_graph(query: str, history=None, use_pubmed: bool = True) -> dict:
    """无持久化的问答路径（原来的 run 逻辑）：建图、invoke，
    图不可用时退回顺序执行。"""
    history = _normalize_history(history)
    retrieval_query = _build_retrieval_query(query, history)
    initial = {
        "query": query, "chunks": [], "draft": "", "final": "", "answer_status": "",
        "route": {}, "plan": {}, "subtasks": [], "tool_calls": [], "figure": {}, "error": "",
        "history": history, "retrieval_query": retrieval_query,
        "retries": 0, "retry": False,
        "pubmed_results": [], "use_pubmed": use_pubmed,
    }
    try:
        result = _build_graph().invoke(initial)
    except Exception as e:
        print(f"[run] LangGraph 不可用，顺序执行：{e}")
        result = _run_sequential(initial)
    return _to_response(result)


def run(query: str, history=None, thread_id=None, use_pubmed: bool = True) -> dict:
    """问答入口。history 可选，不传就是原来的单轮问答；
    传了就是多轮：最近 6 轮进 prompt，追问拼上文去检索。
    thread_id 可选：传了就用 SQLite checkpointer 把这轮对话存下来，
    下次只传 query + thread_id（history 不传）就能接着聊——history
    会从 checkpointer 里恢复；不传就是原来的无状态问答。
    checkpointer 出问题时退回无持久化的图（会话不保存，但问答照常）。
    use_pubmed：本轮是否允许调 PubMed（默认 True）。评测传 False——
    28 道题测的是本地 RAG，PubMed 是外部服务：应拒答题本地没证据，
    按触发规则会走 PubMed，数字会被外部服务带偏、还不可复现；
    PubMed 能力由 tests/test_pubmed.py 的 mock 测试 + 一次真跑验收，
    不走 28 道题。"""
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
                        "query": query, "chunks": [], "draft": "", "final": "", "answer_status": "",
                        "route": {}, "plan": {}, "subtasks": [], "tool_calls": [], "figure": {}, "error": "",
                        "history": history, "retrieval_query": retrieval_query,
                        "retries": 0, "retry": False,
                        "pubmed_results": [], "use_pubmed": use_pubmed,
                    }
                    result = graph.invoke(initial, config=config)
                    return _to_response(result)
            except Exception as e:
                print(f"[run] checkpointer 不可用，退回无持久化：{e}")
    return _run_with_graph(query, history, use_pubmed=use_pubmed)
