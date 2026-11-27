"""LangGraph 三角色：检索员 -> 回答员 -> 核查员。轻量版多智能体。
框架 v2 第一阶段 1.1：入口加了 Router 节点（结构化路由），State 加了
route / plan / subtasks / tool_calls / figure / error，verified 改名
answer_status（四种最终状态）。1.2：planner / executor 节点已实现，
router 的 planning 分支走真正的 planner；2.3：chart_read 节点已实现，
router 的 vision 分支走 VLM 读图（没图可读时退回 retriever 兜底）。
1.4：历史管理抽到 app/memory.py；历史超限时被截掉的轮次先摘要进
State["history_summary"] 再丢，prompt 只拼最近 2 轮完整内容 + 摘要；
checkpointer 里仍是最近 6 轮完整历史，压缩不影响持久化。1.5：核查员加
失败归因（verify_failures：fabricated_citation / missing_citation /
insufficient_evidence / llm_error）+ 部分回答（partial：只留已核实部分，
标出未答的子任务）+ 只重跑失败的子任务（retry_target 决定回 executor
还是 answerer）；PubMed 引用接受 PMID/DOI/摘要三种形式。"""
import json
import os
import re
from typing import TypedDict, List

from .config import CHECKPOINTS_DB, LLM_BASE_URL, LLM_MODEL, PUBMED_CANDIDATE_K, PUBMED_TOP_K, get_api_key, tracing_enabled
from .memory import MAX_HISTORY_CHARS, MAX_HISTORY_TURNS
from . import memory
from .prompts import load_prompt
from .tools import call_tool, list_tools

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
    # 规划（1.2）：Planner 的总体计划；subtasks 是结构化子任务列表，
    # 每个子任务 {task_id, query, status, evidence_ids, result, retry_count}，
    # status 是 pending / running / completed / failed / insufficient_evidence
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
    # retrieval_query 是真正拿去检索的问题（追问会被拼上上文，见 _build_retrieval_query）；
    # history_summary 是较早轮次的压缩摘要（1.4）：历史超限时，被截掉的轮次先摘要
    # 再丢，摘要写在这里，不丢信息；prompt 只拼最近 2 轮完整内容 + 摘要。
    history: List[dict]
    history_summary: str
    retrieval_query: str
    # 核查重试：retries 是已重试次数，retry 是核查员要求再答一次的标志
    retries: int
    retry: bool
    # 1.5 核查升级：verify_failures 是失败归因 [{subtask_id, reason}]，
    # reason 是 fabricated_citation / missing_citation / insufficient_evidence /
    # llm_error（subtask_id 为 "" 表示整篇答案级的问题，不归属到某个子任务）；
    # retry_target 是重试去向（"executor" 只重跑失败的子任务，"answerer" 重答），
    # 只在 retry 为真时有效
    verify_failures: List[dict]
    retry_target: str
    # PubMed（第 5 步）：pubmed_results 是本次 PubMed 返回的文章列表
    # [{pmid, title, authors, journal, year, abstract}]，没调 PubMed 时为空；
    # use_pubmed 是本轮是否允许调 PubMed（评测传 False，保证 28 道题数字不变）
    pubmed_results: List[dict]
    use_pubmed: bool


# PubMed 引用的格式：PubMed PMID:38395541（冒号后面可带空格，中文冒号也认）
PMID_RE = re.compile(r"PubMed\s*PMID[:：]?\s*(\d+)")

# PubMed 引用放宽（1.5）：除 PMID 外，DOI 和摘要引用也接受，不强制全文页码。
# DOI 形如 DOI:10.1002/alz.14174；mcp_tools 只返回 pmid/title/authors/
# journal/year/abstract，没有结构化 DOI 字段，所以 DOI 只认“出现在本次返回的
# 标题/摘要文字里”的——编出来的 DOI 不在返回文字里，仍按编造处理。
# 摘要引用指草稿里原样引用了某篇返回文章的摘要文字（连续 40 字以上）；
# 摘要太短（<20 字）不认，避免一两个通用词就被当成引用。
DOI_RE = re.compile(r"doi[:：]?\s*(10\.\d+/[^\s,;，。）\)\]]+)", re.IGNORECASE)
MIN_ABSTRACT_QUOTE = 40


# 多轮对话的历史上限（定义在 app/memory.py，这里 import 进来，旧测试读
# agent.MAX_HISTORY_TURNS / MAX_HISTORY_CHARS 不受影响）。
# 历史保留最近 6 轮（一轮 = 一问一答），总字数不超过 4000 字，
# 太长就从最早的一轮开始丢；prompt 太长既费 token、模型也容易抓不住重点。
# 1.4 之后：被截掉的轮次先摘要进 history_summary 再丢（见 _append_turn_to_history），
# 摘要写进 State，不丢信息；压缩只影响传给 LLM 的 prompt 上下文。

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
    图里的映射把 chart_read 指向真正的 chart_read 节点（VLM 读图）。
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


# Planner / Executor（框架 v2，第一阶段 1.2）。
# Planner 只负责拆解，不执行、不选工具；Executor 只在单个子任务内做
# 有限自主选工具（不是全量 ReAct：Planner 先定子任务结构）。
PLANNER_PROMPT = load_prompt("planner", "v1")
EXECUTOR_PROMPT = load_prompt("executor", "v1")

# Executor：总共最多 MAX_EXECUTOR_STEPS 次工具调用，超了就停，返回部分结果；
# 每个子任务最多重试 MAX_SUBTASK_RETRIES 次（失败只重跑失败的子任务）。
MAX_EXECUTOR_STEPS = 8
MAX_SUBTASK_RETRIES = 1


def _parse_plan(raw: str) -> tuple:
    """解析 Planner 的 JSON 输出，返回 (goal, 子任务 query 列表)。
    解析失败、结构不对、子任务为空，都返回 ("", [])——调用方退回单个子任务。
    子任务最多取 5 个。"""
    if not raw:
        return "", []
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
    except Exception:
        return "", []
    if not isinstance(data, dict):
        return "", []
    goal = data.get("goal") if isinstance(data.get("goal"), str) else ""
    subtasks = data.get("subtasks")
    if not isinstance(subtasks, list):
        return goal.strip(), []
    queries = []
    for s in subtasks:
        if isinstance(s, dict) and isinstance(s.get("query"), str) and s["query"].strip():
            queries.append(s["query"].strip())
    return goal.strip(), queries[:5]


def planner_node(state: State):
    """Planner 节点：把复杂问题拆解成结构化子任务，写进 State["subtasks"]。
    触发条件是“需要拆解”，不是“跨文献”——单篇内需要多步推理的复杂问题
    也可以拆（Router 的 requires_planning 为真就进这里，不再判断是不是跨文献）。
    拆解失败（LLM 没调通、返回的不是合法 JSON、子任务为空）时退回单个子任务
    （就是原问题本身），问答不断。"""
    query = state.get("query", "")
    goal, queries = "", []
    try:
        raw = _call_llm(PLANNER_PROMPT.format(query=query))
        goal, queries = _parse_plan(raw)
    except Exception as e:
        print(f"[planner] 拆解失败（{e}），退回单个子任务")
    if not queries:
        goal, queries = query, [query]
    subtasks = [
        {"task_id": f"S{i + 1}", "query": q, "status": "pending",
         "evidence_ids": [], "result": "", "retry_count": 0}
        for i, q in enumerate(queries)
    ]
    print(f"[planner] {query} → {len(subtasks)} 个子任务")
    return {"plan": {"goal": goal, "total": len(subtasks)}, "subtasks": subtasks}


def _build_executor_prompt(subtask_query: str) -> str:
    """给 Executor 选工具用的 prompt：工具列表（名字 + 描述）+ 子任务。"""
    tools_text = "\n".join(
        f"- {t['name']}：{t['description']}" for t in list_tools()
    )
    return EXECUTOR_PROMPT.format(tools=tools_text, subtask=subtask_query)


def _parse_tool_call(raw: str) -> tuple:
    """解析 Executor 的选工具 JSON，返回 (工具名, 参数 dict)。
    解析失败、工具名不在注册表里、参数不是 dict，都退回
    ("local_search", {})——本地检索是默认工具，选不出来也不断问答；
    调用方会把子任务的 query 补进 local_search 的参数。"""
    if not raw:
        return "local_search", {}
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
    except Exception:
        return "local_search", {}
    if not isinstance(data, dict):
        return "local_search", {}
    tool = data.get("tool")
    params = data.get("params")
    if not isinstance(tool, str) or tool not in {t["name"] for t in list_tools()}:
        return "local_search", {}
    if not isinstance(params, dict):
        params = {}
    return tool, params


def _merge_chunks(existing: list, new: list) -> list:
    """把新检索到的片段并进已有片段，按 (source, page, text) 去重。"""
    seen = {(str(c.get("source")), str(c.get("page")), str(c.get("text"))) for c in existing}
    out = list(existing)
    for c in new:
        key = (str(c.get("source")), str(c.get("page")), str(c.get("text")))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _merge_pubmed_results(existing: list, new: list) -> list:
    """把新的 PubMed 文章并进已有结果，按 PMID 去重。"""
    seen = {str(r.get("pmid", "")) for r in existing}
    out = list(existing)
    for r in new:
        pmid = str(r.get("pmid", ""))
        if pmid and pmid not in seen:
            seen.add(pmid)
            out.append(r)
    return out


def executor_node(state: State):
    """Executor 节点：按子任务顺序执行。
    每个子任务：LLM 按子任务 + 工具描述选工具 → 校验参数并执行 →
    结果写回 State（子任务的 result、evidence_ids、status）。
    有限循环：总共最多 MAX_EXECUTOR_STEPS 次工具调用，超了就停，
    剩下的子任务保持 pending，返回已有的部分结果。
    失败只重跑失败的子任务（每个子任务最多重试 MAX_SUBTASK_RETRIES 次，
    重试时重新选工具），不重跑全流程。
    chart_read 是占位（available=False），选到它时子任务记为
    insufficient_evidence，不重试。每个工具调用都记进 tool_calls。"""
    subtasks = [dict(s) for s in state.get("subtasks") or []]
    tool_calls = list(state.get("tool_calls") or [])
    chunks = list(state.get("chunks") or [])
    pubmed_results = list(state.get("pubmed_results") or [])
    steps = 0
    for st in subtasks:
        if steps >= MAX_EXECUTOR_STEPS:
            break
        if st.get("status") != "pending":
            continue
        st["status"] = "running"
        while steps < MAX_EXECUTOR_STEPS:
            # 选工具：LLM 按子任务 + 工具描述选；选不出来用默认 local_search
            try:
                raw = _call_llm(_build_executor_prompt(st.get("query", "")))
            except Exception as e:
                print(f"[executor] 选工具失败（{e}），用默认 local_search")
                raw = ""
            tool_name, params = _parse_tool_call(raw)
            if tool_name in ("local_search", "pubmed_search") and not params.get("query"):
                params["query"] = st.get("query", "")
            result = call_tool(tool_name, **params)
            steps += 1
            tool_calls.append({
                "tool": tool_name, "params": params,
                "result": result.get("data"), "success": result.get("success", False),
                "error": result.get("error"),
            })
            # chart_read 占位：工具本身“成功”但返回 available=False，
            # 说明现在读不了图——子任务记 insufficient_evidence，不重试
            if (tool_name == "chart_read" and isinstance(result.get("data"), dict)
                    and result["data"].get("available") is False):
                st["status"] = "insufficient_evidence"
                st["result"] = result["data"].get("reason", "VLM 未实现")
                break
            if result.get("success"):
                data = result.get("data") or {}
                if tool_name == "local_search":
                    new_chunks = data.get("chunks") or []
                    chunks = _merge_chunks(chunks, new_chunks)
                    st["evidence_ids"] = [f"{c.get('source', '未知')} p.{c.get('page', '?')}" for c in new_chunks]
                    st["result"] = f"检索到 {len(new_chunks)} 个片段"
                    st["status"] = "completed"
                elif tool_name == "pubmed_search":
                    new_results = data.get("results") or []
                    pubmed_results = _merge_pubmed_results(pubmed_results, new_results)
                    st["evidence_ids"] = [str(r.get("pmid", "")) for r in new_results]
                    st["result"] = f"检索到 {len(new_results)} 篇文献"
                    st["status"] = "completed"
                elif tool_name == "chart_locate":
                    if data.get("found"):
                        st["evidence_ids"] = [str(data.get("source", ""))]
                        st["result"] = data.get("caption", "")
                        st["status"] = "completed"
                    else:
                        st["evidence_ids"] = []
                        st["result"] = "未找到该图表"
                        st["status"] = "insufficient_evidence"
                else:
                    st["result"] = str(data)
                    st["status"] = "completed"
                break
            # 工具执行失败：还有重试次数就重新选工具再试一次（只重跑这个子任务），
            # 用完就记 failed
            if st.get("retry_count", 0) < MAX_SUBTASK_RETRIES:
                st["retry_count"] = st.get("retry_count", 0) + 1
                print(f"[executor] 子任务 {st.get('task_id')} 失败，重试 {st['retry_count']}/{MAX_SUBTASK_RETRIES}")
                continue
            st["status"] = "failed"
            st["result"] = result.get("error") or "工具执行失败"
            break
    # 步骤上限打断时，正在跑的子任务改回 pending（它没执行完，不是失败）
    for st in subtasks:
        if st.get("status") == "running":
            st["status"] = "pending"
    return {"subtasks": subtasks, "tool_calls": tool_calls, "chunks": chunks, "pubmed_results": pubmed_results}


def _normalize_history(history) -> List[dict]:
    """把调用方传来的 history 洗成干净的 [{"role", "content"}] 列表。
    逻辑在 app/memory.py（normalize_history），这里只做兼容包装：
    只保留最近 MAX_HISTORY_TURNS 轮；再按总字数截断，超了就从最早的一轮
    开始丢。返回的一定是整轮。被截掉的轮次由调用方拿 memory.normalize_history
    的 dropped 再摘要（见 _append_turn_to_history、run），这里不摘要——
    短历史不产生额外 LLM 调用。"""
    kept, _dropped = memory.normalize_history(history)
    return kept


def _build_retrieval_query(query: str, history: List[dict]) -> str:
    """给检索用的问题。追问（带指代词）先改写成完整独立的问题（指代消解、
    省略补全，见 rag.rewrite_query_for_retrieval），改写后的问题才拿去检索；
    普通问题原样返回，不加东西。改写调一次 LLM，失败时退回老做法（把上
    一轮用户问题拼在前面）。中文翻英文不在这里做——检索时 retrieve() 内
    部再翻向量检索词，这样 PubMed 路径的 _to_pubmed_query 仍能拿到中文
    原问题做 OR 拓宽。"""
    from .rag import rewrite_query_for_retrieval

    return rewrite_query_for_retrieval(query, history, llm_call=_call_llm)


def _extract_paper_and_figure(query: str):
    """从图表问题里抽论文名和图号。
    图号：第一个 "图 N"；论文：问题开头的拉丁作者名（连续的拉丁词，
    遇到数字/中文就停），去空格、去重音后返回，供 chart_locate 的子串匹配用
    （Di Biase→dibiase、Coupé→coupe，对上文件名里的 dibiase/coupe）。
    抽不到图号返回 (None, None)，调用方退回 retriever。"""
    import unicodedata

    q = query or ""
    m = re.search(r"图\s*(\d+)", q)
    if not m:
        return None, None
    figure_no = m.group(1)
    lead = re.match(r"^[A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+)*", q.strip())
    if not lead:
        return None, figure_no
    norm = unicodedata.normalize("NFKD", lead.group(0)).encode("ascii", "ignore").decode()
    paper = re.sub(r"\s+", "", norm).lower()
    return (paper or None), figure_no


def chart_read_node(state: State):
    """图表题节点（第二阶段 2.3）：VLM 读图。
    先从问题里抽论文 + 图号，调 chart_read 工具拿 VLM 读图结果；
    成功时把读图结果转成一个可引用片段（source=论文文件名、page=图所在页，
    text=VLM 原文 + 图注），回答员按正常引用流程引用它，核查员按 (source, page)
    校验。降级（数字对不上图注）时在片段里注明，回答员不把存疑数字当定论。
    没图可读（本地没 PDF、图号找不到、VLM 调不通、问题里抽不出图号）时
    退回 retriever（图注/正文路径），图表题不断答——老行为是兜底。"""
    from .rag import retrieve

    query = state.get("query", "")
    tool_calls = list(state.get("tool_calls") or [])
    paper, figure_no = _extract_paper_and_figure(query)
    if paper and figure_no:
        result = call_tool("chart_read", paper=paper, figure_no=figure_no, question=query)
        tool_calls.append({
            "tool": "chart_read",
            "params": {"paper": paper, "figure_no": figure_no, "question": query},
            "result": result.get("data"), "success": result.get("success", False),
            "error": result.get("error"),
        })
        data = result.get("data") or {}
        if result.get("success") and data.get("available"):
            source = data.get("source") or ""
            page = data.get("page")
            reading = data.get("reading") or ""
            caption = data.get("caption") or ""
            text = f"图注：{caption}\nVLM 读图：{reading}" if caption else reading
            if data.get("downgraded"):
                text += "\n（注意：VLM 读出的部分数字与图注不符，以下数字可能不准）"
            chunk = {"source": source, "page": page, "text": text}
            figure = {"paper": source, "page": page, "figure_no": data.get("figure_no") or figure_no,
                      "image_path": "", "caption": caption}
            return {"chunks": [chunk], "tool_calls": tool_calls, "figure": figure}
        print(f"[chart_read] VLM 不可用（{data.get('reason') or result.get('error')}），退回 retriever")
    else:
        print("[chart_read] 问题里抽不出论文/图号，退回 retriever")
    chunks = retrieve(state.get("retrieval_query") or query, top_k=10, llm_call=_call_llm)
    return {"chunks": chunks, "tool_calls": tool_calls}


def retriever_node(state: State):
    from .rag import retrieve

    # 追问已经由 _build_retrieval_query 改写成完整意思，检索用它；
    # 没历史时 retrieval_query 就是原问题。把 _call_llm 传给 retrieve，
    # 让它在内部把中文问题翻成英文再查向量（跨语言向量检索弱，Q09 的教训）；
    # 没 key 时 _call_llm 返回空，翻译退回原问题，检索不断。
    chunks = retrieve(state.get("retrieval_query") or state["query"], top_k=10, llm_call=_call_llm)
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

    # 1.4 压缩：prompt 用的历史不是 state["history"] 原样——历史超限时，
    # 只拼最近 2 轮完整内容 + 摘要（memory.prompt_history），较早轮次在摘要里；
    # 没摘要时原样返回，和原来一模一样。
    prompt_hist = memory.prompt_history(state.get("history") or [], state.get("history_summary") or [])
    pubmed_results = state.get("pubmed_results") or []
    if pubmed_results:
        # PubMed 路径：只用 PubMed 返回的文章回答，本地片段不用——两种引用
        # 体系（PubMed PMID / 本地论文 p.页码）不混在一道答案里，核查也只认 PMID
        prompt = build_pubmed_prompt(state["query"], pubmed_results, history=prompt_hist)
        reminder = PUBMED_RETRY_REMINDER
    elif state["chunks"]:
        prompt = build_prompt(state["query"], state["chunks"], history=prompt_hist)
        reminder = RETRY_REMINDER
    else:
        return {"draft": REFUSAL}
    subtasks = state.get("subtasks") or []
    if subtasks:
        # 2.2 汇总缺口修法：规划任务的回答员必须覆盖全部子任务。
        # 原来 prompt 里只有问题和片段，没有子任务清单，回答员可能悄悄漏掉
        # 一个维度/一篇论文，核查还报 verified（引用都是真的、子任务也都有证据，
        # 只是答案里没用上）。这里把子任务清单写进 prompt，要求每个都覆盖、
        # 每个至少引用一次它的证据。
        sub_lines = "\n".join(
            f"- {st.get('task_id', '')}：{st.get('query', '')}"
            for st in subtasks if st.get("query")
        )
        if sub_lines:
            prompt += (
                "\n\n注意：这是一个多步骤任务，答案必须覆盖以下全部子任务，"
                "每个子任务至少引用一次它的证据，不要漏掉任何一个：\n" + sub_lines + "\n"
            )
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


def _append_turn_to_history(state: State, final: str) -> dict:
    """把本轮（一问一答）追加到历史里，供 checkpointer 持久化。
    只在核查员给出最终答案时调（重试时不调，那轮还没定论）。
    历史照样只留最近 6 轮、4000 字（memory.normalize_history 的上限）；
    被截掉的轮次先摘要进 history_summary（调一次 LLM，失败退回规则式），
    再丢——摘要写进 State，不丢信息；短历史不截断、不调 LLM，和原来一样。
    返回 {"history": ..., "history_summary": ...}，调用方用 ** 展开进返回的 state。"""
    history = list(state.get("history") or [])
    history.append({"role": "user", "content": state.get("query", "")})
    history.append({"role": "assistant", "content": final})
    kept, dropped = memory.normalize_history(history)
    new_summary = memory.append_summary(state.get("history_summary") or "", dropped, llm_call=_call_llm)
    return {"history": kept, "history_summary": new_summary}


# ---- 1.5 核查升级：失败归因、部分回答、只重跑失败的子任务 ----

def _norm_text(text: str) -> str:
    """摘要引用比对用的归一：小写、空白折叠。"""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _abstract_windows(pubmed_results: list):
    """从本次 PubMed 返回的每篇文章摘要里切出 40 字的窗口（步长 5），
    返回 [(窗口, pmid)]。草稿里出现任一窗口，就算引用了该篇的摘要。
    摘要太短（<20 字）不切，避免通用短语被当成引用。"""
    windows = []
    for r in pubmed_results or []:
        abstract = _norm_text(r.get("abstract", ""))
        if len(abstract) < 20:
            continue
        pmid = str(r.get("pmid", ""))
        if len(abstract) <= MIN_ABSTRACT_QUOTE:
            windows.append((abstract, pmid))
            continue
        for i in range(0, len(abstract) - MIN_ABSTRACT_QUOTE + 1, 5):
            windows.append((abstract[i:i + MIN_ABSTRACT_QUOTE], pmid))
    return windows


def _abstract_cited_pmids(draft: str, windows) -> set:
    """草稿里原样引用了哪些文章的摘要（按窗口匹配）。"""
    norm_draft = _norm_text(draft)
    return {pmid for w, pmid in windows if w and w in norm_draft}


def _pubmed_refs(draft: str, pubmed_results: list):
    """PubMed 路径的引用提取。返回 (cited_pmids, cited_dois, abstract_pmids)：
    cited_pmids 是 PubMed PMID:xxx 形式的；cited_dois 是 DOI:10.xxx 形式的
    （归一小写）；abstract_pmids 是草稿里原样引用了其摘要文字的文章 PMID。"""
    cited_pmids = set(PMID_RE.findall(draft or ""))
    cited_dois = {d.lower() for d in DOI_RE.findall(draft or "")}
    windows = _abstract_windows(pubmed_results)
    abstract_pmids = _abstract_cited_pmids(draft, windows)
    return cited_pmids, cited_dois, abstract_pmids


def _doi_in_returned(dois: set, pubmed_results: list) -> set:
    """哪些 DOI 出现在本次 PubMed 返回的标题/摘要文字里（即可接受的 DOI）。"""
    text = " ".join(
        _norm_text(f"{r.get('title', '')} {r.get('abstract', '')}")
        for r in pubmed_results or []
    )
    return {d for d in dois if d in text}


def _local_evidence_pool(chunks: list):
    """本地证据池：{(source, page)}，以及 {source} 文件名集合。"""
    cites = set()
    sources = set()
    for c in chunks or []:
        sources.add(str(c.get("source", "未知")))
        cites.add((str(c.get("source", "未知")), str(c.get("page", "?"))))
    return cites, sources


def _parse_evidence_id(eid: str):
    """把子任务的 evidence_id 还原成可核对的形式：
    - "paper.pdf p.3" → ("local", source, page)
    - "38395541"（纯数字）→ ("pmid", pmid)
    - 其他（chart_locate 的文件名）→ ("source", filename)：图注定位工具
      本身成功就是证据（它不产生检索片段，进不了 chunks 池），直接认。"""
    s = str(eid)
    m = re.match(r"^(.*) p\.(\?|\d+)$", s)
    if m:
        return ("local", m.group(1), m.group(2))
    if s.strip().isdigit():
        return ("pmid", s.strip())
    return ("source", s)


def _check_subtasks(subtasks, local_cites, chunk_sources, real_pmids):
    """逐个子任务核查证据。返回 (verified_ids, failures)：
    verified 的条件：status == "completed"，且 evidence_ids 非空、且每条都
    能在证据池里找到（本地 (source, page)、PMID；chart_locate 的文件名直接认）。
    其余（insufficient_evidence / failed / pending / running、或 evidence_ids
    为空、或证据不在池里）→ insufficient_evidence。
    注意：failed / insufficient 的子任务 evidence_ids 本来就是空的（Executor
    只在工具成功时填 evidence_ids），所以“证据不在池里”实际只发生在
    completed 但 evidence 被漏掉的极端情况。"""
    verified = set()
    failures = []
    for st in subtasks or []:
        tid = st.get("task_id", "")
        if st.get("status") != "completed":
            failures.append({"subtask_id": tid, "reason": "insufficient_evidence"})
            continue
        eids = st.get("evidence_ids") or []
        if not eids:
            failures.append({"subtask_id": tid, "reason": "insufficient_evidence"})
            continue
        ok = True
        for eid in eids:
            kind, *rest = _parse_evidence_id(eid)
            if kind == "local":
                if (rest[0], rest[1]) not in local_cites:
                    ok = False
                    break
            elif kind == "pmid":
                if rest[0] not in real_pmids:
                    ok = False
                    break
            # ("source", ...) 是 chart_locate 的文件名：工具成功即证据，直接认
        if ok:
            verified.add(tid)
        else:
            failures.append({"subtask_id": tid, "reason": "insufficient_evidence"})
    return verified, failures


def _verified_evidence_keys(subtasks, verified_ids):
    """已核实子任务的证据键：本地 {(source, page)}、PubMed {pmid}。
    partial 过滤句子时用：只留引用全在这个集合里的句子。"""
    local_keys = set()
    pmid_keys = set()
    by_id = {st.get("task_id"): st for st in subtasks or []}
    for tid in verified_ids:
        for eid in (by_id.get(tid) or {}).get("evidence_ids") or []:
            kind, *rest = _parse_evidence_id(eid)
            if kind == "local":
                local_keys.add((rest[0], rest[1]))
            elif kind == "pmid":
                pmid_keys.add(rest[0])
    return local_keys, pmid_keys


def _filter_partial_draft(draft, verified_local, verified_pmids, is_pubmed, windows):
    """partial 答案只留已核实的部分：按句过滤。
    保留：一句里的引用全在已核实证据集合里（本地 (source, page)；PubMed PMID）。
    DOI/摘要引用无法归属到具体子任务，全局核查已通过就保留。
    没引用的句子没法归属到已核实的子任务，去掉。
    过滤完是空的就原样返回（防御性：能进 partial 说明至少一个子任务已核实，
    而草稿的引用都已通过全局核查，正常到不了这里）。"""
    sentences = [s for s in re.split(r"(?<=[。！？；\n])", draft or "") if s.strip()]
    kept = []
    for s in sentences:
        if is_pubmed:
            pmids = PMID_RE.findall(s)
            if pmids:
                if all(p in verified_pmids for p in pmids):
                    kept.append(s)
                continue
            # 没 PMID：DOI 引用或摘要引用（全局已核查通过）就保留，否则去掉
            norm_s = _norm_text(s)
            if DOI_RE.search(s) or any(w and w in norm_s for w, _ in windows):
                kept.append(s)
        else:
            cites = [(m.group(1), m.group(2)) for m in CITE_RE.finditer(s)]
            if cites and all(c in verified_local for c in cites):
                kept.append(s)
    if not kept:
        return draft
    return "".join(kept)


def _partial_note(failed_subtasks):
    """partial 答案末尾的标注：哪些子任务因证据不足没答（用子任务的 query 点名）。"""
    items = [st.get("query", "") for st in failed_subtasks if st.get("query")]
    if not items:
        return "\n\n【部分回答】部分内容因证据不足，本次未作回答。"
    return "\n\n【部分回答】以下内容因证据不足，本次未作回答：" + "；".join(items)


def _retry_failed_subtasks(state, failed_ids):
    """把失败的子任务重置为 pending（只重跑失败的，通过的不动），retry_count +1。
    返回 (更新后的 subtasks 列表, 被重置的 task_id 列表)。
    retry_count 已用完（>= MAX_SUBTASK_RETRIES）的子任务不重置——它没重试额度了。"""
    out = []
    reset = []
    for st in state.get("subtasks") or []:
        s = dict(st)
        if s.get("task_id") in failed_ids and s.get("retry_count", 0) < MAX_SUBTASK_RETRIES:
            s["status"] = "pending"
            s["retry_count"] = s.get("retry_count", 0) + 1
            reset.append(s.get("task_id"))
        out.append(s)
    return out, reset


def _verification_failed(state: State, reason: str, failures, retry_target="answerer") -> dict:
    """核查失败的统一收尾：记失败归因（verify_failures）；还有重试次数就按
    retry_target 回去（answerer 重答 / executor 只重跑失败的子任务），
    用完就拒答——绝不编造。重试时 answer_status 还没定论（不写），
    最终定论时由 verifier 填 verified / partial / refused / error。
    注意：引用编造/缺失走 answerer（生成问题，重答能修）；子任务证据不足
    走 executor（证据问题，重答修不了，得重新检索）——两类失败的重试去向不同。"""
    print(f"[verifier] {reason}")
    retries = state.get("retries", 0)
    if retries < MAX_RETRIES:
        return {"final": "", "retry": True, "retry_target": retry_target,
                "retries": retries + 1, "verify_failures": failures}
    return {"final": REFUSAL, "answer_status": "refused", "retry": False,
            "retry_target": "", "verify_failures": failures,
            **_append_turn_to_history(state, REFUSAL)}


def verifier_node(state: State):
    """核查员（1.5）：引用存在性 + 子任务证据逐项核查 + 失败归因。
    顺序：终态（拒答/LLM 异常/无证据）→ 引用核查 → 子任务核查 → 定论。
    - 引用编造/缺失是生成问题：回 answerer 重答（提醒只引用本次提供的依据）。
    - 子任务证据不足是证据问题：回 executor 只重跑失败的子任务（通过的不动），
      每个子任务最多重试 1 次（复用 1.2 的 retry_count），整体重试上限 2 次。
    - 预算用完：有已核实的子任务 → partial（只留已核实部分，标出未答部分）；
      一个都没通过 → refused。error（LLM 没调通）与 refused（文献没依据）区分。
    注意：核查只看 draft 引用 vs chunks/pubmed_results，不读 history_summary——
    摘要只是 prompt 里的上下文入口，不是用户原话，不参与引用校验。"""
    draft = state.get("draft", "")
    pubmed_results = state.get("pubmed_results") or []
    chunks = state.get("chunks") or []
    subtasks = state.get("subtasks") or []

    # --- 终态：不用重试 ---
    if draft == REFUSAL:
        # 回答员主动拒答：正确收尾，不是失败
        return {"final": draft, "answer_status": "refused", "retry": False,
                "retry_target": "", "verify_failures": [],
                **_append_turn_to_history(state, draft)}
    if draft.startswith("[待接 LLM]"):
        # LLM 没调通：系统异常 error（不是知识不足的 refused），不重试
        failures = [{"subtask_id": "", "reason": "llm_error"}]
        return {"final": draft, "answer_status": "error", "retry": False,
                "retry_target": "", "verify_failures": failures,
                "error": "LLM 调用失败",
                **_append_turn_to_history(state, draft)}
    if not chunks and not pubmed_results:
        # 完全没证据：拒答（回答员这时本来就该返回 REFUSAL；防御性收尾）
        return {"final": REFUSAL, "answer_status": "refused", "retry": False,
                "retry_target": "", "verify_failures": [],
                **_append_turn_to_history(state, REFUSAL)}

    # --- 引用核查（路径不同，规则不同）---
    is_pubmed = bool(pubmed_results)
    if is_pubmed:
        # PubMed 路径：引用接受 PMID、DOI、摘要三种形式，不强制全文页码；
        # 假 PMID、假 DOI、混入本地式引用仍按编造处理。
        real_pmids = {str(r.get("pmid", "")) for r in pubmed_results}
        cited_pmids, cited_dois, abstract_pmids = _pubmed_refs(draft, pubmed_results)
        if list(CITE_RE.finditer(draft)):
            return _verification_failed(
                state, "PubMed 回答里混入了本地式引用",
                [{"subtask_id": "", "reason": "fabricated_citation"}])
        bad_pmids = sorted(cited_pmids - real_pmids)
        if bad_pmids:
            return _verification_failed(
                state, f"引用的 PMID 不在本次 PubMed 返回结果中：{bad_pmids}",
                [{"subtask_id": "", "reason": "fabricated_citation"}])
        bad_dois = sorted(cited_dois - _doi_in_returned(cited_dois, pubmed_results))
        if bad_dois:
            return _verification_failed(
                state, f"引用的 DOI 不在本次 PubMed 返回结果中：{bad_dois}",
                [{"subtask_id": "", "reason": "fabricated_citation"}])
        if not cited_pmids and not cited_dois and not abstract_pmids:
            return _verification_failed(
                state, "回答没有给出引用（PMID/DOI/摘要）",
                [{"subtask_id": "", "reason": "missing_citation"}])
    else:
        # 本地路径：每个 [来源：... p.X] 必须对应一个真实检索片段
        cited = set((m.group(1), m.group(2)) for m in CITE_RE.finditer(draft))
        local_cites, _ = _local_evidence_pool(chunks)
        if not cited:
            return _verification_failed(
                state, "回答没有给出引用",
                [{"subtask_id": "", "reason": "missing_citation"}])
        if not cited.issubset(local_cites):
            bad = sorted(cited - local_cites)
            return _verification_failed(
                state, f"引用不在检索结果中：{bad}",
                [{"subtask_id": "", "reason": "fabricated_citation"}])

    # --- 子任务证据核查（有子任务时才做；简单问答没子任务，引用对了就通过）---
    if not subtasks:
        return {"final": draft, "answer_status": "verified", "retry": False,
                "retry_target": "", "verify_failures": [],
                **_append_turn_to_history(state, draft)}
    local_cites, chunk_sources = _local_evidence_pool(chunks)
    real_pmids = {str(r.get("pmid", "")) for r in pubmed_results}
    verified_ids, sub_failures = _check_subtasks(subtasks, local_cites, chunk_sources, real_pmids)
    if not sub_failures:
        return {"final": draft, "answer_status": "verified", "retry": False,
                "retry_target": "", "verify_failures": [],
                **_append_turn_to_history(state, draft)}

    # 有子任务失败：还能重试就只重跑失败的（通过的不动）；预算用完就定论
    by_id = {st.get("task_id"): st for st in subtasks}
    failed_ids = [f["subtask_id"] for f in sub_failures]
    retryable = [tid for tid in failed_ids
                 if (by_id.get(tid) or {}).get("retry_count", 0) < MAX_SUBTASK_RETRIES]
    if retryable and state.get("retries", 0) < MAX_RETRIES:
        new_subtasks, reset = _retry_failed_subtasks(state, set(retryable))
        print(f"[verifier] 子任务证据不足，只重跑失败的：{reset}")
        return {"final": "", "retry": True, "retry_target": "executor",
                "retries": state.get("retries", 0) + 1,
                "subtasks": new_subtasks, "verify_failures": sub_failures}
    if verified_ids:
        # 部分通过 → partial：只留已核实部分，标出未答的子任务
        failed_sts = [by_id[tid] for tid in failed_ids if tid in by_id]
        verified_local, verified_pmids = _verified_evidence_keys(subtasks, verified_ids)
        windows = _abstract_windows(pubmed_results) if is_pubmed else []
        kept = _filter_partial_draft(draft, verified_local, verified_pmids, is_pubmed, windows)
        final = kept + _partial_note(failed_sts)
        return {"final": final, "answer_status": "partial", "retry": False,
                "retry_target": "", "verify_failures": sub_failures,
                **_append_turn_to_history(state, final)}
    # 一个子任务都没通过：拒答
    return {"final": REFUSAL, "answer_status": "refused", "retry": False,
            "retry_target": "", "verify_failures": sub_failures,
            **_append_turn_to_history(state, REFUSAL)}


def _route_after_verifier(state: State) -> str:
    """核查后的走向：通过（verified/partial）就结束；要求重试就按 retry_target
    回去（executor 只重跑失败的子任务，answerer 重答）；否则结束。"""
    if state.get("answer_status") in ("verified", "partial"):
        return "end"
    if state.get("retry"):
        return state.get("retry_target") or "answerer"
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


def _restore_history_summary(graph, config) -> str:
    """从 checkpointer 里恢复上次保存的历史摘要（1.4，和 history 一起存）。
    没存过、读失败都返回 ""——摘要只是 prompt 上下文，没有也能问答。"""
    try:
        snapshot = graph.get_state(config)
    except Exception as e:
        print(f"[checkpointer] 恢复历史摘要失败（{e}），按无摘要处理")
        return ""
    if not snapshot or not snapshot.values:
        return ""
    return snapshot.values.get("history_summary") or ""


def _build_graph(checkpointer=None):
    from langgraph.graph import StateGraph, END

    graph = StateGraph(State)
    graph.add_node("router", router_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("chart_read", chart_read_node)
    graph.add_node("pubmed", pubmed_node)
    graph.add_node("answerer", answerer_node)
    graph.add_node("verifier", verifier_node)
    graph.set_entry_point("router")
    # router 之后的分流：vision 分支走 chart_read 节点（VLM 读图，
    # 没图可读时节点内部退回 retriever）；planning 走 planner，外部搜索走 pubmed。
    graph.add_conditional_edges("router", _route_after_router, {
        "planner": "planner",
        "chart_read": "chart_read",
        "pubmed": "pubmed",
        "retriever": "retriever",
    })
    graph.add_edge("planner", "executor")
    # executor 执行完，证据已汇总进 chunks / pubmed_results，直接去回答员；
    # 不再经过 retriever（证据已经由子任务的工具调用拿到，不用再查一遍）。
    graph.add_edge("executor", "answerer")
    graph.add_edge("chart_read", "answerer")
    graph.add_conditional_edges("retriever", _route_after_retriever, {"answerer": "answerer", "pubmed": "pubmed"})
    graph.add_edge("pubmed", "answerer")
    graph.add_edge("answerer", "verifier")
    # 1.5：核查失败的重试有两个去向——引用错回 answerer 重答，子任务证据不足
    # 回 executor 只重跑失败的子任务（retry_target 由 verifier 定）。
    graph.add_conditional_edges("verifier", _route_after_verifier, {"answerer": "answerer", "executor": "executor", "end": END})
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
    # verify_failures 是 1.5 的失败归因 [{subtask_id, reason}]，没有失败时为空列表。
    answer_status = result.get("answer_status") or "refused"
    verified = answer_status == "verified"
    verify_failures = result.get("verify_failures") or []
    pubmed_results = result.get("pubmed_results") or []
    if pubmed_results:
        return {
            "answer": result.get("final", REFUSAL),
            "verified": verified,
            "answer_status": answer_status,
            "citations": _dedupe_pubmed_citations(pubmed_results),
            "verify_failures": verify_failures,
        }
    return {
        "answer": result.get("final", REFUSAL),
        "verified": verified,
        "answer_status": answer_status,
        "citations": _dedupe_citations(result.get("chunks", [])),
        "verify_failures": verify_failures,
    }


def _run_sequential(initial: dict) -> dict:
    """LangGraph 不可用时的顺序执行兜底，流程和图里一致（含 Router、PubMed 分支和重试循环）。"""
    state: State = dict(initial)
    state.update(router_node(state))
    dest = _route_after_router(state)
    if dest == "planner":
        state.update(planner_node(state))
        state.update(executor_node(state))
    elif dest == "chart_read":
        state.update(chart_read_node(state))
    elif dest == "pubmed":
        state.update(pubmed_node(state))
    else:
        state.update(retriever_node(state))
        if _route_after_retriever(state) == "pubmed":
            state.update(pubmed_node(state))
    while True:
        state.update(answerer_node(state))
        state.update(verifier_node(state))
        if state.get("answer_status") in ("verified", "partial") or not state.get("retry"):
            break
        if state.get("retry_target") == "executor":
            # 子任务证据不足：只重跑失败的子任务，再回回答员重新汇总
            state.update(executor_node(state))
    return state


def _run_with_graph(query: str, history=None, use_pubmed: bool = True) -> dict:
    """无持久化的问答路径（原来的 run 逻辑）：建图、invoke，
    图不可用时退回顺序执行。
    1.4：传入的历史超限时，被截掉的轮次先摘要进 history_summary（调一次 LLM，
    失败退回规则式），再丢；短历史不截断、不调 LLM，和原来一样。"""
    kept, dropped = memory.normalize_history(history)
    history_summary = memory.append_summary("", dropped, llm_call=_call_llm)
    history = kept
    retrieval_query = _build_retrieval_query(query, history)
    initial = {
        "query": query, "chunks": [], "draft": "", "final": "", "answer_status": "",
        "route": {}, "plan": {}, "subtasks": [], "tool_calls": [], "figure": {}, "error": "",
        "history": history, "history_summary": history_summary, "retrieval_query": retrieval_query,
        "retries": 0, "retry": False, "verify_failures": [], "retry_target": "",
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
                        # 没传 history：从 checkpointer 里恢复上一轮的（历史 + 摘要一起恢复）
                        history = _restore_history(graph, config)
                        existing_summary = _restore_history_summary(graph, config)
                    else:
                        # 传了 history：用传来的，摘要从传入的历史现算（不用库里的）
                        existing_summary = ""
                    kept, dropped = memory.normalize_history(history)
                    history_summary = memory.append_summary(existing_summary, dropped, llm_call=_call_llm)
                    history = kept
                    retrieval_query = _build_retrieval_query(query, history)
                    initial = {
                        "query": query, "chunks": [], "draft": "", "final": "", "answer_status": "",
                        "route": {}, "plan": {}, "subtasks": [], "tool_calls": [], "figure": {}, "error": "",
                        "history": history, "history_summary": history_summary, "retrieval_query": retrieval_query,
                        "retries": 0, "retry": False, "verify_failures": [], "retry_target": "",
                        "pubmed_results": [], "use_pubmed": use_pubmed,
                    }
                    result = graph.invoke(initial, config=config)
                    return _to_response(result)
            except Exception as e:
                print(f"[run] checkpointer 不可用，退回无持久化：{e}")
    return _run_with_graph(query, history, use_pubmed=use_pubmed)
