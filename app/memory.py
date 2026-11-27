"""多轮对话的历史管理 + 上下文压缩（框架 v2，第一阶段 1.4）。

从 app/agent.py 抽出：历史清洗、截断、压缩摘要。

两个层次，别混淆：
- state["history"]：最近 6 轮 / 4000 字的完整历史（旧 _normalize_history 的
  行为，一字没动）。checkpointer 持久化它，压缩不影响它——断点续聊靠它。
- state["history_summary"]：较早轮次的压缩摘要。历史超限时，被截断丢掉的
  轮次先摘要再丢（摘要写进 State，不丢信息）；prompt 只拼最近 2 轮完整内容
  + 摘要。
- 摘要优先调 LLM；LLM 失败（没传、抛异常、返回空、返回的不是摘要）时退回
  规则式压缩（每轮保留用户问题 + 回答的第一句关键结论句），不断问答。
- prompt 里的摘要分两段拼：持久化摘要（被截掉的轮次，LLM 摘要）+ 中间轮次
  的规则式摘要（kept 历史里除最近 2 轮外的轮次，还没被截掉、所以不在持久化
  摘要里；规则式是确定性的、不调 LLM，每次按当前历史现算——这些轮次以后
  被截掉时还会再进持久化摘要走一次 LLM，不会漏）。
"""
import re

# 历史上限（沿用旧值）：最近 6 轮（一轮 = 一问一答），总字数不超过 4000 字。
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 4000
# 压缩后保留的完整轮次：最近 2 轮。
KEEP_RECENT_TURNS = 2
# LLM 摘要太长就截断，prompt 不能被摘要撑爆。
MAX_SUMMARY_CHARS = 2000

# 句子切分：取回答的第一句当“关键结论句”（规则式压缩用）。
_SENTENCE_SPLIT_RE = re.compile(r"[。！？；\n]")


def _split_turns(history):
    """把消息列表按轮切分（每轮 = user + assistant 两条；不成对的单独成轮）。"""
    turns = []
    current = []
    for msg in history:
        if msg.get("role") == "user" and current:
            turns.append(current)
            current = []
        current.append(msg)
    if current:
        turns.append(current)
    return turns


def clean_history(history):
    """清洗：只要 user/assistant 两种角色，空内容丢掉；开头是 assistant
    （上一轮的回答没配上问题）时丢掉它，保证整轮。只清洗，不截断。"""
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
    if cleaned and cleaned[0]["role"] == "assistant":
        cleaned = cleaned[1:]
    return cleaned


def normalize_history(history):
    """清洗 + 截断（沿用旧 _normalize_history 的行为，一字没动）：
    只留最近 MAX_HISTORY_TURNS 轮；总字数超 MAX_HISTORY_CHARS 就从最早的
    一轮开始丢。返回 (kept, dropped)：kept 是留下的历史，dropped 是被截掉
    的消息（调用方拿去摘要，别直接丢）。kept 一定是 cleaned 的后缀，
    所以 dropped = cleaned[:len(cleaned) - len(kept)]。"""
    cleaned = clean_history(history)
    # 只留最近 N 轮；开头如果是 assistant（上一轮的回答没配上问题），丢掉它
    keep = cleaned[-MAX_HISTORY_TURNS * 2:]
    if keep and keep[0]["role"] == "assistant":
        keep = keep[1:]
    # 总字数超了就从最早的一轮开始丢（整轮丢，不留半轮）
    while len(keep) > 2 and sum(len(m["content"]) for m in keep) > MAX_HISTORY_CHARS:
        keep = keep[2:]
    dropped = cleaned[:len(cleaned) - len(keep)] if len(keep) < len(cleaned) else []
    return keep, dropped


def _first_sentence(text):
    """取一段回答的第一句（规则式压缩里当“关键结论句”）。"""
    for part in _SENTENCE_SPLIT_RE.split(text or ""):
        part = part.strip()
        if part:
            return part
    return (text or "").strip()


def _rule_summary(turns):
    """规则式压缩：每轮保留用户问题 + 回答的第一句（关键结论句）。
    turns 是按轮切分后的轮次列表（每轮是消息列表）。"""
    lines = []
    for turn in turns:
        q = next((m["content"] for m in turn if m.get("role") == "user"), "")
        a = next((m["content"] for m in turn if m.get("role") == "assistant"), "")
        if not q and not a:
            continue
        if a:
            lines.append(f"用户问：{q}；助手答：{_first_sentence(a)}")
        else:
            lines.append(f"用户问：{q}")
    return "\n".join(lines)


def _turns_text(turns):
    """把轮次渲染成 LLM 摘要用的文本。"""
    lines = []
    for turn in turns:
        for m in turn:
            role = "用户" if m.get("role") == "user" else "助手"
            lines.append(f"{role}：{m.get('content', '')}")
    return "\n".join(lines)


def summarize_history(messages, llm_call=None):
    """把若干轮对话压缩成一段摘要。messages 是消息列表（整轮）。
    优先调 LLM 摘要；LLM 失败（没传、抛异常、返回空、返回的不是摘要）
    时退回规则式压缩。返回 "" 表示没有可摘要的内容。"""
    if not messages:
        return ""
    turns = _split_turns(list(messages))
    if not turns:
        return ""
    if llm_call is not None:
        try:
            raw = llm_call(
                "把下面几轮对话压缩成一段摘要，保留其中的关键事实、数字、结论和"
                "用户问过的问题，只输出摘要，不要解释，不要编造没有的内容：\n\n"
                + _turns_text(turns)
            )
        except Exception as e:
            print(f"[memory] LLM 摘要失败（{e}），退回规则式压缩")
            raw = ""
        raw = (raw or "").strip()
        if raw and not raw.startswith("[待接 LLM]") and "[来源：" not in raw:
            if len(raw) > MAX_SUMMARY_CHARS:
                raw = raw[:MAX_SUMMARY_CHARS]
            return raw
        if raw:
            print("[memory] LLM 摘要结果不像摘要，退回规则式压缩")
    return _rule_summary(turns)


def append_summary(existing_summary, dropped_messages, llm_call=None):
    """把新截掉的轮次摘要进已有摘要（追加式，不丢信息）。
    dropped_messages 是刚被截掉的消息列表；为空时原样返回已有摘要，
    不调 LLM（短历史不产生额外调用）。"""
    if not dropped_messages:
        return existing_summary or ""
    new_part = summarize_history(dropped_messages, llm_call)
    if not new_part:
        return existing_summary or ""
    if existing_summary:
        return existing_summary + "\n" + new_part
    return new_part


def compress_history(history, llm_call=None):
    """把一段历史压缩成 prompt 用的上下文：历史超限（超过 6 轮或 4000 字）时，
    较早轮次摘要成一段，只留最近 2 轮完整内容；没超限时原样返回、不摘要。
    返回 (context_history, summary)：context_history 是拼进 prompt 的历史
    （超限时只有最近 2 轮），summary 是较早轮次的摘要（没超限时为 ""）。"""
    cleaned = clean_history(history)
    turns = _split_turns(cleaned)
    total_chars = sum(len(m["content"]) for m in cleaned)
    if len(turns) <= MAX_HISTORY_TURNS and total_chars <= MAX_HISTORY_CHARS:
        return cleaned, ""
    older_turns = turns[:-KEEP_RECENT_TURNS]
    recent_turns = turns[-KEEP_RECENT_TURNS:]
    older_msgs = [m for t in older_turns for m in t]
    recent_msgs = [m for t in recent_turns for m in t]
    summary = summarize_history(older_msgs, llm_call)
    return recent_msgs, summary


def prompt_history(history, history_summary):
    """拼进回答员 prompt 的历史：有摘要时只拼最近 2 轮完整内容 + 摘要，
    摘要以 [此前对话摘要] 开头、作为第一条（不是用户原话，只是上下文）；
    没摘要时原样返回历史。
    history 是 state 里的 kept 历史（已归一化，最多 6 轮），history_summary
    是 State 里的持久化摘要（被截掉轮次的摘要）。kept 历史里除最近 2 轮外
    的较早轮次（还没被截掉、所以不在持久化摘要里）用规则式摘要补进 prompt
    摘要，避免 prompt 里丢信息；规则式是确定性的、不调 LLM，每次按当前
    历史现算。"""
    if not history:
        return []
    if not history_summary:
        return list(history)
    turns = _split_turns(list(history))
    if len(turns) <= KEEP_RECENT_TURNS:
        recent_msgs = list(history)
        older_msgs = []
    else:
        recent_msgs = [m for t in turns[-KEEP_RECENT_TURNS:] for m in t]
        older_msgs = [m for t in turns[:-KEEP_RECENT_TURNS] for m in t]
    middle = _rule_summary(_split_turns(older_msgs)) if older_msgs else ""
    full = history_summary
    if middle:
        full = full + "\n" + middle if full else middle
    summary_entry = {"role": "user", "content": f"[此前对话摘要] {full}"}
    return [summary_entry] + recent_msgs
