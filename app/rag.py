"""检索 + 引用格式化。核心规则：没有依据就拒答，引用必须带页码。

检索是混合式：向量检索（多语言 embedding）+ 关键词检索（BM25），
再按 RRF（Reciprocal Rank Fusion）按排名融合，不是固定名额拼接。
原因：问题是中文，文献是英文，跨语言向量检索常漏掉含论文名 / 术语 / 数字的
关键片段；BM25 能直接命中这些与语言无关的 token（Bethlehem、MFPR、
CentileBrain、123,984 等）。BM25 是标准算法（词频 + 逆文档频率加权），
索引只建一次并缓存，库扩到 1000 篇也能用；不要用“数 token 个数 + 文件名
加权”这种临时补丁代替它（2026-11-25 复盘：临时补丁每次查询全库扫一遍、
依赖我方的文件名格式、通用 token 一多就同分，扩展性差）。
固定名额拼接（向量 5 + BM25 5）的问题是两路质量不一样时，差的一路也占
一半名额；RRF 按排名加权，两路都靠前的片段分最高，只在一路靠前的也能进，
但分低。RRF 是融合方式，不是重排模型，和 bge-reranker 是两回事；
重排器保持关闭（2026-11-25 已验证重排会误杀正确片段，Q05、Q11）。
"""
import logging
import re

from .config import CHROMA_DIR, EMBED_MODEL, RERANKER_MODEL
from .prompts import load_prompt


class _SuppressChromaTelemetryError(logging.Filter):
    """屏蔽 ChromaDB 遥测报错：ChromaDB 0.5.0 调 posthog.capture() 传 3 个
    位置参数，PostHog 8.x 只接受 1 个，每次建 client / 取集合都会报两条
    “Failed to send telemetry event …”。ChromaDB 自己会捕获异常、只打日志，
    不影响检索和评测；这里只屏蔽这条噪音，不动依赖、不改 ChromaDB 行为。"""

    def filter(self, record):
        return "Failed to send telemetry event" not in record.getMessage()


logging.getLogger("chromadb.telemetry.product.posthog").addFilter(
    _SuppressChromaTelemetryError()
)

# 回答员的 prompt 模板：存在 prompts/answerer/v1.txt，启动时读进来；
# 改 prompt 就改那个文件，不用动代码（模板里的 {history_text}、
# {context}、{query} 是占位符，build_prompt 里填）。
ANSWERER_TEMPLATE = load_prompt("answerer", "v1")

COLLECTION = "medevidence"

# 重排：混合检索先拿一个更大的候选池（向量 15 + BM25 15），cross-encoder
# 再按问题相关性打分，取前 5 给回答员。候选池必须比 5+5 大——重排只能给
# 池子里的片段提名次，不能把池子外的捞进来。池子定 15+15 是实测出来的：
# Q01 的目标片段 BM25 原来排第 6，加图注后图注片段短、BM25 占优，
# 目标片段掉到第 11 名——池子不到 11，重排根本见不到它。这是方案稿里
# “拿 10 个片段重排”的必要修正：池子 30 个、重排后取 5 个。
# 重排模型不可用（没下载、没装 sentence-transformers）时退回旧行为：
# 向量 5 + BM25 5，不重排，检索不断。
RERANK_POOL_PER_SIDE = 15
RERANK_TOP_K = 5

# RRF 融合：score = 1/(k + rank)，rank 从 1 开始；k=60 是 RRF 论文的默认值。
RRF_K = 60
# 每路先取的候选池：RRF 融合需要比 top_k 大的池子才有意义，取 15+15。
RRF_POOL_PER_SIDE = 15
# 上下文组装：返回给回答员的片段总长度预算（字符数），约 10 个片段；
# 超了按相关性顺序截断，排在前面的优先保留。
MAX_CONTEXT_CHARS = 12000

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*|\d+(?:\.\d+)?")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with",
    "is", "are", "was", "were", "by", "at", "from", "as", "et", "al",
    "pdf", "p", "vs", "per", "how", "what", "which", "does", "do",
}

try:
    from rank_bm25 import BM25Okapi

    _BM25_AVAILABLE = True
except Exception:
    BM25Okapi = None
    _BM25_AVAILABLE = False

# BM25 索引缓存：建一次，多次复用；库里片段数变了才重建。
# 注意：重新入库后要重启服务（或重跑评测进程），否则缓存还是旧库的索引。
_BM25_CACHE = {"count": -1, "index": None, "chunks": None}


def _tokenize(text: str) -> list:
    """把文本切成英文词和数字 token（小写、去停用词），BM25 索引和问题共用。"""
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text or "")
        if len(t) >= 2 and t.lower() not in _STOPWORDS
    ]


def _query_tokens(query: str) -> list:
    """从问题里抽英文词和数字 token（论文名、术语、年份、数字），去停用词。
    另外抽图/表引用（"图 4"、"Figure 4"、"Fig. 4"、"表 2"、"Table 2"），
    拼成 "figure4"/"table2" 这样的 token：入库时图注片段的 context 行末尾
    带同名 token（见 _pack_caption），BM25 就能按图号区分图 3 和图 4 的
    图注。只加到 BM25 查询 token 里，向量那一半用原始问题文本，不受影响。
    不改 _tokenize 本身（它动全局）；一位数字只在这里认，不进通用分词。"""
    tokens = set(_tokenize(query))
    for m in re.finditer(r"(?:图|Figure|Fig\.)\s*(\d+)", query or "", re.IGNORECASE):
        tokens.add(f"figure{m.group(1)}")
    for m in re.finditer(r"(?:表|Table)\s*(\d+)", query or "", re.IGNORECASE):
        tokens.add(f"table{m.group(1)}")
    return sorted(tokens)


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# 追问的指代词（和 agent.py 的 FOLLOWUP_MARKERS 一致，那边也用同一组词判断追问）
_FOLLOWUP_MARKERS = ("那", "它", "这个", "那个", "呢")
# 判断一段文本像不像问题：用来校验 LLM 改写/翻译的结果——测试里的假 LLM
# 会对任何 prompt 返回同一段答案文本，不能把它当成改写结果用。
_QUESTION_HINTS = ("？", "?", "吗", "呢", "什么", "几", "多少", "如何", "是否",
                    "是不是", "哪个", "哪些", "为什么", "怎样", "怎么")


def _has_chinese(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _looks_like_question(text: str) -> bool:
    """改写结果像不像一个问题：非空、不太长、带疑问标记、不含引用标记。
    假 LLM 返回的答案（陈述句、带 [来源：...]）会被拦下来，退回拼上文。"""
    t = (text or "").strip()
    if not t or len(t) > 300:
        return False
    if "[来源" in t or "来源：" in t:
        return False
    return any(h in t for h in _QUESTION_HINTS)


def rewrite_query_for_retrieval(query: str, history: list = None, llm_call=None) -> str:
    """多轮追问改写：把带指代词的追问改写成完整独立的问题（指代消解、省略补全），
    改写后的问题才拿去检索。不是追问、没历史时原样返回，不加东西。
    先调 LLM 改写（用历史里的具体实体替换“它/那/这个/那个”，补全省略的部分）；
    改写失败、改出来不像问题（空串、陈述句、带引用的答案）时，退回老做法——
    把上一轮用户问题拼在前面，检索能看到完整意思。llm_call 是调用方传进来的
    LLM（agent.py 传 _call_llm）；不传时只用拼上文的退回做法，不调 LLM。"""
    q = (query or "").strip()
    if not q or not history:
        return q
    if not any(m in q for m in _FOLLOWUP_MARKERS):
        return q
    prev = None
    for msg in reversed(history):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = (msg.get("content") or "").strip()
            if content:
                prev = content
                break
    if not prev:
        return q
    if llm_call is not None:
        try:
            rewritten = llm_call(
                "根据对话历史，把下面的追问改写成一个完整、独立的问题："
                "用历史里提到的具体实体替换追问中的指代词（它/那/这个/那个），"
                "补全省略的部分。只输出改写后的问题，不要解释。\n"
                f"上一轮问题：{prev}\n追问：{q}"
            )
        except Exception as e:
            print(f"[rewrite] 改写失败（{e}），退回拼上文")
            rewritten = ""
        rewritten = (rewritten or "").strip()
        if _looks_like_question(rewritten):
            return rewritten
    return f"{prev} {q}"


def translate_query_to_english(query: str, llm_call=None) -> str:
    """中文问题翻英文检索词：跨语言向量检索弱（Q09 的教训：中文问题对英文论文
    召回差），中文问题先翻成英文再查本地库。只翻中文问题，纯英文原样返回；
    不传 llm_call 时不翻译；翻译失败、翻出来还是中文、翻出来太长，
    都原样返回——检索不断。注意这只是给向量检索用的英文问题，BM25 仍用
    原问题的 token（论文名、数字、图号都在原问题里）；PubMed 路径的 OR
    拓宽也不走这里（_to_pubmed_query 仍拿中文原问题做，见 agent.py）。"""
    q = (query or "").strip()
    if not q or not _has_chinese(q):
        return q
    if llm_call is None:
        return q
    try:
        translated = llm_call(
            "把下面的中文问题翻译成英文问题，只输出英文问题，不要解释：\n" + q
        )
    except Exception as e:
        print(f"[translate] 翻译失败（{e}），用原问题检索")
        return q
    translated = (translated or "").strip()
    if not translated or _has_chinese(translated) or len(translated) > 500:
        return q
    print(f"[translate] 查询翻译：{q} → {translated}")
    return translated


def _get_bm25_index(collection):
    """建 BM25 索引并缓存。索引文档 = 文件名 token + 正文 token。

    文件名也放进索引：问题里提到的论文名（Bethlehem、Fujita）常出现在
    文件名里；文件名是原样用还是我方重命名格式，不影响机制本身——
    原始文件名（如 1-s2.0-...main.pdf）的 token 只是匹配不上，不会出错。
    """
    try:
        count = collection.count()
    except Exception:
        count = -1
    if _BM25_CACHE["index"] is not None and _BM25_CACHE["count"] == count:
        return _BM25_CACHE["index"], _BM25_CACHE["chunks"]
    try:
        data = collection.get(include=["documents", "metadatas"])
    except Exception as e:
        print(f"[bm25] 取全库片段失败：{e}")
        return None, []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    ids = data.get("ids") or []
    corpus, chunks = [], []
    for doc, meta, cid in zip(docs, metas, ids):
        fname = (meta.get("source", "") or "").replace("-", " ").replace("_", " ")
        corpus.append(_tokenize(fname) + _tokenize(doc))
        chunks.append(
            {
                "text": doc,
                "source": meta.get("source", "未知"),
                "page": meta.get("page", "?"),
                "id": cid,
            }
        )
    if not corpus:
        return None, []
    index = BM25Okapi(corpus) if _BM25_AVAILABLE else None
    _BM25_CACHE.update({"count": count, "index": index, "chunks": chunks})
    return index, chunks


def _keyword_chunks(collection, tokens: list, limit: int = 5, exclude_ids=None, source: str = None) -> list:
    """关键词检索：BM25 打分取前 limit 个（跳过向量已召回的）。

    rank_bm25 没装时退回 token 计数兜底（效果差一截，但不断）。
    source 传了就只取那篇论文的片段——多篇点名时按篇取就靠它。
    """
    if not tokens:
        return []
    exclude_ids = exclude_ids or set()
    if _BM25_AVAILABLE:
        index, chunks = _get_bm25_index(collection)
        if index is None:
            return []
        scores = index.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in ranked:
            if scores[i] <= 0:
                break
            c = chunks[i]
            if source is not None and c["source"] != source:
                continue
            if c["id"] in exclude_ids:
                continue
            # 带上 id：retrieve() 做 RRF 融合时按 id 去重（同一片段在向量和
            # BM25 两路都出现时分数相加）；最终返回前再去掉 id，对外格式不变。
            out.append({"id": c["id"], "text": _strip_context(c["text"]), "source": c["source"], "page": c["page"]})
            if len(out) >= limit:
                break
        return out
    try:
        all_data = collection.get(include=["documents", "metadatas"])
    except Exception as e:
        print(f"[keyword] 取全库片段失败：{e}")
        return []
    docs = all_data.get("documents") or []
    metas = all_data.get("metadatas") or []
    ids = all_data.get("ids") or []
    scored = []
    for doc, meta, cid in zip(docs, metas, ids):
        if cid in exclude_ids:
            continue
        if source is not None and meta.get("source") != source:
            continue
        text = (doc or "").lower()
        hits = sum(1 for t in tokens if t in text)
        if hits:
            scored.append((hits, doc, meta, cid))
    scored.sort(key=lambda x: -x[0])
    return [
        {"id": cid, "text": _strip_context(doc), "source": meta.get("source", "未知"), "page": meta.get("page", "?")}
        for _, doc, meta, cid in scored[:limit]
    ]

_embedding_fn = None

# 重排模型单例：None 表示还没试过加载，False 表示加载失败（不要每次检索都重试），
# 其他值就是加载好的 CrossEncoder。加载失败不抛异常——检索退回旧行为。
_reranker = None


def _get_reranker():
    """懒加载 cross-encoder 重排模型（本地跑，不调 API）。
    模型不存在、没装 sentence-transformers、加载失败，都返回 None，
    调用方退回不重排的旧行为。"""
    global _reranker
    if _reranker is not None:
        return _reranker if _reranker is not False else None
    try:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANKER_MODEL)
    except Exception as e:
        print(f"[rerank] 重排模型不可用（{e}），退回不重排："
              f"请用 ModelScope 下模型到 {RERANKER_MODEL}")
        _reranker = False
        return None
    return _reranker


def _rerank(query: str, chunks: list, top_k: int = RERANK_TOP_K) -> list:
    """cross-encoder 重排：把 (问题, 片段) 一对对喂给模型打相关性分，
    按分从高到低取前 top_k 个。bi-encoder（向量检索）只看问题和片段
    各自的向量，重排是让模型同时看问题和片段全文再打分，所以更准——
    代价是慢，只能对几十个候选片段用，不能对全库用。"""
    if not chunks:
        return []
    reranker = _get_reranker()
    if reranker is None:
        return chunks
    try:
        scores = reranker.predict([(query, c["text"]) for c in chunks])
    except Exception as e:
        print(f"[rerank] 打分失败（{e}），用原顺序返回")
        return chunks
    ranked = sorted(range(len(chunks)), key=lambda i: -float(scores[i]))
    return [chunks[i] for i in ranked[:top_k]]


def get_embedding_function():
    """懒加载 embedding，避免 import 时就下载模型。"""
    global _embedding_fn
    if _embedding_fn is None:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        _embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return _embedding_fn


def get_collection():
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        COLLECTION, embedding_function=get_embedding_function()
    )


def _strip_context(text: str) -> str:
    """去掉检索用上下文行（入库时贴在片段前面的那行），回答员只看原文。
    没有上下文行的旧片段原样返回。"""
    if (text or "").startswith("【文献："):
        _, _, rest = text.partition("\n")
        return rest
    return text


def _chunk_ids(chunks: list) -> list:
    """给每个片段一个稳定唯一的入库 ID。

    只用 source-page-chunk_start 不够：按句子切分时，相邻片段会重叠——
    上一个片段如果只有一个句子，下一个片段会从同一句开头，
    两个片段的 chunk_start 相同，Chroma 整批 upsert 直接报 DuplicateIDError、
    一个片段都写不进去（2026-11-25 第四轮评测 0/20 就是这么来的：85 个重复 ID，
    4220 个片段全没入库，库是空的，评测 20 道题全拒答）。页内序号 k 保证唯一；
    同一份数据重复入库时 ID 不变，upsert 仍能按 ID 去重覆盖。
    """
    per_page = {}
    ids = []
    for c in chunks:
        key = (c.get("source", "unknown"), c.get("page", "?"))
        k = per_page.get(key, 0)
        per_page[key] = k + 1
        ids.append(f"{c.get('source', 'unknown')}-{c.get('page', '?')}-{c.get('chunk_start', 0)}-{k}")
    return ids


def add_chunks(chunks: list, batch_size: int = 500) -> int:
    """把 ingest 切好的片段入库。分批 upsert：一批出问题不会把整库
    4220 个片段全拖下水；同一 source+page+chunk_start+页内序号只存一次。"""
    if not chunks:
        return 0
    collection = get_collection()
    ids = _chunk_ids(chunks)
    # Contextual Retrieval 便宜版：入库的文档是“上下文 + 原文”，embedding 一起算，
    # 检索能分清哪篇论文、哪一页、哪个部分；回答时由 retrieve() 把上下文行剥掉，
    # 模型只看到原文，引用也不变。旧片段没有 context 字段，原样入库。
    docs = [
        (c.get("context") + "\n" + c["text"]) if c.get("context") else c["text"]
        for c in chunks
    ]
    metas = [
        {
            "source": c.get("source", "未知"),
            "page": int(c.get("page", 0)),
            "chunk_start": int(c.get("chunk_start", 0)),
            # type 区分正文和图注/表注（caption），默认 text；检索时可用它过滤
            "type": c.get("type", "text"),
        }
        for c in chunks
    ]
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=docs[i:i + batch_size],
            metadatas=metas[i:i + batch_size],
        )
        print(f"[add_chunks] 已入库 {min(i + batch_size, len(ids))}/{len(ids)} 个片段")
    return len(ids)


def format_citation(chunk: dict) -> str:
    # 统一格式：[来源：论文名 p.页码]
    return f"[来源：{chunk.get('source', '未知')} p.{chunk.get('page', '?')}]"


def _vector_chunks(collection, query: str, limit: int, where: dict = None) -> list:
    """向量检索：按问题向量取前 limit 个片段。失败返回 []，检索不断。
    返回的片段带 "id"（RRF 融合按 id 去重），最终返回前再去掉。
    where 是 Chroma 的 metadata 过滤（比如 {"source": "2022-bethlehem-brain-charts.pdf"}），
    传了就只在那篇论文里查——多篇点名时按篇取就靠它。"""
    try:
        kwargs = {"query_texts": [query], "n_results": limit}
        if where:
            kwargs["where"] = where
        res = collection.query(**kwargs)
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        return [
            {"id": cid, "text": _strip_context(doc),
             "source": meta.get("source", "未知"), "page": meta.get("page", "?")}
            for doc, meta, cid in zip(docs, metas, ids)
        ]
    except Exception as e:
        print(f"[retrieve] 查询失败：{e}")
        return []


def _rrf_fuse(ranked_lists: list, k: int = RRF_K, top_k: int = 10) -> list:
    """RRF 融合：多路按排名融合，score = 1/(k + rank)，rank 从 1 开始；
    同一片段在多路都出现时分数相加。按分从高到低取前 top_k 个。
    输入是每路排好序的片段列表（每个片段带 "id"）；输出去重后的片段
    （保留 "id"，调用方按需去掉）。并列时先按各路最好排名、再按首次
    出现顺序，保证结果确定。"""
    scores = {}
    best_rank = {}
    first_seen = {}
    chunks_by_id = {}
    order = 0
    for lst in ranked_lists:
        for rank, c in enumerate(lst, start=1):
            cid = c.get("id")
            if cid is None:
                continue
            if cid not in chunks_by_id:
                chunks_by_id[cid] = c
                first_seen[cid] = order
                order += 1
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            best_rank[cid] = min(best_rank.get(cid, rank), rank)
    ranked = sorted(scores, key=lambda cid: (-scores[cid], best_rank[cid], first_seen[cid]))
    return [chunks_by_id[cid] for cid in ranked[:top_k]]


def _assemble_context(chunks: list, max_chars: int = MAX_CONTEXT_CHARS) -> list:
    """上下文组装：去重 + 长度预算。
    去重：同一 (source, page, text) 只留第一次出现的（RRF 已按 id 去重，
    这里防的是不同 id 但内容相同的片段）。
    预算：总长度超了就按相关性顺序截断，排在前面的优先保留；第一个片段
    再长也保留（不能返回空）。
    相邻证据补充还没做（2.1 明确可选）：要做时按 chunk_start 找相邻片段。"""
    seen = set()
    out = []
    total = 0
    for c in chunks:
        key = (c.get("source"), c.get("page"), c.get("text"))
        if key in seen:
            continue
        seen.add(key)
        n = len(c.get("text") or "")
        if out and total + n > max_chars:
            break
        out.append(c)
        total += n
    return out


def _drop_ids(chunks: list) -> list:
    """去掉 RRF 融合用的内部 "id"，对外返回格式和原来一样
    （只有 text / source / page）。"""
    return [{k: v for k, v in c.items() if k != "id"} for c in chunks]


def _norm_key(text: str) -> str:
    """归一化成“去重音、小写、只剩字母数字”的 key，供论文名匹配用。
    Di Biase→dibiase、Coupé→coupe、Rehak-Buckova→rehakbuckova。"""
    import unicodedata
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", t.lower())


def _source_author_year(source: str):
    """从源文件名解析 (年份, 作者 key)。
    文件名一般是 YYYY-author-....pdf：年份是第一个 token，作者是年份后
    第一个 token（Di Biase 在文件名里连写成 dibiase，Coupé 去重音成 coupe，
    Rehak-Buckova 的第一个 token rehak 已足够区分）。
    解析不到返回 (None, None)。"""
    base = (source or "").lower()
    if base.endswith(".pdf"):
        base = base[:-4]
    parts = [p for p in re.split(r"[-_]", base) if p]
    if not parts or not re.fullmatch(r"\d{4}", parts[0]):
        return None, None
    year = parts[0]
    if len(parts) < 2:
        return year, None
    return year, _norm_key(parts[1]) or None


def _named_papers(query: str, sources: list) -> list:
    """从问题里找出被点名的论文（返回源文件名列表，去重）。
    规则：作者 key 长度 >= 4 时，出现在问题里即算点名；
    短作者名（ge/ho/kia 这类，长度 < 4）必须作者和年份同时出现在问题里，
    否则 "ge" 会误匹配 general/genetic 这类词。
    同一 (作者, 年份) 有多篇时全算上（比如 Rutherford 2022 有脑生长和
    精神病学框架两篇），检索层都取，回答员按内容区分。"""
    q_norm = _norm_key(query)
    q_tokens = set(_tokenize(query))
    named = []
    for src in sources or []:
        year, author = _source_author_year(src)
        if not author:
            continue
        hit = False
        if len(author) >= 4:
            if author in q_tokens or author in q_norm:
                hit = True
        else:
            if year and author in q_tokens and year in q_tokens:
                hit = True
        if hit and src not in named:
            named.append(src)
    return named


def _distinct_sources(collection) -> list:
    """库里所有源文件名（去重、排序）。
    BM25 缓存里已有全库片段，直接用它；BM25 不可用时退回 collection.get。"""
    try:
        _, chunks = _get_bm25_index(collection)
    except Exception:
        chunks = []
    if chunks:
        return sorted({c.get("source", "") for c in chunks if c.get("source")})
    try:
        data = collection.get(include=["metadatas"])
    except Exception:
        return []
    return sorted({
        (m or {}).get("source", "")
        for m in (data.get("metadatas") or [])
        if (m or {}).get("source")
    })


_COMPARISON_DIMENSIONS = (
    (r"数据集|队列|样本|人群|纵向设计|\b(?:datasets?|cohorts?|sample size|study population)\b",
     "What was the total study population? Total number of participants or healthy individuals "
     "included in the study, number of datasets, age range and sex distribution."),
    (r"变量|暴露|结局定义|结局指标|\b(?:variables?|predictors?|covariates?|outcome measures?)\b",
     "What target measure did the models predict from the input features? Which predictors "
     "and covariates were included during model training and prediction?"),
    (r"模型|算法|常模构建方法|建模方法|\b(?:statistical models?|algorithms?|modelling methods?|modeling methods?)\b",
     "Which statistical algorithms and regression models were trained and compared for normative modelling?"),
    (r"结果(?!变量)|常模发现|关键估计|主要发现|关键发现|\b(?:results?|findings?|predictive performance)\b",
     "What were the main findings and results for the predictive performance and accuracy of the models?"),
    (r"局限|偏倚|外部效度限制|未解决的?问题|\b(?:limitations?|biases|external validity)\b",
     "What limitations did the authors report about the study sample, generalisability, and longitudinal follow-up?"),
)


def _comparison_content_query(query: str, named: list) -> str:
    """点名只用于选论文。给单一维度补英文检索词，同时保留原问的具体限定。"""
    citations = {author + year for year, author in map(_source_author_year, named) if author and year}
    def strip_citation(match: re.Match) -> str:
        # 只删与文件对应的作者年份，保留英文作者前的限定词，如 "ABCD in Ge 2024"。
        phrase = match[0]
        for word in re.finditer(r"[A-Za-zÀ-ž]+", phrase):
            if _norm_key(phrase[word.start():]) in citations:
                return phrase[:word.start()]
        return phrase

    subject = re.sub(r"[A-Za-zÀ-ž]+(?:[ -][A-Za-zÀ-ž]+)*\s*[(（]?\d{4}(?!\d)[)）]?", strip_citation, query)
    dimensions = [prompt for terms, prompt in _COMPARISON_DIMENSIONS if re.search(terms, subject, re.I)]
    if len(dimensions) == 1:
        subject = subject.strip(" 、，,和与")
        return f"{dimensions[0]} {subject}"
    return ""


def _retrieve_per_paper(collection, vector_query: str, tokens: list, named: list, top_k: int) -> list:
    """多篇点名时的检索：每篇各取一份证据再合并。
    每篇：向量（where 过滤）+ BM25（source 过滤）各取一个候选池，
    RRF 融合取 per_paper_k 个；各篇结果拼起来，调用方再按原有逻辑
    组装（去重 + 12000 字预算截断）。per_paper_k 保证每篇至少 2 条，
    总数和 top_k 差不多。某篇取不到时返回空，不硬凑。"""
    per_paper_k = max(2, top_k // max(1, len(named)))
    all_chunks = []
    for src in named:
        v = _vector_chunks(collection, vector_query, limit=RRF_POOL_PER_SIDE, where={"source": src})
        k = _keyword_chunks(collection, tokens, limit=RRF_POOL_PER_SIDE, source=src)
        fused = _rrf_fuse([v, k], top_k=per_paper_k)
        all_chunks.extend(fused)
    return all_chunks


def retrieve(query: str, top_k: int = 10, llm_call=None):
    """混合检索：Query 改写（调用方在 _build_retrieval_query 里做）→
    向量 + BM25 各取一个候选池 → RRF 融合 → (重排) → 组装去重 → 返回。

    向量用英文检索词：中文问题先翻成英文再查（跨语言向量检索弱，Q09 的
    教训）；BM25 用原问题的 token（论文名、数字、图号都在原问题里）。
    翻译失败时用原问题查向量，不抛异常。llm_call 是调用方传进来的 LLM
    （retriever_node 传 _call_llm）；不传时不翻译，只用原问题检索。
    库为空返回 []，走拒答分支；不要编造片段。

    多篇点名时按篇取（2026-11-28，B 方案）：问题里点了 2 篇以上论文时，
    每篇各取一份证据再合并，防止同时提到多篇的综述把各篇原文挤出前 10
    （五篇对比表就是这么被综述淹掉的）。只点一篇或没点名时走原有逻辑。

    重排器保持关闭：_get_reranker() 返回 None 时不重排（生产现状，
    RERANKER_MODEL 指向不存在的路径）；万一哪天模型可用了，仍按老路
    对 RRF 融合后的池子重排取前 5。"""
    try:
        collection = get_collection()
    except Exception as e:
        print(f"[retrieve] Chroma 不可用：{e}")
        return []
    # 多篇点名时按篇取：每篇各取一份证据，防止综述挤掉原文。
    # 单篇或没点名时走原有逻辑，行为不变。
    named = _named_papers(query, _distinct_sources(collection))
    # 已按文件过滤时，两路都用维度内容排名，避免作者和年份把参考文献推到前面。
    content_query = _comparison_content_query(query, named) if len(named) >= 2 else ""
    vector_query = content_query or translate_query_to_english(query, llm_call=llm_call)
    tokens = _query_tokens(content_query or query)
    use_rerank = _get_reranker() is not None
    if len(named) >= 2:
        fused = _retrieve_per_paper(collection, vector_query, tokens, named, top_k)
    else:
        # 每路先取一个候选池（15+15），RRF 融合后再取 top_k；
        # 两路独立排名（BM25 不再跳过向量已召回的），同一片段在两路都出现时 RRF 分数相加。
        vector_chunks = _vector_chunks(collection, vector_query, limit=RRF_POOL_PER_SIDE)
        keyword_chunks = _keyword_chunks(collection, tokens, limit=RRF_POOL_PER_SIDE)
        fuse_k = RERANK_POOL_PER_SIDE if use_rerank else top_k
        fused = _rrf_fuse([vector_chunks, keyword_chunks], top_k=fuse_k)
    if use_rerank:
        return _drop_ids(_assemble_context(_rerank(query, fused)))
    return _drop_ids(_assemble_context(fused))


def build_prompt(query: str, chunks: list, history: list = None) -> str:
    """拼回答员的 prompt。history 是最近几轮对话
    [{"role": "user"/"assistant", "content": ...}]，只用来理解追问里的指代
    （比如“那白质呢”里的“白质”指什么）；回答的依据只能是本次提供的
    文献片段，不能引用历史里的页码——历史里的引用可能不在本次检索结果里，
    核查员会按本次片段校验，引用了就会判错。"""
    context = "\n\n".join([f"{c['text']}\n{format_citation(c)}" for c in chunks])
    history_text = ""
    if history:
        lines = []
        for msg in history:
            role = "用户" if msg.get("role") == "user" else "助手"
            lines.append(f"{role}：{msg.get('content', '')}")
        history_text = (
            "对话历史（只用来理解追问里的指代，比如“那白质呢”里的“白质”指什么；\n"
            "回答的引用只能来自下面本次提供的文献片段，不要引用历史里的页码）：\n"
            + "\n".join(lines) + "\n\n"
        )
    return ANSWERER_TEMPLATE.format(history_text=history_text, context=context, query=query)


PUBMED_ANSWERER_TEMPLATE = load_prompt("pubmed_answerer", "v2")  # v2：分两种问法——具体问题严格回答，列文献时把相关论文都列出来（2026-11-26，v1 太严，列文献只剩 1 篇）


def format_pubmed_citation(result: dict) -> str:
    """PubMed 引用的统一格式：PubMed PMID:xxxxxxxx，和本地的
    [来源：论文名 p.页码] 严格区分，答案里一眼看得出这是外部文献。"""
    return f"PubMed PMID:{result.get('pmid', '?')}"


def build_pubmed_prompt(query: str, results: list, history: list = None) -> str:
    """拼 PubMed 回答员的 prompt。results 是 search_pubmed 返回的文章列表，
    每篇只给标题/作者/期刊年份/摘要片段（600 字内），引用格式是
    PubMed PMID:xxxxxxxx。history 只用来理解追问里的指代，引用只能来自
    本次提供的 PubMed 文章，不能引用历史里的 PMID/页码——历史里的引用
    可能不在本次 PubMed 返回结果里，核查员会按本次结果校验，引用了就会判错。"""
    blocks = []
    for r in results:
        blocks.append(
            f"标题：{r.get('title', '')}\n"
            f"作者：{r.get('authors', '')}\n"
            f"期刊：{r.get('journal', '')}（{r.get('year', '')}）\n"
            f"摘要：{r.get('abstract', '')}\n"
            f"引用：{format_pubmed_citation(r)}"
        )
    context = "\n\n".join(blocks)
    history_text = ""
    if history:
        lines = []
        for msg in history:
            role = "用户" if msg.get("role") == "user" else "助手"
            lines.append(f"{role}：{msg.get('content', '')}")
        history_text = (
            "对话历史（只用来理解追问里的指代，比如“那白质呢”里的“白质”指什么；\n"
            "回答的引用只能来自下面本次提供的 PubMed 文献，不要引用历史里的 PMID 或页码）：\n"
            + "\n".join(lines) + "\n\n"
        )
    return PUBMED_ANSWERER_TEMPLATE.format(history_text=history_text, context=context, query=query)
