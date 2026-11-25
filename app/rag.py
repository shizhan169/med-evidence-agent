"""检索 + 引用格式化。核心规则：没有依据就拒答，引用必须带页码。

检索是混合式：向量检索（多语言 embedding）+ 关键词检索（BM25），各占一半名额。
原因：问题是中文，文献是英文，跨语言向量检索常漏掉含论文名 / 术语 / 数字的
关键片段；BM25 能直接命中这些与语言无关的 token（Bethlehem、MFPR、
CentileBrain、123,984 等）。BM25 是标准算法（词频 + 逆文档频率加权），
索引只建一次并缓存，库扩到 1000 篇也能用；不要用“数 token 个数 + 文件名
加权”这种临时补丁代替它（2026-11-25 复盘：临时补丁每次查询全库扫一遍、
依赖我方的文件名格式、通用 token 一多就同分，扩展性差）。
"""
import re

from .config import CHROMA_DIR, EMBED_MODEL, RERANKER_MODEL
from .prompts import load_prompt

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


def _keyword_chunks(collection, tokens: list, limit: int = 5, exclude_ids=None) -> list:
    """关键词检索：BM25 打分取前 limit 个（跳过向量已召回的）。

    rank_bm25 没装时退回 token 计数兜底（效果差一截，但不断）。
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
            if c["id"] in exclude_ids:
                continue
            out.append({"text": _strip_context(c["text"]), "source": c["source"], "page": c["page"]})
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
        text = (doc or "").lower()
        hits = sum(1 for t in tokens if t in text)
        if hits:
            scored.append((hits, doc, meta))
    scored.sort(key=lambda x: -x[0])
    return [
        {"text": _strip_context(doc), "source": meta.get("source", "未知"), "page": meta.get("page", "?")}
        for _, doc, meta in scored[:limit]
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


def retrieve(query: str, top_k: int = 10):
    """混合检索：向量和关键词各占一半名额，保证关键词的片段
    一定能进最终结果。库为空返回 []，走拒答分支；不要编造片段。

    注意：不要先拿满 top_k 个向量结果再往后追加关键词——追加的部分会被
    [:top_k] 截掉，关键词兜底等于没跑（2026-11-25 第二轮评测踩过这个坑）。

    有重排模型时：候选池放宽到向量 15 + BM25 15（共 30 个），重排后取
    前 RERANK_TOP_K（5）个给回答员；没有重排模型时保持旧行为
    （向量 5 + BM25 5，不重排）。"""
    try:
        collection = get_collection()
    except Exception as e:
        print(f"[retrieve] Chroma 不可用：{e}")
        return []
    use_rerank = _get_reranker() is not None
    if use_rerank:
        vector_k = keyword_k = RERANK_POOL_PER_SIDE
    else:
        vector_k = top_k // 2
        keyword_k = top_k - vector_k
    chunks = []
    seen_ids = set()
    try:
        res = collection.query(query_texts=[query], n_results=vector_k)
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        for doc, meta, cid in zip(docs, metas, ids):
            seen_ids.add(cid)
            chunks.append(
                {"text": _strip_context(doc), "source": meta.get("source", "未知"), "page": meta.get("page", "?")}
            )
    except Exception as e:
        print(f"[retrieve] 查询失败：{e}")
    # 关键词兜底：补向量漏掉的含论文名 / 术语 / 数字的片段，有固定名额
    tokens = _query_tokens(query)
    chunks.extend(_keyword_chunks(collection, tokens, limit=keyword_k, exclude_ids=seen_ids))
    if use_rerank:
        return _rerank(query, chunks)
    return chunks[:top_k]


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
