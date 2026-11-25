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

from .config import CHROMA_DIR, EMBED_MODEL

COLLECTION = "medevidence"

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
    """从问题里抽英文词和数字 token（论文名、术语、年份、数字），去停用词。"""
    return sorted(set(_tokenize(query)))


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
            out.append({"text": c["text"], "source": c["source"], "page": c["page"]})
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
        {"text": doc, "source": meta.get("source", "未知"), "page": meta.get("page", "?")}
        for _, doc, meta in scored[:limit]
    ]

_embedding_fn = None


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


def add_chunks(chunks: list) -> int:
    """把 ingest 切好的片段入库。去重：同一 source+page+chunk_start 只存一次。"""
    if not chunks:
        return 0
    collection = get_collection()
    ids, docs, metas = [], [], []
    for c in chunks:
        cid = f"{c.get('source', 'unknown')}-{c.get('page', '?')}-{c.get('chunk_start', 0)}"
        ids.append(cid)
        docs.append(c["text"])
        metas.append(
            {
                "source": c.get("source", "未知"),
                "page": int(c.get("page", 0)),
                "chunk_start": int(c.get("chunk_start", 0)),
            }
        )
    collection.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def format_citation(chunk: dict) -> str:
    # 统一格式：[来源：论文名 p.页码]
    return f"[来源：{chunk.get('source', '未知')} p.{chunk.get('page', '?')}]"


def retrieve(query: str, top_k: int = 10):
    """混合检索：向量和关键词各占一半名额（默认各 5 个），保证关键词的片段
    一定能进最终结果。库为空返回 []，走拒答分支；不要编造片段。

    注意：不要先拿满 top_k 个向量结果再往后追加关键词——追加的部分会被
    [:top_k] 截掉，关键词兜底等于没跑（2026-11-25 第二轮评测踩过这个坑）。
    """
    try:
        collection = get_collection()
    except Exception as e:
        print(f"[retrieve] Chroma 不可用：{e}")
        return []
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
                {"text": doc, "source": meta.get("source", "未知"), "page": meta.get("page", "?")}
            )
    except Exception as e:
        print(f"[retrieve] 查询失败：{e}")
    # 关键词兜底：补向量漏掉的含论文名 / 术语 / 数字的片段，有固定名额
    tokens = _query_tokens(query)
    chunks.extend(_keyword_chunks(collection, tokens, limit=keyword_k, exclude_ids=seen_ids))
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
    return f"""你是医学文献问答助手。只根据下面提供的文献片段回答，不要编造。

规则：
1. 每条关键结论后面必须标注引用，格式 [来源：论文名 p.页码]
2. 如果文献里没有依据，明确说“根据现有文献无法回答”，不要猜
3. 用简体中文回答，简洁直接
4. 每个引用单独写一个括号，不许把多个引用用分号并在一个括号里；一个结论只标一个最相关的引用

{history_text}文献片段：
{context}

用户问题：{query}
"""
