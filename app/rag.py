"""检索 + 引用格式化。核心规则：没有依据就拒答，引用必须带页码。

检索是混合式：向量检索为主，关键词兜底为辅。
原因：问题是中文，文献是英文，中文 embedding 模型（bge-small-zh）做跨语言
向量检索常漏掉含论文名 / 术语 / 数字的关键片段；关键词能直接命中这些
与语言无关的 token（Bethlehem、MFPR、CentileBrain、123,984 等）。
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


def _query_tokens(query: str) -> list:
    """从问题里抽英文词和数字 token（论文名、术语、年份、数字），去停用词。"""
    tokens = set()
    for t in _TOKEN_RE.findall(query):
        t = t.lower()
        if len(t) < 2 or t in _STOPWORDS:
            continue
        tokens.add(t)
    return sorted(tokens)


def _keyword_chunks(collection, tokens: list, limit: int = 5, exclude_ids=None) -> list:
    """关键词兜底：按 token 命中数给全库片段打分，取前 limit 个（跳过向量已召回的）。"""
    if not tokens:
        return []
    try:
        all_data = collection.get(include=["documents", "metadatas"])
    except Exception as e:
        print(f"[keyword] 取全库片段失败：{e}")
        return []
    docs = all_data.get("documents") or []
    metas = all_data.get("metadatas") or []
    ids = all_data.get("ids") or []
    exclude_ids = exclude_ids or set()
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
    """混合检索：向量 top_k 为主，关键词兜底补足。库为空返回 []，走拒答分支；不要编造片段。"""
    try:
        collection = get_collection()
    except Exception as e:
        print(f"[retrieve] Chroma 不可用：{e}")
        return []
    chunks = []
    seen_ids = set()
    try:
        res = collection.query(query_texts=[query], n_results=top_k)
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
    # 关键词兜底：补向量漏掉的含论文名 / 术语 / 数字的片段
    tokens = _query_tokens(query)
    chunks.extend(_keyword_chunks(collection, tokens, limit=5, exclude_ids=seen_ids))
    return chunks[:top_k]


def build_prompt(query: str, chunks: list) -> str:
    context = "\n\n".join([f"{c['text']}\n{format_citation(c)}" for c in chunks])
    return f"""你是医学文献问答助手。只根据下面提供的文献片段回答，不要编造。

规则：
1. 每条关键结论后面必须标注引用，格式 [来源：论文名 p.页码]
2. 如果文献里没有依据，明确说“根据现有文献无法回答”，不要猜
3. 用简体中文回答，简洁直接
4. 每个引用单独写一个括号，不许把多个引用用分号并在一个括号里；一个结论只标一个最相关的引用

文献片段：
{context}

用户问题：{query}
"""
