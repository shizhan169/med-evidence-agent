"""检索 + 引用格式化。核心规则：没有依据就拒答，引用必须带页码。"""
from .config import CHROMA_DIR, EMBED_MODEL

COLLECTION = "medevidence"

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


def retrieve(query: str, top_k: int = 5):
    """Chroma 向量检索。库为空返回 []，走拒答分支；不要编造片段。"""
    try:
        collection = get_collection()
    except Exception as e:
        print(f"[retrieve] Chroma 不可用：{e}")
        return []
    try:
        res = collection.query(query_texts=[query], n_results=top_k)
    except Exception as e:
        print(f"[retrieve] 查询失败：{e}")
        return []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    chunks = []
    for doc, meta in zip(docs, metas):
        chunks.append(
            {"text": doc, "source": meta.get("source", "未知"), "page": meta.get("page", "?")}
        )
    return chunks


def build_prompt(query: str, chunks: list) -> str:
    context = "\n\n".join([f"{c['text']}\n{format_citation(c)}" for c in chunks])
    return f"""你是医学文献问答助手。只根据下面提供的文献片段回答，不要编造。

规则：
1. 每条关键结论后面必须标注引用，格式 [来源：论文名 p.页码]
2. 如果文献里没有依据，明确说“根据现有文献无法回答”，不要猜
3. 用简体中文回答，简洁直接

文献片段：
{context}

用户问题：{query}
"""
