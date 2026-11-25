"""入库 ID 唯一性回归测试：2026-11-25 第四轮评测 0/20 的根因。

按句子切分时，相邻片段会重叠——上一个片段如果只有一个句子，
下一个片段会从同一句开头，两个片段的 chunk_start 相同；
只用 source-page-chunk_start 做 ID 会在 Chroma upsert 时报 DuplicateIDError，
整批 4220 个片段一个都写不进去，库是空的，评测 20 道题全拒答。
"""
from pathlib import Path

import pytest

from app.ingest import chunk_pages
from app.rag import _chunk_ids


def _sent(n_chars: int) -> str:
    """造一个指定长度的单句：句中没有句号，不会被分句切开；
    以大写字母开头，句号后接大写才算分句点。"""
    return ("Word " * ((n_chars - 1) // 5 + 1))[:n_chars - 1] + "."


def test_single_sentence_chunk_gets_unique_ids():
    # 第一句 700 字、第二句 150 字、第三句 150 字：
    # 第一句自己装不满 800，第二句一加就超 -> 第一个片段只有第一句；
    # 重叠把第一句带进第二个片段，两个片段 chunk_start 都是 0。
    text = _sent(700) + " " + _sent(150) + " " + _sent(150)
    chunks = chunk_pages([{"page": 1, "text": text, "source": "test.pdf"}])
    assert len(chunks) >= 2
    starts = [c["chunk_start"] for c in chunks]
    assert len(set(starts)) < len(starts), "这组句子没能复现 chunk_start 重复，先调句子长度"
    ids = _chunk_ids(chunks)
    assert len(set(ids)) == len(ids)


def test_real_papers_have_unique_ids():
    # 真实文献上的全量检查：26 篇论文 4220 个片段，之前有 85 个重复 ID。
    # 跑得慢（约 1 分钟，pypdf 要逐页解析），只在改切分或入库逻辑后跑。
    papers_dir = Path(__file__).resolve().parents[1] / "data" / "papers"
    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        pytest.skip("data/papers 里没有 PDF，跳过真实文献检查")
    from app.ingest import parse_pdf

    chunks = []
    for pdf in pdfs:
        chunks.extend(chunk_pages(parse_pdf(pdf)))
    ids = _chunk_ids(chunks)
    assert len(set(ids)) == len(ids) == len(chunks)
