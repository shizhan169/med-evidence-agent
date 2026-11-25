"""检索级验证：切分改动有没有让答案更容易被检索到。

不需要 LLM、不需要 Chroma、不需要 embedding 模型，只用 pypdf + rank-bm25，
所以在本机（没有 API Key 的机器）也能跑。完整评测仍需用户在本地跑
python eval/run_eval.py，那才是最终数字。

用法（仓库根目录）：
    python eval/check_retrieval.py

查三件事：
1. 目标题（Q04/Q06/Q08/Q09）：要的数字和上下文是不是在同一个片段里。
   只有同框，检索才可能一次命中；数字和结论分在两个片段里，模型只拿到
   结论片段就答不出数字。
2. 回归：15 道可回答题里，新切分有没有丢掉旧切分已有的探针（预期答案
   里的数字/英文术语）；丢了就说明新切分切掉了内容。
3. BM25 排序：有英文 token 的题，预期来源页的片段在新切分下还能不能
   进 BM25 前 5（旧切分作对照）。纯中文题没有英文 token，BM25 帮不上，
   这里记为 n/a；它们的向量检索效果要等用户本地跑完整评测看。

退出码：4 道目标题在新切分下都同框、且没有探针丢失，返回 0；否则返回 1。
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ingest import CHUNK_OVERLAP, CHUNK_SIZE, chunk_pages, parse_pdf  # noqa: E402
from app.rag import _query_tokens, _tokenize  # noqa: E402

try:
    from rank_bm25 import BM25Okapi

    _BM25_AVAILABLE = True
except Exception:
    BM25Okapi = None
    _BM25_AVAILABLE = False

QUESTIONS_FILE = ROOT / "eval" / "test_questions.md"
PAPERS_DIR = ROOT / "data" / "papers"

# 和 eval/run_eval.py 同一套题库格式：### 题号 + “字段：值”行
# 题号含 F/FR（图表题）时也解析，但本工具只分析原 15 道可回答题：
# 图表题测的是图注检索，不是切分回归，BM25 基线不一样，不混进这里的数字。
BLOCK_RE = re.compile(r"^###\s+(Q\d+|R\d+|FR\d+|F\d+)\s*$", re.M)
FIELD_RE = re.compile(r"^(问题|类型|预期答案|预期来源|拒答原因)：(.*)$")
SOURCE_RE = re.compile(r"^(.+\.pdf)\s+p\.(.+)$")

# 目标题：单片段必须同时含有的内容（数字 + 上下文词），来自评测题库的预期答案
TARGETS = {
    "Q04": {"file": "2023-dibiase-mapping-brain-charts.pdf", "pages": [2],
            "need": ["44%", "37%", "35%", "26%"]},
    "Q06": {"file": "2023-dibiase-mapping-brain-charts.pdf", "pages": [3],
            "need": ["18%", "7%", "4%", "2%", "naive"]},
    "Q08": {"file": "2023-fujita-brain-volume-aging.pdf", "pages": [4, 8],
            # 注意是 whole-brain（带连字符）：原文是 “an annual decrease of –0.4%
            # in whole-brain volume”；p.8 的 “Whole brain A” 只是图注里的坐标轴标签，
            # 不能算数
            "need": ["0.4%", "whole-brain"]},
    "Q09": {"file": "2023-fujita-brain-volume-aging.pdf", "pages": [4, 8],
            "need": ["1.8%", "ventricle"]},
}

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*%?")
_WORD_RE = re.compile(r"[A-Za-z]{4,}")


def parse_questions(path: Path) -> list:
    """解析 test_questions.md（和 eval/run_eval.py 同一套格式）。"""
    text = path.read_text(encoding="utf-8")
    blocks = [(m.group(1), m.start()) for m in BLOCK_RE.finditer(text)]
    questions = []
    for i, (qid, start) in enumerate(blocks):
        end = blocks[i + 1][1] if i + 1 < len(blocks) else len(text)
        fields = {}
        for line in text[start:end].splitlines():
            m = FIELD_RE.match(line.strip())
            if m:
                fields[m.group(1)] = m.group(2).strip()
        q = {"id": qid, "question": fields.get("问题", ""),
             "type": fields.get("类型", ""),
             "expected_answer": fields.get("预期答案", ""),
             "expected_source": fields.get("预期来源", "")}
        m = SOURCE_RE.match(q["expected_source"])
        if m:
            q["expected_file"] = m.group(1)
            q["expected_pages"] = [int(n) for n in re.findall(r"\d+", m.group(2))]
        else:
            q["expected_file"] = ""
            q["expected_pages"] = []
        questions.append(q)
    return questions


def old_chunk_pages(pages):
    """改动前的固定网格切分（800 字窗、100 字步进），仅供对照。"""
    chunks = []
    for p in pages:
        text = p["text"]
        for start in range(0, len(text), CHUNK_SIZE - 100):
            piece = text[start:start + CHUNK_SIZE].strip()
            if len(piece) < 50:
                continue
            chunks.append({**p, "text": piece, "chunk_start": start})
    return chunks


def probes_of(expected_answer: str) -> set:
    """从预期答案里抽探针：百分比/带逗号的数字 + 长度>=4 的英文词。"""
    probes = {m.group(0).replace(" ", "") for m in _NUMBER_RE.finditer(expected_answer)}
    probes |= {w.lower() for w in _WORD_RE.findall(expected_answer)}
    # 太短的纯数字（如 70、8）到处都是，不作探针
    return {p for p in probes if not p.isdigit() or len(p) >= 4}


def _chunk_has(chunk_text: str, needle: str) -> bool:
    return needle.lower() in chunk_text.lower()


def main() -> int:
    # 只看原 15 道可回答题：图表题（F/FR）测图注检索，不参与切分回归检查
    questions = [q for q in parse_questions(QUESTIONS_FILE)
                 if q["type"] == "可回答" and not q["id"].startswith("F")]
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    print(f"解析 {len(pdfs)} 篇 PDF，对照旧切分和新切分……")
    old_chunks, new_chunks = [], []
    for pdf in pdfs:
        pages = parse_pdf(pdf)
        old_chunks.extend(old_chunk_pages(pages))
        new_chunks.extend(chunk_pages(pages))
    print(f"旧切分：{len(old_chunks)} 个片段；新切分：{len(new_chunks)} 个片段")
    print()

    # ---- 1. 目标题：数字和上下文是否同框 ----
    print("== 目标题：单片段同框检查 ==")
    target_ok = True
    for qid, spec in TARGETS.items():
        for label, chunks in [("旧", old_chunks), ("新", new_chunks)]:
            hit = [c for c in chunks
                   if c["source"] == spec["file"] and c["page"] in spec["pages"]
                   and all(_chunk_has(c["text"], n) for n in spec["need"])]
            mark = "通过" if hit else "不通过"
            print(f"  {qid} [{label}] {mark}：要 {spec['need']}，"
                  f"同框片段 {len(hit)} 个")
            if label == "新" and not hit:
                target_ok = False
    print()

    # ---- 2. 回归：新切分有没有丢探针 ----
    print("== 回归：预期答案探针是否还在 ==")
    lost_any = False
    for q in questions:
        probes = probes_of(q["expected_answer"])
        if not probes:
            print(f"  {q['id']}：无可用探针，跳过")
            continue
        page_chunks_old = [c for c in old_chunks if c["source"] == q["expected_file"]
                           and c["page"] in q["expected_pages"]]
        page_chunks_new = [c for c in new_chunks if c["source"] == q["expected_file"]
                           and c["page"] in q["expected_pages"]]
        found_old = {p for p in probes
                     if any(_chunk_has(c["text"], p) for c in page_chunks_old)}
        found_new = {p for p in probes
                     if any(_chunk_has(c["text"], p) for c in page_chunks_new)}
        lost = found_old - found_new
        if lost:
            lost_any = True
        print(f"  {q['id']}：探针 {len(probes)} 个，旧切分命中 {len(found_old)}，"
              f"新切分命中 {len(found_new)}" + (f"；丢失 {sorted(lost)}" if lost else ""))
    print()

    # ---- 3. BM25 前 5：预期来源页在不在 ----
    print("== BM25 前 5：预期来源页是否在列 ==")
    if not _BM25_AVAILABLE:
        print("  rank-bm25 未安装，跳过（装上后重跑即可）")
    else:
        for label, chunks in [("旧", old_chunks), ("新", new_chunks)]:
            corpus = [_tokenize(c["source"].replace("-", " ")) + _tokenize(c["text"])
                      for c in chunks]
            index = BM25Okapi(corpus)
            ok = na = 0
            for q in questions:
                tokens = _query_tokens(q["question"])
                if not tokens:
                    na += 1
                    continue
                scores = index.get_scores(tokens)
                top = sorted(range(len(scores)), key=lambda i: -scores[i])[:5]
                if any(chunks[i]["source"] == q["expected_file"]
                       and chunks[i]["page"] in q["expected_pages"] for i in top):
                    ok += 1
            print(f"  [{label}] 预期页进前 5：{ok}/{len(questions) - na} 道"
                  f"（{na} 道纯中文题无英文 token，BM25 不适用）")
    print()

    # ---- 4. 切分质量：干净开头、断词残留 ----
    print("== 切分质量 ==")
    for label, chunks in [("旧", old_chunks), ("新", new_chunks)]:
        # 干净开头：片段以大写字母/数字/中文开头（像句子开头，而不是半句话）
        clean = sum(1 for c in chunks
                    if c["text"][:1].isupper() or c["text"][:1].isdigit()
                    or "\u4e00" <= c["text"][:1] <= "\u9fff")
        broken = sum(c["text"].count("-\n") for c in chunks)
        print(f"  [{label}] 干净开头 {clean}/{len(chunks)}（{clean / len(chunks):.0%}），"
              f"行尾连字符断词残留 {broken} 处")
    print()

    ok = target_ok and not lost_any
    print("结论：" + ("通过：4 道目标题同框、无探针丢失" if ok
                   else "不通过：见上面标出的项"))
    print("注意：这是检索级验证（片段里有没有、排不排得到），不是完整评测；")
    print("向量检索的实际效果要等用户本地跑 python eval/run_eval.py 看数字。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
