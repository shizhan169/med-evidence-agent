"""跨文献复杂任务测试，框架 v2 第二阶段 2.2：不需要 LLM、不需要 Chroma、不联网。
跑法：pytest tests/test_crossdoc.py -v

思路：用一个真实跨文献任务，比较五篇纵向常模论文的数据集、建模变量、
模型、结果和局限性，把 Planner-Executor 跑通。Router、Planner、Executor
选工具、回答员都换成假的，只验证跨文献的汇总与核查逻辑；真的五篇对比
表格要等用户本地跑。

任务定义见 docs/crossdoc-five-papers.md。五篇论文都在开发集里。
Planner 按维度拆，数据集、建模变量、模型、结果、局限性各一个子任务，
这是 Planner prompt 的既定规则，不按论文拆。

验证点：
- 完整表格：每行都有引用，引用真实存在，五篇论文都覆盖到。
- 1 个维度证据不足时，输出 partial 而不是全拒答。
- 1 篇论文证据不足时，按论文拆的分解方式，同样输出 partial。
- 回答员汇总时，prompt 里要带上子任务清单，要求覆盖全部子任务，
  否则答案可能悄悄漏掉一个维度，核查还报 verified。
- 表格里出现编造引用时，核查要拦住重答。

注意：不要在测试文件里写全角括号，会触发写入工具丢缩进。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.agent as agent

QUESTION = "比较五篇纵向常模论文的数据集、建模变量、模型、结果和局限性"

BETH = "2022-bethlehem-brain-charts.pdf"
DIB = "2023-dibiase-mapping-brain-charts.pdf"
GE = "2024-ge-centilebrain.pdf"
SEI = "2026-seidel-baseline-cortical-thickness.pdf"
RUT = "2022-rutherford-brain-growth-aging-precision.pdf"
PAPERS = [BETH, DIB, GE, SEI, RUT]


def _chunk(text, source, page):
    return {"text": text, "source": source, "page": page}


# 各维度的假检索片段，内容取自论文原文要点
CHUNKS_BY_DIM = {
    "数据集": [
        _chunk("Bethlehem 2022 汇集 123984 次 MRI 扫描，101457 名受试者，年龄受孕后 115 天到 100 岁。", BETH, 1),
        _chunk("Di Biase 2023 用 UK Biobank 和 ABCD 两个队列。", DIB, 2),
        _chunk("Ge 2024 纳入 37407 名健康人，87 个数据集，年龄 3 到 90 岁。", GE, 1),
        _chunk("Seidel 2026 用 ABCD 研究 6000 多名青少年。", SEI, 1),
        _chunk("Rutherford 2022 基于 58836 人，82 个站点，年龄 2 到 100 岁。", RUT, 3),
    ],
    "建模变量": [
        _chunk("Bethlehem 用年龄、性别、站点做协变量。", BETH, 3),
        _chunk("Ge 比较 8 种算法和多种协变量组合。", GE, 1),
        _chunk("Seidel 用基线皮层厚度、基线年龄和随访年龄。", SEI, 1),
    ],
    "模型": [
        _chunk("Bethlehem 用 GAMLSS 建脑图谱，centile 分数。", BETH, 1),
        _chunk("Ge 选 MFPR 为最优算法。", GE, 6),
        _chunk("Seidel 提出 B-Norm 和 C-Norm。", SEI, 1),
        _chunk("Rutherford 用 GAMLSS 做常模建模。", RUT, 3),
    ],
    "结果": [
        _chunk("Di Biase 发现横断面低估纵向变化，皮层厚度最多低估 44%。", DIB, 2),
        _chunk("Ge 的 MFPR 样本量超 3000 后性能趋于平稳。", GE, 1),
        _chunk("Seidel 的 B-Norm 预测更准，解释方差高约 60%。", SEI, 2),
    ],
    "局限性": [
        _chunk("Bethlehem 承认 MRI 研究相对全球人口多样性有偏差。", BETH, 1),
        _chunk("Di Biase 只用两个队列，年龄覆盖有限。", DIB, 2),
        _chunk("Rutherford 的站点差异仍需 harmonisation。", RUT, 3),
    ],
}

# 按论文分的假检索片段，用于按论文拆的分解方式
CHUNKS_BY_PAPER = {}
for _dim, _cs in CHUNKS_BY_DIM.items():
    for _c in _cs:
        CHUNKS_BY_PAPER.setdefault(_c["source"], []).append(_c)

PLANNER_DIM_JSON = (
    '{"goal": "比较五篇纵向常模论文的数据集、建模变量、模型、结果和局限性", "subtasks": ['
    '{"query": "五篇纵向常模论文的数据集各是什么"}, '
    '{"query": "五篇纵向常模论文的建模变量有什么不同"}, '
    '{"query": "五篇纵向常模论文用的模型各是什么"}, '
    '{"query": "五篇纵向常模论文的结果是什么"}, '
    '{"query": "五篇纵向常模论文的局限性是什么"}]}'
)

PLANNER_PAPER_JSON = (
    '{"goal": "比较五篇纵向常模论文", "subtasks": ['
    '{"query": "Bethlehem 2022 的数据集、建模变量、模型、结果和局限性"}, '
    '{"query": "Di Biase 2023 的数据集、建模变量、模型、结果和局限性"}, '
    '{"query": "Ge 2024 的数据集、建模变量、模型、结果和局限性"}, '
    '{"query": "Seidel 2026 的数据集、建模变量、模型、结果和局限性"}, '
    '{"query": "Rutherford 2022 的数据集、建模变量、模型、结果和局限性"}]}'
)

# 完整对比表格草稿：5 个维度、5 篇论文都有引用
TABLE_DRAFT = (
    "数据集：Bethlehem 123984 次扫描、101457 人 [来源：2022-bethlehem-brain-charts.pdf p.1]；"
    "Di Biase 用 UKB 和 ABCD [来源：2023-dibiase-mapping-brain-charts.pdf p.2]；"
    "Ge 纳入 37407 人、87 个数据集 [来源：2024-ge-centilebrain.pdf p.1]；"
    "Seidel 用 ABCD 6000 多名青少年 [来源：2026-seidel-baseline-cortical-thickness.pdf p.1]；"
    "Rutherford 基于 58836 人、82 个站点 [来源：2022-rutherford-brain-growth-aging-precision.pdf p.3]。\n"
    "建模变量：Bethlehem 用年龄、性别、站点 [来源：2022-bethlehem-brain-charts.pdf p.3]；"
    "Ge 比较 8 种算法和多种协变量 [来源：2024-ge-centilebrain.pdf p.1]；"
    "Seidel 用基线厚度、基线年龄和随访年龄 [来源：2026-seidel-baseline-cortical-thickness.pdf p.1]。\n"
    "模型：Bethlehem 用 GAMLSS [来源：2022-bethlehem-brain-charts.pdf p.1]；"
    "Ge 选 MFPR 最优 [来源：2024-ge-centilebrain.pdf p.6]；"
    "Seidel 提出 B-Norm 和 C-Norm [来源：2026-seidel-baseline-cortical-thickness.pdf p.1]；"
    "Rutherford 用 GAMLSS [来源：2022-rutherford-brain-growth-aging-precision.pdf p.3]。\n"
    "结果：Di Biase 横断面低估纵向变化，皮层厚度最多低估 44% [来源：2023-dibiase-mapping-brain-charts.pdf p.2]；"
    "Ge 的 MFPR 样本量超 3000 后性能平稳 [来源：2024-ge-centilebrain.pdf p.1]；"
    "Seidel 的 B-Norm 预测更准，解释方差高约 60% [来源：2026-seidel-baseline-cortical-thickness.pdf p.2]。\n"
    "局限性：Bethlehem 承认 MRI 研究相对全球人口多样性有偏差 [来源：2022-bethlehem-brain-charts.pdf p.1]；"
    "Di Biase 只用两个队列，年龄覆盖有限 [来源：2023-dibiase-mapping-brain-charts.pdf p.2]；"
    "Rutherford 的站点差异仍需 harmonisation [来源：2022-rutherford-brain-growth-aging-precision.pdf p.3]。"
)

# 没有局限性行的表格草稿，用于 partial 场景
TABLE_NO_LIMITS_DRAFT = "\n".join(TABLE_DRAFT.split("\n")[:4])

# 表格里混了一个编造引用：Bethlehem p.99 不在检索结果里
TABLE_FABRICATED_DRAFT = TABLE_DRAFT.replace(
    "[来源：2022-bethlehem-brain-charts.pdf p.1]；Di Biase 用 UKB 和 ABCD",
    "[来源：2022-bethlehem-brain-charts.pdf p.99]；Di Biase 用 UKB 和 ABCD", 1)


def _chunks_for_dim(query):
    for dim, chunks in CHUNKS_BY_DIM.items():
        if dim in query:
            return list(chunks)
    return []


def _chunks_for_paper(query):
    # 按论文名分发：文件名里的 dibiase 是连写的，问题里写 Di Biase 带空格，
    # 直接拿文件名切会匹配不到，这里用显示名映射。
    ql = query.lower()
    if "bethlehem" in ql:
        return list(CHUNKS_BY_PAPER[BETH])
    if "di biase" in ql:
        return list(CHUNKS_BY_PAPER[DIB])
    if "ge 2024" in ql:
        return list(CHUNKS_BY_PAPER[GE])
    if "seidel" in ql:
        return list(CHUNKS_BY_PAPER[SEI])
    if "rutherford" in ql:
        return list(CHUNKS_BY_PAPER[RUT])
    return []


def _run_crossdoc(monkeypatch, planner_json, tool_impl, drafts, missing_dims=None):
    """跑一次跨文献任务：Router 规则分类，比较、五篇触发 planner；
    Planner 按 planner_json 拆，Executor 用假 call_tool，回答员按 drafts
    顺序出答案。返回 result。missing_dims 是检索不到证据的维度名集合，
    该维度的子任务返回 0 片段。"""
    missing_dims = missing_dims or set()
    counts = {"planner": 0, "executor": 0, "answerer": 0}

    def fake_llm(prompt):
        if "任务规划器" in prompt:
            counts["planner"] += 1
            return planner_json
        if "执行器" in prompt:
            counts["executor"] += 1
            return '{"tool": "local_search", "params": {}}'
        counts["answerer"] += 1
        return drafts[min(counts["answerer"] - 1, len(drafts) - 1)]

    def fake_tool(name, **kwargs):
        return tool_impl(name, **kwargs)

    monkeypatch.setattr("app.agent._call_llm", fake_llm)
    monkeypatch.setattr("app.agent.call_tool", fake_tool)
    result = agent.run(QUESTION, use_pubmed=False)
    result["_counts"] = counts
    return result


def _dim_tool(missing_dims=None):
    missing_dims = missing_dims or set()

    def tool_impl(name, **kwargs):
        q = kwargs.get("query", "")
        for dim in missing_dims:
            if dim in q:
                return {"success": True, "data": {"chunks": []}, "error": None}
        return {"success": True, "data": {"chunks": _chunks_for_dim(q)}, "error": None}

    return tool_impl


def _paper_tool(missing_papers=None):
    missing_papers = missing_papers or set()

    def tool_impl(name, **kwargs):
        q = kwargs.get("query", "")
        for paper in missing_papers:
            if paper in q:
                return {"success": True, "data": {"chunks": []}, "error": None}
        return {"success": True, "data": {"chunks": _chunks_for_paper(q)}, "error": None}

    return tool_impl


def _all_citations_real(answer, chunks):
    """表格里每个引用都在检索结果里。"""
    pool = {(c["source"], str(c["page"])) for c in chunks}
    cited = {(m.group(1), m.group(2)) for m in agent.CITE_RE.finditer(answer)}
    assert cited, "表格里一个引用都没有"
    assert cited.issubset(pool), f"引用不在检索结果里：{sorted(cited - pool)}"
    return cited


def _all_chunks():
    out = []
    for chunks in CHUNKS_BY_DIM.values():
        out.extend(chunks)
    return out


# ---------- 完整表格：跑通，verified ----------

def test_crossdoc_full_table_verified(monkeypatch):
    result = _run_crossdoc(monkeypatch, PLANNER_DIM_JSON, _dim_tool(), [TABLE_DRAFT])
    assert result["answer_status"] == "verified"
    assert result["verified"] is True
    assert result["verify_failures"] == []
    # 五篇论文都覆盖到，每篇至少被引用一次
    for paper in PAPERS:
        assert paper in result["answer"], f"表格漏了 {paper}"
    # 每行都有引用，引用真实存在
    _all_citations_real(result["answer"], _all_chunks())
    # 五个维度都在答案里
    for dim in ("数据集", "建模变量", "模型", "结果", "局限性"):
        assert dim in result["answer"], f"表格漏了维度 {dim}"
    assert result["_counts"]["planner"] == 1
    assert result["_counts"]["executor"] == 5
    assert result["_counts"]["answerer"] == 1


def test_crossdoc_each_row_has_real_citation(monkeypatch):
    # 逐行核查：表格每行都有引用，且引用真实存在
    result = _run_crossdoc(monkeypatch, PLANNER_DIM_JSON, _dim_tool(), [TABLE_DRAFT])
    pool = {(c["source"], str(c["page"])) for c in _all_chunks()}
    rows = [r for r in result["answer"].split("\n") if r.strip()]
    assert len(rows) == 5
    for row in rows:
        cited = {(m.group(1), m.group(2)) for m in agent.CITE_RE.finditer(row)}
        assert cited, f"这一行没有引用：{row[:40]}"
        assert cited.issubset(pool), f"行内引用不在检索结果里：{sorted(cited - pool)}"


# ---------- 1 个维度证据不足：partial，不是全拒答 ----------

def test_crossdoc_one_dimension_missing_gives_partial(monkeypatch):
    # 局限性维度检索不到证据：子任务 0 片段，回答员出没有局限性行的表格
    result = _run_crossdoc(
        monkeypatch, PLANNER_DIM_JSON, _dim_tool(missing_dims={"局限性"}),
        [TABLE_NO_LIMITS_DRAFT, TABLE_NO_LIMITS_DRAFT])
    assert result["answer_status"] == "partial"
    assert result["verified"] is False
    # 已核实的 4 个维度正常输出
    for dim in ("数据集", "建模变量", "模型", "结果"):
        assert dim in result["answer"], f"partial 丢了已核实的维度 {dim}"
    # 局限性被标出来
    assert "部分回答" in result["answer"]
    assert "局限性" in result["answer"]
    assert {"subtask_id": "S5", "reason": "insufficient_evidence"} in result["verify_failures"]
    # S1-S4 通过了，只重跑 S5，不重跑全流程
    assert result["_counts"]["answerer"] == 2


def test_crossdoc_one_paper_missing_gives_partial(monkeypatch):
    # 按论文拆的分解方式：Rutherford 检索不到证据，回答员出 4 篇的表格
    # 注意：Rutherford 在每行都是最后一项，结尾是句号不是分号
    draft_4 = TABLE_DRAFT.replace(
        "Rutherford 基于 58836 人、82 个站点 [来源：2022-rutherford-brain-growth-aging-precision.pdf p.3]。", "")
    draft_4 = draft_4.replace(
        "Rutherford 用 GAMLSS [来源：2022-rutherford-brain-growth-aging-precision.pdf p.3]。", "")
    draft_4 = draft_4.replace(
        "Rutherford 的站点差异仍需 harmonisation [来源：2022-rutherford-brain-growth-aging-precision.pdf p.3]。", "")
    result = _run_crossdoc(
        monkeypatch, PLANNER_PAPER_JSON, _paper_tool(missing_papers={"Rutherford"}),
        [draft_4, draft_4])
    assert result["answer_status"] == "partial"
    assert result["verified"] is False
    # 其余 4 篇正常输出
    for paper in (BETH, DIB, GE, SEI):
        assert paper in result["answer"], f"partial 丢了已核实的论文 {paper}"
    assert "部分回答" in result["answer"]
    assert "Rutherford" in result["answer"]
    assert {"subtask_id": "S5", "reason": "insufficient_evidence"} in result["verify_failures"]


# ---------- 回答员汇总：prompt 里要带子任务清单 ----------

def test_answerer_prompt_lists_subtasks(monkeypatch):
    # 2.2 的汇总缺口修法：回答员的 prompt 里列出全部子任务，要求每个都覆盖，
    # 否则答案可能悄悄漏掉一个维度，核查还报 verified。
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return TABLE_DRAFT

    monkeypatch.setattr("app.agent._call_llm", fake_llm)
    state = {
        "query": QUESTION, "chunks": _all_chunks(), "draft": "", "final": "",
        "answer_status": "", "route": {}, "plan": {}, "history": [], "history_summary": "",
        "retrieval_query": QUESTION, "retries": 0, "retry": False,
        "verify_failures": [], "retry_target": "", "pubmed_results": [], "use_pubmed": False,
        "subtasks": [
            {"task_id": "S1", "query": "五篇纵向常模论文的数据集各是什么", "status": "completed",
             "evidence_ids": ["2022-bethlehem-brain-charts.pdf p.1"], "result": "", "retry_count": 0},
            {"task_id": "S2", "query": "五篇纵向常模论文的局限性是什么", "status": "completed",
             "evidence_ids": ["2022-rutherford-brain-growth-aging-precision.pdf p.3"], "result": "", "retry_count": 0},
        ],
    }
    result = agent.answerer_node(state)
    assert result["draft"] == TABLE_DRAFT
    assert len(prompts) == 1
    prompt = prompts[0]
    assert "五篇纵向常模论文的数据集各是什么" in prompt
    assert "五篇纵向常模论文的局限性是什么" in prompt


def test_answerer_prompt_without_subtasks_unchanged(monkeypatch):
    # 没子任务的简单问答：prompt 不加子任务清单，和原来一样
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return "总皮层灰质体积在 5.9 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.5]。"

    monkeypatch.setattr("app.agent._call_llm", fake_llm)
    state = {
        "query": "总皮层灰质体积在几岁达到峰值？",
        "chunks": [_chunk("总皮层灰质体积在 5.9 岁达到峰值。", BETH, 5)],
        "draft": "", "final": "", "answer_status": "", "route": {}, "plan": {},
        "history": [], "history_summary": "", "retrieval_query": "总皮层灰质体积在几岁达到峰值？",
        "retries": 0, "retry": False, "verify_failures": [], "retry_target": "",
        "pubmed_results": [], "use_pubmed": False, "subtasks": [],
    }
    result = agent.answerer_node(state)
    assert "5.9" in result["draft"]
    assert "子任务" not in prompts[0]


# ---------- 表格里编造引用：核查拦住 ----------

def test_crossdoc_fabricated_citation_in_table_blocked(monkeypatch):
    # 表格第一行引了 Bethlehem p.99，不在检索结果里：核查判编造，回回答员重答
    counts = {"answerer": 0}

    def fake_llm(prompt):
        if "任务规划器" in prompt:
            return PLANNER_DIM_JSON
        if "执行器" in prompt:
            return '{"tool": "local_search", "params": {}}'
        counts["answerer"] += 1
        if counts["answerer"] == 1:
            return TABLE_FABRICATED_DRAFT
        return TABLE_DRAFT

    monkeypatch.setattr("app.agent._call_llm", fake_llm)
    monkeypatch.setattr("app.agent.call_tool", _dim_tool())
    result = agent.run(QUESTION, use_pubmed=False)
    assert result["answer_status"] == "verified"
    assert "p.99" not in result["answer"]
    assert counts["answerer"] == 2
    _all_citations_real(result["answer"], _all_chunks())


def test_crossdoc_fabricated_citation_exhausted_refuses(monkeypatch):
    # 回答员一直编造，预算用完：拒答，不输出编造的表格
    def fake_llm(prompt):
        if "任务规划器" in prompt:
            return PLANNER_DIM_JSON
        if "执行器" in prompt:
            return '{"tool": "local_search", "params": {}}'
        return TABLE_FABRICATED_DRAFT

    monkeypatch.setattr("app.agent._call_llm", fake_llm)
    monkeypatch.setattr("app.agent.call_tool", _dim_tool())
    result = agent.run(QUESTION, use_pubmed=False)
    assert result["answer_status"] == "refused"
    assert result["answer"] == agent.REFUSAL
    assert "p.99" not in result["answer"]
