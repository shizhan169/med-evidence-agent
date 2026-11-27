"""核查升级测试（框架 v2，第一阶段 1.5）：不需要 LLM、不需要 Chroma、不联网。
跑法：pytest tests/test_verifier.py -v

思路：核查员按子任务逐个校验引用 + 失败归因分类 + 部分回答 + 只重跑失败子任务。
调 LLM 和检索都换成假的（monkeypatch），只验证核查逻辑；真的问答效果要等用户本地
跑完整评测。

注意：核查只看 draft 引用 vs chunks/pubmed_results，不读 history_summary——
摘要只是 prompt 里的上下文入口，不是用户原话，不参与引用校验。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.agent as agent

CHUNKS_A = [{"text": "A 论文的数据集是 X。", "source": "2022-a-paper.pdf", "page": 1}]
CHUNKS_B = [{"text": "B 论文的数据集是 Y。", "source": "2022-b-paper.pdf", "page": 2}]
DRAFT_A = "A 论文的数据集是 X [来源：2022-a-paper.pdf p.1]。"
DRAFT_AB = ("A 论文的数据集是 X [来源：2022-a-paper.pdf p.1]。"
            "B 论文的数据集是 Y [来源：2022-b-paper.pdf p.2]。")
PUBMED_RESULTS = [
    {"pmid": "38395541", "title": "CentileBrain", "authors": "Ge",
     "journal": "Nature", "year": "2024",
     "abstract": "We developed this framework with regional morphometric data "
                 "from 37,407 healthy individuals aged 3-90 years."},
    {"pmid": "39974093", "title": "Spectral normative modeling", "authors": "Smith",
     "journal": "Nature", "year": "2025",
     "abstract": "Spectral normative modeling captures brain development "
                 "trajectories across the lifespan using frequency domain analysis. "
                 "DOI:10.1002/spectral.2025."},
]


def _subtask(task_id, query, status="completed", evidence_ids=None, retry_count=0):
    return {"task_id": task_id, "query": query, "status": status,
            "evidence_ids": evidence_ids or [], "result": "", "retry_count": retry_count}


def _base_state(**overrides):
    """直接调 verifier_node 用的最小 state（历史短，不调 LLM 做摘要）。"""
    state = {
        "query": "比较 A 和 B 论文的数据集",
        "chunks": [], "draft": "", "final": "", "answer_status": "",
        "route": {}, "plan": {}, "subtasks": [], "tool_calls": [], "figure": {}, "error": "",
        "history": [], "history_summary": "", "retrieval_query": "比较 A 和 B 论文的数据集",
        "retries": 0, "retry": False, "verify_failures": [], "retry_target": "",
        "pubmed_results": [], "use_pubmed": False,
    }
    state.update(overrides)
    return state


# ---------- 部分回答 ----------

def test_partial_keeps_verified_part_only():
    # 2 个子任务：S1 通过（有证据），S2 证据不足；重试预算已用完，直接定 partial
    state = _base_state(
        chunks=CHUNKS_A, draft=DRAFT_A + "B 论文的数据集是 Y。",
        subtasks=[
            _subtask("S1", "A 论文的数据集", "completed",
                     ["2022-a-paper.pdf p.1"], retry_count=1),
            _subtask("S2", "B 论文的数据集", "insufficient_evidence",
                     [], retry_count=1),
        ],
        retries=2,
    )
    result = agent.verifier_node(state)
    assert result["answer_status"] == "partial"
    # 注意：verifier_node 的更新字典里没有 verified 键；verified 是 _to_response
    # 从 answer_status 派生的兼容字段，只在 agent.run 返回里出现（见端到端测试）。
    # 只含通过的部分：S2 那句没引用，被过滤掉了
    assert "A 论文的数据集是 X" in result["final"]
    assert "B 论文的数据集是 Y" not in result["final"]
    # 标出未答部分：S2 的 query 点名；“部分回答”四个字只出现在 partial 标注里
    assert "部分回答" in result["final"]
    assert "B 论文的数据集" in result["final"]
    assert result["verify_failures"] == [
        {"subtask_id": "S2", "reason": "insufficient_evidence"}]


def test_refusal_with_failed_subtasks_still_refused():
    # 回答员本来就拒答（没证据）：即使子任务都失败了，核查也直接确认拒答，
    # 不进 partial（partial 要的是“有已核实的部分可输出”，这里没有）
    state = _base_state(
        chunks=[], draft=agent.REFUSAL,
        subtasks=[
            _subtask("S1", "A 论文的数据集", "insufficient_evidence", [], retry_count=1),
            _subtask("S2", "B 论文的数据集", "failed", [], retry_count=1),
        ],
        retries=2,
    )
    result = agent.verifier_node(state)
    assert result["answer_status"] == "refused"
    assert result["final"] == agent.REFUSAL


def test_partial_all_failed_but_draft_has_citations_refuses():
    # 草稿有真实引用、但两个子任务都没证据：一个都没通过 → refused，不是 partial
    state = _base_state(
        chunks=CHUNKS_A, draft=DRAFT_A,
        subtasks=[
            _subtask("S1", "A 论文的数据集", "failed", [], retry_count=1),
            _subtask("S2", "B 论文的数据集", "insufficient_evidence", [], retry_count=1),
        ],
        retries=2,
    )
    result = agent.verifier_node(state)
    assert result["answer_status"] == "refused"
    assert result["final"] == agent.REFUSAL
    assert result["verify_failures"] == [
        {"subtask_id": "S1", "reason": "insufficient_evidence"},
        {"subtask_id": "S2", "reason": "insufficient_evidence"}]


def test_error_llm_down():
    # LLM 没调通：draft 是 [待接 LLM] 占位 → error，不是 refused
    draft = "[待接 LLM] 已生成 Prompt（100 字符），请在 .env 填入 QWEN_API_KEY 或 DEEPSEEK_API_KEY 后重试。"
    state = _base_state(chunks=CHUNKS_A, draft=draft)
    result = agent.verifier_node(state)
    assert result["answer_status"] == "error"
    assert result["final"] == draft  # 返回错误说明，不触发拒答话术
    assert result["final"] != agent.REFUSAL
    assert result["verify_failures"] == [{"subtask_id": "", "reason": "llm_error"}]
    assert result["retry"] is False  # LLM 挂了重试也没用，不重试


def test_refused_no_evidence():
    # 文献里没依据：回答员本来就拒答 → refused，标准拒答话术
    state = _base_state(chunks=[], draft=agent.REFUSAL)
    result = agent.verifier_node(state)
    assert result["answer_status"] == "refused"
    assert result["final"] == agent.REFUSAL
    assert result["verify_failures"] == []


def test_refused_no_chunks_forces_refusal():
    # 完全没证据、草稿却有内容（防御性收尾）：强制拒答，不是 error
    state = _base_state(chunks=[], draft="A 论文的数据集是 X。")
    result = agent.verifier_node(state)
    assert result["answer_status"] == "refused"
    assert result["final"] == agent.REFUSAL


# ---------- 失败归因分类：四种 reason 都能分出来 ----------

def test_failure_reason_fabricated_citation():
    # 引用编造：引了不在检索结果里的 p.99
    state = _base_state(
        chunks=CHUNKS_A,
        draft="A 论文的数据集是 X [来源：2022-a-paper.pdf p.99]。")
    result = agent.verifier_node(state)
    assert result["retry"] is True
    assert result["retry_target"] == "answerer"
    assert result["verify_failures"] == [{"subtask_id": "", "reason": "fabricated_citation"}]


def test_failure_reason_missing_citation():
    # 缺引用：有内容没给引用
    state = _base_state(chunks=CHUNKS_A, draft="A 论文的数据集是 X。")
    result = agent.verifier_node(state)
    assert result["retry"] is True
    assert result["retry_target"] == "answerer"
    assert result["verify_failures"] == [{"subtask_id": "", "reason": "missing_citation"}]


def test_failure_reason_insufficient_evidence():
    # 证据不足：子任务没拿到证据（见 test_partial_keeps_verified_part_only）
    state = _base_state(
        chunks=CHUNKS_A, draft=DRAFT_A,
        subtasks=[
            _subtask("S1", "A 论文的数据集", "completed",
                     ["2022-a-paper.pdf p.1"], retry_count=1),
            _subtask("S2", "B 论文的数据集", "insufficient_evidence", [], retry_count=1),
        ],
        retries=2,
    )
    result = agent.verifier_node(state)
    assert result["answer_status"] == "partial"
    assert {"subtask_id": "S2", "reason": "insufficient_evidence"} in result["verify_failures"]


def test_failure_reason_llm_error():
    # LLM 没调通（见 test_error_llm_down）：reason 是 llm_error
    draft = "[待接 LLM] 已生成 Prompt（100 字符），请在 .env 填入 QWEN_API_KEY 后重试。"
    state = _base_state(chunks=CHUNKS_A, draft=draft)
    result = agent.verifier_node(state)
    assert result["answer_status"] == "error"
    assert result["verify_failures"] == [{"subtask_id": "", "reason": "llm_error"}]


def test_completed_subtask_with_empty_evidence_is_insufficient():
    # 工具“成功”但 0 片段：status 是 completed、evidence_ids 为空 → 证据不足
    state = _base_state(
        chunks=CHUNKS_A, draft=DRAFT_A,
        subtasks=[
            _subtask("S1", "A 论文的数据集", "completed",
                     ["2022-a-paper.pdf p.1"], retry_count=1),
            _subtask("S2", "B 论文的数据集", "completed", [], retry_count=1),
        ],
        retries=2,
    )
    result = agent.verifier_node(state)
    assert result["answer_status"] == "partial"
    assert {"subtask_id": "S2", "reason": "insufficient_evidence"} in result["verify_failures"]


# ---------- 只重跑失败子任务（端到端） ----------

def _run_planned(monkeypatch, tool_impl, drafts):
    """跑一次规划任务：Router 规则分类（“比较”→ planner），Planner 拆 2 个子任务，
    Executor 用假 call_tool，回答员按 drafts 顺序出答案。
    返回 (result, counts)，counts 记各节点调 LLM 的次数和工具调用次数。"""
    counts = {"planner": 0, "executor": 0, "answerer": 0, "tool_A": 0, "tool_B": 0}

    def fake_llm(prompt):
        if "任务规划器" in prompt:
            counts["planner"] += 1
            return ('{"goal": "比较 A 和 B 的数据集", "subtasks": '
                    '[{"query": "A 论文的数据集"}, {"query": "B 论文的数据集"}]}')
        if "执行器" in prompt:
            counts["executor"] += 1
            return '{"tool": "local_search", "params": {}}'
        counts["answerer"] += 1
        return drafts[min(counts["answerer"] - 1, len(drafts) - 1)]

    def fake_tool(name, **kwargs):
        q = kwargs.get("query", "")
        if "A 论文" in q:
            counts["tool_A"] += 1
        if "B 论文" in q:
            counts["tool_B"] += 1
        return tool_impl(name, **kwargs)

    monkeypatch.setattr("app.agent._call_llm", fake_llm)
    monkeypatch.setattr("app.agent.call_tool", fake_tool)
    result = agent.run("比较 A 和 B 论文的数据集", use_pubmed=False)
    return result, counts


def test_retry_only_failed_subtask(monkeypatch):
    # S1 一次拿到证据，S2 第一次 0 片段、重试后拿到证据：S1 不重跑，S2 只重跑一次
    def tool_impl(name, **kwargs):
        q = kwargs.get("query", "")
        if "A 论文" in q:
            return {"success": True, "data": {"chunks": CHUNKS_A}, "error": None}
        if "B 论文" in q:
            # 第一次 0 片段（工具成功但没证据），第二次拿到证据
            tool_impl.calls["B"] += 1
            if tool_impl.calls["B"] == 1:
                return {"success": True, "data": {"chunks": []}, "error": None}
            return {"success": True, "data": {"chunks": CHUNKS_B}, "error": None}
        return {"success": True, "data": {"chunks": []}, "error": None}

    tool_impl.calls = {"B": 0}
    result, counts = _run_planned(monkeypatch, tool_impl, [DRAFT_A, DRAFT_AB])
    assert result["answer_status"] == "verified"
    assert "A 论文的数据集是 X" in result["answer"]
    assert "B 论文的数据集是 Y" in result["answer"]
    assert result["verify_failures"] == []
    # S1 只跑一次（通过的不重跑），S2 跑两次（第一次 0 片段 + 核查后重跑一次）
    assert counts["tool_A"] == 1
    assert counts["tool_B"] == 2
    assert counts["answerer"] == 2  # 第一次答 A，第二次重答 A+B


def test_mixed_retries_respect_overall_cap(monkeypatch):
    # 整体重试上限 2 次：第一次引用编造（回 answerer，retries 0→1），
    # 第二次 S2 证据不足（回 executor，retries 1→2），第三次核查不再重试 → partial
    def tool_impl(name, **kwargs):
        q = kwargs.get("query", "")
        if "A 论文" in q:
            return {"success": True, "data": {"chunks": CHUNKS_A}, "error": None}
        # B 永远 0 片段：S2 拿不到证据
        return {"success": True, "data": {"chunks": []}, "error": None}

    drafts = [
        "A 论文的数据集是 X [来源：2022-a-paper.pdf p.99]。",  # 编造，重答
        DRAFT_A,  # 引用对了，但 S2 没证据 → executor 重跑 S2
        DRAFT_A,  # S2 还是没证据，预算用完 → partial
    ]
    result, counts = _run_planned(monkeypatch, tool_impl, drafts)
    assert result["answer_status"] == "partial"
    assert result["verified"] is False  # 兼容字段：partial 时 verified 为 False
    assert "A 论文的数据集是 X" in result["answer"]
    # partial 标注在答案里；这里只断言“部分回答”四个字，该词只出现在 partial 标注里。
    # 注意：不要在测试文件里写全角括号，会触发写入工具丢缩进。
    assert "部分回答" in result["answer"]
    assert "B 论文的数据集" in result["answer"]
    assert counts["answerer"] == 3  # 1 次原答 + 2 次重试，不再多试
    assert counts["tool_A"] == 1  # S1 通过了，一次都没重跑
    assert counts["tool_B"] == 2  # S2 只重跑一次（executor 一次 + 核查后一次）
    assert {"subtask_id": "S2", "reason": "insufficient_evidence"} in result["verify_failures"]


# ---------- PubMed 引用放宽 ----------

def test_pubmed_doi_citation_accepted():
    # DOI 引用：DOI 在本次返回的摘要文字里 → 通过，不强制 PMID
    state = _base_state(
        pubmed_results=PUBMED_RESULTS,
        draft="Spectral normative modeling 见 DOI:10.1002/spectral.2025。",
        use_pubmed=True)
    result = agent.verifier_node(state)
    assert result["answer_status"] == "verified"
    assert result["verify_failures"] == []


def test_pubmed_doi_citation_fabricated_blocked():
    # 编出来的 DOI（不在返回文字里）→ 编造，重试
    state = _base_state(
        pubmed_results=PUBMED_RESULTS,
        draft="Spectral normative modeling 见 DOI:10.9999/fake.1234。",
        use_pubmed=True)
    result = agent.verifier_node(state)
    assert result["retry"] is True
    assert result["verify_failures"] == [{"subtask_id": "", "reason": "fabricated_citation"}]


def test_pubmed_abstract_citation_accepted():
    # 摘要引用：草稿原样引用了返回摘要的文字（连续 40 字以上）→ 通过，不强制 PMID
    state = _base_state(
        pubmed_results=PUBMED_RESULTS,
        draft=("该框架的摘要说：We developed this framework with regional "
               "morphometric data from 37,407 healthy individuals aged 3-90 years."),
        use_pubmed=True)
    result = agent.verifier_node(state)
    assert result["answer_status"] == "verified"
    assert result["verify_failures"] == []


def test_pubmed_fake_pmid_still_blocked():
    # 假 PMID 仍判编造（放宽的是引用形式，不是放宽真实性）
    state = _base_state(
        pubmed_results=PUBMED_RESULTS,
        draft="有一篇讲常模 PubMed PMID:99999999。",
        use_pubmed=True)
    result = agent.verifier_node(state)
    assert result["retry"] is True
    assert result["retry_target"] == "answerer"
    assert result["verify_failures"] == [{"subtask_id": "", "reason": "fabricated_citation"}]


def test_pubmed_no_citation_at_all_still_retries():
    # 一个引用形式都没有（没 PMID、没 DOI、没引摘要）→ 缺引用，重试
    state = _base_state(
        pubmed_results=PUBMED_RESULTS,
        draft="CentileBrain 用 37,407 人建了常模。",
        use_pubmed=True)
    result = agent.verifier_node(state)
    assert result["retry"] is True
    assert result["verify_failures"] == [{"subtask_id": "", "reason": "missing_citation"}]


# ---------- 失败重试不走偏 ----------

def test_fallback_path_executor_retry(monkeypatch):
    """LangGraph 不可用时走顺序执行兜底，executor 重试逻辑要和图里一致。"""
    def broken_graph():
        raise RuntimeError("LangGraph 不可用")

    monkeypatch.setattr("app.agent._build_graph", broken_graph)

    def tool_impl(name, **kwargs):
        q = kwargs.get("query", "")
        if "A 论文" in q:
            return {"success": True, "data": {"chunks": CHUNKS_A}, "error": None}
        if "B 论文" in q:
            tool_impl.calls["B"] += 1
            if tool_impl.calls["B"] == 1:
                return {"success": True, "data": {"chunks": []}, "error": None}
            return {"success": True, "data": {"chunks": CHUNKS_B}, "error": None}
        return {"success": True, "data": {"chunks": []}, "error": None}

    tool_impl.calls = {"B": 0}
    result, counts = _run_planned(monkeypatch, tool_impl, [DRAFT_A, DRAFT_AB])
    assert result["answer_status"] == "verified"
    assert counts["tool_A"] == 1
    assert counts["tool_B"] == 2
