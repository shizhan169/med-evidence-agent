"""2.4 评测脚本扩展的测试：6 个必做维度 + 跨文档多来源 + 多轮序列。
用假数据验证评测逻辑，不需要真 LLM、不需要 Chroma、不联网。
跑法(仓库根目录)：
    pytest tests/test_eval_dimensions.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

# 先 import run_eval_auto：它 import run_eval 时会把 eval/ 从 sys.path 移除
# (Codex 的修复)，所以必须在 from eval.run_eval import 之前 import 它，
# 否则后面找不到 run_eval_auto。
import run_eval_auto  # noqa: E402

from eval.run_eval import (  # noqa: E402
    citation_status,
    evaluate_result,
    expected_route,
    expected_tool_path,
    multiturn_summary,
    parse_questions,
    parse_sources,
    planning_summary,
    route_correct,
    router_summary,
    run_sequence,
    section_summary,
    task_completion,
    timing_summary,
    vlm_summary,
)

QUESTIONS_FILE = ROOT / "eval" / "test_questions.md"


def _cits(*pairs):
    return [{"source": f, "page": p} for f, p in pairs]


# ---- parse_sources：多来源解析 ----

def test_parse_sources_single():
    srcs = parse_sources("2022-bethlehem-brain-charts.pdf p.1")
    assert len(srcs) == 1
    assert srcs[0]["file"] == "2022-bethlehem-brain-charts.pdf"
    assert srcs[0]["pages"] == [1]


def test_parse_sources_multi_semicolon():
    srcs = parse_sources(
        "2018-fortin-harmonization-cortical-thickness.pdf p.1; "
        "2020-pomponio-harmonization-mri-datasets.pdf p.2、p.3")
    assert len(srcs) == 2
    assert srcs[0]["file"] == "2018-fortin-harmonization-cortical-thickness.pdf"
    assert srcs[0]["pages"] == [1]
    assert srcs[1]["file"] == "2020-pomponio-harmonization-mri-datasets.pdf"
    assert srcs[1]["pages"] == [2, 3]


def test_parse_sources_empty():
    assert parse_sources("") == []
    assert parse_sources("no pdf here") == []


# ---- 跨文档题：Q18/Q19 有两个预期来源 ----

def test_q18_has_two_expected_sources():
    qs = {q["id"]: q for q in parse_questions(QUESTIONS_FILE)}
    q18 = qs["Q18"]
    files = [s["file"] for s in q18["expected_sources"]]
    assert files[0] == "2018-fortin-harmonization-cortical-thickness.pdf"
    assert "2020-pomponio-harmonization-mri-datasets.pdf" in files
    assert q18["expected_sources"][1]["pages"] == [2, 3]
    # 主来源保持不变(向后兼容)
    assert q18["expected_file"] == "2018-fortin-harmonization-cortical-thickness.pdf"
    assert q18["expected_pages"] == [1]


def test_q19_has_two_expected_sources():
    qs = {q["id"]: q for q in parse_questions(QUESTIONS_FILE)}
    q19 = qs["Q19"]
    files = [s["file"] for s in q19["expected_sources"]]
    assert files[0] == "2022-kia-closing-lifecycle-normative-modeling.pdf"
    assert "2026-ho-unified-multi-cohort-harmonisation.pdf" in files


# ---- citation_status：多来源引用核查 ----

def test_citation_status_full_both_sources():
    cits = _cits(
        ("2018-fortin-harmonization-cortical-thickness.pdf", 1),
        ("2020-pomponio-harmonization-mri-datasets.pdf", 2),
    )
    srcs = parse_sources(
        "2018-fortin-harmonization-cortical-thickness.pdf p.1; "
        "2020-pomponio-harmonization-mri-datasets.pdf p.2、p.3")
    status, per = citation_status(cits, srcs)
    assert status == "full"
    assert [s["hit"] for s in per] == [True, True]


def test_citation_status_partial_q18_only_pomponio():
    # 上一轮 Q18 的真实情况：只引了 Pomponio，没引 Fortin。
    # 这是 partial，不是 full。
    cits = _cits(
        ("2020-pomponio-harmonization-mri-datasets.pdf", 2),
        ("2020-pomponio-harmonization-mri-datasets.pdf", 7),
    )
    qs = {q["id"]: q for q in parse_questions(QUESTIONS_FILE)}
    status, per = citation_status(cits, qs["Q18"]["expected_sources"])
    assert status == "partial"
    assert [s["hit"] for s in per] == [False, True]


def test_citation_status_partial_primary_only():
    cits = _cits(("2018-fortin-harmonization-cortical-thickness.pdf", 1))
    qs = {q["id"]: q for q in parse_questions(QUESTIONS_FILE)}
    status, per = citation_status(cits, qs["Q18"]["expected_sources"])
    assert status == "partial"
    assert [s["hit"] for s in per] == [True, False]


def test_citation_status_miss():
    cits = _cits(("2024-ge-centilebrain.pdf", 1))
    qs = {q["id"]: q for q in parse_questions(QUESTIONS_FILE)}
    status, per = citation_status(cits, qs["Q18"]["expected_sources"])
    assert status == "miss"
    assert [s["hit"] for s in per] == [False, False]


def test_citation_status_single_source():
    cits = _cits(("2022-bethlehem-brain-charts.pdf", 1))
    srcs = parse_sources("2022-bethlehem-brain-charts.pdf p.1")
    status, _ = citation_status(cits, srcs)
    assert status == "full"
    status, _ = citation_status([], srcs)
    assert status == "miss"


# ---- evaluate_result：回答正确性 / 拒答 / 引用 ----

def _q(qid="Q01", section="original", qtype="可回答",
       sources="2022-bethlehem-brain-charts.pdf p.1"):
    return {
        "id": qid, "section": section, "type": qtype,
        "question": "test question",
        "expected_answer": "expected",
        "expected_source": sources,
        "refuse_reason": "",
        "expected_sources": parse_sources(sources),
    }


def _result(answer, verified=True, status="verified", cits=None, failures=None):
    return {"answer": answer, "verified": verified, "answer_status": status,
            "citations": cits or [], "verify_failures": failures or []}


def _simple_route():
    return {"task_type": "simple", "requires_planning": False,
            "requires_external_search": False, "requires_vision": False}


def test_evaluate_result_answered_and_cited():
    q = _q()
    r = _result("共汇集 123,984 次 MRI 扫描。",
                cits=_cits(("2022-bethlehem-brain-charts.pdf", 1)))
    rec = evaluate_result(q, r, _simple_route(), 1.5)
    assert rec["refused"] is False
    assert rec["citation_hit"] is True
    assert rec["citation_status"] == "full"
    assert rec["route_correct"] is True
    assert rec["elapsed_seconds"] == 1.5
    assert rec["answer_status"] == "verified"
    assert rec["tool_path"] == "local_search"


def test_evaluate_result_refusal():
    q = _q()
    r = _result("根据现有文献无法回答。", verified=False, status="refused")
    rec = evaluate_result(q, r, _simple_route(), 0.8)
    assert rec["refused"] is True
    assert rec["citation_hit"] is False
    assert rec["citation_status"] == "miss"


def test_evaluate_result_q18_partial_not_hit():
    # Q18 只引 Pomponio 时，citation_hit 必须是 False(不完全命中不算命中)
    qs = {q["id"]: q for q in parse_questions(QUESTIONS_FILE)}
    q18 = qs["Q18"]
    r = _result("Fortin 用 ComBat，Pomponio 用 ComBat-GAM。",
                cits=_cits(("2020-pomponio-harmonization-mri-datasets.pdf", 2)))
    rec = evaluate_result(q18, r, _simple_route(), 2.0)
    assert rec["refused"] is False
    assert rec["citation_status"] == "partial"
    assert rec["citation_hit"] is False


# ---- 路由正确性 ----

def test_expected_route():
    assert expected_route("Q01")["requires_planning"] is False
    assert expected_route("Q01")["requires_vision"] is False
    assert expected_route("Q04") == {"task_type": "complex", "requires_planning": True,
                                     "requires_external_search": False, "requires_vision": False}
    assert expected_route("Q12")["requires_planning"] is True
    assert expected_route("R05")["requires_external_search"] is True
    assert expected_route("F01")["requires_vision"] is True
    assert expected_route("FR02")["requires_vision"] is True


def test_route_correct():
    exp = _simple_route()
    assert route_correct(dict(exp), exp) is True
    wrong = dict(exp, requires_vision=True)
    assert route_correct(wrong, exp) is False
    assert route_correct({}, exp) is None
    assert route_correct(None, exp) is None


# ---- 复杂任务完成度 ----

def test_task_completion():
    assert task_completion("verified") == ("complete", [])
    assert task_completion("partial", [{"subtask_id": "S2", "reason": "insufficient_evidence"}]) == ("partial", ["S2"])
    assert task_completion("refused") == ("incomplete", [])
    assert task_completion("error") == ("incomplete", [])
    assert task_completion("") == ("incomplete", [])


# ---- 工具路径 ----

def test_expected_tool_path():
    assert expected_tool_path({"requires_planning": True}) == "planner+executor"
    assert expected_tool_path({"requires_external_search": True}) == "pubmed_search"
    assert expected_tool_path({"requires_vision": True}) == "chart_locate/chart_read"
    assert expected_tool_path({"requires_planning": False, "requires_external_search": False,
                               "requires_vision": False}) == "local_search"
    assert expected_tool_path({}) == "unknown"


# ---- 各维度汇总 ----

def _rec(qid, section="original", qtype="可回答", refused=False, hit=True,
         route_correct=None, completion=None, route=None, elapsed=1.0, status="full"):
    return {"id": qid, "section": section, "type": qtype, "refused": refused,
            "citation_hit": hit, "citation_status": status,
            "route_correct": route_correct, "task_completion": completion,
            "route": route or {}, "elapsed_seconds": elapsed, "error": ""}


def test_router_summary():
    recs = [
        _rec("Q01", route_correct=True),
        _rec("Q04", route_correct=True),
        _rec("F01", route_correct=False),
        _rec("Q02", route_correct=None),
    ]
    s = router_summary(recs)
    assert s["total"] == 3
    assert s["correct"] == 2
    assert s["accuracy"] == round(2 / 3, 4)
    assert s["wrong_ids"] == ["F01"]


def test_planning_summary():
    recs = [
        _rec("Q04", completion="complete"),
        _rec("Q12", completion="partial"),
        _rec("Q01", completion=None),
    ]
    s = planning_summary(recs)
    assert s["total"] == 2
    assert s["complete"] == 1
    assert s["partial"] == 1
    assert s["incomplete"] == 0
    assert s["partial_ids"] == ["Q12"]


def test_vlm_summary():
    recs = [
        _rec("F01", section="figure", route={"requires_vision": True}),
        _rec("F02", section="figure", refused=True, hit=False,
             route={"requires_vision": True}),
        _rec("F06", section="figure", refused=True, hit=False,
             route={"requires_vision": False}),
        _rec("Q01", section="original"),
    ]
    s = vlm_summary(recs)
    assert s["total"] == 3
    assert s["answered"] == 1
    assert s["citation_hits"] == 1
    assert s["citation_accuracy"] == 1.0
    assert s["vision_routed"] == 2
    assert s["vision_routing_rate"] == round(2 / 3, 4)


def test_multiturn_summary():
    recs = [
        _rec("MT01-T1", section="multiturn"),
        _rec("MT01-T2", section="multiturn"),
        _rec("MT02-T1", section="multiturn", refused=True, hit=False),
        _rec("Q01", section="original"),
    ]
    s = multiturn_summary(recs)
    assert s["total_turns"] == 3
    assert s["sequences"]["MT01"]["turns"] == 2
    assert s["sequences"]["MT01"]["answered"] == 2
    assert s["sequences"]["MT02"]["turns"] == 1
    assert s["sequences"]["MT02"]["answered"] == 0


def test_timing_summary():
    recs = [_rec("Q01", elapsed=2.0), _rec("Q02", elapsed=5.0), _rec("Q03", elapsed=1.0)]
    s = timing_summary(recs)
    assert s["total_seconds"] == 8.0
    assert s["mean_seconds"] == round(8.0 / 3, 1)
    assert [x["id"] for x in s["slowest"]] == ["Q02", "Q01", "Q03"]


def test_section_summary_ignores_new_fields():
    # 新记录多了很多字段，旧的 section_summary 口径不能变
    recs = [
        _rec("Q01", route_correct=True, completion=None),
        _rec("Q02", refused=True, hit=False, route_correct=False),
        _rec("R01", qtype="应拒答", refused=True, hit=False),
    ]
    s = section_summary(recs)
    assert s["total"] == 3
    assert s["answerable_answered"] == 1
    assert s["citation_hits"] == 1
    assert s["should_refuse_correct"] == 1
    assert s["fabrication_count"] == 0


# ---- 多轮序列：按顺序跑，历史往下传 ----

def test_run_sequence_passes_history_in_order():
    seen = []

    def fake_run(question, history=None, use_pubmed=False):
        seen.append([m["content"] for m in (history or []) if m["role"] == "user"])
        return _result("answer to " + question,
                       cits=_cits(("2023-fujita-brain-volume-aging.pdf", 1)))

    def fake_router(state):
        return {"route": _simple_route()}

    seq = {
        "id": "MT01",
        "title": "test",
        "turns": [
            {"question": "first question", "type": "可回答",
             "expected_answer": "a1",
             "expected_source": "2023-fujita-brain-volume-aging.pdf p.1"},
            {"question": "second question", "type": "可回答",
             "expected_answer": "a2",
             "expected_source": "2023-fujita-brain-volume-aging.pdf p.1"},
        ],
    }
    recs = run_sequence(seq, fake_run, fake_router)
    assert [r["id"] for r in recs] == ["MT01-T1", "MT01-T2"]
    assert all(r["section"] == "multiturn" for r in recs)
    # 第一轮历史是空的，第二轮能看到第一轮的问题
    assert seen[0] == []
    assert seen[1] == ["first question"]
    # 每轮都答出且引用命中
    assert all(not r["refused"] for r in recs)
    assert all(r["citation_hit"] for r in recs)


def test_run_sequence_refused_turn_still_in_history():
    seen = []

    def fake_run(question, history=None, use_pubmed=False):
        seen.append([m["content"] for m in (history or [])])
        if "first" in question:
            return _result("根据现有文献无法回答。", verified=False, status="refused")
        return _result("total visits 7915",
                       cits=_cits(("2023-fujita-brain-volume-aging.pdf", 1)))

    seq = {
        "id": "MT02",
        "title": "test",
        "turns": [
            {"question": "first question", "type": "可回答",
             "expected_answer": "a1",
             "expected_source": "2023-fujita-brain-volume-aging.pdf p.1"},
            {"question": "second question", "type": "可回答",
             "expected_answer": "a2",
             "expected_source": "2023-fujita-brain-volume-aging.pdf p.1"},
        ],
    }
    recs = run_sequence(seq, fake_run, None)
    assert recs[0]["refused"] is True
    # 第二轮的历史里有第一轮的问题和拒答
    assert "first question" in seen[1]
    assert "根据现有文献无法回答。" in seen[1]
    assert recs[1]["refused"] is False


# ---- run_eval_auto：新维度对比 ----

def _auto_rec(qid, section="original", qtype="可回答", refused=False, hit=True,
              route_correct=None, status=None, completion=None, error=""):
    return {"id": qid, "section": section, "type": qtype,
            "refused": refused, "citation_hit": hit, "error": error,
            "route_correct": route_correct, "citation_status": status,
            "task_completion": completion}


def test_compare_route_regression():
    prev = [_auto_rec("Q04", route_correct=True)]
    new = [_auto_rec("Q04", route_correct=False)]
    comp = run_eval_auto.compare_records(prev, new)
    assert [r["id"] for r in comp["regressions"]] == ["Q04"]
    assert "路由对变错" in comp["regressions"][0]["description"]


def test_compare_route_improvement():
    prev = [_auto_rec("F01", route_correct=False)]
    new = [_auto_rec("F01", route_correct=True)]
    comp = run_eval_auto.compare_records(prev, new)
    assert [r["id"] for r in comp["improvements"]] == ["F01"]


def test_compare_citation_status_partial_is_regression():
    prev = [_auto_rec("Q18", hit=True, status="full")]
    new = [_auto_rec("Q18", hit=False, status="partial")]
    comp = run_eval_auto.compare_records(prev, new)
    assert [r["id"] for r in comp["regressions"]] == ["Q18"]


def test_compare_citation_status_partial_to_miss():
    # citation_hit 两轮都是 False，但状态从 partial 掉到 miss 也是退步
    prev = [_auto_rec("Q18", hit=False, status="partial")]
    new = [_auto_rec("Q18", hit=False, status="miss")]
    comp = run_eval_auto.compare_records(prev, new)
    assert [r["id"] for r in comp["regressions"]] == ["Q18"]


def test_compare_task_completion_regression():
    prev = [_auto_rec("Q04", completion="complete")]
    new = [_auto_rec("Q04", completion="partial")]
    comp = run_eval_auto.compare_records(prev, new)
    assert [r["id"] for r in comp["regressions"]] == ["Q04"]


def test_compare_old_records_without_new_fields():
    # 老 results.json 没有 route_correct / citation_status / task_completion，
    # 对比不能崩，也不能误报新维度退步
    prev = [{"id": "Q01", "section": "original", "type": "可回答",
             "refused": False, "citation_hit": True, "error": ""}]
    new = [_auto_rec("Q01", route_correct=True, status="full")]
    comp = run_eval_auto.compare_records(prev, new)
    assert comp["regressions"] == []
    assert comp["improvements"] == []


def test_summary_line_with_new_dimensions():
    summary = {
        "original": {"answerable_answered": 22, "answerable_total": 23,
                     "citation_accuracy": 0.9545, "citation_hits": 21,
                     "should_refuse_correct": 7, "should_refuse_total": 7,
                     "fabrication_count": 0},
        "figure": {"total": 10, "answerable_answered": 7, "answerable_total": 8,
                   "citation_accuracy": 1.0, "citation_hits": 7,
                   "should_refuse_correct": 2, "should_refuse_total": 2,
                   "fabrication_count": 0},
        "router": {"total": 40, "correct": 39, "accuracy": 0.975},
        "planning": {"total": 2, "complete": 1, "partial": 1, "incomplete": 0},
        "vlm": {"total": 10, "citation_accuracy": 1.0, "vision_routed": 10},
        "multiturn": {"total_turns": 2},
    }
    line = run_eval_auto._summary_line(summary)
    assert "路由 39/40" in line
    assert "planning 完全完成 1/2" in line
    assert "多轮 2 轮" in line


def test_summary_line_old_summary_no_crash():
    # 老 summary 没有新维度字段，不能崩
    summary = {
        "original": {"answerable_answered": 13, "answerable_total": 15,
                     "citation_accuracy": 0.923, "citation_hits": 12,
                     "should_refuse_correct": 5, "should_refuse_total": 5,
                     "fabrication_count": 0},
        "figure": {"total": 8, "answerable_answered": 4, "answerable_total": 6,
                   "citation_accuracy": 1.0, "citation_hits": 4,
                   "should_refuse_correct": 2, "should_refuse_total": 2,
                   "fabrication_count": 0},
    }
    line = run_eval_auto._summary_line(summary)
    assert "原 20 题" in line
    assert "路由" not in line
