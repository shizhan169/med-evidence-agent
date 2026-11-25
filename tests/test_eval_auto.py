"""评测自动化测试：不需要真 LLM、不需要 Chroma、不联网。
跑法（仓库根目录，先激活环境）：
    pytest tests/test_eval_auto.py -v

思路：只测对比和判定的纯函数（compare_records / stop_rule_verdict /
build_compare_markdown），用合成的两轮数据；真的跑评测要等用户本地
跑 python eval/run_eval_auto.py。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import run_eval_auto  # noqa: E402


def _rec(qid, section="original", qtype="可回答", refused=False, hit=True, error=""):
    return {"id": qid, "section": section, "type": qtype,
            "refused": refused, "citation_hit": hit, "error": error}


def _summary(answered=13, acc=0.923, refuse_ok=5, fab=0, fig_answered=4):
    return {
        "original": {"answerable_answered": answered, "answerable_total": 15,
                     "citation_accuracy": acc, "citation_hits": 12,
                     "should_refuse_correct": refuse_ok, "should_refuse_total": 5,
                     "fabrication_count": fab},
        "figure": {"total": 8, "answerable_answered": fig_answered, "answerable_total": 6,
                   "citation_accuracy": 1.0, "citation_hits": 4,
                   "should_refuse_correct": 2, "should_refuse_total": 2,
                   "fabrication_count": 0},
    }


def test_compare_answered_to_refused_is_regression():
    prev = [_rec("Q01"), _rec("Q02")]
    new = [_rec("Q01", refused=True), _rec("Q02")]
    comp = run_eval_auto.compare_records(prev, new)
    assert [r["id"] for r in comp["regressions"]] == ["Q01"]
    assert "上轮答出，这轮拒答" in comp["regressions"][0]["description"]
    assert comp["improvements"] == []


def test_compare_citation_hit_lost_is_regression():
    prev = [_rec("Q01", hit=True)]
    new = [_rec("Q01", hit=False)]
    comp = run_eval_auto.compare_records(prev, new)
    assert [r["id"] for r in comp["regressions"]] == ["Q01"]
    assert "引用命中变未命中" in comp["regressions"][0]["description"]


def test_compare_refused_to_answered_is_improvement():
    prev = [_rec("Q08", refused=True, hit=False)]
    new = [_rec("Q08", refused=False, hit=True)]
    comp = run_eval_auto.compare_records(prev, new)
    assert [r["id"] for r in comp["improvements"]] == ["Q08"]
    assert "上轮拒答，这轮答出" in comp["improvements"][0]["description"]
    assert comp["regressions"] == []


def test_compare_should_refuse_answered_is_regression():
    # 应拒答题：上轮正确拒答，这轮回答了 = 编造，是退步
    prev = [_rec("R01", qtype="应拒答", refused=True, hit=False)]
    new = [_rec("R01", qtype="应拒答", refused=False, hit=False)]
    comp = run_eval_auto.compare_records(prev, new)
    assert [r["id"] for r in comp["regressions"]] == ["R01"]
    assert "编造" in comp["regressions"][0]["description"]


def test_compare_unchanged_is_neither():
    prev = [_rec("Q01"), _rec("Q02", refused=True, hit=False)]
    new = [_rec("Q01"), _rec("Q02", refused=True, hit=False)]
    comp = run_eval_auto.compare_records(prev, new)
    assert comp["regressions"] == [] and comp["improvements"] == []


def test_compare_new_question_listed_as_improvement():
    prev = [_rec("Q01")]
    new = [_rec("Q01"), _rec("Q99")]
    comp = run_eval_auto.compare_records(prev, new)
    assert [r["id"] for r in comp["improvements"]] == ["Q99"]


def test_stop_rule_passes():
    passed, reasons = run_eval_auto.stop_rule_verdict(_summary(answered=13), [])
    assert passed is True and reasons == []


def test_stop_rule_fails_when_too_few_answered():
    passed, reasons = run_eval_auto.stop_rule_verdict(_summary(answered=11), [])
    assert passed is False
    assert "11/15" in reasons[0] and "低于 12" in reasons[0]


def test_stop_rule_fails_on_original_regression():
    regressions = [{"id": "Q05", "section": "original", "description": "Q05：上轮答出，这轮拒答"}]
    passed, reasons = run_eval_auto.stop_rule_verdict(_summary(answered=13), regressions)
    assert passed is False
    assert "Q05" in reasons[0]


def test_stop_rule_figure_only_regression_does_not_fail():
    # 图表题的退步只列出来、不单独判不通过（图表题还在基线阶段）
    regressions = [{"id": "F02", "section": "figure", "description": "F02：上轮答出，这轮拒答"}]
    passed, reasons = run_eval_auto.stop_rule_verdict(_summary(answered=13), regressions)
    assert passed is True and reasons == []


def test_build_compare_markdown():
    prev = _summary(answered=13, acc=0.923)
    new = _summary(answered=12, acc=0.916)
    comp = {"regressions": [{"id": "Q05", "section": "original", "description": "Q05：上轮答出，这轮拒答"}],
            "improvements": [{"id": "Q08", "section": "original", "description": "Q08：上轮拒答，这轮答出"}]}
    passed, reasons = run_eval_auto.stop_rule_verdict(new, comp["regressions"])
    md = run_eval_auto.build_compare_markdown(prev, new, comp, passed, reasons)
    assert "## 结论：不通过" in md
    assert "Q05：上轮答出，这轮拒答" in md
    assert "Q08：上轮拒答，这轮答出" in md
    assert "上一轮：" in md and "本轮：" in md
