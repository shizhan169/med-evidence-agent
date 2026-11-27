"""评测自动化：一键跑 28 题评测，自动和上一轮对比、按停表规则判通过/不通过。

用法（在仓库根目录，先激活 medevidence 环境）：
    python eval/run_eval_auto.py

流程：
  1. 先把现有的 eval/results.json 读出来（上一轮），备份成 eval/results_prev.json；
     没有上一轮就只报本轮数字、不对比。
  2. 跑 eval/run_eval.py（28 道题，写新的 results.json + results.md）。
  3. 逐题对比：哪些题退步了（上轮答出、这轮拒答，或引用命中变未命中）、
     哪些进步了；汇总对比原 20 题和图表题。
  4. 按停表规则判：原 20 题可回答已答 >= 12/15，且原 20 题没有新的严重退步
     （上轮答出、这轮拒答，或引用命中 True→False）→ 通过，退出码 0；
     否则不通过，退出码 1。
  5. 对比报告写进 eval/results_compare.md，终端也打印一份。

注意：完整评测只能在本地跑（需要 bge-m3 和 DeepSeek key），所以这不是
GitHub Actions——GitHub 上跑不了。是本地一键 + 对比报告。

新维度（2.4 扩展）：逐题对比也看路由正确性、引用状态和复杂任务完成度，
老 results.json 没有这些字段时跳过，不误报。停表规则的“严重退步”现在也包括
原 20 题的新维度退步（路由对变错、引用完全命中变不完全、完成度下降），
因为路由和规划是框架的核心能力；图表题和多轮序列的退步只列出来，不单独判不通过。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run_eval  # noqa: E402

RESULTS_JSON = ROOT / "eval" / "results.json"
RESULTS_PREV_JSON = ROOT / "eval" / "results_prev.json"
COMPARE_MD = ROOT / "eval" / "results_compare.md"

# 停表规则：原 20 题可回答已答 >= 12/15
STOP_MIN_ANSWERED = 12


def _question_change(prev: dict, new: dict):
    """一题两轮的变化。返回 (kind, description)：
    kind 是 'regression'（退步）/ 'improvement'（进步）/ None（没变）。"""
    qid = new["id"]
    if new.get("error") and not prev.get("error"):
        return "regression", f"{qid}：上轮正常，这轮运行出错（{str(new['error'])[:40]}）"
    if prev.get("error") and not new.get("error"):
        return "improvement", f"{qid}：上轮运行出错，这轮正常"
    if prev.get("error") and new.get("error"):
        return None, ""
    if prev["type"] == "应拒答":
        if prev["refused"] and not new["refused"]:
            return "regression", f"{qid}：上轮正确拒答，这轮回答了（编造）"
        if not prev["refused"] and new["refused"]:
            return "improvement", f"{qid}：上轮编造，这轮正确拒答"
        return None, ""
    # 可回答题
    if not prev["refused"] and new["refused"]:
        return "regression", f"{qid}：上轮答出，这轮拒答"
    if prev["refused"] and not new["refused"]:
        return "improvement", f"{qid}：上轮拒答，这轮答出"
    if not prev["refused"] and not new["refused"]:
        if prev["citation_hit"] and not new["citation_hit"]:
            return "regression", f"{qid}：引用命中变未命中"
        if not prev["citation_hit"] and new["citation_hit"]:
            return "improvement", f"{qid}：引用未命中变命中"
    # 新维度（只有两轮都有这个字段时才比；老 results.json 没有这些字段时跳过）
    status_cn = {"full": "完全命中", "partial": "不完全命中", "miss": "未命中"}
    comp_cn = {"complete": "完全完成", "partial": "部分完成", "incomplete": "未完成"}
    prev_route = prev.get("route_correct")
    new_route = new.get("route_correct")
    if prev_route is True and new_route is False:
        return "regression", f"{qid}：路由对变错"
    if prev_route is False and new_route is True:
        return "improvement", f"{qid}：路由错变对"
    prev_status = prev.get("citation_status")
    new_status = new.get("citation_status")
    if prev_status and new_status and prev_status != new_status:
        order = {"full": 2, "partial": 1, "miss": 0}
        if order[new_status] < order[prev_status]:
            return "regression", f"{qid}：引用从{status_cn[prev_status]}变{status_cn[new_status]}"
        if order[new_status] > order[prev_status]:
            return "improvement", f"{qid}：引用从{status_cn[prev_status]}变{status_cn[new_status]}"
    prev_comp = prev.get("task_completion")
    new_comp = new.get("task_completion")
    if prev_comp and new_comp and prev_comp != new_comp:
        order = {"complete": 2, "partial": 1, "incomplete": 0}
        if order[new_comp] < order[prev_comp]:
            return "regression", f"{qid}：复杂任务完成度从{comp_cn[prev_comp]}变{comp_cn[new_comp]}"
        if order[new_comp] > order[prev_comp]:
            return "improvement", f"{qid}：复杂任务完成度从{comp_cn[prev_comp]}变{comp_cn[new_comp]}"
    return None, ""


def compare_records(prev_records: list, new_records: list) -> dict:
    """逐题对比两轮。返回 {"regressions": [...], "improvements": [...]}，
    每条是 {"id", "section", "description"}。"""
    prev_by_id = {r["id"]: r for r in prev_records}
    regressions, improvements = [], []
    for new in new_records:
        prev = prev_by_id.get(new["id"])
        if prev is None:
            improvements.append({
                "id": new["id"], "section": new["section"],
                "description": f"{new['id']}：新题（上轮没有）",
            })
            continue
        kind, desc = _question_change(prev, new)
        entry = {"id": new["id"], "section": new["section"], "description": desc}
        if kind == "regression":
            regressions.append(entry)
        elif kind == "improvement":
            improvements.append(entry)
    return {"regressions": regressions, "improvements": improvements}


def stop_rule_verdict(new_summary: dict, regressions: list):
    """按停表规则判：原 20 题可回答已答 >= 12/15，且原 20 题没有新的严重退步。
    严重退步 = 原 20 题的退步（拒答、引用命中丢了、应拒答题被回答了）；
    图表题的退步只列出来、不单独判不通过（图表题还在基线阶段）。
    返回 (passed, reasons)，reasons 是不通过的原因列表（通过时为空）。"""
    reasons = []
    orig = new_summary.get("original", {})
    answered = orig.get("answerable_answered", 0)
    if answered < STOP_MIN_ANSWERED:
        reasons.append(f"原 20 题可回答已答 {answered}/15，低于 {STOP_MIN_ANSWERED} 道")
    serious = [r for r in regressions if r["section"] == "original"]
    if serious:
        reasons.append("原 20 题新增严重退步：\n" + "\n".join(f"  - {r['description']}" for r in serious))
    return (not reasons), reasons


def _summary_line(summary: dict) -> str:
    orig, fig = summary.get("original", {}), summary.get("figure", {})
    base = (
        f"原 20 题：可回答已答 {orig.get('answerable_answered')}/{orig.get('answerable_total')}，"
        f"引用准确率 {orig.get('citation_accuracy', 0):.1%}，"
        f"应拒答 {orig.get('should_refuse_correct')}/{orig.get('should_refuse_total')}，"
        f"编造 {orig.get('fabrication_count')} 次；"
        f"图表 {fig.get('total')} 题：可回答已答 {fig.get('answerable_answered')}/{fig.get('answerable_total')}，"
        f"引用准确率 {fig.get('citation_accuracy', 0):.1%}，"
        f"应拒答 {fig.get('should_refuse_correct')}/{fig.get('should_refuse_total')}，"
        f"编造 {fig.get('fabrication_count')} 次"
    )
    router = summary.get("router") or {}
    planning = summary.get("planning") or {}
    vlm = summary.get("vlm") or {}
    mt = summary.get("multiturn") or {}
    extra = []
    if router:
        extra.append(f"路由 {router.get('correct')}/{router.get('total')}，准确率 {router.get('accuracy', 0):.1%}")
    if planning:
        extra.append(f"planning 完全完成 {planning.get('complete')}/{planning.get('total')}")
    if vlm:
        extra.append(f"VLM 引用准确率 {vlm.get('citation_accuracy', 0):.1%}，vision 路由 {vlm.get('vision_routed')}/{vlm.get('total')}")
    if mt:
        extra.append(f"多轮 {mt.get('total_turns')} 轮")
    if extra:
        base += "；" + "；".join(extra)
    return base


def build_compare_markdown(prev_summary: dict, new_summary: dict, comparison: dict,
                            passed: bool, reasons: list) -> str:
    """对比报告（写进 eval/results_compare.md 的全文）。"""
    lines = [
        "# 评测对比（上一轮 vs 本轮）",
        "",
        f"## 结论：{'通过' if passed else '不通过'}",
        "",
    ]
    if passed:
        lines.append("停表规则：原 20 题可回答已答 >= 12/15，且原 20 题没有新的严重退步。")
    else:
        lines.append("停表规则不通过，原因：")
        lines += [f"- {r}" for r in reasons]
    lines += [
        "",
        "## 汇总对比",
        "",
        f"- 上一轮：{_summary_line(prev_summary)}",
        f"- 本轮：{_summary_line(new_summary)}",
        "",
        "## 退步",
        "",
    ]
    if comparison["regressions"]:
        lines += [f"- {r['description']}" for r in comparison["regressions"]]
    else:
        lines.append("- 无")
    lines += ["", "## 进步", ""]
    if comparison["improvements"]:
        lines += [f"- {r['description']}" for r in comparison["improvements"]]
    else:
        lines.append("- 无")
    lines += [
        "",
        "## 说明",
        "",
        "- 引用未命中不一定是编造，可能是检索没找到对应片段，需结合回答原文人工判断。",
        "- 完整回答原文见 eval/results.json；上一轮见 eval/results_prev.json。",
        "",
    ]
    return "\n".join(lines)


def main():
    prev = None
    if RESULTS_JSON.exists():
        prev = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        RESULTS_PREV_JSON.write_text(
            json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"上一轮结果已备份到 {RESULTS_PREV_JSON}")
    else:
        print("没有上一轮结果（eval/results.json 不存在），只跑本轮、不对比。")
    # 跑评测（写新的 results.json + results.md）
    run_eval.main()
    new = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    new_summary = new["summary"]
    print()
    print("本轮：" + _summary_line(new_summary))
    if prev is None:
        print("没有上一轮可对比。")
        return 0
    prev_summary = prev["summary"]
    print("上一轮：" + _summary_line(prev_summary))
    comparison = compare_records(prev["records"], new["records"])
    passed, reasons = stop_rule_verdict(new_summary, comparison["regressions"])
    print()
    if comparison["regressions"]:
        print("退步：")
        for r in comparison["regressions"]:
            print(f"  - {r['description']}")
    else:
        print("退步：无")
    if comparison["improvements"]:
        print("进步：")
        for r in comparison["improvements"]:
            print(f"  - {r['description']}")
    else:
        print("进步：无")
    print()
    if passed:
        print("结论：通过（停表规则：原 20 题可回答已答 >= 12/15，且无新的严重退步）")
    else:
        print("结论：不通过")
        for r in reasons:
            print(f"  - {r}")
    COMPARE_MD.write_text(
        build_compare_markdown(prev_summary, new_summary, comparison, passed, reasons),
        encoding="utf-8")
    print(f"对比报告已写入 {COMPARE_MD}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
