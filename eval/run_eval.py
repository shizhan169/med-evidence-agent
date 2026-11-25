"""评测运行脚本：逐题调用 app.agent.run，输出结果与中文总结。

用法（在仓库根目录，先激活 medevidence 环境）：
    python eval/run_eval.py

题库 28 道：原 20 道（Q01–Q15 可回答、R01–R05 应拒答）+ 图表 8 道
（F01–F06 可回答、FR01–FR02 应拒答）。汇总分开报两组数字（原 20 题和
图表 8 题），口径一致；results.json 里总体 + original + figure 三层。

输出：
    eval/results.json  每题完整记录（问题、回答、引用、核查结果）
    eval/results.md    中文总结：总数、已答/拒答、应拒答正确数、引用准确率、逐题表

指标定义：
    已答 / 拒答：回答是否以“根据现有文献无法回答”开头（不带句号：
    模型有时用分号、冒号或直接接解释，如“根据现有文献无法回答；……”、
    “根据现有文献无法回答具体‘快多少’”，这些都是拒答）。
    应拒答答对：应拒答题被正确拒答的题数。
    引用命中：可回答题返回的引用里包含预期来源（论文 + 页码）。
    注意 run() 返回的引用是检索到的片段列表，所以引用命中实际度量的是
    检索是否召回了预期来源，不是答案里写的引用是否正确。
    引用准确率：引用命中的可回答题数 / 已回答的可回答题数。
    编造次数：应拒答题被回答的次数（明确无依据却给出回答），目标 0 次。
    答对率：需人工核对回答与预期答案是否一致，脚本不自动判，
    核对时看 results.md 逐题表里的回答原文。
"""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import REFUSAL, run  # noqa: E402

QUESTIONS_FILE = ROOT / "eval" / "test_questions.md"
RESULTS_JSON = ROOT / "eval" / "results.json"
RESULTS_MD = ROOT / "eval" / "results.md"

BLOCK_RE = re.compile(r"^###\s+(Q\d+|R\d+|FR\d+|F\d+)\s*$", re.M)
FIELD_RE = re.compile(r"^(问题|类型|预期答案|预期来源|拒答原因)：(.*)$")
SOURCE_RE = re.compile(r"^(.+\.pdf)\s+p\.(.+)$")  # 页码部分容忍 "3、p.5" / "3、5" 两种写法


def parse_questions(path: Path) -> list:
    """解析 test_questions.md，返回题目列表（dict）。"""
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
        q = {
            "id": qid,
            "section": "figure" if qid.startswith("F") else "original",
            "question": fields.get("问题", ""),
            "type": fields.get("类型", ""),
            "expected_answer": fields.get("预期答案", ""),
            "expected_source": fields.get("预期来源", ""),
            "refuse_reason": fields.get("拒答原因", ""),
        }
        m = SOURCE_RE.match(q["expected_source"])
        if m:
            q["expected_file"] = m.group(1)
            q["expected_pages"] = [int(n) for n in re.findall(r"\d+", m.group(2))]
        else:
            q["expected_file"] = ""
            q["expected_pages"] = []
        questions.append(q)
    return questions


def citation_hit(citations: list, expected_file: str, expected_pages: list) -> bool:
    """返回的引用里是否包含预期来源（论文 + 任一预期页码）。"""
    if not expected_file or not expected_pages:
        return False
    for c in citations:
        if c.get("source") == expected_file and int(c.get("page", -1)) in expected_pages:
            return True
    return False


def section_summary(records: list) -> dict:
    """一节题目的汇总指标（原 20 题和图表题共用同一口径）。

    口径和原来完全一致：引用准确率 = 引用命中的可回答题数 / 已回答的可回答题数；
    编造次数 = 应拒答题被回答的次数。
    """
    total = len(records)
    answered = [r for r in records if not r["refused"] and not r["error"]]
    refused = [r for r in records if r["refused"]]
    errors = [r for r in records if r["error"]]
    answerable = [r for r in records if r["type"] == "可回答"]
    should_refuse = [r for r in records if r["type"] == "应拒答"]
    correct_refusals = [r for r in should_refuse if r["refused"]]
    fabrication = len(should_refuse) - len(correct_refusals)  # 应拒答题被回答 = 编造
    answered_answerable = [r for r in answerable if not r["refused"] and not r["error"]]
    hits = [r for r in answered_answerable if r["citation_hit"]]
    citation_acc = len(hits) / len(answered_answerable) if answered_answerable else 0.0
    return {
        "total": total,
        "answered": len(answered),
        "refused": len(refused),
        "errors": len(errors),
        "answerable_total": len(answerable),
        "answerable_answered": len(answered_answerable),
        "should_refuse_total": len(should_refuse),
        "should_refuse_correct": len(correct_refusals),
        "fabrication_count": fabrication,
        "citation_hits": len(hits),
        "citation_accuracy": round(citation_acc, 4),
    }


def _table_lines(records: list) -> list:
    """一节题目的逐题表（markdown 表格行）。"""
    lines = [
        "| 题号 | 类型 | 结果 | 引用命中 | 回答摘要 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in records:
        if r["error"]:
            result_text, hit_text = "运行出错", "-"
        elif r["refused"]:
            result_text, hit_text = "拒答", "-"
        else:
            result_text = "已回答"
            hit_text = "命中" if r["citation_hit"] else "未命中"
        brief = r["answer"].replace("\n", " ").replace("|", "｜")[:60]
        lines.append(f"| {r['id']} | {r['type']} | {result_text} | {hit_text} | {brief} |")
    return lines


def main():
    questions = parse_questions(QUESTIONS_FILE)
    print(f"共加载 {len(questions)} 道题，开始评测……")
    records = []
    t0 = time.time()
    for idx, q in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] {q['id']}：{q['question']}")
        try:
            result = run(q["question"])
            answer = result.get("answer", "")
            verified = result.get("verified", False)
            citations = result.get("citations", [])
            error = ""
        except Exception as e:  # 单题失败不中断整轮评测
            answer, verified, citations, error = "", False, [], str(e)
            print(f"    运行出错：{error}")
        refused = answer.strip().startswith(REFUSAL.rstrip("。"))
        hit = citation_hit(citations, q["expected_file"], q["expected_pages"]) if not refused else False
        records.append({
            "id": q["id"],
            "section": q["section"],
            "type": q["type"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "expected_source": q["expected_source"],
            "refuse_reason": q["refuse_reason"],
            "answer": answer,
            "verified": verified,
            "refused": refused,
            "citation_hit": hit,
            "citations": [{"source": c.get("source"), "page": c.get("page")} for c in citations],
            "error": error,
        })
        print(f"    结果：{'拒答' if refused else '已回答'}，核查通过：{verified}，引用命中：{hit}")
    elapsed = time.time() - t0

    # ---- 汇总指标：总体 + 原 20 题 + 图表 8 题（两组分开报，口径一致） ----
    original_records = [r for r in records if r["section"] == "original"]
    figure_records = [r for r in records if r["section"] == "figure"]
    summary = section_summary(records)
    summary["elapsed_seconds"] = round(elapsed, 1)
    summary["original"] = section_summary(original_records)
    summary["figure"] = section_summary(figure_records)
    orig, fig = summary["original"], summary["figure"]

    RESULTS_JSON.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 评测结果",
        "",
        f"评测题数：{summary['total']} 道（原 20 题 + 图表 {fig['total']} 题）",
        f"用时：{elapsed:.1f} 秒",
        "",
        "## 汇总（总体）",
        "",
        f"- 已答：{summary['answered']} 道，拒答：{summary['refused']} 道" + (f"，运行出错：{summary['errors']} 道" if summary['errors'] else ""),
        f"- 应拒答题答对（正确拒答）：{summary['should_refuse_correct']} / {summary['should_refuse_total']}",
        f"- 编造次数（应拒答题被回答）：{summary['fabrication_count']} 次，目标 0 次",
        f"- 可回答题已答：{summary['answerable_answered']} 道，其中引用命中：{summary['citation_hits']} 道",
        f"- 引用准确率：{summary['citation_accuracy']:.1%}（引用命中 / 可回答题已答数）",
        "- 答对率：需人工核对（对照下表每题的回答原文与预期答案），脚本不自动判",
        "",
        "## 原 20 题",
        "",
        f"- 可回答题已答：{orig['answerable_answered']} / {orig['answerable_total']}，其中引用命中：{orig['citation_hits']} 道",
        f"- 引用准确率：{orig['citation_accuracy']:.1%}",
        f"- 应拒答题答对（正确拒答）：{orig['should_refuse_correct']} / {orig['should_refuse_total']}",
        f"- 编造次数：{orig['fabrication_count']} 次",
        "",
        "## 图表题",
        "",
        f"- 可回答题已答：{fig['answerable_answered']} / {fig['answerable_total']}，其中引用命中：{fig['citation_hits']} 道",
        f"- 引用准确率：{fig['citation_accuracy']:.1%}",
        f"- 应拒答题答对（正确拒答）：{fig['should_refuse_correct']} / {fig['should_refuse_total']}",
        f"- 编造次数：{fig['fabrication_count']} 次",
        "",
        "## 逐题表（原 20 题）",
        "",
        *_table_lines(original_records),
        "",
        "## 逐题表（图表题）",
        "",
        *_table_lines(figure_records),
        "",
        "## 说明",
        "",
        "- 引用未命中不一定是编造，可能是检索没找到对应片段，需结合回答原文人工判断。",
        "- 完整回答原文见 eval/results.json。",
        "",
    ]
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("评测完成：")
    print(f"  总数 {summary['total']}，已答 {summary['answered']}，拒答 {summary['refused']}" + (f"，出错 {summary['errors']}" if summary['errors'] else ""))
    print(f"  原 20 题：可回答已答 {orig['answerable_answered']}/{orig['answerable_total']}，"
          f"引用准确率 {orig['citation_accuracy']:.1%}（{orig['citation_hits']}/{orig['answerable_answered']}），"
          f"应拒答 {orig['should_refuse_correct']}/{orig['should_refuse_total']}，编造 {orig['fabrication_count']} 次")
    print(f"  图表 {fig['total']} 题：可回答已答 {fig['answerable_answered']}/{fig['answerable_total']}，"
          f"引用准确率 {fig['citation_accuracy']:.1%}（{fig['citation_hits']}/{fig['answerable_answered']}），"
          f"应拒答 {fig['should_refuse_correct']}/{fig['should_refuse_total']}，编造 {fig['fabrication_count']} 次")
    print(f"  结果已写入 {RESULTS_JSON}")
    print(f"  总结已写入 {RESULTS_MD}")


if __name__ == "__main__":
    main()
