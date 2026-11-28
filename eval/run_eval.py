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
    已答 / 拒答：回答的第一句里是否出现"无法回答"。
    不用固定开头匹配：模型有时会在"根据现有文献"和"无法回答"之间
    插词（如"根据现有文献片段无法回答……"），开头匹配会漏判（第七轮
    Q09 就是这么漏的）。只看第一句，是因为已答题的后半句也可能出现
    "无法回答"（如 Q03 第五轮"峰值后至 50 岁前的具体变化，根据现有文献
    片段无法回答"），那是部分拒答不是整题拒答。
    应拒答答对：应拒答题被正确拒答的题数。
    引用命中：可回答题返回的引用里包含预期来源（论文 + 页码）。
    注意 run() 返回的引用是检索到的片段列表，所以引用命中实际度量的是
    检索是否召回了预期来源，不是答案里写的引用是否正确。
    引用准确率：引用命中的可回答题数 / 已回答的可回答题数。
    编造次数：应拒答题被回答的次数（明确无依据却给出回答），目标 0 次。
    答对率：需人工核对回答与预期答案是否一致，脚本不自动判，
    核对时看 results.md 逐题表里的回答原文。

新维度（2.4 扩展）：
    路由正确性：每题单独调一次 router_node 拿实际路由，和期望路由
    （Q04/Q12 走 planning，R05 走外部搜索，F/FR 题走 vision，其余简单问答）
    四项 flag 全对才算对。期望路由写在 expected_route 里。
    复杂任务完成度：planning 题按 answer_status 分 complete / partial /
    incomplete，partial 时列出失败的子任务。
    VLM 图表理解：图表题的答出率、引用准确率、走 vision 路由的比例。
    当前 chart_read 还是占位，这里是图注和文本路径的基线。
    跨文档多来源引用：预期来源支持多来源（题库里用分号隔开，或写在
    SECONDARY_SOURCES 里）。citation_status 是 full / partial / miss，
    只引了一篇就是 partial，不算完全命中。citation_hit 只在 full 时为真。
    多轮序列：MULTITURN_SEQUENCES 里定义追问题串，按顺序跑，每轮把历史
    传给 agent。结果单独记在 section=multiturn 的记录里。
    耗时统计：每题耗时记在记录里，汇总给总耗时、平均、最慢 3 道。
    Token 数 run() 不返回，精确值要看 LangSmith trace。
    工具选择：每题记期望工具路径（tool_path），供人工判断；严格核查
    需要 run() 返回实际调了哪些工具，那是 app 层的改动，这里不做。
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 直接运行脚本时，eval/datasets 会被识别为顶层 datasets 包，
# 使 sentence-transformers 误以为已安装 Hugging Face datasets。
if str(ROOT / "eval") in sys.path:
    sys.path.remove(str(ROOT / "eval"))
sys.path.insert(0, str(ROOT))

from app.agent import router_node, run  # noqa: E402

QUESTIONS_FILE = ROOT / "eval" / "test_questions.md"
RESULTS_JSON = ROOT / "eval" / "results.json"
RESULTS_MD = ROOT / "eval" / "results.md"
# 冻结测试题（阶段验收专用）：T01–T10，只在阶段验收时跑一次。
# 单独的题目文件和结果文件，不覆盖开发集的 results.json。
FROZEN_QUESTIONS_FILE = ROOT / "eval" / "datasets" / "test_questions_frozen.md"
FROZEN_RESULTS_JSON = ROOT / "eval" / "results_frozen.json"
FROZEN_RESULTS_MD = ROOT / "eval" / "results_frozen.md"

BLOCK_RE = re.compile(r"^###\s+(Q\d+|R\d+|FR\d+|F\d+|T\d+)\s*$", re.M)
FIELD_RE = re.compile(r"^(问题|类型|预期答案|预期来源|拒答原因)：(.*)$")
SOURCE_RE = re.compile(r"^(.+\.pdf)\s+p\.(.+)$")  # 页码部分容忍 "3、p.5" / "3、5" 两种写法


def parse_sources(source_text: str) -> list:
    """解析预期来源文本，返回来源列表，每个是 {"file": 文件名, "pages": [页码]}。
    支持多来源：用分号隔开，例如 "a.pdf p.1; b.pdf p.2、p.3"。
    单来源就是只有一个元素的列表；空文本返回空列表。"""
    sources = []
    if not source_text:
        return sources
    for part in re.split(r"[;；]", source_text):
        part = part.strip()
        if not part:
            continue
        m = SOURCE_RE.match(part)
        if m:
            sources.append({
                "file": m.group(1),
                "pages": [int(n) for n in re.findall(r"\d+", m.group(2))],
            })
    return sources


# 跨文档题的次要来源。题库文件头的说明：Q18 的第二篇是 Pomponio 2020 p.2-3，
# Q19 的第二篇是 Ho 2026 p.2。题库的“预期来源”字段目前只写主来源，
# 所以次要来源先写在这里；等题库改成“来源1; 来源2”格式后，parse_sources
# 会直接读出来，这里只补还没写进题库的。
SECONDARY_SOURCES = {
    "Q18": [{"file": "2020-pomponio-harmonization-mri-datasets.pdf", "pages": [2, 3]}],
    "Q19": [{"file": "2026-ho-unified-multi-cohort-harmonisation.pdf", "pages": [2]}],
}

# 路由预期：按框架设计，哪些题应该走哪条路。
# Q04/Q12 要拆解；R05 要查外部；F/FR 题要读图；其余是简单事实问答。
PLANNING_QUESTIONS = {"Q04", "Q12"}
EXTERNAL_QUESTIONS = {"R05"}


def expected_route(qid: str) -> dict:
    """这道题期望的路由（四项 flag）。"""
    if qid in PLANNING_QUESTIONS:
        return {"task_type": "complex", "requires_planning": True,
                "requires_external_search": False, "requires_vision": False}
    if qid in EXTERNAL_QUESTIONS:
        return {"task_type": "simple", "requires_planning": False,
                "requires_external_search": True, "requires_vision": False}
    if qid.startswith("F"):
        return {"task_type": "simple", "requires_planning": False,
                "requires_external_search": False, "requires_vision": True}
    return {"task_type": "simple", "requires_planning": False,
            "requires_external_search": False, "requires_vision": False}


def route_correct(actual: dict, expected: dict):
    """实际路由和期望路由是否一致（四项 flag 全对）。
    actual 为空表示没拿到路由，返回 None（不计入统计）。"""
    if not actual:
        return None
    return all(actual.get(k) == v for k, v in expected.items())


def task_completion(answer_status: str, verify_failures=None):
    """复杂任务完成度（只对 planning 题有意义）。
    verified 是 complete；partial 是 partial；refused/error 是 incomplete。
    返回 (completion, failed_subtask_ids)。"""
    if answer_status == "verified":
        return "complete", []
    if answer_status == "partial":
        failed = [f.get("subtask_id", "") for f in (verify_failures or [])]
        return "partial", failed
    return "incomplete", []


def expected_tool_path(route: dict) -> str:
    """按路由，这道题应该走什么工具路径（只作记录，供人工判断工具选择是否合理）。
    run() 不返回实际调了哪些工具，所以这里只能记期望路径，不能严格核查。"""
    if not route:
        return "unknown"
    if route.get("requires_planning"):
        return "planner+executor"
    if route.get("requires_external_search"):
        return "pubmed_search"
    if route.get("requires_vision"):
        return "chart_locate/chart_read"
    return "local_search"


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
            "section": "frozen" if qid.startswith("T") else ("figure" if qid.startswith("F") else "original"),
            "question": fields.get("问题", ""),
            "type": fields.get("类型", ""),
            "expected_answer": fields.get("预期答案", ""),
            "expected_source": fields.get("预期来源", ""),
            "refuse_reason": fields.get("拒答原因", ""),
        }
        q["expected_sources"] = parse_sources(q["expected_source"])
        # 跨文档题补次要来源：还没写进题库的才补，已写分号格式的不重复。
        for extra in SECONDARY_SOURCES.get(qid, []):
            if extra["file"] not in [s["file"] for s in q["expected_sources"]]:
                q["expected_sources"].append(extra)
        if q["expected_sources"]:
            q["expected_file"] = q["expected_sources"][0]["file"]
            q["expected_pages"] = q["expected_sources"][0]["pages"]
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


def citation_status(citations: list, expected_sources: list):
    """多来源引用核查。返回 (status, per_source)。
    status 是 full / partial / miss：
    - full：每个预期来源都命中（至少引到该来源的一个预期页）
    - partial：命中一部分来源（跨文档题只引了一篇就是这种情况）
    - miss：一个来源都没命中
    per_source 是每个预期来源的命中情况。
    单来源题只有 full / miss 两种，和原来 citation_hit 的口径一致。"""
    per_source = []
    for src in expected_sources:
        hit = citation_hit(citations, src["file"], src["pages"])
        per_source.append({"file": src["file"], "pages": src["pages"], "hit": hit})
    hits = sum(1 for s in per_source if s["hit"])
    if not per_source or hits == 0:
        return "miss", per_source
    if hits == len(per_source):
        return "full", per_source
    return "partial", per_source


def is_refusal(answer: str) -> bool:
    """答案是不是拒答。

    标志是第一句里出现"无法回答"。不用固定开头匹配：模型有时会在
    "根据现有文献"和"无法回答"之间插词（如"根据现有文献片段无法回答……"），
    开头匹配会漏判（第七轮 Q09）。只看第一句，是因为已答题的后半句也
    可能出现"无法回答"（如 Q03 第五轮"峰值后至 50 岁前的具体变化，根据
    现有文献片段无法回答"），那是部分拒答不是整题拒答。
    """
    first = re.split(r"[。！？；\n]", answer.strip(), maxsplit=1)[0]
    return "无法回答" in first


def evaluate_result(q: dict, result: dict, route: dict, elapsed: float) -> dict:
    """把一次 run() 的结果算成一条评测记录。
    q 是题目 dict；result 是 run() 返回的 dict；route 是 router_node 返回的路由
    （拿不到时传空 dict）；elapsed 是这轮耗时（秒）。
    记录比原来多了：answer_status、citation_status、citation_sources、
    verify_failures、route、route_correct、task_completion、tool_path、elapsed_seconds。
    原有字段口径不变。"""
    answer = result.get("answer", "")
    verified = result.get("verified", False)
    answer_status = result.get("answer_status", "")
    citations = result.get("citations", [])
    verify_failures = result.get("verify_failures", [])
    refused = is_refusal(answer)
    if refused:
        status = "miss"
        per_source = [{"file": s["file"], "pages": s["pages"], "hit": False}
                      for s in q.get("expected_sources", [])]
        hit = False
    else:
        status, per_source = citation_status(citations, q.get("expected_sources", []))
        hit = (status == "full")
    exp_route = expected_route(q["id"])
    rc = route_correct(route, exp_route) if route else None
    completion, failed_subtasks = None, []
    if exp_route.get("requires_planning"):
        completion, failed_subtasks = task_completion(answer_status, verify_failures)
    return {
        "id": q["id"],
        "section": q["section"],
        "type": q["type"],
        "question": q["question"],
        "expected_answer": q["expected_answer"],
        "expected_source": q["expected_source"],
        "refuse_reason": q["refuse_reason"],
        "answer": answer,
        "verified": verified,
        "answer_status": answer_status,
        "refused": refused,
        "citation_hit": hit,
        "citation_status": status,
        "citation_sources": per_source,
        "verify_failures": verify_failures,
        "citations": [{"source": c.get("source"), "page": c.get("page")} for c in citations],
        "route": route or {},
        "route_correct": rc,
        "expected_route": exp_route,
        "task_completion": completion,
        "failed_subtasks": failed_subtasks,
        "tool_path": expected_tool_path(route) if route else "unknown",
        "elapsed_seconds": round(elapsed, 1),
        "error": "",
    }


def router_summary(records: list) -> dict:
    """路由正确性汇总：只统计拿到了路由的题（route_correct 不是 None 的）。"""
    judged = [r for r in records if r.get("route_correct") is not None]
    correct = [r for r in judged if r["route_correct"]]
    wrong = [r["id"] for r in judged if not r["route_correct"]]
    return {
        "total": len(judged),
        "correct": len(correct),
        "accuracy": round(len(correct) / len(judged), 4) if judged else 0.0,
        "wrong_ids": wrong,
    }


def planning_summary(records: list) -> dict:
    """复杂任务完成度汇总：只统计 planning 题（task_completion 不是 None 的）。"""
    planning = [r for r in records if r.get("task_completion") is not None]
    complete = [r for r in planning if r["task_completion"] == "complete"]
    partial = [r for r in planning if r["task_completion"] == "partial"]
    incomplete = [r for r in planning if r["task_completion"] == "incomplete"]
    return {
        "total": len(planning),
        "complete": len(complete),
        "partial": len(partial),
        "incomplete": len(incomplete),
        "partial_ids": [r["id"] for r in partial],
        "incomplete_ids": [r["id"] for r in incomplete],
    }


def vlm_summary(records: list) -> dict:
    """VLM 图表理解汇总：只统计图表题。
    注意当前 chart_read 还是占位，测的是图注和文本路径的图表理解基线；
    2.3 真 VLM 落地后，同一维度会反映 VLM 的效果。"""
    fig = [r for r in records if r["section"] == "figure"]
    answered = [r for r in fig if not r["refused"] and not r["error"]]
    hits = [r for r in answered if r["citation_hit"]]
    routed = [r for r in fig if (r.get("route") or {}).get("requires_vision")]
    return {
        "total": len(fig),
        "answered": len(answered),
        "citation_hits": len(hits),
        "citation_accuracy": round(len(hits) / len(answered), 4) if answered else 0.0,
        "vision_routed": len(routed),
        "vision_routing_rate": round(len(routed) / len(fig), 4) if fig else 0.0,
    }


def multiturn_summary(records: list) -> dict:
    """多轮序列汇总：只统计多轮记录，按序列 id 分组。"""
    mt = [r for r in records if r["section"] == "multiturn"]
    by_seq = {}
    for r in mt:
        seq_id = r["id"].split("-T")[0]
        by_seq.setdefault(seq_id, []).append(r)
    sequences = {}
    for seq_id, turns in by_seq.items():
        answered = [t for t in turns if not t["refused"] and not t["error"]]
        sequences[seq_id] = {
            "turns": len(turns),
            "answered": len(answered),
            "citation_hits": len([t for t in answered if t["citation_hit"]]),
            "turn_ids": [t["id"] for t in turns],
        }
    return {"total_turns": len(mt), "sequences": sequences}


def timing_summary(records: list) -> dict:
    """耗时统计：总耗时、平均每题、最慢的 3 道题。
    Token 数 run() 不返回，精确值要看 LangSmith trace，这里不估算。"""
    timed = [r for r in records if r.get("elapsed_seconds") is not None]
    total = round(sum(r["elapsed_seconds"] for r in timed), 1)
    slowest = sorted(timed, key=lambda r: r["elapsed_seconds"], reverse=True)[:3]
    return {
        "total_seconds": total,
        "mean_seconds": round(total / len(timed), 1) if timed else 0.0,
        "slowest": [{"id": r["id"], "seconds": r["elapsed_seconds"]} for r in slowest],
    }


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


# 多轮序列：真正的多轮追问。Q20 原来是“题面自带上文”的形式，
# 这里改成真多轮：第一轮问 Fujita 的纳入和随访，第二轮只追问总访视次数，
# 上文靠历史传，不再写进题面。加新序列往这个列表里添就行。
MULTITURN_SEQUENCES = [
    {
        "id": "MT01",
        "title": "Fujita 2023 多轮追问",
        "turns": [
            {
                "question": "Fujita 2023 研究纳入多少名认知正常者，随访多久，总访视多少次？",
                "type": "可回答",
                "expected_answer": "纳入 653 名认知正常者，平均随访 11.5 年，总访视 7915 次。",
                "expected_source": "2023-fujita-brain-volume-aging.pdf p.1、p.4",
            },
            {
                "question": "那么这项研究的总访视次数是多少？",
                "type": "可回答",
                "expected_answer": "总访视 7,915 次。",
                "expected_source": "2023-fujita-brain-volume-aging.pdf p.1、p.4",
            },
        ],
    },
]


def run_sequence(seq: dict, run_fn, router_fn=None) -> list:
    """跑一串多轮追问，返回每轮的评测记录。
    run_fn 是问答函数，签名 run_fn(question, history=..., use_pubmed=...)，
    返回 run() 同格式的 dict。router_fn 可选，用来拿每轮的路由。
    每轮把到目前为止的历史（user/assistant 成对）传给 run_fn，
    跑完把本轮问答追加进历史。拒答的轮次也追加，下一轮能看到它拒答了。"""
    history = []
    records = []
    for i, turn in enumerate(seq["turns"], 1):
        q = {
            "id": f"{seq['id']}-T{i}",
            "section": "multiturn",
            "type": turn.get("type", "可回答"),
            "question": turn["question"],
            "expected_answer": turn.get("expected_answer", ""),
            "expected_source": turn.get("expected_source", ""),
            "refuse_reason": turn.get("refuse_reason", ""),
            "expected_sources": parse_sources(turn.get("expected_source", "")),
        }
        t0 = time.time()
        try:
            result = run_fn(turn["question"], history=history, use_pubmed=False)
            error = ""
        except Exception as e:
            result = {"answer": "", "verified": False, "answer_status": "error",
                      "citations": [], "verify_failures": []}
            error = str(e)
        elapsed = time.time() - t0
        route = {}
        if router_fn is not None:
            try:
                route = router_fn({"query": turn["question"]}).get("route", {})
            except Exception:
                route = {}
        rec = evaluate_result(q, result, route, elapsed)
        if error:
            rec["error"] = error
        records.append(rec)
        history.append({"role": "user", "content": turn["question"]})
        history.append({"role": "assistant", "content": rec["answer"]})
    return records


def main(frozen: bool = False):
    # 冻结测试（--frozen）：只跑 T01–T10，结果写 results_frozen.json/md，
    # 不覆盖开发集的 results.json；多轮序列 MT01 是开发集专用的，冻结跑跳过。
    questions_file = FROZEN_QUESTIONS_FILE if frozen else QUESTIONS_FILE
    results_json = FROZEN_RESULTS_JSON if frozen else RESULTS_JSON
    results_md = FROZEN_RESULTS_MD if frozen else RESULTS_MD
    questions = parse_questions(questions_file)
    print(f"共加载 {len(questions)} 道题，开始评测……")
    records = []
    t0 = time.time()
    for idx, q in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] {q['id']}：{q['question']}")
        try:
            # 评测只测本地 RAG：use_pubmed=False 关掉 PubMed。28 道题里应拒答题
            # 本地本来就没证据，按触发规则会走 PubMed——数字会被外部服务带偏、
            # 还不可复现（PubMed 的内容和通不通都会变）。PubMed 能力由
            # tests/test_pubmed.py 的 mock 测试 + 一次真跑验收，不走 28 道题。
            t1 = time.time()
            result = run(q["question"], use_pubmed=False)
            error = ""
        except Exception as e:  # 单题失败不中断整轮评测
            result = {"answer": "", "verified": False, "answer_status": "error",
                      "citations": [], "verify_failures": []}
            error = str(e)
            print(f"    运行出错：{error}")
        elapsed_q = time.time() - t1
        route = {}
        try:
            route = router_node({"query": q["question"]}).get("route", {})
        except Exception as e:
            print(f"    路由检查失败：{e}")
            route = {}
        rec = evaluate_result(q, result, route, elapsed_q)
        if error:
            rec["error"] = error
        records.append(rec)
        print(f"    结果：{'拒答' if rec['refused'] else '已回答'}，核查通过：{rec['verified']}，"
              f"引用：{rec['citation_status']}，路由：{'对' if rec['route_correct'] else '错' if rec['route_correct'] is False else '未知'}")
    # ---- 多轮序列：按顺序跑追问题，每轮把历史传给 agent ----
    # 冻结测试没有多轮题，跳过（MT01 是开发集专用的）。
    if frozen:
        multiturn_sequences = []
    else:
        multiturn_sequences = MULTITURN_SEQUENCES
    for seq in multiturn_sequences:
        print(f"多轮序列 {seq['id']}：{seq['title']}")
        seq_records = run_sequence(seq, run, router_node)
        for sr in seq_records:
            print(f"    {sr['id']}：{'拒答' if sr['refused'] else '已回答'}，引用：{sr['citation_status']}")
        records.extend(seq_records)
    elapsed = time.time() - t0

    # ---- 汇总指标：总体 + 原 20 题 + 图表 8 题（两组分开报，口径一致） ----
    original_records = [r for r in records if r["section"] == "original"]
    figure_records = [r for r in records if r["section"] == "figure"]
    summary = section_summary(records)
    summary["elapsed_seconds"] = round(elapsed, 1)
    summary["original"] = section_summary(original_records)
    summary["figure"] = section_summary(figure_records)
    summary["router"] = router_summary(records)
    summary["planning"] = planning_summary(records)
    summary["vlm"] = vlm_summary(records)
    summary["multiturn"] = multiturn_summary(records)
    summary["timing"] = timing_summary(records)
    orig, fig = summary["original"], summary["figure"]
    router_s, planning_s, vlm_s, mt_s, timing_s = (
        summary["router"], summary["planning"], summary["vlm"],
        summary["multiturn"], summary["timing"],
    )

    results_json.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 跨文档题引用详情（Q18/Q19 这类多来源题，每个来源单独列命中情况）
    status_cn = {"full": "完全命中", "partial": "不完全命中", "miss": "未命中"}
    cross_doc_lines = []
    for r in records:
        if len(r.get("citation_sources", [])) > 1:
            parts = []
            for s in r["citation_sources"]:
                pages = "、".join(f"p.{p}" for p in s["pages"])
                parts.append(f"{s['file']} {pages}：{'命中' if s['hit'] else '未命中'}")
            cross_doc_lines.append(
                f"- {r['id']}：{status_cn.get(r['citation_status'], r['citation_status'])}；" + "；".join(parts))
    multiturn_records = [r for r in records if r["section"] == "multiturn"]
    mt_lines = [f"- {sid}：{s['answered']} / {s['turns']} 轮答出，引用命中 {s['citation_hits']} 轮"
                for sid, s in mt_s["sequences"].items()]
    slowest_text = "、".join(f"{s['id']} {s['seconds']} 秒" for s in timing_s["slowest"])

    lines = [
        "# 评测结果",
        "",
        f"评测题数：{summary['total']} 道（单轮 {len(original_records) + len(figure_records)} 道 + 多轮 {mt_s['total_turns']} 轮）",
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
        "## 路由正确性",
        "",
        f"- 路由对：{router_s['correct']} / {router_s['total']}，准确率：{router_s['accuracy']:.1%}",
        f"- 路由错的题：{', '.join(router_s['wrong_ids']) if router_s['wrong_ids'] else '无'}",
        "",
        "## 复杂任务完成度",
        "",
        f"- planning 题共 {planning_s['total']} 道：完全完成 {planning_s['complete']} 道，部分完成 {planning_s['partial']} 道，未完成 {planning_s['incomplete']} 道",
        f"- 部分完成的题：{', '.join(planning_s['partial_ids']) if planning_s['partial_ids'] else '无'}",
        f"- 未完成的题：{', '.join(planning_s['incomplete_ids']) if planning_s['incomplete_ids'] else '无'}",
        "",
        "## VLM 图表理解",
        "",
        f"- 图表题已答：{vlm_s['answered']} / {vlm_s['total']}，引用准确率：{vlm_s['citation_accuracy']:.1%}",
        f"- 走 vision 路由：{vlm_s['vision_routed']} / {vlm_s['total']}，占比：{vlm_s['vision_routing_rate']:.1%}",
        "- 当前 chart_read 还是占位，这里测的是图注和文本路径的图表理解基线；2.3 真 VLM 落地后同一维度会反映 VLM 的效果。",
        "",
        "## 跨文档引用",
        "",
        *(cross_doc_lines if cross_doc_lines else ["- 本轮没有多来源题。"]),
        "",
        "## 多轮序列",
        "",
        f"- 共 {mt_s['total_turns']} 轮，分布在 {len(mt_s['sequences'])} 个序列里",
        *mt_lines,
        "",
        "## 耗时统计",
        "",
        f"- 总耗时：{timing_s['total_seconds']} 秒，平均每题：{timing_s['mean_seconds']} 秒",
        f"- 最慢的 3 道：{slowest_text}",
        "- Token 数 run() 不返回，精确值要看 LangSmith trace。",
        "",
        "## 逐题表（原 20 题）",
        "",
        *_table_lines(original_records),
        "",
        "## 逐题表（图表题）",
        "",
        *_table_lines(figure_records),
        "",
        "## 逐题表（多轮序列）",
        "",
        *_table_lines(multiturn_records),
        "",
        "## 说明",
        "",
        "- 引用未命中不一定是编造，可能是检索没找到对应片段，需结合回答原文人工判断。",
        "- 完整回答原文见 eval/results.json。",
        "",
    ]
    results_md.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("评测完成：")
    print(f"  总数 {summary['total']}，已答 {summary['answered']}，拒答 {summary['refused']}" + (f"，出错 {summary['errors']}" if summary['errors'] else ""))
    print(f"  原 20 题：可回答已答 {orig['answerable_answered']}/{orig['answerable_total']}，"
          f"引用准确率 {orig['citation_accuracy']:.1%}（{orig['citation_hits']}/{orig['answerable_answered']}），"
          f"应拒答 {orig['should_refuse_correct']}/{orig['should_refuse_total']}，编造 {orig['fabrication_count']} 次")
    print(f"  图表 {fig['total']} 题：可回答已答 {fig['answerable_answered']}/{fig['answerable_total']}，"
          f"引用准确率 {fig['citation_accuracy']:.1%}（{fig['citation_hits']}/{fig['answerable_answered']}），"
          f"应拒答 {fig['should_refuse_correct']}/{fig['should_refuse_total']}，编造 {fig['fabrication_count']} 次")
    print(f"  路由正确性：{router_s['correct']}/{router_s['total']}，准确率 {router_s['accuracy']:.1%}"
          + (f"，路由错的题：{', '.join(router_s['wrong_ids'])}" if router_s['wrong_ids'] else ""))
    print(f"  复杂任务完成度：planning 题 {planning_s['total']} 道，完全完成 {planning_s['complete']} 道，"
          f"部分完成 {planning_s['partial']} 道，未完成 {planning_s['incomplete']} 道")
    print(f"  VLM 图表理解：图表题已答 {vlm_s['answered']}/{vlm_s['total']}，"
          f"引用准确率 {vlm_s['citation_accuracy']:.1%}，走 vision 路由 {vlm_s['vision_routed']}/{vlm_s['total']}")
    print(f"  多轮序列：{mt_s['total_turns']} 轮，{len(mt_s['sequences'])} 个序列")
    print(f"  耗时：总 {timing_s['total_seconds']} 秒，平均每题 {timing_s['mean_seconds']} 秒")
    print(f"  结果已写入 {results_json}")
    print(f"  总结已写入 {results_md}")


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="MedEvidence 评测：开发集或冻结测试集")
    _parser.add_argument("--frozen", action="store_true",
                         help="跑冻结测试集 T01–T10（阶段验收专用），结果写 results_frozen.json/md")
    _args = _parser.parse_args()
    main(frozen=_args.frozen)
