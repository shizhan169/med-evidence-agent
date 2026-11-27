"""图表题评测：题库解析 + 分节汇总。

原 20 道题不动：解析后前 20 道的题号、字段必须和改动前一致；
新增的 8 道图表题单独成节，分开计分。
"""
from pathlib import Path

from eval.run_eval import is_refusal, parse_questions, section_summary

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = ROOT / "eval" / "test_questions.md"
PAPERS_DIR = ROOT / "data" / "papers"


def test_parse_28_questions_original_untouched():
    qs = parse_questions(QUESTIONS_FILE)
    # 2.4 扩充后共 40 道：原 28 道 + 新增 12 道（Q16–Q23、R06–R07、F07–F08）
    assert len(qs) == 40
    ids = [q["id"] for q in qs]
    # 原 28 道题没动：相对顺序不变
    orig_ids = ([f"Q{i:02d}" for i in range(1, 16)] + [f"R{i:02d}" for i in range(1, 6)]
                + ["F01", "F02", "F03", "F04", "F05", "F06", "FR01", "FR02"])
    assert [i for i in ids if i in orig_ids] == orig_ids
    # 新增 12 道插在各自小节末尾
    assert ids[15:23] == [f"Q{i:02d}" for i in range(16, 24)]
    assert ids[28:30] == ["R06", "R07"]
    assert ids[36:38] == ["F07", "F08"]
    # 原 20 道题没动：抽查 Q01、R05 的字段
    by_id = {q["id"]: q for q in qs}
    q01 = by_id["Q01"]
    assert q01["question"].startswith("Bethlehem 2022")
    assert q01["type"] == "可回答"
    assert q01["expected_file"] == "2022-bethlehem-brain-charts.pdf"
    assert q01["expected_pages"] == [1]
    assert q01["section"] == "original"
    r05 = by_id["R05"]
    assert r05["question"] == "今天上海天气怎么样？"
    assert r05["type"] == "应拒答"
    assert r05["section"] == "original"


def test_figure_questions_parse():
    qs = {q["id"]: q for q in parse_questions(QUESTIONS_FILE)}
    for fid in ["F01", "F02", "F03", "F04", "F05", "F06"]:
        q = qs[fid]
        assert q["type"] == "可回答", fid
        assert q["section"] == "figure", fid
        assert q["expected_answer"], fid
        assert q["expected_file"].endswith(".pdf"), fid
        assert q["expected_pages"], fid
        # 预期来源的论文必须在库里
        assert (PAPERS_DIR / q["expected_file"]).exists(), fid
    # F02 是按图注原文收紧过的版本，不是只有标题翻译
    f02 = qs["F02"]
    assert "三幅面板" in f02["expected_answer"]
    assert "95% 置信区间" in f02["expected_answer"]
    assert "广义加性模型" in f02["expected_answer"]
    for fid in ["FR01", "FR02"]:
        q = qs[fid]
        assert q["type"] == "应拒答", fid
        assert q["section"] == "figure", fid
        assert q["refuse_reason"], fid
        assert q["expected_answer"] == ""


def test_is_refusal_first_sentence():
    # 固定开头 "根据现有文献无法回答" 及其变体：第七轮 Q09 在中间插了"片段"，
    # 旧的 startswith 判定漏判了它
    assert is_refusal("根据现有文献无法回答。")
    assert is_refusal("根据现有文献片段无法回答“健康老年人脑室体积年均变化率”的具体数值；片段中的脑室变化率图未给出可读的具体数值。")
    assert is_refusal("根据现有文献无法回答：提供的片段中仅出现“Lecanemab in early Alzheimer’s disease”这一参考文献题名。")
    assert is_refusal("根据现有文献无法回答具体“快多少”。片段仅报告 APOE 风险变异 rs429358 与海马体积变化率相关。")
    assert is_refusal("根据现有文献无法回答“健康老年人全脑体积年均变化率”的具体数值及其不同年龄段差异；片段仅以按年龄组分层的分布图展示全脑体积变化率。")
    # 正常回答：第一句没有"无法回答"
    assert not is_refusal("总灰质体积（GMV）在 5.9 岁达到峰值（95% bootstrap CI 5.8–6.1）[来源：2022-bethlehem-brain-charts.pdf p.3]。")
    # 后半句的部分拒答不是整题拒答（Q03 第五轮的写法）
    assert not is_refusal("总白质体积（WMV）在 28.7岁达到峰值（95% bootstrap CI 28.1–29.2）。峰值后至 50 岁前的具体变化，根据现有文献片段无法回答。")
    assert not is_refusal("")


def test_section_summary_splits_original_and_figure():
    def rec(qid, section, qtype, refused, hit, error=""):
        return {"id": qid, "section": section, "type": qtype,
                "refused": refused, "citation_hit": hit, "error": error}

    records = [
        rec("Q01", "original", "可回答", False, True),
        rec("Q02", "original", "可回答", False, False),
        rec("R01", "original", "应拒答", True, False),
        rec("F01", "figure", "可回答", False, True),
        rec("FR01", "figure", "应拒答", False, False),
    ]
    overall = section_summary(records)
    assert overall["total"] == 5
    assert overall["answerable_answered"] == 3
    assert overall["citation_hits"] == 2
    assert overall["should_refuse_correct"] == 1
    assert overall["fabrication_count"] == 1
    orig = section_summary([r for r in records if r["section"] == "original"])
    assert (orig["answerable_total"], orig["answerable_answered"]) == (2, 2)
    assert orig["citation_hits"] == 1
    assert orig["citation_accuracy"] == 0.5
    assert orig["fabrication_count"] == 0
    fig = section_summary([r for r in records if r["section"] == "figure"])
    assert (fig["answerable_total"], fig["answerable_answered"]) == (1, 1)
    assert fig["citation_hits"] == 1
    assert fig["citation_accuracy"] == 1.0
    assert fig["should_refuse_correct"] == 0
    assert fig["fabrication_count"] == 1
