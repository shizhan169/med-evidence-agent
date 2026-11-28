"""冻结测试集：T01–T10 能被评测脚本解析。

冻结测试题（eval/datasets/test_questions_frozen.md）用 T 开头的题号，
和开发集的 Q/R/F 区分开。parse_questions 必须认 T 题号，否则冻结文件
只是个文档，验收时跑不起来。
"""
import inspect
from pathlib import Path

from eval.run_eval import (
    FROZEN_QUESTIONS_FILE,
    FROZEN_RESULTS_JSON,
    FROZEN_RESULTS_MD,
    QUESTIONS_FILE,
    RESULTS_JSON,
    RESULTS_MD,
    main,
    parse_questions,
)

ROOT = Path(__file__).resolve().parent.parent
FROZEN_FILE = ROOT / "eval" / "datasets" / "test_questions_frozen.md"


def test_frozen_questions_parse():
    qs = parse_questions(FROZEN_FILE)
    assert len(qs) == 10
    assert [q["id"] for q in qs] == [f"T{i:02d}" for i in range(1, 11)]
    for q in qs:
        assert q["type"] == "可回答", q["id"]
        assert q["section"] == "frozen", q["id"]
        assert q["expected_answer"], q["id"]
        assert q["expected_file"].endswith(".pdf"), q["id"]
        assert q["expected_pages"], q["id"]
    # 每篇测试集论文 2 道：5 篇各 2 道
    from collections import Counter

    files = Counter(q["expected_file"] for q in qs)
    assert len(files) == 5
    assert all(n == 2 for n in files.values())
    # 注意：4 篇新论文的 PDF 还在用户本地，VM 的 data/papers/ 里没有，
    # 这里不 assert 文件存在；验收跑题之前要先把 4 篇同步进来再入库。


def test_frozen_runner_uses_separate_files():
    """--frozen 跑冻结集：题目文件、结果文件都和开发集分开，
    冻结跑不会覆盖开发集的 results.json。"""
    assert FROZEN_QUESTIONS_FILE == FROZEN_FILE
    assert FROZEN_QUESTIONS_FILE != QUESTIONS_FILE
    assert FROZEN_RESULTS_JSON != RESULTS_JSON
    assert FROZEN_RESULTS_MD != RESULTS_MD
    assert FROZEN_RESULTS_JSON.name == "results_frozen.json"
    assert FROZEN_RESULTS_MD.name == "results_frozen.md"
    assert "frozen" in inspect.signature(main).parameters
