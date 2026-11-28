"""第三阶段 3.5 补覆盖测试：第二阶段模块（router / planner / executor /
tools / vision / verifier）里现有测试没走到分支。

只测已有行为，不改 app/ 代码。假 LLM + 假工具 + 假 VLM，不联网、
不需要 Chroma、不需要 API key。跑法：pytest tests/test_phase3_coverage.py -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.agent as agent
from app import config, tools, vision

FUJITA = ROOT / "data" / "papers" / "2023-fujita-brain-volume-aging.pdf"


# ---------- Router：_classify_by_llm 直接测 ----------

def test_classify_by_llm_derives_complex_from_planning_flag(monkeypatch):
    # LLM 只给了 requires_planning，没给 task_type：task_type 按规则推导成 complex
    monkeypatch.setattr("app.agent._call_llm", lambda p: '{"requires_planning": true}')
    route = agent._classify_by_llm("这个结果说明什么？")
    assert route["requires_planning"] is True
    assert route["task_type"] == "complex"
    assert route["requires_vision"] is False
    assert route["requires_external_search"] is False


def test_classify_by_llm_explicit_task_type_kept(monkeypatch):
    monkeypatch.setattr(
        "app.agent._call_llm",
        lambda p: '{"task_type": "complex", "requires_vision": true}',
    )
    route = agent._classify_by_llm("Fujita 图 4 讲了什么？")
    assert route["task_type"] == "complex"
    assert route["requires_vision"] is True
    assert route["requires_planning"] is False


def test_classify_by_llm_empty_response_falls_back(monkeypatch):
    monkeypatch.setattr("app.agent._call_llm", lambda p: "")
    assert agent._classify_by_llm("这个结果可靠吗？") == agent._empty_route()


def test_classify_by_llm_non_bool_flags_ignored(monkeypatch):
    # flag 不是布尔值时不采纳（字符串 "yes"、数字 1 都不算 True）
    monkeypatch.setattr(
        "app.agent._call_llm",
        lambda p: '{"requires_planning": "yes", "requires_vision": 1}',
    )
    assert agent._classify_by_llm("这个结果可靠吗？") == agent._empty_route()


def test_classify_by_rules_english_and_combined():
    route = agent._classify_by_rules("Compare the longitudinal models across papers")
    assert route["requires_planning"] is True
    assert route["task_type"] == "complex"
    route = agent._classify_by_rules("Table 2 shows the sample sizes")
    assert route["requires_vision"] is True
    assert route["task_type"] == "simple"
    route = agent._classify_by_rules("比较五篇论文的图 4")
    assert route["requires_planning"] is True
    assert route["requires_vision"] is True
    assert route["task_type"] == "complex"


# ---------- Planner：_parse_plan 直接测 ----------

def test_parse_plan_empty_and_garbage():
    assert agent._parse_plan("") == ("", [])
    assert agent._parse_plan("我不知道怎么拆") == ("", [])
    assert agent._parse_plan(None) == ("", [])


def test_parse_plan_subtasks_not_list():
    assert agent._parse_plan('{"goal": "比较五篇", "subtasks": "不是列表"}') == ("比较五篇", [])


def test_parse_plan_skips_bad_entries():
    raw = ('{"goal": "g", "subtasks": ['
           '{"query": "数据集"}, "不是字典", {"noquery": 1}, '
           '{"query": "   "}, {"query": 123}, {"query": "模型"}]}')
    goal, queries = agent._parse_plan(raw)
    assert goal == "g"
    assert queries == ["数据集", "模型"]


def test_parse_plan_caps_at_five():
    subs = ", ".join(f'{{"query": "子任务 {i}"}}' for i in range(1, 8))
    _, queries = agent._parse_plan('{"goal": "g", "subtasks": [' + subs + "]}")
    assert len(queries) == 5
    assert queries[0] == "子任务 1"
    assert queries[-1] == "子任务 5"


def test_parse_plan_non_string_goal():
    goal, queries = agent._parse_plan('{"goal": 123, "subtasks": [{"query": "a"}]}')
    assert goal == ""
    assert queries == ["a"]


# ---------- Executor：_parse_tool_call 直接测 ----------

def test_parse_tool_call_empty_and_bad_json():
    assert agent._parse_tool_call("") == ("local_search", {})
    assert agent._parse_tool_call("{not json") == ("local_search", {})
    assert agent._parse_tool_call(None) == ("local_search", {})


def test_parse_tool_call_unknown_tool():
    assert agent._parse_tool_call('{"tool": "nope", "params": {}}') == ("local_search", {})
    assert agent._parse_tool_call('{"tool": 123, "params": {}}') == ("local_search", {})


def test_parse_tool_call_params_not_dict():
    assert agent._parse_tool_call('{"tool": "local_search", "params": "xxx"}') == (
        "local_search", {})
    assert agent._parse_tool_call('{"tool": "chart_locate", "params": null}') == (
        "chart_locate", {})


def test_parse_tool_call_valid():
    tool, params = agent._parse_tool_call(
        '{"tool": "chart_locate", "params": {"paper": "Fujita", "figure_no": 4}}')
    assert tool == "chart_locate"
    assert params == {"paper": "Fujita", "figure_no": 4}


# ---------- Executor：合并去重 ----------

def test_merge_chunks_dedups():
    a = {"source": "x.pdf", "page": 1, "text": "same"}
    b = {"source": "x.pdf", "page": 2, "text": "same"}
    c = {"source": "x.pdf", "page": 1, "text": "same"}
    out = agent._merge_chunks([a], [b, c])
    assert len(out) == 2
    assert out[0] is a
    assert out[1] == b


def test_merge_pubmed_results_dedups():
    r1 = {"pmid": "111", "title": "A"}
    r2 = {"pmid": "222", "title": "B"}
    r3 = {"pmid": "111", "title": "A again"}
    r4 = {"pmid": "", "title": "no pmid"}
    out = agent._merge_pubmed_results([r1], [r2, r3, r4])
    assert [r["pmid"] for r in out] == ["111", "222"]


# ---------- Executor 节点：选工具 LLM 挂了、chart_locate 找到 ----------

def _subtask(task_id, query):
    return {"task_id": task_id, "query": query, "status": "pending",
            "evidence_ids": [], "result": "", "retry_count": 0}


def _state(subtasks):
    return {"query": "比较五篇论文", "subtasks": subtasks,
            "tool_calls": [], "chunks": [], "pubmed_results": []}


def test_executor_llm_selection_failure_uses_default(monkeypatch):
    # 选工具时 LLM 抛异常：退回默认 local_search，子任务照常完成
    calls = []

    def boom(prompt):
        raise RuntimeError("LLM 挂了")

    def fake_tool(name, **kwargs):
        calls.append(name)
        return {"success": True,
                "data": {"chunks": [{"text": "t", "source": "x.pdf", "page": 1}]},
                "error": None}

    monkeypatch.setattr("app.agent._call_llm", boom)
    monkeypatch.setattr("app.agent.call_tool", fake_tool)
    result = agent.executor_node(_state([_subtask("S1", "五篇论文的数据集")]))
    assert calls == ["local_search"]
    assert result["subtasks"][0]["status"] == "completed"
    assert result["subtasks"][0]["retry_count"] == 0


def test_executor_chart_locate_found(monkeypatch):
    # chart_locate 找到图：子任务完成，证据是图所在论文，result 是图注
    def fake_tool(name, **kwargs):
        return {"success": True,
                "data": {"found": True, "source": "2023-fujita-brain-volume-aging.pdf",
                         "caption": "Figure 4. Volume change rate."},
                "error": None}

    monkeypatch.setattr("app.agent._call_llm",
                        lambda p: '{"tool": "chart_locate", "params": {"paper": "Fujita", "figure_no": 4}}')
    monkeypatch.setattr("app.agent.call_tool", fake_tool)
    result = agent.executor_node(_state([_subtask("S1", "Fujita 图 4 在哪一页")]))
    st = result["subtasks"][0]
    assert st["status"] == "completed"
    assert st["evidence_ids"] == ["2023-fujita-brain-volume-aging.pdf"]
    assert st["result"] == "Figure 4. Volume change rate."


# ---------- PubMed 辅助函数 ----------

def test_to_pubmed_query_english_unchanged(monkeypatch):
    # 纯英文问题不翻译，不花一次 LLM 调用
    calls = {"n": 0}

    def fake_llm(prompt):
        calls["n"] += 1
        return "should not be used"

    monkeypatch.setattr("app.agent._call_llm", fake_llm)
    assert agent._to_pubmed_query("longitudinal brain normative modeling") == (
        "longitudinal brain normative modeling")
    assert calls["n"] == 0
    assert agent._to_pubmed_query("") == ""


def test_to_pubmed_query_llm_failure_returns_original(monkeypatch):
    def boom(prompt):
        raise RuntimeError("LLM 挂了")

    monkeypatch.setattr("app.agent._call_llm", boom)
    q = "纵向脑常模模型最新的文献有哪些"
    assert agent._to_pubmed_query(q) == q


def test_to_pubmed_query_still_chinese_returns_original(monkeypatch):
    monkeypatch.setattr("app.agent._call_llm", lambda p: "纵向脑常模模型")
    q = "纵向脑常模模型最新的文献有哪些"
    assert agent._to_pubmed_query(q) == q


def test_to_pubmed_query_strips_outer_quotes(monkeypatch):
    # 整个输出被一对引号包起来时去掉外层引号（引号恰好 2 个才 strip）
    monkeypatch.setattr(
        "app.agent._call_llm",
        lambda p: '"longitudinal brain normative model OR brain charts"')
    out = agent._to_pubmed_query("纵向脑常模模型")
    assert out == "longitudinal brain normative model OR brain charts"


def test_classify_by_llm_bad_json_falls_back(monkeypatch):
    # 有大括号但不是合法 JSON：解析失败，退回简单问答
    monkeypatch.setattr("app.agent._call_llm", lambda p: "{not json")
    assert agent._classify_by_llm("这个结果可靠吗？") == agent._empty_route()


def test_parse_plan_bad_json():
    assert agent._parse_plan("{not json") == ("", [])


def test_parse_tool_call_bad_json():
    assert agent._parse_tool_call("{not json") == ("local_search", {})


def test_to_pubmed_query_non_or_fallback(monkeypatch):
    # 没 OR、没成对引号：退回关键词拼接，不把不合规的式子送给 PubMed
    monkeypatch.setattr("app.agent._call_llm", lambda p: "longitudinal brain model")
    assert agent._to_pubmed_query("纵向脑常模模型") == "longitudinal brain model"


def test_norm_pubmed_word():
    assert agent._norm_pubmed_word("modelling") == "model"
    assert agent._norm_pubmed_word("models") == "model"
    assert agent._norm_pubmed_word("charts") == "chart"
    assert agent._norm_pubmed_word("centiles") == "centile"
    assert agent._norm_pubmed_word("analysis") == "analysis"
    assert agent._norm_pubmed_word("model") == "model"


def test_phrase_hit():
    words = {"longitudinal", "brain", "normative", "model"}
    assert agent._phrase_hit("longitudinal brain model", words) is True
    assert agent._phrase_hit("longitudinal hippocampus", words) is False
    assert agent._phrase_hit("", words) is False


def test_pubmed_year():
    assert agent._pubmed_year({"year": "2024"}) == 2024
    assert agent._pubmed_year({"year": 2024}) == 2024
    assert agent._pubmed_year({}) == 0
    assert agent._pubmed_year({"year": "abc"}) == 0
    assert agent._pubmed_year({"year": None}) == 0


def test_route_after_retriever():
    # PubMed 关了（评测）直接回答
    assert agent._route_after_retriever({"use_pubmed": False, "chunks": []}) == "answerer"
    # 本地有证据、没问新文献：直接回答
    assert agent._route_after_retriever(
        {"use_pubmed": True, "query": "Bethlehem 脑图谱多少扫描",
         "chunks": [{"text": "x"}]}) == "answerer"
    # 本地没证据：走 PubMed
    assert agent._route_after_retriever(
        {"use_pubmed": True, "query": "Bethlehem 脑图谱多少扫描", "chunks": []}) == "pubmed"
    # 问新文献：哪怕本地有证据也走 PubMed
    assert agent._route_after_retriever(
        {"use_pubmed": True, "query": "纵向脑常模模型最新的文献有哪些",
         "chunks": [{"text": "x"}]}) == "pubmed"


# ---------- tools：_type_ok、图注查找、读图失败分支 ----------

def test_type_ok_list_types():
    assert tools._type_ok("x", ["str", "int"]) is True
    assert tools._type_ok(3, ["str", "int"]) is True
    assert tools._type_ok(3.5, ["str", "int"]) is False
    assert tools._type_ok(True, ["int"]) is False  # bool 不是 int
    assert tools._type_ok(True, ["bool"]) is True
    assert tools._type_ok(True, ["str", "bool"]) is True


class _FakeCollection:
    def __init__(self, docs, metas):
        self._docs = docs
        self._metas = metas

    def get(self, where=None, include=None):
        return {"documents": self._docs, "metadatas": self._metas}


def test_find_caption_no_digits():
    assert tools._find_caption("Fujita", "abc") == {"found": False}
    assert tools._find_caption("Fujita", "") == {"found": False}
    assert tools._find_caption("Fujita", None) == {"found": False}


def test_find_caption_paper_mismatch(monkeypatch):
    # 论文名和库里任何 source 都对不上：找不到，不抛异常
    docs = ["Figure 4. Volume change rate."]
    metas = [{"source": "2023-fujita-brain-volume-aging.pdf", "page": 8}]
    monkeypatch.setattr("app.tools._open_collection",
                        lambda: _FakeCollection(docs, metas))
    assert tools._find_caption("Shaw", 4) == {"found": False}


def test_chart_read_image_vlm_failure(tmp_path, monkeypatch):
    # 图片存在，但 VLM 调不通：available=False，不抛异常
    img = tmp_path / "fig.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    def boom(image_bytes, prompt):
        raise RuntimeError("VLM 挂了")

    monkeypatch.setattr("app.vision.call_vlm", boom)
    out = tools._chart_read(image_path=str(img), question="这张图讲了什么？")
    assert out["available"] is False
    assert "VLM 调用失败" in out["reason"]


def test_chart_read_image_path_is_dir(tmp_path):
    # image_path 是个目录不是文件：available=False
    out = tools._chart_read(image_path=str(tmp_path), question="这张图讲了什么？")
    assert out["available"] is False
    assert "图片不存在" in out["reason"]


# ---------- vision：数字格式、提示词兜底、VLM 调用、找图页 ----------

def test_fmt_num():
    assert vision._fmt_num(95.0) == "95"
    assert vision._fmt_num(3.5) == "3.5"
    assert vision._fmt_num(0.0) == "0"


def test_chart_reader_prompt_fallback(monkeypatch):
    # 提示词文件缺了：用内置最小版，照样带图注和问题
    def boom(name, version):
        raise FileNotFoundError("no prompt file")

    monkeypatch.setattr("app.vision.load_prompt", boom)
    prompt = vision._chart_reader_prompt("曲线趋势如何？", "Figure 4. Volume change.")
    assert "Figure 4. Volume change." in prompt
    assert "曲线趋势如何？" in prompt


def test_default_vlm_call_no_key(monkeypatch):
    monkeypatch.setattr("app.vision.get_vlm_api_key", lambda: "")
    with pytest.raises(RuntimeError, match="没有 API key"):
        vision._default_vlm_call(b"fake-png-bytes", "看这张图")


class _FakeChatCompletions:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        msg = type("M", (), {"content": self.text})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


class _FakeOpenAI:
    instances = []

    def __init__(self, api_key=None, base_url=None, timeout=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.chat = type("Chat", (), {"completions": _FakeChatCompletions("图里有三幅面板")})()
        _FakeOpenAI.instances.append(self)


def test_default_vlm_call_ok(monkeypatch):
    monkeypatch.setattr("app.vision.get_vlm_api_key", lambda: "sk-test")
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    _FakeOpenAI.instances.clear()
    out = vision._default_vlm_call(b"fake-png-bytes", "看这张图")
    assert out == "图里有三幅面板"
    client = _FakeOpenAI.instances[0]
    assert client.api_key == "sk-test"
    call = client.chat.completions.calls[0]
    assert call["model"] == vision.VLM_MODEL
    assert call["temperature"] == 0
    assert call["messages"][0]["content"][1]["type"] == "image_url"


def test_find_figure_page_no_images(tmp_path):
    # 一整本 PDF 都没有图片：退回原页，found=False
    import pymupdf

    pdf = tmp_path / "noimg.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Figure 4. No image here, only text.")
    doc.save(str(pdf))
    doc.close()
    page_no, found = vision._find_figure_page(str(pdf), 1)
    assert page_no == 1
    assert found is False


def test_find_figure_clip_no_digits():
    assert vision.find_figure_clip(FUJITA, 8, "abc") is None
    assert vision.find_figure_clip(FUJITA, 8, "") is None


def test_find_figure_clip_page_out_of_range():
    assert vision.find_figure_clip(FUJITA, 99, 4) is None
    assert vision.find_figure_clip(FUJITA, 0, 4) is None


def test_find_figure_clip_caption_not_on_page():
    # Fujita p.8 没有 Figure 9 的图注：找不到，退回整页
    assert vision.find_figure_clip(FUJITA, 8, 9) is None


def test_find_figure_clip_too_narrow(tmp_path):
    # 两条图注行靠得太近（带子不到 80pt）：不裁剪，退回整页
    import pymupdf

    pdf = tmp_path / "narrow.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Figure 4. First caption.")
    page.insert_text((72, 120), "Figure 5. Second caption.")
    doc.save(str(pdf))
    doc.close()
    assert vision.find_figure_clip(str(pdf), 1, 4) is None


def test_render_page_empty_clip_falls_back():
    # 裁剪框完全在页外：交集为空，退回整页渲染，照样出 PNG
    data = vision.render_page_to_png(FUJITA, 8, clip=(5000, 5000, 6000, 6000))
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 10000


# ---------- _call_llm：没 key、假 OpenAI ----------

def test_call_llm_no_key_returns_empty(monkeypatch):
    monkeypatch.setattr("app.agent.get_api_key", lambda: "")
    assert agent._call_llm("你好") == ""


def test_call_llm_uses_fake_openai(monkeypatch):
    monkeypatch.setattr("app.agent.get_api_key", lambda: "sk-test")
    monkeypatch.setattr("app.agent.tracing_enabled", lambda: False)
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    _FakeOpenAI.instances.clear()
    out = agent._call_llm("Bethlehem 脑图谱多少扫描？")
    assert out == "图里有三幅面板"
    client = _FakeOpenAI.instances[0]
    assert client.api_key == "sk-test"
    call = client.chat.completions.calls[0]
    assert call["model"] == agent.LLM_MODEL
    assert call["temperature"] == 0


# ---------- config：追踪开关、key 优先级 ----------

def test_tracing_enabled(monkeypatch):
    monkeypatch.setattr(config, "LANGSMITH_TRACING", "true")
    assert config.tracing_enabled() is True
    monkeypatch.setattr(config, "LANGSMITH_TRACING", "false")
    assert config.tracing_enabled() is False
    monkeypatch.setattr(config, "LANGSMITH_TRACING", "")
    assert config.tracing_enabled() is False


def test_get_api_key_precedence(monkeypatch):
    monkeypatch.setattr(config, "QWEN_API_KEY", "qwen-key")
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "deepseek-key")
    assert config.get_api_key() == "qwen-key"
    monkeypatch.setattr(config, "QWEN_API_KEY", "")
    assert config.get_api_key() == "deepseek-key"
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    assert config.get_api_key() == ""




# ---------- 第二批：核查员、回答员、执行器剩余分支 ----------

def test_executor_chart_read_available_goes_to_else_branch(monkeypatch):
    # chart_read 成功且 available=True：不属于三类检索工具，走通用 else 分支完成
    def fake_tool(name, **kwargs):
        return {"success": True,
                "data": {"available": True, "reading": "图里有三幅面板"},
                "error": None}

    monkeypatch.setattr("app.agent._call_llm",
                        lambda p: '{"tool": "chart_read", "params": {"paper": "Fujita", "figure_no": 4}}')
    monkeypatch.setattr("app.agent.call_tool", fake_tool)
    result = agent.executor_node(_state([_subtask("S1", "Fujita 图 4 的曲线趋势")]))
    st = result["subtasks"][0]
    assert st["status"] == "completed"
    assert "三幅面板" in st["result"]


def test_executor_step_cap_leaves_running_pending(monkeypatch):
    # 前 7 个子任务各用 1 步完成，第 8 个子任务第 1 次失败（还有重试额度）
    # 但步数已到上限：while 退出时它还停在 running，收尾时改回 pending
    calls = {"n": 0}

    def fake_tool(name, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 7:
            return {"success": True,
                    "data": {"chunks": [{"text": "t", "source": "x.pdf", "page": 1}]},
                    "error": None}
        return {"success": False, "data": None, "error": "第 8 步失败"}

    monkeypatch.setattr("app.agent._call_llm",
                        lambda p: '{"tool": "local_search", "params": {"query": "q"}}')
    monkeypatch.setattr("app.agent.call_tool", fake_tool)
    subtasks = [_subtask(f"S{i}", f"子任务 {i}") for i in range(1, 9)]
    result = agent.executor_node(_state(subtasks))
    assert len(result["tool_calls"]) == 8
    assert [s["status"] for s in result["subtasks"][:7]] == ["completed"] * 7
    last = result["subtasks"][7]
    assert last["status"] == "pending"
    assert last["retry_count"] == 1


def test_extract_paper_no_latin_lead():
    # 问题有图号但开头不是拉丁作者名：抽不出论文，只返回图号
    assert agent._extract_paper_and_figure("图 4 的曲线趋势如何？") == (None, "4")
    assert agent._extract_paper_and_figure("这张图 4 讲了什么？") == (None, "4")


def test_call_llm_tracing_wrap_failure_falls_back(monkeypatch):
    # 追踪开着但 wrap_openai 抛异常：用原 client，问答不受影响
    import langsmith.wrappers

    monkeypatch.setattr("app.agent.get_api_key", lambda: "sk-test")
    monkeypatch.setattr("app.agent.tracing_enabled", lambda: True)
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    monkeypatch.setattr(langsmith.wrappers, "wrap_openai",
                        lambda client: (_ for _ in ()).throw(RuntimeError("wrap 失败")))
    _FakeOpenAI.instances.clear()
    out = agent._call_llm("Bethlehem 脑图谱多少扫描？")
    assert out == "图里有三幅面板"


def test_answerer_llm_failure_returns_placeholder(monkeypatch):
    # 回答员 LLM 抛异常：不编答案，返回待接 LLM 占位
    def boom(prompt):
        raise RuntimeError("LLM 挂了")

    monkeypatch.setattr("app.agent._call_llm", boom)
    state = {"query": "Bethlehem 脑图谱多少扫描？",
             "chunks": [{"text": "t", "source": "x.pdf", "page": 1}],
             "history": [], "history_summary": "", "retries": 0,
             "pubmed_results": []}
    out = agent.answerer_node(state)
    assert out["draft"].startswith("[待接 LLM]")


def test_parse_evidence_id_forms():
    assert agent._parse_evidence_id("2022-bethlehem-brain-charts.pdf p.3") == (
        "local", "2022-bethlehem-brain-charts.pdf", "3")
    assert agent._parse_evidence_id("38395541") == ("pmid", "38395541")
    assert agent._parse_evidence_id("2023-fujita-brain-volume-aging.pdf") == (
        "source", "2023-fujita-brain-volume-aging.pdf")


def test_check_subtasks_evidence_not_in_pool():
    # completed 但证据不在池里：local 引用对不上、PMID 对不上都算证据不足
    subs = [
        {"task_id": "S1", "status": "completed",
         "evidence_ids": ["missing.pdf p.99"]},
        {"task_id": "S2", "status": "completed",
         "evidence_ids": ["99999999"]},
        {"task_id": "S3", "status": "completed",
         "evidence_ids": ["2023-fujita-brain-volume-aging.pdf"]},
        {"task_id": "S4", "status": "completed",
         "evidence_ids": ["2022-bethlehem-brain-charts.pdf p.3"]},
    ]
    verified, failures = agent._check_subtasks(
        subs, {("2022-bethlehem-brain-charts.pdf", "3")}, set(), {"111"})
    assert verified == {"S3", "S4"}
    assert {f["subtask_id"] for f in failures} == {"S1", "S2"}
    assert all(f["reason"] == "insufficient_evidence" for f in failures)


def test_verified_evidence_keys_pmid():
    subs = [
        {"task_id": "S1", "status": "completed",
         "evidence_ids": ["38395541", "2022-bethlehem-brain-charts.pdf p.3"]},
    ]
    local_keys, pmid_keys = agent._verified_evidence_keys(subs, {"S1"})
    assert local_keys == {("2022-bethlehem-brain-charts.pdf", "3")}
    assert pmid_keys == {"38395541"}


def test_filter_partial_draft_pubmed():
    # PubMed 路径：PMID 全在已核实集合里的句子保留，别的去掉
    draft = ("第一句 PubMed PMID:111。第二句 PubMed PMID:999。"
             "第三句见 DOI: 10.1000/xyz。")
    out = agent._filter_partial_draft(draft, set(), {"111"}, True, [])
    assert "第一句" in out
    assert "第二句" not in out
    assert "第三句" in out  # DOI 引用全局已核查通过，保留
    # 一句都留不下：原样返回（防御性分支）
    out2 = agent._filter_partial_draft("只有一句 PubMed PMID:999。",
                                       set(), {"111"}, True, [])
    assert out2 == "只有一句 PubMed PMID:999。"


def test_filter_partial_draft_local():
    draft = ("Bethlehem 有 123984 次扫描 [来源：2022-bethlehem-brain-charts.pdf p.1]。"
             "Di Biase 用了两个队列 [来源：2023-dibiase-mapping-brain-charts.pdf p.2]。")
    out = agent._filter_partial_draft(
        draft, {("2022-bethlehem-brain-charts.pdf", "1")}, set(), False, [])
    assert "Bethlehem" in out
    assert "Di Biase" not in out


def test_partial_note_empty_and_named():
    assert agent._partial_note([]) == "\n\n【部分回答】部分内容因证据不足，本次未作回答。"
    note = agent._partial_note([{"query": "数据集"}, {"query": ""}, {}])
    assert "数据集" in note
    assert note.startswith("\n\n【部分回答】以下内容因证据不足，本次未作回答：")


def test_verifier_pubmed_mixed_local_citation_blocked():
    # PubMed 回答里混入本地式引用：按编造处理
    state = {
        "query": "纵向脑常模模型最新的文献有哪些",
        "draft": "某研究纳入 1000 人 [来源：2022-bethlehem-brain-charts.pdf p.3]。",
        "chunks": [],
        "pubmed_results": [{"pmid": "111", "title": "A", "abstract": "x" * 50}],
        "subtasks": [], "retries": 0, "history": [], "history_summary": "",
    }
    out = agent.verifier_node(state)
    # 还有重试次数：回 answerer 重答，answer_status 这时还没定论（不写）
    assert out["retry"] is True
    assert out["retry_target"] == "answerer"
    assert out["retries"] == 1
    assert any(f["reason"] == "fabricated_citation"
               for f in out.get("verify_failures", []))
    # 重试用完：直接拒答，定论 refused
    out2 = agent.verifier_node({**state, "retries": 2})
    assert out2["answer_status"] == "refused"
    assert out2["retry"] is False
    assert out2["final"] == agent.REFUSAL


class _FakeSnapshot:
    def __init__(self, values):
        self.values = values


class _FakeGraph:
    def __init__(self, values=None, boom=False):
        self._values = values
        self._boom = boom

    def get_state(self, config):
        if self._boom:
            raise RuntimeError("库坏了")
        return _FakeSnapshot(self._values)


def test_restore_history_graph_failure_and_empty():
    assert agent._restore_history(_FakeGraph(boom=True), {}) == []
    assert agent._restore_history(_FakeGraph(None), {}) == []
    assert agent._restore_history(_FakeGraph({}), {}) == []
    assert agent._restore_history_summary(_FakeGraph(boom=True), {}) == ""
    assert agent._restore_history_summary(_FakeGraph(None), {}) == ""


def test_restore_history_ok():
    hist = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    graph = _FakeGraph({"history": hist, "history_summary": "摘要"})
    assert agent._restore_history(graph, {}) == hist
    assert agent._restore_history_summary(graph, {}) == "摘要"


def test_dedupe_pubmed_citations():
    results = [
        {"pmid": "111", "title": "A", "journal": "J", "year": "2024"},
        {"pmid": "111", "title": "A again", "journal": "J", "year": "2024"},
        {"pmid": "", "title": "no pmid", "journal": "J", "year": "2024"},
        {"pmid": "222", "title": "B", "journal": "J", "year": "2023"},
    ]
    out = agent._dedupe_pubmed_citations(results)
    assert [c["pmid"] for c in out] == ["111", "222"]
    assert out[0]["source"] == "PubMed"


def test_to_response_pubmed_citations():
    result = {"final": "答案 PubMed PMID:111。",
              "answer_status": "verified", "verify_failures": [],
              "pubmed_results": [{"pmid": "111", "title": "A",
                                  "journal": "J", "year": "2024"}]}
    out = agent._to_response(result)
    assert out["answer"] == "答案 PubMed PMID:111。"
    assert out["verified"] is True
    assert [c["pmid"] for c in out["citations"]] == ["111"]


# ---------- tools：_chart_read 剩余分支 ----------

def test_chart_read_image_read_failure(tmp_path, monkeypatch):
    # 图片存在但读不出来：available=False
    import pathlib

    img = tmp_path / "fig.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    def boom(self):
        raise OSError("磁盘读失败")

    monkeypatch.setattr(pathlib.Path, "read_bytes", boom)
    out = tools._chart_read(image_path=str(img), question="这张图讲了什么？")
    assert out["available"] is False
    assert "读图片失败" in out["reason"]


def test_chart_read_image_success(monkeypatch):
    # 图片路径 + VLM 正常：available=True，带 reading 和 verification
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        path = f.name
    monkeypatch.setattr("app.vision.call_vlm",
                        lambda image_bytes, prompt: "图里有三幅面板，95% 置信区间。")
    try:
        out = tools._chart_read(image_path=path, question="这张图讲了什么？")
    finally:
        Path(path).unlink(missing_ok=True)
    assert out["available"] is True
    assert "三幅面板" in out["reading"]
    assert out["verification"]["verified"] is False  # 没图注可核对，降级
    assert out["downgraded"] is True


def test_chart_read_caption_not_found(monkeypatch):
    # 论文 + 图号，但图注库里没有这张图：available=False
    monkeypatch.setattr("app.tools._find_caption", lambda paper, no: {"found": False})
    out = tools._chart_read(paper="Fujita", figure_no=9, question="图 9 讲了什么？")
    assert out["available"] is False
    assert "本地库里没找到" in out["reason"]


# ---------- 第三批：图跑通、checkpointer、长摘要窗口 ----------

def test_classify_by_llm_bad_json_with_braces(monkeypatch):
    # 大括号齐全但内容不是合法 JSON：json.loads 抛异常，退回简单问答
    monkeypatch.setattr("app.agent._call_llm", lambda p: "{not json}")
    assert agent._classify_by_llm("这个结果可靠吗？") == agent._empty_route()


def test_parse_plan_bad_json_with_braces():
    assert agent._parse_plan("{not json}") == ("", [])


def test_parse_tool_call_bad_json_with_braces():
    assert agent._parse_tool_call("{not json}") == ("local_search", {})


def test_abstract_windows_long_and_short():
    # 摘要太短（<20 字）不切；40 字以内整段一个窗口；更长按 40 字、步长 5 切
    short = {"pmid": "1", "abstract": "too short"}
    medium = {"pmid": "2", "abstract": "a" * 40}
    long_abs = {"pmid": "3", "abstract": "b" * 100}
    out = agent._abstract_windows([short, medium, long_abs])
    assert all(pmid != "1" for _, pmid in out)
    assert ("a" * 40, "2") in out
    long_windows = [w for w, pmid in out if pmid == "3"]
    assert long_windows[0] == "b" * 40
    assert len(long_windows) == (100 - 40) // 5 + 1


def test_get_checkpointer_cm_import_failure(monkeypatch):
    # langgraph.checkpoint.sqlite 导不进来：返回 None，问答退回无持久化
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.sqlite", None)
    assert agent._get_checkpointer_cm() is None


def test_get_checkpointer_cm_conn_string_failure(monkeypatch, tmp_path):
    # 打开库文件失败：返回 None
    import langgraph.checkpoint.sqlite as sqlite_mod

    monkeypatch.setattr(agent, "CHECKPOINTS_DB", str(tmp_path / "ckpt.db"))
    monkeypatch.setattr(sqlite_mod.SqliteSaver, "from_conn_string",
                        classmethod(lambda cls, s: (_ for _ in ()).throw(RuntimeError("打不开"))))
    assert agent._get_checkpointer_cm() is None


def test_run_figure_goes_through_chart_read(monkeypatch):
    # 图表题走 chart_read 节点：VLM 不可用时退回 retriever，问答照常完成
    chunks = [{"text": "总白质体积在 28.7 岁达到峰值。",
               "source": "2022-bethlehem-brain-charts.pdf", "page": 3}]
    answer = "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]"
    monkeypatch.setattr("app.rag.retrieve", lambda query, top_k=10, llm_call=None: chunks)
    monkeypatch.setattr("app.agent._call_llm", lambda prompt: answer)
    monkeypatch.setattr("app.agent.call_tool",
                        lambda name, **kw: {"success": False, "data": None,
                                            "error": "VLM 不可用"})
    result = agent.run("Fujita 图 4 的曲线趋势如何？")
    assert result["answer_status"] == "verified"
    assert result["verified"] is True


def test_run_retriever_then_pubmed(monkeypatch):
    # 本地没证据、PubMed 也没结果：检索员之后走 PubMed 节点，最终拒答
    monkeypatch.setattr("app.rag.retrieve", lambda query, top_k=10, llm_call=None: [])
    monkeypatch.setattr("app.agent._call_llm", lambda prompt: agent.REFUSAL)
    monkeypatch.setattr("app.mcp_tools.search_pubmed",
                        lambda query, top_k=30, sort="relevance": [])
    result = agent.run("总白质体积在几岁达到峰值？", use_pubmed=True)
    assert result["answer_status"] == "refused"
    assert result["answer"] == agent.REFUSAL


def test_run_with_history_and_thread(monkeypatch, tmp_path):
    # 传了 history + thread_id：用传来的历史，不用库里恢复的
    chunks = [{"text": "总白质体积在 28.7 岁达到峰值。",
               "source": "2022-bethlehem-brain-charts.pdf", "page": 3}]
    answer = "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]"
    monkeypatch.setattr("app.agent.CHECKPOINTS_DB", str(tmp_path / "ckpt.db"))
    monkeypatch.setattr("app.rag.retrieve", lambda query, top_k=10, llm_call=None: chunks)
    monkeypatch.setattr("app.agent._call_llm", lambda prompt: answer)
    hist = [{"role": "user", "content": "上一轮的问题"},
            {"role": "assistant", "content": "上一轮的回答"}]
    result = agent.run("总白质体积在几岁达到峰值？", history=hist, thread_id="t1")
    assert result["answer_status"] == "verified"
    assert result["verified"] is True


def test_run_checkpointer_failure_falls_back(monkeypatch):
    # checkpointer 建不起来：退回无持久化，问答照常
    class _BoomCM:
        def __enter__(self):
            raise RuntimeError("库坏了")

        def __exit__(self, *args):
            return False

    canned = {"answer": "x", "verified": True, "answer_status": "verified",
              "citations": [], "verify_failures": []}
    monkeypatch.setattr("app.agent._get_checkpointer_cm", lambda: _BoomCM())
    monkeypatch.setattr("app.agent._run_with_graph",
                        lambda query, history=None, use_pubmed=True: canned)
    assert agent.run("总白质体积在几岁达到峰值？", thread_id="t1") == canned


# ---------- 第四批：顺序执行兜底（LangGraph 不可用时） ----------

def test_sequential_figure_goes_through_chart_read(monkeypatch):
    # 图不可用时退回顺序执行：图表题走 chart_read 节点
    chunks = [{"text": "总白质体积在 28.7 岁达到峰值。",
               "source": "2022-bethlehem-brain-charts.pdf", "page": 3}]
    answer = "总白质体积在 28.7 岁达到峰值 [来源：2022-bethlehem-brain-charts.pdf p.3]"
    monkeypatch.setattr("app.agent._build_graph",
                        lambda checkpointer=None: (_ for _ in ()).throw(RuntimeError("图挂了")))
    monkeypatch.setattr("app.rag.retrieve", lambda query, top_k=10, llm_call=None: chunks)
    monkeypatch.setattr("app.agent._call_llm", lambda prompt: answer)
    monkeypatch.setattr("app.agent.call_tool",
                        lambda name, **kw: {"success": False, "data": None,
                                            "error": "VLM 不可用"})
    result = agent.run("Fujita 图 4 的曲线趋势如何？")
    assert result["answer_status"] == "verified"
    assert result["verified"] is True


def test_sequential_retriever_then_pubmed(monkeypatch):
    # 图不可用时退回顺序执行：本地没证据时检索员之后走 PubMed 节点
    monkeypatch.setattr("app.agent._build_graph",
                        lambda checkpointer=None: (_ for _ in ()).throw(RuntimeError("图挂了")))
    monkeypatch.setattr("app.rag.retrieve", lambda query, top_k=10, llm_call=None: [])
    monkeypatch.setattr("app.agent._call_llm", lambda prompt: agent.REFUSAL)
    monkeypatch.setattr("app.mcp_tools.search_pubmed",
                        lambda query, top_k=30, sort="relevance": [])
    result = agent.run("总白质体积在几岁达到峰值？", use_pubmed=True)
    assert result["answer_status"] == "refused"
    assert result["answer"] == agent.REFUSAL


def test_chart_read_pdf_missing(monkeypatch):
    # 图注找到了，但本地没有这篇 PDF：available=False
    monkeypatch.setattr("app.tools._find_caption",
                        lambda paper, no: {"found": True, "kind": "Figure",
                                           "page": 8, "caption": "Figure 4. x.",
                                           "source": "2099-nonexistent-paper.pdf"})
    out = tools._chart_read(paper="Nonexistent", figure_no=4, question="图 4 讲了什么？")
    assert out["available"] is False
    assert "本地没有该论文的 PDF" in out["reason"]
