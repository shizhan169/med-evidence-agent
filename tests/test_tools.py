"""工具注册表测试（框架 v2，第一阶段 1.3）：不需要 Chroma、不需要 LLM、不联网。
检索和 PubMed 都换成假的；chart_locate 的 Chroma 库换成假集合。
跑法：pytest tests/test_tools.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import tools

FAKE_CHUNKS = [{"text": "总白质体积在 28.7 岁达到峰值。", "source": "2022-bethlehem-brain-charts.pdf", "page": 3}]
FAKE_PUBMED = [{"pmid": "38395541", "title": "CentileBrain", "authors": "Ge et al.",
                "journal": "Nat Ment Health", "year": 2024, "abstract": "xxx"}]


def test_list_tools_has_four():
    listed = tools.list_tools()
    names = {t["name"] for t in listed}
    assert names == {"local_search", "pubmed_search", "chart_locate", "chart_read"}
    # 每个工具都有描述（Planner/Executor 选工具时靠它判断）
    assert all(t["description"] for t in listed)


# ---------- 三个可用工具 ----------

def test_local_search(monkeypatch):
    seen = {}

    def fake_retrieve(query, top_k=10):
        seen["query"] = query
        seen["top_k"] = top_k
        return FAKE_CHUNKS

    monkeypatch.setattr("app.rag.retrieve", fake_retrieve)
    result = tools.call_tool("local_search", query="总白质体积峰值")
    assert result["success"] is True
    assert result["error"] is None
    assert result["data"]["chunks"] == FAKE_CHUNKS
    assert seen["query"] == "总白质体积峰值"
    assert seen["top_k"] == 10  # 没传 top_k 时用默认值


def test_pubmed_search(monkeypatch):
    monkeypatch.setattr(
        "app.mcp_tools.search_pubmed",
        lambda query, top_k=10, sort="relevance": FAKE_PUBMED)
    result = tools.call_tool("pubmed_search", query="longitudinal normative modeling", top_k=5)
    assert result["success"] is True
    assert result["error"] is None
    assert result["data"]["results"] == FAKE_PUBMED


class _FakeCollection:
    """假 Chroma 集合：只实现 chart_locate 用到的 get()。"""

    def __init__(self, docs, metas):
        self._docs = docs
        self._metas = metas

    def get(self, where=None, include=None):
        return {"documents": self._docs, "metadatas": self._metas}


CAPTION_DOCS = ["Figure 3. Ridgeline plots showing distributions of rates of change.",
                "Figure 4. GAM smooths with 95% confidence intervals."]
CAPTION_METAS = [{"source": "2023-fujita-brain-volume-aging.pdf", "page": 8},
                 {"source": "2023-fujita-brain-volume-aging.pdf", "page": 8}]


def test_chart_locate_found(monkeypatch):
    monkeypatch.setattr("app.tools._open_collection",
                        lambda: _FakeCollection(CAPTION_DOCS, CAPTION_METAS))
    result = tools.call_tool("chart_locate", paper="fujita", figure_no=3)
    assert result["success"] is True
    assert result["data"]["found"] is True
    assert result["data"]["page"] == 8
    assert "Ridgeline" in result["data"]["caption"]


def test_chart_locate_found_by_string_no(monkeypatch):
    # figure_no 传字符串 "图 4" 也能定位（只取里面的数字）
    monkeypatch.setattr("app.tools._open_collection",
                        lambda: _FakeCollection(CAPTION_DOCS, CAPTION_METAS))
    result = tools.call_tool("chart_locate", paper="2023-fujita-brain-volume-aging.pdf",
                             figure_no="图 4")
    assert result["data"]["found"] is True
    assert "GAM smooths" in result["data"]["caption"]


def test_chart_locate_not_found(monkeypatch):
    monkeypatch.setattr("app.tools._open_collection",
                        lambda: _FakeCollection(CAPTION_DOCS, CAPTION_METAS))
    result = tools.call_tool("chart_locate", paper="fujita", figure_no=99)
    assert result["success"] is True
    assert result["data"]["found"] is False


def test_chart_locate_real_stored_format(monkeypatch):
    # 回归：真实库里存的文档是 context 行 + 图注正文
    # （rag.add_chunks 把 context 贴在前面），不能只认裸 "Figure N." 开头。
    # 2026-11-28 用户本地验证发现：不剥 context 行，_find_caption 在真实库里
    # 一张图都找不到，chart_read 直接报 available=False，VLM 根本没被调到。
    real_docs = [
        "【文献：Fujita 2023｜第 8 页｜结果】 figure3\n"
        "Figure 3. Ridgeline plots showing distributions of rates of change.",
        "【文献：Fujita 2023｜第 8 页｜结果】 figure4\n"
        "Figure 4. GAM smooths with 95% confidence intervals.",
    ]
    monkeypatch.setattr("app.tools._open_collection",
                        lambda: _FakeCollection(real_docs, CAPTION_METAS))
    result = tools.call_tool("chart_locate", paper="fujita", figure_no=4)
    assert result["data"]["found"] is True
    assert result["data"]["page"] == 8
    assert result["data"]["caption"].startswith("Figure 4.")
    assert "【文献" not in result["data"]["caption"]


def test_chart_locate_collection_unavailable(monkeypatch):
    # 库打不开（不存在 / 损坏）：返回 found=False，不抛异常
    def boom():
        raise RuntimeError("库不存在")

    monkeypatch.setattr("app.tools._open_collection", boom)
    result = tools.call_tool("chart_locate", paper="fujita", figure_no=3)
    assert result["success"] is True
    assert result["data"]["found"] is False


def test_chart_read_placeholder():
    # chart_read 是占位（VLM 第二阶段接入）：工具本身调用成功，
    # 但返回 available=False，告诉调用方现在读不了图
    result = tools.call_tool("chart_read", image_path="p8.png", question="图里峰值是多少？")
    assert result["success"] is True
    assert result["data"]["available"] is False


# ---------- 参数校验：白名单 + 类型检查 ----------

def test_unknown_tool():
    result = tools.call_tool("no_such_tool", query="x")
    assert result["success"] is False
    assert "未知工具" in result["error"]


def test_missing_required_param():
    result = tools.call_tool("local_search")
    assert result["success"] is False
    assert "缺少必填参数" in result["error"]


def test_wrong_param_type(monkeypatch):
    monkeypatch.setattr("app.rag.retrieve", lambda query, top_k=10: FAKE_CHUNKS)
    result = tools.call_tool("local_search", query="x", top_k="10")
    assert result["success"] is False
    assert "应为 int" in result["error"]


def test_bool_is_not_int(monkeypatch):
    # Python 里 bool 是 int 的子类，但 top_k=True 显然不是合法的 top_k，
    # 类型检查要把 bool 和 int 严格区分开
    monkeypatch.setattr("app.rag.retrieve", lambda query, top_k=10: FAKE_CHUNKS)
    result = tools.call_tool("local_search", query="x", top_k=True)
    assert result["success"] is False


def test_unknown_param_rejected():
    result = tools.call_tool("local_search", query="x", bogus=1)
    assert result["success"] is False
    assert "没有参数" in result["error"]


def test_tool_exception_returns_failure(monkeypatch):
    # 工具执行中抛异常：返回 success=False，不把异常抛给调用方
    def boom(query, top_k=10):
        raise RuntimeError("检索挂了")

    monkeypatch.setattr("app.rag.retrieve", boom)
    result = tools.call_tool("local_search", query="x")
    assert result["success"] is False
    assert "执行失败" in result["error"]
