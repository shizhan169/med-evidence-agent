"""统一工具注册表（框架 v2，第一阶段 1.3）。

每个工具 = 名字 + 描述 + 输入 schema + 执行函数。
工具调用统一经过参数校验（白名单 + 类型检查），返回统一格式
{success, data, error}，失败不抛异常——调用方按失败分支走，问答不断。

注册的工具：
- local_search：本地 RAG 检索（包装 app.rag.retrieve）
- pubmed_search：PubMed 搜索（包装 app.mcp_tools.search_pubmed）
- chart_locate：图表定位（按图号/图注定位页面，复用入库的图注片段）
- chart_read：VLM 读图（第二阶段 2.3）：按图号定位页面，渲染成图片调 Qwen-VL；
  没有本地图片时返回 available=False，不强行读图

注意：检索文本视为数据而非指令——不把检索到的文本直接拼进工具调用参数。
"""
import re
from pathlib import Path

from .config import CHROMA_DIR

# 图注文本的开头格式（见 ingest._extract_captions）："Figure 3. ..." / "Table 2. ..."
_FIGURE_RE = re.compile(r"^(Figure|Table)\s*(\d+)", re.IGNORECASE)


def _type_ok(value, want) -> bool:
    """参数类型检查。want 是 "str"/"int"/"bool" 或它们的列表。
    bool 和 int 严格区分（Python 里 bool 是 int 的子类，要单拎出来）。"""
    wants = [want] if isinstance(want, str) else want
    for w in wants:
        if w == "bool":
            if isinstance(value, bool):
                return True
        elif w == "int":
            if isinstance(value, int) and not isinstance(value, bool):
                return True
        elif w == "str":
            if isinstance(value, str):
                return True
    return False


class Tool:
    """一个工具：名字 + 描述 + 输入 schema + 执行函数。
    params 的形状：{参数名: {"type": "str"/"int"/"bool" 或列表,
    "required": True/False, "default": 默认值（可选参数用）}}。"""

    def __init__(self, name, description, params, func):
        self.name = name
        self.description = description
        self.params = params
        self.func = func


_TOOLS = {}


def register(tool: Tool) -> None:
    """往注册表里加一个工具（同名会覆盖）。"""
    _TOOLS[tool.name] = tool


def list_tools() -> list:
    """注册表里的工具（名字 + 描述），供 Planner/Executor 选工具时参考。"""
    return [{"name": t.name, "description": t.description} for t in _TOOLS.values()]


def call_tool(name: str, **kwargs) -> dict:
    """统一调用入口：先校验（白名单 + 类型），再执行，返回统一格式。
    任何失败都返回 {"success": False, "data": None, "error": "..."}，不抛异常。"""
    tool = _TOOLS.get(name)
    if tool is None:
        return {"success": False, "data": None, "error": f"未知工具：{name}"}
    # 白名单：不在 schema 里的参数不接受
    for key in kwargs:
        if key not in tool.params:
            return {"success": False, "data": None, "error": f"工具 {name} 没有参数 {key}"}
    call_kwargs = {}
    for pname, spec in tool.params.items():
        if pname in kwargs:
            if not _type_ok(kwargs[pname], spec["type"]):
                want = spec["type"] if isinstance(spec["type"], str) else "/".join(spec["type"])
                return {"success": False, "data": None,
                        "error": f"参数 {pname} 应为 {want}，实际是 {type(kwargs[pname]).__name__}"}
            call_kwargs[pname] = kwargs[pname]
        elif spec.get("required"):
            return {"success": False, "data": None, "error": f"工具 {name} 缺少必填参数 {pname}"}
        elif "default" in spec:
            call_kwargs[pname] = spec["default"]
    try:
        data = tool.func(**call_kwargs)
    except Exception as e:
        return {"success": False, "data": None, "error": f"工具 {name} 执行失败：{e}"}
    return {"success": True, "data": data, "error": None}


def _local_search(query: str, top_k: int = 10) -> dict:
    """本地 RAG 检索：包装 app.rag.retrieve。库不可用时 retrieve 自己返回 []。"""
    from .rag import retrieve

    return {"chunks": retrieve(query, top_k=top_k)}


def _pubmed_search(query: str, top_k: int = 10) -> dict:
    """PubMed 搜索：包装 app.mcp_tools.search_pubmed。失败返回 []，不抛异常。"""
    from .mcp_tools import search_pubmed

    return {"results": search_pubmed(query, top_k=top_k)}


def _open_collection():
    """打开 Chroma 库（只读图注用，不加载 embedding 模型）。
    collection.get() 不算向量，不需要 embedding 函数，用默认的即可。"""
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection("medevidence")


def _find_caption(paper: str, figure_no) -> dict:
    """在已入库的图注片段里找指定论文、指定图号的图注。
    图注片段的文本以 "Figure N. ..." / "Table N. ..." 开头（见
    ingest._extract_captions），元数据 type=caption。库不存在、没这篇、
    没这张图，都返回 {found: False}，不抛异常。"""
    num_m = re.search(r"\d+", str(figure_no or ""))
    if not num_m:
        return {"found": False}
    num = num_m.group(0)
    try:
        collection = _open_collection()
        data = collection.get(where={"type": "caption"}, include=["documents", "metadatas"])
    except Exception as e:
        print(f"[chart_locate] 打开图注库失败：{e}")
        return {"found": False}
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    paper_l = (paper or "").lower()
    for doc, meta in zip(docs, metas):
        source = str(meta.get("source", "") or "")
        # 论文名匹配：传全名或关键词都行，双向包含就算（大小写不敏感）
        if paper_l and paper_l not in source.lower() and source.lower() not in paper_l:
            continue
        m = _FIGURE_RE.match((doc or "").strip())
        if not m or m.group(2) != num:
            continue
        return {
            "found": True,
            "kind": m.group(1).capitalize(),
            "page": meta.get("page", "?"),
            "caption": (doc or "").strip(),
            "source": source,
        }
    return {"found": False}


def _chart_locate(paper: str, figure_no) -> dict:
    return _find_caption(paper, figure_no)


def _chart_read(paper: str = "", figure_no="", question: str = "",
                image_path: str = "") -> dict:
    """VLM 读图（第二阶段 2.3）：按图号定位页面，渲染成图片，调 Qwen-VL 读图。
    两种输入：
    - image_path：已有图片路径，直接读这张图（没有图注可供交叉核对）。
    - paper + figure_no：本地论文 + 图号，先定位图注拿页码和图注，
      再从 PAPERS_DIR 找 PDF 渲染该页，读完和图注交叉核对。
    没有图片时（本地没这篇 PDF、图号找不到、图片不存在）返回
    available=False，不强行调 VLM——调用方按“证据不足”走。
    返回 available=True 时带 reading（VLM 原文）、verification（交叉核对）、
    downgraded（有数字对不上时 True，只降级说明）。"""
    from .vision import call_vlm, extract_numbers, read_chart, verify_reading

    if image_path:
        p = Path(image_path)
        if not p.is_file():
            return {"available": False, "reason": f"图片不存在：{image_path}"}
        try:
            image_bytes = p.read_bytes()
        except Exception as e:
            return {"available": False, "reason": f"读图片失败：{e}"}
        try:
            reading = call_vlm(image_bytes, question)
        except Exception as e:
            return {"available": False, "reason": f"VLM 调用失败：{e}"}
        verification = verify_reading(reading, "")
        return {"available": True, "source": "", "page": None, "figure_no": "",
                "caption": "", "reading": reading, "verification": verification,
                "downgraded": bool(extract_numbers(reading)) and not verification["verified"],
                "cropped": False, "reason": ""}
    if not paper or figure_no in ("", None):
        return {"available": False,
                "reason": "读图需要图片路径，或论文名 + 图号（本地没有图片时不强行调 VLM）"}
    located = _find_caption(paper, figure_no)
    if not located.get("found"):
        return {"available": False,
                "reason": f"本地库里没找到 {paper} 的图 {figure_no}，没有图片可读"}
    from .config import PAPERS_DIR

    pdf_path = Path(PAPERS_DIR) / located["source"]
    if not pdf_path.is_file():
        return {"available": False,
                "reason": f"本地没有该论文的 PDF（{located['source']}），无法渲染图片"}
    return read_chart(pdf_path, located["page"], figure_no, question,
                      caption=located.get("caption", ""))


register(Tool(
    name="local_search",
    description="在本地文献库里检索与问题相关的片段（向量 + BM25 混合检索）。用于回答库里已有论文的问题。",
    params={
        "query": {"type": "str", "required": True},
        "top_k": {"type": "int", "required": False, "default": 10},
    },
    func=_local_search,
))

register(Tool(
    name="pubmed_search",
    description="在 PubMed 上搜索最新或库外文献（英文检索式）。用于问最新文献、库里没有的内容。",
    params={
        "query": {"type": "str", "required": True},
        "top_k": {"type": "int", "required": False, "default": 10},
    },
    func=_pubmed_search,
))

register(Tool(
    name="chart_locate",
    description="按图号定位某篇论文里的图表：返回图注所在的页码和图注文字。用于问某张图之前先找到它在哪。",
    params={
        "paper": {"type": "str", "required": True},
        "figure_no": {"type": ["str", "int"], "required": True},
    },
    func=_chart_locate,
))

register(Tool(
    name="chart_read",
    description="用 VLM 读取本地论文里的图表（坐标轴、趋势、数值、图例）。参数 paper 写论文名、figure_no 写图号、question 写要读的问题；已有图片路径时传 image_path 直接读那张图。没有本地图片时返回 available=False，不强行读图。",
    params={
        "paper": {"type": "str", "required": False, "default": ""},
        "figure_no": {"type": ["str", "int"], "required": False, "default": ""},
        "question": {"type": "str", "required": False, "default": ""},
        "image_path": {"type": "str", "required": False, "default": ""},
    },
    func=_chart_read,
))
