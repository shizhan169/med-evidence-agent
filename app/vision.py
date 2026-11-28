"""真正多模态：VLM 读图（第二阶段 2.3）。

P0：按图号/图注定位页面，用 PyMuPDF 把指定 PDF 页面渲染成图片，
连同问题发给 Qwen-VL。
P1：把图表区域裁剪后再发给 VLM（按图注位置切出图表所在的纵向带子）。
P2：自动识别复杂图表边界及子图——未做，有空再做，不阻塞。

VLM 输出核查：读出的数字与图注交叉核对；对不上的数字降级说明，
不硬编精确数。引用指向真实图表（source 是论文文件名，page 是图注所在页）。

没有图片时不强行调 VLM——调用方（chart_read 工具）在没图时直接返回
available=False，走“证据不足/部分回答”分支。

VM 上没有 Qwen-VL 的 key，跑不了真实 VLM：测试用注入的假 vlm_call
验证逻辑；真实读图验证由用户在本地做。
"""
import base64
import re
from pathlib import Path

from .config import VLM_BASE_URL, VLM_MODEL, get_vlm_api_key
from .prompts import load_prompt

# 图注行的开头格式（和 ingest._extract_captions、tools._FIGURE_RE 一致）：
# "Figure 3. ..." / "Table 2. ..."
_FIGURE_LINE_RE = re.compile(r"^(Figure|Table)\s*(\d+)", re.IGNORECASE)
# 提数字前先去掉的引用型 token，避免“Figure 4”里的 4 被当成数据去核对
_REF_TOKEN_RE = re.compile(r"(Figure|Table)\s*\d+|p\.\s*\d+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<!\d)-?\d[\d,]*\.?\d*\s*%?")
# (?<!\d) 是关键：区间里的连字符不是负号，"18-21" 要拆成 18 和 21，
# 不能把后一个读成 -21。

DEFAULT_DPI = 150
# 裁剪出的带子低于这个高度（PDF 坐标点）就不像一张图，退回整页渲染
_MIN_CLIP_HEIGHT = 80


def render_page_to_png(pdf_path, page, dpi=DEFAULT_DPI, clip=None):
    """把 PDF 指定页渲染成 PNG 字节。page 是 1-based（和入库的元数据一致，
    fitz 内部用 0-based，这里换算）。clip 是可选的 (x0, y0, x1, y1)
    裁剪框（PDF 坐标点），超出页面自动夹回页面内。"""
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    try:
        if page < 1 or page > len(doc):
            raise ValueError(f"页码 {page} 超出范围（共 {len(doc)} 页）")
        pg = doc[page - 1]
        rect = None
        if clip is not None:
            rect = pymupdf.Rect(*clip) & pg.rect
            if rect.is_empty:
                rect = None
        pix = pg.get_pixmap(dpi=dpi, clip=rect)
        return pix.tobytes("png")
    finally:
        doc.close()


def find_figure_clip(pdf_path, page, figure_no):
    """P1：按图号在页面上找图注行，切出“图注到下一个图注”之间的纵向带子。
    返回 (x0, y0, x1, y1)（PDF 坐标点）；找不到图注、或带子太窄，
    返回 None（调用方退回整页渲染）。
    假设图注在图的上方（Fujita 这类论文是这样）；图注在下方的论文
    会切偏，届时退回整页仍能读图，只是多花点 token。"""
    import pymupdf

    num_m = re.search(r"\d+", str(figure_no or ""))
    if not num_m:
        return None
    num = num_m.group(0)
    doc = pymupdf.open(str(pdf_path))
    try:
        if page < 1 or page > len(doc):
            return None
        pg = doc[page - 1]
        # 收集本页所有图注/表注行的纵坐标（按行拼文本再认，比按 span 稳）
        caps = []
        for block in pg.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                text = "".join(s["text"] for s in line["spans"]).strip()
                m = _FIGURE_LINE_RE.match(text)
                if not m:
                    continue
                y0 = min(s["bbox"][1] for s in line["spans"])
                caps.append((y0, m.group(2)))
        caps.sort()
        target_y = None
        for y0, n in caps:
            if n == num:
                target_y = y0
                break
        if target_y is None:
            return None
        # 下一个图注的纵坐标；没有就用页底
        next_y = float(pg.rect.height)
        for y0, n in caps:
            if y0 > target_y + 1:
                next_y = y0
                break
        x0 = 0.0
        x1 = float(pg.rect.width)
        y0 = max(0.0, target_y - 10)
        y1 = min(float(pg.rect.height), next_y - 10)
        if y1 - y0 < _MIN_CLIP_HEIGHT:
            return None
        return (x0, y0, x1, y1)
    finally:
        doc.close()


def _fmt_num(n):
    """数字显示：整数不带 .0（95.0 印成 95），小数保留原值。"""
    if float(n).is_integer():
        return str(int(n))
    return str(n)


def extract_numbers(text):
    """从文本里抽数字（去千分位逗号、去百分号），返回 float 列表。
    先去掉 Figure N / Table N / p.N 这类引用型 token，避免把图号、
    页码当成数据。例子：“95% 置信区间”得 [95.0]，“18-21 岁”得 [18.0, 21.0]。"""
    clean = _REF_TOKEN_RE.sub(" ", text or "")
    nums = []
    for m in _NUMBER_RE.finditer(clean):
        tok = m.group(0).replace(",", "").replace("%", "").strip()
        try:
            nums.append(float(tok))
        except ValueError:
            continue
    return nums


def verify_reading(reading, reference_text):
    """把 VLM 读出的数字和图注（reference）交叉核对。
    返回 {checked, verified, verified_numbers, unverified_numbers, note}。
    - reference 为空：没东西可核对，checked=False，数字全算未核对
      （不删除，只提示；调用方据此降级）。
    - 数字在 reference 里出现过（相对误差 1% 内）：verified。
    - 对不上：unverified，整体降级。
    注意：核对不上不等于读错了——图里的数值本来就不一定写在图注里；
    所以对不上的数字只降级说明，不删除、不硬改。"""
    read_nums = extract_numbers(reading or "")
    ref_text = (reference_text or "").strip()
    if not read_nums:
        return {"checked": True, "verified": True, "verified_numbers": [],
                "unverified_numbers": [], "note": ""}
    if not ref_text:
        return {"checked": False, "verified": False, "verified_numbers": [],
                "unverified_numbers": read_nums,
                "note": "无图注可供交叉核对，读出的数字未经核实"}
    ref_nums = extract_numbers(ref_text)
    verified, unverified = [], []
    for n in read_nums:
        ok = any(abs(n - r) <= max(1e-9, abs(r) * 0.01) for r in ref_nums)
        (verified if ok else unverified).append(n)
    if not unverified:
        return {"checked": True, "verified": True,
                "verified_numbers": verified, "unverified_numbers": [], "note": ""}
    return {"checked": True, "verified": False,
            "verified_numbers": verified, "unverified_numbers": unverified,
            "note": "以下数字在图注里对不上，仅供参考："
                    + "、".join(_fmt_num(n) for n in unverified)}


def _chart_reader_prompt(question, caption):
    """拼给 VLM 的提示词：图注 + 问题。优先用 prompts/chart_reader/v1.txt，
    文件缺了就用内置的最小版（VLM 提示词不是问答核心，不必像 answerer
    那样硬报错）。"""
    try:
        template = load_prompt("chart_reader", "v1")
    except FileNotFoundError:
        template = None
    if template:
        return template.format(question=question or "", caption=caption or "")
    parts = ["请仔细看这张图表，并回答问题。"]
    if caption:
        parts.append("图注：" + caption)
    parts.append("问题：" + (question or "这张图展示了什么？"))
    parts.append("只说图里能清楚读到的内容；数字读不准时说“约”或说明读不出来，不要硬编精确数。")
    return "\n".join(parts)


def _default_vlm_call(image_bytes, prompt_text):
    """真实 VLM 调用：Qwen-VL，走 OpenAI 兼容接口。key 用 get_vlm_api_key()
   （优先 VLM_API_KEY，没填就跟 LLM 用同一个）。没 key 时直接报错，
    调用方转成 available=False，不让问答断掉。"""
    api_key = get_vlm_api_key()
    if not api_key:
        raise RuntimeError("没有 API key，无法调用 VLM")
    from openai import OpenAI

    b64 = base64.b64encode(image_bytes).decode("ascii")
    client = OpenAI(api_key=api_key, base_url=VLM_BASE_URL)
    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


def call_vlm(image_bytes, question, caption="", vlm_call=None):
    """调 VLM 读一张图。vlm_call 是注入的调用函数（测试用假 VLM），
    不传时用真实 Qwen-VL。返回 VLM 的原始回答文本。"""
    prompt = _chart_reader_prompt(question, caption)
    fn = vlm_call or _default_vlm_call
    return fn(image_bytes, prompt)


def _find_figure_page(pdf_path, page):
    """图实际在哪页：图注所在的页不一定有图（Fujita 图 4 的图注在 p.7，
    图本身在 p.8，p.7 一张图片都没有）。先看图注页有没有图片，
    没有就往附近页找（+1、-1、+2、-2），找到有图片的页就用它；
    附近都没有图片时退回原页（调用方按老路走，不中断）。
    返回 (page, found_images)。"""
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    try:
        total = len(doc)
        for cand in [page, page + 1, page - 1, page + 2, page - 2]:
            if cand < 1 or cand > total:
                continue
            if doc[cand - 1].get_images():
                return cand, True
        return page, False
    finally:
        doc.close()


def read_chart(pdf_path, page, figure_no, question, caption="",
               vlm_call=None, dpi=DEFAULT_DPI, crop=True):
    """读一张图（P0 + P1）：定位、渲染（能裁就裁）、VLM、数字交叉核对。
    返回结构化结果；任何一步失败都返回 available=False（不抛异常），
    调用方按“证据不足/部分回答”走。
    成功时：available=True，reading 是 VLM 原文，verification 是交叉核对结果，
    downgraded=True 表示有数字在图注里对不上（只降级说明，不删除）。
    注意：先由 _find_figure_page 确认图实际在哪页——图注页不一定有图，
    读错页时 VLM 只能看到图注文字、看不到图。"""
    source = Path(pdf_path).name
    try:
        page, _has_images = _find_figure_page(pdf_path, page)
    except Exception:
        pass
    try:
        clip = find_figure_clip(pdf_path, page, figure_no) if crop else None
    except Exception:
        clip = None
    try:
        image_bytes = render_page_to_png(pdf_path, page, dpi=dpi, clip=clip)
    except Exception as e:
        return {"available": False, "reason": f"渲染页面失败：{e}",
                "source": source, "page": page, "figure_no": str(figure_no)}
    try:
        reading = call_vlm(image_bytes, question, caption=caption, vlm_call=vlm_call)
    except Exception as e:
        return {"available": False, "reason": f"VLM 调用失败：{e}",
                "source": source, "page": page, "figure_no": str(figure_no)}
    verification = verify_reading(reading, caption)
    downgraded = bool(extract_numbers(reading)) and not verification["verified"]
    return {
        "available": True,
        "source": source,
        "page": page,
        "figure_no": str(figure_no),
        "caption": caption,
        "reading": reading,
        "verification": verification,
        "downgraded": downgraded,
        "cropped": clip is not None,
        "reason": "",
    }
