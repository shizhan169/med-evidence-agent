"""VLM 读图测试（第二阶段 2.3）：不需要真实 VLM、不需要 API key、不联网。
PyMuPDF 渲染用 VM 里 data/papers/ 的真实 Fujita PDF；VLM 调用全部换成假的。
跑法：pytest tests/test_vision.py -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import tools
from app import vision

FUJITA = ROOT / "data" / "papers" / "2023-fujita-brain-volume-aging.pdf"
# Fujita 图 4 的图注（p.8，内容和入库的图注片段一致）
FIG4_CAPTION = ("Figure 4. Visualization of Age-Dependent Volume Change Rate "
                "for Each Tissue Type. Curves fitted with a generalized additive "
                "model using cubic splines; shaded areas represent 95% "
                "confidence intervals.")
FIG3_CAPTION = ("Figure 3. Ridgeline Plots of the Distribution of Brain Structure "
                "Volume Change Rates for Each Decade.")
# F02 的合理读图结果（假 VLM 返回）：数字 95% 在图注里对得上
F02_READING = ("这张图有三幅面板，分别展示脑室与脑沟、各脑叶、与神经退行性疾病相关"
               "的结构的体积变化率随年龄的变化。每条曲线用广义加性模型（三次样条）"
               "拟合，阴影表示 95% 置信区间。")
F01_READING = "这张图是山脊图，展示各年龄组（每十年）脑结构体积变化率的分布。"


class _FakeVLM:
    """假 VLM：记下收到的图片和提示词，返回写死的读图结果。"""

    def __init__(self, text):
        self.text = text
        self.calls = []

    def __call__(self, image_bytes, prompt):
        self.calls.append({"image_bytes": image_bytes, "prompt": prompt})
        return self.text


class _FakeCollection:
    """假 Chroma 集合：只实现 chart_locate 用到的 get()。"""

    def __init__(self, docs, metas):
        self._docs = docs
        self._metas = metas

    def get(self, where=None, include=None):
        return {"documents": self._docs, "metadatas": self._metas}


# ---------- 渲染 ----------

def test_render_page_png():
    data = vision.render_page_to_png(FUJITA, 8)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 10000


def test_render_page_bad_page():
    with pytest.raises(ValueError):
        vision.render_page_to_png(FUJITA, 99)


# ---------- P1 裁剪 ----------

def test_find_figure_clip_fig4():
    clip = vision.find_figure_clip(FUJITA, 8, 4)
    assert clip is not None
    x0, y0, x1, y1 = clip
    assert x0 == 0.0
    assert x1 == 612.0  # Letter 纸宽
    assert y0 > 400  # 图 4 图注在 y 约 448，裁剪从图注附近开始
    assert y1 > y0 + 100  # 带子够高，像一张图
    assert y1 <= 792.0  # 不超过页底


def test_find_figure_clip_fig3_above_fig4():
    clip3 = vision.find_figure_clip(FUJITA, 8, 3)
    clip4 = vision.find_figure_clip(FUJITA, 8, 4)
    assert clip3 is not None and clip4 is not None
    assert clip3[1] < clip4[1]  # 图 3 在图 4 上面
    assert clip3[3] <= clip4[1] + 20  # 图 3 的带子在图 4 的带子开始前结束


def test_find_figure_clip_not_found():
    assert vision.find_figure_clip(FUJITA, 8, 99) is None


# ---------- 数字提取与交叉核对 ----------

def test_extract_numbers():
    assert vision.extract_numbers("阴影表示 95% 置信区间") == [95.0]
    assert vision.extract_numbers("纳入 2,944 名受试者") == [2944.0]
    assert vision.extract_numbers("Figure 4 展示的是什么") == []
    assert vision.extract_numbers("18-21 岁那段是虚线") == [18.0, 21.0]


def test_verify_reading_match():
    v = vision.verify_reading("阴影表示 95% 置信区间", FIG4_CAPTION)
    assert v["verified"] is True
    assert v["unverified_numbers"] == []


def test_verify_reading_wrong_number():
    # VLM 把 95% 读成 80%：交叉核对要发现并降级
    v = vision.verify_reading("阴影表示 80% 置信区间", FIG4_CAPTION)
    assert v["verified"] is False
    assert v["unverified_numbers"] == [80.0]
    assert "80" in v["note"]


def test_verify_reading_no_numbers():
    v = vision.verify_reading("这张图是山脊图", FIG4_CAPTION)
    assert v["verified"] is True


def test_verify_reading_no_reference():
    v = vision.verify_reading("阴影表示 95% 置信区间", "")
    assert v["checked"] is False
    assert v["verified"] is False


# ---------- 读图主流程（假 VLM） ----------

def test_read_chart_mock_f02():
    fake = _FakeVLM(F02_READING)
    result = vision.read_chart(FUJITA, 8, 4, "Fujita 2023 的图 4 展示的是什么？",
                               caption=FIG4_CAPTION, vlm_call=fake)
    assert result["available"] is True
    assert result["source"] == "2023-fujita-brain-volume-aging.pdf"
    assert result["page"] == 8
    assert result["figure_no"] == "4"
    assert result["reading"] == F02_READING
    assert result["cropped"] is True  # P1 裁剪生效
    assert result["verification"]["verified"] is True
    assert result["downgraded"] is False
    # VLM 真收到了一张渲染出来的图片（不是空调用）
    assert len(fake.calls) == 1
    assert fake.calls[0]["image_bytes"][:8] == b"\x89PNG\r\n\x1a\n"
    assert "图 4" in fake.calls[0]["prompt"] or "Figure 4" in fake.calls[0]["prompt"]


def test_read_chart_mock_f01():
    fake = _FakeVLM(F01_READING)
    result = vision.read_chart(FUJITA, 8, 3, "Fujita 2023 的图 3 是什么图？",
                               caption=FIG3_CAPTION, vlm_call=fake)
    assert result["available"] is True
    assert result["cropped"] is True
    assert result["reading"] == F01_READING


def test_read_chart_caption_page_has_no_image():
    # 回归：Fujita 图 4 的图注在 p.7，图本身在 p.8，p.7 一张图片都没有。
    # 2026-11-28 用户本地验证：chart_locate 给的是图注页 p.7，read_chart
    # 直接渲染 p.7，VLM 只看到图注文字、看不到图（还以为图里没图表）。
    # 修法：_find_figure_page 先确认图实际在哪页，没图就往附近页找。
    fake = _FakeVLM(F02_READING)
    result = vision.read_chart(FUJITA, 7, 4, "Fujita 2023 的图 4 展示的是什么？",
                               caption=FIG4_CAPTION, vlm_call=fake)
    assert result["available"] is True
    assert result["page"] == 8  # 图注页 p.7 没图，自动切到有图的 p.8
    assert len(fake.calls) == 1


def test_read_chart_no_crop():
    fake = _FakeVLM(F02_READING)
    result = vision.read_chart(FUJITA, 8, 4, "图 4 展示什么？",
                               caption=FIG4_CAPTION, vlm_call=fake, crop=False)
    assert result["available"] is True
    assert result["cropped"] is False


def test_read_chart_missing_pdf():
    fake = _FakeVLM(F02_READING)
    result = vision.read_chart(ROOT / "data" / "papers" / "不存在.pdf", 8, 4,
                               "图 4 展示什么？", caption=FIG4_CAPTION, vlm_call=fake)
    assert result["available"] is False
    assert len(fake.calls) == 0  # 没图就不调 VLM


def test_read_chart_vlm_raises():
    def boom(image_bytes, prompt):
        raise RuntimeError("VLM 挂了")

    result = vision.read_chart(FUJITA, 8, 4, "图 4 展示什么？",
                               caption=FIG4_CAPTION, vlm_call=boom)
    assert result["available"] is False
    assert "VLM" in result["reason"]


def test_read_chart_wrong_number_downgraded():
    fake = _FakeVLM("阴影表示 80% 置信区间")
    result = vision.read_chart(FUJITA, 8, 4, "图 4 的阴影表示什么？",
                               caption=FIG4_CAPTION, vlm_call=fake)
    assert result["available"] is True
    assert result["downgraded"] is True
    assert result["verification"]["unverified_numbers"] == [80.0]


# ---------- chart_read 工具 ----------

def _mock_caption_collection():
    return _FakeCollection(
        [FIG4_CAPTION],
        [{"source": "2023-fujita-brain-volume-aging.pdf", "page": 8}])


def test_chart_read_tool_mock(monkeypatch):
    fake = _FakeVLM(F02_READING)
    monkeypatch.setattr("app.tools._open_collection", _mock_caption_collection)
    monkeypatch.setattr("app.config.PAPERS_DIR", str(ROOT / "data" / "papers"))
    monkeypatch.setattr("app.vision._default_vlm_call", fake)
    result = tools.call_tool("chart_read", paper="fujita", figure_no=4,
                             question="Fujita 2023 的图 4 展示的是什么？")
    assert result["success"] is True
    assert result["data"]["available"] is True
    assert result["data"]["reading"] == F02_READING
    assert result["data"]["source"] == "2023-fujita-brain-volume-aging.pdf"
    assert result["data"]["page"] == 8
    assert len(fake.calls) == 1


def test_chart_read_tool_no_figure(monkeypatch):
    fake = _FakeVLM(F02_READING)
    monkeypatch.setattr("app.tools._open_collection",
                        lambda: _FakeCollection([], []))
    monkeypatch.setattr("app.vision._default_vlm_call", fake)
    result = tools.call_tool("chart_read", paper="fujita", figure_no=99,
                             question="图 99 展示什么？")
    assert result["success"] is True
    assert result["data"]["available"] is False
    assert len(fake.calls) == 0  # 没图就不调 VLM


def test_chart_read_tool_missing_image():
    result = tools.call_tool("chart_read", image_path="不存在.png",
                             question="图里峰值是多少？")
    assert result["success"] is True
    assert result["data"]["available"] is False


def test_chart_read_tool_no_params():
    result = tools.call_tool("chart_read")
    assert result["success"] is True
    assert result["data"]["available"] is False


# ---------- VLM 单独 key（2026-11-28：LLM 走 ModelScope、VLM 走 DashScope） ----------

def test_get_vlm_api_key_prefers_vlm_key(monkeypatch):
    # VLM_API_KEY 填了就用它，不受 QWEN_API_KEY / DEEPSEEK_API_KEY 影响。
    # 不然直接填 QWEN_API_KEY 会让 get_api_key() 优先返回 DashScope 的 key，
    # LLM 调 ModelScope 时会被拒。
    from app import config
    monkeypatch.setattr(config, "VLM_API_KEY", "vlm-key")
    monkeypatch.setattr(config, "QWEN_API_KEY", "dashscope-key")
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "modelscope-key")
    assert config.get_vlm_api_key() == "vlm-key"
    # LLM 用的 key 不变，还是 ModelScope 那个
    assert config.get_api_key() == "dashscope-key"


def test_get_vlm_api_key_falls_back_to_llm_key(monkeypatch):
    # VLM_API_KEY 没填时跟 LLM 用同一个 key（老行为不变）
    from app import config
    monkeypatch.setattr(config, "VLM_API_KEY", "")
    monkeypatch.setattr(config, "QWEN_API_KEY", "")
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "modelscope-key")
    assert config.get_vlm_api_key() == "modelscope-key"
