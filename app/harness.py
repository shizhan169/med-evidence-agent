"""Harness（框架 v2 第三阶段 3.1 + 3.4）：request_id、超时、Token 预算、
统一日志、异常处理、工具调用参数校验。

只依赖标准库 + app.config，不 import agent / tools / rag，避免循环引用。
main.py / agent.py / tools.py 只做最小接线：生成 request_id、调
run_with_timeout、_call_llm 里记耗时和 Token、answerer 里按预算截断、
call_tool 里调参数校验。

设计要点：
- request_id + TokenBudget 用 ContextVar 跟着一次问答走：run() 里 set，
  _call_llm 里读，不用把参数一路传进每个节点。
- Token 估算是粗估（中文 1 字符 1 token、英文约 4 字符 1 token），
  够用来截断和告警，不够用来精确计费。
- 预算分两层：单条 prompt 上限（MAX_PROMPT_TOKENS，answerer 里丢片段
  截断）和单次问答累计上限（MAX_TOKENS_PER_RUN，超了记日志告警、
  降级继续，不直接掐断问答——中途掐断比超预算更糟）。
"""
import concurrent.futures
import logging
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .config import PAPERS_DIR

logger = logging.getLogger("medevidence.harness")

# 单条 prompt 的 Token 上限（默认 8000）：answerer 的 prompt 超了就丢
# 最末尾的片段再重建，至少留 1 条；还超就记日志、降级继续。
try:
    MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "8000"))
except ValueError:
    MAX_PROMPT_TOKENS = 8000

# 单次问答累计 Token 上限（默认 30000）：所有 LLM 调用（路由、规划、
# 执行器选工具、回答、核查、翻译、摘要）的 prompt + 回答都计入；
# 超了记日志告警，不掐断。
try:
    MAX_TOKENS_PER_RUN = int(os.getenv("MAX_TOKENS_PER_RUN", "30000"))
except ValueError:
    MAX_TOKENS_PER_RUN = 30000

# 单次问答整体超时（默认 300 秒）：评测里最慢的一题约 172 秒，
# 300 秒留有余量；LLM 单次调用超时默认 120 秒。
try:
    RUN_TIMEOUT_SECONDS = int(os.getenv("RUN_TIMEOUT_SECONDS", "300"))
except ValueError:
    RUN_TIMEOUT_SECONDS = 300
try:
    LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
except ValueError:
    LLM_TIMEOUT_SECONDS = 120


class HarnessError(Exception):
    """Harness 层的异常基类。"""


class HarnessTimeout(HarnessError):
    """执行超时（run_with_timeout 抛）。"""


def new_request_id() -> str:
    """生成一次问答的 request_id：uuid 前 12 位十六进制，日志里够短、
    碰撞概率可忽略。"""
    return uuid.uuid4().hex[:12]


_current_request_id = None
_current_budget = None
_current_node = None
try:
    from contextvars import ContextVar

    _current_request_id = ContextVar("medevidence_request_id", default=None)
    _current_budget = ContextVar("medevidence_token_budget", default=None)
    _current_node = ContextVar("medevidence_node", default=None)
except Exception:
    _current_request_id = None
    _current_budget = None
    _current_node = None


def set_request_context(request_id: str, budget=None):
    """把 request_id 和 TokenBudget 挂到当前上下文（run() 里调），
    返回 token，调用方在 finally 里用 reset_request_context 还原。"""
    if _current_request_id is None:
        return None
    t1 = _current_request_id.set(request_id)
    t2 = _current_budget.set(budget)
    return (t1, t2)


def reset_request_context(token) -> None:
    if token is None or _current_request_id is None:
        return
    _current_request_id.reset(token[0])
    _current_budget.reset(token[1])


def get_request_id():
    return _current_request_id.get() if _current_request_id is not None else None


def get_token_budget():
    return _current_budget.get() if _current_budget is not None else None


@contextmanager
def request_context(request_id=None, budget=None):
    """with 写法：进上下文时 set，退出时还原；request_id 不给就新生成一个。"""
    token = set_request_context(request_id or new_request_id(), budget)
    try:
        yield get_request_id()
    finally:
        reset_request_context(token)


def get_node():
    """当前节点名（router / planner / executor / retriever / answerer / memory），
    调用方节点用 node() 上下文设定；没设定时返回 None。"""
    return _current_node.get() if _current_node is not None else None


@contextmanager
def node(name: str):
    """with 写法：把当前节点名设成 name，退出时还原。
    _call_llm 记日志时从这里读节点名——不用改 _call_llm 的签名，
    现有测试里的 1 参数 fake 照样能用。"""
    if _current_node is None:
        yield
        return
    token = _current_node.set(name)
    try:
        yield
    finally:
        _current_node.reset(token)


class Timer:
    """计时器：创建时开始，elapsed_ms() 读毫秒数；也可用作 with。"""

    def __init__(self):
        self._start = time.monotonic()

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._start) * 1000)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def log_event(request_id, node: str, message: str, elapsed_ms=None, tokens=None) -> None:
    """统一日志格式，一行一个事件：
    [harness] request_id=xxx node=answerer <消息> elapsed_ms=123 tokens=456
    elapsed_ms / tokens 没给就不写。走 print，和仓库现有 "[router] ..."、
    "[executor] ..." 的风格一致；logger 也记一份，方便以后接日志采集。"""
    parts = [f"[harness] request_id={request_id or '-'}", f"node={node}", str(message)]
    if elapsed_ms is not None:
        parts.append(f"elapsed_ms={elapsed_ms}")
    if tokens is not None:
        parts.append(f"tokens={tokens}")
    line = " ".join(parts)
    print(line)
    logger.info(line)


def run_with_timeout(fn, timeout_seconds, *args, **kwargs):
    """在超时内跑 fn，超了抛 HarnessTimeout。
    timeout_seconds <= 0 或 None 时不限时、直接跑。
    注意：超时只停等结果，不杀线程——真正的 HTTP 调用由 OpenAI 客户端
    自己的 timeout 兜底（_call_llm 里传 LLM_TIMEOUT_SECONDS）。"""
    if not timeout_seconds or timeout_seconds <= 0:
        return fn(*args, **kwargs)
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as e:
            raise HarnessTimeout(f"执行超时（超过 {timeout_seconds} 秒）") from e
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def estimate_tokens(text: str) -> int:
    """估算 token 数（预算用，不是精确计费）：
    中文字符按 1 token；其余字符每 4 个算 1 token（向上取整）。
    英文约 4 字符/token、中文约 1 字符/token，是业内常用的粗估法。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    rest = len(text) - cjk
    return cjk + (rest + 3) // 4


class TokenBudget:
    """单次问答的 Token 预算：max_tokens 上限，used 已用。
    consume 超了返回 False（不扣），调用方决定截断还是降级。"""

    def __init__(self, max_tokens: int = MAX_TOKENS_PER_RUN):
        self.max_tokens = max_tokens
        self.used = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used)

    def fits(self, text: str) -> bool:
        return self.used + estimate_tokens(text) <= self.max_tokens

    def consume(self, text: str) -> bool:
        return self.consume_tokens(estimate_tokens(text))

    def consume_tokens(self, n: int) -> bool:
        if self.used + n > self.max_tokens:
            return False
        self.used += n
        return True


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """把文本截到 max_tokens 以内：保留开头，末尾加截断标记。
    回答员 prompt 的截断不走这个（走丢片段重建）；这个给超长用户输入等场景用。"""
    marker = "\n...[已截断]"
    if estimate_tokens(text) <= max_tokens:
        return text
    budget = max(0, max_tokens - estimate_tokens(marker))
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + marker


def error_message(exc: Exception) -> str:
    """给 API 层的友好错误文本：超时单独说明，其余只说失败、不泄露堆栈。"""
    if isinstance(exc, HarnessTimeout):
        return f"问答超时（超过 {RUN_TIMEOUT_SECONDS} 秒），请缩短问题或稍后重试。"
    return "问答失败，请稍后重试。"


# ---- 工具调用参数校验（3.4 注入防护的第二道关） ----
# call_tool 已有白名单 + 类型检查；这里再加值域检查，防 LLM 生成的
# 参数把问答带偏：top_k 过大拖慢检索、query 过长撑爆 prompt、
# image_path 越出数据目录读任意文件、paper 里带路径分隔符。
# 返回 None 表示通过，否则返回错误文本（调用方按失败分支走，不抛异常）。
_PARAM_LIMITS = {
    "local_search": {"query": ("str", 2000), "top_k": ("int", 1, 50)},
    "pubmed_search": {"query": ("str", 2000), "top_k": ("int", 1, 50)},
    "chart_locate": {"paper": ("str", 200), "figure_no": ("numstr", 4)},
    "chart_read": {
        "paper": ("str", 200),
        "figure_no": ("numstr", 4),
        "question": ("str", 2000),
        "image_path": ("image_path", None),
    },
}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _default_image_roots():
    return [PAPERS_DIR, "./data"]


def _validate_image_path(value, allowed_dirs=None):
    """image_path 只能是项目目录内的图片文件：
    空字符串表示没传（走 paper+figure_no 分支），放过；
    否则后缀必须是图片、解析后的路径必须落在项目目录（cwd）或
    allowed_dirs 内，防绝对路径和 "../" 越出项目目录读任意文件。
    注意：不存在的文件不拦——_chart_read 自己会返回 available=False。"""
    if not value:
        return None
    if not isinstance(value, str):
        return "参数 image_path 应为字符串"
    p = Path(value)
    if p.suffix.lower() not in _IMAGE_EXTS:
        return "参数 image_path 只接受图片文件"
    roots = [Path.cwd()]
    roots += [Path(r) for r in (allowed_dirs if allowed_dirs is not None else _default_image_roots())]
    try:
        resolved = p.resolve()
    except Exception:
        return f"参数 image_path 无法解析：{value}"
    for root in roots:
        try:
            if resolved.is_relative_to(root.resolve()):
                return None
        except Exception:
            continue
    return "参数 image_path 必须在项目目录内"


def validate_tool_params(name: str, params: dict, allowed_dirs=None):
    """校验一次工具调用的参数值（call_tool 里调）。
    不在 _PARAM_LIMITS 里的工具名返回错误；不在表里的参数不管
    （白名单由 call_tool 拦）。"""
    if name not in _PARAM_LIMITS:
        return f"未知工具：{name}"
    for pname, value in (params or {}).items():
        rule = _PARAM_LIMITS[name].get(pname)
        if rule is None:
            continue
        kind = rule[0]
        if kind == "str":
            if not isinstance(value, str):
                return f"参数 {pname} 应为字符串"
            if len(value) > rule[1]:
                return f"参数 {pname} 长度超过 {rule[1]} 字符"
        elif kind == "int":
            if (not isinstance(value, int)) or isinstance(value, bool):
                return f"参数 {pname} 应为整数"
            if not (rule[1] <= value <= rule[2]):
                return f"参数 {pname} 应为 {rule[1]}~{rule[2]} 的整数"
        elif kind == "numstr":
            s = str(value)
            if s == "":
                continue  # 空表示没传（走默认分支），不拦
            m = re.search(r"\d+", s)
            if not m or len(m.group(0)) > rule[1]:
                return f"参数 {pname} 应包含 {rule[1]} 位以内的数字"
        elif kind == "image_path":
            err = _validate_image_path(value, allowed_dirs)
            if err:
                return err
    paper = (params or {}).get("paper")
    if isinstance(paper, str) and ("/" in paper or "\\" in paper or ".." in paper):
        return "参数 paper 不能包含路径分隔符"
    return None
