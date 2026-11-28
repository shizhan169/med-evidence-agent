import json
import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from .agent import run

logger = logging.getLogger(__name__)

app = FastAPI(title="MedEvidence Agent")

# 请求追踪：每个请求一个 request_id，全链路日志、响应头、SSE 事件都用
# 同一个 ID，一次问答的日志能按 ID 串起来。
# 客户端传了 X-Request-ID 就沿用（方便跨服务追踪），没传就生成一个。
REQUEST_ID_HEADER = "X-Request-ID"


def _new_request_id() -> str:
    return str(uuid.uuid4())


class RequestIdMiddleware(BaseHTTPMiddleware):
    """生成 request_id 并挂到 request.state 上，响应头里带回；
    记录请求开始和结束（带耗时），一次问答的日志都带同一个 ID。
    注意：只做追踪和日志，不改业务逻辑、不动 agent.py。"""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or _new_request_id()
        request.state.request_id = request_id
        start = time.monotonic()
        logger.info("[%s] %s %s", request_id, request.method, request.url.path)
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info("[%s] %s %s -> %s %.1fms", request_id, request.method,
                    request.url.path, response.status_code, elapsed_ms)
        return response


app.add_middleware(RequestIdMiddleware)


def _request_id(request: Request) -> str:
    """从 request.state 取 request_id；中间件没跑到时（比如直接调函数）
    生成一个临时的，保证日志和错误体里总有 ID。"""
    return getattr(request.state, "request_id", None) or _new_request_id()


@app.exception_handler(Exception)
async def _structured_error_handler(request: Request, exc: Exception):
    """结构化异常处理：业务异常不再返回笼统的 500，而是明确的错误信息。
    HTTPException（404/422 校验错等）保持 FastAPI 默认格式不动，
    只把未处理的异常包成 {"error": {"code", "message", "request_id"}}。
    """
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    request_id = _request_id(request)
    logger.exception("[%s] unhandled error: %s", request_id, exc)
    message = str(exc) or "服务器内部错误"
    return JSONResponse(status_code=500, content={
        "error": {"code": "internal_error", "message": message, "request_id": request_id},
    }, headers={REQUEST_ID_HEADER: request_id})

class AskRequest(BaseModel):
    query: str
    # 可选的多轮对话历史：[{"role": "user"/"assistant", "content": "..."}]。
    # 不传（None）且给了 thread_id 时，history 从 checkpointer 里恢复；
    # 传了（哪怕是 []）就用传来的；不传 thread_id 时就是原来的单轮问答
    history: list | None = None
    # 可选的会话 id：传了就用 SQLite checkpointer 把这轮对话存下来，
    # 下次只传 query + thread_id（history 不传）就能接着聊；
    # 不传就是原来的无状态问答
    thread_id: str = ""

class AskResponse(BaseModel):
    answer: str
    verified: bool  # 兼容字段：只有 answer_status == "verified" 时为 True
    answer_status: str = "refused"  # 最终状态：verified / partial / refused / error
    citations: list = []

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    request_id = _request_id(request)
    logger.info("[%s] agent run start query=%s", request_id, (req.query or "")[:80])
    result = run(req.query, history=req.history, thread_id=req.thread_id or None)
    logger.info("[%s] agent run done answer_status=%s", request_id,
                result.get("answer_status", "refused"))
    return AskResponse(answer=result["answer"], verified=result["verified"],
                       answer_status=result.get("answer_status", "refused"),
                       citations=result["citations"])


def _split_stream_chunks(text: str, max_chars: int = 24) -> list:
    """把一段文字切成小块供 SSE 逐块推送：每块尽量不超过 max_chars 字，
    优先在空格或句末标点（。！？；，、：\n,.!?;:）处断，不断词；超长无标点
    的段落硬切（每块不超过 2*max_chars）。空文本返回 []。
    切完拼回去和原文一模一样（无损），测试里会断言这一点。"""
    if not text:
        return []
    boundaries = set(" 。！？；，、：\n,.!?;:")
    chunks, buf = [], ""
    for ch in text:
        buf += ch
        if len(buf) >= max_chars and ch in boundaries:
            chunks.append(buf)
            buf = ""
        elif len(buf) >= max_chars * 2:
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


@app.post("/ask/stream")
def ask_stream(req: AskRequest, request: Request):
    """流式问答（SSE）：先跑完整流程（检索→回答→核查，核查不过会重答），
    核查通过后再把最终答案逐块推给前端。
    之所以不直接流 LLM 的原始输出：核查没通过要重答，流到一半的答案
    收不回来——对要求引用必须核查通过的系统，先核查、再流式输出才是
    正确的取舍。首字延迟和 /ask 一样，好处是长答案不用等整段写完才看到。
    事件格式（每行 `data: {json}`，空行分隔）：
      {"type": "token", "text": "..."}  答案的一小块
      {"type": "done", "verified": true/false, "answer_status": "verified", "citations": [...], "request_id": "..."}  结束
      {"type": "error", "message": "...", "request_id": "..."}  出错（run 抛异常时）
    块与块之间不加人为延迟。
    """
    request_id = _request_id(request)
    logger.info("[%s] agent run start query=%s", request_id, (req.query or "")[:80])

    def events():
        try:
            result = run(req.query, history=req.history, thread_id=req.thread_id or None)
        except Exception as e:
            logger.exception("[%s] agent run error: %s", request_id, e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'request_id': request_id}, ensure_ascii=False)}\n\n"
            return
        logger.info("[%s] agent run done answer_status=%s", request_id,
                    result.get("answer_status", "refused"))
        for chunk in _split_stream_chunks(result["answer"]):
            yield f"data: {json.dumps({'type': 'token', 'text': chunk}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'verified': result['verified'], 'answer_status': result.get('answer_status', 'refused'), 'citations': result['citations'], 'request_id': request_id}, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
