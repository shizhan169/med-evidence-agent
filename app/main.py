import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from .agent import run

app = FastAPI(title="MedEvidence Agent")

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
    verified: bool
    citations: list = []

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = run(req.query, history=req.history, thread_id=req.thread_id or None)
    return AskResponse(answer=result["answer"], verified=result["verified"], citations=result["citations"])


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
def ask_stream(req: AskRequest):
    """流式问答（SSE）：先跑完整流程（检索→回答→核查，核查不过会重答），
    核查通过后再把最终答案逐块推给前端。
    之所以不直接流 LLM 的原始输出：核查没通过要重答，流到一半的答案
    收不回来——对要求引用必须核查通过的系统，先核查、再流式输出才是
    正确的取舍。首字延迟和 /ask 一样，好处是长答案不用等整段写完才看到。
    事件格式（每行 `data: {json}`，空行分隔）：
      {"type": "token", "text": "..."}  答案的一小块
      {"type": "done", "verified": true/false, "citations": [...]}  结束
      {"type": "error", "message": "..."}  出错（run 抛异常时）
    """
    def events():
        try:
            result = run(req.query, history=req.history, thread_id=req.thread_id or None)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            return
        for chunk in _split_stream_chunks(result["answer"]):
            yield f"data: {json.dumps({'type': 'token', 'text': chunk}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'verified': result['verified'], 'citations': result['citations']}, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
