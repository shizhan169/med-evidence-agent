from fastapi import FastAPI
from pydantic import BaseModel
from .agent import run

app = FastAPI(title="MedEvidence Agent")

class AskRequest(BaseModel):
    query: str
    # 可选的多轮对话历史：[{"role": "user"/"assistant", "content": "..."}]，
    # 不传就是单轮问答；传了就按最近 6 轮截断后用
    history: list = []

class AskResponse(BaseModel):
    answer: str
    verified: bool
    citations: list = []

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = run(req.query, history=req.history)
    return AskResponse(answer=result["answer"], verified=result["verified"], citations=result["citations"])
