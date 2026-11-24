from fastapi import FastAPI
from pydantic import BaseModel
from .agent import run

app = FastAPI(title="MedEvidence Agent")

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    verified: bool
    citations: list = []

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = run(req.query)
    return AskResponse(answer=result["answer"], verified=result["verified"], citations=result["citations"])
