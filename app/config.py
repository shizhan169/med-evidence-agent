import os
from dotenv import load_dotenv

load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
EMBED_MODEL = os.getenv("EMBED_MODEL", "./models/bge-m3")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
PAPERS_DIR = os.getenv("PAPERS_DIR", "./data/papers")

# 至少要有一个 key 才能跑问答，检索可以先不用 key
def get_api_key():
    return QWEN_API_KEY or DEEPSEEK_API_KEY
