import os
from dotenv import load_dotenv

load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
EMBED_MODEL = os.getenv("EMBED_MODEL", "./models/bge-m3")
# 重排模型：bge-reranker-base，和 bge-m3 同一个 BGE 家族；本地跑，不调 API。
# 用 ModelScope 下：modelscope download --model BAAI/bge-reranker-base --local_dir ./models/bge-reranker-base
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "./models/bge-reranker-base")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
PAPERS_DIR = os.getenv("PAPERS_DIR", "./data/papers")
# 会话持久化：LangGraph SQLite checkpointer 的数据库文件（thread_id 的对话存在这里）；
# 删掉它就是清空所有会话，问答本身不受影响（.gitignore 已排除，不进仓库）
CHECKPOINTS_DB = os.getenv("CHECKPOINTS_DB", "./data/checkpoints.db")

# 至少要有一个 key 才能跑问答，检索可以先不用 key
def get_api_key():
    return QWEN_API_KEY or DEEPSEEK_API_KEY

# LangSmith 追踪（可选）：只在 LANGSMITH_TRACING=true 且给了 key 时启用；
# 不填就完全不追踪，问答不受影响。key 只在本地 .env，不进仓库。
# 去 https://smith.langchain.com 免费注册，拿到 key 填进 .env。
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "medevidence")

# langsmith 的开关只认小写 "true"（LANGSMITH_TRACING=true），写成 True/1
# 图的追踪就悄悄不工作——这里统一归一化成小写，避免这个坑。
if LANGSMITH_TRACING.strip().lower() in ("true", "1", "yes"):
    os.environ["LANGSMITH_TRACING"] = "true"
    LANGSMITH_TRACING = "true"

def tracing_enabled():
    """LangSmith 追踪是否开启。只认环境变量；没开时应用和原来一模一样，
    不调 langsmith、不要求 key。开着时 LangGraph 图的执行过程（检索→回答
    →核查）和 LLM 调用都会被记成 trace（需要装 langsmith 包，见
    requirements.txt）。"""
    return LANGSMITH_TRACING == "true"
