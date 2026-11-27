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


# VLM 读图（第二阶段 2.3）：Qwen-VL，走 OpenAI 兼容接口。
# 默认走 DashScope 的 qwen-vl-plus；如果 LLM_BASE_URL 是 ModelScope，
# 把 VLM_MODEL 改成 ModelScope 上有的 VL 模型（比如 Qwen/Qwen2.5-VL-72B-Instruct）。
# VLM_BASE_URL 默认跟 LLM_BASE_URL 走；想让 VLM 单独走另一个服务再改它。
# VLM_API_KEY：VLM 单独的 key。不填时跟 LLM 用同一个 key（get_api_key）；
# LLM 和 VLM 走不同服务、key 不一样时才需要填——比如 LLM 走 ModelScope、
# VLM 走 DashScope：只填 VLM_API_KEY，不动 QWEN_API_KEY，不然 get_api_key()
# 会优先返回 DashScope 的 key，LLM 调 ModelScope 时会被拒。
VLM_MODEL = os.getenv("VLM_MODEL", "qwen-vl-plus")
VLM_BASE_URL = os.getenv("VLM_BASE_URL", LLM_BASE_URL)
VLM_API_KEY = os.getenv("VLM_API_KEY", "")


def get_vlm_api_key():
    """VLM 用的 key：优先 VLM_API_KEY，没填就跟 LLM 用同一个。"""
    return VLM_API_KEY or get_api_key()

# MCP PubMed 搜索（第 5 步，5a：只搜不下）。
# 服务端是 @cyanheads/pubmed-mcp-server（npx 拉起，stdio 方式），版本钉死在
# MCP_PUBMED_ARGS 里（2026-11-26 实测 2.12.4）。需要本机装 Node.js（npx 能跑）；
# NCBI 要求调用时带邮箱（E-utilities 规范），填进本地 .env 的 NCBI_EMAIL，
# 不填也能用，但 NCBI 建议填。MCP_PUBMED_ENABLED=false 可以彻底关掉 PubMed
# （只用本地库）。默认值写死在代码里，不放秘密。
def _mcp_enabled():
    return os.getenv("MCP_PUBMED_ENABLED", "true").strip().lower() in ("true", "1", "yes")


MCP_PUBMED_ENABLED = _mcp_enabled()
MCP_PUBMED_COMMAND = os.getenv("MCP_PUBMED_COMMAND", "npx")
MCP_PUBMED_ARGS = os.getenv("MCP_PUBMED_ARGS", "-y @cyanheads/pubmed-mcp-server@2.12.4")
# 工具名按服务端 2.12.4 实测：搜文章（只返回 PMID）+ 取文章（拿元数据），
# 方案稿写的 search_pubmed 在这个版本里不存在
MCP_PUBMED_TOOL_SEARCH = os.getenv("MCP_PUBMED_TOOL_SEARCH", "pubmed_search_articles")
MCP_PUBMED_TOOL_FETCH = os.getenv("MCP_PUBMED_TOOL_FETCH", "pubmed_fetch_articles")
try:
    MCP_PUBMED_TIMEOUT = int(os.getenv("MCP_PUBMED_TIMEOUT", "120"))
except ValueError:
    MCP_PUBMED_TIMEOUT = 120
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
# 一次 PubMed 查询最多返回几篇（搜 + 取都是这个数）；列文献的问题 5 篇太少，2026-11-26 调到 10
PUBMED_TOP_K = 10
# PubMed 先取多少候选再按相关性重排（2026-11-26）：问“最新文献”时按日期倒序
# 只取 10 篇会把真正相关的旧论文顶下去；先取 30 篇候选（按相关性排的池子）
# 再重排，取前 10 篇给回答员。VM 真跑：取 30 篇约 8 秒，池子里有 CentileBrain
#（2024）和 Brain Morphometry Normative Modeling（2024）这类直接相关的
PUBMED_CANDIDATE_K = 30
