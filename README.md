# MedEvidence Agent — 面向专业文档的带引用 RAG Agent

首个垂类：医学文献。架构通用，可换成法律、金融等任意专业文档。

## 解决什么问题

普通 LLM 回答医学问题容易编造，且没有来源。MedEvidence 强制每条回答标注来源页码（`[来源：论文名 p.页码]`），核查员逐条校验引用是否真实存在于检索结果中，不通过就重答（最多 2 次），再不过就明确拒答。没有依据的问题，宁可拒答也不猜。

## 当前状态（2026-11-26）

- 文献库：26 篇纵向常模方向的论文，596 页，4456 个片段（含 147 条图注 + 28 条表注）。
- 最新一轮评测（28 道题：原 20 道 + 图表 8 道，Qwen/Qwen3-8B）：
  - 原 20 题：可回答已答 **13/15**，引用准确率 **92.3%（12/13）**，应拒答 5/5，编造 0 次。
  - 图表题：全对 **5/6**（F01–F05），引用 100%（5/5）；应拒答 2/2，编造 0 次。
  - 总计应拒答 7/7，编造 0 次；整体引用准确率 94.4%（17/18）。
- 检索：向量（bge-m3）+ BM25 混合，各取前 5；重排器（bge-reranker-base）已停用——实测发现它会把正确片段排掉。

## 架构：输入 - 处理 - 输出

输入：用户问题 + PDF 论文库
处理（LangGraph 三角色）：
1. 检索员 Retriever：混合检索——向量（bge-m3）+ 关键词（BM25）各取前 5。入库时给每个片段附加上下文（来源论文、页码、章节），与正文一起计算 embedding，检索相关性更好；回答时剥离上下文，引用不受影响。
2. 回答员 Answerer：Qwen / DeepSeek API 生成带引用的回答；多轮对话带最近 6 轮历史（4000 字上限），追问拼上文去检索。
3. 核查员 Verifier：检查每个引用是否在真实检索片段里；引用错了或没给引用就让回答员重答（最多 2 次），再不过就拒答。
输出：回答 + 引用列表 + 核查是否通过。

其他机制：
- 流式输出：`POST /ask/stream`（SSE），先核查、再把最终答案逐块推给前端。
- 会话持久化：LangGraph SQLite checkpointer，`thread_id` 标识一个会话，服务重启后还能接着聊（数据文件 `data/checkpoints.db`）。
- LangSmith 追踪：只认环境变量（见下），开了能在后台看到每次运行的完整 trace。
- 提示词管理：prompt 在 `prompts/` 目录按版本管，改 prompt 只改文件，不动代码。
- MCP PubMed 搜索：本地没证据、或用户明确问新文献时，调 PubMed 搜外部文献（引用格式 `PubMed PMID:xxxxxxxx`），只搜不下，挂了退回本地-only（见下）。

## 技术栈

Python + FastAPI + LangGraph + Chroma + SQLite + Streamlit + Docker
大模型：Qwen / DeepSeek API（通过 `.env` 配置）；embedding：bge-m3（本地）；BM25：rank-bm25

## 快速开始（本地）

```bash
conda create -n medevidence python=3.11 -y
conda activate medevidence
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple  # 国内用清华源，快很多
cp .env.example .env  # 填入 QWEN_API_KEY 或 DEEPSEEK_API_KEY
```

模型用 ModelScope 下载到本地（bge-m3 是多语言模型，中英文都支持）：
```bash
modelscope download --model BAAI/bge-m3 --local_dir ./models/bge-m3
# 重排模型已停用，不用下；想启用再下：
# modelscope download --model BAAI/bge-reranker-base --local_dir ./models/bge-reranker-base
```

入库 + 启动：
```bash
python -m app.ingest --papers data/papers  # 首次运行会建库（4456 个片段），耐心等几分钟，只需一次
uvicorn app.main:app --reload --port 8000
streamlit run app/streamlit_app.py  # 另开一个终端，演示页面
```

换模型后必须删掉 `data/chroma` 重新跑 ingest（向量维度变了，旧库不能用）。

## Docker

模型和数据不进镜像，运行时挂载；API key 只在本地 `.env`，不进镜像：

```powershell
# PowerShell，在项目根目录
docker build -t medevidence .
docker run -d --name medevidence -p 8000:8000 `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/data:/app/data `
  --env-file .env medevidence
docker logs medevidence  # 看启动日志
curl.exe http://localhost:8000/health  # 应返回 {"status":"ok"}
```

停掉 / 重启：
```powershell
docker stop medevidence
docker start medevidence
docker rm -f medevidence  # 彻底删掉容器（数据在挂载的 ./data 里，不会丢）
```

## 接口

- `GET /health` → `{"status": "ok"}`
- `POST /ask` → `{"answer": "...", "verified": true/false, "citations": [{"source": "...", "page": 3}]}`，请求体 `{"query": "...", "history": [...], "thread_id": "..."}`（`history` 和 `thread_id` 都可选）。
  - `history`：多轮对话历史 `[{"role": "user"/"assistant", "content": "..."}]`，不传就是单轮。
  - `thread_id`：会话 id，传了就用 checkpointer 持久化；下次只传 `query` + `thread_id`（`history` 不传）就能接着聊，history 从库里恢复。
- `POST /ask/stream`：流式问答（SSE），请求体同 `/ask`。先跑完整流程（检索→回答→核查），核查通过后再把最终答案逐块推回来。事件（每行 `data: {json}`）：
  - `{"type": "token", "text": "..."}`：答案的一小块
  - `{"type": "done", "verified": true/false, "citations": [...]}`：结束
  - `{"type": "error", "message": "..."}`：出错

之所以不直接流 LLM 的原始输出：核查没通过要重答，流到一半的答案收不回来。首字延迟和 `/ask` 一样，好处是长答案不用等整段写完才看到。

## LangSmith 追踪（可选）

去 https://smith.langchain.com 免费注册，拿到 API key 后在本地 `.env` 里填：
```bash
LANGSMITH_TRACING=true  # 必须是小写 true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=medevidence
```
之后每次问答都会在 LangSmith 后台留一条 trace：检索到了什么、prompt 是什么、哪步慢、哪步错。不填就完全不追踪，问答不受影响；开了追踪但 key 错了/没填，trace 发不出去（后台报 warning），问答照常跑。

## MCP PubMed 搜索

本地库只有 26 篇论文，库外的问题原来只能拒答。现在检索员后面多了一条 PubMed 分支：

- **什么时候调**：本地检索不到证据（原来会拒答的那条路），先去 PubMed 找找再决定拒不拒答；用户明确问“最新 / 最近 / 新文献”时，直接调 PubMed（按发表日期倒序，最新在前）。本地能答的，PubMed 不插手。
- **引用格式**：PubMed 的引用是 `PubMed PMID:xxxxxxxx`，和本地的 `[来源：论文名 p.页码]` 严格区分，答案里一眼看得出这是外部文献。核查员只认本次 PubMed 返回的 PMID，编一个没返回过的 PMID 会被拦下来（走重试/拒答）。
- **只搜不下**：搜到的文章只用于本次回答，不写入本地库。下次问同样的问题还会再搜一次。
- **异常时退回本地**：MCP 服务端起不来、NCBI 调不通、超时——一律退回本地-only，问答照常。

配置（本地 `.env`，见 `.env.example`）：
```bash
NCBI_EMAIL=you@example.com  # NCBI 要求调用时带邮箱，填你自己的；不填也能搜
# MCP_PUBMED_ENABLED=true   # 设为 false 可以彻底关掉 PubMed
```

服务端是 `@cyanheads/pubmed-mcp-server` 2.12.4（npx 拉起，stdio 方式），需要本机装 Node.js。实现上有三点要注意：
- 用 Python 的 `mcp` SDK 直连服务端（`langchain-mcp-adapters` 要求 langchain-core ≥ 0.3，与本项目 langgraph 0.2.x / langchain-core 0.2.x 不兼容）。
- 搜索分两步：`pubmed_search_articles` 只返回 PMID 列表，再调 `pubmed_fetch_articles` 拿标题、作者、期刊、年份、摘要。所以一次 PubMed 查询是两次工具调用。
- 中文问题先翻成英文再搜（PubMed 只认英文），翻译只在 PubMed 路径上进行。

评测时 PubMed 默认关闭（`eval/run_eval.py` 传 `use_pubmed=False`），28 道题只测本地检索。

## 提示词管理

prompt 不写在代码里，放在 `prompts/` 目录按版本管：

- `prompts/answerer/v1.txt`：回答员的 prompt 模板（`{history_text}`、`{context}`、`{query}` 是占位符）。
- `prompts/retry/v1.txt`：核查失败后重答时的提醒。

改 prompt 就改这里的文件，不用动 `app/` 里的代码；改完重启服务生效。想试新版：新建一个版本文件（比如 `v2.txt`），再把调用处（`app/rag.py`、`app/agent.py`）的版本号改成 `v2`——旧版文件留着，随时能切回去。

## 评测

题库 28 道：原 20 道（Q01–Q15 可回答、R01–R05 应拒答）+ 图表 8 道（F01–F06 可回答、FR01–FR02 应拒答）。预期答案/来源都用原文核对过。

```bash
python eval/run_eval.py        # 跑 28 题，结果写进 eval/results.json + eval/results.md
python eval/run_eval_auto.py   # 一键：跑评测 + 自动和上一轮对比 + 按通过标准判通过/不通过
```

`run_eval_auto.py` 的流程：先把现有的 `eval/results.json` 备份成 `eval/results_prev.json`，再跑评测，逐题对比（哪些退步、哪些进步），按通过标准判定——原 20 题可回答已答 >= 12/15，且原 20 题没有新的严重退步→通过（退出码 0），否则不通过（退出码 1）。对比报告写进 `eval/results_compare.md`。

注意：完整评测在本地运行（需要 bge-m3 和 DeepSeek key）。

指标口径（`eval/run_eval.py` 开头有完整说明）：
- 已答/拒答：回答的第一句里是否出现“无法回答”。
- 引用准确率 = 引用命中的可回答题数 / 已回答的可回答题数（引用是检索到的片段列表，度量的是检索是否召回了预期来源）。
- 编造次数 = 应拒答题被回答的次数，目标 0 次。
- 答对率需人工核对（对照 `eval/results.md` 逐题表里的回答原文与预期答案），脚本不自动判。

## 测试

```bash
pytest -q  # 全量，不需要真 LLM、不需要 Chroma、不联网（假 LLM + 假检索）
```

## 目录

- `app/main.py`：FastAPI 入口（`/health`、`/ask`、`/ask/stream`）
- `app/agent.py`：LangGraph 三角色流程（检索→回答→核查→重试）+ checkpointer + PubMed 分支
- `app/mcp_tools.py`：MCP PubMed 搜索（mcp SDK 直连服务端，搜 + 取元数据，只搜不下）
- `app/rag.py`：混合检索（向量 + BM25）+ 引用格式化 + prompt 拼装
- `app/ingest.py`：PDF 解析、按句子切分、图注/表注提取、入库
- `app/config.py`：配置（`.env`）
- `app/prompts.py`：提示词 loader
- `app/streamlit_app.py`：演示页面（调 `localhost:8000`，流式显示）
- `prompts/`：提示词文件（按版本）
- `eval/test_questions.md`：28 道评测题；`eval/run_eval.py`：评测脚本；`eval/run_eval_auto.py`：评测自动化
- `eval/check_retrieval.py`：检索级验证（不需要 LLM/Chroma/embedding）
- `data/papers/`：放公开论文 PDF，不要放 ADNI/NACC 受限数据
