# MedEvidence Agent — 面向专业文档的带引用 RAG Agent

首个垂类：医学文献。架构通用，可换成法律、金融等任意专业文档。

## 解决什么问题

普通 LLM 回答医学问题容易编造，且没有来源。MedEvidence 强制每条回答标注来源页码（`[来源：论文名 p.页码]`），核查员逐条校验引用是否真实存在于检索结果中，不通过就重答（最多 2 次），再不过就明确拒答；证据只够答一部分时给部分回答，并标出没答的部分。没有依据的问题，宁可拒答也不猜。

## 当前状态（2026-11-28）

- 文献库：30 篇纵向常模方向的论文，5085 个片段。开发集 25 篇 + 冻结测试集 5 篇（Verdi 2024、Dimitrova 2021、Bayer 2022、Wolfers 2018、Parker 2025）。
- 评测题库：40 道开发题（Q01–Q23 可回答 23 道、R01–R07 应拒答 7 道、F01–F08 + FR01–FR02 图表 10 道）+ 多轮追问 MT01（两轮），共 42 条记录。冻结测试题 T01–T10（每篇测试集论文 2 道）已冻结，只在阶段验收时跑一次，开发时不跑不调。
- 最新一轮评测（第 13 轮，2026-11-28，42 题，Qwen，约 18 分钟）：
  - 可回答已答 **32/33**，应拒答 8/9，编造 1 次（R02：路由器误判成复杂任务，答案本身没写假内容，是评测的计数口径）；
  - 原题：答出 22/23，引用 86.4%（19/22）；图表题：答出 **8/8**，引用 **100%**（8/8），应拒答 2/2；
  - 整体引用 90.6%（29/32）；路由 41/42；复杂任务完成度 2/2；VLM 图表题已答 8/10，引用 100%，走 vision 路由 10/10；多轮 MT01 两轮全对。
  - 引用率比第 12 轮（95.5%）低，主因是评测检查变严（多来源引用）加上 Q17/Q19 的 LLM 措辞漂移，按停表规则不追个别题。
- 检索：向量（bge-m3）+ BM25 混合，RRF 融合；重排器已停用——实测发现它会把正确片段排掉。
- VLM 读图已打通：Fujita 图 4 能真读图（不是只读图注），图表题 8/8，F06 修好。

## 架构：输入 - 处理 - 输出

输入：用户问题 + PDF 论文库

处理（LangGraph 编排，节点分工）：

```
用户问题
  → 路由器 Router：输出结构化 flag（task_type / requires_planning /
      requires_external_search / requires_vision），规则明确时直接分类，
      否则调 LLM；没有“其余 → 单篇”兜底
  → 按路由分流（优先级：规划 → 读图 → 外部搜索 → 本地检索）：
      需要规划 → 规划器 Planner：拆成最多 5 个结构化子任务
                → 执行器 Executor：每个子任务由 LLM 选工具执行，
                  总共最多 8 步工具调用，失败子任务最多重试 1 次
      需要读图 → 读图节点 chart_read：VLM 读图，失败退回检索员
      需要外部搜索 → PubMed 节点：搜 PubMed 外部文献
      否则 → 检索员 Retriever：混合检索
  → 回答员 Answerer：生成带引用的回答；规划任务要求每个子任务都覆盖、
      每个至少引用一次它的证据
  → 核查员 Verifier：校验引用真实性 + 子任务证据；
      引用问题回回答员重答（最多 2 次），证据不足回执行器只重跑失败的子任务；
      预算用完 → 部分回答（partial，只留已核实部分）或拒答
输出：回答 + 引用列表 + answer_status（verified / partial / refused / error）
```

工具（统一注册表，4 个；参数校验 + 白名单，失败返回 `{success: False}`，不抛异常）：

- `local_search`：本地 RAG 检索（包装 `app.rag.retrieve`）
- `pubmed_search`：PubMed 搜索（包装 `app.mcp_tools.search_pubmed`）
- `chart_locate`：按图号定位图注和页码（复用入库的图注片段）
- `chart_read`：VLM 读图（按图号定位页面，渲染成图片调 Qwen-VL）

检索细节：

- 混合检索：向量（bge-m3）+ BM25 各取 15 候选，RRF 融合（k=60）取前 10；重排器关闭。
- 中文问题先翻成英文再查向量（跨语言向量检索弱）；BM25 用原问题 token（论文名、数字、图号都在原问题里）。
- 图注片段带 `figureN` token，BM25 能按图号区分图 3 和图 4 的图注。
- 入库时给每个片段附加上下文行（来源论文、页码、章节），与正文一起算 embedding，回答时剥掉，引用不受影响。
- 问题里点了 2 篇以上论文时按篇取：每篇各取一份证据再合并，防止同时提到多篇的综述把各篇原文挤出前 10；只点一篇或没点名时走全局逻辑。
- 多轮对话：历史超限时摘要较早轮次、只留最近 2 轮完整；checkpointer 仍存完整历史，压缩只影响传给 LLM 的上下文。

VLM 读图：

- 模型 qwen3-vl-plus，走 DashScope OpenAI 兼容接口；`VLM_API_KEY` 单独配（没填就跟 LLM 用同一个 key；LLM 走 ModelScope、VLM 走 DashScope 时必须分开填，见下）。
- 流程：`chart_locate` 按图号定位图注和页码 → 渲染该页（图注页没图时往附近页找有图片的页）→ 按图注行裁剪 → VLM 读图 → 读出的数字和图注交叉核对（相对误差 1%），对不上就降级说明、不硬改数字。
- PubMed 的图没有原文图片时不强行调 VLM，返回部分答案。

其他机制：

- 流式输出：`POST /ask/stream`（SSE），先核查、再把最终答案逐块推给前端。
- 会话持久化：LangGraph SQLite checkpointer，`thread_id` 标识一个会话，服务重启后还能接着聊（数据文件 `data/checkpoints.db`）。
- LangSmith 追踪：只认环境变量（见下），开了能在后台看到每次运行的完整 trace。
- 提示词管理：prompt 在 `prompts/` 目录按版本管，改 prompt 只改文件，不动代码。
- MCP PubMed 搜索：本地没证据、或用户明确问新文献时，调 PubMed 搜外部文献（引用格式 `PubMed PMID:xxxxxxxx`），只搜不下，挂了退回本地-only（见下）。

## 技术栈

Python + FastAPI + LangGraph + Chroma + SQLite + Streamlit + Docker
大模型：Qwen / DeepSeek API（通过 `.env` 配置）；VLM：Qwen-VL（DashScope）；embedding：bge-m3（本地）；BM25：rank-bm25

## 快速开始（本地）

```bash
conda create -n medevidence python=3.11 -y
conda activate medevidence
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple  # 国内用清华源，快很多
cp .env.example .env  # 填入 QWEN_API_KEY 或 DEEPSEEK_API_KEY；VLM 另外填 VLM_API_KEY（见下）
```

模型用 ModelScope 下载到本地（bge-m3 是多语言模型，中英文都支持）：
```bash
modelscope download --model BAAI/bge-m3 --local_dir ./models/bge-m3
# 重排模型已停用，不用下；想启用再下：
# modelscope download --model BAAI/bge-reranker-base --local_dir ./models/bge-reranker-base
```

入库 + 启动：
```bash
python -m app.ingest --papers data/papers  # 首次运行会建库（30 篇，5085 个片段），耐心等几分钟，只需一次
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

注意：改 `.env` 后 `docker restart` 不重读，必须 `docker rm -f` + 原样 `docker run --env-file .env` 重建容器。Dockerfile 没有 Node.js，所以 PubMed 在容器里不工作，本地用 uvicorn 验证。

## 接口

- `GET /health` → `{"status": "ok"}`
- `POST /ask` → `{"answer": "...", "verified": true/false, "answer_status": "verified", "citations": [{"source": "...", "page": 3}]}`，请求体 `{"query": "...", "history": [...], "thread_id": "..."}`（`history` 和 `thread_id` 都可选）。
  - `verified` 是兼容字段：只有 `answer_status == "verified"` 时为 True；`answer_status` 是最终状态：`verified` / `partial` / `refused` / `error`（`error` 是 LLM 没调通，不是知识不足的拒答）。
  - `history`：多轮对话历史 `[{"role": "user"/"assistant", "content": "..."}]`，不传就是单轮。
  - `thread_id`：会话 id，传了就用 checkpointer 持久化；下次只传 `query` + `thread_id`（`history` 不传）就能接着聊，history 从库里恢复。
- `POST /ask/stream`：流式问答（SSE），请求体同 `/ask`。先跑完整流程（检索→回答→核查），核查通过后再把最终答案逐块推回来。事件（每行 `data: {json}`）：
  - `{"type": "token", "text": "..."}`：答案的一小块
  - `{"type": "done", "verified": true/false, "answer_status": "verified", "citations": [...]}`：结束
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

本地库只有 30 篇论文，库外的问题原来只能拒答。现在检索员后面多了一条 PubMed 分支：

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
- 中文问题先翻成英文再搜（PubMed 只认英文），翻译只在 PubMed 路径上进行；还会让 LLM 给核心概念的 2–3 个英文同义表达，用 OR 连接，扩大召回；先取 30 篇候选再按相关性重排取前 10。

评测时 PubMed 默认关闭（`eval/run_eval.py` 传 `use_pubmed=False`），42 条记录只测本地检索。

## VLM 读图配置

VLM 走 DashScope，不走 ModelScope（ModelScope 免费 API-Inference 对 VL 模型没有可用 provider）：

```bash
VLM_API_KEY=sk-...  # 阿里云百炼的 API key，和 LLM 的 key 分开填
VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1  # 北京地域
VLM_MODEL=qwen3-vl-plus
```

注意两点：百炼北京和新加坡地域的 key 不通用，key 和 base URL 必须同一个地域；`VLM_API_KEY` 不填时 VLM 跟 LLM 用同一个 key——LLM 走 ModelScope、VLM 走 DashScope 时必须分开填，不然 LLM 调 ModelScope 会被 DashScope 的 key 拒掉。

本地验证读图：
```bash
python -c "from app.tools import call_tool; print(call_tool('chart_read', paper='fujita', figure_no='4', question='test'))"
```
应返回 `available=True`，`page` 是图实际所在的页（图注页和图本身可能不在同一页，代码会自动往附近页找有图片的页）。

## 提示词管理

prompt 不写在代码里，放在 `prompts/` 目录按版本管：

- `prompts/answerer/v1.txt`：回答员的 prompt 模板（`{history_text}`、`{context}`、`{query}` 是占位符）。
- `prompts/retry/v1.txt`：核查失败后重答时的提醒。
- `prompts/router/v1.txt`：路由器分类 prompt（输出结构化 flag）。
- `prompts/planner/v1.txt`：规划器拆解 prompt。
- `prompts/executor/v1.txt`：执行器选工具 prompt。
- `prompts/chart_reader/v1.txt`：VLM 读图 prompt。
- `prompts/pubmed_answerer/v2.txt`：PubMed 路径回答员 prompt（分事实型 vs 文献罗列型）。
- `prompts/pubmed_retry/v1.txt`：PubMed 路径重答提醒。

改 prompt 就改这里的文件，不用动 `app/` 里的代码；改完重启服务生效。想试新版：新建一个版本文件（比如 `v2.txt`），再把调用处的版本号改成 `v2`——旧版文件留着，随时能切回去。

## 评测

题库 40 道开发题（`eval/test_questions.md`：Q01–Q23 可回答、R01–R07 应拒答、F01–F08 + FR01–FR02 图表）+ 多轮 MT01（`eval/run_eval.py` 里的 `MULTITURN_SEQUENCES`，两轮），共 42 条记录。冻结测试题 T01–T10 在 `eval/datasets/test_questions_frozen.md`，只在阶段验收时跑一次，开发时不跑不调。

6 个评测维度：回答、引用、拒答、路由正确性、复杂任务完成度、VLM 图表理解；另有多轮和耗时统计。引用状态分三档：`citation_status` 是 `full` / `partial` / `miss`（`citation_hit` 只在 full 时为真）；跨文档题支持多来源引用（`预期来源` 用分号分隔，Q18、Q19 的次要来源写在题库文件头的 `SECONDARY_SOURCES`）。

```bash
python eval/run_eval.py        # 跑 42 条记录，结果写进 eval/results.json + eval/results.md
python eval/run_eval_auto.py   # 一键：跑评测 + 自动和上一轮对比 + 按停表规则判通过/不通过
```

`run_eval_auto.py` 的流程：先把现有的 `eval/results.json` 备份成 `eval/results_prev.json`，再跑评测，逐题对比（哪些退步、哪些进步），按停表规则判定——原 20 题可回答已答 ≥ 12/15，且原 20 题没有新的严重退步（路由对变错、引用 full→partial、完成度下降）→通过（退出码 0），否则不通过（退出码 1）。图表题和多轮的退步只列出来、不单独判不通过。对比报告写进 `eval/results_compare.md`。

注意：完整评测在本地运行（需要 bge-m3 和 LLM key，VM 上跑不了）。

指标口径（`eval/run_eval.py` 开头有完整说明）：
- 已答/拒答：回答的第一句里是否出现“无法回答”。
- 引用准确率 = 引用命中的可回答题数 / 已回答的可回答题数（引用是检索到的片段列表，度量的是检索是否召回了预期来源）。
- 编造次数 = 应拒答题被回答的次数，目标 0 次。
- 答对率需人工核对（对照 `eval/results.md` 逐题表里的回答原文与预期答案），脚本不自动判。

## 测试

```bash
pytest -q  # 全量 376 个测试（截至 2026-11-29），不需要真 LLM、不需要 Chroma、不联网（假 LLM + 假检索 + 假 VLM）
```

覆盖重点：路由器、规划器、执行器、工具注册、VLM 读图、核查员（含部分回答与重试）、跨文献汇总、多轮对话、评测维度解析。`app/agent.py`、`app/tools.py`、`app/vision.py` 三个第二阶段核心模块覆盖率 97%–99%，剩余是构造上到不了的防御分支。

## 目录

- `app/main.py`：FastAPI 入口（`/health`、`/ask`、`/ask/stream`）
- `app/agent.py`：LangGraph 流程（路由器→规划器/执行器/读图/PubMed/检索员→回答员→核查员→重试）+ checkpointer + 多轮历史
- `app/tools.py`：统一工具注册表（local_search、pubmed_search、chart_locate、chart_read）+ 参数校验
- `app/vision.py`：VLM 读图（页面渲染、图注裁剪、数字提取、图注交叉核对）
- `app/memory.py`：历史管理 + 上下文压缩（最近 6 轮/4000 字，超限摘要较早轮次）
- `app/mcp_tools.py`：MCP PubMed 搜索（mcp SDK 直连服务端，搜 + 取元数据，只搜不下）
- `app/rag.py`：混合检索（向量 + BM25，RRF 融合，多篇点名按篇取）+ 引用格式化 + prompt 拼装
- `app/ingest.py`：PDF 解析、按句子切分、图注/表注提取、入库
- `app/config.py`：配置（`.env`，含 VLM 和 PubMed 配置）
- `app/prompts.py`：提示词 loader
- `app/streamlit_app.py`：演示页面（调 `localhost:8000`，流式显示）
- `prompts/`：提示词文件（按版本）
- `eval/test_questions.md`：40 道开发评测题；`eval/run_eval.py`：评测脚本；`eval/run_eval_auto.py`：评测自动化
- `eval/datasets/`：开发集/冻结测试集划分（`SPLIT.md`、`dev.txt`、`test.txt`）+ 冻结测试题（`test_questions_frozen.md`）
- `eval/check_retrieval.py`：检索级验证（不需要 LLM/Chroma/embedding）
- `docs/crossdoc-five-papers.md`：五篇纵向常模论文对比表任务定义
- `docs/engineering-notes.md`：工程问题记录（开始的效果 / 遇到的问题 / 原因 / 怎么改的 / 提效）
- `data/papers/`：放公开论文 PDF，不要放 ADNI/NACC 受限数据
