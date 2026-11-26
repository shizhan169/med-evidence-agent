# 学习笔记：MCP 协议与 PubMed 工具接入（第 5 步）

> 给学习板块（Space）的一课。对应实现：第 5 步 5a，`app/mcp_tools.py` + `app/agent.py` 的 PubMed 分支。

## 1. MCP 是什么

MCP（Model Context Protocol，模型上下文协议）是 Anthropic 2024 年底提出的一个开放协议，解决的是同一个问题：**大模型 / Agent 怎么调用外部工具**。

没有 MCP 之前，每个工具都要单独写一套调用代码：调 PubMed 要写一套，调天气 API 又要写一套，格式各不一样。有了 MCP，工具的提供方（服务端）和调用方（客户端）都按同一种规矩说话：

- **服务端（server）**：暴露“工具列表”和“调用工具”两个能力。比如 PubMed 的服务端说：我有 `pubmed_search_articles`（搜文章）和 `pubmed_fetch_articles`（取元数据）这两个工具，每个工具要哪些参数、返回什么格式，都在协议里写清楚。
- **客户端（client）**：先问服务端“你有什么工具”，再按名字调，参数用 JSON 传。
- **传输（transport）**：两边怎么把消息传过去。最常用的是两种：stdio（客户端把服务端当子进程拉起来，用标准输入输出传 JSON-RPC 消息）和 Streamable HTTP（走 HTTP 请求）。我们这个项目用 stdio：Python 进程用 `npx` 拉起 Node 写的 PubMed 服务端，通过管道对话。

关键点：MCP 只规定“怎么发现工具、怎么调工具”，不管工具内部怎么做。PubMed 服务端内部调的是 NCBI 的 E-utilities（esearch + esummary），那是 NCBI 自己的 API；MCP 只是把它包了一层标准接口。

## 2. 为什么用现成的服务端，不自己写

主流做法是**直接用现成的服务端 + 客户端库接进来**，不要自己写协议层：

- 现成的 PubMed 服务端（`@cyanheads/pubmed-mcp-server`）已经处理了 NCBI 的各种细节：限流、重试、分页、返回格式、错误恢复。自己写要重新踩一遍这些坑。
- 客户端库（`langchain-mcp-adapters` 的 `MultiServerMCPClient`）把 MCP 工具直接转成 LangChain/LangGraph 能用的 tool，不用自己写 JSON-RPC。

但这个项目里，主流做法走不通，改用了次主流的做法：

- `langchain-mcp-adapters` 所有已发布版本都要求 `langchain-core>=0.3.36`，而本项目钉的是 `langgraph 0.2.0` / `langchain-core 0.2.x`，pip 解析直接失败（`ResolutionImpossible`，2026-11-26 实测）。升级 `langchain-core` 意味着连 `langgraph` 一起升，是另一个大动作，不能为了接一个工具顺手升全家。
- 所以改用 Python 的 `mcp` SDK（`mcp` 包）直连服务端。`mcp` SDK 就是 `langchain-mcp-adapters` 底层用的那个包，没有 `langchain-core` 依赖；等项目升级到 `langchain-core 0.3+` / `langgraph 1.x` 再换成 `MultiServerMCPClient`。接口长得差不多（`stdio_client` + `ClientSession` + `call_tool`），换的时候只改 `app/mcp_tools.py` 内部，不动图的逻辑。

**和手写的区别**：如果自己用 FastMCP 写一个小的服务端（约 100 行，包一层 E-utilities），确实更能学明白 MCP 协议（自己定义工具、写 JSON Schema、处理 stdio 传输）。但那是“为了学而手写”：多一个自己维护的组件，NCBI 的限流重试还得自己写。先学会用现成的，知道协议长什么样；以后真要定制（比如 5b 的下载、5c 的入库），再考虑自己写。

## 3. 我们这个项目里 PubMed 是怎么接进来的

流程（图里检索员后面多了一条条件边）：

```
用户问题
  → 检索员：先查本地库（向量 + BM25）
  → 本地有证据、且用户没问新文献 → 回答员（老路，本地引用）
  → 本地没证据、或用户明确问新文献 → PubMed 节点
      → 搜（pubmed_search_articles，拿 PMID）
      → 取（pubmed_fetch_articles，拿标题/作者/期刊/年份/摘要）
      → 回答员（PubMed 专用 prompt，引用格式 PubMed PMID:xxx）
  → 核查员：PubMed 路径只认本次返回的 PMID，假 PMID 按编造处理
```

三个容易错的点，都是实测跑出来的：

1. **搜和取是两个工具**。方案稿写的是“一个工具 search_pubmed，返回前 5 篇的 PMID、标题、作者、期刊、年份、摘要”。实测服务端 2.12.4 的搜索工具只返回 PMID 列表，标题摘要要再调取文章的工具。所以“一个工具”实际是“一个能力（PubMed 搜索），两次工具调用”。——方案稿里的工具名和返回格式不能照抄，要拿真服务端跑一遍再定。
2. **中文问题要先翻成英文再搜**。PubMed 的索引是英文文献，中文词一个都匹配不到。所以 PubMed 路径上调一次 LLM，把中文问题翻成英文检索词（纯英文问题不翻译）。这只是给 PubMed 搜的，本地检索的中文问题翻英文是另一回事，以后再说。
3. **子进程的环境变量不是全继承的**。MCP SDK 拉子进程时只继承少量安全变量（PATH、HOME 等），代理变量（HTTP_PROXY 等）和证书变量（NODE_EXTRA_CA_CERTS）不继承。VM 上实测：不显式传这些变量，子进程连不上 NCBI。所以 `app/mcp_tools.py` 里把这些变量显式拼进子进程的环境里（有才传，没有不传；用户本机 VPN 直连一般没有这些变量，不影响）。

## 4. 防编造是怎么做的（和本地引用同一条规矩）

- PubMed 的引用格式是 `PubMed PMID:xxxxxxxx`，和本地的 `[来源：论文名 p.页码]` 严格区分，答案里一眼看得出这是外部文献。
- 核查员只认**本次** PubMed 返回的 PMID 列表：答案里出现一个工具没返回过的 PMID，就按引用编造处理（走第 3 步的重试/拒答）。和本地引用的核查是同一条规矩，不开后门。
- PubMed 答案里不许混本地式引用（`[来源：x.pdf p.N]`），混了也算核查失败——两种引用体系不混在一道答案里。

## 5. 评测为什么不受影响

28 道评测题测的是本地 RAG。应拒答题的定义就是“本地没证据”，正好撞上 PubMed 的触发条件（本地没证据 → 调 PubMed），所以评测时把 PubMed 关掉（`run(..., use_pubmed=False)`），不然数字会被外部服务带偏、还不可复现。PubMed 能力由 mock 测试 + 一次真跑验收，不走 28 道题。评测题库一个字没动。

## 6. 面试可以讲的点

- “我怎么把一个 MCP 工具接进 LangGraph”：检索员后面加条件边，本地优先、PubMed 补位；失败退回本地-only，问答不断。
- “主流做法和实际取舍”：主流是 `langchain-mcp-adapters` + 现成服务端；但它和项目钉的 `langchain-core 0.2.x` 冲突，改用 `mcp` SDK 直连，并写清楚以后升级了再换回来。
- “防编造”：核查员只认本次工具返回的 PMID，假 PMID 走重试/拒答，和本地引用同一条规矩。
