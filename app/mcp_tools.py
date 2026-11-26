"""MCP PubMed 搜索（第 5 步，5a：只搜不下）。

接法：Python 的 mcp SDK 直连 @cyanheads/pubmed-mcp-server（npx 拉起，stdio）。
为什么不用 langchain-mcp-adapters 的 MultiServerMCPClient（主流做法）：
它所有已发布版本都要求 langchain-core>=0.3.36，和本项目 langgraph 0.2.0 /
langchain-core 0.2.x 的钉版冲突，pip 解析直接失败（ResolutionImpossible，
2026-11-26 实测）。mcp SDK 就是 langchain-mcp-adapters 底层用的那个包，
没有 langchain-core 依赖；等项目升级到 langchain-core 0.3+ / langgraph 1.x
再换成 MultiServerMCPClient。

服务端实测（2026-11-26，v2.12.4，VM 上真跑过）：
- 搜索工具叫 pubmed_search_articles（不是方案稿写的 search_pubmed），
  只返回 PMID 列表，不返回标题摘要；要标题/作者/期刊/年份/摘要，
  必须再调 pubmed_fetch_articles（按 PMID 取元数据）。所以一次 PubMed
  查询 = 两次工具调用（搜 + 取），都是只读，不写库、不下载全文。
- 两个工具返回的都是 Markdown 文本，不是 JSON，靠正则解析（见 _parse_*）。
- 只搜 PubMed，不下载、不入库：搜到的文章只用来回答这一次，下次问
  同样的问题还会再搜一次（5b/5c 再做下载和入库）。

失败一律返回 []（不断）：服务端没起来（没装 Node、npx 拉包失败）、
NCBI 调不通、超时、返回格式不对——调用方按本地-only/拒答走，问答不断。
"""
import asyncio
import os
import re
import shlex

from .config import (
    MCP_PUBMED_ARGS,
    MCP_PUBMED_COMMAND,
    MCP_PUBMED_ENABLED,
    MCP_PUBMED_TIMEOUT,
    MCP_PUBMED_TOOL_FETCH,
    MCP_PUBMED_TOOL_SEARCH,
    NCBI_EMAIL,
    PUBMED_TOP_K,
)

# 传给 MCP 子进程的环境变量：MCP SDK 默认只继承少量安全变量（PATH、HOME 等），
# 代理和证书变量不继承——VM 上实测，不显式传 HTTPS_PROXY + NODE_EXTRA_CA_CERTS，
# 子进程连不上 NCBI（fetch 直接失败）。有这些变量才传，没有就不传；
# 用户本机（VPN 直连）一般没有这些变量，不影响。
_FORWARDED_ENV_KEYS = (
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
    "NODE_USE_ENV_PROXY", "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE",
)


def _server_env() -> dict:
    """拼给 MCP 服务端子进程的环境变量。"""
    env = {"MCP_TRANSPORT_TYPE": "stdio"}
    if NCBI_EMAIL:
        env["NCBI_EMAIL"] = NCBI_EMAIL
    for key in _FORWARDED_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def _tool_text(result) -> str:
    """MCP 工具调用结果转纯文本：只取 text 块拼起来；出错（isError）返回 ""。"""
    if result is None or getattr(result, "isError", False):
        return ""
    parts = []
    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", "") == "text":
            parts.append(getattr(block, "text", "") or "")
    return "\n".join(parts)


def _parse_search_result(text: str) -> list:
    """解析 pubmed_search_articles 的返回：只认 `**PMIDs:** 38395541, ...` 那一行，
    按顺序去重。行里没有 PMID（0 命中、或格式变了）就返回 []。"""
    m = re.search(r"\*\*PMIDs:\*\*\s*([0-9,\s]+)", text or "")
    if not m:
        return []
    seen = set()
    pmids = []
    for pmid in m.group(1).replace(",", " ").split():
        if pmid.isdigit() and pmid not in seen:
            seen.add(pmid)
            pmids.append(pmid)
    return pmids


def _parse_fetch_result(text: str) -> list:
    """解析 pubmed_fetch_articles 的返回（Markdown，每篇一个 `### 标题` 块），
    每篇取 pmid / title / authors（前 3 位 + 等）/ journal / year / abstract
    （600 字以内）。缺 PMID 的块不要（没法引用）。"""
    articles = []
    # 按 "### 标题" 切块；第一块是 "## PubMed Articles" 的文件头，不要
    blocks = re.split(r"(?m)^### ", text or "")
    for block in blocks[1:]:
        lines = block.splitlines()
        title = (lines[0] if lines else "").strip()
        pmid_m = re.search(r"\*\*PMID:\*\*\s*(\d+)", block)
        if not pmid_m:
            continue
        journal_m = re.search(r"\*\*Journal:\*\*\s*(.+)", block)
        journal = journal_m.group(1).strip() if journal_m else ""
        year_m = re.search(r"\b(19|20)\d{2}\b", journal)
        year = year_m.group(0) if year_m else ""
        authors = _parse_authors(block)
        abstract = _parse_abstract(block)
        articles.append({
            "pmid": pmid_m.group(1),
            "title": title,
            "authors": authors,
            "journal": journal,
            "year": year,
            "abstract": abstract,
        })
    return articles


def _parse_authors(block: str) -> str:
    """取作者列表的前 3 位（"Ruiyang Ge (R) [aff 0]" 这种行只取括号前的名字），
    标题里的总数（Authors (31)）比展示的多就加“等”。没有作者列表就返回 ""。"""
    m = re.search(r"\*\*Authors(?:\s*\((\d+)\))?:\*\*\s*\n((?:- .+\n?)+)", block)
    if not m:
        return ""
    total = int(m.group(1)) if m.group(1) else None
    names = []
    for line in m.group(2).splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        name = line[2:].split(" (")[0].strip()
        if name:
            names.append(name)
    if not names:
        return ""
    shown = names[:3]
    more = (total is not None and total > len(shown)) or len(names) > 3
    return "、".join(shown) + (" 等" if more else "")


def _parse_abstract(block: str) -> str:
    """取 `#### Abstract` 段的文字，到下一个 `####`/`###` 标题为止；超 600 字截断。
    有的文章没有摘要，返回 ""。"""
    m = re.search(r"#### Abstract\s*\n(.*?)(?=\n#### |\n### |\Z)", block, re.S)
    if not m:
        return ""
    abstract = re.sub(r"\s+", " ", m.group(1)).strip()
    return abstract[:600]


async def _search_pubmed_async(query: str, top_k: int, sort: str) -> list:
    """一次 PubMed 查询：搜（拿 PMID）+ 取（拿元数据）。任何一步失败都抛异常，
    由 search_pubmed 接住、返回 []。每次调用都新起一个 MCP 子进程（npx），
    用完即关——简单可靠，慢一点（首跑要拉包）也能接受，因为这只是补位路径。"""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=MCP_PUBMED_COMMAND,
        args=shlex.split(MCP_PUBMED_ARGS),
        env=_server_env(),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            search = await session.call_tool(
                MCP_PUBMED_TOOL_SEARCH,
                {"query": query, "maxResults": top_k, "sort": sort},
            )
            pmids = _parse_search_result(_tool_text(search))[:top_k]
            if not pmids:
                return []
            fetch = await session.call_tool(
                MCP_PUBMED_TOOL_FETCH, {"pmids": pmids}
            )
            return _parse_fetch_result(_tool_text(fetch))


def search_pubmed(query: str, top_k: int = PUBMED_TOP_K, sort: str = "relevance") -> list:
    """同步入口：搜 PubMed，返回文章列表 [{pmid, title, authors, journal,
    year, abstract}]，最多 top_k 篇。PubMed 关了、服务端起不来、NCBI 调不通、
    超时、没搜到——一律返回 []，绝不抛异常（调用方按本地-only/拒答走）。
    sort：relevance（默认）或 pub_date（问“最新文献”时用，最新在前）。"""
    if not MCP_PUBMED_ENABLED:
        return []
    if not (query or "").strip():
        return []
    try:
        return asyncio.run(
            asyncio.wait_for(
                _search_pubmed_async(query, top_k, sort),
                timeout=MCP_PUBMED_TIMEOUT,
            )
        )
    except Exception as e:
        print(f"[pubmed] PubMed 查询失败（{e}），退回本地-only")
        return []
