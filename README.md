# MedEvidence Agent — 面向专业文档的带引用 RAG Agent

首个垂类：医学文献。架构通用，可换成法律、金融等任意专业文档。

## 解决什么问题
普通 LLM 回答医学问题容易编造，且没有来源。MedEvidence 强制每条回答标注来源页码，没有依据就明确拒答。

## 架构：输入 - 处理 - 输出
输入：用户问题 + PDF 论文库
处理：
1. 检索员 Retriever：Chroma 向量检索 Top-K 片段
2. 回答员 Answerer：Qwen / DeepSeek API 生成带引用 [来源：论文名 p.X] 的回答
3. 核查员 Verifier：检查引用是否真实存在、是否支持结论，不支持就拒答或改写
输出：回答 + 引用列表 + 置信度

## 技术栈
Python + FastAPI + LangGraph + Chroma + SQLite + Streamlit + Docker
大模型：Qwen / DeepSeek API（通过 .env 配置）

## 今晚 MVP 目标（第 1 周）
- 5 篇公开论文 PDF 放进 data/papers/
- 跑通：解析 -> 切分 -> embedding -> 检索 -> 带引用问答
- 接口：POST /ask 返回 {answer, citations}
- 演示：Streamlit 页面能问、能看到引用

## 快速开始
```bash
conda create -n medevidence python=3.11 -y
conda activate medevidence
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple  # 国内用清华源，快很多
cp .env.example .env  # 填入 QWEN_API_KEY 或 DEEPSEEK_API_KEY
python -m app.ingest --papers data/papers  # 首次运行会下载 embedding 模型（约 90MB），耐心等几分钟，只需一次
uvicorn app.main:app --reload --port 8000
streamlit run app/streamlit_app.py
```

## 目录
- app/main.py：FastAPI 入口
- app/ingest.py：PDF 解析、切分、入库
- app/rag.py：检索 + 引用格式化
- app/agent.py：LangGraph 三角色流程
- app/config.py：配置
- eval/test_questions.md：20 道评测题（第 3 周做）
- data/papers/：放公开论文 PDF，不要放 ADNI/NACC 受限数据

## 轻量版保留的三项（不阻塞投递）
- 多智能体：检索/回答/核查三角色，第 2 周
- 多模态：论文图表图注解析 + 看图提问演示，第 2 周
- 微调：限时 2 天小规模 LoRA 对照实验，只比引用格式、拒答、答对率，第 4 周
完整版放到第 5 周起边面试边做。
