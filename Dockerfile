# MedEvidence Agent：FastAPI 服务（uvicorn），端口 8000。
# 模型和数据不进镜像，运行时从本机挂载（见 README）：
#   -v <项目>/models:/app/models   本地 bge-m3（embedding）
#   -v <项目>/data:/app/data       本地 Chroma 库 + 论文 PDF
#   --env-file .env                API key 只在本地 .env，不进镜像
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
