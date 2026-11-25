# 提示词

prompt 不写在代码里，放在这里按版本管理：`prompts/<名字>/<版本>.txt`。

- `answerer/v1.txt`：回答员的 prompt 模板（`{history_text}`、`{context}`、`{query}` 是占位符，代码里填）。
- `retry/v1.txt`：核查失败后重答时的提醒。

改 prompt 就改这里的文件，不用动 `app/` 里的代码；改完重启服务生效。
想试新版 prompt：新建一个版本文件（比如 `v2.txt`），再把调用处
（`app/rag.py`、`app/agent.py`）的版本号改成 `v2`——旧版文件留着，
随时能切回去。
