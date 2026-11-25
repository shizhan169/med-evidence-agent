import json
import uuid

import streamlit as st
import httpx

st.set_page_config(page_title="MedEvidence Agent", layout="wide")
st.title("MedEvidence Agent — 带引用溯源的医学文献问答")
st.caption("每条结论带来源页码，没有依据就明确拒答。")

# 多轮对话历史存在 session_state 里：每轮一问一答，刷新页面不丢，
# 点“清空对话”才清掉；每次提问都把历史一起发给后端，后端只取最近 6 轮。
if "history" not in st.session_state:
    st.session_state.history = []

# 会话 id：每个浏览器会话固定一个，后端用它把对话存进 SQLite checkpointer，
# 服务重启后换个浏览器也不会串；点“清空对话”时换一个新的。
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 先显示历史对话
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.text_input("问一个医学文献问题", placeholder="例如：仑卡奈单抗对早期阿尔茨海默病 MMSE 的影响是什么？")
col_ask, col_clear = st.columns([1, 5])
ask_clicked = col_ask.button("提问")
clear_clicked = col_clear.button("清空对话")
if clear_clicked:
    st.session_state.history = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()
if ask_clicked and query:
    # 流式问答：后端先跑完检索→回答→核查，再把最终答案逐块推回来，
    # 这里一边收一边显示；收完才把本轮写进历史、展示引用。
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.write("正在检索文献...")
        full = ""
        verified = False
        citations = []
        try:
            with httpx.stream(
                "POST",
                "http://localhost:8000/ask/stream",
                json={"query": query, "history": st.session_state.history,
                      "thread_id": st.session_state.thread_id},
                timeout=120,
            ) as r:
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:].strip())
                    if event.get("type") == "token":
                        full += event.get("text", "")
                        placeholder.write(full)
                    elif event.get("type") == "done":
                        verified = event.get("verified", False)
                        citations = event.get("citations", [])
                    elif event.get("type") == "error":
                        placeholder.write("")
                        st.error(f"后端出错：{event.get('message', '')}")
                        full = ""
                        break
        except Exception as e:
            placeholder.write("")
            st.error(f"后端未启动，请先跑 uvicorn app.main:app --reload --port 8000。错误：{e}")
            full = ""
    if full:
        # 先把本轮问答存进历史，下次追问（比如“那白质呢？”）才能接上文
        st.session_state.history.append({"role": "user", "content": query})
        st.session_state.history.append({"role": "assistant", "content": full})
        if citations:
            st.subheader("引用")
            for c in citations:
                st.write(f"- {c.get('source')} p.{c.get('page')}")
        else:
            st.info("本次无引用，说明文献库中暂无依据。")
