import streamlit as st
import httpx

st.set_page_config(page_title="MedEvidence Agent", layout="wide")
st.title("MedEvidence Agent — 带引用溯源的医学文献问答")
st.caption("每条结论带来源页码，没有依据就明确拒答。")

# 多轮对话历史存在 session_state 里：每轮一问一答，刷新页面不丢，
# 点“清空对话”才清掉；每次提问都把历史一起发给后端，后端只取最近 6 轮。
if "history" not in st.session_state:
    st.session_state.history = []

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
    st.rerun()
if ask_clicked and query:
    with st.spinner("正在检索文献..."):
        try:
            r = httpx.post(
                "http://localhost:8000/ask",
                json={"query": query, "history": st.session_state.history},
                timeout=60,
            )
            data = r.json()
            answer = data.get("answer", "")
            # 先把本轮问答存进历史，下次追问（比如“那白质呢？”）才能接上文
            st.session_state.history.append({"role": "user", "content": query})
            st.session_state.history.append({"role": "assistant", "content": answer})
            st.write(answer)
            if data.get("citations"):
                st.subheader("引用")
                for c in data["citations"]:
                    st.write(f"- {c.get('source')} p.{c.get('page')}")
            else:
                st.info("本次无引用，说明文献库中暂无依据。")
        except Exception as e:
            st.error(f"后端未启动，请先跑 uvicorn app.main:app --reload --port 8000。错误：{e}")
