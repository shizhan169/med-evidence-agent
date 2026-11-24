import streamlit as st
import httpx

st.set_page_config(page_title="MedEvidence Agent", layout="wide")
st.title("MedEvidence Agent — 带引用溯源的医学文献问答")
st.caption("每条结论带来源页码，没有依据就明确拒答。")

query = st.text_input("问一个医学文献问题", placeholder="例如：仑卡奈单抗对早期阿尔茨海默病 MMSE 的影响是什么？")
if st.button("提问") and query:
    with st.spinner("正在检索文献..."):
        try:
            r = httpx.post("http://localhost:8000/ask", json={"query": query}, timeout=60)
            data = r.json()
            st.write(data.get("answer", ""))
            if data.get("citations"):
                st.subheader("引用")
                for c in data["citations"]:
                    st.write(f"- {c.get('source')} p.{c.get('page')}")
            else:
                st.info("本次无引用，说明文献库中暂无依据。")
        except Exception as e:
            st.error(f"后端未启动，请先跑 uvicorn app.main:app --reload --port 8000。错误：{e}")
