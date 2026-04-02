# app/ui.py

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(
    page_title="SEBI Compliance RAG Assistant",
    layout="wide",
)

st.title("SEBI Compliance RAG Assistant")
st.caption("Conversational AI over SEBI regulations (RAG + Groq LLaMA)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask a question about SEBI regulations...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.spinner("Retrieving SEBI-compliant answer..."):
        payload = {
            "question": user_input,
            "history": st.session_state.history,
        }
        res = requests.post(API_URL, json=payload, timeout=60)

    if res.status_code == 200:
        data = res.json()
        answer = data["answer"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.history.append({"user": user_input, "assistant": answer})
        st.chat_message("assistant").write(answer)

        with st.expander("Evidence (Retrieved Chunks)"):
            for item in data["evidence"]:
                score = item.get("score")
                source = item.get("source_file", "unknown")
                chunk_index = item.get("chunk_index", "n/a")
                label = f"**Source:** {source} - chunk {chunk_index}"
                if score is not None:
                    label += f" (score {score:.4f})"
                st.write(label)
                st.write(item["text"])
                st.write("---")
    else:
        st.error(f"API error: {res.status_code} - {res.text}")
