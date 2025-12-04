# app/ui.py

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(
    page_title="SEBI Compliance RAG Assistant",
    layout="wide"
)

st.title("📘 SEBI Compliance RAG Assistant")
st.caption("AI-driven regulatory intelligence powered by RAG + LLaMA on Groq")

query = st.text_area(
    "Enter your question about SEBI regulations:",
    placeholder="Example: What are the disclosure requirements for listed entities?"
)

if st.button("Submit Query"):
    if not query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Retrieving information from SEBI documents..."):
            payload = {"question": query}
            response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            data = response.json()

            st.subheader("✅ SEBI-Compliant Answer")
            st.write(data["answer"])

            st.subheader("📑 Evidence (Retrieved Chunks)")
            for item in data["evidence"]:
                with st.expander(f"{item['source_file']} — chunk {item['chunk_index']}  (score {item['score']:.4f})"):
                    st.write(item["text"])
        else:
            st.error("API error: Unable to retrieve data. Check FastAPI logs.")
