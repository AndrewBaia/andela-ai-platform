import streamlit as st
import requests
import time

st.set_page_config(page_title="Andela AI Platform - Chat", layout="wide")

st.title("🚀 Andela AI Platform - RAG Chat")
st.sidebar.header("Settings")

API_URL = st.sidebar.text_input("API URL", value="http://localhost:8005")
API_KEY = st.sidebar.text_input("X-API-Key", value="andela-secret-key", type="password")

if st.sidebar.button("Ingest Documents (/data folder)"):
    with st.spinner("Processing documents..."):
        try:
            res = requests.post(f"{API_URL}/ingest", headers={"X-API-Key": API_KEY})
            if res.status_code == 200:
                st.sidebar.success("Documents ingested successfully!")
            else:
                st.sidebar.error(f"Error: {res.text}")
        except Exception as e:
            st.sidebar.error(f"Connection error: {e}")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "latency" in message:
            st.caption(f"Latency: {message['latency']:.2f}ms")

if prompt := st.chat_input("Ask something about the job or the documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    headers={"X-API-Key": API_KEY},
                    json={"query": prompt}
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    latency = data["latency_ms"]
                    
                    st.markdown(answer)
                    st.caption(f"Latency: {latency:.2f}ms")
                    
                    with st.expander("View Sources"):
                        for i, source in enumerate(data["sources"]):
                            st.info(f"Source {i+1}: {source}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "latency": latency
                    })
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection error: {e}")
