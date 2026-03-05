import streamlit as st
import requests
import time

st.set_page_config(page_title="Andela AI Platform - Chat", layout="wide")

st.title("🚀 Andela AI Platform - RAG Chat")
st.sidebar.header("Configurações")

API_URL = st.sidebar.text_input("API URL", value="http://localhost:8005")
API_KEY = st.sidebar.text_input("X-API-Key", value="andela-secret-key", type="password")

if st.sidebar.button("Ingerir Documentos (Pasta /data)"):
    with st.spinner("Processando documentos..."):
        try:
            res = requests.post(f"{API_URL}/ingest", headers={"X-API-Key": API_KEY})
            if res.status_code == 200:
                st.sidebar.success("Documentos ingeridos com sucesso!")
            else:
                st.sidebar.error(f"Erro: {res.text}")
        except Exception as e:
            st.sidebar.error(f"Erro de conexão: {e}")

# Histórico de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "latency" in message:
            st.caption(f"Latência: {message['latency']:.2f}ms")

if prompt := st.chat_input("Pergunte algo sobre a vaga ou os documentos..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
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
                    st.caption(f"Latência: {latency:.2f}ms")
                    
                    with st.expander("Ver Fontes (Sources)"):
                        for i, source in enumerate(data["sources"]):
                            st.info(f"Fonte {i+1}: {source}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "latency": latency
                    })
                else:
                    st.error(f"Erro na API: {response.status_code}")
            except Exception as e:
                st.error(f"Erro de conexão: {e}")
