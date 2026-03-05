# Andela AI Platform - Production RAG Backend

This project implements a production-grade AI backend based on the **Software, AI Engineer** role requirements at Andela.

## 🚀 Features
- **FastAPI Backend:** Scalable, asynchronous API with contract-first design.
- **Advanced RAG:** LlamaIndex with **Sentence Window Retrieval** for superior context management.
- **Vector Database:** Qdrant integration for high-performance retrieval.
- **Operational Rigor:** 
  - API Key Authentication (`X-API-Key: andela-secret-key`).
  - Latency tracking.
  - Evaluation harness using **DeepEval** (Faithfulness & Relevancy).
  - Pydantic models for strict data validation.

## 🛠 Prerequisites
1. **Python 3.10+**
2. **OpenAI API Key** (for GPT-4o and Embeddings).
3. **Qdrant** (Optional: can run locally in Docker).

## 📦 Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   Create a `.env` file from `.env.example` and add your `OPENAI_API_KEY`.

3. **Run Qdrant (Docker):**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

4. **Ingest Data:**
   Place your PDF or text files in the `data/` directory, then run:
   ```bash
   # Use the API to ingest
   curl -X POST http://localhost:8005/ingest -H "X-API-Key: andela-secret-key"
   ```

5. **Start the API:**
   ```bash
   uvicorn app.main:app --reload --port 8005
   ```

## 🧪 Testing & Evaluation
Run the evaluation suite to measure RAG accuracy:
```bash
pytest tests/test_evals.py
```

## 🔐 Security & Governance
- **Auth:** Implemented via `APIKeyHeader`.
- **Validation:** All inputs/outputs are validated via Pydantic.
- **Observability:** Latency is tracked per request.
