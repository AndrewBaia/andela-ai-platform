from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from app.schemas.api_schemas import QueryRequest, QueryResponse, IngestionResponse
from app.services.rag_service import rag_engine
from app.config import settings
import time
from loguru import logger

app = FastAPI(
    title="Andela AI Platform - Backend Service",
    description="Production-grade RAG API for intelligent content systems.",
    version="1.0.0"
)

# Simple API Key Auth for Production Rigor
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == "andela-secret-key": # In production, use a secure vault
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(get_api_key)])
async def query_ai(request: QueryRequest):
    start_time = time.time()
    try:
        query_engine = rag_engine.get_query_engine()
        response = query_engine.query(request.query)
        
        latency = (time.time() - start_time) * 1000
        
        return QueryResponse(
            answer=str(response),
            sources=[node.node.get_content()[:200] + "..." for node in response.source_nodes],
            latency_ms=latency
        )
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal AI Engine Error")

@app.post("/ingest", response_model=IngestionResponse, dependencies=[Depends(get_api_key)])
async def ingest_data():
    try:
        rag_engine.ingest_documents("./data")
        return IngestionResponse(status="success", message="Documents ingested successfully")
    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}
