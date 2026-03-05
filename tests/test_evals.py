from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from llama_index.llms.ollama import Ollama
from app.services.rag_service import rag_engine
import pytest
import re
import asyncio

# Custom Resilient Metric (Andrew Baía - Technical Governance)
# This metric is optimized for local models (Ollama) and avoids complex JSON parsing errors.
class LocalRAGFidelityMetric(BaseMetric):
    def __init__(self, threshold=0.5, model_name="llama3.1:8b"):
        self.threshold = threshold
        self.model = Ollama(model=model_name, base_url="http://localhost:11434", request_timeout=120.0)
        self.score = 0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase):
        """Measures response fidelity to context using a direct prompt."""
        prompt = f"""
        You are an AI system evaluator. Your task is to verify if the RESPONSE is faithful to the provided CONTEXT.
        
        CONTEXT:
        {test_case.retrieval_context}
        
        RESPONSE:
        {test_case.actual_output}
        
        Evaluate the fidelity and give a score from 0 to 10.
        Respond ONLY with the score number (e.g., 8). Do not write anything else.
        """
        try:
            response = str(self.model.complete(prompt)).strip()
            # Extract only the first number found in the response
            match = re.search(r"(\d+)", response)
            if match:
                val = int(match.group(1))
                self.score = val / 10.0
            else:
                self.score = 0
        except Exception as e:
            print(f"Evaluation error: {e}")
            self.score = 0
        
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Local RAG Fidelity"

@pytest.mark.asyncio
async def test_rag_performance():
    """RAG evaluation optimized for local environment (Anti-Hallucination)."""
    
    query = "What are the key skills required for the AI Engineer role?"
    query_engine = rag_engine.get_query_engine()
    
    # 1. Response Generation
    response = query_engine.query(query)
    actual_output = str(response)
    
    # 2. Context Retrieval
    retrieval_context = [node.node.get_content() for node in response.source_nodes]
    
    # 3. Evaluation with Custom Metric
    metric = LocalRAGFidelityMetric(threshold=0.5, model_name="llama3.1:8b")
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )
    
    score = metric.measure(test_case)
    
    print(f"\n" + "="*30)
    print(f"📊 RAG EVALUATION REPORT")
    print(f"="*30)
    print(f"Query: {query}")
    print(f"Fidelity Score: {score * 10}/10")
    print(f"Status: {'✅ PASS' if metric.is_successful() else '❌ FAIL'}")
    print(f"="*30 + "\n")
    
    assert metric.is_successful(), f"Insufficient fidelity: {score*10}/10"
