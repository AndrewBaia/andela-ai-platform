from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from llama_index.llms.ollama import Ollama
from app.services.rag_service import rag_engine
import pytest
import re
import asyncio

# Métrica Customizada e Resiliente (Andrew Baía - Technical Governance)
# Esta métrica é otimizada para modelos locais (Ollama) e evita erros de parsing JSON complexos.
class LocalRAGFidelityMetric(BaseMetric):
    def __init__(self, threshold=0.5, model_name="llama3.1:8b"):
        self.threshold = threshold
        self.model = Ollama(model=model_name, base_url="http://localhost:11434", request_timeout=120.0)
        self.score = 0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase):
        """Mede a fidelidade da resposta ao contexto usando um prompt direto."""
        prompt = f"""
        Você é um avaliador de sistemas de IA. Sua tarefa é verificar se a RESPOSTA é fiel ao CONTEXTO fornecido.
        
        CONTEXTO:
        {test_case.retrieval_context}
        
        RESPOSTA:
        {test_case.actual_output}
        
        Avalie a fidelidade e dê uma nota de 0 a 10.
        Responda APENAS com o número da nota (ex: 8). Não escreva mais nada.
        """
        try:
            response = str(self.model.complete(prompt)).strip()
            # Extrai apenas o primeiro número encontrado na resposta
            match = re.search(r"(\d+)", response)
            if match:
                val = int(match.group(1))
                self.score = val / 10.0
            else:
                self.score = 0
        except Exception as e:
            print(f"Erro na avaliação: {e}")
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
    """Avaliação de RAG otimizada para ambiente local (Anti-Alucinação)."""
    
    query = "What are the key skills required for the AI Engineer role?"
    query_engine = rag_engine.get_query_engine()
    
    # 1. Geração da Resposta
    response = query_engine.query(query)
    actual_output = str(response)
    
    # 2. Recuperação do Contexto
    retrieval_context = [node.node.get_content() for node in response.source_nodes]
    
    # 3. Avaliação com Métrica Customizada
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
    print(f"Pergunta: {query}")
    print(f"Score de Fidelidade: {score * 10}/10")
    print(f"Status: {'✅ PASS' if metric.is_successful() else '❌ FAIL'}")
    print(f"="*30 + "\n")
    
    assert metric.is_successful(), f"Fidelidade insuficiente: {score*10}/10"
