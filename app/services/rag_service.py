from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.settings import Settings
import qdrant_client
from app.config import settings
from loguru import logger

class RAGEngine:
    def __init__(self):
        Settings.llm = Ollama(
            model=settings.OLLAMA_MODEL, 
            base_url=settings.OLLAMA_BASE_URL, 
            request_timeout=300.0  # Increased to 5 minutes
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=settings.OLLAMA_EMBED_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            request_timeout=120.0
        )
        
        # Initialize Qdrant Client
        self.client = qdrant_client.QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        
        # Advanced Node Parser: Sentence Window for better context
        self.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        
        self.vector_store = QdrantVectorStore(
            client=self.client, 
            collection_name="andela_knowledge_base"
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = None

    def ingest_documents(self, directory_path: str):
        """Ingest documents from a directory using advanced chunking."""
        logger.info(f"Ingesting documents from {directory_path}")
        documents = SimpleDirectoryReader(directory_path).load_data()
        
        # Parse nodes using Sentence Window
        nodes = self.node_parser.get_nodes_from_documents(documents)
        
        self.index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
        )
        logger.info("Ingestion complete.")

    def get_query_engine(self):
        """Returns a query engine with metadata replacement for sentence window."""
        if not self.index:
            # Try to load existing index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )
            
        return self.index.as_query_engine(
            similarity_top_k=5,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ]
        )

rag_engine = RAGEngine()
