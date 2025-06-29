"""
PriceVision RAG System using LlamaIndex
RTX 3050 optimized Retrieval-Augmented Generation system
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
import json
import os
from datetime import datetime
from contextlib import asynccontextmanager

try:
    from llama_index.core import VectorStoreIndex, Document, Settings
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.llms.huggingface import HuggingFaceLLM
except ImportError:
    # Fallback for development without LlamaIndex
    logging.warning("LlamaIndex not available, using mock implementations")

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import config
from .database import get_chroma_client, get_mysql_session
from .memory_manager import memory_manager, ModelPriority

logger = logging.getLogger(__name__)

class RTX3050RAGSystem:
    """
    RAG system optimized for RTX 3050 (4GB VRAM)
    Uses lightweight models and efficient memory management
    """
    
    def __init__(self):
        self.embedding_model = None
        self.llm_model = None
        self.vector_store = None
        self.index = None
        self.query_engine = None
        self.chroma_client = None
        self.collection_name = "pricevision_knowledge"
        
        # RTX 3050 optimizations
        self.max_chunk_size = 512  # Smaller chunks for memory efficiency
        self.max_context_length = 1024  # Reduced context window
        self.embedding_batch_size = 8  # Small batch size
        
    async def initialize(self):
        """Initialize the RAG system with RTX 3050 optimizations"""
        try:
            logger.info("Initializing RTX 3050 optimized RAG system...")
            
            # Initialize ChromaDB client
            self.chroma_client = await get_chroma_client()
            
            # Setup embedding model (lightweight)
            await self._setup_embedding_model()
            
            # Setup LLM (CPU-based for RTX 3050)
            await self._setup_llm_model()
            
            # Setup vector store
            await self._setup_vector_store()
            
            # Setup query engine
            await self._setup_query_engine()
            
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    async def _setup_embedding_model(self):
        """Setup lightweight embedding model for RTX 3050"""
        try:
            # Use sentence-transformers with RTX 3050 optimization
            model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 22MB model
            
            async with memory_manager.load_model_context(
                "embedding", 
                ModelPriority.HIGH,
                estimated_memory_mb=100
            ) as model_context:
                
                if 'llama_index' in globals():
                    self.embedding_model = HuggingFaceEmbedding(
                        model_name=model_name,
                        device="cuda" if config.rtx3050.gpu_available else "cpu",
                        max_length=512,
                        normalize=True
                    )
                    
                    # Configure LlamaIndex settings
                    Settings.embed_model = self.embedding_model
                    Settings.chunk_size = self.max_chunk_size
                    Settings.chunk_overlap = 50
                else:
                    # Mock embedding model for development
                    self.embedding_model = MockEmbeddingModel()
                
                logger.info(f"Embedding model loaded: {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to setup embedding model: {e}")
            # Fallback to mock model
            self.embedding_model = MockEmbeddingModel()
    
    async def _setup_llm_model(self):
        """Setup lightweight LLM model (CPU-based for RTX 3050)"""
        try:
            # Use CPU-based model to preserve GPU memory for embeddings
            model_name = "microsoft/DialoGPT-small"  # 117MB model
            
            if 'llama_index' in globals():
                self.llm_model = HuggingFaceLLM(
                    model_name=model_name,
                    device_map="cpu",  # Force CPU to save GPU memory
                    max_new_tokens=256,
                    generate_kwargs={
                        "do_sample": True,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "pad_token_id": 50256
                    }
                )
                
                Settings.llm = self.llm_model
            else:
                # Mock LLM for development
                self.llm_model = MockLLMModel()
            
            logger.info(f"LLM model loaded: {model_name} (CPU)")
            
        except Exception as e:
            logger.error(f"Failed to setup LLM model: {e}")
            self.llm_model = MockLLMModel()
    
    async def _setup_vector_store(self):
        """Setup ChromaDB vector store"""
        try:
            # Get or create collection
            collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PriceVision game pricing knowledge base"}
            )
            
            if 'llama_index' in globals():
                self.vector_store = ChromaVectorStore(chroma_collection=collection)
            else:
                self.vector_store = MockVectorStore()
            
            logger.info("Vector store initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup vector store: {e}")
            self.vector_store = MockVectorStore()
    
    async def _setup_query_engine(self):
        """Setup query engine with RTX 3050 optimizations"""
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            if 'llama_index' in globals():
                # Create storage context
                storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store
                )
                
                # Create or load index
                try:
                    self.index = VectorStoreIndex.from_vector_store(
                        vector_store=self.vector_store,
                        storage_context=storage_context
                    )
                except:
                    # Create new index if none exists
                    self.index = VectorStoreIndex(
                        nodes=[],
                        storage_context=storage_context
                    )
                
                # Create retriever with RTX 3050 optimizations
                retriever = VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=5,  # Reduced for memory efficiency
                )
                
                # Create query engine
                self.query_engine = RetrieverQueryEngine(
                    retriever=retriever,
                    node_postprocessors=[
                        SimilarityPostprocessor(similarity_cutoff=0.7)
                    ]
                )
            else:
                self.query_engine = MockQueryEngine()
            
            logger.info("Query engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup query engine: {e}")
            self.query_engine = MockQueryEngine()
    
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the knowledge base"""
        try:
            if not self.index:
                logger.error("Index not initialized")
                return False
            
            # Convert documents to LlamaIndex format
            llama_docs = []
            for doc in documents:
                llama_doc = Document(
                    text=doc.get("content", ""),
                    metadata={
                        "title": doc.get("title", ""),
                        "source": doc.get("source", ""),
                        "category": doc.get("category", ""),
                        "timestamp": datetime.now().isoformat()
                    }
                )
                llama_docs.append(llama_doc)
            
            # Process documents in batches for memory efficiency
            batch_size = 10
            for i in range(0, len(llama_docs), batch_size):
                batch = llama_docs[i:i + batch_size]
                
                if 'llama_index' in globals():
                    # Parse nodes
                    parser = SimpleNodeParser.from_defaults(
                        chunk_size=self.max_chunk_size,
                        chunk_overlap=50
                    )
                    nodes = parser.get_nodes_from_documents(batch)
                    
                    # Add to index
                    self.index.insert_nodes(nodes)
                
                logger.info(f"Added batch {i//batch_size + 1} of documents")
                
                # Memory cleanup between batches
                await memory_manager.cleanup_if_needed()
            
            logger.info(f"Added {len(documents)} documents to knowledge base")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    async def query(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            if not self.query_engine:
                return {
                    "answer": "RAG system not initialized",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Enhance query with context
            enhanced_query = self._enhance_query(question, context)
            
            # Execute query with memory management
            async with memory_manager.load_model_context(
                "rag_query",
                ModelPriority.HIGH,
                estimated_memory_mb=200
            ):
                if 'llama_index' in globals():
                    response = self.query_engine.query(enhanced_query)
                    
                    result = {
                        "answer": str(response),
                        "sources": self._extract_sources(response),
                        "confidence": self._calculate_confidence(response),
                        "query_time": datetime.now().isoformat()
                    }
                else:
                    # Mock response for development
                    result = {
                        "answer": f"Mock response for: {enhanced_query}",
                        "sources": ["mock_source_1", "mock_source_2"],
                        "confidence": 0.8,
                        "query_time": datetime.now().isoformat()
                    }
            
            logger.info(f"RAG query completed: {question[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                "answer": "Sorry, I couldn't process your question right now.",
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _enhance_query(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Enhance query with context information"""
        enhanced = question
        
        if context:
            if context.get("game_title"):
                enhanced = f"Game: {context['game_title']}. {enhanced}"
            if context.get("platform"):
                enhanced = f"Platform: {context['platform']}. {enhanced}"
            if context.get("condition"):
                enhanced = f"Condition: {context['condition']}. {enhanced}"
        
        return enhanced
    
    def _extract_sources(self, response) -> List[str]:
        """Extract source information from response"""
        try:
            if hasattr(response, 'source_nodes'):
                sources = []
                for node in response.source_nodes:
                    if hasattr(node, 'metadata'):
                        source = node.metadata.get('source', 'Unknown')
                        if source not in sources:
                            sources.append(source)
                return sources
        except:
            pass
        
        return ["knowledge_base"]
    
    def _calculate_confidence(self, response) -> float:
        """Calculate confidence score for response"""
        try:
            if hasattr(response, 'source_nodes') and response.source_nodes:
                # Average similarity scores
                scores = []
                for node in response.source_nodes:
                    if hasattr(node, 'score') and node.score:
                        scores.append(node.score)
                
                if scores:
                    return sum(scores) / len(scores)
            
            return 0.7  # Default confidence
            
        except:
            return 0.5
    
    async def load_game_knowledge(self):
        """Load game pricing knowledge from database"""
        try:
            async with get_mysql_session() as session:
                # Query game data from database
                # This would be implemented with actual SQLAlchemy queries
                
                # Mock data for now
                game_docs = [
                    {
                        "title": "Game Pricing Guidelines",
                        "content": "Game pricing depends on platform, condition, rarity, and market demand. "
                                 "Mint condition games typically sell for 20% above average market price. "
                                 "Excellent condition games sell at market price. "
                                 "Good condition games sell for 20% below market price.",
                        "source": "pricing_guidelines",
                        "category": "pricing"
                    },
                    {
                        "title": "Platform Value Factors",
                        "content": "Nintendo games typically hold value better than other platforms. "
                                 "Limited editions and collector's items command premium prices. "
                                 "Digital-only releases have lower resale value.",
                        "source": "platform_analysis",
                        "category": "market_trends"
                    },
                    {
                        "title": "Seasonal Pricing Trends",
                        "content": "Game prices typically increase during holiday seasons. "
                                 "Summer months see lower demand for most games. "
                                 "New releases affect prices of older games in the same series.",
                        "source": "seasonal_analysis",
                        "category": "market_trends"
                    }
                ]
                
                await self.add_documents(game_docs)
                logger.info("Game knowledge loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load game knowledge: {e}")
    
    async def update_knowledge(self, new_data: List[Dict[str, Any]]):
        """Update knowledge base with new information"""
        try:
            await self.add_documents(new_data)
            logger.info(f"Knowledge base updated with {len(new_data)} new documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update knowledge: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear models from memory
            self.embedding_model = None
            self.llm_model = None
            self.query_engine = None
            
            # Force garbage collection
            await memory_manager.cleanup_if_needed()
            
            logger.info("RAG system cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Mock classes for development without LlamaIndex
class MockEmbeddingModel:
    def __init__(self):
        self.model_name = "mock_embedding_model"
    
    def embed_query(self, text: str) -> List[float]:
        # Return mock embedding
        return [0.1] * 384

class MockLLMModel:
    def __init__(self):
        self.model_name = "mock_llm_model"
    
    def complete(self, prompt: str) -> str:
        return f"Mock response to: {prompt[:50]}..."

class MockVectorStore:
    def __init__(self):
        self.documents = []
    
    def add_documents(self, docs):
        self.documents.extend(docs)

class MockQueryEngine:
    def __init__(self):
        self.responses = [
            "Based on current market data, this game typically sells for $35-45.",
            "The condition significantly affects the price. Mint condition adds 20% value.",
            "This platform generally holds value well in the resale market.",
        ]
    
    def query(self, question: str):
        import random
        return random.choice(self.responses)

# Global RAG system instance
rag_system = RTX3050RAGSystem()

async def initialize_rag_system():
    """Initialize the global RAG system"""
    await rag_system.initialize()
    await rag_system.load_game_knowledge()

async def query_rag_system(question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Query the RAG system"""
    return await rag_system.query(question, context)

async def update_rag_knowledge(new_data: List[Dict[str, Any]]) -> bool:
    """Update RAG system knowledge"""
    return await rag_system.update_knowledge(new_data)

# Context manager for RAG operations
@asynccontextmanager
async def rag_context():
    """Context manager for RAG operations with memory management"""
    try:
        if not rag_system.query_engine:
            await initialize_rag_system()
        yield rag_system
    finally:
        # Cleanup if needed
        await memory_manager.cleanup_if_needed()