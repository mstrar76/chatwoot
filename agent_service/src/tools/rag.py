"""
RAG (Retrieval Augmented Generation) Tool for Chatwoot Agent MVP.

This tool provides production-ready RAG functionality with:
- LangChain tool integration using @tool decorator
- pgvector database connection and querying  
- OpenAI embedding generation (text-embedding-3-small)
- Contact-specific context retrieval with metadata filtering
- Relevance scoring and context window management
- Async operations for performance
- Comprehensive error handling and fallback strategies

Technical Requirements:
- Database: postgres://omniadmin:omni4518pgdb@omnineural_postgres:5432/omnicore
- Embeddings table: 1536-dimension vectors for OpenAI text-embedding-3-small
- Contact isolation: Filter by contact_phone for privacy
- Performance: <2s retrieval time, configurable top-K, similarity threshold
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field, validator
import tiktoken

from ..services.database import get_database_service, DatabaseError, QueryError
from ..models.schemas import RAGConfig
from ..utils.config import get_openai_config
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Custom exceptions for RAG operations
class RAGError(Exception):
    """Base exception for RAG operations."""
    pass

class EmbeddingError(RAGError):
    """Exception for embedding generation issues."""
    pass

class RetrievalError(RAGError):
    """Exception for document retrieval issues."""
    pass

class ConfigurationError(RAGError):
    """Exception for RAG configuration issues."""
    pass


class RAGQuery(BaseModel):
    """Input schema for RAG queries."""
    query: str = Field(..., description="The user query to search for relevant context")
    contact_phone: str = Field(..., description="Contact phone number to filter results")
    top_k: Optional[int] = Field(4, description="Number of similar documents to retrieve")
    similarity_threshold: Optional[float] = Field(0.7, description="Minimum similarity threshold")
    time_window_days: Optional[int] = Field(90, description="Limit search to documents within N days")
    include_metadata: bool = Field(True, description="Include metadata in results")
    
    @validator('query')
    def validate_query(cls, v):
        """Ensure query is not empty and within reasonable length."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > 2000:
            raise ValueError("Query too long (max 2000 characters)")
        return v.strip()
    
    @validator('contact_phone')
    def validate_contact_phone(cls, v):
        """Ensure contact phone is provided and valid format."""
        if not v or not v.strip():
            raise ValueError("Contact phone is required")
        # Basic phone validation - adjust regex as needed for your format
        import re
        phone_pattern = r'^[\+]?[1-9][\d\s\-\(\)]{8,15}$'
        if not re.match(phone_pattern, v.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')):
            logger.warning("Phone number format validation failed", contact_phone=v)
        return v.strip()
    
    @validator('top_k')
    def validate_top_k(cls, v):
        """Ensure top_k is within reasonable bounds."""
        if v is not None and (v < 1 or v > 20):
            raise ValueError("top_k must be between 1 and 20")
        return v
    
    @validator('similarity_threshold')  
    def validate_similarity_threshold(cls, v):
        """Ensure similarity threshold is valid."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        return v


class RAGResult(BaseModel):
    """Schema for RAG retrieval results."""
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    total_found: int = Field(0, description="Total number of documents found")
    retrieval_time_ms: int = Field(0, description="Time taken for retrieval in milliseconds")
    query_embedding_time_ms: int = Field(0, description="Time taken to generate query embedding")
    context_length: int = Field(0, description="Total character length of retrieved context")
    truncated: bool = Field(False, description="Whether results were truncated due to length")
    error: Optional[str] = Field(None, description="Error message if retrieval failed")
    
    def get_context_text(self, max_length: Optional[int] = None) -> str:
        """
        Combine all document content into a single context string.
        
        Args:
            max_length: Maximum length of context (characters)
            
        Returns:
            Combined context string
        """
        if not self.documents:
            return ""
        
        context_parts = []
        total_length = 0
        
        for doc in self.documents:
            content = doc.get('content_chunk', '')
            similarity = doc.get('similarity', 0)
            
            # Format with metadata for context
            formatted_content = f"[Similarity: {similarity:.3f}] {content}"
            
            if max_length and total_length + len(formatted_content) > max_length:
                # Truncate if needed
                remaining = max_length - total_length
                if remaining > 100:  # Only add if meaningful content fits
                    formatted_content = formatted_content[:remaining] + "..."
                    context_parts.append(formatted_content)
                self.truncated = True
                break
                
            context_parts.append(formatted_content)
            total_length += len(formatted_content)
        
        return "\n\n".join(context_parts)


class RAGService:
    """
    Production-ready RAG service with caching, performance monitoring, and error handling.
    
    Provides vector similarity search capabilities using pgvector and OpenAI embeddings
    with contact-specific filtering for privacy and security.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG service with configuration.
        
        Args:
            config: RAG configuration, uses defaults if None
        """
        self.config = config or RAGConfig()
        self._embeddings_client: Optional[OpenAIEmbeddings] = None
        self._tokenizer: Optional[tiktoken.Encoding] = None
        self._initialized = False
        
        # Performance metrics
        self._query_count = 0
        self._total_retrieval_time = 0.0
        self._cache_hits = 0
        self._embedding_cache: Dict[str, List[float]] = {}
        
        logger.info("RAG service initialized", config=self.config.dict())
    
    async def initialize(self) -> None:
        """Initialize the RAG service components."""
        if self._initialized:
            return
            
        try:
            # Initialize OpenAI embeddings client
            openai_config = get_openai_config()
            self._embeddings_client = OpenAIEmbeddings(
                openai_api_key=openai_config.api_key,
                model=self.config.embedding_model,
                openai_api_base=openai_config.base_url,
                openai_organization=openai_config.organization,
                timeout=openai_config.timeout_seconds,
                max_retries=openai_config.max_retries
            )
            
            # Initialize tokenizer for text processing
            self._tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            
            # Test database connection
            db = await get_database_service()
            await db.health_check()
            
            self._initialized = True
            logger.info("RAG service initialization completed successfully")
            
        except Exception as e:
            logger.error("Failed to initialize RAG service", error=str(e))
            raise ConfigurationError(f"RAG service initialization failed: {e}")
    
    def _get_cache_key(self, query: str, contact_phone: str) -> str:
        """Generate cache key for query embeddings."""
        import hashlib
        content = f"{query}:{contact_phone}:{self.config.embedding_model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _generate_embedding(
        self,
        text: str,
        cache_key: Optional[str] = None
    ) -> Tuple[List[float], int]:
        """
        Generate embedding for input text with caching.
        
        Args:
            text: Input text to embed
            cache_key: Optional cache key for caching embeddings
            
        Returns:
            Tuple of (embedding vector, generation time in ms)
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        # Check cache first
        if cache_key and cache_key in self._embedding_cache:
            self._cache_hits += 1
            embedding = self._embedding_cache[cache_key]
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.debug("Embedding cache hit", cache_key=cache_key, time_ms=elapsed_ms)
            return embedding, elapsed_ms
        
        try:
            # Generate embedding using OpenAI
            embedding = await self._embeddings_client.aembed_query(text)
            
            # Validate embedding dimensions
            if len(embedding) != 1536:
                raise EmbeddingError(
                    f"Invalid embedding dimensions: {len(embedding)}, expected 1536"
                )
            
            # Cache the result
            if cache_key:
                self._embedding_cache[cache_key] = embedding
                # Limit cache size to prevent memory issues
                if len(self._embedding_cache) > 1000:
                    # Remove oldest entries (simple FIFO)
                    oldest_key = next(iter(self._embedding_cache))
                    del self._embedding_cache[oldest_key]
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.debug("Embedding generated successfully", 
                        text_length=len(text),
                        dimensions=len(embedding),
                        time_ms=elapsed_ms)
            
            return embedding, elapsed_ms
            
        except Exception as e:
            logger.error("Failed to generate embedding", 
                        text_length=len(text), 
                        error=str(e))
            raise EmbeddingError(f"Embedding generation failed: {e}")
    
    def _chunk_text(self, text: str, max_tokens: int = 512) -> List[str]:
        """
        Split text into chunks that fit within token limits.
        
        Args:
            text: Input text to chunk
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        if not self._tokenizer:
            # Fallback to character-based chunking
            max_chars = max_tokens * 4  # Rough approximation
            return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
        
        tokens = self._tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i+max_tokens]
            chunk_text = self._tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    async def retrieve_context(
        self,
        query: str,
        contact_phone: str,
        top_k: int = 4,
        similarity_threshold: float = 0.7,
        time_window_days: Optional[int] = None,
        include_metadata: bool = True
    ) -> RAGResult:
        """
        Retrieve relevant context for a query using vector similarity search.
        
        Args:
            query: User query text
            contact_phone: Contact phone for filtering
            top_k: Number of similar documents to retrieve
            similarity_threshold: Minimum similarity threshold
            time_window_days: Limit search to recent documents
            include_metadata: Include metadata in results
            
        Returns:
            RAGResult with retrieved documents and metadata
            
        Raises:
            RetrievalError: If retrieval fails
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            rag_query = RAGQuery(
                query=query,
                contact_phone=contact_phone,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                time_window_days=time_window_days or self.config.time_window_days,
                include_metadata=include_metadata
            )
            
            # Generate query embedding
            cache_key = self._get_cache_key(query, contact_phone)
            query_embedding, embedding_time_ms = await self._generate_embedding(
                rag_query.query, cache_key
            )
            
            # Perform similarity search in database
            db = await get_database_service()
            retrieval_start = time.time()
            
            similar_docs = await db.similarity_search(
                query_embedding=query_embedding,
                contact_phone=rag_query.contact_phone,
                top_k=rag_query.top_k,
                similarity_threshold=rag_query.similarity_threshold,
                time_window_days=rag_query.time_window_days
            )
            
            retrieval_time_ms = int((time.time() - retrieval_start) * 1000)
            
            # Calculate context length
            context_length = sum(len(doc.get('content_chunk', '')) for doc in similar_docs)
            
            # Update performance metrics
            self._query_count += 1
            total_time = time.time() - start_time
            self._total_retrieval_time += total_time
            
            result = RAGResult(
                documents=similar_docs,
                total_found=len(similar_docs),
                retrieval_time_ms=retrieval_time_ms,
                query_embedding_time_ms=embedding_time_ms,
                context_length=context_length,
                truncated=False
            )
            
            logger.info("RAG context retrieval completed",
                       contact_phone=contact_phone,
                       query_length=len(query),
                       documents_found=len(similar_docs),
                       retrieval_time_ms=retrieval_time_ms,
                       total_time_ms=int(total_time * 1000),
                       context_length=context_length)
            
            return result
            
        except (DatabaseError, QueryError) as e:
            logger.error("Database error during RAG retrieval",
                        contact_phone=contact_phone,
                        error=str(e))
            return RAGResult(error=f"Database retrieval failed: {e}")
            
        except EmbeddingError as e:
            logger.error("Embedding error during RAG retrieval",
                        contact_phone=contact_phone,
                        error=str(e))  
            return RAGResult(error=f"Embedding generation failed: {e}")
            
        except Exception as e:
            logger.error("Unexpected error during RAG retrieval",
                        contact_phone=contact_phone,
                        error=str(e))
            return RAGResult(error=f"Retrieval failed: {e}")
    
    async def store_context(
        self,
        content: str,
        contact_phone: str,
        conversation_id: int,
        message_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store content with generated embeddings for future retrieval.
        
        Args:
            content: Text content to store and embed
            contact_phone: Contact phone number
            conversation_id: Chatwoot conversation ID
            message_type: Type of message (text, image, audio, file)
            metadata: Additional metadata to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        try:
            if not content or not content.strip():
                logger.warning("Attempted to store empty content", 
                              contact_phone=contact_phone)
                return False
            
            # Store the message first
            db = await get_database_service()
            message_id = await db.store_message(
                contact_phone=contact_phone,
                conversation_id=conversation_id,
                role="user",  # Assuming user message, adjust as needed
                content=content,
                message_type=message_type,
                metadata=metadata
            )
            
            # Generate embeddings for content chunks
            chunks = self._chunk_text(content, max_tokens=512)
            stored_embeddings = 0
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                    
                try:
                    # Generate embedding for chunk
                    embedding, _ = await self._generate_embedding(chunk)
                    
                    # Store embedding in database
                    embedding_id = await db.store_embedding(
                        message_id=message_id,
                        contact_phone=contact_phone,
                        embedding=embedding,
                        content_chunk=chunk,
                        chunk_index=i,
                        metadata={
                            **(metadata or {}),
                            "message_type": message_type,
                            "chunk_count": len(chunks)
                        }
                    )
                    
                    stored_embeddings += 1
                    logger.debug("Embedding stored successfully",
                               embedding_id=embedding_id,
                               chunk_index=i,
                               chunk_length=len(chunk))
                    
                except Exception as e:
                    logger.error("Failed to store embedding for chunk",
                               message_id=message_id,
                               chunk_index=i,
                               error=str(e))
                    continue
            
            logger.info("Content stored with embeddings",
                       message_id=message_id,
                       contact_phone=contact_phone,
                       chunks_processed=len(chunks),
                       embeddings_stored=stored_embeddings)
            
            return stored_embeddings > 0
            
        except Exception as e:
            logger.error("Failed to store content for RAG",
                        contact_phone=contact_phone,
                        conversation_id=conversation_id,
                        error=str(e))
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the RAG service.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_retrieval_time = (
            self._total_retrieval_time / self._query_count 
            if self._query_count > 0 else 0
        )
        
        cache_hit_rate = (
            self._cache_hits / self._query_count
            if self._query_count > 0 else 0
        )
        
        return {
            "total_queries": self._query_count,
            "average_retrieval_time_seconds": avg_retrieval_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._embedding_cache),
            "embedding_model": self.config.embedding_model,
            "top_k_default": self.config.top_k,
            "similarity_threshold_default": self.config.similarity_threshold
        }


# Global RAG service instance
_rag_service: Optional[RAGService] = None

async def get_rag_service() -> RAGService:
    """Get the global RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
        await _rag_service.initialize()
    return _rag_service


# LangChain Tool Implementation
@tool
async def retrieve_relevant_context(
    query: str,
    contact_phone: str, 
    top_k: int = 4,
    similarity_threshold: float = 0.7,
    max_context_length: int = 4000
) -> str:
    """
    Retrieve relevant conversation context using RAG (Retrieval Augmented Generation).
    
    This tool searches through the contact's conversation history using vector similarity 
    to find the most relevant messages and context for answering their current query.
    It uses OpenAI embeddings and pgvector for efficient semantic search.
    
    Args:
        query: The user's current query or message to find relevant context for
        contact_phone: The contact's phone number to filter results (required for privacy)
        top_k: Number of most similar messages to retrieve (1-20, default: 4)
        similarity_threshold: Minimum similarity score to include results (0.0-1.0, default: 0.7)
        max_context_length: Maximum total character length of returned context (default: 4000)
        
    Returns:
        Formatted string with relevant conversation context, or error message if retrieval fails.
        Context includes similarity scores and is formatted for LLM consumption.
        
    Example:
        context = await retrieve_relevant_context(
            query="What was my last order status?",
            contact_phone="+5511999999999",
            top_k=3
        )
    """
    try:
        # Input validation
        if not query or not query.strip():
            return "Error: Query cannot be empty"
        
        if not contact_phone or not contact_phone.strip():
            return "Error: Contact phone number is required"
        
        # Get RAG service and perform retrieval
        rag_service = await get_rag_service()
        result = await rag_service.retrieve_context(
            query=query.strip(),
            contact_phone=contact_phone.strip(),
            top_k=max(1, min(20, top_k)),  # Clamp to valid range
            similarity_threshold=max(0.0, min(1.0, similarity_threshold))
        )
        
        # Handle retrieval errors
        if result.error:
            logger.error("RAG retrieval failed", 
                        contact_phone=contact_phone,
                        error=result.error)
            return f"Context retrieval failed: {result.error}"
        
        # Handle no results found
        if not result.documents:
            logger.info("No relevant context found", 
                       contact_phone=contact_phone,
                       query=query[:100])
            return "No relevant conversation context found for this query."
        
        # Generate formatted context
        context_text = result.get_context_text(max_length=max_context_length)
        
        # Add metadata to help the agent understand the context quality
        metadata_info = []
        if result.total_found > 0:
            metadata_info.append(f"Found {result.total_found} relevant messages")
        if result.truncated:
            metadata_info.append("Results truncated due to length limits")
        if result.retrieval_time_ms > 2000:
            metadata_info.append("Note: Retrieval took longer than expected")
        
        # Format final response
        response_parts = []
        if metadata_info:
            response_parts.append(f"[RAG Context: {', '.join(metadata_info)}]")
        response_parts.append(context_text)
        
        final_context = "\n\n".join(response_parts)
        
        logger.info("RAG context retrieved successfully",
                   contact_phone=contact_phone,
                   documents_found=result.total_found,
                   context_length=len(final_context),
                   retrieval_time_ms=result.retrieval_time_ms)
        
        return final_context
        
    except Exception as e:
        logger.error("Unexpected error in RAG tool",
                    contact_phone=contact_phone,
                    query=query[:100] if query else "None",
                    error=str(e))
        return f"Error retrieving context: {str(e)}"


@tool
async def store_conversation_context(
    content: str,
    contact_phone: str,
    conversation_id: int,
    message_type: str = "text"
) -> str:
    """
    Store conversation content for future RAG retrieval.
    
    This tool processes and stores conversation content with generated embeddings
    for future semantic search and context retrieval. Use this to build up
    conversation history that can be searched later.
    
    Args:
        content: The message or content to store
        contact_phone: Contact's phone number for privacy filtering
        conversation_id: Chatwoot conversation ID
        message_type: Type of content (text, image, audio, file)
        
    Returns:
        Success or error message about the storage operation
        
    Example:
        result = await store_conversation_context(
            content="Customer ordered 3 pizzas for delivery to Main St",
            contact_phone="+5511999999999", 
            conversation_id=12345
        )
    """
    try:
        if not content or not content.strip():
            return "Error: Content cannot be empty"
        
        if not contact_phone or not contact_phone.strip():
            return "Error: Contact phone number is required"
        
        if not isinstance(conversation_id, int) or conversation_id <= 0:
            return "Error: Valid conversation ID is required"
        
        # Get RAG service and store content
        rag_service = await get_rag_service()
        success = await rag_service.store_context(
            content=content.strip(),
            contact_phone=contact_phone.strip(),
            conversation_id=conversation_id,
            message_type=message_type,
            metadata={
                "stored_at": datetime.utcnow().isoformat(),
                "tool": "rag_store"
            }
        )
        
        if success:
            logger.info("Content stored for RAG",
                       contact_phone=contact_phone,
                       conversation_id=conversation_id,
                       content_length=len(content))
            return f"Successfully stored {len(content)} characters of content for future retrieval"
        else:
            logger.warning("Failed to store content for RAG",
                          contact_phone=contact_phone,
                          conversation_id=conversation_id)
            return "Failed to store content - please try again"
            
    except Exception as e:
        logger.error("Error storing conversation context",
                    contact_phone=contact_phone,
                    conversation_id=conversation_id,
                    error=str(e))
        return f"Error storing context: {str(e)}"


@tool
async def get_rag_performance_stats() -> str:
    """
    Get RAG system performance statistics and health information.
    
    Returns performance metrics, cache statistics, and configuration information
    for monitoring and debugging the RAG system.
    
    Returns:
        Formatted string with RAG performance statistics
        
    Example:
        stats = await get_rag_performance_stats()
        print(stats)  # Shows query count, cache hit rate, etc.
    """
    try:
        rag_service = await get_rag_service()
        stats = rag_service.get_performance_stats()
        
        formatted_stats = [
            f"RAG Performance Statistics:",
            f"- Total queries processed: {stats['total_queries']}",
            f"- Average retrieval time: {stats['average_retrieval_time_seconds']:.3f}s",
            f"- Cache hit rate: {stats['cache_hit_rate']:.2%}",
            f"- Cache size: {stats['cache_size']} entries",
            f"- Embedding model: {stats['embedding_model']}",
            f"- Default top-K: {stats['top_k_default']}",
            f"- Default similarity threshold: {stats['similarity_threshold_default']}"
        ]
        
        return "\n".join(formatted_stats)
        
    except Exception as e:
        logger.error("Error getting RAG performance stats", error=str(e))
        return f"Error retrieving performance stats: {str(e)}"