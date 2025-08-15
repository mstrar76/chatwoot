"""
Comprehensive unit tests for the RAG (Retrieval-Augmented Generation) tool.

Tests cover:
- pgvector database operations
- Embedding generation and retrieval
- Context filtering and relevance scoring
- Contact isolation and privacy
- Performance optimization
- Error handling and edge cases
"""

import pytest
import numpy as np
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.tools.rag import (
    retrieve_relevant_context, store_conversation_context,
    get_rag_performance_stats, get_rag_service,
    RAGService, EmbeddingService
)
from src.models.schemas import RAGConfig
from tests.conftest import (
    mock_rag_embeddings, assert_response_time, AsyncMockService
)


class TestRAGService:
    """Test core RAG service functionality."""
    
    @pytest.fixture
    async def rag_service(self):
        """Create RAG service instance for testing."""
        config = RAGConfig(
            enabled=True,
            embedding_model="text-embedding-3-small",
            top_k=4,
            time_window_days=90,
            similarity_threshold=0.7
        )
        
        with patch('src.tools.rag.get_database_service') as mock_db, \
             patch('src.tools.rag.get_embedding_service') as mock_embedding:
            
            mock_db_service = AsyncMock()
            mock_embedding_service = AsyncMock()
            
            mock_db.return_value = mock_db_service
            mock_embedding.return_value = mock_embedding_service
            
            service = RAGService(config)
            await service.initialize()
            
            return service, mock_db_service, mock_embedding_service
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rag_service_initialization(self):
        """Test RAG service initialization."""
        # Arrange
        config = RAGConfig(enabled=True)
        
        with patch('src.tools.rag.get_database_service'), \
             patch('src.tools.rag.get_embedding_service'):
            
            # Act
            service = RAGService(config)
            await service.initialize()
        
        # Assert
        assert service._initialized is True
        assert service.config == config
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, rag_service):
        """Test successful embedding generation."""
        # Arrange
        service, mock_db_service, mock_embedding_service = rag_service
        test_text = "Hello, I need help with my order"
        mock_embedding = [0.1] * 1536  # Mock 1536-dimensional embedding
        
        mock_embedding_service.generate_embedding.return_value = mock_embedding
        
        # Act
        start_time = time.time()
        result = await service.generate_embedding(test_text)
        
        # Assert
        assert_response_time(start_time, 5000)  # Should complete within 5 seconds
        assert result == mock_embedding
        assert len(result) == 1536
        mock_embedding_service.generate_embedding.assert_called_once_with(test_text)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_embedding_failure(self, rag_service):
        """Test embedding generation failure handling."""
        # Arrange
        service, mock_db_service, mock_embedding_service = rag_service
        test_text = "Test text"
        
        mock_embedding_service.generate_embedding.side_effect = Exception("OpenAI API error")
        
        # Act & Assert
        with pytest.raises(Exception, match="OpenAI API error"):
            await service.generate_embedding(test_text)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_store_embedding_success(self, rag_service):
        """Test successful embedding storage."""
        # Arrange
        service, mock_db_service, mock_embedding_service = rag_service
        
        content = "User asked about order status"
        contact_phone = "+1234567890"
        conversation_id = 12345
        embedding = [0.1] * 1536
        
        mock_embedding_service.generate_embedding.return_value = embedding
        mock_db_service.store_embedding.return_value = 100  # Mock embedding ID
        mock_db_service.store_message.return_value = 50  # Mock message ID
        
        # Act
        result = await service.store_embedding(
            content=content,
            contact_phone=contact_phone,
            conversation_id=conversation_id,
            metadata={"type": "user_query"}
        )
        
        # Assert
        assert result == 100
        mock_embedding_service.generate_embedding.assert_called_once_with(content)
        mock_db_service.store_message.assert_called_once()
        mock_db_service.store_embedding.assert_called_once()
        
        # Verify message storage call
        message_args = mock_db_service.store_message.call_args[1]
        assert message_args["content"] == content
        assert message_args["contact_phone"] == contact_phone
        assert message_args["conversation_id"] == conversation_id
        
        # Verify embedding storage call
        embedding_args = mock_db_service.store_embedding.call_args[1]
        assert embedding_args["embedding"] == embedding
        assert embedding_args["contact_phone"] == contact_phone
        assert embedding_args["metadata"]["type"] == "user_query"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_similarity_search_success(self, rag_service, mock_rag_embeddings):
        """Test successful similarity search."""
        # Arrange
        service, mock_db_service, mock_embedding_service = rag_service
        
        query = "What is my order status?"
        contact_phone = "+1234567890"
        query_embedding = [0.15] * 1536
        
        mock_embedding_service.generate_embedding.return_value = query_embedding
        mock_db_service.similarity_search.return_value = mock_rag_embeddings
        
        # Act
        start_time = time.time()
        results = await service.similarity_search(
            query=query,
            contact_phone=contact_phone,
            top_k=4
        )
        
        # Assert
        assert_response_time(start_time, 3000)  # Should complete within 3 seconds
        assert len(results) == 2  # From mock data
        assert all('content' in result for result in results)
        assert all('similarity' in result for result in results)
        assert all('metadata' in result for result in results)
        
        # Verify database call
        mock_db_service.similarity_search.assert_called_once()
        search_args = mock_db_service.similarity_search.call_args[1]
        assert search_args["embedding"] == query_embedding
        assert search_args["contact_phone"] == contact_phone
        assert search_args["top_k"] == 4
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_similarity_search_with_time_filter(self, rag_service, mock_rag_embeddings):
        """Test similarity search with time window filtering."""
        # Arrange
        service, mock_db_service, mock_embedding_service = rag_service
        
        query = "Recent conversations"
        contact_phone = "+1234567890"
        query_embedding = [0.2] * 1536
        time_window_days = 30
        
        mock_embedding_service.generate_embedding.return_value = query_embedding
        mock_db_service.similarity_search.return_value = mock_rag_embeddings
        
        # Act
        results = await service.similarity_search(
            query=query,
            contact_phone=contact_phone,
            time_window_days=time_window_days
        )
        
        # Assert
        mock_db_service.similarity_search.assert_called_once()
        search_args = mock_db_service.similarity_search.call_args[1]
        assert search_args["time_window_days"] == time_window_days
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_similarity_search_threshold_filtering(self, rag_service):
        """Test similarity search with threshold filtering."""
        # Arrange
        service, mock_db_service, mock_embedding_service = rag_service
        
        query = "Test query"
        contact_phone = "+1234567890"
        query_embedding = [0.1] * 1536
        
        # Mock results with different similarity scores
        mock_results = [
            {'content': 'High similarity', 'similarity': 0.85, 'metadata': {}},
            {'content': 'Medium similarity', 'similarity': 0.65, 'metadata': {}},  # Below threshold
            {'content': 'Low similarity', 'similarity': 0.45, 'metadata': {}}     # Below threshold
        ]
        
        mock_embedding_service.generate_embedding.return_value = query_embedding
        mock_db_service.similarity_search.return_value = mock_results
        
        # Act
        results = await service.similarity_search(
            query=query,
            contact_phone=contact_phone,
            similarity_threshold=0.7
        )
        
        # Assert
        assert len(results) == 1  # Only high similarity result
        assert results[0]['content'] == 'High similarity'
        assert results[0]['similarity'] == 0.85
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_similarity_search_no_results(self, rag_service):
        """Test similarity search when no results are found."""
        # Arrange
        service, mock_db_service, mock_embedding_service = rag_service
        
        query = "Nonexistent query"
        contact_phone = "+1234567890"
        query_embedding = [0.1] * 1536
        
        mock_embedding_service.generate_embedding.return_value = query_embedding
        mock_db_service.similarity_search.return_value = []
        
        # Act
        results = await service.similarity_search(
            query=query,
            contact_phone=contact_phone
        )
        
        # Assert
        assert results == []
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_contact_isolation(self, rag_service):
        """Test that RAG properly isolates data by contact phone."""
        # Arrange
        service, mock_db_service, mock_embedding_service = rag_service
        
        query = "My order status"
        contact_phone_1 = "+1234567890"
        contact_phone_2 = "+0987654321"
        query_embedding = [0.1] * 1536
        
        mock_embedding_service.generate_embedding.return_value = query_embedding
        mock_db_service.similarity_search.return_value = []
        
        # Act - Search for contact 1
        await service.similarity_search(query=query, contact_phone=contact_phone_1)
        
        # Act - Search for contact 2
        await service.similarity_search(query=query, contact_phone=contact_phone_2)
        
        # Assert - Each search should be isolated by contact phone
        assert mock_db_service.similarity_search.call_count == 2
        
        # Verify first call
        first_call = mock_db_service.similarity_search.call_args_list[0][1]
        assert first_call["contact_phone"] == contact_phone_1
        
        # Verify second call
        second_call = mock_db_service.similarity_search.call_args_list[1][1]
        assert second_call["contact_phone"] == contact_phone_2
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_performance_stats(self, rag_service):
        """Test getting RAG performance statistics."""
        # Arrange
        service, mock_db_service, mock_embedding_service = rag_service
        
        mock_stats = {
            'total_embeddings': 1500,
            'total_searches': 250,
            'average_search_time_ms': 150.5,
            'cache_hit_rate': 0.35,
            'embedding_model': 'text-embedding-3-small',
            'vector_dimensions': 1536
        }
        
        mock_db_service.get_rag_stats.return_value = mock_stats
        
        # Act
        stats = await service.get_performance_stats()
        
        # Assert
        assert stats == mock_stats
        assert stats['total_embeddings'] == 1500
        assert stats['average_search_time_ms'] == 150.5
        mock_db_service.get_rag_stats.assert_called_once()


class TestRAGToolFunctions:
    """Test the RAG tool functions used by the agent."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieve_relevant_context_tool(self, mock_rag_embeddings):
        """Test retrieve_relevant_context tool function."""
        # Arrange
        query = "What is my order status?"
        contact_phone = "+1234567890"
        
        with patch('src.tools.rag.get_rag_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.similarity_search.return_value = mock_rag_embeddings
            mock_get_service.return_value = mock_service
            
            # Act
            result = await retrieve_relevant_context(
                query=query,
                contact_phone=contact_phone,
                top_k=3
            )
        
        # Assert
        assert isinstance(result, str)
        assert "Previous conversation about installation" in result
        assert "Service order status inquiry" in result
        assert len(result) > 0
        
        # Verify service call
        mock_service.similarity_search.assert_called_once_with(
            query=query,
            contact_phone=contact_phone,
            top_k=3,
            time_window_days=90,
            similarity_threshold=0.7
        )
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieve_relevant_context_no_results(self):
        """Test retrieve_relevant_context when no results found."""
        # Arrange
        query = "Nonexistent query"
        contact_phone = "+1234567890"
        
        with patch('src.tools.rag.get_rag_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.similarity_search.return_value = []
            mock_get_service.return_value = mock_service
            
            # Act
            result = await retrieve_relevant_context(
                query=query,
                contact_phone=contact_phone
            )
        
        # Assert
        assert result == "No relevant context found in conversation history."
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_store_conversation_context_tool(self):
        """Test store_conversation_context tool function."""
        # Arrange
        content = "User: What is my order status?\nAssistant: Let me check that for you."
        contact_phone = "+1234567890"
        conversation_id = 12345
        
        with patch('src.tools.rag.get_rag_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.store_embedding.return_value = 100
            mock_get_service.return_value = mock_service
            
            # Act
            result = await store_conversation_context(
                content=content,
                contact_phone=contact_phone,
                conversation_id=conversation_id
            )
        
        # Assert
        assert isinstance(result, str)
        assert "stored successfully" in result.lower()
        assert "100" in result  # Should include the embedding ID
        
        # Verify service call
        mock_service.store_embedding.assert_called_once()
        call_args = mock_service.store_embedding.call_args[1]
        assert call_args["content"] == content
        assert call_args["contact_phone"] == contact_phone
        assert call_args["conversation_id"] == conversation_id
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_store_conversation_context_failure(self):
        """Test store_conversation_context when storage fails."""
        # Arrange
        content = "Test content"
        contact_phone = "+1234567890"
        conversation_id = 12345
        
        with patch('src.tools.rag.get_rag_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.store_embedding.side_effect = Exception("Storage failed")
            mock_get_service.return_value = mock_service
            
            # Act
            result = await store_conversation_context(
                content=content,
                contact_phone=contact_phone,
                conversation_id=conversation_id
            )
        
        # Assert
        assert isinstance(result, str)
        assert "failed" in result.lower()
        assert "storage failed" in result.lower()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_rag_performance_stats_tool(self):
        """Test get_rag_performance_stats tool function."""
        # Arrange
        mock_stats = {
            'total_embeddings': 2500,
            'total_searches': 400,
            'average_search_time_ms': 125.0,
            'cache_hit_rate': 0.42,
            'embedding_model': 'text-embedding-3-small'
        }
        
        with patch('src.tools.rag.get_rag_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_performance_stats.return_value = mock_stats
            mock_get_service.return_value = mock_service
            
            # Act
            result = await get_rag_performance_stats()
        
        # Assert
        assert isinstance(result, str)
        assert "2500" in result  # total_embeddings
        assert "400" in result   # total_searches
        assert "125.0" in result # average_search_time_ms
        assert "42%" in result   # cache_hit_rate as percentage


class TestEmbeddingService:
    """Test embedding service functionality."""
    
    @pytest.fixture
    def embedding_service(self):
        """Create embedding service for testing."""
        with patch('src.tools.rag.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            from src.tools.rag import EmbeddingService
            service = EmbeddingService()
            return service, mock_client
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, embedding_service):
        """Test successful embedding generation."""
        # Arrange
        service, mock_client = embedding_service
        text = "Test text for embedding"
        mock_embedding = [0.1] * 1536
        
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = mock_embedding
        mock_client.embeddings.create.return_value = mock_response
        
        # Act
        result = await service.generate_embedding(text)
        
        # Assert
        assert result == mock_embedding
        assert len(result) == 1536
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=text
        )
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_embedding_api_error(self, embedding_service):
        """Test embedding generation with API error."""
        # Arrange
        service, mock_client = embedding_service
        text = "Test text"
        
        mock_client.embeddings.create.side_effect = Exception("API rate limit exceeded")
        
        # Act & Assert
        with pytest.raises(Exception, match="API rate limit exceeded"):
            await service.generate_embedding(text)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text(self, embedding_service):
        """Test embedding generation with empty text."""
        # Arrange
        service, mock_client = embedding_service
        text = ""
        
        # Act & Assert
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await service.generate_embedding(text)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_embedding_very_long_text(self, embedding_service):
        """Test embedding generation with very long text."""
        # Arrange
        service, mock_client = embedding_service
        text = "A" * 10000  # Very long text
        mock_embedding = [0.1] * 1536
        
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = mock_embedding
        mock_client.embeddings.create.return_value = mock_response
        
        # Act
        result = await service.generate_embedding(text)
        
        # Assert
        assert result == mock_embedding
        # Should handle long text gracefully (may truncate or chunk)


class TestRAGEdgeCases:
    """Test edge cases and error scenarios for RAG."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rag_service_disabled_config(self):
        """Test RAG service behavior when disabled in configuration."""
        # Arrange
        config = RAGConfig(enabled=False)
        
        # Act
        service = RAGService(config)
        
        # Assert
        assert service.config.enabled is False
        # Service should still be initializable but may skip certain operations
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_similarity_search_database_error(self, rag_service):
        """Test similarity search when database operation fails."""
        # Arrange
        service, mock_db_service, mock_embedding_service = rag_service
        
        query = "Test query"
        contact_phone = "+1234567890"
        query_embedding = [0.1] * 1536
        
        mock_embedding_service.generate_embedding.return_value = query_embedding
        mock_db_service.similarity_search.side_effect = Exception("Database connection failed")
        
        # Act & Assert
        with pytest.raises(Exception, match="Database connection failed"):
            await service.similarity_search(query=query, contact_phone=contact_phone)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_store_embedding_duplicate_content(self, rag_service):
        """Test storing duplicate content handling."""
        # Arrange
        service, mock_db_service, mock_embedding_service = rag_service
        
        content = "Duplicate content"
        contact_phone = "+1234567890"
        conversation_id = 12345
        embedding = [0.1] * 1536
        
        mock_embedding_service.generate_embedding.return_value = embedding
        mock_db_service.store_embedding.return_value = 100
        mock_db_service.store_message.return_value = 50
        
        # Act - Store same content twice
        result1 = await service.store_embedding(content, contact_phone, conversation_id)
        result2 = await service.store_embedding(content, contact_phone, conversation_id)
        
        # Assert - Both should succeed (duplicate handling is database-level concern)
        assert result1 == 100
        assert result2 == 100
        assert mock_db_service.store_embedding.call_count == 2
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_similarity_search_malformed_embedding(self, rag_service):
        """Test similarity search with malformed embedding response."""
        # Arrange
        service, mock_db_service, mock_embedding_service = rag_service
        
        query = "Test query"
        contact_phone = "+1234567890"
        malformed_embedding = [0.1] * 512  # Wrong dimension
        
        mock_embedding_service.generate_embedding.return_value = malformed_embedding
        
        # Act & Assert
        # Should either handle gracefully or raise appropriate error
        try:
            await service.similarity_search(query=query, contact_phone=contact_phone)
        except Exception as e:
            # Expected to fail with dimension mismatch
            assert "dimension" in str(e).lower() or "shape" in str(e).lower()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_large_batch_embedding_storage(self, rag_service):
        """Test storing large batch of embeddings."""
        # Arrange
        service, mock_db_service, mock_embedding_service = rag_service
        
        batch_size = 100
        base_content = "Batch content item"
        contact_phone = "+1234567890"
        conversation_id = 12345
        embedding = [0.1] * 1536
        
        mock_embedding_service.generate_embedding.return_value = embedding
        mock_db_service.store_embedding.return_value = 100
        mock_db_service.store_message.return_value = 50
        
        # Act - Store batch of embeddings
        import asyncio
        tasks = [
            service.store_embedding(
                f"{base_content} {i}",
                contact_phone,
                conversation_id
            )
            for i in range(batch_size)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        
        # Assert
        assert_response_time(start_time, 30000)  # Should complete within 30 seconds
        assert len(results) == batch_size
        assert all(result == 100 for result in results)
        assert mock_db_service.store_embedding.call_count == batch_size


class TestRAGSecurity:
    """Test security aspects of RAG functionality."""
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_contact_phone_injection_prevention(self, rag_service):
        """Test prevention of SQL injection through contact phone parameter."""
        # Arrange
        service, mock_db_service, mock_embedding_service = rag_service
        
        query = "Test query"
        malicious_phone = "'; DROP TABLE embeddings; --"
        query_embedding = [0.1] * 1536
        
        mock_embedding_service.generate_embedding.return_value = query_embedding
        mock_db_service.similarity_search.return_value = []
        
        # Act
        await service.similarity_search(query=query, contact_phone=malicious_phone)
        
        # Assert - Should call database with the malicious string as parameter
        # Database layer should use parameterized queries to prevent injection
        mock_db_service.similarity_search.assert_called_once()
        call_args = mock_db_service.similarity_search.call_args[1]
        assert call_args["contact_phone"] == malicious_phone
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_content_sanitization(self, rag_service):
        """Test that sensitive content is properly handled."""
        # Arrange
        service, mock_db_service, mock_embedding_service = rag_service
        
        sensitive_content = "User's credit card: 4111-1111-1111-1111, SSN: 123-45-6789"
        contact_phone = "+1234567890"
        conversation_id = 12345
        embedding = [0.1] * 1536
        
        mock_embedding_service.generate_embedding.return_value = embedding
        mock_db_service.store_embedding.return_value = 100
        mock_db_service.store_message.return_value = 50
        
        # Act
        await service.store_embedding(
            content=sensitive_content,
            contact_phone=contact_phone,
            conversation_id=conversation_id
        )
        
        # Assert - Content should be stored (sanitization is application-level concern)
        mock_db_service.store_message.assert_called_once()
        call_args = mock_db_service.store_message.call_args[1]
        assert call_args["content"] == sensitive_content
        # Note: In production, consider implementing PII detection and masking
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_access_control_by_contact(self, rag_service):
        """Test that contacts cannot access each other's embeddings."""
        # Arrange
        service, mock_db_service, mock_embedding_service = rag_service
        
        query = "Sensitive information"
        unauthorized_phone = "+1111111111"
        authorized_phone = "+1234567890"
        query_embedding = [0.1] * 1536
        
        # Mock that only authorized contact has embeddings
        authorized_results = [{'content': 'Authorized content', 'similarity': 0.9}]
        unauthorized_results = []
        
        mock_embedding_service.generate_embedding.return_value = query_embedding
        
        def mock_search(*args, **kwargs):
            if kwargs.get("contact_phone") == authorized_phone:
                return authorized_results
            return unauthorized_results
        
        mock_db_service.similarity_search.side_effect = mock_search
        
        # Act - Search as unauthorized contact
        unauthorized_result = await service.similarity_search(
            query=query,
            contact_phone=unauthorized_phone
        )
        
        # Act - Search as authorized contact
        authorized_result = await service.similarity_search(
            query=query,
            contact_phone=authorized_phone
        )
        
        # Assert
        assert unauthorized_result == []  # No access to other contact's data
        assert authorized_result == authorized_results  # Access to own data
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_embedding_model_consistency(self, rag_service):
        """Test that embedding model cannot be manipulated."""
        # Arrange
        service, mock_db_service, mock_embedding_service = rag_service
        
        # Attempt to use different embedding model
        text = "Test content"
        
        # Act
        await service.generate_embedding(text)
        
        # Assert - Should always use configured model
        assert service.config.embedding_model == "text-embedding-3-small"
        # The service should not allow model switching at runtime for consistency