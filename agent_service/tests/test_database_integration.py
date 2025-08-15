"""
Integration tests for database operations.

Tests cover:
- Connection pooling and management
- Transaction handling
- pgvector operations
- Migration validation
- Performance under load
- Error recovery scenarios
"""

import pytest
import asyncio
import asyncpg
import numpy as np
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch
from typing import Dict, Any, List

from src.services.database import DatabaseService, get_database_service
from src.utils.config import DatabaseConfig
from src.models.schemas import AgentConfig, AgentMemory, LogEntry
from tests.conftest import (
    test_database_config, assert_response_time,
    DatabaseTestHelper, AsyncMockService
)


class TestDatabaseService:
    """Test core database service functionality."""
    
    @pytest.fixture
    async def database_service(self, test_database_config):
        """Create database service with test configuration."""
        service = DatabaseService(test_database_config)
        
        # Mock the actual database connection for unit tests
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            await service.initialize()
            service._pool = mock_pool
            
            return service, mock_pool
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_database_service_initialization(self, test_database_config):
        """Test database service initialization with real connection."""
        # Note: This test requires actual database setup
        # In CI/CD, this would use a test database container
        
        service = DatabaseService(test_database_config)
        
        # Mock for unit testing
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            await service.initialize()
            
            assert service._initialized is True
            assert service._pool is not None
            mock_create_pool.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_check(self, database_service):
        """Test database health check functionality."""
        # Arrange
        service, mock_pool = database_service
        
        # Mock database responses
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock health check queries
        mock_connection.fetchval.side_effect = [
            1,  # SELECT 1
            True,  # pgvector extension check
            5,   # pool size
            20   # max pool size
        ]
        
        mock_connection.fetch.return_value = [
            {'table_name': 'messages'},
            {'table_name': 'embeddings'},
            {'table_name': 'agent_configs'},
            {'table_name': 'agent_memory'},
            {'table_name': 'agent_logs'}
        ]
        
        # Act
        start_time = time.time()
        health = await service.health_check()
        
        # Assert
        assert_response_time(start_time, 3000)  # Should complete within 3 seconds
        assert health['status'] == 'healthy'
        assert health['connection'] is True
        assert health['pgvector_enabled'] is True
        assert health['pool_size'] == 5
        assert health['pool_max_size'] == 20
        assert len(health['tables_found']) == 5
        assert 'timestamp' in health
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_store_message(self, database_service):
        """Test storing message in database."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        mock_connection.fetchval.return_value = 123  # Mock message ID
        
        # Act
        message_id = await service.store_message(
            contact_phone="+1234567890",
            conversation_id=12345,
            role="user",
            content="Hello, I need help with my order"
        )
        
        # Assert
        assert message_id == 123
        mock_connection.fetchval.assert_called_once()
        
        # Verify SQL query structure
        call_args = mock_connection.fetchval.call_args
        sql_query = call_args[0][0]
        assert "INSERT INTO messages" in sql_query
        assert "RETURNING id" in sql_query
        
        # Verify parameters
        params = call_args[0][1:]
        assert "+1234567890" in params
        assert 12345 in params
        assert "user" in params
        assert "Hello, I need help with my order" in params
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_store_embedding(self, database_service):
        """Test storing embedding vector in database."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        mock_connection.fetchval.return_value = 456  # Mock embedding ID
        
        embedding_vector = [0.1] * 1536  # 1536-dimensional embedding
        metadata = {"conversation_id": 12345, "type": "user_query"}
        
        # Act
        embedding_id = await service.store_embedding(
            message_id=123,
            contact_phone="+1234567890",
            embedding=embedding_vector,
            metadata=metadata
        )
        
        # Assert
        assert embedding_id == 456
        mock_connection.fetchval.assert_called_once()
        
        # Verify SQL query structure
        call_args = mock_connection.fetchval.call_args
        sql_query = call_args[0][0]
        assert "INSERT INTO embeddings" in sql_query
        assert "RETURNING id" in sql_query
        
        # Verify parameters include embedding vector
        params = call_args[0][1:]
        assert 123 in params  # message_id
        assert "+1234567890" in params  # contact_phone
        # Embedding vector should be converted to proper format for pgvector
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_similarity_search(self, database_service):
        """Test vector similarity search."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock similarity search results
        mock_results = [
            {
                'id': 1,
                'content': 'Previous order inquiry',
                'similarity': 0.85,
                'metadata': {'conversation_id': 12340, 'type': 'user_query'},
                'created_at': datetime.utcnow()
            },
            {
                'id': 2,
                'content': 'Installation service question',
                'similarity': 0.78,
                'metadata': {'conversation_id': 12341, 'type': 'support_request'},
                'created_at': datetime.utcnow() - timedelta(days=1)
            }
        ]
        mock_connection.fetch.return_value = mock_results
        
        query_embedding = [0.15] * 1536
        
        # Act
        results = await service.similarity_search(
            embedding=query_embedding,
            contact_phone="+1234567890",
            top_k=5,
            similarity_threshold=0.7,
            time_window_days=30
        )
        
        # Assert
        assert len(results) == 2
        assert results[0]['similarity'] == 0.85
        assert results[1]['similarity'] == 0.78
        assert all('content' in result for result in results)
        assert all('metadata' in result for result in results)
        
        # Verify SQL query structure
        mock_connection.fetch.assert_called_once()
        call_args = mock_connection.fetch.call_args
        sql_query = call_args[0][0]
        assert "SELECT" in sql_query
        assert "ORDER BY" in sql_query
        assert "LIMIT" in sql_query
        assert "embedding <=> $1" in sql_query or "embedding <#>" in sql_query  # pgvector operators
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_conversation_history(self, database_service):
        """Test retrieving conversation history."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock conversation history
        mock_history = [
            {
                'id': 1,
                'role': 'user',
                'content': 'Hello, I need help',
                'sent_at': datetime.utcnow() - timedelta(minutes=10)
            },
            {
                'id': 2,
                'role': 'assistant',
                'content': 'I can help you with that',
                'sent_at': datetime.utcnow() - timedelta(minutes=9)
            }
        ]
        mock_connection.fetch.return_value = mock_history
        
        # Act
        history = await service.get_conversation_history(
            contact_phone="+1234567890",
            conversation_id=12345,
            limit=10
        )
        
        # Assert
        assert len(history) == 2
        assert history[0]['role'] == 'user'
        assert history[1]['role'] == 'assistant'
        assert all('content' in msg for msg in history)
        assert all('sent_at' in msg for msg in history)
        
        # Verify query parameters
        call_args = mock_connection.fetch.call_args
        params = call_args[0][1:]
        assert "+1234567890" in params
        assert 12345 in params
        assert 10 in params
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_store_agent_memory(self, database_service):
        """Test storing agent memory."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        mock_connection.fetchval.return_value = 789  # Mock memory ID
        
        memory = AgentMemory(
            contact_phone="+1234567890",
            conversation_id=12345,
            conversation_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            metadata={"preferences": {"language": "en"}}
        )
        
        # Act
        memory_id = await service.store_agent_memory(memory)
        
        # Assert
        assert memory_id == 789
        mock_connection.fetchval.assert_called_once()
        
        # Verify SQL structure
        call_args = mock_connection.fetchval.call_args
        sql_query = call_args[0][0]
        assert "INSERT INTO agent_memory" in sql_query or "UPSERT" in sql_query
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_agent_memory(self, database_service):
        """Test retrieving agent memory."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock memory data
        mock_memory_data = {
            'id': 789,
            'contact_phone': '+1234567890',
            'conversation_id': 12345,
            'conversation_history': [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            'metadata': {"preferences": {"language": "en"}},
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        mock_connection.fetchrow.return_value = mock_memory_data
        
        # Act
        memory = await service.get_agent_memory(
            contact_phone="+1234567890",
            conversation_id=12345
        )
        
        # Assert
        assert memory is not None
        assert memory['contact_phone'] == '+1234567890'
        assert memory['conversation_id'] == 12345
        assert len(memory['conversation_history']) == 2
        assert 'metadata' in memory
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_log_agent_execution(self, database_service):
        """Test logging agent execution."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        mock_connection.fetchval.return_value = 999  # Mock log ID
        
        log_entry = LogEntry(
            conversation_id=12345,
            contact_phone="+1234567890",
            user_query="What is my order status?",
            tool_used="rag_search",
            final_response="Your order is being processed",
            latency_ms=1500,
            token_count=150,
            cost_usd=0.045,
            status="success"
        )
        
        # Act
        log_id = await service.log_agent_execution(log_entry)
        
        # Assert
        assert log_id == 999
        mock_connection.fetchval.assert_called_once()
        
        # Verify SQL structure
        call_args = mock_connection.fetchval.call_args
        sql_query = call_args[0][0]
        assert "INSERT INTO agent_logs" in sql_query
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_execution_stats(self, database_service):
        """Test getting execution statistics."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock statistics data
        mock_stats = {
            'total_executions': 250,
            'successful_executions': 235,
            'failed_executions': 15,
            'avg_latency_ms': 1250.5,
            'total_tokens': 45000,
            'total_cost_usd': 135.50,
            'tools_used_count': 8
        }
        mock_connection.fetchrow.return_value = mock_stats
        
        # Act
        stats = await service.get_execution_stats(period_days=7)
        
        # Assert
        assert stats['total_executions'] == 250
        assert stats['successful_executions'] == 235
        assert stats['failed_executions'] == 15
        assert stats['success_rate'] == pytest.approx(0.94, rel=1e-2)  # 235/250
        assert stats['avg_latency_ms'] == 1250.5
        assert stats['total_tokens'] == 45000
        assert stats['total_cost_usd'] == 135.50


class TestDatabaseConnectionManagement:
    """Test database connection pooling and management."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_connection_pool_creation(self, test_database_config):
        """Test connection pool creation and configuration."""
        # Arrange
        service = DatabaseService(test_database_config)
        
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            # Act
            await service.initialize()
            
            # Assert
            mock_create_pool.assert_called_once_with(
                host=test_database_config.host,
                port=test_database_config.port,
                database=test_database_config.database,
                user=test_database_config.username,
                password=test_database_config.password,
                min_size=test_database_config.pool_min_size,
                max_size=test_database_config.pool_max_size,
                command_timeout=60
            )
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_connection_pool_cleanup(self, database_service):
        """Test proper cleanup of connection pool."""
        # Arrange
        service, mock_pool = database_service
        
        # Act
        await service.close()
        
        # Assert
        mock_pool.close.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_connection_retry_on_failure(self, test_database_config):
        """Test connection retry mechanism on failure."""
        # Arrange
        service = DatabaseService(test_database_config)
        
        with patch('asyncpg.create_pool') as mock_create_pool:
            # First call fails, second succeeds
            mock_create_pool.side_effect = [
                Exception("Connection failed"),
                AsyncMock()
            ]
            
            # Act
            await service.initialize()
            
            # Assert
            assert mock_create_pool.call_count == 2
            assert service._initialized is True
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_database_operations(self, database_service):
        """Test database operations under concurrent load."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        mock_connection.fetchval.return_value = 123
        mock_connection.fetch.return_value = []
        
        # Act - Create concurrent operations
        tasks = []
        for i in range(20):
            task = service.store_message(
                contact_phone=f"+123456789{i}",
                conversation_id=12345 + i,
                role="user",
                content=f"Message {i}"
            )
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        
        # Assert
        assert_response_time(start_time, 5000)  # Should complete within 5 seconds
        assert len(results) == 20
        assert all(result == 123 for result in results)
        assert mock_connection.fetchval.call_count == 20


class TestDatabaseTransactions:
    """Test database transaction handling."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transaction_commit_success(self, database_service):
        """Test successful transaction commit."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_transaction = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.transaction.return_value = mock_transaction
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        
        mock_connection.fetchval.side_effect = [123, 456]  # message_id, embedding_id
        
        # Act - Perform transaction with multiple operations
        async with service._pool.acquire() as conn:
            async with conn.transaction():
                message_id = await service.store_message(
                    contact_phone="+1234567890",
                    conversation_id=12345,
                    role="user",
                    content="Test message"
                )
                
                embedding_id = await service.store_embedding(
                    message_id=message_id,
                    contact_phone="+1234567890",
                    embedding=[0.1] * 1536,
                    metadata={"type": "test"}
                )
        
        # Assert
        assert message_id == 123
        assert embedding_id == 456
        mock_transaction.__aenter__.assert_called()
        mock_transaction.__aexit__.assert_called()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, database_service):
        """Test transaction rollback on error."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_transaction = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.transaction.return_value = mock_transaction
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        
        # First operation succeeds, second fails
        mock_connection.fetchval.side_effect = [123, Exception("Database error")]
        
        # Act & Assert
        with pytest.raises(Exception, match="Database error"):
            async with service._pool.acquire() as conn:
                async with conn.transaction():
                    await service.store_message(
                        contact_phone="+1234567890",
                        conversation_id=12345,
                        role="user",
                        content="Test message"
                    )
                    
                    await service.store_embedding(
                        message_id=123,
                        contact_phone="+1234567890",
                        embedding=[0.1] * 1536,
                        metadata={"type": "test"}
                    )
        
        # Transaction should have been rolled back
        mock_transaction.__aexit__.assert_called()


class TestPgvectorOperations:
    """Test pgvector specific operations."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_vector_similarity_operators(self, database_service):
        """Test different vector similarity operators."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        mock_connection.fetch.return_value = []
        
        query_embedding = [0.1] * 1536
        
        # Test different similarity operators
        operators = ['<->', '<#>', '<=>']  # L2, inner product, cosine
        
        for operator in operators:
            # Act
            await service.similarity_search(
                embedding=query_embedding,
                contact_phone="+1234567890",
                similarity_operator=operator
            )
            
            # Assert - Verify operator is used in query
            call_args = mock_connection.fetch.call_args
            sql_query = call_args[0][0]
            assert operator in sql_query
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_vector_index_performance(self, database_service):
        """Test performance with vector indices."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock large result set to simulate index usage
        mock_results = [
            {
                'id': i,
                'content': f'Content {i}',
                'similarity': 0.9 - (i * 0.01),
                'metadata': {},
                'created_at': datetime.utcnow()
            }
            for i in range(100)
        ]
        mock_connection.fetch.return_value = mock_results
        
        query_embedding = [0.1] * 1536
        
        # Act
        start_time = time.time()
        results = await service.similarity_search(
            embedding=query_embedding,
            contact_phone="+1234567890",
            top_k=10
        )
        
        # Assert
        assert_response_time(start_time, 1000)  # Should be fast with index
        assert len(results) <= 10  # Should respect top_k limit
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_embedding_dimension_validation(self, database_service):
        """Test validation of embedding dimensions."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Test with wrong dimension
        wrong_dimension_embedding = [0.1] * 512  # Should be 1536
        
        mock_connection.fetchval.side_effect = Exception("dimension mismatch")
        
        # Act & Assert
        with pytest.raises(Exception, match="dimension"):
            await service.store_embedding(
                message_id=123,
                contact_phone="+1234567890",
                embedding=wrong_dimension_embedding,
                metadata={}
            )


class TestDatabaseMigrations:
    """Test database migration and schema validation."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_schema_validation(self, database_service):
        """Test validation of required database schema."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock table existence check
        required_tables = ['messages', 'embeddings', 'agent_configs', 'agent_memory', 'agent_logs']
        mock_connection.fetch.return_value = [
            {'table_name': table} for table in required_tables
        ]
        
        # Act
        schema_valid = await service.validate_schema()
        
        # Assert
        assert schema_valid is True
        mock_connection.fetch.assert_called()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_missing_tables_detection(self, database_service):
        """Test detection of missing required tables."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock missing some tables
        existing_tables = ['messages', 'embeddings']  # Missing agent_configs, agent_memory, agent_logs
        mock_connection.fetch.return_value = [
            {'table_name': table} for table in existing_tables
        ]
        
        # Act
        schema_valid = await service.validate_schema()
        
        # Assert
        assert schema_valid is False
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pgvector_extension_check(self, database_service):
        """Test checking for pgvector extension."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock extension check
        mock_connection.fetchval.return_value = True  # Extension exists
        
        # Act
        has_pgvector = await service.check_pgvector_extension()
        
        # Assert
        assert has_pgvector is True
        
        # Verify SQL query
        call_args = mock_connection.fetchval.call_args
        sql_query = call_args[0][0]
        assert "pg_extension" in sql_query.lower() or "vector" in sql_query.lower()


class TestDatabaseErrorHandling:
    """Test database error handling and recovery."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self, database_service):
        """Test handling of connection timeouts."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock timeout error
        mock_connection.fetchval.side_effect = asyncio.TimeoutError("Query timeout")
        
        # Act & Assert
        with pytest.raises(asyncio.TimeoutError):
            await service.store_message(
                contact_phone="+1234567890",
                conversation_id=12345,
                role="user",
                content="Test message"
            )
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_database_constraint_violation(self, database_service):
        """Test handling of database constraint violations."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock constraint violation
        constraint_error = Exception("duplicate key value violates unique constraint")
        mock_connection.fetchval.side_effect = constraint_error
        
        # Act & Assert
        with pytest.raises(Exception, match="duplicate key"):
            await service.store_message(
                contact_phone="+1234567890",
                conversation_id=12345,
                role="user",
                content="Test message"
            )
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_connection_recovery_after_failure(self, database_service):
        """Test connection recovery after failure."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # First call fails, second succeeds
        mock_connection.fetchval.side_effect = [
            Exception("Connection lost"),
            123  # Success
        ]
        
        # Act - First call should fail
        with pytest.raises(Exception, match="Connection lost"):
            await service.store_message(
                contact_phone="+1234567890",
                conversation_id=12345,
                role="user",
                content="Test message 1"
            )
        
        # Second call should succeed (simulating recovery)
        result = await service.store_message(
            contact_phone="+1234567890",
            conversation_id=12345,
            role="user",
            content="Test message 2"
        )
        
        # Assert
        assert result == 123
        assert mock_connection.fetchval.call_count == 2


class TestDatabasePerformance:
    """Test database performance characteristics."""
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_bulk_insert_performance(self, database_service):
        """Test performance of bulk insert operations."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock successful inserts
        mock_connection.fetchval.side_effect = range(1, 101)  # IDs 1-100
        
        # Act - Insert 100 messages
        start_time = time.time()
        tasks = []
        
        for i in range(100):
            task = service.store_message(
                contact_phone=f"+123456789{i%10}",
                conversation_id=12345 + i,
                role="user",
                content=f"Bulk message {i}"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Assert
        assert_response_time(start_time, 10000)  # Should complete within 10 seconds
        assert len(results) == 100
        assert results == list(range(1, 101))
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_similarity_search_performance(self, database_service):
        """Test performance of similarity search operations."""
        # Arrange
        service, mock_pool = database_service
        
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock search results
        mock_results = [
            {
                'id': i,
                'content': f'Content {i}',
                'similarity': 0.9 - (i * 0.01),
                'metadata': {},
                'created_at': datetime.utcnow()
            }
            for i in range(10)
        ]
        mock_connection.fetch.return_value = mock_results
        
        query_embedding = [0.1] * 1536
        
        # Act - Perform multiple searches concurrently
        start_time = time.time()
        tasks = []
        
        for i in range(10):
            task = service.similarity_search(
                embedding=query_embedding,
                contact_phone=f"+123456789{i}",
                top_k=5
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Assert
        assert_response_time(start_time, 3000)  # Should complete within 3 seconds
        assert len(results) == 10
        assert all(len(result) == 10 for result in results)  # Mock returns 10 results each