"""
Tests for the database service.
"""

import json
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

import pytest
import asyncpg

from src.services.database import (
    DatabaseService, DatabaseError, ConnectionError, QueryError
)
from src.models.schemas import AgentConfig, InboxConfig, LogEntry, AgentMemory
from src.utils.config import DatabaseConfig


class TestDatabaseService:
    """Tests for the DatabaseService class."""
    
    @pytest.fixture
    def database_service(self, test_database_config):
        """Create database service instance."""
        return DatabaseService(config=test_database_config)
    
    @pytest.fixture
    def mock_pool(self):
        """Mock AsyncPG pool."""
        pool = AsyncMock()
        pool.get_size.return_value = 5
        return pool
    
    @pytest.fixture
    def mock_connection(self):
        """Mock AsyncPG connection."""
        connection = AsyncMock()
        connection.fetchval.return_value = 1
        connection.fetchrow.return_value = None
        connection.fetch.return_value = []
        connection.execute.return_value = None
        connection.transaction.return_value = AsyncMock()
        return connection
    
    @pytest.mark.asyncio
    async def test_initialization_default_config(self):
        """Test initialization with default config."""
        with patch('src.services.database.get_database_config') as mock_get_config:
            mock_config = DatabaseConfig()
            mock_get_config.return_value = mock_config
            
            service = DatabaseService()
            assert service.config is mock_config
            assert service._pool is None
            assert service._initialized is False
    
    @pytest.mark.asyncio
    async def test_initialization_custom_config(self, test_database_config):
        """Test initialization with custom config."""
        service = DatabaseService(config=test_database_config)
        assert service.config is test_database_config
        assert service._pool is None
        assert service._initialized is False
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, database_service, mock_pool, mock_connection):
        """Test successful database initialization."""
        with patch('asyncpg.create_pool', return_value=mock_pool):
            mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            await database_service.initialize()
            
            assert database_service._pool is mock_pool
            assert database_service._initialized is True
    
    @pytest.mark.asyncio
    async def test_initialize_connection_failure(self, database_service):
        """Test initialization with connection failure."""
        with patch('asyncpg.create_pool', side_effect=Exception("Connection failed")):
            with pytest.raises(ConnectionError, match="Database connection failed"):
                await database_service.initialize()
            
            assert database_service._pool is None
            assert database_service._initialized is False
    
    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, database_service, mock_pool, mock_connection):
        """Test that initialize is idempotent."""
        with patch('asyncpg.create_pool', return_value=mock_pool):
            mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            # First call should initialize
            await database_service.initialize()
            assert database_service._initialized is True
            
            # Second call should not call create_pool again
            with patch('asyncpg.create_pool') as mock_create_pool:
                await database_service.initialize()
                mock_create_pool.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_close(self, database_service, mock_pool):
        """Test database connection pool closure."""
        database_service._pool = mock_pool
        database_service._initialized = True
        
        await database_service.close()
        
        mock_pool.close.assert_called_once()
        assert database_service._pool is None
        assert database_service._initialized is False
    
    @pytest.mark.asyncio
    async def test_close_no_pool(self, database_service):
        """Test close when no pool exists."""
        await database_service.close()  # Should not raise exception
    
    @pytest.mark.asyncio
    async def test_get_connection_success(self, database_service, mock_pool, mock_connection):
        """Test successful connection acquisition."""
        database_service._pool = mock_pool
        database_service._initialized = True
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        
        async with database_service.get_connection() as conn:
            assert conn is mock_connection
        
        mock_pool.acquire.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_connection_not_initialized(self, database_service, mock_pool, mock_connection):
        """Test connection acquisition when not initialized."""
        with patch('asyncpg.create_pool', return_value=mock_pool):
            mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            async with database_service.get_connection() as conn:
                assert conn is mock_connection
            
            assert database_service._initialized is True
    
    @pytest.mark.asyncio
    async def test_get_connection_no_pool(self, database_service):
        """Test connection acquisition failure."""
        database_service._initialized = True
        database_service._pool = None
        
        with pytest.raises(ConnectionError, match="Database pool not initialized"):
            async with database_service.get_connection():
                pass
    
    @pytest.mark.asyncio
    async def test_transaction_context(self, database_service, mock_pool, mock_connection):
        """Test transaction context manager."""
        database_service._pool = mock_pool
        database_service._initialized = True
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        
        async with database_service.transaction() as conn:
            assert conn is mock_connection
        
        mock_connection.transaction.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, database_service, mock_pool, mock_connection):
        """Test successful health check."""
        database_service._pool = mock_pool
        database_service._initialized = True
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        
        # Mock database queries
        mock_connection.fetchval.return_value = 1
        mock_connection.fetch.side_effect = [
            # Tables query
            [
                {'table_name': 'messages'},
                {'table_name': 'embeddings'},
                {'table_name': 'agent_configs'},
                {'table_name': 'agent_memory'},
                {'table_name': 'agent_logs'}
            ],
            # Vector extension query
            [{'extname': 'vector'}]
        ]
        
        health = await database_service.health_check()
        
        assert health['status'] == 'healthy'
        assert health['connection'] is True
        assert len(health['tables_found']) == 5
        assert health['pgvector_enabled'] is True
        assert health['pool_size'] == 5
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, database_service, mock_pool, mock_connection):
        """Test health check failure."""
        database_service._pool = mock_pool
        database_service._initialized = True
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        
        # Mock connection failure
        mock_connection.fetchval.side_effect = Exception("Connection failed")
        
        with pytest.raises(ConnectionError, match="Health check failed"):
            await database_service.health_check()


class TestConfigurationMethods:
    """Tests for configuration management methods."""
    
    @pytest.fixture
    def database_service_with_connection(self, database_service, mock_pool, mock_connection):
        """Database service with mocked connection."""
        database_service._pool = mock_pool
        database_service._initialized = True
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        return database_service, mock_connection
    
    @pytest.mark.asyncio
    async def test_get_global_config_success(self, database_service_with_connection):
        """Test successful global configuration retrieval."""
        service, mock_connection = database_service_with_connection
        
        config_data = {
            "enabled": True,
            "llm_provider": "openai",
            "model": "gpt-4"
        }
        mock_connection.fetchrow.return_value = {'config_data': config_data}
        
        config = await service.get_global_config()
        
        assert config is not None
        assert isinstance(config, AgentConfig)
        assert config.enabled is True
        assert config.model == "gpt-4"
    
    @pytest.mark.asyncio
    async def test_get_global_config_not_found(self, database_service_with_connection):
        """Test global configuration not found."""
        service, mock_connection = database_service_with_connection
        mock_connection.fetchrow.return_value = None
        
        config = await service.get_global_config()
        assert config is None
    
    @pytest.mark.asyncio
    async def test_get_inbox_config_success(self, database_service_with_connection):
        """Test successful inbox configuration retrieval."""
        service, mock_connection = database_service_with_connection
        
        config_data = {
            "enabled": True,
            "language": "pt",
            "persona": "helpful assistant"
        }
        mock_connection.fetchrow.return_value = {'config_data': config_data}
        
        config = await service.get_inbox_config(42)
        
        assert config is not None
        assert isinstance(config, InboxConfig)
        assert config.inbox_id == 42
        assert config.language == "pt"
        assert config.persona == "helpful assistant"
    
    @pytest.mark.asyncio
    async def test_save_global_config(self, database_service_with_connection):
        """Test saving global configuration."""
        service, mock_connection = database_service_with_connection
        
        config = AgentConfig(model="gpt-4", temperature=0.7)
        await service.save_global_config(config)
        
        mock_connection.execute.assert_called_once()
        call_args = mock_connection.execute.call_args
        assert "INSERT INTO agent_configs" in call_args[0][0]
        assert "global" in str(call_args[0])
    
    @pytest.mark.asyncio
    async def test_save_inbox_config(self, database_service_with_connection):
        """Test saving inbox configuration."""
        service, mock_connection = database_service_with_connection
        
        config = InboxConfig(inbox_id=42, language="pt", persona="helpful")
        await service.save_inbox_config(config)
        
        mock_connection.execute.assert_called_once()
        call_args = mock_connection.execute.call_args
        assert "INSERT INTO agent_configs" in call_args[0][0]
        assert 42 in call_args[0]


class TestMessageStorage:
    """Tests for message storage methods."""
    
    @pytest.fixture
    def database_service_with_connection(self, database_service, mock_pool, mock_connection):
        """Database service with mocked connection."""
        database_service._pool = mock_pool
        database_service._initialized = True
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        return database_service, mock_connection
    
    @pytest.mark.asyncio
    async def test_store_message_success(self, database_service_with_connection):
        """Test successful message storage."""
        service, mock_connection = database_service_with_connection
        mock_connection.fetchval.return_value = 123  # Message ID
        
        message_id = await service.store_message(
            contact_phone="+1234567890",
            conversation_id=12345,
            role="user",
            content="Hello, world!",
            message_type="text",
            metadata={"source": "whatsapp"},
            sent_at=datetime.utcnow()
        )
        
        assert message_id == 123
        mock_connection.fetchval.assert_called_once()
        
        call_args = mock_connection.fetchval.call_args
        assert "INSERT INTO messages" in call_args[0][0]
        assert "+1234567890" in call_args[0]
        assert 12345 in call_args[0]
        assert "user" in call_args[0]
        assert "Hello, world!" in call_args[0]
    
    @pytest.mark.asyncio
    async def test_store_message_with_defaults(self, database_service_with_connection):
        """Test message storage with default values."""
        service, mock_connection = database_service_with_connection
        mock_connection.fetchval.return_value = 124
        
        message_id = await service.store_message(
            contact_phone="+1234567890",
            conversation_id=12345,
            role="assistant",
            content="Response message"
        )
        
        assert message_id == 124
        call_args = mock_connection.fetchval.call_args
        assert "text" in str(call_args[0])  # Default message_type
    
    @pytest.mark.asyncio
    async def test_get_conversation_history_success(self, database_service_with_connection):
        """Test successful conversation history retrieval."""
        service, mock_connection = database_service_with_connection
        
        mock_records = [
            {
                'id': 1,
                'conversation_id': 12345,
                'role': 'user',
                'content': 'Hello',
                'message_type': 'text',
                'metadata': {},
                'sent_at': datetime.utcnow(),
                'created_at': datetime.utcnow()
            },
            {
                'id': 2,
                'conversation_id': 12345,
                'role': 'assistant',
                'content': 'Hi there!',
                'message_type': 'text',
                'metadata': {},
                'sent_at': datetime.utcnow(),
                'created_at': datetime.utcnow()
            }
        ]
        mock_connection.fetch.return_value = mock_records
        
        messages = await service.get_conversation_history(
            contact_phone="+1234567890",
            conversation_id=12345,
            limit=50
        )
        
        assert len(messages) == 2
        assert messages[0]['role'] == 'user'
        assert messages[1]['role'] == 'assistant'
        
        call_args = mock_connection.fetch.call_args
        assert "SELECT" in call_args[0][0]
        assert "FROM messages" in call_args[0][0]
        assert "+1234567890" in call_args[0]
    
    @pytest.mark.asyncio
    async def test_get_conversation_history_with_time_window(self, database_service_with_connection):
        """Test conversation history with time window filter."""
        service, mock_connection = database_service_with_connection
        mock_connection.fetch.return_value = []
        
        await service.get_conversation_history(
            contact_phone="+1234567890",
            time_window_days=7
        )
        
        call_args = mock_connection.fetch.call_args
        assert "INTERVAL '7 days'" in call_args[0][0]


class TestEmbeddingMethods:
    """Tests for vector/embedding methods."""
    
    @pytest.fixture
    def database_service_with_connection(self, database_service, mock_pool, mock_connection):
        """Database service with mocked connection."""
        database_service._pool = mock_pool
        database_service._initialized = True
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        return database_service, mock_connection
    
    @pytest.mark.asyncio
    async def test_store_embedding_success(self, database_service_with_connection):
        """Test successful embedding storage."""
        service, mock_connection = database_service_with_connection
        mock_connection.fetchval.return_value = 456  # Embedding ID
        
        embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
        
        embedding_id = await service.store_embedding(
            message_id=123,
            contact_phone="+1234567890",
            embedding=embedding,
            content_chunk="Hello, world!",
            chunk_index=0,
            metadata={"model": "text-embedding-3-small"}
        )
        
        assert embedding_id == 456
        mock_connection.fetchval.assert_called_once()
        
        call_args = mock_connection.fetchval.call_args
        assert "INSERT INTO embeddings" in call_args[0][0]
        assert 123 in call_args[0]
        assert "+1234567890" in call_args[0]
        assert embedding in call_args[0]
    
    @pytest.mark.asyncio
    async def test_similarity_search_success(self, database_service_with_connection):
        """Test successful similarity search."""
        service, mock_connection = database_service_with_connection
        
        mock_records = [
            {
                'id': 1,
                'message_id': 123,
                'conversation_id': 12345,
                'content_chunk': 'Similar content',
                'chunk_index': 0,
                'similarity': 0.85,
                'metadata': {},
                'message_role': 'user',
                'sent_at': datetime.utcnow(),
                'created_at': datetime.utcnow()
            }
        ]
        mock_connection.fetch.return_value = mock_records
        
        query_embedding = [0.1, 0.2, 0.3] * 512
        results = await service.similarity_search(
            query_embedding=query_embedding,
            contact_phone="+1234567890",
            top_k=4,
            similarity_threshold=0.7
        )
        
        assert len(results) == 1
        assert results[0]['similarity'] == 0.85
        assert results[0]['content_chunk'] == 'Similar content'
        
        call_args = mock_connection.fetch.call_args
        assert "SELECT" in call_args[0][0]
        assert "FROM embeddings" in call_args[0][0]
        assert "similarity DESC" in call_args[0][0]
        assert "+1234567890" in call_args[0]


class TestAgentMemory:
    """Tests for agent memory methods."""
    
    @pytest.fixture
    def database_service_with_connection(self, database_service, mock_pool, mock_connection):
        """Database service with mocked connection."""
        database_service._pool = mock_pool
        database_service._initialized = True
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        return database_service, mock_connection
    
    @pytest.mark.asyncio
    async def test_get_agent_memory_success(self, database_service_with_connection):
        """Test successful agent memory retrieval."""
        service, mock_connection = database_service_with_connection
        
        memory_data = {
            'conversation_history': [{'role': 'user', 'content': 'Hello'}],
            'metadata': {'last_topic': 'greeting'}
        }
        
        mock_connection.fetchrow.return_value = {
            'contact_phone': '+1234567890',
            'conversation_id': 12345,
            'memory_data': memory_data,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        memory = await service.get_agent_memory(
            contact_phone="+1234567890",
            conversation_id=12345
        )
        
        assert memory is not None
        assert isinstance(memory, AgentMemory)
        assert memory.contact_phone == "+1234567890"
        assert memory.conversation_id == 12345
        assert len(memory.conversation_history) == 1
    
    @pytest.mark.asyncio
    async def test_save_agent_memory(self, database_service_with_connection):
        """Test saving agent memory."""
        service, mock_connection = database_service_with_connection
        
        memory = AgentMemory(
            contact_phone="+1234567890",
            conversation_id=12345,
            conversation_history=[{'role': 'user', 'content': 'Hello'}],
            metadata={'last_topic': 'greeting'}
        )
        
        await service.save_agent_memory(memory)
        
        mock_connection.execute.assert_called_once()
        call_args = mock_connection.execute.call_args
        assert "INSERT INTO agent_memory" in call_args[0][0]
        assert "+1234567890" in call_args[0]
        assert 12345 in call_args[0]


class TestLoggingMethods:
    """Tests for logging methods."""
    
    @pytest.fixture
    def database_service_with_connection(self, database_service, mock_pool, mock_connection):
        """Database service with mocked connection."""
        database_service._pool = mock_pool
        database_service._initialized = True
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        return database_service, mock_connection
    
    @pytest.mark.asyncio
    async def test_log_agent_execution_success(self, database_service_with_connection):
        """Test successful agent execution logging."""
        service, mock_connection = database_service_with_connection
        mock_connection.fetchval.return_value = 789  # Log ID
        
        log_entry = LogEntry(
            conversation_id=12345,
            contact_phone="+1234567890",
            user_query="What's my order status?",
            tool_used="order_lookup",
            final_response="Your order is shipped",
            latency_ms=150,
            token_count=50,
            cost_usd=0.001,
            status="success"
        )
        
        log_id = await service.log_agent_execution(log_entry)
        
        assert log_id == 789
        mock_connection.fetchval.assert_called_once()
        
        call_args = mock_connection.fetchval.call_args
        assert "INSERT INTO agent_logs" in call_args[0][0]
        assert 12345 in call_args[0]
        assert "+1234567890" in call_args[0]
        assert "order_lookup" in call_args[0]
    
    @pytest.mark.asyncio
    async def test_get_execution_stats_success(self, database_service_with_connection):
        """Test successful execution statistics retrieval."""
        service, mock_connection = database_service_with_connection
        
        mock_connection.fetchrow.return_value = {
            'total_executions': 100,
            'successful_executions': 95,
            'failed_executions': 5,
            'avg_latency_ms': 125.5,
            'total_tokens': 5000,
            'total_cost_usd': Decimal('0.050'),
            'tools_used_count': 3
        }
        
        stats = await service.get_execution_stats(
            contact_phone="+1234567890",
            days=7
        )
        
        assert stats['total_executions'] == 100
        assert stats['successful_executions'] == 95
        assert stats['failed_executions'] == 5
        assert stats['success_rate'] == 0.95
        assert stats['avg_latency_ms'] == 125.5
        assert stats['total_tokens'] == 5000
        assert stats['total_cost_usd'] == 0.05
        assert stats['period_days'] == 7


class TestErrorHandling:
    """Tests for error handling in database service."""
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self, database_service):
        """Test connection error handling."""
        with patch('asyncpg.create_pool', side_effect=asyncpg.ConnectionError("Failed to connect")):
            with pytest.raises(ConnectionError):
                await database_service.initialize()
    
    @pytest.mark.asyncio
    async def test_query_error_handling(self, database_service, mock_pool, mock_connection):
        """Test query error handling."""
        database_service._pool = mock_pool
        database_service._initialized = True
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        
        # Mock database query failure
        mock_connection.fetchval.side_effect = Exception("Query failed")
        
        with pytest.raises(QueryError, match="Global config query failed"):
            await database_service.get_global_config()
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, database_service, mock_pool, mock_connection):
        """Test transaction rollback on error."""
        database_service._pool = mock_pool
        database_service._initialized = True
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        
        # Mock transaction that raises an exception
        mock_connection.execute.side_effect = Exception("Transaction failed")
        
        config = AgentConfig()
        with pytest.raises(QueryError):
            await database_service.save_global_config(config)
        
        # Transaction context manager should handle rollback
        mock_connection.transaction.assert_called_once()


class TestGlobalServiceManagement:
    """Tests for global service management functions."""
    
    @pytest.mark.asyncio
    async def test_get_database_service(self):
        """Test get_database_service function."""
        from src.services.database import get_database_service, _database_service
        
        # Clear global service
        import src.services.database
        src.services.database._database_service = None
        
        with patch.object(DatabaseService, 'initialize') as mock_initialize:
            service = await get_database_service()
            assert isinstance(service, DatabaseService)
            mock_initialize.assert_called_once()
            
            # Second call should return same instance
            service2 = await get_database_service()
            assert service2 is service
            assert mock_initialize.call_count == 1
    
    @pytest.mark.asyncio
    async def test_close_database_service(self):
        """Test close_database_service function."""
        from src.services.database import close_database_service
        import src.services.database
        
        # Set up mock service
        mock_service = AsyncMock()
        src.services.database._database_service = mock_service
        
        await close_database_service()
        
        mock_service.close.assert_called_once()
        assert src.services.database._database_service is None
    
    @pytest.mark.asyncio
    async def test_close_database_service_no_service(self):
        """Test closing database service when none exists."""
        from src.services.database import close_database_service
        import src.services.database
        
        src.services.database._database_service = None
        
        # Should not raise exception
        await close_database_service()