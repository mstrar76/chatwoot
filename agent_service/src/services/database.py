"""
Database services for the Chatwoot Agent Service.

Provides AsyncPG-based database operations with connection pooling,
configuration management, and comprehensive error handling.
"""

import json
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple, Union
from decimal import Decimal

import asyncpg
from asyncpg import Pool, Connection, Record

from ..models.schemas import (
    AgentConfig, InboxConfig, LogEntry, AgentMemory, 
    Message as MessageSchema, EventType
)
from ..utils.config import get_database_config, DatabaseConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class ConnectionError(DatabaseError):
    """Exception for database connection issues."""
    pass


class QueryError(DatabaseError):
    """Exception for database query issues."""
    pass


class DatabaseService:
    """
    Async database service with connection pooling and transaction management.
    
    Provides high-level database operations for the Chatwoot Agent Service
    with automatic retry, connection pooling, and comprehensive error handling.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or get_database_config()
        self._pool: Optional[Pool] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize database connection pool.
        
        Raises:
            ConnectionError: If unable to establish database connection
        """
        if self._initialized:
            return
        
        try:
            logger.info("Initializing database connection pool",
                       host=self.config.host,
                       database=self.config.database,
                       pool_min_size=self.config.pool_min_size,
                       pool_max_size=self.config.pool_max_size)
            
            self._pool = await asyncpg.create_pool(
                self.config.asyncpg_dsn,
                min_size=self.config.pool_min_size,
                max_size=self.config.pool_max_size,
                max_queries=self.config.pool_max_queries,
                max_inactive_connection_lifetime=self.config.pool_max_inactive_connection_lifetime,
                command_timeout=30
            )
            
            # Test connection
            await self.health_check()
            
            self._initialized = True
            logger.info("Database connection pool initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize database connection pool", error=str(e))
            raise ConnectionError(f"Database connection failed: {e}")
    
    async def close(self) -> None:
        """Close database connection pool."""
        if self._pool:
            logger.info("Closing database connection pool")
            await self._pool.close()
            self._pool = None
            self._initialized = False
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[Connection, None]:
        """
        Get database connection from pool.
        
        Yields:
            AsyncPG connection instance
            
        Raises:
            ConnectionError: If unable to get connection from pool
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._pool:
            raise ConnectionError("Database pool not initialized")
        
        try:
            async with self._pool.acquire() as connection:
                yield connection
        except Exception as e:
            logger.error("Failed to acquire database connection", error=str(e))
            raise ConnectionError(f"Connection acquisition failed: {e}")
    
    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Connection, None]:
        """
        Get database connection with transaction management.
        
        Yields:
            AsyncPG connection with active transaction
            
        Raises:
            ConnectionError: If unable to get connection
        """
        async with self.get_connection() as conn:
            async with conn.transaction():
                yield conn
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform database health check.
        
        Returns:
            Dictionary with health check results
            
        Raises:
            ConnectionError: If health check fails
        """
        try:
            async with self.get_connection() as conn:
                # Test basic connectivity
                result = await conn.fetchval("SELECT 1")
                
                # Check if required tables exist
                tables_query = """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('messages', 'embeddings', 'agent_configs', 'agent_memory', 'agent_logs')
                """
                tables = await conn.fetch(tables_query)
                table_names = [record['table_name'] for record in tables]
                
                # Check pgvector extension
                vector_query = "SELECT extname FROM pg_extension WHERE extname = 'vector'"
                vector_enabled = bool(await conn.fetch(vector_query))
                
                health_status = {
                    'status': 'healthy',
                    'connection': result == 1,
                    'tables_found': table_names,
                    'required_tables': ['messages', 'embeddings', 'agent_configs', 'agent_memory', 'agent_logs'],
                    'pgvector_enabled': vector_enabled,
                    'pool_size': self._pool.get_size() if self._pool else 0,
                    'pool_max_size': self.config.pool_max_size,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                logger.debug("Database health check completed", **health_status)
                return health_status
                
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            raise ConnectionError(f"Health check failed: {e}")
    
    # Configuration Management Methods
    
    async def get_global_config(self) -> Optional[AgentConfig]:
        """
        Get global agent configuration from database.
        
        Returns:
            AgentConfig instance if found, None otherwise
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT config_data 
                    FROM agent_configs 
                    WHERE config_type = 'global' AND inbox_id IS NULL
                    ORDER BY updated_at DESC 
                    LIMIT 1
                """
                record = await conn.fetchrow(query)
                
                if record:
                    return AgentConfig(**record['config_data'])
                return None
                
        except Exception as e:
            logger.error("Failed to get global configuration", error=str(e))
            raise QueryError(f"Global config query failed: {e}")
    
    async def get_inbox_config(self, inbox_id: int) -> Optional[InboxConfig]:
        """
        Get inbox-specific configuration from database.
        
        Args:
            inbox_id: Chatwoot inbox ID
            
        Returns:
            InboxConfig instance if found, None otherwise
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT config_data 
                    FROM agent_configs 
                    WHERE config_type = 'inbox' AND inbox_id = $1
                    ORDER BY updated_at DESC 
                    LIMIT 1
                """
                record = await conn.fetchrow(query, inbox_id)
                
                if record:
                    config_data = record['config_data']
                    config_data['inbox_id'] = inbox_id
                    return InboxConfig(**config_data)
                return None
                
        except Exception as e:
            logger.error("Failed to get inbox configuration", 
                        inbox_id=inbox_id, error=str(e))
            raise QueryError(f"Inbox config query failed: {e}")
    
    async def save_global_config(self, config: AgentConfig) -> None:
        """
        Save global agent configuration to database.
        
        Args:
            config: AgentConfig instance to save
        """
        try:
            async with self.transaction() as conn:
                query = """
                    INSERT INTO agent_configs (config_type, config_data, updated_at)
                    VALUES ('global', $1, NOW())
                    ON CONFLICT (config_type, inbox_id) 
                    DO UPDATE SET config_data = $1, updated_at = NOW()
                """
                config_json = json.loads(config.json())
                await conn.execute(query, config_json)
                
                logger.info("Global configuration saved successfully")
                
        except Exception as e:
            logger.error("Failed to save global configuration", error=str(e))
            raise QueryError(f"Global config save failed: {e}")
    
    async def save_inbox_config(self, config: InboxConfig) -> None:
        """
        Save inbox-specific configuration to database.
        
        Args:
            config: InboxConfig instance to save
        """
        try:
            async with self.transaction() as conn:
                query = """
                    INSERT INTO agent_configs (config_type, inbox_id, config_data, updated_at)
                    VALUES ('inbox', $1, $2, NOW())
                    ON CONFLICT (config_type, inbox_id) 
                    DO UPDATE SET config_data = $2, updated_at = NOW()
                """
                config_json = json.loads(config.json())
                # Remove inbox_id from config_data since it's stored separately
                config_json.pop('inbox_id', None)
                await conn.execute(query, config.inbox_id, config_json)
                
                logger.info("Inbox configuration saved successfully", 
                           inbox_id=config.inbox_id)
                
        except Exception as e:
            logger.error("Failed to save inbox configuration", 
                        inbox_id=config.inbox_id, error=str(e))
            raise QueryError(f"Inbox config save failed: {e}")
    
    # Message Storage Methods
    
    async def store_message(
        self,
        contact_phone: str,
        conversation_id: int,
        role: str,
        content: str,
        message_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
        sent_at: Optional[datetime] = None
    ) -> int:
        """
        Store a message in the database.
        
        Args:
            contact_phone: Contact phone number
            conversation_id: Chatwoot conversation ID
            role: Message role (user, assistant, system)
            content: Message content
            message_type: Message type (text, image, audio, file)
            metadata: Additional message metadata
            sent_at: Message timestamp
            
        Returns:
            Message ID
        """
        try:
            async with self.transaction() as conn:
                query = """
                    INSERT INTO messages (contact_phone, conversation_id, role, content, message_type, metadata, sent_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                """
                message_id = await conn.fetchval(
                    query,
                    contact_phone,
                    conversation_id,
                    role,
                    content,
                    message_type,
                    json.dumps(metadata or {}),
                    sent_at or datetime.utcnow()
                )
                
                logger.debug("Message stored successfully", 
                           message_id=message_id,
                           contact_phone=contact_phone,
                           conversation_id=conversation_id)
                
                return message_id
                
        except Exception as e:
            logger.error("Failed to store message", 
                        contact_phone=contact_phone,
                        conversation_id=conversation_id,
                        error=str(e))
            raise QueryError(f"Message storage failed: {e}")
    
    async def get_conversation_history(
        self,
        contact_phone: str,
        conversation_id: Optional[int] = None,
        limit: int = 50,
        time_window_days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a contact.
        
        Args:
            contact_phone: Contact phone number
            conversation_id: Specific conversation ID (optional)
            limit: Maximum number of messages to return
            time_window_days: Limit to messages within N days
            
        Returns:
            List of message dictionaries
        """
        try:
            async with self.get_connection() as conn:
                conditions = ["contact_phone = $1"]
                params = [contact_phone]
                param_count = 1
                
                if conversation_id:
                    param_count += 1
                    conditions.append(f"conversation_id = ${param_count}")
                    params.append(conversation_id)
                
                if time_window_days:
                    param_count += 1
                    conditions.append(f"sent_at >= NOW() - INTERVAL '{time_window_days} days'")
                
                query = f"""
                    SELECT id, conversation_id, role, content, message_type, metadata, sent_at, created_at
                    FROM messages
                    WHERE {' AND '.join(conditions)}
                    ORDER BY sent_at DESC
                    LIMIT {limit}
                """
                
                records = await conn.fetch(query, *params)
                
                messages = []
                for record in records:
                    messages.append({
                        'id': record['id'],
                        'conversation_id': record['conversation_id'],
                        'role': record['role'],
                        'content': record['content'],
                        'message_type': record['message_type'],
                        'metadata': record['metadata'],
                        'sent_at': record['sent_at'].isoformat(),
                        'created_at': record['created_at'].isoformat()
                    })
                
                logger.debug("Retrieved conversation history",
                           contact_phone=contact_phone,
                           conversation_id=conversation_id,
                           message_count=len(messages))
                
                return messages
                
        except Exception as e:
            logger.error("Failed to get conversation history",
                        contact_phone=contact_phone,
                        conversation_id=conversation_id,
                        error=str(e))
            raise QueryError(f"Conversation history query failed: {e}")
    
    # Vector/Embedding Methods
    
    async def store_embedding(
        self,
        message_id: int,
        contact_phone: str,
        embedding: List[float],
        content_chunk: str,
        chunk_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store message embedding for RAG.
        
        Args:
            message_id: Reference to the message
            contact_phone: Contact phone number
            embedding: Vector embedding (1536 dimensions)
            content_chunk: Text content that was embedded
            chunk_index: Chunk index for large messages
            metadata: Additional embedding metadata
            
        Returns:
            Embedding ID
        """
        try:
            async with self.transaction() as conn:
                query = """
                    INSERT INTO embeddings (message_id, contact_phone, embedding, content_chunk, chunk_index, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                """
                embedding_id = await conn.fetchval(
                    query,
                    message_id,
                    contact_phone,
                    embedding,
                    content_chunk,
                    chunk_index,
                    json.dumps(metadata or {})
                )
                
                logger.debug("Embedding stored successfully",
                           embedding_id=embedding_id,
                           message_id=message_id,
                           contact_phone=contact_phone)
                
                return embedding_id
                
        except Exception as e:
            logger.error("Failed to store embedding",
                        message_id=message_id,
                        contact_phone=contact_phone,
                        error=str(e))
            raise QueryError(f"Embedding storage failed: {e}")
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        contact_phone: str,
        top_k: int = 4,
        similarity_threshold: float = 0.7,
        time_window_days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search for RAG.
        
        Args:
            query_embedding: Query vector embedding
            contact_phone: Contact phone number
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            time_window_days: Limit to embeddings within N days
            
        Returns:
            List of similar embeddings with metadata
        """
        try:
            async with self.get_connection() as conn:
                conditions = ["contact_phone = $2"]
                params = [query_embedding, contact_phone]
                param_count = 2
                
                if time_window_days:
                    param_count += 1
                    conditions.append(f"created_at >= NOW() - INTERVAL '{time_window_days} days'")
                
                # Using inner product for similarity (OpenAI embeddings are normalized)
                query = f"""
                    SELECT 
                        e.id,
                        e.message_id,
                        e.content_chunk,
                        e.chunk_index,
                        e.metadata,
                        e.created_at,
                        m.conversation_id,
                        m.role,
                        m.sent_at,
                        (e.embedding <#> $1) * -1 AS similarity
                    FROM embeddings e
                    JOIN messages m ON e.message_id = m.id
                    WHERE {' AND '.join(conditions)}
                    AND (e.embedding <#> $1) * -1 >= {similarity_threshold}
                    ORDER BY similarity DESC
                    LIMIT {top_k}
                """
                
                records = await conn.fetch(query, *params)
                
                results = []
                for record in records:
                    results.append({
                        'id': record['id'],
                        'message_id': record['message_id'],
                        'conversation_id': record['conversation_id'],
                        'content_chunk': record['content_chunk'],
                        'chunk_index': record['chunk_index'],
                        'similarity': float(record['similarity']),
                        'metadata': record['metadata'],
                        'message_role': record['role'],
                        'sent_at': record['sent_at'].isoformat(),
                        'created_at': record['created_at'].isoformat()
                    })
                
                logger.debug("Similarity search completed",
                           contact_phone=contact_phone,
                           results_count=len(results),
                           top_k=top_k)
                
                return results
                
        except Exception as e:
            logger.error("Failed to perform similarity search",
                        contact_phone=contact_phone,
                        error=str(e))
            raise QueryError(f"Similarity search failed: {e}")
    
    # Agent Memory Methods
    
    async def get_agent_memory(
        self,
        contact_phone: str,
        conversation_id: Optional[int] = None
    ) -> Optional[AgentMemory]:
        """
        Get agent memory for a contact.
        
        Args:
            contact_phone: Contact phone number
            conversation_id: Specific conversation ID (optional)
            
        Returns:
            AgentMemory instance if found, None otherwise
        """
        try:
            async with self.get_connection() as conn:
                if conversation_id:
                    query = """
                        SELECT * FROM agent_memory 
                        WHERE contact_phone = $1 AND conversation_id = $2
                        ORDER BY updated_at DESC LIMIT 1
                    """
                    record = await conn.fetchrow(query, contact_phone, conversation_id)
                else:
                    query = """
                        SELECT * FROM agent_memory 
                        WHERE contact_phone = $1
                        ORDER BY updated_at DESC LIMIT 1
                    """
                    record = await conn.fetchrow(query, contact_phone)
                
                if record:
                    return AgentMemory(
                        contact_phone=record['contact_phone'],
                        conversation_id=record['conversation_id'],
                        conversation_history=record['memory_data'].get('conversation_history', []),
                        metadata=record['memory_data'].get('metadata', {}),
                        created_at=record['created_at'],
                        updated_at=record['updated_at']
                    )
                return None
                
        except Exception as e:
            logger.error("Failed to get agent memory",
                        contact_phone=contact_phone,
                        conversation_id=conversation_id,
                        error=str(e))
            raise QueryError(f"Agent memory query failed: {e}")
    
    async def save_agent_memory(self, memory: AgentMemory) -> None:
        """
        Save agent memory to database.
        
        Args:
            memory: AgentMemory instance to save
        """
        try:
            async with self.transaction() as conn:
                query = """
                    INSERT INTO agent_memory (contact_phone, conversation_id, memory_data, last_interaction, updated_at)
                    VALUES ($1, $2, $3, NOW(), NOW())
                    ON CONFLICT (contact_phone, conversation_id)
                    DO UPDATE SET 
                        memory_data = $3, 
                        last_interaction = NOW(), 
                        updated_at = NOW()
                """
                
                memory_data = {
                    'conversation_history': memory.conversation_history,
                    'metadata': memory.metadata
                }
                
                await conn.execute(
                    query,
                    memory.contact_phone,
                    memory.conversation_id,
                    json.dumps(memory_data)
                )
                
                logger.debug("Agent memory saved successfully",
                           contact_phone=memory.contact_phone,
                           conversation_id=memory.conversation_id)
                
        except Exception as e:
            logger.error("Failed to save agent memory",
                        contact_phone=memory.contact_phone,
                        conversation_id=memory.conversation_id,
                        error=str(e))
            raise QueryError(f"Agent memory save failed: {e}")
    
    # Logging Methods
    
    async def log_agent_execution(self, log_entry: LogEntry) -> int:
        """
        Log agent execution details.
        
        Args:
            log_entry: LogEntry instance with execution details
            
        Returns:
            Log entry ID
        """
        try:
            async with self.transaction() as conn:
                query = """
                    INSERT INTO agent_logs (
                        conversation_id, contact_phone, user_query, tool_used, 
                        final_response, latency_ms, token_count, cost_usd, 
                        status, error_message, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING id
                """
                
                log_id = await conn.fetchval(
                    query,
                    log_entry.conversation_id,
                    log_entry.contact_phone,
                    log_entry.user_query,
                    log_entry.tool_used,
                    log_entry.final_response,
                    log_entry.latency_ms,
                    log_entry.token_count,
                    Decimal(str(log_entry.cost_usd)) if log_entry.cost_usd else None,
                    log_entry.status,
                    log_entry.error_message,
                    json.dumps(log_entry.metadata)
                )
                
                logger.debug("Agent execution logged successfully",
                           log_id=log_id,
                           conversation_id=log_entry.conversation_id)
                
                return log_id
                
        except Exception as e:
            logger.error("Failed to log agent execution",
                        conversation_id=log_entry.conversation_id,
                        error=str(e))
            raise QueryError(f"Agent execution logging failed: {e}")
    
    async def get_execution_stats(
        self,
        contact_phone: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get agent execution statistics.
        
        Args:
            contact_phone: Filter by contact phone (optional)
            days: Number of days to include in stats
            
        Returns:
            Dictionary with execution statistics
        """
        try:
            async with self.get_connection() as conn:
                conditions = [f"created_at >= NOW() - INTERVAL '{days} days'"]
                params = []
                
                if contact_phone:
                    conditions.append("contact_phone = $1")
                    params.append(contact_phone)
                
                where_clause = " AND ".join(conditions)
                
                query = f"""
                    SELECT 
                        COUNT(*) as total_executions,
                        COUNT(*) FILTER (WHERE status = 'success') as successful_executions,
                        COUNT(*) FILTER (WHERE status = 'error') as failed_executions,
                        AVG(latency_ms) as avg_latency_ms,
                        SUM(token_count) as total_tokens,
                        SUM(cost_usd) as total_cost_usd,
                        COUNT(DISTINCT tool_used) FILTER (WHERE tool_used IS NOT NULL) as tools_used_count
                    FROM agent_logs
                    WHERE {where_clause}
                """
                
                record = await conn.fetchrow(query, *params)
                
                stats = {
                    'total_executions': record['total_executions'] or 0,
                    'successful_executions': record['successful_executions'] or 0,
                    'failed_executions': record['failed_executions'] or 0,
                    'success_rate': 0.0,
                    'avg_latency_ms': float(record['avg_latency_ms'] or 0),
                    'total_tokens': record['total_tokens'] or 0,
                    'total_cost_usd': float(record['total_cost_usd'] or 0),
                    'tools_used_count': record['tools_used_count'] or 0,
                    'period_days': days
                }
                
                if stats['total_executions'] > 0:
                    stats['success_rate'] = stats['successful_executions'] / stats['total_executions']
                
                logger.debug("Execution statistics retrieved",
                           contact_phone=contact_phone,
                           **stats)
                
                return stats
                
        except Exception as e:
            logger.error("Failed to get execution statistics",
                        contact_phone=contact_phone,
                        error=str(e))
            raise QueryError(f"Execution statistics query failed: {e}")


# Global database service instance
_database_service: Optional[DatabaseService] = None


async def get_database_service() -> DatabaseService:
    """
    Get the global database service instance.
    
    Returns:
        Initialized DatabaseService instance
    """
    global _database_service
    if _database_service is None:
        _database_service = DatabaseService()
        await _database_service.initialize()
    return _database_service


async def close_database_service() -> None:
    """Close the global database service."""
    global _database_service
    if _database_service:
        await _database_service.close()
        _database_service = None