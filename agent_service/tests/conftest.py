"""
Pytest configuration and fixtures for the Chatwoot Agent Service tests.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, AsyncGenerator
from datetime import datetime, timedelta

import pytest
import asyncpg
from unittest.mock import AsyncMock, MagicMock

from src.utils.config import DatabaseConfig, ChatwootConfig, AppConfig
from src.services.database import DatabaseService
from src.services.chatwoot_api import ChatwootAPIClient
from src.handlers.webhook import WebhookHandler
from src.models.schemas import (
    WebhookPayload, Message, Contact, Conversation, Account, Sender,
    EventType, MessageType, ContentType, SenderType, ConversationStatus,
    AgentConfig, InboxConfig
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_database_config():
    """Test database configuration."""
    return DatabaseConfig(
        host="localhost",
        port=5432,
        database="test_chatwoot_agent",
        username="test_user",
        password="test_password",
        pool_min_size=1,
        pool_max_size=5
    )


@pytest.fixture
def test_chatwoot_config():
    """Test Chatwoot API configuration."""
    return ChatwootConfig(
        base_url="http://test-chatwoot:3000",
        api_token="test_api_token_123456789",
        webhook_secret="test_webhook_secret",
        rate_limit_per_minute=10,  # Lower limit for testing
        timeout_seconds=5,
        max_retries=2
    )


@pytest.fixture
def test_app_config(test_database_config, test_chatwoot_config):
    """Test application configuration."""
    return AppConfig(
        environment="test",
        debug=True,
        log_level="DEBUG",
        database=test_database_config,
        chatwoot=test_chatwoot_config,
        agent_enabled=True
    )


@pytest.fixture
def mock_database_service():
    """Mock database service for testing."""
    service = AsyncMock(spec=DatabaseService)
    
    # Mock health check
    service.health_check.return_value = {
        'status': 'healthy',
        'connection': True,
        'tables_found': ['messages', 'embeddings', 'agent_configs', 'agent_memory', 'agent_logs'],
        'required_tables': ['messages', 'embeddings', 'agent_configs', 'agent_memory', 'agent_logs'],
        'pgvector_enabled': True,
        'pool_size': 5,
        'pool_max_size': 20,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    # Mock configuration methods
    service.get_global_config.return_value = AgentConfig()
    service.get_inbox_config.return_value = None
    
    # Mock message storage
    service.store_message.return_value = 1
    service.get_conversation_history.return_value = []
    
    # Mock embedding operations
    service.store_embedding.return_value = 1
    service.similarity_search.return_value = []
    
    # Mock agent memory
    service.get_agent_memory.return_value = None
    
    # Mock logging
    service.log_agent_execution.return_value = 1
    service.get_execution_stats.return_value = {
        'total_executions': 0,
        'successful_executions': 0,
        'failed_executions': 0,
        'success_rate': 0.0,
        'avg_latency_ms': 0.0,
        'total_tokens': 0,
        'total_cost_usd': 0.0,
        'tools_used_count': 0,
        'period_days': 7
    }
    
    return service


@pytest.fixture
def mock_chatwoot_client():
    """Mock Chatwoot API client for testing."""
    client = AsyncMock(spec=ChatwootAPIClient)
    
    # Mock health check
    client.health_check.return_value = {
        'status': 'healthy',
        'api_accessible': True,
        'authentication': 'valid',
        'response_time_ms': 100,
        'rate_limit_available': True,
        'base_url': 'http://test-chatwoot:3000',
        'timestamp': datetime.utcnow().isoformat()
    }
    
    # Mock message operations
    client.send_message.return_value = {
        'id': 123,
        'content': 'Test response',
        'message_type': 'outgoing',
        'content_type': 'text',
        'created_at': datetime.utcnow().isoformat()
    }
    
    client.get_conversation.return_value = {
        'id': 12345,
        'status': 'open',
        'assignee_id': None,
        'inbox_id': 42,
        'contact_id': 555
    }
    
    client.get_conversation_messages.return_value = []
    
    # Mock loop prevention methods
    client.is_agent_message.return_value = False
    client.should_process_message.return_value = True
    
    return client


@pytest.fixture
def sample_webhook_payload():
    """Sample webhook payload for testing."""
    return {
        "event": "message_created",
        "timestamp": "2025-08-14T13:45:00Z",
        "message": {
            "id": 987654,
            "account_id": 1,
            "inbox_id": 42,
            "conversation_id": 12345,
            "content": "Hello, I need help with my order #1234",
            "message_type": "incoming",
            "content_type": "text",
            "content_attributes": {},
            "created_at": "2025-08-14T13:44:59Z",
            "sender": {
                "id": 555,
                "name": "John Doe",
                "avatar_url": None,
                "type": "contact"
            },
            "attachments": [],
            "source_id": "contact:+1234567890"
        },
        "contact": {
            "id": 555,
            "name": "John Doe",
            "phone_number": "+1234567890",
            "email": None,
            "additional_attributes": {}
        },
        "conversation": {
            "id": 12345,
            "status": "open",
            "assignee_id": None,
            "inbox_id": 42,
            "contact_id": 555,
            "custom_attributes": {},
            "labels": ["whatsapp"],
            "created_at": "2025-08-14T13:00:00Z"
        },
        "account": {
            "id": 1,
            "name": "Test Company"
        }
    }


@pytest.fixture
def agent_message_webhook_payload():
    """Sample webhook payload for agent-generated message (should be filtered out)."""
    return {
        "event": "message_created",
        "timestamp": "2025-08-14T13:45:00Z",
        "message": {
            "id": 987655,
            "account_id": 1,
            "inbox_id": 42,
            "conversation_id": 12345,
            "content": "This is an agent response",
            "message_type": "outgoing",
            "content_type": "text",
            "content_attributes": {
                "agent_processed": True,
                "echo_id": "agent_12345_987655"
            },
            "created_at": "2025-08-14T13:45:00Z",
            "sender": {
                "id": 1,
                "name": "Agent Bot",
                "avatar_url": None,
                "type": "agent_bot"
            },
            "attachments": [],
            "source_id": None
        },
        "contact": {
            "id": 555,
            "name": "John Doe",
            "phone_number": "+1234567890",
            "email": None,
            "additional_attributes": {}
        },
        "conversation": {
            "id": 12345,
            "status": "open",
            "assignee_id": 1,
            "inbox_id": 42,
            "contact_id": 555,
            "custom_attributes": {},
            "labels": ["whatsapp"],
            "created_at": "2025-08-14T13:00:00Z"
        },
        "account": {
            "id": 1,
            "name": "Test Company"
        }
    }


@pytest.fixture
def parsed_webhook_payload(sample_webhook_payload):
    """Parsed and validated webhook payload."""
    return WebhookPayload(**sample_webhook_payload)


@pytest.fixture
def webhook_handler(mock_chatwoot_client, mock_database_service):
    """Webhook handler with mocked dependencies."""
    return WebhookHandler(
        chatwoot_client=mock_chatwoot_client,
        database_service=mock_database_service
    )


# Environment variable fixtures

@pytest.fixture(autouse=True)
def set_test_env_vars():
    """Set test environment variables."""
    test_env_vars = {
        'ENVIRONMENT': 'test',
        'DEBUG': 'true',
        'LOG_LEVEL': 'DEBUG',
        'DATABASE_HOST': 'localhost',
        'DATABASE_PORT': '5432',
        'DATABASE_NAME': 'test_chatwoot_agent',
        'DATABASE_USER': 'test_user',
        'DATABASE_PASSWORD': 'test_password',
        'CHATWOOT_BASE_URL': 'http://test-chatwoot:3000',
        'CHATWOOT_API_TOKEN': 'test_api_token_123456789',
        'CHATWOOT_WEBHOOK_SECRET': 'test_webhook_secret',
        'OPENAI_API_KEY': 'test_openai_key_123456789',
        'REDIS_HOST': 'localhost',
        'REDIS_PORT': '6379'
    }
    
    # Set environment variables
    original_env = {}
    for key, value in test_env_vars.items():
        original_env[key] = os.getenv(key)
        os.environ[key] = value
    
    yield
    
    # Restore original environment variables
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


# Enhanced test data fixtures for comprehensive testing

@pytest.fixture
def test_csv_data(temp_dir):
    """Create test CSV data for spreadsheet tool testing."""
    csv_files = {}
    
    # Service orders CSV
    service_orders_path = temp_dir / "service_orders.csv"
    service_orders_data = [
        "order_id,customer_phone,status,service_type,scheduled_date,technician",
        "1234,+1234567890,scheduled,installation,2025-08-15,John Tech",
        "1235,+1987654321,completed,maintenance,2025-08-14,Jane Tech",
        "1236,+1122334455,pending,repair,2025-08-16,Bob Tech",
        "1237,+1234567890,in_progress,upgrade,2025-08-15,Alice Tech"
    ]
    service_orders_path.write_text("\n".join(service_orders_data))
    csv_files['service_orders'] = str(service_orders_path)
    
    # Appointments CSV
    appointments_path = temp_dir / "appointments.csv"
    appointments_data = [
        "appointment_id,customer_phone,date,time,service,status,notes",
        "A001,+1234567890,2025-08-15,09:00,consultation,confirmed,First visit",
        "A002,+1987654321,2025-08-15,14:00,follow_up,completed,Check-up",
        "A003,+1122334455,2025-08-16,10:30,installation,pending,New customer",
        "A004,+1234567890,2025-08-17,16:00,maintenance,confirmed,Regular service"
    ]
    appointments_path.write_text("\n".join(appointments_data))
    csv_files['appointments'] = str(appointments_path)
    
    # Price list CSV
    price_list_path = temp_dir / "price_list.csv"
    price_list_data = [
        "service_type,price_usd,description,category",
        "installation,299.99,Basic installation service,installation",
        "maintenance,149.99,Regular maintenance check,maintenance",
        "repair,199.99,Standard repair service,repair",
        "emergency_repair,399.99,Emergency repair service,repair",
        "consultation,99.99,Technical consultation,consultation"
    ]
    price_list_path.write_text("\n".join(price_list_data))
    csv_files['price_list'] = str(price_list_path)
    
    return csv_files


@pytest.fixture
def mock_rag_embeddings():
    """Mock embeddings for RAG testing."""
    return [
        {
            'id': 1,
            'content': 'Previous conversation about installation scheduling',
            'embedding': [0.1] * 1536,  # Mock embedding vector
            'similarity': 0.85,
            'metadata': {
                'conversation_id': 12340,
                'timestamp': '2025-08-14T10:00:00Z',
                'message_type': 'user_query'
            }
        },
        {
            'id': 2,
            'content': 'Service order status inquiry from last week',
            'embedding': [0.2] * 1536,
            'similarity': 0.78,
            'metadata': {
                'conversation_id': 12341,
                'timestamp': '2025-08-13T15:30:00Z',
                'message_type': 'support_request'
            }
        }
    ]


@pytest.fixture
def multimodal_test_payloads():
    """Test payloads for multimodal processing."""
    return {
        'audio_message': {
            'event': 'message_created',
            'message': {
                'id': 987655,
                'content': '',
                'content_type': 'audio',
                'attachments': [{
                    'id': 1001,
                    'file_url': 'https://test.com/audio.mp3',
                    'file_type': 'audio/mp3',
                    'data_url': None
                }],
                'content_attributes': {
                    'transcript': 'Hello, can you check my appointment for tomorrow?'
                }
            }
        },
        'image_message': {
            'event': 'message_created',
            'message': {
                'id': 987656,
                'content': 'Can you help me with this error?',
                'content_type': 'image',
                'attachments': [{
                    'id': 1002,
                    'file_url': 'https://test.com/error_screenshot.png',
                    'file_type': 'image/png',
                    'data_url': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
                }]
            }
        }
    }


@pytest.fixture
def governance_test_scenarios():
    """Test scenarios for governance controls."""
    return {
        'price_sensitive': {
            'user_query': 'How much does installation cost?',
            'agent_response': 'Our installation service costs $299.99 for basic setup.',
            'should_require_confirmation': True,
            'detected_price': 299.99
        },
        'emergency_response': {
            'user_query': 'I need emergency repair service immediately!',
            'agent_response': 'I can schedule emergency repair for $399.99. Let me check availability.',
            'should_require_confirmation': True,
            'detected_price': 399.99
        },
        'general_inquiry': {
            'user_query': 'What services do you offer?',
            'agent_response': 'We offer installation, maintenance, repair, and consultation services.',
            'should_require_confirmation': False,
            'detected_price': None
        }
    }


@pytest.fixture
def performance_test_data():
    """Test data for performance benchmarks."""
    return {
        'message_batches': [
            [f"Test message {i}" for i in range(10)],  # Small batch
            [f"Test message {i}" for i in range(50)],  # Medium batch
            [f"Test message {i}" for i in range(100)]  # Large batch
        ],
        'response_time_targets': {
            'simple_query': 1000,  # 1 second max
            'rag_query': 3000,     # 3 seconds max
            'complex_query': 5000   # 5 seconds max
        },
        'throughput_targets': {
            'messages_per_minute': 30,
            'concurrent_conversations': 10
        }
    }


# Utility functions for tests

def create_test_message(
    message_id: int = 987654,
    conversation_id: int = 12345,
    content: str = "Test message",
    message_type: str = "incoming",
    sender_type: str = "contact",
    content_attributes: Dict[str, Any] = None,
    content_type: str = "text",
    attachments: List[Dict] = None
) -> Dict[str, Any]:
    """Create test message data with enhanced options."""
    return {
        "id": message_id,
        "account_id": 1,
        "inbox_id": 42,
        "conversation_id": conversation_id,
        "content": content,
        "message_type": message_type,
        "content_type": content_type,
        "content_attributes": content_attributes or {},
        "created_at": datetime.utcnow().isoformat(),
        "sender": {
            "id": 555 if sender_type == "contact" else 1,
            "name": "John Doe" if sender_type == "contact" else "Agent Bot",
            "avatar_url": None,
            "type": sender_type
        },
        "attachments": attachments or [],
        "source_id": "contact:+1234567890" if sender_type == "contact" else None
    }


def create_test_webhook_payload(
    event: str = "message_created",
    message_data: Dict[str, Any] = None,
    conversation_id: int = 12345,
    contact_id: int = 555,
    phone_number: str = "+1234567890",
    contact_name: str = "John Doe",
    inbox_id: int = 42,
    conversation_status: str = "open",
    labels: List[str] = None
) -> Dict[str, Any]:
    """Create test webhook payload with enhanced customization."""
    if message_data is None:
        message_data = create_test_message(conversation_id=conversation_id)
    
    return {
        "event": event,
        "timestamp": datetime.utcnow().isoformat(),
        "message": message_data,
        "contact": {
            "id": contact_id,
            "name": contact_name,
            "phone_number": phone_number,
            "email": None,
            "additional_attributes": {}
        },
        "conversation": {
            "id": conversation_id,
            "status": conversation_status,
            "assignee_id": None,
            "inbox_id": inbox_id,
            "contact_id": contact_id,
            "custom_attributes": {},
            "labels": labels or ["whatsapp"],
            "created_at": datetime.utcnow().isoformat()
        },
        "account": {
            "id": 1,
            "name": "Test Company"
        }
    }


def create_agent_message_payload(
    conversation_id: int = 12345,
    content: str = "Agent response",
    echo_id: str = None
) -> Dict[str, Any]:
    """Create agent-generated message payload for loop prevention testing."""
    echo_id = echo_id or f"agent_{conversation_id}_{int(time.time())}"
    
    message_data = create_test_message(
        conversation_id=conversation_id,
        content=content,
        message_type="outgoing",
        sender_type="agent_bot",
        content_attributes={
            "agent_processed": True,
            "echo_id": echo_id
        }
    )
    
    return create_test_webhook_payload(
        message_data=message_data,
        conversation_id=conversation_id
    )


def create_price_sensitive_message(
    conversation_id: int = 12345,
    price: float = 299.99,
    service: str = "installation"
) -> str:
    """Create message containing price information for governance testing."""
    return f"Our {service} service costs ${price:.2f}. Would you like to proceed?"


def assert_response_time(start_time: float, max_time_ms: int) -> None:
    """Assert that response time is within acceptable limits."""
    elapsed_ms = (time.time() - start_time) * 1000
    assert elapsed_ms <= max_time_ms, f"Response time {elapsed_ms:.2f}ms exceeded limit {max_time_ms}ms"


def mock_openai_response(content: str = "Test response", tools_used: List[str] = None) -> Dict[str, Any]:
    """Create mock OpenAI response for agent testing."""
    return {
        "output": content,
        "intermediate_steps": [
            (type('Step', (), {'tool': tool, 'tool_input': {}, 'log': f"Used {tool}"})(), f"Result from {tool}")
            for tool in (tools_used or [])
        ]
    }


# Async test helpers

async def wait_for(coro, timeout=1.0):
    """Wait for coroutine with timeout."""
    return await asyncio.wait_for(coro, timeout=timeout)


async def assert_async_performance(coro, max_time_ms: int):
    """Assert that async operation completes within time limit."""
    start_time = time.time()
    result = await coro
    elapsed_ms = (time.time() - start_time) * 1000
    assert elapsed_ms <= max_time_ms, f"Async operation took {elapsed_ms:.2f}ms, exceeded {max_time_ms}ms limit"
    return result


class AsyncMockService:
    """Enhanced async mock service for testing."""
    
    def __init__(self, service_name: str = "test_service"):
        self.service_name = service_name
        self.call_count = 0
        self.call_history = []
        self.responses = {}
        self.errors = {}
        
    def set_response(self, method_name: str, response: Any):
        """Set mock response for a method."""
        self.responses[method_name] = response
        
    def set_error(self, method_name: str, error: Exception):
        """Set mock error for a method."""
        self.errors[method_name] = error
        
    async def mock_method(self, method_name: str, *args, **kwargs):
        """Generic mock method handler."""
        self.call_count += 1
        self.call_history.append({
            'method': method_name,
            'args': args,
            'kwargs': kwargs,
            'timestamp': datetime.utcnow()
        })
        
        if method_name in self.errors:
            raise self.errors[method_name]
            
        return self.responses.get(method_name, None)
        
    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.call_history = []
        self.responses = {}
        self.errors = {}


class DatabaseTestHelper:
    """Helper class for database testing scenarios."""
    
    @staticmethod
    def create_test_tables() -> List[str]:
        """SQL commands to create test tables."""
        return [
            """CREATE TABLE IF NOT EXISTS test_messages (
                id SERIAL PRIMARY KEY,
                contact_phone TEXT NOT NULL,
                conversation_id BIGINT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sent_at TIMESTAMP WITH TIME ZONE DEFAULT now()
            )""",
            """CREATE TABLE IF NOT EXISTS test_embeddings (
                id SERIAL PRIMARY KEY,
                message_id BIGINT,
                contact_phone TEXT NOT NULL,
                embedding VECTOR(1536),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
                metadata JSONB
            )"""
        ]
    
    @staticmethod
    def create_test_data() -> Dict[str, List[Dict]]:
        """Create test data for database operations."""
        return {
            'messages': [
                {
                    'contact_phone': '+1234567890',
                    'conversation_id': 12345,
                    'role': 'user',
                    'content': 'Hello, I need help with my order'
                },
                {
                    'contact_phone': '+1234567890',
                    'conversation_id': 12345,
                    'role': 'assistant',
                    'content': 'I can help you with that. What is your order number?'
                }
            ],
            'embeddings': [
                {
                    'message_id': 1,
                    'contact_phone': '+1234567890',
                    'embedding': [0.1] * 1536,
                    'metadata': {'conversation_id': 12345, 'type': 'user_query'}
                }
            ]
        }


# Security test helpers

def create_malicious_payloads() -> List[Dict[str, Any]]:
    """Create payloads for security testing."""
    return [
        # SQL injection attempt
        create_test_webhook_payload(
            message_data=create_test_message(
                content="'; DROP TABLE messages; --"
            )
        ),
        # XSS attempt
        create_test_webhook_payload(
            message_data=create_test_message(
                content="<script>alert('xss')</script>"
            )
        ),
        # Large payload
        create_test_webhook_payload(
            message_data=create_test_message(
                content="A" * 10000  # Very long content
            )
        ),
        # Invalid data types
        {
            "event": "message_created",
            "message": {
                "id": "not_an_integer",  # Should be int
                "content": None,  # Should be string
                "conversation_id": "invalid"
            }
        }
    ]