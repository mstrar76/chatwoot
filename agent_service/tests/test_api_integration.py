"""
Integration tests for API interactions.

Tests cover:
- Chatwoot API client operations
- Rate limiting and retry logic
- Authentication and error handling
- Message sending workflows
- WebSocket connections
- API response validation
"""

import pytest
import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.services.chatwoot_api import ChatwootAPIClient, get_chatwoot_client
from src.utils.config import ChatwootConfig
from src.models.schemas import Message, Conversation, Contact
from tests.conftest import (
    test_chatwoot_config, assert_response_time, AsyncMockService
)


class TestChatwootAPIClient:
    """Test core Chatwoot API client functionality."""
    
    @pytest.fixture
    async def api_client(self, test_chatwoot_config):
        """Create API client with test configuration."""
        client = ChatwootAPIClient(test_chatwoot_config)
        
        # Mock HTTP session
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            await client.initialize()
            client._session = mock_session
            
            return client, mock_session
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_client_initialization(self, test_chatwoot_config):
        """Test API client initialization."""
        # Arrange
        client = ChatwootAPIClient(test_chatwoot_config)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            # Act
            await client.initialize()
            
            # Assert
            assert client._initialized is True
            assert client._session is not None
            mock_session_class.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_check(self, api_client):
        """Test API health check functionality."""
        # Arrange
        client, mock_session = api_client
        
        # Mock successful health check response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_response.headers = {"X-RateLimit-Remaining": "59"}
        
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Act
        start_time = time.time()
        health = await client.health_check()
        
        # Assert
        assert_response_time(start_time, 5000)  # Should complete within 5 seconds
        assert health['status'] == 'healthy'
        assert health['api_accessible'] is True
        assert health['authentication'] == 'valid'
        assert health['rate_limit_available'] is True
        assert 'response_time_ms' in health
        assert 'base_url' in health
        assert 'timestamp' in health
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_send_message_success(self, api_client):
        """Test successful message sending."""
        # Arrange
        client, mock_session = api_client
        
        # Mock successful message send response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "id": 987654,
            "content": "Thank you for your message",
            "message_type": "outgoing",
            "content_type": "text",
            "created_at": datetime.utcnow().isoformat()
        }
        
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Act
        result = await client.send_message(
            account_id=1,
            conversation_id=12345,
            content="Thank you for your message",
            message_type="outgoing"
        )
        
        # Assert
        assert result['id'] == 987654
        assert result['content'] == "Thank you for your message"
        assert result['message_type'] == "outgoing"
        
        # Verify API call
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert '/api/v1/accounts/1/conversations/12345/messages' in call_args[1]['url']
        
        # Verify headers
        headers = call_args[1]['headers']
        assert 'api_access_token' in headers
        assert headers['Content-Type'] == 'application/json'
        
        # Verify payload
        json_data = call_args[1]['json']
        assert json_data['content'] == "Thank you for your message"
        assert json_data['message_type'] == "outgoing"
        assert json_data['content_type'] == "text"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_send_message_with_attachments(self, api_client):
        """Test sending message with attachments."""
        # Arrange
        client, mock_session = api_client
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "id": 987655,
            "content": "Here's the document you requested",
            "message_type": "outgoing",
            "attachments": [{"id": 1001, "file_url": "https://example.com/doc.pdf"}]
        }
        
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Act
        result = await client.send_message(
            account_id=1,
            conversation_id=12345,
            content="Here's the document you requested",
            message_type="outgoing",
            attachments=[{"file_url": "https://example.com/doc.pdf", "file_type": "application/pdf"}]
        )
        
        # Assert
        assert result['id'] == 987655
        assert len(result['attachments']) == 1
        
        # Verify attachment in payload
        call_args = mock_session.post.call_args
        json_data = call_args[1]['json']
        assert 'attachments' in json_data
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_conversation(self, api_client):
        """Test retrieving conversation details."""
        # Arrange
        client, mock_session = api_client
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "id": 12345,
            "status": "open",
            "assignee_id": None,
            "inbox_id": 42,
            "contact_id": 555,
            "custom_attributes": {},
            "labels": ["whatsapp"],
            "created_at": datetime.utcnow().isoformat()
        }
        
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Act
        conversation = await client.get_conversation(account_id=1, conversation_id=12345)
        
        # Assert
        assert conversation['id'] == 12345
        assert conversation['status'] == "open"
        assert conversation['inbox_id'] == 42
        
        # Verify API call
        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        assert '/api/v1/accounts/1/conversations/12345' in call_args[1]['url']
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_conversation_messages(self, api_client):
        """Test retrieving conversation messages."""
        # Arrange
        client, mock_session = api_client
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "payload": [
                {
                    "id": 1001,
                    "content": "Hello, I need help",
                    "message_type": "incoming",
                    "created_at": (datetime.utcnow() - timedelta(minutes=5)).isoformat()
                },
                {
                    "id": 1002,
                    "content": "I can help you with that",
                    "message_type": "outgoing",
                    "created_at": (datetime.utcnow() - timedelta(minutes=4)).isoformat()
                }
            ],
            "meta": {
                "count": 2,
                "current_page": 1
            }
        }
        
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Act
        messages = await client.get_conversation_messages(
            account_id=1,
            conversation_id=12345,
            limit=20
        )
        
        # Assert
        assert len(messages) == 2
        assert messages[0]['id'] == 1001
        assert messages[0]['message_type'] == "incoming"
        assert messages[1]['id'] == 1002
        assert messages[1]['message_type'] == "outgoing"
        
        # Verify API call with pagination
        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        params = call_args[1]['params']
        assert params['per_page'] == 20
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_update_conversation_status(self, api_client):
        """Test updating conversation status."""
        # Arrange
        client, mock_session = api_client
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "id": 12345,
            "status": "resolved"
        }
        
        mock_session.patch.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.patch.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Act
        result = await client.update_conversation_status(
            account_id=1,
            conversation_id=12345,
            status="resolved"
        )
        
        # Assert
        assert result['status'] == "resolved"
        
        # Verify API call
        mock_session.patch.assert_called_once()
        call_args = mock_session.patch.call_args
        json_data = call_args[1]['json']
        assert json_data['status'] == "resolved"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_is_agent_message_detection(self, api_client):
        """Test detection of agent-generated messages."""
        # Arrange
        client, _ = api_client
        
        # Test cases
        test_cases = [
            # Agent message
            {
                "content_attributes": {"agent_processed": True},
                "sender": {"type": "agent_bot"},
                "expected": True
            },
            # User message
            {
                "content_attributes": {},
                "sender": {"type": "contact"},
                "expected": False
            },
            # Message with echo ID
            {
                "content_attributes": {"echo_id": "agent_12345_987654"},
                "sender": {"type": "user"},
                "expected": True
            }
        ]
        
        # Act & Assert
        for case in test_cases:
            is_agent = await client.is_agent_message(case)
            assert is_agent == case["expected"]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_should_process_message_filtering(self, api_client):
        """Test message processing filtering logic."""
        # Arrange
        client, _ = api_client
        
        # Test cases
        test_cases = [
            # Should process - incoming user message
            {
                "message_type": "incoming",
                "sender": {"type": "contact"},
                "content_attributes": {},
                "expected": True
            },
            # Should not process - agent message
            {
                "message_type": "outgoing",
                "sender": {"type": "agent_bot"},
                "content_attributes": {"agent_processed": True},
                "expected": False
            },
            # Should not process - activity message
            {
                "message_type": "activity",
                "sender": {"type": "user"},
                "content_attributes": {},
                "expected": False
            }
        ]
        
        # Act & Assert
        for case in test_cases:
            should_process = await client.should_process_message(case)
            assert should_process == case["expected"]


class TestChatwootAPIRateLimiting:
    """Test rate limiting and retry logic."""
    
    @pytest.fixture
    async def rate_limited_client(self, test_chatwoot_config):
        """Create API client with rate limiting configuration."""
        # Configure for aggressive rate limiting
        config = test_chatwoot_config
        config.rate_limit_per_minute = 10  # Low limit for testing
        config.max_retries = 3
        
        client = ChatwootAPIClient(config)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            await client.initialize()
            client._session = mock_session
            
            return client, mock_session
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, rate_limited_client):
        """Test rate limit enforcement."""
        # Arrange
        client, mock_session = rate_limited_client
        
        # Mock rate limit exceeded response
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_response.json.return_value = {"error": "Rate limit exceeded"}
        
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Act & Assert
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await client.send_message(
                account_id=1,
                conversation_id=12345,
                content="Test message",
                message_type="outgoing"
            )
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_retry_on_transient_errors(self, rate_limited_client):
        """Test retry logic for transient errors."""
        # Arrange
        client, mock_session = rate_limited_client
        
        # Mock responses: first fails, second succeeds
        responses = [
            # First call - network error
            AsyncMock(),
            # Second call - success
            AsyncMock()
        ]
        
        responses[0].status = 502
        responses[0].json.side_effect = Exception("Network error")
        
        responses[1].status = 200
        responses[1].json.return_value = {
            "id": 987654,
            "content": "Test message",
            "message_type": "outgoing"
        }
        
        mock_session.post.side_effect = [
            AsyncMock(__aenter__=AsyncMock(return_value=responses[0]), __aexit__=AsyncMock()),
            AsyncMock(__aenter__=AsyncMock(return_value=responses[1]), __aexit__=AsyncMock())
        ]
        
        # Act
        result = await client.send_message(
            account_id=1,
            conversation_id=12345,
            content="Test message",
            message_type="outgoing"
        )
        
        # Assert
        assert result['id'] == 987654
        assert mock_session.post.call_count == 2  # Retried once
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rate_limit_tracking(self, rate_limited_client):
        """Test rate limit tracking and throttling."""
        # Arrange
        client, mock_session = rate_limited_client
        
        # Mock successful responses with rate limit headers
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {
            "X-RateLimit-Limit": "60",
            "X-RateLimit-Remaining": "5",  # Low remaining
            "X-RateLimit-Reset": str(int(time.time()) + 3600)
        }
        mock_response.json.return_value = {"id": 987654}
        
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Act
        await client.send_message(
            account_id=1,
            conversation_id=12345,
            content="Test message",
            message_type="outgoing"
        )
        
        # Check rate limit status
        rate_limit_info = await client.get_rate_limit_status()
        
        # Assert
        assert rate_limit_info['remaining'] == 5
        assert rate_limit_info['limit'] == 60
        assert 'reset_time' in rate_limit_info
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_requests_throttling(self, rate_limited_client):
        """Test throttling of concurrent requests."""
        # Arrange
        client, mock_session = rate_limited_client
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"id": 987654}
        
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Act - Send many concurrent requests
        tasks = []
        for i in range(20):  # More than rate limit
            task = client.send_message(
                account_id=1,
                conversation_id=12345 + i,
                content=f"Test message {i}",
                message_type="outgoing"
            )
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assert
        # Some requests should be throttled or delayed
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        # Should have some successful requests
        assert len(successful_results) > 0
        
        # May have some failures due to rate limiting
        # The exact behavior depends on implementation


class TestChatwootAPIAuthentication:
    """Test authentication and authorization."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_token_authentication(self, test_chatwoot_config):
        """Test API token authentication."""
        # Arrange
        client = ChatwootAPIClient(test_chatwoot_config)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            await client.initialize()
            client._session = mock_session
            
            # Mock successful authenticated request
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"status": "authenticated"}
            
            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Act
            await client.get_conversation(account_id=1, conversation_id=12345)
            
            # Assert - Verify auth header is included
            mock_session.get.assert_called_once()
            call_args = mock_session.get.call_args
            headers = call_args[1]['headers']
            assert 'api_access_token' in headers
            assert headers['api_access_token'] == test_chatwoot_config.api_token
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_authentication_failure(self, test_chatwoot_config):
        """Test handling of authentication failures."""
        # Arrange
        client = ChatwootAPIClient(test_chatwoot_config)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            await client.initialize()
            client._session = mock_session
            
            # Mock authentication failure
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_response.json.return_value = {"error": "Unauthorized"}
            
            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Act & Assert
            with pytest.raises(Exception, match="Unauthorized|401"):
                await client.get_conversation(account_id=1, conversation_id=12345)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_forbidden_access(self, test_chatwoot_config):
        """Test handling of forbidden access."""
        # Arrange
        client = ChatwootAPIClient(test_chatwoot_config)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            await client.initialize()
            client._session = mock_session
            
            # Mock forbidden response
            mock_response = AsyncMock()
            mock_response.status = 403
            mock_response.json.return_value = {"error": "Forbidden"}
            
            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Act & Assert
            with pytest.raises(Exception, match="Forbidden|403"):
                await client.get_conversation(account_id=1, conversation_id=12345)


class TestChatwootAPIErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    async def api_client(self, test_chatwoot_config):
        """Create API client for error testing."""
        client = ChatwootAPIClient(test_chatwoot_config)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            await client.initialize()
            client._session = mock_session
            
            return client, mock_session
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, api_client):
        """Test handling of network timeouts."""
        # Arrange
        client, mock_session = api_client
        
        # Mock timeout error
        mock_session.post.side_effect = asyncio.TimeoutError("Request timeout")
        
        # Act & Assert
        with pytest.raises(asyncio.TimeoutError):
            await client.send_message(
                account_id=1,
                conversation_id=12345,
                content="Test message",
                message_type="outgoing"
            )
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_connection_error_handling(self, api_client):
        """Test handling of connection errors."""
        # Arrange
        client, mock_session = api_client
        
        # Mock connection error
        mock_session.post.side_effect = aiohttp.ClientConnectionError("Connection failed")
        
        # Act & Assert
        with pytest.raises(aiohttp.ClientConnectionError):
            await client.send_message(
                account_id=1,
                conversation_id=12345,
                content="Test message",
                message_type="outgoing"
            )
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_invalid_json_response(self, api_client):
        """Test handling of invalid JSON responses."""
        # Arrange
        client, mock_session = api_client
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text.return_value = "Invalid response"
        
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Act & Assert
        with pytest.raises(json.JSONDecodeError):
            await client.send_message(
                account_id=1,
                conversation_id=12345,
                content="Test message",
                message_type="outgoing"
            )
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_server_error_handling(self, api_client):
        """Test handling of server errors."""
        # Arrange
        client, mock_session = api_client
        
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.json.return_value = {"error": "Internal server error"}
        
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Act & Assert
        with pytest.raises(Exception, match="500|Internal server error"):
            await client.send_message(
                account_id=1,
                conversation_id=12345,
                content="Test message",
                message_type="outgoing"
            )
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_malformed_request_handling(self, api_client):
        """Test handling of malformed requests."""
        # Arrange
        client, mock_session = api_client
        
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json.return_value = {
            "error": "Bad Request",
            "details": {"content": ["can't be blank"]}
        }
        
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Act & Assert
        with pytest.raises(Exception, match="400|Bad Request"):
            await client.send_message(
                account_id=1,
                conversation_id=12345,
                content="",  # Empty content should trigger error
                message_type="outgoing"
            )
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_resource_not_found_handling(self, api_client):
        """Test handling of resource not found errors."""
        # Arrange
        client, mock_session = api_client
        
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.json.return_value = {"error": "Conversation not found"}
        
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Act & Assert
        with pytest.raises(Exception, match="404|not found"):
            await client.get_conversation(account_id=1, conversation_id=99999)


class TestChatwootAPIPerformance:
    """Test API performance characteristics."""
    
    @pytest.fixture
    async def performance_client(self, test_chatwoot_config):
        """Create API client for performance testing."""
        client = ChatwootAPIClient(test_chatwoot_config)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            await client.initialize()
            client._session = mock_session
            
            return client, mock_session
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_request_performance(self, performance_client):
        """Test performance of single API requests."""
        # Arrange
        client, mock_session = performance_client
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"id": 987654}
        
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Act
        start_time = time.time()
        await client.send_message(
            account_id=1,
            conversation_id=12345,
            content="Performance test message",
            message_type="outgoing"
        )
        
        # Assert
        assert_response_time(start_time, 1000)  # Should complete within 1 second
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self, performance_client):
        """Test performance under concurrent load."""
        # Arrange
        client, mock_session = performance_client
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"id": 987654}
        
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Act - Send 10 concurrent messages
        tasks = []
        for i in range(10):
            task = client.send_message(
                account_id=1,
                conversation_id=12345 + i,
                content=f"Concurrent message {i}",
                message_type="outgoing"
            )
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        
        # Assert
        assert_response_time(start_time, 3000)  # Should complete within 3 seconds
        assert len(results) == 10
        assert all(r['id'] == 987654 for r in results)
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sustained_load_performance(self, performance_client):
        """Test performance under sustained load."""
        # Arrange
        client, mock_session = performance_client
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"id": 987654}
        
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Act - Send messages continuously for 30 seconds
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < 30:  # 30 seconds
            await client.send_message(
                account_id=1,
                conversation_id=12345 + request_count,
                content=f"Sustained load message {request_count}",
                message_type="outgoing"
            )
            request_count += 1
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)
        
        # Assert
        total_time = time.time() - start_time
        requests_per_second = request_count / total_time
        
        # Should maintain reasonable throughput
        assert requests_per_second > 5  # At least 5 requests per second
        assert request_count > 100  # Should have sent many requests


class TestAPIClientUtilities:
    """Test API client utility functions."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_chatwoot_client_singleton(self, test_chatwoot_config):
        """Test get_chatwoot_client singleton pattern."""
        with patch('src.services.chatwoot_api.get_chatwoot_config', return_value=test_chatwoot_config), \
             patch('aiohttp.ClientSession'):
            
            # Act
            client1 = await get_chatwoot_client()
            client2 = await get_chatwoot_client()
            
            # Assert
            assert client1 is client2  # Should be same instance
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_request_id_tracking(self, test_chatwoot_config):
        """Test request ID tracking for debugging."""
        # Arrange
        client = ChatwootAPIClient(test_chatwoot_config)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            await client.initialize()
            client._session = mock_session
            
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {"X-Request-ID": "req_123456"}
            mock_response.json.return_value = {"id": 987654}
            
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Act
            result = await client.send_message(
                account_id=1,
                conversation_id=12345,
                content="Test message",
                message_type="outgoing"
            )
            
            # Assert
            # Request ID should be tracked for debugging
            assert hasattr(client, '_last_request_id')
            # Implementation would store request ID for logging/debugging
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_response_validation(self, test_chatwoot_config):
        """Test validation of API responses."""
        # Arrange
        client = ChatwootAPIClient(test_chatwoot_config)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            await client.initialize()
            client._session = mock_session
            
            # Mock invalid response structure
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                # Missing required fields like 'id'
                "content": "Test message"
            }
            
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Act & Assert
            # Should validate response structure
            result = await client.send_message(
                account_id=1,
                conversation_id=12345,
                content="Test message",
                message_type="outgoing"
            )
            
            # Implementation should handle missing fields gracefully
            assert isinstance(result, dict)