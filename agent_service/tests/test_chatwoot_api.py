"""
Tests for the Chatwoot API client.
"""

import asyncio
import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from src.services.chatwoot_api import (
    ChatwootAPIClient, RateLimiter,
    ChatwootAPIError, AuthenticationError, RateLimitError, 
    ValidationError, NetworkError
)
from src.models.schemas import WebhookPayload, MessageType, ContentType, SenderType
from src.utils.config import ChatwootConfig


class TestRateLimiter:
    """Tests for the RateLimiter class."""
    
    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_requests_per_minute=60)
        assert limiter.max_tokens == 60
        assert limiter.refill_rate == 1.0  # 60/60
        assert limiter.tokens == 60.0
    
    def test_can_proceed_with_tokens(self):
        """Test can_proceed when tokens are available."""
        limiter = RateLimiter(max_requests_per_minute=60)
        assert limiter.can_proceed() is True
        assert limiter.tokens > 0
    
    def test_consume_token_success(self):
        """Test successful token consumption."""
        limiter = RateLimiter(max_requests_per_minute=60)
        initial_tokens = limiter.tokens
        
        result = limiter.consume_token()
        assert result is True
        assert limiter.tokens == initial_tokens - 1.0
    
    def test_consume_token_exhausted(self):
        """Test token consumption when exhausted."""
        limiter = RateLimiter(max_requests_per_minute=1)
        limiter.tokens = 0.5  # Not enough for a full request
        
        result = limiter.consume_token()
        assert result is False
        assert limiter.tokens == 0.5  # Unchanged
    
    def test_token_refill(self):
        """Test token refill over time."""
        limiter = RateLimiter(max_requests_per_minute=60)
        limiter.tokens = 30.0
        
        # Simulate time passage
        with patch('time.time', return_value=limiter.last_refill + 30):  # 30 seconds
            limiter._refill_tokens()
            # Should have refilled 30 tokens (1 per second)
            assert limiter.tokens == 60.0  # Capped at max
    
    def test_get_wait_time_no_wait(self):
        """Test get_wait_time when no wait is needed."""
        limiter = RateLimiter(max_requests_per_minute=60)
        wait_time = limiter.get_wait_time()
        assert wait_time == 0.0
    
    def test_get_wait_time_with_wait(self):
        """Test get_wait_time when wait is needed."""
        limiter = RateLimiter(max_requests_per_minute=60)
        limiter.tokens = 0.5  # Not enough for a request
        
        wait_time = limiter.get_wait_time()
        assert wait_time > 0  # Should need to wait
        assert wait_time == (1.0 - 0.5) / 1.0  # (needed - available) / refill_rate


class TestChatwootAPIClient:
    """Tests for the ChatwootAPIClient class."""
    
    @pytest.fixture
    def api_client(self, test_chatwoot_config):
        """Create API client instance."""
        return ChatwootAPIClient(config=test_chatwoot_config)
    
    @pytest.fixture
    def mock_response(self):
        """Mock HTTP response."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"id": 123, "status": "success"}
        return response
    
    def test_initialization_default_config(self):
        """Test initialization with default config."""
        with patch('src.services.chatwoot_api.get_chatwoot_config') as mock_get_config:
            mock_config = ChatwootConfig(api_token="test_token")
            mock_get_config.return_value = mock_config
            
            client = ChatwootAPIClient()
            assert client.config is mock_config
            assert client.headers['api_access_token'] == "test_token"
    
    def test_initialization_custom_config(self, test_chatwoot_config):
        """Test initialization with custom config."""
        client = ChatwootAPIClient(config=test_chatwoot_config)
        assert client.config is test_chatwoot_config
        assert client.headers['api_access_token'] == test_chatwoot_config.api_token
    
    @pytest.mark.asyncio
    async def test_context_manager(self, api_client):
        """Test async context manager."""
        with patch.object(api_client, '_ensure_client') as mock_ensure:
            with patch.object(api_client, 'close') as mock_close:
                async with api_client as client:
                    assert client is api_client
                    mock_ensure.assert_called_once()
                
                mock_close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ensure_client(self, api_client):
        """Test HTTP client initialization."""
        assert api_client._client is None
        
        await api_client._ensure_client()
        
        assert api_client._client is not None
        assert isinstance(api_client._client, httpx.AsyncClient)
    
    @pytest.mark.asyncio
    async def test_close(self, api_client):
        """Test HTTP client closure."""
        # Set up client
        api_client._client = AsyncMock()
        
        await api_client.close()
        
        api_client._client.aclose.assert_called_once()
        assert api_client._client is None
    
    @pytest.mark.asyncio
    async def test_make_request_success(self, api_client, mock_response):
        """Test successful API request."""
        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        api_client._client = mock_client
        
        response = await api_client._make_request("GET", "/api/test")
        
        assert response is mock_response
        mock_client.request.assert_called_once_with(
            method="GET",
            url="http://test-chatwoot:3000/api/test",
            json=None,
            params=None
        )
    
    @pytest.mark.asyncio
    async def test_make_request_rate_limiting(self, api_client, mock_response):
        """Test rate limiting in API requests."""
        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        api_client._client = mock_client
        
        # Exhaust rate limiter
        api_client.rate_limiter.tokens = 0.0
        
        with patch('asyncio.sleep') as mock_sleep:
            response = await api_client._make_request("GET", "/api/test")
            
            # Should have waited for rate limit
            mock_sleep.assert_called_once()
            assert response is mock_response
    
    @pytest.mark.asyncio
    async def test_make_request_authentication_error(self, api_client):
        """Test authentication error handling."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_client.request.return_value = mock_response
        api_client._client = mock_client
        
        with pytest.raises(AuthenticationError, match="Invalid API token"):
            await api_client._make_request("GET", "/api/test")
    
    @pytest.mark.asyncio
    async def test_make_request_rate_limit_error(self, api_client):
        """Test rate limit error handling."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_client.request.return_value = mock_response
        api_client._client = mock_client
        
        with pytest.raises(RateLimitError, match="API rate limit exceeded"):
            await api_client._make_request("GET", "/api/test")
    
    @pytest.mark.asyncio
    async def test_make_request_client_error(self, api_client):
        """Test client error handling."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"message": "Bad request"}
        mock_client.request.return_value = mock_response
        api_client._client = mock_client
        
        with pytest.raises(ValidationError, match="Client error: 400"):
            await api_client._make_request("GET", "/api/test")
    
    @pytest.mark.asyncio
    async def test_make_request_server_error_retry(self, api_client, mock_response):
        """Test server error with retry logic."""
        mock_client = AsyncMock()
        
        # First call returns 500, second call succeeds
        mock_error_response = MagicMock()
        mock_error_response.status_code = 500
        mock_client.request.side_effect = [mock_error_response, mock_response]
        api_client._client = mock_client
        
        with patch('asyncio.sleep') as mock_sleep:
            response = await api_client._make_request("GET", "/api/test")
            
            assert response is mock_response
            assert mock_client.request.call_count == 2
            mock_sleep.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_request_timeout_retry(self, api_client, mock_response):
        """Test timeout with retry logic."""
        mock_client = AsyncMock()
        
        # First call times out, second call succeeds
        mock_client.request.side_effect = [
            httpx.TimeoutException("Request timeout"),
            mock_response
        ]
        api_client._client = mock_client
        
        with patch('asyncio.sleep') as mock_sleep:
            response = await api_client._make_request("GET", "/api/test")
            
            assert response is mock_response
            assert mock_client.request.call_count == 2
            mock_sleep.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_request_connection_error_retry(self, api_client, mock_response):
        """Test connection error with retry logic."""
        mock_client = AsyncMock()
        
        # First call has connection error, second call succeeds
        mock_client.request.side_effect = [
            httpx.ConnectError("Connection failed"),
            mock_response
        ]
        api_client._client = mock_client
        
        with patch('asyncio.sleep') as mock_sleep:
            response = await api_client._make_request("GET", "/api/test")
            
            assert response is mock_response
            assert mock_client.request.call_count == 2
            mock_sleep.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_request_max_retries_exceeded(self, api_client):
        """Test max retries exceeded."""
        mock_client = AsyncMock()
        
        # All calls return 500
        mock_error_response = MagicMock()
        mock_error_response.status_code = 500
        mock_client.request.return_value = mock_error_response
        api_client._client = mock_client
        
        with patch('asyncio.sleep'):
            with pytest.raises(NetworkError, match="Server error: 500"):
                await api_client._make_request("GET", "/api/test")
        
        # Should have made initial call + max retries
        expected_calls = 1 + api_client.config.max_retries
        assert mock_client.request.call_count == expected_calls


class TestMessageOperations:
    """Tests for message-related API operations."""
    
    @pytest.fixture
    def api_client_with_mock(self, api_client):
        """API client with mocked _make_request method."""
        with patch.object(api_client, '_make_request') as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": 123,
                "content": "Test message",
                "message_type": "outgoing",
                "created_at": datetime.utcnow().isoformat()
            }
            mock_request.return_value = mock_response
            yield api_client, mock_request
    
    @pytest.mark.asyncio
    async def test_send_message_success(self, api_client_with_mock):
        """Test successful message sending."""
        client, mock_request = api_client_with_mock
        
        result = await client.send_message(
            account_id=1,
            conversation_id=12345,
            content="Hello, world!",
            echo_id="test_echo_123"
        )
        
        assert result["id"] == 123
        assert result["content"] == "Test message"
        
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v1/accounts/1/conversations/12345/messages" in call_args[0][1]
        
        # Check payload
        payload = call_args[1]["data"]
        assert payload["content"] == "Hello, world!"
        assert payload["message_type"] == "outgoing"
        assert payload["content_attributes"]["agent_processed"] is True
        assert payload["content_attributes"]["echo_id"] == "test_echo_123"
    
    @pytest.mark.asyncio
    async def test_send_message_with_custom_attributes(self, api_client_with_mock):
        """Test message sending with custom attributes."""
        client, mock_request = api_client_with_mock
        
        await client.send_message(
            account_id=1,
            conversation_id=12345,
            content="Custom message",
            content_attributes={"confidence": 0.95, "tool": "custom_tool"}
        )
        
        payload = mock_request.call_args[1]["data"]
        assert payload["content_attributes"]["agent_processed"] is True
        assert payload["content_attributes"]["confidence"] == 0.95
        assert payload["content_attributes"]["tool"] == "custom_tool"
    
    @pytest.mark.asyncio
    async def test_get_conversation_success(self, api_client_with_mock):
        """Test successful conversation retrieval."""
        client, mock_request = api_client_with_mock
        
        mock_request.return_value.json.return_value = {
            "id": 12345,
            "status": "open",
            "assignee_id": None,
            "inbox_id": 42
        }
        
        result = await client.get_conversation(account_id=1, conversation_id=12345)
        
        assert result["id"] == 12345
        assert result["status"] == "open"
        
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0] == "GET"
        assert "/api/v1/accounts/1/conversations/12345" in call_args[0][1]
    
    @pytest.mark.asyncio
    async def test_get_conversation_messages_success(self, api_client_with_mock):
        """Test successful conversation messages retrieval."""
        client, mock_request = api_client_with_mock
        
        mock_messages = [
            {"id": 1, "content": "Hello"},
            {"id": 2, "content": "Hi there!"}
        ]
        mock_request.return_value.json.return_value = mock_messages
        
        result = await client.get_conversation_messages(
            account_id=1,
            conversation_id=12345,
            limit=20
        )
        
        assert len(result) == 2
        assert result[0]["content"] == "Hello"
        
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0] == "GET"
        assert "/api/v1/accounts/1/conversations/12345/messages" in call_args[0][1]
        
        # Check query parameters
        params = call_args[1]["params"]
        assert params["limit"] == 20
    
    @pytest.mark.asyncio
    async def test_update_conversation_success(self, api_client_with_mock):
        """Test successful conversation update."""
        client, mock_request = api_client_with_mock
        
        mock_request.return_value.json.return_value = {
            "id": 12345,
            "status": "resolved",
            "assignee_id": 1
        }
        
        result = await client.update_conversation(
            account_id=1,
            conversation_id=12345,
            status="resolved",
            assignee_id=1,
            labels=["support", "resolved"]
        )
        
        assert result["status"] == "resolved"
        assert result["assignee_id"] == 1
        
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0] == "PATCH"
        assert "/api/v1/accounts/1/conversations/12345" in call_args[0][1]
        
        # Check payload
        payload = call_args[1]["data"]
        assert payload["status"] == "resolved"
        assert payload["assignee_id"] == 1
        assert payload["labels"] == ["support", "resolved"]
    
    @pytest.mark.asyncio
    async def test_update_conversation_no_parameters(self, api_client_with_mock):
        """Test conversation update with no parameters."""
        client, mock_request = api_client_with_mock
        
        with pytest.raises(ValidationError, match="No update parameters provided"):
            await client.update_conversation(account_id=1, conversation_id=12345)


class TestContactOperations:
    """Tests for contact-related API operations."""
    
    @pytest.mark.asyncio
    async def test_get_contact_success(self, api_client):
        """Test successful contact retrieval."""
        with patch.object(api_client, '_make_request') as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": 555,
                "name": "John Doe",
                "phone_number": "+1234567890",
                "email": "john@example.com"
            }
            mock_request.return_value = mock_response
            
            result = await api_client.get_contact(account_id=1, contact_id=555)
            
            assert result["id"] == 555
            assert result["name"] == "John Doe"
            
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "GET"
            assert "/api/v1/accounts/1/contacts/555" in call_args[0][1]


class TestUtilityMethods:
    """Tests for utility methods."""
    
    def test_is_agent_message_with_agent_processed_flag(self, api_client):
        """Test agent message detection with agent_processed flag."""
        message_data = {
            "id": 123,
            "content": "Agent response",
            "message_type": "outgoing",
            "content_attributes": {"agent_processed": True}
        }
        
        result = api_client.is_agent_message(message_data)
        assert result is True
    
    def test_is_agent_message_with_echo_id(self, api_client):
        """Test agent message detection with echo_id."""
        message_data = {
            "id": 123,
            "content": "Agent response",
            "message_type": "outgoing",
            "content_attributes": {"echo_id": "agent_123_456"}
        }
        
        result = api_client.is_agent_message(message_data)
        assert result is True
    
    def test_is_agent_message_outgoing_type(self, api_client):
        """Test agent message detection with outgoing type."""
        message_data = {
            "id": 123,
            "content": "Agent response",
            "message_type": "outgoing",
            "sender": {"type": "agent_bot"},
            "content_attributes": {}
        }
        
        result = api_client.is_agent_message(message_data)
        assert result is True
    
    def test_is_agent_message_user_message(self, api_client):
        """Test user message detection."""
        message_data = {
            "id": 123,
            "content": "User message",
            "message_type": "incoming",
            "sender": {"type": "contact"},
            "content_attributes": {}
        }
        
        result = api_client.is_agent_message(message_data)
        assert result is False
    
    def test_should_process_message_valid(self, api_client, parsed_webhook_payload):
        """Test should_process_message with valid message."""
        result = api_client.should_process_message(parsed_webhook_payload)
        assert result is True
    
    def test_should_process_message_wrong_event_type(self, api_client, sample_webhook_payload):
        """Test should_process_message with wrong event type."""
        sample_webhook_payload["event"] = "conversation_updated"
        payload = WebhookPayload(**sample_webhook_payload)
        
        result = api_client.should_process_message(payload)
        assert result is False
    
    def test_should_process_message_outgoing_type(self, api_client, sample_webhook_payload):
        """Test should_process_message with outgoing message."""
        sample_webhook_payload["message"]["message_type"] = "outgoing"
        payload = WebhookPayload(**sample_webhook_payload)
        
        result = api_client.should_process_message(payload)
        assert result is False
    
    def test_should_process_message_non_text_content(self, api_client, sample_webhook_payload):
        """Test should_process_message with non-text content."""
        sample_webhook_payload["message"]["content_type"] = "image"
        payload = WebhookPayload(**sample_webhook_payload)
        
        result = api_client.should_process_message(payload)
        assert result is False
    
    def test_should_process_message_empty_content(self, api_client, sample_webhook_payload):
        """Test should_process_message with empty content."""
        sample_webhook_payload["message"]["content"] = ""
        payload = WebhookPayload(**sample_webhook_payload)
        
        result = api_client.should_process_message(payload)
        assert result is False
    
    def test_should_process_message_agent_generated(self, api_client, sample_webhook_payload):
        """Test should_process_message with agent-generated message."""
        sample_webhook_payload["message"]["content_attributes"] = {"agent_processed": True}
        payload = WebhookPayload(**sample_webhook_payload)
        
        result = api_client.should_process_message(payload)
        assert result is False


class TestHealthCheck:
    """Tests for health check functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, api_client):
        """Test successful health check."""
        with patch.object(api_client, '_make_request') as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {"id": 1, "name": "Test User"}
            mock_request.return_value = mock_response
            
            result = await api_client.health_check()
            
            assert result["status"] == "healthy"
            assert result["api_accessible"] is True
            assert result["authentication"] == "valid"
            assert "response_time_ms" in result
            assert result["base_url"] == api_client.config.base_url
    
    @pytest.mark.asyncio
    async def test_health_check_authentication_failure(self, api_client):
        """Test health check with authentication failure."""
        with patch.object(api_client, '_make_request', side_effect=AuthenticationError("Invalid token")):
            result = await api_client.health_check()
            
            assert result["status"] == "unhealthy"
            assert result["api_accessible"] is False
            assert result["authentication"] == "invalid"
            assert "Invalid token" in result["error"]
    
    @pytest.mark.asyncio
    async def test_health_check_network_failure(self, api_client):
        """Test health check with network failure."""
        with patch.object(api_client, '_make_request', side_effect=NetworkError("Connection failed")):
            result = await api_client.health_check()
            
            assert result["status"] == "unhealthy"
            assert result["api_accessible"] is False
            assert result["authentication"] == "unknown"
            assert "Connection failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_health_check_unexpected_error(self, api_client):
        """Test health check with unexpected error."""
        with patch.object(api_client, '_make_request', side_effect=Exception("Unexpected error")):
            result = await api_client.health_check()
            
            assert result["status"] == "unhealthy"
            assert result["api_accessible"] is False
            assert result["authentication"] == "unknown"
            assert "Unexpected error" in result["error"]


class TestGlobalClientManagement:
    """Tests for global client management functions."""
    
    @pytest.mark.asyncio
    async def test_get_chatwoot_client(self):
        """Test get_chatwoot_client function."""
        from src.services.chatwoot_api import get_chatwoot_client, _api_client
        
        # Clear global client
        import src.services.chatwoot_api
        src.services.chatwoot_api._api_client = None
        
        client = await get_chatwoot_client()
        assert isinstance(client, ChatwootAPIClient)
        
        # Second call should return same instance
        client2 = await get_chatwoot_client()
        assert client2 is client
    
    @pytest.mark.asyncio
    async def test_close_chatwoot_client(self):
        """Test close_chatwoot_client function."""
        from src.services.chatwoot_api import close_chatwoot_client
        import src.services.chatwoot_api
        
        # Set up mock client
        mock_client = AsyncMock()
        src.services.chatwoot_api._api_client = mock_client
        
        await close_chatwoot_client()
        
        mock_client.close.assert_called_once()
        assert src.services.chatwoot_api._api_client is None
    
    @pytest.mark.asyncio
    async def test_close_chatwoot_client_no_client(self):
        """Test closing client when none exists."""
        from src.services.chatwoot_api import close_chatwoot_client
        import src.services.chatwoot_api
        
        src.services.chatwoot_api._api_client = None
        
        # Should not raise exception
        await close_chatwoot_client()


class TestErrorScenarios:
    """Tests for various error scenarios."""
    
    @pytest.mark.asyncio
    async def test_request_with_invalid_json_response(self, api_client):
        """Test request with invalid JSON response."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_client.request.return_value = mock_response
        api_client._client = mock_client
        
        # Should still return the response object even if JSON parsing fails
        response = await api_client._make_request("GET", "/api/test")
        assert response is mock_response
    
    @pytest.mark.asyncio
    async def test_send_message_with_network_error(self, api_client):
        """Test send_message with network error."""
        with patch.object(api_client, '_make_request', side_effect=NetworkError("Network failed")):
            with pytest.raises(NetworkError):
                await api_client.send_message(
                    account_id=1,
                    conversation_id=12345,
                    content="Test message"
                )


class TestRateLimitingIntegration:
    """Integration tests for rate limiting."""
    
    @pytest.mark.asyncio
    async def test_multiple_requests_with_rate_limiting(self, api_client):
        """Test multiple requests with rate limiting."""
        # Set very low rate limit for testing
        api_client.rate_limiter = RateLimiter(max_requests_per_minute=2)
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_client.request.return_value = mock_response
        api_client._client = mock_client
        
        # First two requests should go through quickly
        await api_client._make_request("GET", "/api/test1")
        await api_client._make_request("GET", "/api/test2")
        
        # Third request should be rate limited
        with patch('asyncio.sleep') as mock_sleep:
            await api_client._make_request("GET", "/api/test3")
            mock_sleep.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rate_limit_token_exhaustion(self, api_client):
        """Test rate limit when tokens are completely exhausted."""
        # Exhaust all tokens
        api_client.rate_limiter.tokens = 0.0
        
        mock_client = AsyncMock()
        api_client._client = mock_client
        
        with patch('asyncio.sleep') as mock_sleep:
            with patch.object(api_client.rate_limiter, 'consume_token', side_effect=[False, True]):
                await api_client._make_request("GET", "/api/test")
                mock_sleep.assert_called_once()


class TestURLConstruction:
    """Tests for URL construction and endpoint handling."""
    
    @pytest.mark.asyncio
    async def test_url_construction_with_leading_slash(self, api_client):
        """Test URL construction with leading slash in endpoint."""
        with patch.object(api_client, '_ensure_client'):
            api_client._client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            api_client._client.request.return_value = mock_response
            
            await api_client._make_request("GET", "/api/test")
            
            call_args = api_client._client.request.call_args
            assert call_args[1]['url'] == "http://test-chatwoot:3000/api/test"
    
    @pytest.mark.asyncio
    async def test_url_construction_without_leading_slash(self, api_client):
        """Test URL construction without leading slash in endpoint."""
        with patch.object(api_client, '_ensure_client'):
            api_client._client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            api_client._client.request.return_value = mock_response
            
            await api_client._make_request("GET", "api/test")
            
            call_args = api_client._client.request.call_args
            assert call_args[1]['url'] == "http://test-chatwoot:3000/api/test"