"""
Tests for the webhook handler.
"""

import hmac
import hashlib
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.handlers.webhook import (
    WebhookHandler, WebhookError, ValidationError, SecurityError, ProcessingError
)
from src.models.schemas import WebhookPayload, MessageType, ContentType, SenderType
from src.services.chatwoot_api import ChatwootAPIClient
from src.services.database import DatabaseService


class TestWebhookHandler:
    """Tests for the WebhookHandler class."""
    
    @pytest.fixture
    def webhook_handler_with_mocks(self, mock_chatwoot_client, mock_database_service):
        """Webhook handler with mocked dependencies."""
        return WebhookHandler(
            chatwoot_client=mock_chatwoot_client,
            database_service=mock_database_service
        )
    
    def test_initialization_with_dependencies(self, mock_chatwoot_client, mock_database_service):
        """Test initialization with provided dependencies."""
        handler = WebhookHandler(
            chatwoot_client=mock_chatwoot_client,
            database_service=mock_database_service
        )
        
        assert handler.chatwoot_client is mock_chatwoot_client
        assert handler.database_service is mock_database_service
    
    def test_initialization_without_dependencies(self):
        """Test initialization without dependencies."""
        handler = WebhookHandler()
        
        assert handler.chatwoot_client is None
        assert handler.database_service is None
    
    @pytest.mark.asyncio
    async def test_ensure_dependencies(self, webhook_handler):
        """Test dependency initialization."""
        # Initially None
        assert webhook_handler.chatwoot_client is None
        assert webhook_handler.database_service is None
        
        with patch('src.handlers.webhook.get_chatwoot_client') as mock_get_client:
            with patch('src.handlers.webhook.get_database_service') as mock_get_db:
                mock_client = AsyncMock()
                mock_db = AsyncMock()
                mock_get_client.return_value = mock_client
                mock_get_db.return_value = mock_db
                
                await webhook_handler._ensure_dependencies()
                
                assert webhook_handler.chatwoot_client is mock_client
                assert webhook_handler.database_service is mock_db


class TestWebhookSignatureVerification:
    """Tests for webhook signature verification."""
    
    def test_verify_webhook_signature_valid(self, webhook_handler):
        """Test valid webhook signature verification."""
        payload = b'{"test": "data"}'
        secret = "test_secret"
        
        # Calculate expected signature
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        result = webhook_handler.verify_webhook_signature(
            payload, f"sha256={expected_signature}", secret
        )
        
        assert result is True
    
    def test_verify_webhook_signature_valid_without_prefix(self, webhook_handler):
        """Test valid signature without sha256= prefix."""
        payload = b'{"test": "data"}'
        secret = "test_secret"
        
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        result = webhook_handler.verify_webhook_signature(
            payload, expected_signature, secret
        )
        
        assert result is True
    
    def test_verify_webhook_signature_invalid(self, webhook_handler):
        """Test invalid webhook signature."""
        payload = b'{"test": "data"}'
        secret = "test_secret"
        wrong_signature = "invalid_signature"
        
        result = webhook_handler.verify_webhook_signature(
            payload, wrong_signature, secret
        )
        
        assert result is False
    
    def test_verify_webhook_signature_no_secret(self, webhook_handler):
        """Test signature verification when no secret is configured."""
        payload = b'{"test": "data"}'
        
        # No secret configured should return True (allow)
        result = webhook_handler.verify_webhook_signature(payload, "any_signature", None)
        assert result is True
    
    def test_verify_webhook_signature_no_signature_with_secret(self, webhook_handler):
        """Test missing signature when secret is configured."""
        payload = b'{"test": "data"}'
        secret = "test_secret"
        
        result = webhook_handler.verify_webhook_signature(payload, None, secret)
        assert result is False
    
    def test_verify_webhook_signature_exception_handling(self, webhook_handler):
        """Test signature verification exception handling."""
        payload = b'{"test": "data"}'
        secret = None  # This will cause an exception in hmac.new
        signature = "any_signature"
        
        # Should handle exception gracefully
        result = webhook_handler.verify_webhook_signature(payload, signature, secret)
        assert result is False


class TestWebhookPayloadParsing:
    """Tests for webhook payload parsing."""
    
    def test_parse_webhook_payload_success(self, webhook_handler, sample_webhook_payload):
        """Test successful webhook payload parsing."""
        payload = webhook_handler.parse_webhook_payload(sample_webhook_payload)
        
        assert isinstance(payload, WebhookPayload)
        assert payload.event == "message_created"
        assert payload.message is not None
        assert payload.message.id == 987654
        assert payload.message.content == "Hello, I need help with my order #1234"
        assert payload.contact is not None
        assert payload.conversation is not None
        assert payload.account is not None
    
    def test_parse_webhook_payload_with_string_timestamp(self, webhook_handler, sample_webhook_payload):
        """Test payload parsing with string timestamp."""
        # Ensure timestamp is string format
        sample_webhook_payload["timestamp"] = "2025-08-14T13:45:00Z"
        
        payload = webhook_handler.parse_webhook_payload(sample_webhook_payload)
        
        assert isinstance(payload, WebhookPayload)
        assert payload.timestamp is not None
    
    def test_parse_webhook_payload_invalid_data(self, webhook_handler):
        """Test parsing invalid webhook payload."""
        invalid_payload = {
            "event": "invalid_event",
            "message": {
                "id": "not_an_integer",  # Should be int
                "content": "test"
                # Missing required fields
            }
        }
        
        with pytest.raises(ValidationError, match="Invalid webhook payload"):
            webhook_handler.parse_webhook_payload(invalid_payload)
    
    def test_parse_webhook_payload_missing_required_fields(self, webhook_handler):
        """Test parsing payload with missing required fields."""
        invalid_payload = {
            "event": "message_created"
            # Missing account, which is required
        }
        
        with pytest.raises(ValidationError):
            webhook_handler.parse_webhook_payload(invalid_payload)
    
    def test_parse_webhook_payload_unexpected_error(self, webhook_handler, sample_webhook_payload):
        """Test handling of unexpected parsing errors."""
        # Mock WebhookPayload to raise unexpected error
        with patch('src.handlers.webhook.WebhookPayload', side_effect=Exception("Unexpected error")):
            with pytest.raises(ValidationError, match="Failed to parse webhook payload"):
                webhook_handler.parse_webhook_payload(sample_webhook_payload)


class TestAgentMessageDetection:
    """Tests for agent message detection (loop prevention)."""
    
    def test_is_agent_generated_message_with_agent_processed_flag(self, webhook_handler):
        """Test detection with agent_processed flag."""
        message = MagicMock()
        message.id = 123
        message.conversation_id = 12345
        message.content_attributes = {"agent_processed": True}
        message.message_type = MessageType.INCOMING
        message.sender.type = SenderType.CONTACT
        
        result = webhook_handler.is_agent_generated_message(message)
        assert result is True
    
    def test_is_agent_generated_message_with_echo_id(self, webhook_handler):
        """Test detection with echo_id."""
        message = MagicMock()
        message.id = 123
        message.conversation_id = 12345
        message.content_attributes = {"echo_id": "agent_123_456"}
        message.message_type = MessageType.INCOMING
        message.sender.type = SenderType.CONTACT
        
        result = webhook_handler.is_agent_generated_message(message)
        assert result is True
    
    def test_is_agent_generated_message_outgoing_type(self, webhook_handler):
        """Test detection with outgoing message type."""
        message = MagicMock()
        message.id = 123
        message.conversation_id = 12345
        message.content_attributes = {}
        message.message_type = MessageType.OUTGOING
        message.sender.type = SenderType.CONTACT
        
        result = webhook_handler.is_agent_generated_message(message)
        assert result is True
    
    def test_is_agent_generated_message_agent_bot_sender(self, webhook_handler):
        """Test detection with agent_bot sender."""
        message = MagicMock()
        message.id = 123
        message.conversation_id = 12345
        message.content_attributes = {}
        message.message_type = MessageType.INCOMING
        message.sender.type = SenderType.AGENT_BOT
        
        result = webhook_handler.is_agent_generated_message(message)
        assert result is True
    
    def test_is_agent_generated_message_user_outgoing(self, webhook_handler):
        """Test detection with user sender and outgoing type."""
        message = MagicMock()
        message.id = 123
        message.conversation_id = 12345
        message.content_attributes = {}
        message.message_type = MessageType.OUTGOING
        message.sender.type = SenderType.USER
        
        result = webhook_handler.is_agent_generated_message(message)
        assert result is True
    
    def test_is_agent_generated_message_user_message(self, webhook_handler):
        """Test detection with genuine user message."""
        message = MagicMock()
        message.id = 123
        message.conversation_id = 12345
        message.content_attributes = {}
        message.message_type = MessageType.INCOMING
        message.sender.type = SenderType.CONTACT
        
        result = webhook_handler.is_agent_generated_message(message)
        assert result is False


class TestMessageFiltering:
    """Tests for message filtering logic."""
    
    def test_should_process_message_valid(self, webhook_handler, parsed_webhook_payload):
        """Test should process valid message."""
        should_process, reason = webhook_handler.should_process_message(parsed_webhook_payload)
        
        assert should_process is True
        assert "should be processed" in reason
    
    def test_should_process_message_wrong_event_type(self, webhook_handler, sample_webhook_payload):
        """Test filtering wrong event type."""
        sample_webhook_payload["event"] = "conversation_updated"
        payload = WebhookPayload(**sample_webhook_payload)
        
        should_process, reason = webhook_handler.should_process_message(payload)
        
        assert should_process is False
        assert "not message_created" in reason
    
    def test_should_process_message_no_message_data(self, webhook_handler):
        """Test filtering payload without message data."""
        payload_data = {
            "event": "message_created",
            "account": {"id": 1, "name": "Test"},
            "conversation": {
                "id": 12345,
                "status": "open",
                "inbox_id": 42,
                "contact_id": 555,
                "labels": [],
                "created_at": datetime.utcnow().isoformat()
            }
        }
        payload = WebhookPayload(**payload_data)
        
        should_process, reason = webhook_handler.should_process_message(payload)
        
        assert should_process is False
        assert "No message data" in reason
    
    def test_should_process_message_outgoing_type(self, webhook_handler, sample_webhook_payload):
        """Test filtering outgoing messages."""
        sample_webhook_payload["message"]["message_type"] = "outgoing"
        payload = WebhookPayload(**sample_webhook_payload)
        
        should_process, reason = webhook_handler.should_process_message(payload)
        
        assert should_process is False
        assert "not incoming" in reason
    
    def test_should_process_message_non_text_content(self, webhook_handler, sample_webhook_payload):
        """Test filtering non-text content."""
        sample_webhook_payload["message"]["content_type"] = "image"
        payload = WebhookPayload(**sample_webhook_payload)
        
        should_process, reason = webhook_handler.should_process_message(payload)
        
        assert should_process is False
        assert "not text" in reason
    
    def test_should_process_message_empty_content(self, webhook_handler, sample_webhook_payload):
        """Test filtering empty content."""
        sample_webhook_payload["message"]["content"] = ""
        payload = WebhookPayload(**sample_webhook_payload)
        
        should_process, reason = webhook_handler.should_process_message(payload)
        
        assert should_process is False
        assert "empty" in reason
    
    def test_should_process_message_whitespace_content(self, webhook_handler, sample_webhook_payload):
        """Test filtering whitespace-only content."""
        sample_webhook_payload["message"]["content"] = "   \n\t   "
        payload = WebhookPayload(**sample_webhook_payload)
        
        should_process, reason = webhook_handler.should_process_message(payload)
        
        assert should_process is False
        assert "empty" in reason
    
    def test_should_process_message_agent_generated(self, webhook_handler, agent_message_webhook_payload):
        """Test filtering agent-generated messages."""
        payload = WebhookPayload(**agent_message_webhook_payload)
        
        should_process, reason = webhook_handler.should_process_message(payload)
        
        assert should_process is False
        assert "agent" in reason and "loop prevention" in reason
    
    def test_should_process_message_non_contact_sender(self, webhook_handler, sample_webhook_payload):
        """Test filtering non-contact senders."""
        sample_webhook_payload["message"]["sender"]["type"] = "user"
        payload = WebhookPayload(**sample_webhook_payload)
        
        should_process, reason = webhook_handler.should_process_message(payload)
        
        assert should_process is False
        assert "not contact" in reason
    
    def test_should_process_message_no_conversation_data(self, webhook_handler, sample_webhook_payload):
        """Test filtering payload without conversation data."""
        sample_webhook_payload["conversation"] = None
        payload = WebhookPayload(**sample_webhook_payload)
        
        should_process, reason = webhook_handler.should_process_message(payload)
        
        assert should_process is False
        assert "No conversation data" in reason
    
    def test_should_process_message_invalid_source_id(self, webhook_handler, sample_webhook_payload):
        """Test filtering invalid source_id format."""
        sample_webhook_payload["message"]["source_id"] = "contact:"  # Empty phone number
        payload = WebhookPayload(**sample_webhook_payload)
        
        should_process, reason = webhook_handler.should_process_message(payload)
        
        assert should_process is False
        assert "Invalid source_id format" in reason


class TestMessageStorage:
    """Tests for message storage functionality."""
    
    @pytest.mark.asyncio
    async def test_store_message_in_database_success(self, webhook_handler_with_mocks, parsed_webhook_payload):
        """Test successful message storage."""
        handler = webhook_handler_with_mocks
        handler.database_service.store_message.return_value = 123
        
        message_id = await handler.store_message_in_database(parsed_webhook_payload)
        
        assert message_id == 123
        handler.database_service.store_message.assert_called_once()
        
        call_args = handler.database_service.store_message.call_args
        assert call_args[1]['contact_phone'] == "+1234567890"
        assert call_args[1]['conversation_id'] == 12345
        assert call_args[1]['role'] == 'user'
        assert call_args[1]['content'] == "Hello, I need help with my order #1234"
    
    @pytest.mark.asyncio
    async def test_store_message_phone_from_contact(self, webhook_handler_with_mocks, sample_webhook_payload):
        """Test message storage using phone from contact data."""
        # Remove source_id to test fallback to contact phone
        sample_webhook_payload["message"]["source_id"] = None
        sample_webhook_payload["contact"]["phone_number"] = "+1987654321"
        
        payload = WebhookPayload(**sample_webhook_payload)
        handler = webhook_handler_with_mocks
        handler.database_service.store_message.return_value = 124
        
        message_id = await handler.store_message_in_database(payload)
        
        assert message_id == 124
        call_args = handler.database_service.store_message.call_args
        assert call_args[1]['contact_phone'] == "+1987654321"
    
    @pytest.mark.asyncio
    async def test_store_message_no_phone_number(self, webhook_handler_with_mocks, sample_webhook_payload):
        """Test message storage when no phone number is available."""
        sample_webhook_payload["message"]["source_id"] = None
        sample_webhook_payload["contact"]["phone_number"] = None
        
        payload = WebhookPayload(**sample_webhook_payload)
        handler = webhook_handler_with_mocks
        
        message_id = await handler.store_message_in_database(payload)
        
        assert message_id is None
        handler.database_service.store_message.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_store_message_database_error(self, webhook_handler_with_mocks, parsed_webhook_payload):
        """Test message storage with database error."""
        handler = webhook_handler_with_mocks
        handler.database_service.store_message.side_effect = Exception("Database error")
        
        message_id = await handler.store_message_in_database(parsed_webhook_payload)
        
        assert message_id is None
    
    @pytest.mark.asyncio
    async def test_store_message_metadata(self, webhook_handler_with_mocks, parsed_webhook_payload):
        """Test message storage with correct metadata."""
        handler = webhook_handler_with_mocks
        handler.database_service.store_message.return_value = 125
        
        await handler.store_message_in_database(parsed_webhook_payload)
        
        call_args = handler.database_service.store_message.call_args
        metadata = call_args[1]['metadata']
        
        assert metadata['chatwoot_message_id'] == 987654
        assert metadata['sender_name'] == "John Doe"
        assert metadata['sender_id'] == 555
        assert metadata['inbox_id'] == 42
        assert metadata['account_id'] == 1
        assert metadata['source_id'] == "contact:+1234567890"


class TestWebhookProcessing:
    """Tests for complete webhook processing."""
    
    @pytest.mark.asyncio
    async def test_process_webhook_success(self, webhook_handler_with_mocks, sample_webhook_payload):
        """Test successful webhook processing."""
        handler = webhook_handler_with_mocks
        handler.database_service.store_message.return_value = 123
        
        result = await handler.process_webhook(sample_webhook_payload)
        
        assert result['status'] == 'processed'
        assert result['event'] == 'message_created'
        assert result['conversation_id'] == 12345
        assert result['message_id'] == 987654
        assert result['stored_message_id'] == 123
        assert result['contact_phone'] == "+1234567890"
        assert 'processing_time_ms' in result
    
    @pytest.mark.asyncio
    async def test_process_webhook_skipped_message(self, webhook_handler_with_mocks, agent_message_webhook_payload):
        """Test webhook processing with skipped message."""
        handler = webhook_handler_with_mocks
        
        result = await handler.process_webhook(agent_message_webhook_payload)
        
        assert result['status'] == 'skipped'
        assert result['event'] == 'message_created'
        assert 'agent' in result['reason']
        assert 'processing_time_ms' in result
        
        # Should not store message
        handler.database_service.store_message.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_process_webhook_with_signature_verification(self, webhook_handler_with_mocks, sample_webhook_payload):
        """Test webhook processing with signature verification."""
        handler = webhook_handler_with_mocks
        handler.database_service.store_message.return_value = 123
        
        # Create valid signature
        raw_body = json.dumps(sample_webhook_payload).encode('utf-8')
        secret = "test_webhook_secret"
        signature = hmac.new(
            secret.encode('utf-8'),
            raw_body,
            hashlib.sha256
        ).hexdigest()
        
        with patch.object(handler, 'verify_webhook_signature', return_value=True) as mock_verify:
            result = await handler.process_webhook(
                sample_webhook_payload,
                signature=f"sha256={signature}",
                raw_body=raw_body
            )
            
            assert result['status'] == 'processed'
            mock_verify.assert_called_once_with(raw_body, f"sha256={signature}")
    
    @pytest.mark.asyncio
    async def test_process_webhook_invalid_signature(self, webhook_handler_with_mocks, sample_webhook_payload):
        """Test webhook processing with invalid signature."""
        handler = webhook_handler_with_mocks
        raw_body = json.dumps(sample_webhook_payload).encode('utf-8')
        
        with patch.object(handler, 'verify_webhook_signature', return_value=False):
            with pytest.raises(SecurityError, match="Invalid webhook signature"):
                await handler.process_webhook(
                    sample_webhook_payload,
                    signature="invalid_signature",
                    raw_body=raw_body
                )
    
    @pytest.mark.asyncio
    async def test_process_webhook_invalid_payload(self, webhook_handler_with_mocks):
        """Test webhook processing with invalid payload."""
        handler = webhook_handler_with_mocks
        
        invalid_payload = {"invalid": "data"}
        
        with pytest.raises(ValidationError):
            await handler.process_webhook(invalid_payload)
    
    @pytest.mark.asyncio
    async def test_process_webhook_unexpected_error(self, webhook_handler_with_mocks, sample_webhook_payload):
        """Test webhook processing with unexpected error."""
        handler = webhook_handler_with_mocks
        
        # Mock parse_webhook_payload to raise unexpected error
        with patch.object(handler, 'parse_webhook_payload', side_effect=Exception("Unexpected error")):
            with pytest.raises(ProcessingError, match="Failed to process webhook"):
                await handler.process_webhook(sample_webhook_payload)


class TestHealthCheck:
    """Tests for webhook handler health check."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, webhook_handler_with_mocks):
        """Test successful health check."""
        handler = webhook_handler_with_mocks
        
        # Mock successful health checks
        handler.chatwoot_client.health_check.return_value = {
            'status': 'healthy',
            'api_accessible': True
        }
        handler.database_service.health_check.return_value = {
            'status': 'healthy',
            'connection': True
        }
        
        result = await handler.health_check()
        
        assert result['status'] == 'healthy'
        assert result['dependencies']['chatwoot_api']['status'] == 'healthy'
        assert result['dependencies']['database']['status'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_health_check_chatwoot_failure(self, webhook_handler_with_mocks):
        """Test health check with Chatwoot API failure."""
        handler = webhook_handler_with_mocks
        
        handler.chatwoot_client.health_check.side_effect = Exception("API error")
        handler.database_service.health_check.return_value = {
            'status': 'healthy',
            'connection': True
        }
        
        result = await handler.health_check()
        
        assert result['status'] == 'unhealthy'
        assert result['dependencies']['chatwoot_api']['status'] == 'unhealthy'
        assert result['dependencies']['database']['status'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_health_check_database_failure(self, webhook_handler_with_mocks):
        """Test health check with database failure."""
        handler = webhook_handler_with_mocks
        
        handler.chatwoot_client.health_check.return_value = {
            'status': 'healthy',
            'api_accessible': True
        }
        handler.database_service.health_check.side_effect = Exception("Database error")
        
        result = await handler.health_check()
        
        assert result['status'] == 'unhealthy'
        assert result['dependencies']['chatwoot_api']['status'] == 'healthy'
        assert result['dependencies']['database']['status'] == 'unhealthy'
    
    @pytest.mark.asyncio
    async def test_health_check_ensure_dependencies_failure(self, webhook_handler):
        """Test health check when dependency initialization fails."""
        with patch.object(webhook_handler, '_ensure_dependencies', side_effect=Exception("Init error")):
            result = await webhook_handler.health_check()
            
            assert result['status'] == 'unhealthy'
            assert 'error' in result


class TestUtilityMethods:
    """Tests for utility methods."""
    
    def test_validate_message_format_valid(self, webhook_handler):
        """Test message format validation with valid message."""
        message_data = {
            "id": 123,
            "content": "Test message",
            "message_type": "incoming",
            "content_type": "text",
            "sender": {
                "id": 555,
                "type": "contact"
            }
        }
        
        issues = webhook_handler.validate_message_format(message_data)
        assert len(issues) == 0
    
    def test_validate_message_format_missing_required_fields(self, webhook_handler):
        """Test message format validation with missing fields."""
        message_data = {
            "id": 123,
            "content": "Test message"
            # Missing message_type, content_type, sender
        }
        
        issues = webhook_handler.validate_message_format(message_data)
        
        assert len(issues) > 0
        assert any("Missing required field: message_type" in issue for issue in issues)
        assert any("Missing required field: content_type" in issue for issue in issues)
        assert any("Missing required field: sender" in issue for issue in issues)
    
    def test_validate_message_format_invalid_sender(self, webhook_handler):
        """Test message format validation with invalid sender."""
        message_data = {
            "id": 123,
            "content": "Test message",
            "message_type": "incoming",
            "content_type": "text",
            "sender": "not_an_object"  # Should be object
        }
        
        issues = webhook_handler.validate_message_format(message_data)
        
        assert len(issues) > 0
        assert any("Sender must be an object" in issue for issue in issues)
    
    def test_validate_message_format_missing_sender_fields(self, webhook_handler):
        """Test message format validation with missing sender fields."""
        message_data = {
            "id": 123,
            "content": "Test message",
            "message_type": "incoming",
            "content_type": "text",
            "sender": {
                "name": "John"
                # Missing id and type
            }
        }
        
        issues = webhook_handler.validate_message_format(message_data)
        
        assert len(issues) > 0
        assert any("Missing required sender field: id" in issue for issue in issues)
        assert any("Missing required sender field: type" in issue for issue in issues)
    
    def test_validate_message_format_invalid_enum_values(self, webhook_handler):
        """Test message format validation with invalid enum values."""
        message_data = {
            "id": 123,
            "content": "Test message",
            "message_type": "invalid_type",
            "content_type": "invalid_content_type",
            "sender": {
                "id": 555,
                "type": "contact"
            }
        }
        
        issues = webhook_handler.validate_message_format(message_data)
        
        assert len(issues) > 0
        assert any("Invalid message_type: invalid_type" in issue for issue in issues)
        assert any("Invalid content_type: invalid_content_type" in issue for issue in issues)


class TestGlobalFunctions:
    """Tests for global convenience functions."""
    
    @pytest.mark.asyncio
    async def test_handle_chatwoot_webhook(self, sample_webhook_payload):
        """Test global handle_chatwoot_webhook function."""
        from src.handlers.webhook import handle_chatwoot_webhook
        
        with patch('src.handlers.webhook.get_webhook_handler') as mock_get_handler:
            mock_handler = AsyncMock()
            mock_handler.process_webhook.return_value = {"status": "processed"}
            mock_get_handler.return_value = mock_handler
            
            result = await handle_chatwoot_webhook(
                sample_webhook_payload,
                signature="test_signature",
                raw_body=b"test_body"
            )
            
            assert result["status"] == "processed"
            mock_handler.process_webhook.assert_called_once_with(
                sample_webhook_payload,
                "test_signature",
                b"test_body"
            )
    
    def test_get_webhook_handler(self):
        """Test get_webhook_handler function."""
        from src.handlers.webhook import get_webhook_handler
        
        handler1 = get_webhook_handler()
        handler2 = get_webhook_handler()
        
        # Should return same instance (singleton)
        assert handler1 is handler2
        assert isinstance(handler1, WebhookHandler)


class TestErrorHandling:
    """Tests for error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_webhook_processing_with_network_timeout(self, webhook_handler_with_mocks, sample_webhook_payload):
        """Test webhook processing resilience to network timeouts."""
        handler = webhook_handler_with_mocks
        
        # Simulate network timeout during database operation
        handler.database_service.store_message.side_effect = asyncio.TimeoutError("Database timeout")
        
        result = await handler.process_webhook(sample_webhook_payload)
        
        # Should still process but with None message_id
        assert result['status'] == 'processed'
        assert result['stored_message_id'] is None
    
    @pytest.mark.asyncio
    async def test_concurrent_webhook_processing(self, webhook_handler_with_mocks, sample_webhook_payload):
        """Test concurrent webhook processing."""
        handler = webhook_handler_with_mocks
        handler.database_service.store_message.return_value = 123
        
        # Process multiple webhooks concurrently
        tasks = [
            handler.process_webhook(sample_webhook_payload)
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(result['status'] == 'processed' for result in results)
        assert handler.database_service.store_message.call_count == 5
    
    @pytest.mark.asyncio
    async def test_webhook_processing_memory_cleanup(self, webhook_handler_with_mocks, sample_webhook_payload):
        """Test memory cleanup during webhook processing."""
        handler = webhook_handler_with_mocks
        handler.database_service.store_message.return_value = 123
        
        # Process many webhooks to test memory usage
        for _ in range(100):
            result = await handler.process_webhook(sample_webhook_payload)
            assert result['status'] == 'processed'
        
        # Verify call count (should be exactly 100)
        assert handler.database_service.store_message.call_count == 100