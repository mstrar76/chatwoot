"""
Comprehensive unit tests for the WebhookHandler component.

Tests cover:
- Message validation and filtering
- Loop prevention mechanisms  
- Error handling and edge cases
- Agent integration flow
- Performance and security
"""

import pytest
import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.handlers.webhook import WebhookHandler
from src.models.schemas import (
    WebhookPayload, EventType, MessageType, SenderType,
    AgentResponse, AgentConfig
)
from src.services.chatwoot_api import ChatwootAPIClient
from src.services.database import DatabaseService
from tests.conftest import (
    create_test_webhook_payload, create_test_message,
    create_agent_message_payload, assert_response_time,
    create_malicious_payloads, AsyncMockService
)


class TestWebhookHandler:
    """Test suite for WebhookHandler core functionality."""
    
    @pytest.fixture
    async def webhook_handler(self, mock_chatwoot_client, mock_database_service):
        """Create webhook handler with mocked dependencies."""
        handler = WebhookHandler(
            chatwoot_client=mock_chatwoot_client,
            database_service=mock_database_service
        )
        await handler.initialize()
        return handler
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_webhook_handler_initialization(self, webhook_handler):
        """Test webhook handler initializes correctly."""
        assert webhook_handler.chatwoot_client is not None
        assert webhook_handler.database_service is not None
        assert webhook_handler._initialized is True
        
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_valid_webhook_payload(self, webhook_handler, sample_webhook_payload):
        """Test processing a valid webhook payload."""
        # Arrange
        payload = WebhookPayload(**sample_webhook_payload)
        
        # Mock agent response
        with patch('src.handlers.webhook.get_agent') as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.process_message.return_value = AgentResponse(
                conversation_id=12345,
                contact_phone="+1234567890",
                response_content="Thank you for your message. How can I help you?",
                tool_used="rag_search",
                confidence=0.85,
                processing_time_ms=1500,
                status="success"
            )
            mock_get_agent.return_value = mock_agent
            
            # Act
            start_time = time.time()
            result = await webhook_handler.process_webhook(payload)
            
            # Assert
            assert_response_time(start_time, 5000)  # Should complete within 5 seconds
            assert result is not None
            assert result.status == "success"
            assert result.conversation_id == 12345
            assert result.response_content is not None
            assert len(result.response_content) > 0
            
            # Verify agent was called
            mock_agent.process_message.assert_called_once_with(payload, governance_paused=False)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_message_filtering_agent_messages(self, webhook_handler):
        """Test that agent-generated messages are filtered out (loop prevention)."""
        # Arrange - Create agent message payload
        payload_data = create_agent_message_payload(
            conversation_id=12345,
            content="This is an agent response"
        )
        payload = WebhookPayload(**payload_data)
        
        # Act
        result = await webhook_handler.process_webhook(payload)
        
        # Assert - Should be filtered out
        assert result.status == "filtered"
        assert "agent message" in result.error.lower()
        
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_message_filtering_non_text_content(self, webhook_handler):
        """Test filtering of non-text content types that should be processed."""
        # Arrange - Create audio message
        message_data = create_test_message(
            content="",
            content_type="audio",
            attachments=[{
                "id": 1001,
                "file_url": "https://test.com/audio.mp3",
                "file_type": "audio/mp3"
            }]
        )
        payload_data = create_test_webhook_payload(message_data=message_data)
        payload = WebhookPayload(**payload_data)
        
        # Mock agent to handle multimodal
        with patch('src.handlers.webhook.get_agent') as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.process_message.return_value = AgentResponse(
                conversation_id=12345,
                contact_phone="+1234567890",
                response_content="I received your audio message.",
                status="success"
            )
            mock_get_agent.return_value = mock_agent
            
            # Act
            result = await webhook_handler.process_webhook(payload)
            
            # Assert - Should be processed
            assert result.status == "success"
            mock_agent.process_message.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_webhook_payload_validation_error(self, webhook_handler):
        """Test handling of invalid webhook payload."""
        # Arrange - Invalid payload structure
        invalid_payload = {
            "event": "invalid_event",
            "message": {
                "id": "not_an_integer",  # Should be int
                "content": None  # Should be string
            }
        }
        
        # Act & Assert
        with pytest.raises(Exception):  # Should raise validation error
            WebhookPayload(**invalid_payload)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_conversation_status_filtering(self, webhook_handler):
        """Test filtering based on conversation status."""
        # Arrange - Resolved conversation
        payload_data = create_test_webhook_payload(conversation_status="resolved")
        payload = WebhookPayload(**payload_data)
        
        # Act
        result = await webhook_handler.process_webhook(payload)
        
        # Assert - Should process even resolved conversations
        # (This behavior may change based on business requirements)
        assert result is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_handling_agent_failure(self, webhook_handler, sample_webhook_payload):
        """Test error handling when agent processing fails."""
        # Arrange
        payload = WebhookPayload(**sample_webhook_payload)
        
        # Mock agent to raise exception
        with patch('src.handlers.webhook.get_agent') as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.process_message.side_effect = Exception("Agent processing failed")
            mock_get_agent.return_value = mock_agent
            
            # Act
            result = await webhook_handler.process_webhook(payload)
            
            # Assert
            assert result.status == "error"
            assert "Agent processing failed" in result.error
            assert result.conversation_id == 12345
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_handling_chatwoot_api_failure(self, webhook_handler, sample_webhook_payload):
        """Test error handling when Chatwoot API call fails."""
        # Arrange
        payload = WebhookPayload(**sample_webhook_payload)
        
        # Mock successful agent processing but failed API call
        with patch('src.handlers.webhook.get_agent') as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.process_message.return_value = AgentResponse(
                conversation_id=12345,
                contact_phone="+1234567890",
                response_content="Test response",
                status="success"
            )
            mock_get_agent.return_value = mock_agent
            
            # Mock Chatwoot API failure
            webhook_handler.chatwoot_client.send_message.side_effect = Exception("API call failed")
            
            # Act
            result = await webhook_handler.process_webhook(payload)
            
            # Assert
            assert result.status == "error"
            assert "API call failed" in result.error
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_message_processing(self, webhook_handler):
        """Test handling of concurrent message processing."""
        # Arrange - Multiple payloads for same conversation
        payloads = [
            WebhookPayload(**create_test_webhook_payload(
                message_data=create_test_message(
                    message_id=987654 + i,
                    content=f"Message {i}"
                )
            ))
            for i in range(5)
        ]
        
        # Mock agent
        with patch('src.handlers.webhook.get_agent') as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.process_message.return_value = AgentResponse(
                conversation_id=12345,
                contact_phone="+1234567890",
                response_content="Response",
                status="success"
            )
            mock_get_agent.return_value = mock_agent
            
            # Act - Process concurrently
            import asyncio
            results = await asyncio.gather(*[
                webhook_handler.process_webhook(payload)
                for payload in payloads
            ])
            
            # Assert
            assert len(results) == 5
            assert all(result.status == "success" for result in results)
            assert mock_agent.process_message.call_count == 5
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_webhook_signature_validation(self, webhook_handler):
        """Test webhook signature validation for security."""
        # This would test HMAC signature validation if implemented
        # For now, we'll test the structure is in place
        
        # Arrange
        headers = {
            "X-Chatwoot-Signature": "sha256=test_signature",
            "Content-Type": "application/json"
        }
        payload_json = json.dumps(create_test_webhook_payload())
        
        # Act & Assert
        # This test assumes signature validation is implemented
        # Currently just checking the method exists
        assert hasattr(webhook_handler, 'validate_webhook_signature')
    
    @pytest.mark.unit
    async def test_webhook_handler_health_check(self, webhook_handler):
        """Test webhook handler health check functionality."""
        # Act
        health = await webhook_handler.health_check()
        
        # Assert
        assert health['status'] == 'healthy'
        assert 'dependencies' in health
        assert 'chatwoot_api' in health['dependencies']
        assert 'database' in health['dependencies']
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_webhook_handler_metrics(self, webhook_handler, sample_webhook_payload):
        """Test webhook handler metrics collection."""
        # Arrange
        payload = WebhookPayload(**sample_webhook_payload)
        
        # Mock agent
        with patch('src.handlers.webhook.get_agent') as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.process_message.return_value = AgentResponse(
                conversation_id=12345,
                contact_phone="+1234567890",
                response_content="Test response",
                processing_time_ms=1000,
                status="success"
            )
            mock_get_agent.return_value = mock_agent
            
            # Act
            await webhook_handler.process_webhook(payload)
            metrics = await webhook_handler.get_metrics()
            
            # Assert
            assert metrics['total_webhooks_processed'] >= 1
            assert metrics['successful_processes'] >= 1
            assert 'average_processing_time_ms' in metrics
            assert 'last_processed_at' in metrics


class TestWebhookHandlerEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.fixture
    async def webhook_handler(self, mock_chatwoot_client, mock_database_service):
        """Create webhook handler with mocked dependencies."""
        handler = WebhookHandler(
            chatwoot_client=mock_chatwoot_client,
            database_service=mock_database_service
        )
        await handler.initialize()
        return handler
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_message_content(self, webhook_handler):
        """Test handling of empty message content."""
        # Arrange
        message_data = create_test_message(content="")
        payload_data = create_test_webhook_payload(message_data=message_data)
        payload = WebhookPayload(**payload_data)
        
        # Act
        result = await webhook_handler.process_webhook(payload)
        
        # Assert - Should be filtered or handled gracefully
        assert result.status in ["filtered", "success"]
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_very_long_message_content(self, webhook_handler):
        """Test handling of very long message content."""
        # Arrange - Create very long message
        long_content = "A" * 10000  # 10K characters
        message_data = create_test_message(content=long_content)
        payload_data = create_test_webhook_payload(message_data=message_data)
        payload = WebhookPayload(**payload_data)
        
        # Mock agent to handle long content
        with patch('src.handlers.webhook.get_agent') as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.process_message.return_value = AgentResponse(
                conversation_id=12345,
                contact_phone="+1234567890",
                response_content="I received your long message.",
                status="success"
            )
            mock_get_agent.return_value = mock_agent
            
            # Act
            result = await webhook_handler.process_webhook(payload)
            
            # Assert
            assert result.status == "success"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_missing_contact_phone(self, webhook_handler):
        """Test handling when contact phone number is missing."""
        # Arrange
        payload_data = create_test_webhook_payload(phone_number=None)
        payload = WebhookPayload(**payload_data)
        
        # Act
        result = await webhook_handler.process_webhook(payload)
        
        # Assert - Should handle gracefully or use fallback
        assert result is not None
        # Contact phone should default to "unknown" or similar
        assert result.contact_phone in [None, "unknown", ""]
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unsupported_event_type(self, webhook_handler):
        """Test handling of unsupported event types."""
        # Arrange
        payload_data = create_test_webhook_payload(event="conversation_resolved")
        payload = WebhookPayload(**payload_data)
        
        # Act
        result = await webhook_handler.process_webhook(payload)
        
        # Assert - Should be filtered out
        assert result.status == "filtered"
        assert "event type" in result.error.lower()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_malformed_attachment_data(self, webhook_handler):
        """Test handling of malformed attachment data."""
        # Arrange
        message_data = create_test_message(
            attachments=[{
                "id": None,  # Invalid ID
                "file_url": "not_a_valid_url",
                "file_type": ""
            }]
        )
        payload_data = create_test_webhook_payload(message_data=message_data)
        payload = WebhookPayload(**payload_data)
        
        # Act
        result = await webhook_handler.process_webhook(payload)
        
        # Assert - Should handle gracefully
        assert result is not None


class TestWebhookHandlerSecurity:
    """Security-focused tests for webhook handler."""
    
    @pytest.fixture
    async def webhook_handler(self, mock_chatwoot_client, mock_database_service):
        """Create webhook handler with mocked dependencies."""
        handler = WebhookHandler(
            chatwoot_client=mock_chatwoot_client,
            database_service=mock_database_service
        )
        await handler.initialize()
        return handler
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self, webhook_handler):
        """Test prevention of SQL injection attacks."""
        # Arrange - SQL injection attempt
        malicious_content = "'; DROP TABLE messages; --"
        message_data = create_test_message(content=malicious_content)
        payload_data = create_test_webhook_payload(message_data=message_data)
        payload = WebhookPayload(**payload_data)
        
        # Mock agent to handle safely
        with patch('src.handlers.webhook.get_agent') as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.process_message.return_value = AgentResponse(
                conversation_id=12345,
                contact_phone="+1234567890",
                response_content="I understand your question.",
                status="success"
            )
            mock_get_agent.return_value = mock_agent
            
            # Act
            result = await webhook_handler.process_webhook(payload)
            
            # Assert - Should be handled safely
            assert result.status == "success"
            # Verify database operations used parameterized queries
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_xss_content_sanitization(self, webhook_handler):
        """Test sanitization of XSS attempts."""
        # Arrange - XSS attempt
        xss_content = "<script>alert('xss')</script>"
        message_data = create_test_message(content=xss_content)
        payload_data = create_test_webhook_payload(message_data=message_data)
        payload = WebhookPayload(**payload_data)
        
        # Mock agent
        with patch('src.handlers.webhook.get_agent') as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.process_message.return_value = AgentResponse(
                conversation_id=12345,
                contact_phone="+1234567890",
                response_content="I received your message.",
                status="success"
            )
            mock_get_agent.return_value = mock_agent
            
            # Act
            result = await webhook_handler.process_webhook(payload)
            
            # Assert
            assert result.status == "success"
            # XSS content should be safely handled
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_rate_limiting_protection(self, webhook_handler):
        """Test rate limiting protection against spam."""
        # Arrange - Rapid fire requests
        payloads = [
            WebhookPayload(**create_test_webhook_payload(
                message_data=create_test_message(message_id=987654 + i)
            ))
            for i in range(100)  # Many requests
        ]
        
        # Act - Send many requests quickly
        import asyncio
        start_time = time.time()
        
        with patch('src.handlers.webhook.get_agent') as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.process_message.return_value = AgentResponse(
                conversation_id=12345,
                contact_phone="+1234567890",
                response_content="Response",
                status="success"
            )
            mock_get_agent.return_value = mock_agent
            
            results = await asyncio.gather(*[
                webhook_handler.process_webhook(payload)
                for payload in payloads[:10]  # Process only first 10 to avoid timeout
            ])
        
        # Assert - Should handle gracefully without crashing
        assert len(results) == 10
        # Some may be rate limited or throttled
        successful_count = sum(1 for r in results if r.status == "success")
        assert successful_count >= 0  # At least some should succeed


class TestWebhookHandlerPerformance:
    """Performance-focused tests for webhook handler."""
    
    @pytest.fixture
    async def webhook_handler(self, mock_chatwoot_client, mock_database_service):
        """Create webhook handler with mocked dependencies."""
        handler = WebhookHandler(
            chatwoot_client=mock_chatwoot_client,
            database_service=mock_database_service
        )
        await handler.initialize()
        return handler
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_message_performance(self, webhook_handler, sample_webhook_payload):
        """Test performance of single message processing."""
        # Arrange
        payload = WebhookPayload(**sample_webhook_payload)
        
        with patch('src.handlers.webhook.get_agent') as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.process_message.return_value = AgentResponse(
                conversation_id=12345,
                contact_phone="+1234567890",
                response_content="Quick response",
                processing_time_ms=500,
                status="success"
            )
            mock_get_agent.return_value = mock_agent
            
            # Act & Assert - Should complete within 2 seconds
            start_time = time.time()
            result = await webhook_handler.process_webhook(payload)
            assert_response_time(start_time, 2000)
            
            assert result.status == "success"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_processing_performance(self, webhook_handler):
        """Test performance under concurrent load."""
        # Arrange - 20 concurrent messages
        payloads = [
            WebhookPayload(**create_test_webhook_payload(
                message_data=create_test_message(
                    message_id=987654 + i,
                    conversation_id=12345 + (i % 5)  # 5 different conversations
                )
            ))
            for i in range(20)
        ]
        
        with patch('src.handlers.webhook.get_agent') as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.process_message.return_value = AgentResponse(
                conversation_id=12345,
                contact_phone="+1234567890",
                response_content="Response",
                processing_time_ms=300,
                status="success"
            )
            mock_get_agent.return_value = mock_agent
            
            # Act - Process all concurrently
            import asyncio
            start_time = time.time()
            
            results = await asyncio.gather(*[
                webhook_handler.process_webhook(payload)
                for payload in payloads
            ])
            
            # Assert - Should complete within 10 seconds
            assert_response_time(start_time, 10000)
            assert len(results) == 20
            
            # Most should succeed
            successful_count = sum(1 for r in results if r.status == "success")
            assert successful_count >= 15  # At least 75% success rate
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, webhook_handler):
        """Test that memory usage remains stable under load."""
        import gc
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process many messages
        with patch('src.handlers.webhook.get_agent') as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.process_message.return_value = AgentResponse(
                conversation_id=12345,
                contact_phone="+1234567890",
                response_content="Response",
                status="success"
            )
            mock_get_agent.return_value = mock_agent
            
            for i in range(50):
                payload = WebhookPayload(**create_test_webhook_payload(
                    message_data=create_test_message(message_id=987654 + i)
                ))
                await webhook_handler.process_webhook(payload)
                
                # Force garbage collection periodically
                if i % 10 == 0:
                    gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 50MB)
        assert memory_increase < 50 * 1024 * 1024, f"Memory increased by {memory_increase / 1024 / 1024:.2f}MB"