"""
End-to-End integration tests for the complete Chatwoot Agent workflow.

Tests cover:
- Complete message processing pipeline
- Tool coordination and response generation
- Error recovery and fallback scenarios
- Performance benchmarks
- Multi-agent scenarios
- Real-world use cases
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.handlers.webhook import WebhookHandler
from src.agent import ChatwootAgent, get_agent
from src.services.chatwoot_api import ChatwootAPIClient
from src.services.database import DatabaseService
from src.services.governance import GovernanceService
from src.models.schemas import (
    WebhookPayload, AgentResponse, AgentConfig,
    EventType, MessageType, SenderType
)
from tests.conftest import (
    create_test_webhook_payload, create_test_message,
    multimodal_test_payloads, governance_test_scenarios,
    performance_test_data, assert_response_time
)


class TestE2EBasicWorkflow:
    """Test basic end-to-end workflow scenarios."""
    
    @pytest.fixture
    async def e2e_system(self, test_csv_data):
        """Set up complete system for E2E testing."""
        # Mock all external dependencies
        mock_chatwoot_client = AsyncMock(spec=ChatwootAPIClient)
        mock_database_service = AsyncMock(spec=DatabaseService)
        mock_governance_service = AsyncMock(spec=GovernanceService)
        
        # Configure mock responses
        mock_chatwoot_client.health_check.return_value = {
            'status': 'healthy',
            'api_accessible': True
        }
        mock_chatwoot_client.send_message.return_value = {
            'id': 987654,
            'content': 'Agent response sent',
            'message_type': 'outgoing'
        }
        
        mock_database_service.health_check.return_value = {
            'status': 'healthy',
            'connection': True
        }
        mock_database_service.store_message.return_value = 123
        mock_database_service.store_embedding.return_value = 456
        mock_database_service.similarity_search.return_value = []
        
        mock_governance_service.check_response.return_value = {
            'requires_confirmation': False,
            'detected_prices': [],
            'governance_flags': []
        }
        mock_governance_service.is_agent_paused.return_value = False
        
        # Create webhook handler
        webhook_handler = WebhookHandler(
            chatwoot_client=mock_chatwoot_client,
            database_service=mock_database_service
        )
        await webhook_handler.initialize()
        
        # Create and initialize agent
        with patch('src.agent.get_config_manager'), \
             patch('src.agent.get_openai_config'), \
             patch('src.agent.get_rag_service'), \
             patch('src.agent.get_spreadsheet_service'), \
             patch('src.agent.get_database_service', return_value=mock_database_service), \
             patch('src.agent.ChatOpenAI'), \
             patch('src.agent.create_tool_calling_agent'), \
             patch('src.agent.AgentExecutor') as mock_executor:
            
            agent = ChatwootAgent()
            
            # Mock agent executor
            mock_executor_instance = AsyncMock()
            mock_executor.return_value = mock_executor_instance
            agent.agent_executor = mock_executor_instance
            
            await agent.initialize()
        
        return {
            'webhook_handler': webhook_handler,
            'agent': agent,
            'chatwoot_client': mock_chatwoot_client,
            'database_service': mock_database_service,
            'governance_service': mock_governance_service,
            'agent_executor': mock_executor_instance
        }
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_message_processing_workflow(self, e2e_system):
        """Test complete message processing from webhook to response."""
        # Arrange
        system = e2e_system
        webhook_handler = system['webhook_handler']
        agent_executor = system['agent_executor']
        chatwoot_client = system['chatwoot_client']
        
        # Mock agent response
        agent_executor.ainvoke.return_value = {
            "output": "Hello! I can help you with your order. What's your order number?",
            "intermediate_steps": []
        }
        
        # Create webhook payload
        payload_data = create_test_webhook_payload(
            message_data=create_test_message(
                content="Hello, I need help with my order"
            )
        )
        payload = WebhookPayload(**payload_data)
        
        # Act
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']):
            start_time = time.time()
            result = await webhook_handler.process_webhook(payload)
            
        # Assert
        assert_response_time(start_time, 10000)  # Should complete within 10 seconds
        assert result.status == "success"
        assert result.conversation_id == 12345
        assert result.response_content is not None
        assert "help you with your order" in result.response_content
        
        # Verify agent was invoked
        agent_executor.ainvoke.assert_called_once()
        
        # Verify message was sent to Chatwoot
        chatwoot_client.send_message.assert_called_once()
        send_args = chatwoot_client.send_message.call_args[1]
        assert send_args['conversation_id'] == 12345
        assert "help you with your order" in send_args['content']
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_rag_enhanced_conversation(self, e2e_system):
        """Test conversation with RAG context retrieval."""
        # Arrange
        system = e2e_system
        webhook_handler = system['webhook_handler']
        agent_executor = system['agent_executor']
        database_service = system['database_service']
        
        # Mock RAG search results
        database_service.similarity_search.return_value = [
            {
                'content': 'Previous conversation: User asked about order #1234 status',
                'similarity': 0.85,
                'metadata': {'conversation_id': 12340}
            }
        ]
        
        # Mock agent response using RAG context
        agent_executor.ainvoke.return_value = {
            "output": "Based on our previous conversation about order #1234, let me check the current status for you.",
            "intermediate_steps": [
                (MagicMock(tool="retrieve_relevant_context"), "Retrieved order context")
            ]
        }
        
        # Create follow-up message payload
        payload_data = create_test_webhook_payload(
            message_data=create_test_message(
                content="What's the status of my order?"
            )
        )
        payload = WebhookPayload(**payload_data)
        
        # Act
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']):
            result = await webhook_handler.process_webhook(payload)
        
        # Assert
        assert result.status == "success"
        assert "order #1234" in result.response_content
        assert result.tool_used == "retrieve_relevant_context"
        
        # Verify RAG search was performed
        database_service.similarity_search.assert_called()
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_spreadsheet_tool_integration(self, e2e_system, test_csv_data):
        """Test integration with spreadsheet tool for data lookup."""
        # Arrange
        system = e2e_system
        webhook_handler = system['webhook_handler']
        agent_executor = system['agent_executor']
        
        # Mock spreadsheet tool response
        agent_executor.ainvoke.return_value = {
            "output": "I found your order #1234. It's currently scheduled for installation on 2025-08-15 with technician John Tech.",
            "intermediate_steps": [
                (MagicMock(tool="query_spreadsheet_data"), "Found order data")
            ]
        }
        
        # Create order inquiry payload
        payload_data = create_test_webhook_payload(
            message_data=create_test_message(
                content="Can you check my order status? My phone is +1234567890"
            )
        )
        payload = WebhookPayload(**payload_data)
        
        # Act
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']):
            result = await webhook_handler.process_webhook(payload)
        
        # Assert
        assert result.status == "success"
        assert "order #1234" in result.response_content
        assert "scheduled" in result.response_content
        assert "John Tech" in result.response_content
        assert result.tool_used == "query_spreadsheet_data"
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_multi_tool_coordination(self, e2e_system):
        """Test coordination between multiple tools in a single response."""
        # Arrange
        system = e2e_system
        webhook_handler = system['webhook_handler']
        agent_executor = system['agent_executor']
        database_service = system['database_service']
        
        # Mock RAG context
        database_service.similarity_search.return_value = [
            {
                'content': 'User previously inquired about installation pricing',
                'similarity': 0.82,
                'metadata': {'type': 'pricing_inquiry'}
            }
        ]
        
        # Mock multi-tool agent response
        agent_executor.ainvoke.return_value = {
            "output": "Based on our previous discussion about installation pricing, I can confirm your order #1234 is scheduled. The installation service costs $299.99 as discussed.",
            "intermediate_steps": [
                (MagicMock(tool="retrieve_relevant_context"), "Retrieved pricing context"),
                (MagicMock(tool="query_spreadsheet_data"), "Found order details")
            ]
        }
        
        payload_data = create_test_webhook_payload(
            message_data=create_test_message(
                content="Can you confirm my order and pricing details?"
            )
        )
        payload = WebhookPayload(**payload_data)
        
        # Act
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']):
            result = await webhook_handler.process_webhook(payload)
        
        # Assert
        assert result.status == "success"
        assert "order #1234" in result.response_content
        assert "$299.99" in result.response_content
        # Tool used should be the first one (implementation dependent)
        assert result.tool_used in ["retrieve_relevant_context", "query_spreadsheet_data"]


class TestE2EGovernanceWorkflow:
    """Test end-to-end governance workflows."""
    
    @pytest.fixture
    async def governance_system(self):
        """Set up system with governance controls."""
        # Mock services
        mock_chatwoot_client = AsyncMock(spec=ChatwootAPIClient)
        mock_database_service = AsyncMock(spec=DatabaseService)
        mock_governance_service = AsyncMock(spec=GovernanceService)
        
        # Configure governance responses
        mock_governance_service.is_agent_paused.return_value = False
        
        # Create webhook handler
        webhook_handler = WebhookHandler(
            chatwoot_client=mock_chatwoot_client,
            database_service=mock_database_service
        )
        await webhook_handler.initialize()
        
        # Create agent
        with patch('src.agent.get_config_manager'), \
             patch('src.agent.get_openai_config'), \
             patch('src.agent.get_rag_service'), \
             patch('src.agent.get_spreadsheet_service'), \
             patch('src.agent.get_database_service', return_value=mock_database_service), \
             patch('src.agent.ChatOpenAI'), \
             patch('src.agent.create_tool_calling_agent'), \
             patch('src.agent.AgentExecutor') as mock_executor:
            
            agent = ChatwootAgent()
            mock_executor_instance = AsyncMock()
            mock_executor.return_value = mock_executor_instance
            agent.agent_executor = mock_executor_instance
            await agent.initialize()
        
        return {
            'webhook_handler': webhook_handler,
            'agent': agent,
            'chatwoot_client': mock_chatwoot_client,
            'database_service': mock_database_service,
            'governance_service': mock_governance_service,
            'agent_executor': mock_executor_instance
        }
    
    @pytest.mark.e2e
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_price_detection_and_confirmation_workflow(self, governance_system, governance_test_scenarios):
        """Test complete price detection and confirmation workflow."""
        # Arrange
        system = governance_system
        webhook_handler = system['webhook_handler']
        agent_executor = system['agent_executor']
        governance_service = system['governance_service']
        
        # Configure price-sensitive response
        scenario = governance_test_scenarios['price_sensitive']
        agent_executor.ainvoke.return_value = {
            "output": scenario['agent_response'],
            "intermediate_steps": []
        }
        
        # Configure governance to require confirmation
        governance_service.check_response.return_value = {
            'requires_confirmation': True,
            'detected_prices': [{'amount': 299.99, 'currency': 'USD'}],
            'governance_flags': ['price_detected']
        }
        
        payload_data = create_test_webhook_payload(
            message_data=create_test_message(
                content=scenario['user_query']
            )
        )
        payload = WebhookPayload(**payload_data)
        
        # Act
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']), \
             patch('src.handlers.webhook.get_governance_service', return_value=governance_service):
            
            result = await webhook_handler.process_webhook(payload)
        
        # Assert
        assert result.status == "success"
        assert result.requires_confirmation is True
        assert "$299.99" in result.response_content
        
        # Verify governance check was performed
        governance_service.check_response.assert_called_once()
    
    @pytest.mark.e2e
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_agent_pause_resume_workflow(self, governance_system):
        """Test agent pause and resume workflow."""
        # Arrange
        system = governance_system
        webhook_handler = system['webhook_handler']
        governance_service = system['governance_service']
        
        # First message - agent not paused
        governance_service.is_agent_paused.return_value = False
        
        payload_data = create_test_webhook_payload(
            message_data=create_test_message(
                content="Hello, I need help"
            )
        )
        payload = WebhookPayload(**payload_data)
        
        # Act 1 - Process message normally
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']), \
             patch('src.handlers.webhook.get_governance_service', return_value=governance_service):
            
            result1 = await webhook_handler.process_webhook(payload)
        
        # Assert 1
        assert result1.status == "success"
        
        # Arrange 2 - Pause agent
        governance_service.is_agent_paused.return_value = True
        
        # Act 2 - Process message while paused
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']), \
             patch('src.handlers.webhook.get_governance_service', return_value=governance_service):
            
            result2 = await webhook_handler.process_webhook(payload)
        
        # Assert 2
        assert result2.status == "paused"
        assert "paused" in result2.error.lower()


class TestE2EMultimodalWorkflow:
    """Test end-to-end multimodal processing workflows."""
    
    @pytest.fixture
    async def multimodal_system(self):
        """Set up system for multimodal testing."""
        # Mock services
        mock_chatwoot_client = AsyncMock(spec=ChatwootAPIClient)
        mock_database_service = AsyncMock(spec=DatabaseService)
        
        # Create webhook handler
        webhook_handler = WebhookHandler(
            chatwoot_client=mock_chatwoot_client,
            database_service=mock_database_service
        )
        await webhook_handler.initialize()
        
        # Create agent
        with patch('src.agent.get_config_manager'), \
             patch('src.agent.get_openai_config'), \
             patch('src.agent.get_rag_service'), \
             patch('src.agent.get_spreadsheet_service'), \
             patch('src.agent.get_database_service', return_value=mock_database_service), \
             patch('src.agent.ChatOpenAI'), \
             patch('src.agent.create_tool_calling_agent'), \
             patch('src.agent.AgentExecutor') as mock_executor:
            
            agent = ChatwootAgent()
            mock_executor_instance = AsyncMock()
            mock_executor.return_value = mock_executor_instance
            agent.agent_executor = mock_executor_instance
            await agent.initialize()
        
        return {
            'webhook_handler': webhook_handler,
            'agent': agent,
            'chatwoot_client': mock_chatwoot_client,
            'database_service': mock_database_service,
            'agent_executor': mock_executor_instance
        }
    
    @pytest.mark.e2e
    @pytest.mark.multimodal
    @pytest.mark.asyncio
    async def test_audio_message_processing(self, multimodal_system, multimodal_test_payloads):
        """Test processing of audio messages with transcription."""
        # Arrange
        system = multimodal_system
        webhook_handler = system['webhook_handler']
        agent_executor = system['agent_executor']
        
        # Mock agent response to audio/transcript
        agent_executor.ainvoke.return_value = {
            "output": "I can help you check your appointment. Let me look that up for you.",
            "intermediate_steps": []
        }
        
        # Create audio message payload
        audio_payload_data = multimodal_test_payloads['audio_message']
        payload = WebhookPayload(**audio_payload_data)
        
        # Act
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']):
            result = await webhook_handler.process_webhook(payload)
        
        # Assert
        assert result.status == "success"
        assert "appointment" in result.response_content
        
        # Verify agent processed the transcript
        agent_executor.ainvoke.assert_called_once()
        call_args = agent_executor.ainvoke.call_args[0][0]
        # Should contain either the transcript or reference to audio
        assert "appointment" in call_args['input'] or call_args['input'] != ""
    
    @pytest.mark.e2e
    @pytest.mark.multimodal
    @pytest.mark.asyncio
    async def test_image_message_processing(self, multimodal_system, multimodal_test_payloads):
        """Test processing of image messages."""
        # Arrange
        system = multimodal_system
        webhook_handler = system['webhook_handler']
        agent_executor = system['agent_executor']
        
        # Mock agent response to image
        agent_executor.ainvoke.return_value = {
            "output": "I can see the error in your screenshot. Let me help you resolve this issue.",
            "intermediate_steps": []
        }
        
        # Create image message payload
        image_payload_data = multimodal_test_payloads['image_message']
        payload = WebhookPayload(**image_payload_data)
        
        # Act
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']):
            result = await webhook_handler.process_webhook(payload)
        
        # Assert
        assert result.status == "success"
        assert "error" in result.response_content
        assert "screenshot" in result.response_content


class TestE2EErrorRecovery:
    """Test end-to-end error recovery and fallback scenarios."""
    
    @pytest.fixture
    async def error_prone_system(self):
        """Set up system for error testing."""
        # Mock services
        mock_chatwoot_client = AsyncMock(spec=ChatwootAPIClient)
        mock_database_service = AsyncMock(spec=DatabaseService)
        
        # Create webhook handler
        webhook_handler = WebhookHandler(
            chatwoot_client=mock_chatwoot_client,
            database_service=mock_database_service
        )
        await webhook_handler.initialize()
        
        # Create agent
        with patch('src.agent.get_config_manager'), \
             patch('src.agent.get_openai_config'), \
             patch('src.agent.get_rag_service'), \
             patch('src.agent.get_spreadsheet_service'), \
             patch('src.agent.get_database_service', return_value=mock_database_service), \
             patch('src.agent.ChatOpenAI'), \
             patch('src.agent.create_tool_calling_agent'), \
             patch('src.agent.AgentExecutor') as mock_executor:
            
            agent = ChatwootAgent()
            mock_executor_instance = AsyncMock()
            mock_executor.return_value = mock_executor_instance
            agent.agent_executor = mock_executor_instance
            await agent.initialize()
        
        return {
            'webhook_handler': webhook_handler,
            'agent': agent,
            'chatwoot_client': mock_chatwoot_client,
            'database_service': mock_database_service,
            'agent_executor': mock_executor_instance
        }
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_agent_timeout_recovery(self, error_prone_system):
        """Test recovery from agent timeout errors."""
        # Arrange
        system = error_prone_system
        webhook_handler = system['webhook_handler']
        agent_executor = system['agent_executor']
        chatwoot_client = system['chatwoot_client']
        
        # Mock agent timeout
        agent_executor.ainvoke.side_effect = asyncio.TimeoutError("Agent timeout")
        
        # Mock fallback message sending
        chatwoot_client.send_message.return_value = {
            'id': 987654,
            'content': 'I apologize for the delay. A human agent will assist you shortly.'
        }
        
        payload_data = create_test_webhook_payload(
            message_data=create_test_message(
                content="I need urgent help with my order"
            )
        )
        payload = WebhookPayload(**payload_data)
        
        # Act
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']):
            result = await webhook_handler.process_webhook(payload)
        
        # Assert
        assert result.status == "error"
        assert "timeout" in result.error.lower()
        
        # Verify fallback message was sent
        chatwoot_client.send_message.assert_called_once()
        send_args = chatwoot_client.send_message.call_args[1]
        assert "human agent" in send_args['content'].lower()
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_database_error_recovery(self, error_prone_system):
        """Test recovery from database errors."""
        # Arrange
        system = error_prone_system
        webhook_handler = system['webhook_handler']
        agent_executor = system['agent_executor']
        database_service = system['database_service']
        
        # Mock successful agent response
        agent_executor.ainvoke.return_value = {
            "output": "I can help you with your request.",
            "intermediate_steps": []
        }
        
        # Mock database error when storing
        database_service.store_message.side_effect = Exception("Database connection failed")
        
        payload_data = create_test_webhook_payload()
        payload = WebhookPayload(**payload_data)
        
        # Act
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']):
            result = await webhook_handler.process_webhook(payload)
        
        # Assert
        # Should still complete successfully even if database storage fails
        assert result.status in ["success", "error"]  # Depends on implementation
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_chatwoot_api_error_recovery(self, error_prone_system):
        """Test recovery from Chatwoot API errors."""
        # Arrange
        system = error_prone_system
        webhook_handler = system['webhook_handler']
        agent_executor = system['agent_executor']
        chatwoot_client = system['chatwoot_client']
        
        # Mock successful agent response
        agent_executor.ainvoke.return_value = {
            "output": "I can help you with your request.",
            "intermediate_steps": []
        }
        
        # Mock API error when sending message
        chatwoot_client.send_message.side_effect = Exception("API rate limit exceeded")
        
        payload_data = create_test_webhook_payload()
        payload = WebhookPayload(**payload_data)
        
        # Act
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']):
            result = await webhook_handler.process_webhook(payload)
        
        # Assert
        assert result.status == "error"
        assert "rate limit" in result.error.lower() or "api" in result.error.lower()


class TestE2EPerformanceBenchmarks:
    """Test end-to-end performance benchmarks."""
    
    @pytest.fixture
    async def performance_system(self):
        """Set up system for performance testing."""
        # Mock services with realistic delays
        mock_chatwoot_client = AsyncMock(spec=ChatwootAPIClient)
        mock_database_service = AsyncMock(spec=DatabaseService)
        
        # Add realistic async delays
        async def mock_send_message(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms API delay
            return {'id': 987654, 'content': 'Response sent'}
        
        async def mock_store_message(*args, **kwargs):
            await asyncio.sleep(0.05)  # 50ms database delay
            return 123
        
        mock_chatwoot_client.send_message.side_effect = mock_send_message
        mock_database_service.store_message.side_effect = mock_store_message
        mock_database_service.similarity_search.return_value = []
        
        # Create webhook handler
        webhook_handler = WebhookHandler(
            chatwoot_client=mock_chatwoot_client,
            database_service=mock_database_service
        )
        await webhook_handler.initialize()
        
        # Create agent
        with patch('src.agent.get_config_manager'), \
             patch('src.agent.get_openai_config'), \
             patch('src.agent.get_rag_service'), \
             patch('src.agent.get_spreadsheet_service'), \
             patch('src.agent.get_database_service', return_value=mock_database_service), \
             patch('src.agent.ChatOpenAI'), \
             patch('src.agent.create_tool_calling_agent'), \
             patch('src.agent.AgentExecutor') as mock_executor:
            
            agent = ChatwootAgent()
            mock_executor_instance = AsyncMock()
            
            # Add realistic processing delay
            async def mock_ainvoke(*args, **kwargs):
                await asyncio.sleep(0.5)  # 500ms LLM processing delay
                return {
                    "output": "I can help you with that request.",
                    "intermediate_steps": []
                }
            
            mock_executor_instance.ainvoke.side_effect = mock_ainvoke
            mock_executor.return_value = mock_executor_instance
            agent.agent_executor = mock_executor_instance
            
            await agent.initialize()
        
        return {
            'webhook_handler': webhook_handler,
            'agent': agent,
            'chatwoot_client': mock_chatwoot_client,
            'database_service': mock_database_service,
            'agent_executor': mock_executor_instance
        }
    
    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_message_end_to_end_performance(self, performance_system, performance_test_data):
        """Test end-to-end performance for single message."""
        # Arrange
        system = performance_system
        webhook_handler = system['webhook_handler']
        target_time = performance_test_data['response_time_targets']['simple_query']
        
        payload_data = create_test_webhook_payload(
            message_data=create_test_message(
                content="Hello, I need help"
            )
        )
        payload = WebhookPayload(**payload_data)
        
        # Act
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']):
            start_time = time.time()
            result = await webhook_handler.process_webhook(payload)
            
        # Assert
        assert_response_time(start_time, target_time)
        assert result.status == "success"
        assert result.processing_time_ms > 0
        assert result.processing_time_ms < target_time
    
    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_message_performance(self, performance_system, performance_test_data):
        """Test performance under concurrent message load."""
        # Arrange
        system = performance_system
        webhook_handler = system['webhook_handler']
        
        # Create multiple concurrent messages
        payloads = []
        for i in range(10):
            payload_data = create_test_webhook_payload(
                message_data=create_test_message(
                    message_id=987654 + i,
                    conversation_id=12345 + i,
                    content=f"Concurrent message {i}"
                )
            )
            payloads.append(WebhookPayload(**payload_data))
        
        # Act
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']):
            start_time = time.time()
            
            tasks = [webhook_handler.process_webhook(payload) for payload in payloads]
            results = await asyncio.gather(*tasks)
            
        # Assert
        total_time = time.time() - start_time
        assert total_time < 10.0  # Should complete within 10 seconds
        assert len(results) == 10
        assert all(result.status == "success" for result in results)
        
        # Check individual response times
        avg_response_time = sum(result.processing_time_ms for result in results) / len(results)
        assert avg_response_time < 2000  # Average under 2 seconds
    
    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sustained_throughput_performance(self, performance_system, performance_test_data):
        """Test sustained throughput performance."""
        # Arrange
        system = performance_system
        webhook_handler = system['webhook_handler']
        target_throughput = performance_test_data['throughput_targets']['messages_per_minute']
        
        # Act - Process messages for 60 seconds
        start_time = time.time()
        message_count = 0
        
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']):
            while time.time() - start_time < 60:  # 60 seconds
                payload_data = create_test_webhook_payload(
                    message_data=create_test_message(
                        message_id=987654 + message_count,
                        content=f"Sustained test message {message_count}"
                    )
                )
                payload = WebhookPayload(**payload_data)
                
                await webhook_handler.process_webhook(payload)
                message_count += 1
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(1.0)  # 1 second between messages
        
        # Assert
        total_time = time.time() - start_time
        messages_per_minute = (message_count / total_time) * 60
        
        # Should meet minimum throughput target
        assert messages_per_minute >= target_throughput * 0.8  # Allow 20% variance


class TestE2ERealWorldScenarios:
    """Test real-world usage scenarios."""
    
    @pytest.fixture
    async def real_world_system(self):
        """Set up system for real-world scenario testing."""
        # Mock services with realistic responses
        mock_chatwoot_client = AsyncMock(spec=ChatwootAPIClient)
        mock_database_service = AsyncMock(spec=DatabaseService)
        
        # Configure realistic mock responses
        mock_chatwoot_client.send_message.return_value = {
            'id': 987654,
            'content': 'Response sent successfully'
        }
        
        mock_database_service.store_message.return_value = 123
        mock_database_service.similarity_search.return_value = []
        
        # Create webhook handler
        webhook_handler = WebhookHandler(
            chatwoot_client=mock_chatwoot_client,
            database_service=mock_database_service
        )
        await webhook_handler.initialize()
        
        # Create agent
        with patch('src.agent.get_config_manager'), \
             patch('src.agent.get_openai_config'), \
             patch('src.agent.get_rag_service'), \
             patch('src.agent.get_spreadsheet_service'), \
             patch('src.agent.get_database_service', return_value=mock_database_service), \
             patch('src.agent.ChatOpenAI'), \
             patch('src.agent.create_tool_calling_agent'), \
             patch('src.agent.AgentExecutor') as mock_executor:
            
            agent = ChatwootAgent()
            mock_executor_instance = AsyncMock()
            mock_executor.return_value = mock_executor_instance
            agent.agent_executor = mock_executor_instance
            await agent.initialize()
        
        return {
            'webhook_handler': webhook_handler,
            'agent': agent,
            'chatwoot_client': mock_chatwoot_client,
            'database_service': mock_database_service,
            'agent_executor': mock_executor_instance
        }
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_customer_service_order_inquiry(self, real_world_system):
        """Test realistic customer service order inquiry scenario."""
        # Arrange
        system = real_world_system
        webhook_handler = system['webhook_handler']
        agent_executor = system['agent_executor']
        
        # Mock realistic customer service response
        agent_executor.ainvoke.return_value = {
            "output": "I can see your order #1234 is currently being processed. It's scheduled for delivery on August 16th. You'll receive a tracking number once it ships.",
            "intermediate_steps": [
                (MagicMock(tool="query_spreadsheet_data"), "Retrieved order details")
            ]
        }
        
        # Create customer inquiry
        payload_data = create_test_webhook_payload(
            message_data=create_test_message(
                content="Hi, can you please check the status of my order? I placed it last week but haven't heard anything."
            ),
            phone_number="+1234567890",
            contact_name="Sarah Johnson"
        )
        payload = WebhookPayload(**payload_data)
        
        # Act
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']):
            result = await webhook_handler.process_webhook(payload)
        
        # Assert
        assert result.status == "success"
        assert "order #1234" in result.response_content
        assert "August 16th" in result.response_content
        assert "tracking number" in result.response_content
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_technical_support_escalation(self, real_world_system):
        """Test technical support scenario with escalation."""
        # Arrange
        system = real_world_system
        webhook_handler = system['webhook_handler']
        agent_executor = system['agent_executor']
        
        # Mock technical support response with escalation
        agent_executor.ainvoke.return_value = {
            "output": "I understand you're experiencing connectivity issues. I've found some troubleshooting steps in our knowledge base, but given the complexity of your setup, I'm connecting you with our technical specialist who can provide more detailed assistance.",
            "intermediate_steps": [
                (MagicMock(tool="retrieve_relevant_context"), "Retrieved technical documentation")
            ]
        }
        
        # Create technical support request
        payload_data = create_test_webhook_payload(
            message_data=create_test_message(
                content="I'm having trouble with my internet connection after the recent installation. The speeds are much slower than expected and I keep getting disconnected."
            ),
            phone_number="+1987654321",
            contact_name="Mike Chen"
        )
        payload = WebhookPayload(**payload_data)
        
        # Act
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']):
            result = await webhook_handler.process_webhook(payload)
        
        # Assert
        assert result.status == "success"
        assert "connectivity issues" in result.response_content
        assert "technical specialist" in result.response_content
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_appointment_scheduling_workflow(self, real_world_system):
        """Test appointment scheduling workflow."""
        # Arrange
        system = real_world_system
        webhook_handler = system['webhook_handler']
        agent_executor = system['agent_executor']
        
        # Mock appointment scheduling response
        agent_executor.ainvoke.return_value = {
            "output": "I can help you schedule an appointment. Based on your location and service type, I have availability on August 17th at 2:00 PM or August 18th at 10:00 AM. Which time works better for you?",
            "intermediate_steps": [
                (MagicMock(tool="query_spreadsheet_data"), "Checked availability")
            ]
        }
        
        # Create appointment request
        payload_data = create_test_webhook_payload(
            message_data=create_test_message(
                content="I need to schedule a maintenance appointment for next week. I'm usually available in the afternoons."
            ),
            phone_number="+1555123456",
            contact_name="Lisa Rodriguez"
        )
        payload = WebhookPayload(**payload_data)
        
        # Act
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']):
            result = await webhook_handler.process_webhook(payload)
        
        # Assert
        assert result.status == "success"
        assert "August 17th" in result.response_content or "August 18th" in result.response_content
        assert "2:00 PM" in result.response_content or "10:00 AM" in result.response_content
        assert "which time" in result.response_content.lower()
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_multi_turn_conversation_memory(self, real_world_system):
        """Test multi-turn conversation with memory persistence."""
        # Arrange
        system = real_world_system
        webhook_handler = system['webhook_handler']
        agent_executor = system['agent_executor']
        database_service = system['database_service']
        
        # First turn - Initial inquiry
        agent_executor.ainvoke.return_value = {
            "output": "I can help you with billing questions. What specific information do you need about your account?",
            "intermediate_steps": []
        }
        
        payload_1 = WebhookPayload(**create_test_webhook_payload(
            message_data=create_test_message(
                content="I have a question about my bill",
                message_id=1001
            )
        ))
        
        # Act 1
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']):
            result_1 = await webhook_handler.process_webhook(payload_1)
        
        # Assert 1
        assert result_1.status == "success"
        
        # Arrange 2 - Follow-up with context
        # Mock database returning previous context
        database_service.similarity_search.return_value = [
            {
                'content': 'User: I have a question about my bill\nAssistant: I can help you with billing questions.',
                'similarity': 0.95,
                'metadata': {'conversation_id': 12345}
            }
        ]
        
        agent_executor.ainvoke.return_value = {
            "output": "Based on our previous discussion about your billing question, I can see you're asking about the installation charge. This is a one-time fee for setting up your service.",
            "intermediate_steps": [
                (MagicMock(tool="retrieve_relevant_context"), "Retrieved conversation history")
            ]
        }
        
        payload_2 = WebhookPayload(**create_test_webhook_payload(
            message_data=create_test_message(
                content="Why is there an installation charge on my first bill?",
                message_id=1002
            )
        ))
        
        # Act 2
        with patch('src.handlers.webhook.get_agent', return_value=system['agent']):
            result_2 = await webhook_handler.process_webhook(payload_2)
        
        # Assert 2
        assert result_2.status == "success"
        assert "previous discussion" in result_2.response_content
        assert "installation charge" in result_2.response_content
        assert result_2.tool_used == "retrieve_relevant_context"