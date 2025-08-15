"""
Comprehensive unit tests for the ChatwootAgent core logic.

Tests cover:
- LangChain agent initialization and configuration
- Tool registration and orchestration
- Message processing workflows
- Configuration loading and validation
- Performance monitoring and metrics
- Error handling and fallbacks
"""

import pytest
import json
import time
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import Dict, Any, List

from src.agent import ChatwootAgent, AgentMetricsCallback, ConversationContext
from src.models.schemas import (
    WebhookPayload, AgentResponse, AgentConfig, InboxConfig,
    RAGConfig, SheetsToolConfig, MultimodalConfig, CostLimitsConfig,
    MetricsResponse, MessageType, SenderType, EventType
)
from tests.conftest import (
    create_test_webhook_payload, create_test_message,
    assert_response_time, mock_openai_response, AsyncMockService
)


class TestChatwootAgentInitialization:
    """Test agent initialization and configuration."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_agent_initialization_success(self):
        """Test successful agent initialization."""
        # Arrange & Act
        agent = ChatwootAgent()
        
        with patch('src.agent.get_config_manager'), \
             patch('src.agent.get_openai_config'), \
             patch('src.agent.get_rag_service'), \
             patch('src.agent.get_spreadsheet_service'), \
             patch('src.agent.get_database_service') as mock_db:
            
            mock_db.return_value.health_check = AsyncMock(return_value={"status": "healthy"})
            
            await agent.initialize()
        
        # Assert
        assert agent._initialized is True
        assert agent.llm is not None
        assert agent.agent_executor is not None
        assert isinstance(agent.tools, list)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_agent_initialization_failure(self):
        """Test agent initialization failure handling."""
        # Arrange
        agent = ChatwootAgent()
        
        # Mock configuration loading to fail
        with patch('src.agent.get_config_manager') as mock_config:
            mock_config.side_effect = Exception("Config loading failed")
            
            # Act & Assert
            with pytest.raises(RuntimeError, match="Agent initialization failed"):
                await agent.initialize()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_load_configuration_default(self):
        """Test loading default configuration when none provided."""
        # Arrange
        agent = ChatwootAgent()
        
        with patch('src.agent.get_config_manager') as mock_config_manager:
            mock_manager = MagicMock()
            mock_manager.get_agent_config.return_value = None
            mock_config_manager.return_value = mock_manager
            
            # Act
            await agent._load_configuration()
        
        # Assert
        assert agent.global_config is not None
        assert isinstance(agent.global_config, AgentConfig)
        assert agent.global_config.enabled is True
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_load_custom_configuration(self):
        """Test loading custom configuration."""
        # Arrange
        agent = ChatwootAgent()
        custom_config = AgentConfig(
            model="gpt-4o",
            temperature=0.5,
            max_tokens=2048,
            enabled=True
        )
        
        with patch('src.agent.get_config_manager') as mock_config_manager:
            mock_manager = MagicMock()
            mock_manager.get_agent_config.return_value = custom_config
            mock_config_manager.return_value = mock_manager
            
            # Act
            await agent._load_configuration()
        
        # Assert
        assert agent.global_config == custom_config
        assert agent.global_config.model == "gpt-4o"
        assert agent.global_config.temperature == 0.5
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_llm_with_config(self):
        """Test LLM initialization with configuration."""
        # Arrange
        agent = ChatwootAgent()
        agent.global_config = AgentConfig(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=1024
        )
        
        with patch('src.agent.get_openai_config') as mock_openai_config, \
             patch('src.agent.ChatOpenAI') as mock_chat_openai:
            
            mock_openai_config.return_value = MagicMock(
                api_key="test_key",
                base_url=None,
                organization=None,
                timeout_seconds=30,
                max_retries=3
            )
            
            # Act
            await agent._initialize_llm()
        
        # Assert
        mock_chat_openai.assert_called_once_with(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=1024,
            openai_api_key="test_key",
            openai_api_base=None,
            openai_organization=None,
            timeout=30,
            max_retries=3,
            streaming=False
        )
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_register_tools_rag_enabled(self):
        """Test tool registration when RAG is enabled."""
        # Arrange
        agent = ChatwootAgent()
        agent.global_config = AgentConfig(
            rag=RAGConfig(enabled=True),
            sheets_tool=SheetsToolConfig(enabled=False)
        )
        
        with patch('src.agent.retrieve_relevant_context') as mock_rag_tool, \
             patch('src.agent.store_conversation_context') as mock_store_tool, \
             patch('src.agent.get_rag_performance_stats') as mock_stats_tool:
            
            # Act
            await agent._register_tools()
        
        # Assert
        assert len(agent.tools) == 3  # Three RAG tools
        tool_names = [tool.name for tool in agent.tools]
        assert 'retrieve_relevant_context' in tool_names
        assert 'store_conversation_context' in tool_names
        assert 'get_rag_performance_stats' in tool_names
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_register_tools_sheets_enabled(self):
        """Test tool registration when spreadsheet tool is enabled."""
        # Arrange
        agent = ChatwootAgent()
        agent.global_config = AgentConfig(
            rag=RAGConfig(enabled=False),
            sheets_tool=SheetsToolConfig(enabled=True)
        )
        
        with patch('src.agent.query_spreadsheet_data') as mock_query_tool, \
             patch('src.agent.list_available_spreadsheets') as mock_list_tool, \
             patch('src.agent.get_spreadsheet_performance_stats') as mock_stats_tool:
            
            # Act
            await agent._register_tools()
        
        # Assert
        assert len(agent.tools) == 3  # Three spreadsheet tools
        tool_names = [tool.name for tool in agent.tools]
        assert 'query_spreadsheet_data' in tool_names
        assert 'list_available_spreadsheets' in tool_names
        assert 'get_spreadsheet_performance_stats' in tool_names
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_register_tools_all_disabled(self):
        """Test tool registration when all tools are disabled."""
        # Arrange
        agent = ChatwootAgent()
        agent.global_config = AgentConfig(
            rag=RAGConfig(enabled=False),
            sheets_tool=SheetsToolConfig(enabled=False)
        )
        
        # Act
        await agent._register_tools()
        
        # Assert
        assert len(agent.tools) == 0


class TestChatwootAgentMessageProcessing:
    """Test agent message processing functionality."""
    
    @pytest.fixture
    async def initialized_agent(self):
        """Create and initialize agent for testing."""
        agent = ChatwootAgent()
        
        # Mock all dependencies
        with patch('src.agent.get_config_manager'), \
             patch('src.agent.get_openai_config'), \
             patch('src.agent.get_rag_service'), \
             patch('src.agent.get_spreadsheet_service'), \
             patch('src.agent.get_database_service') as mock_db, \
             patch('src.agent.ChatOpenAI'), \
             patch('src.agent.create_tool_calling_agent'), \
             patch('src.agent.AgentExecutor') as mock_executor:
            
            mock_db.return_value.health_check = AsyncMock(return_value={"status": "healthy"})
            
            # Mock agent executor
            mock_executor_instance = AsyncMock()
            mock_executor.return_value = mock_executor_instance
            agent.agent_executor = mock_executor_instance
            
            await agent.initialize()
        
        return agent
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_message_success(self, initialized_agent, sample_webhook_payload):
        """Test successful message processing."""
        # Arrange
        payload = WebhookPayload(**sample_webhook_payload)
        
        # Mock successful agent execution
        initialized_agent.agent_executor.ainvoke.return_value = {
            "output": "Thank you for your message. How can I help you?",
            "intermediate_steps": []
        }
        
        # Act
        start_time = time.time()
        result = await initialized_agent.process_message(payload)
        
        # Assert
        assert_response_time(start_time, 10000)  # Should complete within 10 seconds
        assert isinstance(result, AgentResponse)
        assert result.status == "success"
        assert result.conversation_id == 12345
        assert result.contact_phone == "+1234567890"
        assert result.response_content == "Thank you for your message. How can I help you?"
        assert result.processing_time_ms > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_message_agent_disabled_globally(self, initialized_agent, sample_webhook_payload):
        """Test message processing when agent is globally disabled."""
        # Arrange
        payload = WebhookPayload(**sample_webhook_payload)
        initialized_agent.global_config.enabled = False
        
        # Act
        result = await initialized_agent.process_message(payload)
        
        # Assert
        assert result.status == "disabled"
        assert "globally disabled" in result.error
        assert result.conversation_id == 12345
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_message_governance_paused(self, initialized_agent, sample_webhook_payload):
        """Test message processing when governance is paused."""
        # Arrange
        payload = WebhookPayload(**sample_webhook_payload)
        
        # Act
        result = await initialized_agent.process_message(payload, governance_paused=True)
        
        # Assert
        assert result.status == "paused"
        assert "paused" in result.error
        assert result.conversation_id == 12345
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_message_inbox_disabled(self, initialized_agent, sample_webhook_payload):
        """Test message processing when agent is disabled for specific inbox."""
        # Arrange
        payload = WebhookPayload(**sample_webhook_payload)
        
        # Mock inbox config with agent disabled
        inbox_config = InboxConfig(inbox_id=42, enabled=False)
        with patch.object(initialized_agent, 'get_inbox_config', return_value=inbox_config):
            # Act
            result = await initialized_agent.process_message(payload)
        
        # Assert
        assert result.status == "disabled"
        assert "disabled for inbox" in result.error
        assert result.conversation_id == 12345
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_message_with_tools(self, initialized_agent, sample_webhook_payload):
        """Test message processing with tool usage."""
        # Arrange
        payload = WebhookPayload(**sample_webhook_payload)
        
        # Mock agent execution with tools
        mock_tool_step = MagicMock()
        mock_tool_step.tool = "retrieve_relevant_context"
        
        initialized_agent.agent_executor.ainvoke.return_value = {
            "output": "Based on our previous conversation, I can help you with your order.",
            "intermediate_steps": [(mock_tool_step, "Retrieved context")]
        }
        
        # Act
        result = await initialized_agent.process_message(payload)
        
        # Assert
        assert result.status == "success"
        assert result.tool_used == "retrieve_relevant_context"
        assert "previous conversation" in result.response_content
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_message_agent_execution_failure(self, initialized_agent, sample_webhook_payload):
        """Test message processing when agent execution fails."""
        # Arrange
        payload = WebhookPayload(**sample_webhook_payload)
        
        # Mock agent execution failure
        initialized_agent.agent_executor.ainvoke.side_effect = Exception("LLM timeout")
        
        # Act
        result = await initialized_agent.process_message(payload)
        
        # Assert
        assert result.status == "error"
        assert "LLM timeout" in result.error
        assert result.conversation_id == 12345
        assert result.processing_time_ms > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_message_conversation_context_creation(self, initialized_agent, sample_webhook_payload):
        """Test conversation context creation and management."""
        # Arrange
        payload = WebhookPayload(**sample_webhook_payload)
        
        # Mock successful processing
        initialized_agent.agent_executor.ainvoke.return_value = {
            "output": "Hello! How can I help you?",
            "intermediate_steps": []
        }
        
        # Act
        result = await initialized_agent.process_message(payload)
        
        # Assert
        assert result.status == "success"
        
        # Check conversation context was created
        conversation_id = payload.conversation.id
        assert conversation_id in initialized_agent._conversation_contexts
        
        context = initialized_agent._conversation_contexts[conversation_id]
        assert context.conversation_id == conversation_id
        assert context.contact_phone == "+1234567890"
        assert context.inbox_id == 42
        assert len(context.conversation_history) == 2  # User message + agent response
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_message_with_persona(self, initialized_agent, sample_webhook_payload):
        """Test message processing with inbox persona configuration."""
        # Arrange
        payload = WebhookPayload(**sample_webhook_payload)
        
        # Mock inbox config with persona
        inbox_config = InboxConfig(
            inbox_id=42,
            enabled=True,
            persona="You are a friendly and professional tech support agent.",
            language="en"
        )
        
        with patch.object(initialized_agent, 'get_inbox_config', return_value=inbox_config):
            # Mock successful processing
            initialized_agent.agent_executor.ainvoke.return_value = {
                "output": "Hello! I'm here to help with your technical needs.",
                "intermediate_steps": []
            }
            
            # Act
            result = await initialized_agent.process_message(payload)
        
        # Assert
        assert result.status == "success"
        # Verify persona was used in agent input
        initialized_agent.agent_executor.ainvoke.assert_called_once()
        call_args = initialized_agent.agent_executor.ainvoke.call_args[0][0]
        assert "friendly and professional" in call_args["persona_instructions"]
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_message_conversation_history_limit(self, initialized_agent, sample_webhook_payload):
        """Test that conversation history is limited to recent messages."""
        # Arrange
        payload = WebhookPayload(**sample_webhook_payload)
        conversation_id = payload.conversation.id
        
        # Create conversation context with long history
        context = ConversationContext(
            conversation_id=conversation_id,
            contact_phone="+1234567890",
            inbox_id=42,
            account_id=1
        )
        
        # Add 15 messages to history (should be trimmed to 10)
        for i in range(15):
            context.conversation_history.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i}",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        initialized_agent._conversation_contexts[conversation_id] = context
        
        # Mock successful processing
        initialized_agent.agent_executor.ainvoke.return_value = {
            "output": "Response to recent message",
            "intermediate_steps": []
        }
        
        # Act
        result = await initialized_agent.process_message(payload)
        
        # Assert
        assert result.status == "success"
        
        # Verify only last 10 messages were passed to agent
        call_args = initialized_agent.agent_executor.ainvoke.call_args[0][0]
        chat_history = call_args["chat_history"]
        assert len(chat_history) == 10  # Should be limited to last 10


class TestChatwootAgentConfiguration:
    """Test agent configuration management."""
    
    @pytest.fixture
    async def agent(self):
        """Create agent instance for testing."""
        return ChatwootAgent()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_inbox_config_from_cache(self, agent):
        """Test getting inbox config from cache."""
        # Arrange
        inbox_id = 42
        cached_config = InboxConfig(inbox_id=inbox_id, enabled=True, language="es")
        agent.inbox_configs[inbox_id] = cached_config
        
        # Act
        result = await agent.get_inbox_config(inbox_id)
        
        # Assert
        assert result == cached_config
        assert result.language == "es"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_inbox_config_load_from_file(self, agent):
        """Test loading inbox config from file."""
        # Arrange
        inbox_id = 42
        file_config = InboxConfig(inbox_id=inbox_id, enabled=True, persona="Test persona")
        
        with patch.object(agent.config_manager, 'load_inbox_config', return_value=file_config):
            # Act
            result = await agent.get_inbox_config(inbox_id)
        
        # Assert
        assert result == file_config
        assert result.persona == "Test persona"
        assert inbox_id in agent.inbox_configs  # Should be cached
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_inbox_config_create_default(self, agent):
        """Test creating default inbox config when none found."""
        # Arrange
        inbox_id = 42
        
        with patch.object(agent.config_manager, 'load_inbox_config', return_value=None):
            # Act
            result = await agent.get_inbox_config(inbox_id)
        
        # Assert
        assert isinstance(result, InboxConfig)
        assert result.inbox_id == inbox_id
        assert result.enabled is True  # Default value
        assert inbox_id in agent.inbox_configs  # Should be cached
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_build_persona_instructions_with_persona(self, agent):
        """Test building persona instructions when persona is configured."""
        # Arrange
        inbox_config = InboxConfig(
            inbox_id=42,
            persona="You are a helpful technical support agent specializing in network issues.",
            language="en"
        )
        
        # Act
        instructions = await agent._build_persona_instructions(inbox_config)
        
        # Assert
        assert "technical support agent" in instructions
        assert "network issues" in instructions
        assert "Language: en" in instructions
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_build_persona_instructions_without_persona(self, agent):
        """Test building persona instructions when no persona is configured."""
        # Arrange
        inbox_config = InboxConfig(inbox_id=42, persona=None)
        
        # Act
        instructions = await agent._build_persona_instructions(inbox_config)
        
        # Assert
        assert "professional and helpful" in instructions
        assert instructions != ""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_global_config(self, agent):
        """Test getting global configuration as dict."""
        # Arrange
        agent.global_config = AgentConfig(
            model="gpt-4o",
            temperature=0.5,
            enabled=True
        )
        
        # Act
        config_dict = await agent.get_global_config()
        
        # Assert
        assert isinstance(config_dict, dict)
        assert config_dict["model"] == "gpt-4o"
        assert config_dict["temperature"] == 0.5
        assert config_dict["enabled"] is True
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_global_config_when_none(self, agent):
        """Test getting global configuration when none is set."""
        # Arrange
        agent.global_config = None
        
        # Act
        config_dict = await agent.get_global_config()
        
        # Assert
        assert config_dict == {}


class TestChatwootAgentMetrics:
    """Test agent metrics collection and reporting."""
    
    @pytest.fixture
    async def initialized_agent(self):
        """Create initialized agent for metrics testing."""
        agent = ChatwootAgent()
        
        with patch('src.agent.get_config_manager'), \
             patch('src.agent.get_openai_config'), \
             patch('src.agent.get_rag_service'), \
             patch('src.agent.get_spreadsheet_service'), \
             patch('src.agent.get_database_service') as mock_db:
            
            mock_db.return_value.health_check = AsyncMock(return_value={"status": "healthy"})
            await agent.initialize()
        
        return agent
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_metrics_success(self, initialized_agent):
        """Test metrics update for successful processing."""
        # Act
        await initialized_agent._update_metrics(
            conversation_id=12345,
            processing_time_ms=1500,
            tools_used=["retrieve_relevant_context"],
            token_count=150,
            success=True
        )
        
        # Assert
        assert initialized_agent._metrics['total_messages_processed'] == 1
        assert initialized_agent._metrics['successful_responses'] == 1
        assert initialized_agent._metrics['failed_responses'] == 0
        assert initialized_agent._metrics['total_tokens_used'] == 150
        assert initialized_agent._metrics['tools_usage']['retrieve_relevant_context'] == 1
        assert len(initialized_agent._metrics['response_times']) == 1
        assert initialized_agent._metrics['response_times'][0] == 1500
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_metrics_failure(self, initialized_agent):
        """Test metrics update for failed processing."""
        # Act
        await initialized_agent._update_metrics(
            conversation_id=12345,
            processing_time_ms=2000,
            tools_used=[],
            token_count=0,
            success=False,
            error="Processing failed"
        )
        
        # Assert
        assert initialized_agent._metrics['total_messages_processed'] == 1
        assert initialized_agent._metrics['successful_responses'] == 0
        assert initialized_agent._metrics['failed_responses'] == 1
        assert initialized_agent._metrics['total_tokens_used'] == 0
        assert len(initialized_agent._metrics['response_times']) == 1
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_metrics(self, initialized_agent):
        """Test getting agent metrics."""
        # Arrange - Add some test data
        await initialized_agent._update_metrics(
            conversation_id=12345,
            processing_time_ms=1000,
            tools_used=["rag_search"],
            token_count=100,
            success=True
        )
        
        await initialized_agent._update_metrics(
            conversation_id=12346,
            processing_time_ms=2000,
            tools_used=["spreadsheet_query"],
            token_count=75,
            success=True
        )
        
        # Act
        metrics = await initialized_agent.get_metrics()
        
        # Assert
        assert isinstance(metrics, MetricsResponse)
        assert metrics.total_messages_processed == 2
        assert metrics.successful_responses == 2
        assert metrics.failed_responses == 0
        assert metrics.average_response_time_ms == 1500.0  # (1000 + 2000) / 2
        assert metrics.total_tokens_used == 175
        assert metrics.tools_usage["rag_search"] == 1
        assert metrics.tools_usage["spreadsheet_query"] == 1
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metrics_cost_calculation(self, initialized_agent):
        """Test cost calculation in metrics."""
        # Act
        await initialized_agent._update_metrics(
            conversation_id=12345,
            processing_time_ms=1000,
            tools_used=[],
            token_count=1000,  # 1000 tokens
            success=True
        )
        
        # Assert
        expected_cost = (1000 / 1000) * 0.03  # $0.03 per 1K tokens
        assert initialized_agent._metrics['total_cost_usd'] == expected_cost
        assert initialized_agent._metrics['daily_cost_usd'] == expected_cost
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metrics_daily_reset(self, initialized_agent):
        """Test daily metrics reset functionality."""
        # Arrange - Set yesterday as last reset date
        yesterday = datetime.utcnow().date() - timedelta(days=1)
        initialized_agent._metrics['last_reset_date'] = yesterday
        initialized_agent._metrics['daily_cost_usd'] = 10.0
        
        # Act
        await initialized_agent._update_metrics(
            conversation_id=12345,
            processing_time_ms=1000,
            tools_used=[],
            token_count=100,
            success=True
        )
        
        # Assert - Daily cost should reset
        today = datetime.utcnow().date()
        assert initialized_agent._metrics['last_reset_date'] == today
        # Daily cost should be reset and then updated with new cost
        expected_new_cost = (100 / 1000) * 0.03
        assert initialized_agent._metrics['daily_cost_usd'] == expected_new_cost
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_response_times_limit(self, initialized_agent):
        """Test that response times list is limited to prevent memory growth."""
        # Act - Add more than 1000 response times
        for i in range(1100):
            await initialized_agent._update_metrics(
                conversation_id=12345 + i,
                processing_time_ms=1000 + i,
                tools_used=[],
                token_count=10,
                success=True
            )
        
        # Assert - Should be limited to last 1000
        assert len(initialized_agent._metrics['response_times']) == 1000
        # Should contain the most recent times
        assert 2099 in initialized_agent._metrics['response_times']  # 1000 + 1099
        assert 1000 not in initialized_agent._metrics['response_times']  # Should be removed


class TestChatwootAgentHealthCheck:
    """Test agent health check functionality."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_healthy_agent(self):
        """Test health check for healthy initialized agent."""
        # Arrange
        agent = ChatwootAgent()
        
        with patch('src.agent.get_config_manager'), \
             patch('src.agent.get_openai_config'), \
             patch('src.agent.get_rag_service'), \
             patch('src.agent.get_spreadsheet_service'), \
             patch('src.agent.get_database_service') as mock_db, \
             patch('src.agent.ChatOpenAI'), \
             patch('src.agent.create_tool_calling_agent'), \
             patch('src.agent.AgentExecutor') as mock_executor:
            
            mock_db.return_value.health_check = AsyncMock(return_value={"status": "healthy"})
            
            # Mock successful agent execution for health check
            mock_executor_instance = AsyncMock()
            mock_executor_instance.ainvoke.return_value = {"output": "healthy"}
            mock_executor.return_value = mock_executor_instance
            agent.agent_executor = mock_executor_instance
            
            await agent.initialize()
            
            # Act
            is_healthy = await agent.is_healthy()
        
        # Assert
        assert is_healthy is True
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_uninitialized_agent(self):
        """Test health check for uninitialized agent."""
        # Arrange
        agent = ChatwootAgent()
        
        # Act
        is_healthy = await agent.is_healthy()
        
        # Assert
        assert is_healthy is False
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_agent_execution_timeout(self):
        """Test health check when agent execution times out."""
        # Arrange
        agent = ChatwootAgent()
        
        with patch('src.agent.get_config_manager'), \
             patch('src.agent.get_openai_config'), \
             patch('src.agent.get_rag_service'), \
             patch('src.agent.get_spreadsheet_service'), \
             patch('src.agent.get_database_service') as mock_db, \
             patch('src.agent.ChatOpenAI'), \
             patch('src.agent.create_tool_calling_agent'), \
             patch('src.agent.AgentExecutor') as mock_executor:
            
            mock_db.return_value.health_check = AsyncMock(return_value={"status": "healthy"})
            
            # Mock agent execution that times out
            mock_executor_instance = AsyncMock()
            mock_executor_instance.ainvoke.side_effect = asyncio.TimeoutError()
            mock_executor.return_value = mock_executor_instance
            agent.agent_executor = mock_executor_instance
            
            await agent.initialize()
            
            # Act
            is_healthy = await agent.is_healthy()
        
        # Assert
        assert is_healthy is False
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_agent_execution_error(self):
        """Test health check when agent execution fails."""
        # Arrange
        agent = ChatwootAgent()
        
        with patch('src.agent.get_config_manager'), \
             patch('src.agent.get_openai_config'), \
             patch('src.agent.get_rag_service'), \
             patch('src.agent.get_spreadsheet_service'), \
             patch('src.agent.get_database_service') as mock_db, \
             patch('src.agent.ChatOpenAI'), \
             patch('src.agent.create_tool_calling_agent'), \
             patch('src.agent.AgentExecutor') as mock_executor:
            
            mock_db.return_value.health_check = AsyncMock(return_value={"status": "healthy"})
            
            # Mock agent execution that fails
            mock_executor_instance = AsyncMock()
            mock_executor_instance.ainvoke.side_effect = Exception("Health check failed")
            mock_executor.return_value = mock_executor_instance
            agent.agent_executor = mock_executor_instance
            
            await agent.initialize()
            
            # Act
            is_healthy = await agent.is_healthy()
        
        # Assert
        assert is_healthy is False


class TestAgentMetricsCallback:
    """Test the AgentMetricsCallback functionality."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent for callback testing."""
        agent = MagicMock()
        agent._update_metrics = AsyncMock()
        return agent
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_callback_lifecycle_success(self, mock_agent):
        """Test complete callback lifecycle for successful execution."""
        # Arrange
        callback = AgentMetricsCallback(mock_agent)
        
        # Act - Simulate callback lifecycle
        await callback.on_chain_start({}, {"conversation_id": 12345})
        await callback.on_tool_start({"name": "test_tool"}, "test input")
        await callback.on_tool_end("test output")
        await callback.on_chain_end({"output": "success"})
        
        # Assert
        assert callback.current_conversation_id == 12345
        assert "test_tool" in callback.tool_calls
        mock_agent._update_metrics.assert_called_once()
        call_args = mock_agent._update_metrics.call_args[1]
        assert call_args["conversation_id"] == 12345
        assert call_args["success"] is True
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_callback_lifecycle_error(self, mock_agent):
        """Test callback lifecycle for failed execution."""
        # Arrange
        callback = AgentMetricsCallback(mock_agent)
        
        # Act - Simulate callback lifecycle with error
        await callback.on_chain_start({}, {"conversation_id": 12345})
        await callback.on_tool_start({"name": "test_tool"}, "test input")
        await callback.on_tool_error(Exception("Tool failed"))
        await callback.on_chain_error(Exception("Chain failed"))
        
        # Assert
        mock_agent._update_metrics.assert_called_once()
        call_args = mock_agent._update_metrics.call_args[1]
        assert call_args["conversation_id"] == 12345
        assert call_args["success"] is False
        assert call_args["error"] == "Chain failed"


class TestConversationContext:
    """Test ConversationContext functionality."""
    
    @pytest.mark.unit
    def test_conversation_context_creation(self):
        """Test creating conversation context."""
        # Act
        context = ConversationContext(
            conversation_id=12345,
            contact_phone="+1234567890",
            inbox_id=42,
            account_id=1,
            contact_name="John Doe"
        )
        
        # Assert
        assert context.conversation_id == 12345
        assert context.contact_phone == "+1234567890"
        assert context.inbox_id == 42
        assert context.account_id == 1
        assert context.contact_name == "John Doe"
        assert isinstance(context.conversation_history, list)
        assert len(context.conversation_history) == 0
        assert isinstance(context.custom_attributes, dict)
        assert isinstance(context.labels, list)
    
    @pytest.mark.unit
    def test_conversation_context_defaults(self):
        """Test conversation context with default values."""
        # Act
        context = ConversationContext(
            conversation_id=12345,
            contact_phone="+1234567890",
            inbox_id=42,
            account_id=1
        )
        
        # Assert
        assert context.contact_name is None
        assert context.conversation_history == []
        assert context.custom_attributes == {}
        assert context.labels == []
        assert isinstance(context.created_at, datetime)