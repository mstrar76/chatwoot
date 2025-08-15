"""
Chatwoot Agent MVP - Main LangChain Agent Implementation.

This module provides the core agent orchestration using modern LangChain patterns
with ReAct agent architecture, tool integration, and production-ready features.

Features:
- LangChain ReAct agent with tool orchestration
- Tool registration (RAG tool, Spreadsheet tool)
- Persona injection based on inbox configuration
- Async message processing with governance integration
- Configuration loading and management
- Health checks and metrics collection
- Performance monitoring and cost tracking
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import AsyncCallbackHandler

from .tools.rag import (
    retrieve_relevant_context, 
    store_conversation_context, 
    get_rag_performance_stats,
    get_rag_service
)
from .tools.spreadsheet import (
    query_spreadsheet_data,
    list_available_spreadsheets,
    get_spreadsheet_performance_stats,
    get_spreadsheet_service
)
from .models.schemas import (
    AgentConfig, InboxConfig, AgentResponse, MetricsResponse,
    LogEntry, WebhookPayload, Message, AgentMemory
)
from .services.chatwoot_api import ChatwootAPIClient, get_chatwoot_client
from .services.database import get_database_service
from .utils.config import get_config_manager, get_openai_config
from .utils.logging import get_logger

logger = get_logger(__name__)


class AgentMetricsCallback(AsyncCallbackHandler):
    """
    Async callback handler for collecting agent performance metrics.
    
    Tracks token usage, response times, tool usage, and costs for monitoring
    and optimization of the agent system.
    """
    
    def __init__(self, agent_instance: 'ChatwootAgent'):
        super().__init__()
        self.agent = agent_instance
        self.start_time: Optional[float] = None
        self.tool_calls: List[str] = []
        self.token_count: int = 0
        self.current_conversation_id: Optional[int] = None
    
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when the agent chain starts."""
        self.start_time = time.time()
        self.tool_calls = []
        self.token_count = 0
        # Extract conversation ID from inputs if available
        if 'conversation_id' in inputs:
            self.current_conversation_id = inputs['conversation_id']
    
    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool starts executing."""
        tool_name = serialized.get('name', 'unknown_tool')
        self.tool_calls.append(tool_name)
        logger.debug("Tool execution started", tool=tool_name, input_length=len(input_str))
    
    async def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes executing."""
        logger.debug("Tool execution completed", output_length=len(output))
    
    async def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when a tool encounters an error."""
        logger.error("Tool execution failed", error=str(error))
    
    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when the agent chain ends."""
        if self.start_time:
            duration_ms = int((time.time() - self.start_time) * 1000)
            
            # Update agent metrics
            await self.agent._update_metrics(
                conversation_id=self.current_conversation_id,
                processing_time_ms=duration_ms,
                tools_used=self.tool_calls,
                token_count=self.token_count,
                success=True
            )
    
    async def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Called when the agent chain encounters an error."""
        if self.start_time:
            duration_ms = int((time.time() - self.start_time) * 1000)
            
            # Update agent metrics with error
            await self.agent._update_metrics(
                conversation_id=self.current_conversation_id,
                processing_time_ms=duration_ms,
                tools_used=self.tool_calls,
                token_count=self.token_count,
                success=False,
                error=str(error)
            )


@dataclass
class ConversationContext:
    """Context for agent conversations including memory and configuration."""
    conversation_id: int
    contact_phone: str
    inbox_id: int
    account_id: int
    contact_name: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    labels: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


class ChatwootAgent:
    """
    Production-ready Chatwoot Agent using modern LangChain patterns.
    
    Features:
    - ReAct agent with tool calling capabilities
    - Async processing with performance monitoring
    - Persona injection based on inbox configuration
    - Memory management and conversation context
    - Governance controls integration
    - Comprehensive error handling and fallbacks
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Chatwoot Agent.
        
        Args:
            config_path: Optional path to agent configuration file
        """
        self.config_manager = get_config_manager()
        self.global_config: Optional[AgentConfig] = None
        self.inbox_configs: Dict[int, InboxConfig] = {}
        
        # LangChain components
        self.llm: Optional[ChatOpenAI] = None
        self.tools: List[BaseTool] = []
        self.agent_executor: Optional[AgentExecutor] = None
        
        # System state
        self._initialized = False
        self._conversation_contexts: Dict[int, ConversationContext] = {}
        self._memory_cache: Dict[str, AgentMemory] = {}
        
        # Performance metrics
        self._metrics = {
            'total_messages_processed': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            'total_tokens_used': 0,
            'total_cost_usd': 0.0,
            'daily_cost_usd': 0.0,
            'tools_usage': {},
            'response_times': [],
            'last_reset_date': datetime.utcnow().date()
        }
        
        logger.info("ChatwootAgent initialized", config_path=config_path)
    
    async def initialize(self) -> None:
        """Initialize all agent components and dependencies."""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing Chatwoot Agent...")
            
            # Load configuration
            await self._load_configuration()
            
            # Initialize LLM
            await self._initialize_llm()
            
            # Register tools
            await self._register_tools()
            
            # Create agent executor
            await self._create_agent_executor()
            
            # Initialize dependencies
            await self._initialize_dependencies()
            
            self._initialized = True
            logger.info("Chatwoot Agent initialization completed successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Chatwoot Agent", error=str(e))
            raise RuntimeError(f"Agent initialization failed: {e}")
    
    async def _load_configuration(self) -> None:
        """Load agent and inbox configurations."""
        try:
            # Load global agent configuration
            self.global_config = self.config_manager.get_agent_config()
            if not self.global_config:
                # Use default configuration
                self.global_config = AgentConfig()
                logger.warning("Using default agent configuration")
            
            logger.info("Agent configuration loaded", 
                       model=self.global_config.model,
                       enabled=self.global_config.enabled)
            
        except Exception as e:
            logger.error("Failed to load agent configuration", error=str(e))
            raise
    
    async def _initialize_llm(self) -> None:
        """Initialize the language model with configuration."""
        try:
            openai_config = get_openai_config()
            
            self.llm = ChatOpenAI(
                model=self.global_config.model,
                temperature=self.global_config.temperature,
                max_tokens=self.global_config.max_tokens,
                openai_api_key=openai_config.api_key,
                openai_api_base=openai_config.base_url,
                openai_organization=openai_config.organization,
                timeout=openai_config.timeout_seconds,
                max_retries=openai_config.max_retries,
                streaming=False  # Disable streaming for production stability
            )
            
            logger.info("LLM initialized successfully",
                       model=self.global_config.model,
                       temperature=self.global_config.temperature,
                       max_tokens=self.global_config.max_tokens)
            
        except Exception as e:
            logger.error("Failed to initialize LLM", error=str(e))
            raise
    
    async def _register_tools(self) -> None:
        """Register all available tools for the agent."""
        try:
            self.tools = []
            
            # Register RAG tool if enabled
            if self.global_config.rag.enabled:
                self.tools.extend([
                    retrieve_relevant_context,
                    store_conversation_context,
                    get_rag_performance_stats
                ])
                logger.info("RAG tools registered")
            
            # Register spreadsheet tool if enabled
            if self.global_config.sheets_tool.enabled:
                self.tools.extend([
                    query_spreadsheet_data,
                    list_available_spreadsheets,
                    get_spreadsheet_performance_stats
                ])
                logger.info("Spreadsheet tools registered")
            
            logger.info("Tools registration completed", tool_count=len(self.tools))
            
        except Exception as e:
            logger.error("Failed to register tools", error=str(e))
            raise
    
    async def _create_agent_executor(self) -> None:
        """Create the LangChain agent executor with ReAct pattern."""
        try:
            # Create system prompt template with persona injection
            system_prompt = """You are a helpful AI assistant integrated with Chatwoot, designed to provide excellent customer support.

Your role:
- Answer customer questions accurately and helpfully
- Use available tools to search for relevant information
- Maintain conversation context and provide personalized responses
- Be professional, friendly, and solution-oriented

Available tools:
{tools}

Tool usage guidelines:
- Always search for relevant context using RAG when customers reference past conversations
- Use spreadsheet tools to lookup order status, appointments, or structured data
- Store important conversation details for future reference
- Provide performance stats when requested for monitoring

{persona_instructions}

Current conversation context:
- Contact: {contact_name} ({contact_phone})
- Conversation ID: {conversation_id}
- Inbox: {inbox_id}

Remember to:
1. Be conversational and natural
2. Ask clarifying questions when needed
3. Provide specific, actionable answers
4. Use tools proactively to give accurate information
5. Maintain context throughout the conversation"""
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Create tool-calling agent
            agent = create_tool_calling_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            # Create agent executor with configuration
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=self.global_config.log_level.lower() == "debug",
                return_intermediate_steps=True,
                max_execution_time=300,  # 5 minutes max execution
                max_iterations=10,  # Limit iterations to prevent runaway
                handle_parsing_errors=True
            )
            
            logger.info("Agent executor created successfully")
            
        except Exception as e:
            logger.error("Failed to create agent executor", error=str(e))
            raise
    
    async def _initialize_dependencies(self) -> None:
        """Initialize external dependencies and services."""
        try:
            # Initialize RAG service
            if self.global_config.rag.enabled:
                rag_service = await get_rag_service()
                logger.debug("RAG service initialized")
            
            # Initialize spreadsheet service
            if self.global_config.sheets_tool.enabled:
                spreadsheet_service = await get_spreadsheet_service()
                logger.debug("Spreadsheet service initialized")
            
            # Test database connection
            db_service = await get_database_service()
            await db_service.health_check()
            logger.debug("Database service verified")
            
        except Exception as e:
            logger.error("Failed to initialize dependencies", error=str(e))
            raise
    
    async def get_inbox_config(self, inbox_id: int) -> InboxConfig:
        """
        Get configuration for a specific inbox.
        
        Args:
            inbox_id: Chatwoot inbox ID
            
        Returns:
            InboxConfig for the inbox, using defaults if not found
        """
        if inbox_id not in self.inbox_configs:
            # Try to load from file
            config = self.config_manager.load_inbox_config(inbox_id)
            if config:
                self.inbox_configs[inbox_id] = config
            else:
                # Create default inbox config
                self.inbox_configs[inbox_id] = InboxConfig(inbox_id=inbox_id)
                logger.debug("Using default inbox configuration", inbox_id=inbox_id)
        
        return self.inbox_configs[inbox_id]
    
    async def _get_conversation_context(self, webhook_payload: WebhookPayload) -> ConversationContext:
        """
        Get or create conversation context from webhook payload.
        
        Args:
            webhook_payload: Webhook data from Chatwoot
            
        Returns:
            ConversationContext for the conversation
        """
        conversation_id = webhook_payload.conversation.id
        
        if conversation_id not in self._conversation_contexts:
            # Create new context
            context = ConversationContext(
                conversation_id=conversation_id,
                contact_phone=webhook_payload.contact.phone_number or "unknown",
                inbox_id=webhook_payload.conversation.inbox_id,
                account_id=webhook_payload.account.id,
                contact_name=webhook_payload.contact.name,
                custom_attributes=webhook_payload.conversation.custom_attributes,
                labels=webhook_payload.conversation.labels
            )
            
            self._conversation_contexts[conversation_id] = context
            logger.debug("Created new conversation context", conversation_id=conversation_id)
        
        return self._conversation_contexts[conversation_id]
    
    async def _build_persona_instructions(self, inbox_config: InboxConfig) -> str:
        """
        Build persona instructions based on inbox configuration.
        
        Args:
            inbox_config: Configuration for the specific inbox
            
        Returns:
            Formatted persona instructions
        """
        if not inbox_config.persona:
            return "Use a professional and helpful tone appropriate for customer support."
        
        return f"""Persona Instructions:
{inbox_config.persona}

Language: {inbox_config.language}
Always respond in the specified language and maintain the persona throughout the conversation."""
    
    async def process_message(
        self,
        webhook_payload: WebhookPayload,
        governance_paused: bool = False
    ) -> AgentResponse:
        """
        Process an incoming message using the LangChain agent.
        
        Args:
            webhook_payload: Webhook data from Chatwoot
            governance_paused: Whether agent is paused for this conversation
            
        Returns:
            AgentResponse with processing results
        """
        start_time = time.time()
        conversation_id = webhook_payload.conversation.id
        contact_phone = webhook_payload.contact.phone_number or "unknown"
        user_message = webhook_payload.message.content
        
        logger.info("Processing message",
                   conversation_id=conversation_id,
                   contact_phone=contact_phone,
                   message_length=len(user_message),
                   governance_paused=governance_paused)
        
        try:
            # Check if agent is enabled
            if not self.global_config.enabled:
                return AgentResponse(
                    conversation_id=conversation_id,
                    contact_phone=contact_phone,
                    status="disabled",
                    error="Agent is globally disabled"
                )
            
            # Check governance controls
            if governance_paused:
                logger.info("Agent paused for conversation", conversation_id=conversation_id)
                return AgentResponse(
                    conversation_id=conversation_id,
                    contact_phone=contact_phone,
                    status="paused",
                    error="Agent processing is paused for this conversation"
                )
            
            # Get conversation context
            context = await self._get_conversation_context(webhook_payload)
            
            # Get inbox configuration
            inbox_config = await self.get_inbox_config(context.inbox_id)
            
            # Check if agent is enabled for this inbox
            if not inbox_config.enabled:
                return AgentResponse(
                    conversation_id=conversation_id,
                    contact_phone=contact_phone,
                    status="disabled",
                    error=f"Agent is disabled for inbox {context.inbox_id}"
                )
            
            # Build persona instructions
            persona_instructions = await self._build_persona_instructions(inbox_config)
            
            # Prepare agent input
            agent_input = {
                "input": user_message,
                "chat_history": context.conversation_history[-10:],  # Last 10 messages
                "contact_name": context.contact_name or "Customer",
                "contact_phone": contact_phone,
                "conversation_id": conversation_id,
                "inbox_id": context.inbox_id,
                "persona_instructions": persona_instructions,
                "tools": [tool.name for tool in self.tools]
            }
            
            # Create metrics callback
            metrics_callback = AgentMetricsCallback(self)
            
            # Create runnable config with callback
            config = RunnableConfig(
                callbacks=[metrics_callback],
                metadata={
                    "conversation_id": conversation_id,
                    "contact_phone": contact_phone,
                    "inbox_id": context.inbox_id
                }
            )
            
            # Execute agent
            logger.debug("Executing agent", conversation_id=conversation_id)
            result = await self.agent_executor.ainvoke(agent_input, config=config)
            
            # Extract response
            response_content = result.get("output", "I'm sorry, I couldn't process your request.")
            tools_used = [step[0].tool for step in result.get("intermediate_steps", [])]
            
            # Update conversation history
            context.conversation_history.extend([
                {"role": "user", "content": user_message, "timestamp": datetime.utcnow().isoformat()},
                {"role": "assistant", "content": response_content, "timestamp": datetime.utcnow().isoformat()}
            ])
            
            # Store conversation context in RAG if enabled
            if self.global_config.rag.enabled:
                try:
                    await store_conversation_context(
                        content=f"User: {user_message}\nAssistant: {response_content}",
                        contact_phone=contact_phone,
                        conversation_id=conversation_id
                    )
                except Exception as e:
                    logger.warning("Failed to store conversation context", error=str(e))
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            response = AgentResponse(
                conversation_id=conversation_id,
                contact_phone=contact_phone,
                response_content=response_content,
                tool_used=tools_used[0] if tools_used else None,
                confidence=0.85,  # Default confidence score
                processing_time_ms=processing_time_ms,
                requires_confirmation=False,  # Could be enhanced with governance rules
                status="success"
            )
            
            logger.info("Message processed successfully",
                       conversation_id=conversation_id,
                       response_length=len(response_content) if response_content else 0,
                       tools_used=tools_used,
                       processing_time_ms=processing_time_ms)
            
            return response
            
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            logger.error("Failed to process message",
                        conversation_id=conversation_id,
                        contact_phone=contact_phone,
                        error=str(e),
                        processing_time_ms=processing_time_ms)
            
            # Update error metrics
            await self._update_metrics(
                conversation_id=conversation_id,
                processing_time_ms=processing_time_ms,
                tools_used=[],
                token_count=0,
                success=False,
                error=str(e)
            )
            
            return AgentResponse(
                conversation_id=conversation_id,
                contact_phone=contact_phone,
                processing_time_ms=processing_time_ms,
                status="error",
                error=f"Processing failed: {str(e)}"
            )
    
    async def _update_metrics(
        self,
        conversation_id: Optional[int],
        processing_time_ms: int,
        tools_used: List[str],
        token_count: int,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """
        Update agent performance metrics.
        
        Args:
            conversation_id: Conversation ID being processed
            processing_time_ms: Processing time in milliseconds
            tools_used: List of tools used in processing
            token_count: Number of tokens consumed
            success: Whether processing was successful
            error: Error message if failed
        """
        try:
            # Reset daily metrics if needed
            current_date = datetime.utcnow().date()
            if current_date > self._metrics['last_reset_date']:
                self._metrics['daily_cost_usd'] = 0.0
                self._metrics['last_reset_date'] = current_date
            
            # Update counters
            self._metrics['total_messages_processed'] += 1
            if success:
                self._metrics['successful_responses'] += 1
            else:
                self._metrics['failed_responses'] += 1
            
            # Update performance metrics
            self._metrics['response_times'].append(processing_time_ms)
            if len(self._metrics['response_times']) > 1000:  # Keep last 1000 times
                self._metrics['response_times'] = self._metrics['response_times'][-1000:]
            
            # Update token and cost metrics
            self._metrics['total_tokens_used'] += token_count
            
            # Estimate cost (rough calculation for gpt-4o)
            estimated_cost = (token_count / 1000) * 0.03  # $0.03 per 1K tokens
            self._metrics['total_cost_usd'] += estimated_cost
            self._metrics['daily_cost_usd'] += estimated_cost
            
            # Update tool usage
            for tool in tools_used:
                self._metrics['tools_usage'][tool] = self._metrics['tools_usage'].get(tool, 0) + 1
            
            # Log metrics periodically
            if self._metrics['total_messages_processed'] % 100 == 0:
                logger.info("Agent metrics update",
                           total_processed=self._metrics['total_messages_processed'],
                           success_rate=self._metrics['successful_responses'] / self._metrics['total_messages_processed'],
                           daily_cost=self._metrics['daily_cost_usd'])
            
        except Exception as e:
            logger.error("Failed to update metrics", error=str(e))
    
    async def get_global_config(self) -> Dict[str, Any]:
        """Get global agent configuration."""
        if not self.global_config:
            return {}
        return self.global_config.dict()
    
    async def get_metrics(self) -> MetricsResponse:
        """
        Get agent performance metrics.
        
        Returns:
            MetricsResponse with current metrics
        """
        avg_response_time = (
            sum(self._metrics['response_times']) / len(self._metrics['response_times'])
            if self._metrics['response_times'] else 0.0
        )
        
        return MetricsResponse(
            total_messages_processed=self._metrics['total_messages_processed'],
            successful_responses=self._metrics['successful_responses'],
            failed_responses=self._metrics['failed_responses'],
            average_response_time_ms=avg_response_time,
            total_tokens_used=self._metrics['total_tokens_used'],
            total_cost_usd=self._metrics['total_cost_usd'],
            daily_cost_usd=self._metrics['daily_cost_usd'],
            tools_usage=self._metrics['tools_usage'].copy()
        )
    
    async def get_conversation_logs(self, conversation_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get conversation logs for debugging and analysis.
        
        Args:
            conversation_id: Conversation ID to get logs for
            limit: Maximum number of log entries
            
        Returns:
            List of conversation log entries
        """
        try:
            context = self._conversation_contexts.get(conversation_id)
            if not context:
                return []
            
            # Return recent conversation history
            history = context.conversation_history[-limit:]
            return [
                {
                    "timestamp": entry.get("timestamp"),
                    "role": entry.get("role"),
                    "content": entry.get("content", "")[:500],  # Truncate for logs
                    "conversation_id": conversation_id
                }
                for entry in history
            ]
            
        except Exception as e:
            logger.error("Failed to get conversation logs", 
                        conversation_id=conversation_id, error=str(e))
            return []
    
    async def is_healthy(self) -> bool:
        """
        Check if agent is healthy and ready to process messages.
        
        Returns:
            True if agent is healthy, False otherwise
        """
        try:
            if not self._initialized:
                return False
            
            if not self.llm or not self.agent_executor:
                return False
            
            # Test a simple agent invocation
            test_input = {
                "input": "health check",
                "chat_history": [],
                "contact_name": "Test",
                "contact_phone": "+1234567890",
                "conversation_id": 0,
                "inbox_id": 0,
                "persona_instructions": "Respond briefly",
                "tools": []
            }
            
            # Quick health check with timeout
            config = RunnableConfig(metadata={"timeout": 10})
            await asyncio.wait_for(
                self.agent_executor.ainvoke(test_input, config=config),
                timeout=10.0
            )
            
            return True
            
        except asyncio.TimeoutError:
            logger.warning("Agent health check timed out")
            return False
        except Exception as e:
            logger.error("Agent health check failed", error=str(e))
            return False


# Global agent instance
_agent_instance: Optional[ChatwootAgent] = None


async def get_agent() -> ChatwootAgent:
    """
    Get the global agent instance.
    
    Returns:
        Initialized ChatwootAgent instance
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ChatwootAgent()
        await _agent_instance.initialize()
    return _agent_instance


async def shutdown_agent() -> None:
    """Shutdown the global agent instance."""
    global _agent_instance
    if _agent_instance:
        logger.info("Shutting down Chatwoot Agent")
        _agent_instance = None