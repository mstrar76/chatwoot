"""
Pydantic models and schemas for the Chatwoot Agent Service.

These models define the data structures for webhook payloads, configurations,
and API responses used throughout the agent system.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class EventType(str, Enum):
    """Chatwoot webhook event types."""
    MESSAGE_CREATED = "message_created"
    MESSAGE_UPDATED = "message_updated"
    CONVERSATION_CREATED = "conversation_created"
    CONVERSATION_UPDATED = "conversation_updated"
    CONVERSATION_STATUS_CHANGED = "conversation_status_changed"


class MessageType(str, Enum):
    """Message types supported by the agent."""
    INCOMING = "incoming"
    OUTGOING = "outgoing"
    ACTIVITY = "activity"
    TEMPLATE = "template"


class ContentType(str, Enum):
    """Content types for messages."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"


class SenderType(str, Enum):
    """Sender types in Chatwoot."""
    CONTACT = "contact"
    USER = "user"
    AGENT_BOT = "agent_bot"


class ConversationStatus(str, Enum):
    """Conversation status values."""
    OPEN = "open"
    RESOLVED = "resolved"
    PENDING = "pending"
    SNOOZED = "snoozed"


# Webhook payload models
class Sender(BaseModel):
    """Sender information in webhook payload."""
    id: int
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    type: SenderType


class Attachment(BaseModel):
    """Attachment information in messages."""
    id: Optional[int] = None
    file_url: Optional[str] = None
    file_type: Optional[str] = None
    data_url: Optional[str] = None


class Message(BaseModel):
    """Message data from Chatwoot webhook."""
    id: int
    account_id: int
    inbox_id: int
    conversation_id: int
    content: str
    message_type: MessageType
    content_type: ContentType
    content_attributes: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    sender: Sender
    attachments: List[Attachment] = Field(default_factory=list)
    source_id: Optional[str] = None


class Contact(BaseModel):
    """Contact information from Chatwoot."""
    id: int
    name: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    additional_attributes: Dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    """Conversation data from Chatwoot webhook."""
    id: int
    status: ConversationStatus
    assignee_id: Optional[int] = None
    inbox_id: int
    contact_id: int
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)
    labels: List[str] = Field(default_factory=list)
    created_at: datetime


class Account(BaseModel):
    """Account information from Chatwoot."""
    id: int
    name: str


class WebhookPayload(BaseModel):
    """Complete webhook payload from Chatwoot."""
    event: EventType
    timestamp: Optional[datetime] = None
    message: Optional[Message] = None
    contact: Optional[Contact] = None
    conversation: Optional[Conversation] = None
    account: Account

    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        """Set timestamp to current time if not provided."""
        return v or datetime.utcnow()


# Configuration models
class RAGConfig(BaseModel):
    """RAG (Retrieval Augmented Generation) configuration."""
    enabled: bool = True
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 4
    time_window_days: int = 90
    vector_table: str = "public.embeddings"
    similarity_threshold: float = 0.7


class SheetsToolConfig(BaseModel):
    """Spreadsheet tool configuration."""
    enabled: bool = True
    sheet_configs: List[str] = Field(default_factory=list)
    cache_ttl_minutes: int = 5
    csv_path: Optional[str] = None


class MultimodalConfig(BaseModel):
    """Multimodal processing configuration."""
    audio_transcripts: bool = True
    image_ocr: bool = False
    max_file_size_mb: int = 10


class CostLimitsConfig(BaseModel):
    """Cost control configuration."""
    daily_budget_usd: float = 50.0
    monthly_budget_usd: float = 1000.0
    per_message_max_tokens: int = 2000
    alert_threshold_percent: float = 80.0


class AgentConfig(BaseModel):
    """Global agent configuration."""
    enabled: bool = True
    llm_provider: str = "openai"
    model: str = "gpt-4o"
    max_tokens: int = 1024
    temperature: float = 0.3
    preview_delay_seconds: int = 15
    rag: RAGConfig = Field(default_factory=RAGConfig)
    sheets_tool: SheetsToolConfig = Field(default_factory=SheetsToolConfig)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)
    cost_limits: CostLimitsConfig = Field(default_factory=CostLimitsConfig)
    log_level: str = "info"


class InboxConfig(BaseModel):
    """Per-inbox agent configuration."""
    inbox_id: int
    enabled: bool = True
    language: str = "en"
    preview_delay_seconds: Optional[int] = None
    persona: Optional[str] = None
    rag: Optional[RAGConfig] = None
    sheets_tool: Optional[SheetsToolConfig] = None
    multimodal: Optional[MultimodalConfig] = None


# Memory models
class AgentMemory(BaseModel):
    """Per-contact memory storage."""
    contact_phone: str
    conversation_id: Optional[int] = None
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# API response models
class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    services: Dict[str, bool] = Field(default_factory=dict)
    version: str
    error: Optional[str] = None


class AgentResponse(BaseModel):
    """Agent processing response."""
    conversation_id: int
    contact_phone: str
    response_content: Optional[str] = None
    tool_used: Optional[str] = None
    confidence: float = 0.0
    processing_time_ms: int = 0
    requires_confirmation: bool = False
    status: str = "success"
    error: Optional[str] = None


class MetricsResponse(BaseModel):
    """Agent metrics response."""
    total_messages_processed: int = 0
    successful_responses: int = 0
    failed_responses: int = 0
    average_response_time_ms: float = 0.0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    daily_cost_usd: float = 0.0
    tools_usage: Dict[str, int] = Field(default_factory=dict)


class LogEntry(BaseModel):
    """Structured log entry."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    conversation_id: int
    contact_phone: str
    user_query: str
    tool_used: Optional[str] = None
    final_response: Optional[str] = None
    latency_ms: int = 0
    token_count: int = 0
    cost_usd: float = 0.0
    status: str = "success"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)