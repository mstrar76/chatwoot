"""
Structured logging setup for the Chatwoot Agent Service.

Provides production-ready JSON structured logging with custom formatters,
correlation IDs, and comprehensive error handling.
"""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional, Union
from pathlib import Path

from pythonjsonlogger import jsonlogger


# Context variables for request tracing
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
conversation_id: ContextVar[Optional[int]] = ContextVar('conversation_id', default=None)
contact_phone: ContextVar[Optional[str]] = ContextVar('contact_phone', default=None)


class ChatwootJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter that includes structured fields for Chatwoot agent logging.
    
    Adds correlation ID, conversation context, and performance metrics to all log entries.
    """
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """Add custom fields to the log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Always include timestamp in ISO format
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        
        # Add service identification
        log_record['service'] = 'chatwoot-agent'
        log_record['version'] = '1.0.0'
        
        # Add context variables if available
        if correlation_id.get():
            log_record['correlation_id'] = correlation_id.get()
        
        if conversation_id.get():
            log_record['conversation_id'] = conversation_id.get()
            
        if contact_phone.get():
            log_record['contact_phone'] = contact_phone.get()
        
        # Add execution context
        log_record['logger_name'] = record.name
        log_record['level'] = record.levelname
        
        # Add performance metrics if available in extra
        if hasattr(record, 'duration_ms'):
            log_record['duration_ms'] = record.duration_ms
        
        if hasattr(record, 'tokens_used'):
            log_record['tokens_used'] = record.tokens_used
            
        if hasattr(record, 'cost_usd'):
            log_record['cost_usd'] = record.cost_usd


class StructuredLogger:
    """
    Production-grade structured logger with context management.
    
    Provides methods for logging with consistent structure and automatic
    correlation ID management for request tracing.
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Configure the logger with JSON formatting."""
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = ChatwootJsonFormatter(
                fmt='%(timestamp)s %(level)s %(name)s %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Set default level
        self.logger.setLevel(logging.INFO)
    
    def set_level(self, level: Union[str, int]) -> None:
        """Set the logging level."""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.logger.setLevel(level)
    
    def with_context(
        self,
        correlation_id_val: Optional[str] = None,
        conversation_id_val: Optional[int] = None,
        contact_phone_val: Optional[str] = None
    ) -> 'ContextLogger':
        """
        Create a context logger with specific context variables.
        
        Args:
            correlation_id_val: Request correlation ID
            conversation_id_val: Chatwoot conversation ID
            contact_phone_val: Contact phone number
            
        Returns:
            ContextLogger instance with set context
        """
        return ContextLogger(
            self.logger,
            correlation_id_val or str(uuid.uuid4()),
            conversation_id_val,
            contact_phone_val
        )
    
    def info(self, message: str, **kwargs) -> None:
        """Log info level message with structured data."""
        self.logger.info(message, extra=kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error level message with structured data."""
        self.logger.error(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning level message with structured data."""
        self.logger.warning(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug level message with structured data."""
        self.logger.debug(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with stack trace and structured data."""
        self.logger.exception(message, extra=kwargs)


class ContextLogger:
    """
    Logger with preset context variables for request tracing.
    
    Automatically includes correlation ID and conversation context
    in all log messages within the context manager.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        correlation_id_val: str,
        conversation_id_val: Optional[int] = None,
        contact_phone_val: Optional[str] = None
    ):
        self.logger = logger
        self.correlation_id_val = correlation_id_val
        self.conversation_id_val = conversation_id_val
        self.contact_phone_val = contact_phone_val
        self._tokens = []
    
    def __enter__(self):
        """Enter context manager and set context variables."""
        self._tokens = [
            correlation_id.set(self.correlation_id_val),
            conversation_id.set(self.conversation_id_val),
            contact_phone.set(self.contact_phone_val)
        ]
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and reset context variables."""
        for token in reversed(self._tokens):
            if token:
                token.var.set(token.old_value)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info level message with context."""
        self.logger.info(message, extra=kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error level message with context."""
        self.logger.error(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning level message with context."""
        self.logger.warning(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug level message with context."""
        self.logger.debug(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with stack trace and context."""
        self.logger.exception(message, extra=kwargs)
    
    def log_performance(
        self,
        operation: str,
        duration_ms: int,
        tokens_used: Optional[int] = None,
        cost_usd: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Log performance metrics for an operation.
        
        Args:
            operation: Name of the operation performed
            duration_ms: Duration in milliseconds
            tokens_used: Number of LLM tokens used
            cost_usd: Cost in USD for the operation
            **kwargs: Additional structured data
        """
        extra = {
            'operation': operation,
            'duration_ms': duration_ms,
            **kwargs
        }
        
        if tokens_used is not None:
            extra['tokens_used'] = tokens_used
        
        if cost_usd is not None:
            extra['cost_usd'] = cost_usd
        
        self.logger.info(f"Performance: {operation}", extra=extra)
    
    def log_agent_response(
        self,
        user_query: str,
        agent_response: str,
        tool_used: Optional[str] = None,
        confidence: Optional[float] = None,
        duration_ms: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Log agent response with structured data for analysis.
        
        Args:
            user_query: User's original query
            agent_response: Agent's response
            tool_used: Tool used by the agent
            confidence: Confidence score (0-1)
            duration_ms: Response generation time
            **kwargs: Additional structured data
        """
        extra = {
            'event_type': 'agent_response',
            'user_query': user_query,
            'agent_response': agent_response,
            **kwargs
        }
        
        if tool_used:
            extra['tool_used'] = tool_used
        
        if confidence is not None:
            extra['confidence'] = confidence
            
        if duration_ms is not None:
            extra['duration_ms'] = duration_ms
        
        self.logger.info("Agent response generated", extra=extra)


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Setup global logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
    """
    logging.basicConfig(level=getattr(logging, level.upper()))
    
    # Configure root logger
    root_logger = logging.getLogger()
    
    # Remove default handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add structured JSON handler
    handler = logging.StreamHandler(sys.stdout)
    formatter = ChatwootJsonFormatter(
        fmt='%(timestamp)s %(level)s %(name)s %(message)s'
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set level
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Reduce noise from external libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('asyncpg').setLevel(logging.WARNING)
    logging.getLogger('aiofiles').setLevel(logging.WARNING)


# Global logger instance
logger = StructuredLogger(__name__)


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)


# Context manager for setting request context
def log_context(
    correlation_id_val: Optional[str] = None,
    conversation_id_val: Optional[int] = None,
    contact_phone_val: Optional[str] = None
) -> ContextLogger:
    """
    Create a logging context manager.
    
    Args:
        correlation_id_val: Request correlation ID
        conversation_id_val: Chatwoot conversation ID  
        contact_phone_val: Contact phone number
        
    Returns:
        ContextLogger instance
    """
    return logger.with_context(
        correlation_id_val=correlation_id_val or str(uuid.uuid4()),
        conversation_id_val=conversation_id_val,
        contact_phone_val=contact_phone_val
    )