"""
Tests for the structured logging utility.
"""

import json
import uuid
from datetime import datetime
from io import StringIO

import pytest
from unittest.mock import patch, MagicMock

from src.utils.logging import (
    StructuredLogger, ChatwootJsonFormatter, ContextLogger,
    correlation_id, conversation_id, contact_phone,
    get_logger, log_context, setup_logging
)


class TestChatwootJsonFormatter:
    """Tests for the custom JSON formatter."""
    
    def test_add_fields_basic(self):
        """Test basic field addition."""
        formatter = ChatwootJsonFormatter()
        log_record = {}
        
        # Create a mock log record
        record = MagicMock()
        record.name = "test_logger"
        record.levelname = "INFO"
        record.created = datetime.utcnow().timestamp()
        
        formatter.add_fields(log_record, record, {})
        
        assert 'timestamp' in log_record
        assert 'service' in log_record
        assert log_record['service'] == 'chatwoot-agent'
        assert 'version' in log_record
        assert log_record['logger_name'] == 'test_logger'
        assert log_record['level'] == 'INFO'
    
    def test_add_fields_with_context(self):
        """Test field addition with context variables."""
        formatter = ChatwootJsonFormatter()
        
        with patch.object(correlation_id, 'get', return_value='test-correlation-123'):
            with patch.object(conversation_id, 'get', return_value=12345):
                with patch.object(contact_phone, 'get', return_value='+1234567890'):
                    log_record = {}
                    record = MagicMock()
                    record.name = "test"
                    record.levelname = "INFO"
                    
                    formatter.add_fields(log_record, record, {})
                    
                    assert log_record['correlation_id'] == 'test-correlation-123'
                    assert log_record['conversation_id'] == 12345
                    assert log_record['contact_phone'] == '+1234567890'
    
    def test_add_fields_with_performance_metrics(self):
        """Test field addition with performance metrics."""
        formatter = ChatwootJsonFormatter()
        log_record = {}
        
        record = MagicMock()
        record.name = "test"
        record.levelname = "INFO"
        record.duration_ms = 150
        record.tokens_used = 500
        record.cost_usd = 0.001
        
        formatter.add_fields(log_record, record, {})
        
        assert log_record['duration_ms'] == 150
        assert log_record['tokens_used'] == 500
        assert log_record['cost_usd'] == 0.001


class TestStructuredLogger:
    """Tests for the StructuredLogger class."""
    
    def test_initialization(self):
        """Test logger initialization."""
        logger = StructuredLogger("test_logger")
        assert logger.logger.name == "test_logger"
    
    def test_with_context(self):
        """Test context logger creation."""
        logger = StructuredLogger("test_logger")
        
        context_logger = logger.with_context(
            correlation_id_val="test-123",
            conversation_id_val=12345,
            contact_phone_val="+1234567890"
        )
        
        assert isinstance(context_logger, ContextLogger)
        assert context_logger.correlation_id_val == "test-123"
        assert context_logger.conversation_id_val == 12345
        assert context_logger.contact_phone_val == "+1234567890"
    
    @patch('src.utils.logging.logging.StreamHandler')
    def test_logging_methods(self, mock_handler):
        """Test basic logging methods."""
        logger = StructuredLogger("test_logger")
        
        # Mock the handler to capture log calls
        mock_logger = MagicMock()
        logger.logger = mock_logger
        
        # Test each logging method
        logger.info("Test info", extra_field="value")
        mock_logger.info.assert_called_with("Test info", extra={'extra_field': 'value'})
        
        logger.error("Test error", error_code=500)
        mock_logger.error.assert_called_with("Test error", extra={'error_code': 500})
        
        logger.warning("Test warning", component="api")
        mock_logger.warning.assert_called_with("Test warning", extra={'component': 'api'})
        
        logger.debug("Test debug", request_id="123")
        mock_logger.debug.assert_called_with("Test debug", extra={'request_id': '123'})


class TestContextLogger:
    """Tests for the ContextLogger class."""
    
    def test_context_manager(self):
        """Test context manager functionality."""
        mock_logger = MagicMock()
        
        context_logger = ContextLogger(
            mock_logger,
            "test-correlation-123",
            12345,
            "+1234567890"
        )
        
        with context_logger as ctx:
            # Context variables should be set
            assert ctx is context_logger
            # We can't easily test the context variable setting due to async nature
            # but we can test that the context manager works
        
        # Context should be cleaned up after exiting
        # This is handled by the context variable system
    
    def test_logging_methods(self):
        """Test logging methods in context logger."""
        mock_logger = MagicMock()
        
        context_logger = ContextLogger(
            mock_logger,
            "test-correlation-123",
            12345,
            "+1234567890"
        )
        
        context_logger.info("Test message", component="webhook")
        mock_logger.info.assert_called_with("Test message", extra={'component': 'webhook'})
        
        context_logger.error("Test error", error="Something went wrong")
        mock_logger.error.assert_called_with("Test error", extra={'error': 'Something went wrong'})
    
    def test_log_performance(self):
        """Test performance logging."""
        mock_logger = MagicMock()
        
        context_logger = ContextLogger(
            mock_logger,
            "test-correlation-123",
            12345,
            "+1234567890"
        )
        
        context_logger.log_performance(
            "api_call",
            150,
            tokens_used=500,
            cost_usd=0.001,
            endpoint="/api/v1/messages"
        )
        
        expected_extra = {
            'operation': 'api_call',
            'duration_ms': 150,
            'tokens_used': 500,
            'cost_usd': 0.001,
            'endpoint': '/api/v1/messages'
        }
        
        mock_logger.info.assert_called_with("Performance: api_call", extra=expected_extra)
    
    def test_log_agent_response(self):
        """Test agent response logging."""
        mock_logger = MagicMock()
        
        context_logger = ContextLogger(
            mock_logger,
            "test-correlation-123",
            12345,
            "+1234567890"
        )
        
        context_logger.log_agent_response(
            "What is my order status?",
            "Your order #1234 is shipped",
            tool_used="order_lookup",
            confidence=0.95,
            duration_ms=200,
            model="gpt-4"
        )
        
        expected_extra = {
            'event_type': 'agent_response',
            'user_query': 'What is my order status?',
            'agent_response': 'Your order #1234 is shipped',
            'tool_used': 'order_lookup',
            'confidence': 0.95,
            'duration_ms': 200,
            'model': 'gpt-4'
        }
        
        mock_logger.info.assert_called_with("Agent response generated", extra=expected_extra)


class TestGlobalFunctions:
    """Tests for global logging functions."""
    
    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("test_module")
        assert isinstance(logger, StructuredLogger)
        assert logger.logger.name == "test_module"
    
    def test_log_context_function(self):
        """Test log_context function."""
        context = log_context(
            correlation_id_val="test-123",
            conversation_id_val=12345,
            contact_phone_val="+1234567890"
        )
        
        assert isinstance(context, ContextLogger)
        assert context.correlation_id_val == "test-123"
        assert context.conversation_id_val == 12345
        assert context.contact_phone_val == "+1234567890"
    
    def test_log_context_with_default_correlation_id(self):
        """Test log_context with auto-generated correlation ID."""
        context = log_context(
            conversation_id_val=12345,
            contact_phone_val="+1234567890"
        )
        
        # Should generate a UUID for correlation ID
        assert context.correlation_id_val is not None
        assert len(context.correlation_id_val) == 36  # UUID length
        assert context.conversation_id_val == 12345
        assert context.contact_phone_val == "+1234567890"


class TestSetupLogging:
    """Tests for logging setup."""
    
    @patch('src.utils.logging.logging.basicConfig')
    @patch('src.utils.logging.logging.getLogger')
    @patch('src.utils.logging.logging.StreamHandler')
    @patch('src.utils.logging.logging.FileHandler')
    def test_setup_logging_basic(self, mock_file_handler, mock_stream_handler, mock_get_logger, mock_basic_config):
        """Test basic logging setup."""
        mock_root_logger = MagicMock()
        mock_root_logger.handlers = []
        mock_get_logger.return_value = mock_root_logger
        
        setup_logging(level="INFO")
        
        # Should configure basic logging
        mock_basic_config.assert_called_once()
        
        # Should add stream handler
        mock_stream_handler.assert_called_once()
        mock_root_logger.addHandler.assert_called()
        
        # Should set log level
        mock_root_logger.setLevel.assert_called()
    
    @patch('src.utils.logging.logging.basicConfig')
    @patch('src.utils.logging.logging.getLogger')
    @patch('src.utils.logging.logging.StreamHandler')
    @patch('src.utils.logging.logging.FileHandler')
    def test_setup_logging_with_file(self, mock_file_handler, mock_stream_handler, mock_get_logger, mock_basic_config, temp_dir):
        """Test logging setup with file output."""
        mock_root_logger = MagicMock()
        mock_root_logger.handlers = []
        mock_get_logger.return_value = mock_root_logger
        
        log_file = temp_dir / "test.log"
        setup_logging(level="DEBUG", log_file=log_file)
        
        # Should add both stream and file handlers
        mock_stream_handler.assert_called_once()
        mock_file_handler.assert_called_once_with(log_file)
        
        # Should call addHandler twice (stream + file)
        assert mock_root_logger.addHandler.call_count == 2


@pytest.mark.asyncio
class TestAsyncLogging:
    """Tests for async logging scenarios."""
    
    async def test_context_preservation_across_awaits(self):
        """Test that context is preserved across async operations."""
        logger = get_logger("test_async")
        
        async def async_operation():
            await pytest.importorskip("asyncio").sleep(0.01)
            return "done"
        
        with logger.with_context(
            correlation_id_val="async-test-123",
            conversation_id_val=99999
        ) as ctx:
            # Context should be available
            result = await async_operation()
            assert result == "done"
            
            # Log after async operation
            with patch.object(ctx.logger, 'info') as mock_info:
                ctx.info("After async operation")
                mock_info.assert_called_once_with("After async operation", extra={})


class TestErrorHandling:
    """Tests for error handling in logging."""
    
    def test_formatter_with_invalid_timestamp(self):
        """Test formatter handles invalid timestamp gracefully."""
        formatter = ChatwootJsonFormatter()
        log_record = {'timestamp': 'invalid-timestamp'}
        record = MagicMock()
        record.name = "test"
        record.levelname = "ERROR"
        
        # Should not raise exception
        formatter.add_fields(log_record, record, {})
        
        # Should have a valid timestamp
        assert 'timestamp' in log_record
    
    def test_context_logger_exception_handling(self):
        """Test context logger exception logging."""
        mock_logger = MagicMock()
        
        context_logger = ContextLogger(
            mock_logger,
            "test-correlation-123"
        )
        
        try:
            raise ValueError("Test exception")
        except Exception:
            context_logger.exception("An error occurred", component="test")
            mock_logger.exception.assert_called_with(
                "An error occurred", 
                extra={'component': 'test'}
            )