"""
Webhook handler for Chatwoot message_created events.

Processes incoming webhooks, validates payloads, implements loop prevention,
and routes messages to agent processing pipeline.
"""

import hmac
import hashlib
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from pydantic import ValidationError as PydanticValidationError

from ..models.schemas import (
    WebhookPayload, EventType, MessageType, ContentType, SenderType,
    Message, Contact, Conversation, Account, Sender
)
from ..services.chatwoot_api import get_chatwoot_client, ChatwootAPIClient
from ..services.database import get_database_service, DatabaseService
from ..utils.config import get_chatwoot_config
from ..utils.logging import get_logger, log_context

logger = get_logger(__name__)


class WebhookError(Exception):
    """Base exception for webhook processing errors."""
    pass


class ValidationError(WebhookError):
    """Exception for webhook payload validation errors."""
    pass


class SecurityError(WebhookError):
    """Exception for webhook security/authentication errors."""
    pass


class ProcessingError(WebhookError):
    """Exception for webhook processing errors."""
    pass


class WebhookHandler:
    """
    Production-ready webhook handler for Chatwoot events.
    
    Features:
    - Webhook payload validation and parsing
    - HMAC signature verification for security
    - Loop prevention with multiple strategies
    - Message filtering for relevant content
    - Agent integration with governance controls
    - Structured logging for all events
    - Database integration for message storage
    """
    
    def __init__(
        self,
        agent=None,
        chatwoot_client: Optional[ChatwootAPIClient] = None,
        governance=None
    ):
        self.agent = agent
        self.chatwoot_client = chatwoot_client
        self.governance = governance
        self.config = get_chatwoot_config()
    
    async def _ensure_dependencies(self) -> None:
        """Ensure required dependencies are initialized."""
        if self.chatwoot_client is None:
            self.chatwoot_client = await get_chatwoot_client()
    
    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
        secret: Optional[str] = None
    ) -> bool:
        """
        Verify webhook HMAC signature for security.
        
        Args:
            payload: Raw webhook payload bytes
            signature: Provided signature header
            secret: Webhook secret (from config if not provided)
            
        Returns:
            True if signature is valid, False otherwise
        """
        if not secret:
            secret = self.config.webhook_secret
        
        if not secret:
            logger.warning("No webhook secret configured, skipping signature verification")
            return True  # Allow if no secret is configured
        
        if not signature:
            logger.error("No signature provided but webhook secret is configured")
            return False
        
        try:
            # Remove 'sha256=' prefix if present
            if signature.startswith('sha256='):
                signature = signature[7:]
            
            # Calculate expected signature
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            # Use secure comparison to prevent timing attacks
            is_valid = hmac.compare_digest(signature, expected_signature)
            
            if not is_valid:
                logger.error("Webhook signature verification failed",
                           provided_signature=signature,
                           expected_signature=expected_signature)
            
            return is_valid
            
        except Exception as e:
            logger.error("Error during signature verification", error=str(e))
            return False
    
    def parse_webhook_payload(self, raw_payload: Dict[str, Any]) -> WebhookPayload:
        """
        Parse and validate webhook payload.
        
        Args:
            raw_payload: Raw webhook payload dictionary
            
        Returns:
            Validated WebhookPayload instance
            
        Raises:
            ValidationError: If payload validation fails
        """
        try:
            # Handle different timestamp formats
            if 'timestamp' in raw_payload and isinstance(raw_payload['timestamp'], str):
                # Convert ISO string to datetime if needed
                try:
                    raw_payload['timestamp'] = datetime.fromisoformat(
                        raw_payload['timestamp'].replace('Z', '+00:00')
                    )
                except ValueError:
                    # If parsing fails, let Pydantic handle it
                    pass
            
            # Validate and parse the payload
            webhook_payload = WebhookPayload(**raw_payload)
            
            logger.debug("Webhook payload parsed successfully",
                        event=webhook_payload.event,
                        account_id=webhook_payload.account.id,
                        conversation_id=webhook_payload.conversation.id if webhook_payload.conversation else None,
                        message_id=webhook_payload.message.id if webhook_payload.message else None)
            
            return webhook_payload
            
        except PydanticValidationError as e:
            logger.error("Webhook payload validation failed", 
                        validation_errors=str(e),
                        raw_payload_keys=list(raw_payload.keys()))
            raise ValidationError(f"Invalid webhook payload: {e}")
        
        except Exception as e:
            logger.error("Unexpected error parsing webhook payload", 
                        error=str(e),
                        error_type=type(e).__name__)
            raise ValidationError(f"Failed to parse webhook payload: {e}")
    
    def is_agent_generated_message(self, message: Message) -> bool:
        """
        Check if message was generated by an agent (loop prevention).
        
        Args:
            message: Message instance to check
            
        Returns:
            True if message is agent-generated, False otherwise
        """
        # Check content_attributes for agent markers
        content_attrs = message.content_attributes or {}
        
        # Our agent marks messages with agent_processed flag
        if content_attrs.get('agent_processed'):
            logger.debug("Message marked as agent_processed",
                        message_id=message.id,
                        conversation_id=message.conversation_id)
            return True
        
        # Check for echo_id (unique identifier we add to our messages)
        if content_attrs.get('echo_id'):
            logger.debug("Message has echo_id (agent-generated)",
                        message_id=message.id,
                        conversation_id=message.conversation_id,
                        echo_id=content_attrs.get('echo_id'))
            return True
        
        # Check message type and sender type combinations
        if message.message_type == MessageType.OUTGOING:
            logger.debug("Outgoing message detected (likely agent-generated)",
                        message_id=message.id,
                        conversation_id=message.conversation_id)
            return True
        
        # Check sender type - agent_bot messages are from agents
        if message.sender.type == SenderType.AGENT_BOT:
            logger.debug("Message from agent_bot sender",
                        message_id=message.id,
                        conversation_id=message.conversation_id)
            return True
        
        # Additional check: messages from system users
        if (message.sender.type == SenderType.USER and 
            message.message_type == MessageType.OUTGOING):
            logger.debug("Outgoing message from user (likely agent)",
                        message_id=message.id,
                        conversation_id=message.conversation_id)
            return True
        
        return False
    
    def should_process_message(self, webhook_payload: WebhookPayload) -> tuple[bool, str]:
        """
        Determine if webhook message should be processed.
        
        Args:
            webhook_payload: Validated webhook payload
            
        Returns:
            Tuple of (should_process: bool, reason: str)
        """
        # Only process message_created events
        if webhook_payload.event != EventType.MESSAGE_CREATED:
            return False, f"Event type {webhook_payload.event} is not message_created"
        
        # Must have message data
        if not webhook_payload.message:
            return False, "No message data in webhook payload"
        
        message = webhook_payload.message
        
        # Only process incoming messages
        if message.message_type != MessageType.INCOMING:
            return False, f"Message type {message.message_type} is not incoming"
        
        # Only process text messages for now
        if message.content_type != ContentType.TEXT:
            return False, f"Content type {message.content_type} is not text"
        
        # Skip empty messages
        if not message.content or not message.content.strip():
            return False, "Message content is empty"
        
        # Loop prevention - skip agent-generated messages
        if self.is_agent_generated_message(message):
            return False, "Message was generated by agent (loop prevention)"
        
        # Only process messages from contacts
        if message.sender.type != SenderType.CONTACT:
            return False, f"Sender type {message.sender.type} is not contact"
        
        # Must have conversation data
        if not webhook_payload.conversation:
            return False, "No conversation data in webhook payload"
        
        # Additional validation for WhatsApp source_id format
        if message.source_id and message.source_id.startswith('contact:'):
            # Extract phone number from source_id
            phone_number = message.source_id.replace('contact:', '')
            if not phone_number:
                return False, "Invalid source_id format - no phone number"
        
        return True, "Message passes all filters and should be processed"
    
    async def store_message_in_database(
        self,
        webhook_payload: WebhookPayload
    ) -> Optional[int]:
        """
        Store incoming message in database for context and history.
        
        Args:
            webhook_payload: Validated webhook payload
            
        Returns:
            Message ID if stored successfully, None otherwise
        """
        try:
            # Get database service
            database_service = await get_database_service()
            
            message = webhook_payload.message
            contact = webhook_payload.contact
            
            # Extract phone number from source_id or contact data
            phone_number = None
            if message.source_id and message.source_id.startswith('contact:'):
                phone_number = message.source_id.replace('contact:', '')
            elif contact and contact.phone_number:
                phone_number = contact.phone_number
            
            if not phone_number:
                logger.warning("No phone number found for message storage",
                             message_id=message.id,
                             conversation_id=message.conversation_id)
                return None
            
            # Store message
            message_id = await database_service.store_message(
                contact_phone=phone_number,
                conversation_id=message.conversation_id,
                role='user',  # Incoming messages are from user
                content=message.content,
                message_type=message.content_type.value,
                metadata={
                    'chatwoot_message_id': message.id,
                    'sender_name': message.sender.name,
                    'sender_id': message.sender.id,
                    'inbox_id': message.inbox_id,
                    'account_id': message.account_id,
                    'source_id': message.source_id,
                    'content_attributes': message.content_attributes
                },
                sent_at=message.created_at
            )
            
            logger.info("Message stored in database",
                       message_id=message_id,
                       chatwoot_message_id=message.id,
                       conversation_id=message.conversation_id,
                       contact_phone=phone_number)
            
            return message_id
            
        except Exception as e:
            logger.error("Failed to store message in database",
                        message_id=message.id if message else None,
                        conversation_id=message.conversation_id if message else None,
                        error=str(e))
            return None
    
    async def process_webhook(
        self,
        raw_payload: Dict[str, Any],
        signature: Optional[str] = None,
        raw_body: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Process incoming webhook with full validation and error handling.
        
        Args:
            raw_payload: Raw webhook payload dictionary
            signature: HMAC signature for verification
            raw_body: Raw request body for signature verification
            
        Returns:
            Processing result dictionary
            
        Raises:
            WebhookError: Various webhook-related errors
        """
        processing_start = datetime.utcnow()
        
        try:
            # Verify webhook signature if provided
            if raw_body and signature:
                if not self.verify_webhook_signature(raw_body, signature):
                    logger.error("Webhook signature verification failed")
                    raise SecurityError("Invalid webhook signature")
            
            # Parse and validate payload
            webhook_payload = self.parse_webhook_payload(raw_payload)
            
            # Set up logging context
            conversation_id = webhook_payload.conversation.id if webhook_payload.conversation else None
            contact_phone = None
            
            if webhook_payload.message and webhook_payload.message.source_id:
                contact_phone = webhook_payload.message.source_id.replace('contact:', '')
            elif webhook_payload.contact and webhook_payload.contact.phone_number:
                contact_phone = webhook_payload.contact.phone_number
            
            with log_context(
                conversation_id_val=conversation_id,
                contact_phone_val=contact_phone
            ) as ctx:
                
                # Log webhook received
                ctx.info("Webhook received",
                        event=webhook_payload.event,
                        account_id=webhook_payload.account.id,
                        inbox_id=webhook_payload.message.inbox_id if webhook_payload.message else None)
                
                # Check if message should be processed
                should_process, reason = self.should_process_message(webhook_payload)
                
                if not should_process:
                    ctx.info("Webhook message skipped", reason=reason)
                    return {
                        'status': 'skipped',
                        'reason': reason,
                        'event': webhook_payload.event,
                        'processing_time_ms': int((datetime.utcnow() - processing_start).total_seconds() * 1000)
                    }
                
                # Store message in database
                message_id = await self.store_message_in_database(webhook_payload)
                
                # Process message with agent
                agent_response = None
                if self.agent and self.governance:
                    try:
                        # Check if conversation is paused
                        is_paused = await self.governance.is_conversation_paused(conversation_id)
                        
                        # Process message through agent
                        agent_response = await self.agent.process_message(
                            webhook_payload, 
                            governance_paused=is_paused
                        )
                        
                        # Send response if agent generated one and it's not paused
                        if (agent_response.response_content and 
                            agent_response.status == "success" and 
                            not is_paused):
                            
                            # Generate unique echo_id for loop prevention
                            import uuid
                            echo_id = f"agent_{uuid.uuid4().hex[:8]}"
                            
                            await self.chatwoot_client.send_message(
                                account_id=webhook_payload.account.id,
                                conversation_id=conversation_id,
                                content=agent_response.response_content,
                                content_attributes={
                                    "agent_processed": True,
                                    "agent_response_time_ms": agent_response.processing_time_ms,
                                    "tool_used": agent_response.tool_used,
                                    "confidence": agent_response.confidence
                                },
                                echo_id=echo_id
                            )
                            
                            ctx.info("Agent response sent",
                                   response_length=len(agent_response.response_content),
                                   tool_used=agent_response.tool_used,
                                   processing_time_ms=agent_response.processing_time_ms)
                        
                    except Exception as e:
                        ctx.error("Agent processing failed", error=str(e))
                        agent_response = None
                
                ctx.info("Message processing completed",
                        chatwoot_message_id=webhook_payload.message.id,
                        stored_message_id=message_id,
                        agent_processed=agent_response is not None,
                        user_query=webhook_payload.message.content[:100])
                
                processing_time_ms = int((datetime.utcnow() - processing_start).total_seconds() * 1000)
                
                result = {
                    'status': 'processed',
                    'event': webhook_payload.event,
                    'conversation_id': conversation_id,
                    'message_id': webhook_payload.message.id,
                    'stored_message_id': message_id,
                    'contact_phone': contact_phone,
                    'processing_time_ms': processing_time_ms
                }
                
                ctx.log_performance(
                    operation='webhook_processing',
                    duration_ms=processing_time_ms
                )
                
                return result
                
        except (ValidationError, SecurityError) as e:
            logger.error("Webhook validation/security error", 
                        error=str(e),
                        error_type=type(e).__name__)
            raise
        
        except Exception as e:
            logger.error("Unexpected error processing webhook",
                        error=str(e),
                        error_type=type(e).__name__)
            raise ProcessingError(f"Failed to process webhook: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for webhook handler dependencies.
        
        Returns:
            Health check results
        """
        health_data = {
            'status': 'healthy',
            'dependencies': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            await self._ensure_dependencies()
            
            # Check Chatwoot API client
            try:
                chatwoot_health = await self.chatwoot_client.health_check()
                health_data['dependencies']['chatwoot_api'] = {
                    'status': 'healthy' if chatwoot_health['status'] == 'healthy' else 'unhealthy',
                    'details': chatwoot_health
                }
            except Exception as e:
                health_data['dependencies']['chatwoot_api'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
            
            # Check database service
            try:
                database_service = await get_database_service()
                db_health = await database_service.health_check()
                health_data['dependencies']['database'] = {
                    'status': 'healthy' if db_health['status'] == 'healthy' else 'unhealthy',
                    'details': db_health
                }
            except Exception as e:
                health_data['dependencies']['database'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
            
            # Overall status
            all_healthy = all(
                dep['status'] == 'healthy' 
                for dep in health_data['dependencies'].values()
            )
            health_data['status'] = 'healthy' if all_healthy else 'unhealthy'
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            health_data['status'] = 'unhealthy'
            health_data['error'] = str(e)
        
        return health_data
    
    # Utility methods for testing and debugging
    
    def validate_message_format(self, message_data: Dict[str, Any]) -> List[str]:
        """
        Validate message format and return list of issues.
        
        Args:
            message_data: Raw message data
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        required_fields = ['id', 'content', 'message_type', 'content_type', 'sender']
        for field in required_fields:
            if field not in message_data:
                issues.append(f"Missing required field: {field}")
        
        if 'sender' in message_data:
            sender = message_data['sender']
            if not isinstance(sender, dict):
                issues.append("Sender must be an object")
            else:
                sender_required = ['id', 'type']
                for field in sender_required:
                    if field not in sender:
                        issues.append(f"Missing required sender field: {field}")
        
        # Validate enum values
        if 'message_type' in message_data:
            valid_types = [t.value for t in MessageType]
            if message_data['message_type'] not in valid_types:
                issues.append(f"Invalid message_type: {message_data['message_type']}")
        
        if 'content_type' in message_data:
            valid_types = [t.value for t in ContentType]
            if message_data['content_type'] not in valid_types:
                issues.append(f"Invalid content_type: {message_data['content_type']}")
        
        return issues


# Global webhook handler instance
_webhook_handler: Optional[WebhookHandler] = None


def get_webhook_handler() -> WebhookHandler:
    """
    Get the global webhook handler instance.
    
    Returns:
        WebhookHandler instance
    """
    global _webhook_handler
    if _webhook_handler is None:
        _webhook_handler = WebhookHandler()
    return _webhook_handler


# Convenience function for FastAPI/Flask integration
async def handle_chatwoot_webhook(
    payload: Dict[str, Any],
    signature: Optional[str] = None,
    raw_body: Optional[bytes] = None
) -> Dict[str, Any]:
    """
    Convenience function to handle Chatwoot webhook.
    
    Args:
        payload: Webhook payload dictionary
        signature: HMAC signature header
        raw_body: Raw request body for verification
        
    Returns:
        Processing result dictionary
        
    Raises:
        WebhookError: Various webhook-related errors
    """
    handler = get_webhook_handler()
    return await handler.process_webhook(payload, signature, raw_body)