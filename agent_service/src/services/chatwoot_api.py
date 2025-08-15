"""
Chatwoot API client for the Agent Service.

Provides HTTP client with authentication, rate limiting, retry logic,
and comprehensive error handling for Chatwoot API integration.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx
from httpx import AsyncClient, Response

from ..models.schemas import (
    WebhookPayload, Message, Conversation, Contact, 
    MessageType, ContentType, SenderType
)
from ..utils.config import get_chatwoot_config, ChatwootConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ChatwootAPIError(Exception):
    """Base exception for Chatwoot API errors."""
    pass


class AuthenticationError(ChatwootAPIError):
    """Exception for authentication failures."""
    pass


class RateLimitError(ChatwootAPIError):
    """Exception for rate limiting issues."""
    pass


class ValidationError(ChatwootAPIError):
    """Exception for request validation errors."""
    pass


class NetworkError(ChatwootAPIError):
    """Exception for network-related errors."""
    pass


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter for API requests.
    
    Implements token bucket algorithm to respect Chatwoot API rate limits
    (60 requests per minute by default).
    """
    
    max_tokens: int
    refill_rate: float  # tokens per second
    tokens: float
    last_refill: float
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_tokens = max_requests_per_minute
        self.refill_rate = max_requests_per_minute / 60.0  # tokens per second
        self.tokens = float(max_requests_per_minute)
        self.last_refill = time.time()
    
    def can_proceed(self) -> bool:
        """Check if a request can proceed without being rate limited."""
        self._refill_tokens()
        return self.tokens >= 1.0
    
    def consume_token(self) -> bool:
        """
        Consume a token for a request.
        
        Returns:
            True if token was consumed, False if rate limited
        """
        self._refill_tokens()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
    
    def get_wait_time(self) -> float:
        """Get time to wait before next request is allowed."""
        if self.can_proceed():
            return 0.0
        return (1.0 - self.tokens) / self.refill_rate


class ChatwootAPIClient:
    """
    Production-ready Chatwoot API client with comprehensive error handling.
    
    Features:
    - Authentication with API tokens
    - Rate limiting with token bucket algorithm
    - Exponential backoff retry logic
    - Request/response logging
    - Loop prevention with echo_id mechanism
    """
    
    def __init__(self, config: Optional[ChatwootConfig] = None):
        self.config = config or get_chatwoot_config()
        self.rate_limiter = RateLimiter(self.config.rate_limit_per_minute)
        self._client: Optional[AsyncClient] = None
        
        # Request headers
        self.headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Chatwoot-Agent-Service/1.0.0',
            'api_access_token': self.config.api_token
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_client(self) -> None:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_seconds),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                headers=self.headers,
                follow_redirects=True
            )
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> Response:
        """
        Make HTTP request with rate limiting and retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            retry_count: Current retry attempt
            
        Returns:
            HTTP response
            
        Raises:
            Various ChatwootAPIError subclasses based on error type
        """
        await self._ensure_client()
        
        # Rate limiting
        if not self.rate_limiter.consume_token():
            wait_time = self.rate_limiter.get_wait_time()
            logger.warning("Rate limit reached, waiting", wait_time=wait_time)
            await asyncio.sleep(wait_time)
            # Try again after waiting
            if not self.rate_limiter.consume_token():
                raise RateLimitError("Unable to acquire rate limit token after waiting")
        
        # Build full URL
        url = urljoin(self.config.base_url, endpoint.lstrip('/'))
        
        # Log request
        logger.debug("Making API request",
                    method=method,
                    url=url,
                    retry_count=retry_count,
                    has_data=data is not None)
        
        try:
            start_time = time.time()
            
            response = await self._client.request(
                method=method,
                url=url,
                json=data,
                params=params
            )
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Log response
            logger.debug("API request completed",
                        method=method,
                        url=url,
                        status_code=response.status_code,
                        duration_ms=duration_ms,
                        retry_count=retry_count)
            
            # Handle different response codes
            if response.status_code == 200 or response.status_code == 201:
                return response
            elif response.status_code == 401:
                logger.error("Authentication failed", status_code=response.status_code)
                raise AuthenticationError("Invalid API token or authentication failed")
            elif response.status_code == 429:
                logger.warning("Rate limit exceeded", status_code=response.status_code)
                raise RateLimitError("API rate limit exceeded")
            elif response.status_code >= 400 and response.status_code < 500:
                error_msg = f"Client error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = f"{error_msg} - {error_data.get('message', 'Unknown error')}"
                except:
                    pass
                logger.error("API client error", status_code=response.status_code, error=error_msg)
                raise ValidationError(error_msg)
            elif response.status_code >= 500:
                error_msg = f"Server error: {response.status_code}"
                logger.error("API server error", status_code=response.status_code)
                
                # Retry on server errors
                if retry_count < self.config.max_retries:
                    wait_time = self.config.backoff_factor * (2 ** retry_count)
                    logger.info("Retrying request after server error", 
                               retry_count=retry_count + 1,
                               wait_time=wait_time)
                    await asyncio.sleep(wait_time)
                    return await self._make_request(method, endpoint, data, params, retry_count + 1)
                
                raise NetworkError(error_msg)
            else:
                raise ChatwootAPIError(f"Unexpected response code: {response.status_code}")
                
        except httpx.TimeoutException as e:
            logger.error("Request timeout", method=method, url=url, error=str(e))
            
            # Retry on timeout
            if retry_count < self.config.max_retries:
                wait_time = self.config.backoff_factor * (2 ** retry_count)
                logger.info("Retrying request after timeout",
                           retry_count=retry_count + 1,
                           wait_time=wait_time)
                await asyncio.sleep(wait_time)
                return await self._make_request(method, endpoint, data, params, retry_count + 1)
            
            raise NetworkError(f"Request timeout after {self.config.max_retries} retries")
        
        except httpx.ConnectError as e:
            logger.error("Connection error", method=method, url=url, error=str(e))
            
            # Retry on connection error
            if retry_count < self.config.max_retries:
                wait_time = self.config.backoff_factor * (2 ** retry_count)
                logger.info("Retrying request after connection error",
                           retry_count=retry_count + 1,
                           wait_time=wait_time)
                await asyncio.sleep(wait_time)
                return await self._make_request(method, endpoint, data, params, retry_count + 1)
            
            raise NetworkError(f"Connection error after {self.config.max_retries} retries: {e}")
        
        except Exception as e:
            logger.error("Unexpected error during API request", 
                        method=method, url=url, error=str(e), error_type=type(e).__name__)
            raise ChatwootAPIError(f"Unexpected error: {e}")
    
    # Message Operations
    
    async def send_message(
        self,
        account_id: int,
        conversation_id: int,
        content: str,
        message_type: str = "outgoing",
        content_type: str = "text",
        content_attributes: Optional[Dict[str, Any]] = None,
        echo_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a message to a Chatwoot conversation.
        
        Args:
            account_id: Chatwoot account ID
            conversation_id: Conversation ID to send message to
            content: Message content
            message_type: Message type (outgoing, etc.)
            content_type: Content type (text, etc.)
            content_attributes: Additional message attributes
            echo_id: Unique ID to prevent echo loops
            
        Returns:
            Message data from API response
            
        Raises:
            ChatwootAPIError: If message sending fails
        """
        endpoint = f"/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
        
        # Build message payload
        payload = {
            "content": content,
            "message_type": message_type,
            "content_type": content_type,
            "content_attributes": {
                "agent_processed": True,
                **(content_attributes or {})
            }
        }
        
        # Add echo_id for loop prevention
        if echo_id:
            payload["content_attributes"]["echo_id"] = echo_id
        
        try:
            response = await self._make_request("POST", endpoint, data=payload)
            message_data = response.json()
            
            logger.info("Message sent successfully",
                       conversation_id=conversation_id,
                       message_id=message_data.get('id'),
                       content_length=len(content),
                       echo_id=echo_id)
            
            return message_data
            
        except Exception as e:
            logger.error("Failed to send message",
                        account_id=account_id,
                        conversation_id=conversation_id,
                        error=str(e))
            raise
    
    async def get_conversation(
        self,
        account_id: int,
        conversation_id: int
    ) -> Dict[str, Any]:
        """
        Get conversation details from Chatwoot.
        
        Args:
            account_id: Chatwoot account ID
            conversation_id: Conversation ID
            
        Returns:
            Conversation data from API
            
        Raises:
            ChatwootAPIError: If conversation retrieval fails
        """
        endpoint = f"/api/v1/accounts/{account_id}/conversations/{conversation_id}"
        
        try:
            response = await self._make_request("GET", endpoint)
            conversation_data = response.json()
            
            logger.debug("Conversation retrieved successfully",
                        conversation_id=conversation_id,
                        status=conversation_data.get('status'),
                        assignee_id=conversation_data.get('assignee_id'))
            
            return conversation_data
            
        except Exception as e:
            logger.error("Failed to get conversation",
                        account_id=account_id,
                        conversation_id=conversation_id,
                        error=str(e))
            raise
    
    async def get_conversation_messages(
        self,
        account_id: int,
        conversation_id: int,
        before: Optional[int] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get messages from a conversation.
        
        Args:
            account_id: Chatwoot account ID
            conversation_id: Conversation ID
            before: Get messages before this message ID
            limit: Number of messages to retrieve
            
        Returns:
            List of message data from API
            
        Raises:
            ChatwootAPIError: If message retrieval fails
        """
        endpoint = f"/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
        
        params = {"limit": limit}
        if before:
            params["before"] = before
        
        try:
            response = await self._make_request("GET", endpoint, params=params)
            messages = response.json()
            
            logger.debug("Conversation messages retrieved",
                        conversation_id=conversation_id,
                        message_count=len(messages),
                        limit=limit)
            
            return messages
            
        except Exception as e:
            logger.error("Failed to get conversation messages",
                        account_id=account_id,
                        conversation_id=conversation_id,
                        error=str(e))
            raise
    
    async def update_conversation(
        self,
        account_id: int,
        conversation_id: int,
        status: Optional[str] = None,
        assignee_id: Optional[int] = None,
        labels: Optional[List[str]] = None,
        custom_attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update conversation properties.
        
        Args:
            account_id: Chatwoot account ID
            conversation_id: Conversation ID
            status: New conversation status
            assignee_id: New assignee ID
            labels: New labels list
            custom_attributes: Custom attributes to update
            
        Returns:
            Updated conversation data
            
        Raises:
            ChatwootAPIError: If conversation update fails
        """
        endpoint = f"/api/v1/accounts/{account_id}/conversations/{conversation_id}"
        
        payload = {}
        if status:
            payload["status"] = status
        if assignee_id is not None:
            payload["assignee_id"] = assignee_id
        if labels is not None:
            payload["labels"] = labels
        if custom_attributes:
            payload["custom_attributes"] = custom_attributes
        
        if not payload:
            raise ValidationError("No update parameters provided")
        
        try:
            response = await self._make_request("PATCH", endpoint, data=payload)
            conversation_data = response.json()
            
            logger.info("Conversation updated successfully",
                       conversation_id=conversation_id,
                       updates=list(payload.keys()))
            
            return conversation_data
            
        except Exception as e:
            logger.error("Failed to update conversation",
                        account_id=account_id,
                        conversation_id=conversation_id,
                        error=str(e))
            raise
    
    # Contact Operations
    
    async def get_contact(
        self,
        account_id: int,
        contact_id: int
    ) -> Dict[str, Any]:
        """
        Get contact details from Chatwoot.
        
        Args:
            account_id: Chatwoot account ID
            contact_id: Contact ID
            
        Returns:
            Contact data from API
            
        Raises:
            ChatwootAPIError: If contact retrieval fails
        """
        endpoint = f"/api/v1/accounts/{account_id}/contacts/{contact_id}"
        
        try:
            response = await self._make_request("GET", endpoint)
            contact_data = response.json()
            
            logger.debug("Contact retrieved successfully",
                        contact_id=contact_id,
                        name=contact_data.get('name'),
                        phone=contact_data.get('phone_number'))
            
            return contact_data
            
        except Exception as e:
            logger.error("Failed to get contact",
                        account_id=account_id,
                        contact_id=contact_id,
                        error=str(e))
            raise
    
    # Utility Methods
    
    def is_agent_message(self, message_data: Dict[str, Any]) -> bool:
        """
        Check if a message was sent by this agent (loop prevention).
        
        Args:
            message_data: Message data from webhook or API
            
        Returns:
            True if message was sent by agent, False otherwise
        """
        content_attributes = message_data.get('content_attributes', {})
        
        # Check for agent_processed flag
        if content_attributes.get('agent_processed'):
            return True
        
        # Check for echo_id (our messages should have this)
        if content_attributes.get('echo_id'):
            return True
        
        # Check message type and sender type
        message_type = message_data.get('message_type')
        sender = message_data.get('sender', {})
        sender_type = sender.get('type')
        
        # Agent messages are typically outgoing from agent_bot or user
        if message_type == 'outgoing' and sender_type in ['agent_bot', 'user']:
            return True
        
        return False
    
    def should_process_message(self, webhook_payload: WebhookPayload) -> bool:
        """
        Determine if a webhook message should be processed by the agent.
        
        Args:
            webhook_payload: Validated webhook payload
            
        Returns:
            True if message should be processed, False otherwise
        """
        # Only process message_created events
        if webhook_payload.event != "message_created":
            return False
        
        # Must have message data
        if not webhook_payload.message:
            return False
        
        message = webhook_payload.message
        
        # Only process incoming text messages
        if message.message_type != MessageType.INCOMING:
            return False
        
        if message.content_type != ContentType.TEXT:
            return False
        
        # Skip empty messages
        if not message.content or not message.content.strip():
            return False
        
        # Skip agent messages (loop prevention)
        if self.is_agent_message(message.dict()):
            logger.debug("Skipping agent message to prevent loop",
                        message_id=message.id,
                        conversation_id=message.conversation_id)
            return False
        
        # Only process messages from contacts
        if message.sender.type != SenderType.CONTACT:
            return False
        
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check against Chatwoot API.
        
        Returns:
            Health check results
        """
        try:
            await self._ensure_client()
            
            # Test basic connectivity with a simple endpoint
            # Note: Using a generic endpoint that should work with most API tokens
            start_time = time.time()
            response = await self._make_request("GET", "/api/v1/profile")
            duration_ms = int((time.time() - start_time) * 1000)
            
            profile_data = response.json()
            
            health_data = {
                'status': 'healthy',
                'api_accessible': True,
                'authentication': 'valid',
                'response_time_ms': duration_ms,
                'rate_limit_available': self.rate_limiter.can_proceed(),
                'base_url': self.config.base_url,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info("Chatwoot API health check passed", **health_data)
            return health_data
            
        except AuthenticationError as e:
            health_data = {
                'status': 'unhealthy',
                'api_accessible': False,
                'authentication': 'invalid',
                'error': str(e),
                'base_url': self.config.base_url,
                'timestamp': datetime.utcnow().isoformat()
            }
            logger.error("Chatwoot API health check failed - authentication", **health_data)
            return health_data
            
        except (NetworkError, RateLimitError) as e:
            health_data = {
                'status': 'unhealthy',
                'api_accessible': False,
                'authentication': 'unknown',
                'error': str(e),
                'base_url': self.config.base_url,
                'timestamp': datetime.utcnow().isoformat()
            }
            logger.error("Chatwoot API health check failed - network", **health_data)
            return health_data
            
        except Exception as e:
            health_data = {
                'status': 'unhealthy',
                'api_accessible': False,
                'authentication': 'unknown',
                'error': f"Unexpected error: {e}",
                'base_url': self.config.base_url,
                'timestamp': datetime.utcnow().isoformat()
            }
            logger.error("Chatwoot API health check failed - unexpected error", **health_data)
            return health_data


# Global API client instance
_api_client: Optional[ChatwootAPIClient] = None


async def get_chatwoot_client() -> ChatwootAPIClient:
    """
    Get the global Chatwoot API client instance.
    
    Returns:
        Initialized ChatwootAPIClient instance
    """
    global _api_client
    if _api_client is None:
        _api_client = ChatwootAPIClient()
    return _api_client


async def close_chatwoot_client() -> None:
    """Close the global Chatwoot API client."""
    global _api_client
    if _api_client:
        await _api_client.close()
        _api_client = None