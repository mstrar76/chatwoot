"""
Governance Service for Chatwoot Agent MVP.

Provides Human-in-the-Loop (HITL) controls including:
- Pause/resume agent for specific conversations
- Pre-send confirmation for responses
- Response approval workflows
- Agent control and monitoring
"""

import asyncio
from typing import Dict, Set, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationState:
    """State tracking for individual conversations."""
    conversation_id: int
    paused: bool = False
    paused_at: Optional[datetime] = None
    paused_by: Optional[str] = None
    pending_responses: Dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False


class GovernanceService:
    """
    Governance service for agent control and oversight.
    
    Provides HITL controls including pause/resume functionality,
    response confirmation, and conversation monitoring.
    """
    
    def __init__(self):
        """Initialize the governance service."""
        self._conversation_states: Dict[int, ConversationState] = {}
        self._globally_paused: bool = False
        self._paused_conversations: Set[int] = set()
        self._pending_confirmations: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Governance service initialized")
    
    async def pause_conversation(self, conversation_id: int, paused_by: str = "admin") -> bool:
        """
        Pause agent processing for a specific conversation.
        
        Args:
            conversation_id: Conversation to pause
            paused_by: Who requested the pause
            
        Returns:
            True if paused successfully
        """
        try:
            if conversation_id not in self._conversation_states:
                self._conversation_states[conversation_id] = ConversationState(
                    conversation_id=conversation_id
                )
            
            state = self._conversation_states[conversation_id]
            state.paused = True
            state.paused_at = datetime.utcnow()
            state.paused_by = paused_by
            
            self._paused_conversations.add(conversation_id)
            
            logger.info("Conversation paused",
                       conversation_id=conversation_id,
                       paused_by=paused_by)
            
            return True
            
        except Exception as e:
            logger.error("Failed to pause conversation",
                        conversation_id=conversation_id,
                        error=str(e))
            return False
    
    async def resume_conversation(self, conversation_id: int) -> bool:
        """
        Resume agent processing for a specific conversation.
        
        Args:
            conversation_id: Conversation to resume
            
        Returns:
            True if resumed successfully
        """
        try:
            if conversation_id in self._conversation_states:
                state = self._conversation_states[conversation_id]
                state.paused = False
                state.paused_at = None
                state.paused_by = None
            
            self._paused_conversations.discard(conversation_id)
            
            logger.info("Conversation resumed", conversation_id=conversation_id)
            
            return True
            
        except Exception as e:
            logger.error("Failed to resume conversation",
                        conversation_id=conversation_id,
                        error=str(e))
            return False
    
    async def is_conversation_paused(self, conversation_id: int) -> bool:
        """
        Check if a conversation is paused.
        
        Args:
            conversation_id: Conversation to check
            
        Returns:
            True if conversation is paused
        """
        # Check global pause
        if self._globally_paused:
            return True
        
        # Check conversation-specific pause
        return conversation_id in self._paused_conversations
    
    async def pause_all_conversations(self, paused_by: str = "admin") -> bool:
        """
        Pause agent processing for all conversations.
        
        Args:
            paused_by: Who requested the global pause
            
        Returns:
            True if paused successfully
        """
        try:
            self._globally_paused = True
            
            logger.warning("All conversations paused globally", paused_by=paused_by)
            
            return True
            
        except Exception as e:
            logger.error("Failed to pause all conversations", error=str(e))
            return False
    
    async def resume_all_conversations(self) -> bool:
        """
        Resume agent processing for all conversations.
        
        Returns:
            True if resumed successfully
        """
        try:
            self._globally_paused = False
            
            logger.info("All conversations resumed globally")
            
            return True
            
        except Exception as e:
            logger.error("Failed to resume all conversations", error=str(e))
            return False
    
    async def request_response_confirmation(
        self,
        conversation_id: int,
        response_content: str,
        confidence: float = 0.0
    ) -> str:
        """
        Request confirmation for a response before sending.
        
        Args:
            conversation_id: Conversation ID
            response_content: Proposed response content
            confidence: Confidence score for the response
            
        Returns:
            Confirmation ID for tracking
        """
        try:
            confirmation_id = f"conf_{conversation_id}_{int(datetime.utcnow().timestamp())}"
            
            confirmation_data = {
                "conversation_id": conversation_id,
                "response_content": response_content,
                "confidence": confidence,
                "created_at": datetime.utcnow(),
                "status": "pending"
            }
            
            self._pending_confirmations[confirmation_id] = confirmation_data
            
            # Mark conversation as requiring confirmation
            if conversation_id not in self._conversation_states:
                self._conversation_states[conversation_id] = ConversationState(
                    conversation_id=conversation_id
                )
            
            state = self._conversation_states[conversation_id]
            state.requires_confirmation = True
            state.pending_responses[confirmation_id] = confirmation_data
            
            logger.info("Response confirmation requested",
                       conversation_id=conversation_id,
                       confirmation_id=confirmation_id,
                       confidence=confidence)
            
            return confirmation_id
            
        except Exception as e:
            logger.error("Failed to request response confirmation",
                        conversation_id=conversation_id,
                        error=str(e))
            raise
    
    async def confirm_response(
        self,
        conversation_id: int,
        confirmation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Confirm a pending response for sending.
        
        Args:
            conversation_id: Conversation ID
            confirmation_data: Confirmation details from the API
            
        Returns:
            Result of the confirmation
        """
        try:
            confirmation_id = confirmation_data.get("confirmation_id")
            action = confirmation_data.get("action", "approve")  # approve, reject, modify
            
            if confirmation_id not in self._pending_confirmations:
                raise ValueError(f"Confirmation {confirmation_id} not found")
            
            pending = self._pending_confirmations[confirmation_id]
            
            if action == "approve":
                pending["status"] = "approved"
                pending["approved_at"] = datetime.utcnow()
                
                result = {
                    "status": "approved",
                    "conversation_id": conversation_id,
                    "response_content": pending["response_content"],
                    "should_send": True
                }
                
                logger.info("Response approved",
                           conversation_id=conversation_id,
                           confirmation_id=confirmation_id)
                
            elif action == "reject":
                pending["status"] = "rejected"
                pending["rejected_at"] = datetime.utcnow()
                
                result = {
                    "status": "rejected",
                    "conversation_id": conversation_id,
                    "should_send": False,
                    "reason": confirmation_data.get("reason", "Response rejected")
                }
                
                logger.info("Response rejected",
                           conversation_id=conversation_id,
                           confirmation_id=confirmation_id)
                
            elif action == "modify":
                modified_content = confirmation_data.get("modified_content")
                if not modified_content:
                    raise ValueError("Modified content is required for modify action")
                
                pending["status"] = "modified"
                pending["modified_at"] = datetime.utcnow()
                pending["modified_content"] = modified_content
                
                result = {
                    "status": "modified",
                    "conversation_id": conversation_id,
                    "response_content": modified_content,
                    "should_send": True
                }
                
                logger.info("Response modified",
                           conversation_id=conversation_id,
                           confirmation_id=confirmation_id)
                
            else:
                raise ValueError(f"Invalid action: {action}")
            
            # Clean up pending confirmations
            del self._pending_confirmations[confirmation_id]
            
            # Update conversation state
            if conversation_id in self._conversation_states:
                state = self._conversation_states[conversation_id]
                if confirmation_id in state.pending_responses:
                    del state.pending_responses[confirmation_id]
                
                # Check if there are more pending responses
                state.requires_confirmation = len(state.pending_responses) > 0
            
            return result
            
        except Exception as e:
            logger.error("Failed to confirm response",
                        conversation_id=conversation_id,
                        error=str(e))
            raise
    
    async def get_conversation_state(self, conversation_id: int) -> Dict[str, Any]:
        """
        Get the current state of a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Dictionary with conversation state information
        """
        try:
            state = self._conversation_states.get(conversation_id)
            
            if not state:
                return {
                    "conversation_id": conversation_id,
                    "paused": conversation_id in self._paused_conversations,
                    "globally_paused": self._globally_paused,
                    "requires_confirmation": False,
                    "pending_responses": 0
                }
            
            return {
                "conversation_id": conversation_id,
                "paused": state.paused,
                "globally_paused": self._globally_paused,
                "paused_at": state.paused_at.isoformat() if state.paused_at else None,
                "paused_by": state.paused_by,
                "requires_confirmation": state.requires_confirmation,
                "pending_responses": len(state.pending_responses)
            }
            
        except Exception as e:
            logger.error("Failed to get conversation state",
                        conversation_id=conversation_id,
                        error=str(e))
            return {"error": str(e)}
    
    async def get_pending_confirmations(self) -> Dict[str, Any]:
        """
        Get all pending response confirmations.
        
        Returns:
            Dictionary with pending confirmations
        """
        try:
            pending = {}
            
            for conf_id, conf_data in self._pending_confirmations.items():
                pending[conf_id] = {
                    "confirmation_id": conf_id,
                    "conversation_id": conf_data["conversation_id"],
                    "response_content": conf_data["response_content"][:200] + "..." 
                                      if len(conf_data["response_content"]) > 200 
                                      else conf_data["response_content"],
                    "confidence": conf_data["confidence"],
                    "created_at": conf_data["created_at"].isoformat(),
                    "status": conf_data["status"]
                }
            
            return {
                "total_pending": len(pending),
                "confirmations": pending
            }
            
        except Exception as e:
            logger.error("Failed to get pending confirmations", error=str(e))
            return {"error": str(e)}
    
    async def cleanup_old_states(self, max_age_hours: int = 24) -> int:
        """
        Clean up old conversation states and confirmations.
        
        Args:
            max_age_hours: Maximum age for states to keep
            
        Returns:
            Number of items cleaned up
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            cleaned_count = 0
            
            # Clean up old confirmations
            expired_confirmations = [
                conf_id for conf_id, conf_data in self._pending_confirmations.items()
                if conf_data["created_at"] < cutoff_time
            ]
            
            for conf_id in expired_confirmations:
                del self._pending_confirmations[conf_id]
                cleaned_count += 1
            
            # Clean up old conversation states
            expired_states = [
                conv_id for conv_id, state in self._conversation_states.items()
                if state.paused_at and state.paused_at < cutoff_time
                and not state.paused  # Don't clean up currently paused conversations
            ]
            
            for conv_id in expired_states:
                del self._conversation_states[conv_id]
                cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info("Cleaned up old governance states", cleaned_count=cleaned_count)
            
            return cleaned_count
            
        except Exception as e:
            logger.error("Failed to cleanup old states", error=str(e))
            return 0
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get governance service statistics.
        
        Returns:
            Dictionary with service statistics
        """
        try:
            total_conversations = len(self._conversation_states)
            paused_conversations = len(self._paused_conversations)
            pending_confirmations = len(self._pending_confirmations)
            
            conversations_requiring_confirmation = sum(
                1 for state in self._conversation_states.values()
                if state.requires_confirmation
            )
            
            return {
                "globally_paused": self._globally_paused,
                "total_conversations_tracked": total_conversations,
                "paused_conversations": paused_conversations,
                "pending_confirmations": pending_confirmations,
                "conversations_requiring_confirmation": conversations_requiring_confirmation,
                "service_status": "healthy"
            }
            
        except Exception as e:
            logger.error("Failed to get governance statistics", error=str(e))
            return {"error": str(e)}


# Global governance service instance
_governance_service: Optional[GovernanceService] = None


def get_governance_service() -> GovernanceService:
    """
    Get the global governance service instance.
    
    Returns:
        GovernanceService instance
    """
    global _governance_service
    if _governance_service is None:
        _governance_service = GovernanceService()
    return _governance_service