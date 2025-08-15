"""
Chatwoot Agent Service - Main FastAPI Application

This is the main entry point for the Chatwoot Agent Service that provides
intelligent conversational AI capabilities integrated with Chatwoot.

Features:
- Webhook handling for Chatwoot events
- RAG (Retrieval Augmented Generation) with pgvector
- In-memory spreadsheet tool for structured data queries
- Governance controls (HITL, pre-send confirmation)
- Multi-modal support (text, images, audio)
"""

import asyncio
import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .handlers.webhook import WebhookHandler
from .services.chatwoot_api import ChatwootAPIClient
from .agent import ChatwootAgent, get_agent
from .models.schemas import WebhookPayload, AgentConfig, HealthCheck
from .utils.logging import setup_logging, get_logger
from .services.governance import GovernanceService, get_governance_service

# Setup structured logging
setup_logging()
logger = get_logger(__name__)


class AgentService:
    """Main agent service class that orchestrates all components."""
    
    def __init__(self):
        self.agent: ChatwootAgent = None
        self.webhook_handler: WebhookHandler = None
        self.chatwoot_client: ChatwootAPIClient = None
        self.governance: GovernanceService = None
        
    async def initialize(self):
        """Initialize all service components."""
        try:
            logger.info("Initializing Chatwoot Agent Service...")
            
            # Load configuration
            config_path = os.getenv("AGENT_CONFIG_PATH", "/app/config/agent_config.json")
            
            # Initialize Chatwoot API client
            chatwoot_base_url = os.getenv("CHATWOOT_API_URL", "http://omnineural_chatwoot:3000")
            chatwoot_token = os.getenv("CHATWOOT_API_TOKEN")
            
            if not chatwoot_token:
                raise ValueError("CHATWOOT_API_TOKEN environment variable is required")
                
            self.chatwoot_client = ChatwootAPIClient(chatwoot_base_url, chatwoot_token)
            
            # Initialize governance service
            self.governance = get_governance_service()
            
            # Initialize agent using global instance
            self.agent = await get_agent()
            
            # Initialize webhook handler
            self.webhook_handler = WebhookHandler(
                agent=self.agent,
                chatwoot_client=self.chatwoot_client,
                governance=self.governance
            )
            
            logger.info("Agent service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent service: {e}")
            raise


# Global service instance
service = AgentService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    await service.initialize()
    yield
    # Shutdown
    logger.info("Shutting down agent service...")


# Create FastAPI application
app = FastAPI(
    title="Chatwoot Agent Service",
    description="Intelligent conversational AI service for Chatwoot",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your security requirements
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint for container health monitoring."""
    try:
        # Check if all services are healthy
        services_status = {
            "agent": service.agent is not None and await service.agent.is_healthy(),
            "chatwoot_client": service.chatwoot_client is not None,
            "webhook_handler": service.webhook_handler is not None,
            "governance": service.governance is not None,
        }
        
        all_healthy = all(services_status.values())
        
        return HealthCheck(
            status="healthy" if all_healthy else "unhealthy",
            services=services_status,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            services={},
            version="1.0.0",
            error=str(e)
        )


# Main webhook endpoint
@app.post("/webhook")
async def handle_webhook(payload: WebhookPayload, request: Request):
    """
    Handle incoming webhooks from Chatwoot.
    
    This endpoint receives message_created events and other webhook events
    from Chatwoot and processes them through the agent system.
    """
    try:
        logger.info(f"Received webhook event: {payload.event}")
        
        # Validate webhook signature if configured
        # TODO: Implement webhook signature validation for security
        
        # Process webhook through handler
        result = await service.webhook_handler.handle_webhook(payload)
        
        return JSONResponse(
            status_code=200,
            content={"status": "processed", "result": result}
        )
        
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")


# Agent control endpoints
@app.post("/agent/pause/{conversation_id}")
async def pause_agent(conversation_id: int):
    """Pause agent for a specific conversation (HITL control)."""
    try:
        await service.governance.pause_conversation(conversation_id)
        return {"status": "paused", "conversation_id": conversation_id}
    except Exception as e:
        logger.error(f"Failed to pause agent for conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/resume/{conversation_id}")
async def resume_agent(conversation_id: int):
    """Resume agent for a specific conversation (HITL control)."""
    try:
        await service.governance.resume_conversation(conversation_id)
        return {"status": "resumed", "conversation_id": conversation_id}
    except Exception as e:
        logger.error(f"Failed to resume agent for conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/confirm/{conversation_id}")
async def confirm_response(conversation_id: int, response: Dict[str, Any]):
    """Confirm a pending response (pre-send confirmation)."""
    try:
        result = await service.governance.confirm_response(conversation_id, response)
        return {"status": "confirmed", "result": result}
    except Exception as e:
        logger.error(f"Failed to confirm response for conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration endpoints
@app.get("/config/global")
async def get_global_config():
    """Get global agent configuration."""
    try:
        config = await service.agent.get_global_config()
        return config
    except Exception as e:
        logger.error(f"Failed to get global config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/inbox/{inbox_id}")
async def get_inbox_config(inbox_id: int):
    """Get inbox-specific agent configuration."""
    try:
        config = await service.agent.get_inbox_config(inbox_id)
        return config
    except Exception as e:
        logger.error(f"Failed to get inbox config for inbox {inbox_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Metrics and observability endpoints
@app.get("/metrics")
async def get_metrics():
    """Get agent performance metrics."""
    try:
        metrics = await service.agent.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/{conversation_id}")
async def get_conversation_logs(conversation_id: int, limit: int = 50):
    """Get agent logs for a specific conversation."""
    try:
        logs = await service.agent.get_conversation_logs(conversation_id, limit)
        return logs
    except Exception as e:
        logger.error(f"Failed to get logs for conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured logging."""
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with structured logging."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    import uvicorn
    
    # For development only
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8082,
        reload=True,
        log_level="info"
    )