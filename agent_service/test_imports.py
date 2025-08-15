#!/usr/bin/env python3
"""
Simple test script to verify all imports are working correctly.
"""

import sys
import os

# Change working directory to src
src_dir = os.path.join(os.path.dirname(__file__), 'src')
os.chdir(src_dir)
sys.path.insert(0, src_dir)

def test_imports():
    """Test that all major components can be imported."""
    try:
        print("Testing agent imports...")
        from agent import ChatwootAgent, get_agent
        print("✓ Agent imports successful")
        
        print("Testing governance imports...")
        from services.governance import GovernanceService, get_governance_service
        print("✓ Governance imports successful")
        
        print("Testing webhook handler imports...")
        from handlers.webhook import WebhookHandler, get_webhook_handler
        print("✓ Webhook handler imports successful")
        
        print("Testing main service imports...")
        from main import AgentService
        print("✓ Main service imports successful")
        
        print("Testing configuration imports...")
        from utils.config import get_config, get_chatwoot_config, get_openai_config
        print("✓ Configuration imports successful")
        
        print("Testing model schema imports...")
        from models.schemas import AgentConfig, WebhookPayload, AgentResponse
        print("✓ Model schema imports successful")
        
        print("Testing tool imports...")
        from tools.rag import retrieve_relevant_context, store_conversation_context
        from tools.spreadsheet import query_spreadsheet_data, list_available_spreadsheets
        print("✓ Tool imports successful")
        
        print("Testing API client imports...")
        from services.chatwoot_api import ChatwootAPIClient, get_chatwoot_client
        print("✓ API client imports successful")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ All dependencies are properly configured!")
        sys.exit(0)
    else:
        print("\n❌ Some imports failed. Please check dependencies.")
        sys.exit(1)