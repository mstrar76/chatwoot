#!/usr/bin/env python3
"""
Simple test script to verify core LangChain imports are working correctly.
"""

def test_core_imports():
    """Test that core LangChain components can be imported."""
    try:
        print("Testing LangChain core imports...")
        from langchain_core.tools import BaseTool
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        from langchain_core.runnables import RunnableConfig
        from langchain_core.callbacks import AsyncCallbackHandler
        print("✓ LangChain core imports successful")
        
        print("Testing LangChain OpenAI imports...")
        from langchain_openai import ChatOpenAI
        print("✓ LangChain OpenAI imports successful")
        
        print("Testing LangChain agents imports...")
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        print("✓ LangChain agents imports successful")
        
        print("Testing Pydantic imports...")
        from pydantic import BaseModel, Field, validator
        print("✓ Pydantic imports successful")
        
        print("Testing async libraries...")
        import asyncio
        import time
        from typing import Dict, List, Any, Optional, Union, Tuple
        from datetime import datetime, timedelta
        print("✓ Standard library imports successful")
        
        print("\n🎉 All core imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_core_imports()
    if success:
        print("\n✅ Core dependencies are properly configured!")
        print("✅ The LangChain agent implementation should work correctly!")
    else:
        print("\n❌ Some core imports failed. Please check dependencies.")