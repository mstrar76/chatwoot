"""
LangChain tools for the Chatwoot Agent MVP.

This module provides production-ready tools for the agent system including:
- RAG (Retrieval Augmented Generation) tool for conversation context
- Spreadsheet tool for CSV data querying and analysis

All tools follow LangChain patterns with @tool decorators and async operations.
"""

from .rag import (
    retrieve_relevant_context,
    store_conversation_context,
    get_rag_performance_stats,
    get_rag_service,
    RAGService,
    RAGResult,
    RAGQuery
)

from .spreadsheet import (
    query_spreadsheet_data,
    list_available_spreadsheets,
    get_spreadsheet_performance_stats,
    get_spreadsheet_service,
    SpreadsheetService,
    SpreadsheetResult,
    SpreadsheetQuery
)

# Export all tools for easy importing
__all__ = [
    # RAG tools
    "retrieve_relevant_context",
    "store_conversation_context", 
    "get_rag_performance_stats",
    "get_rag_service",
    "RAGService",
    "RAGResult",
    "RAGQuery",
    
    # Spreadsheet tools
    "query_spreadsheet_data",
    "list_available_spreadsheets",
    "get_spreadsheet_performance_stats",
    "get_spreadsheet_service",
    "SpreadsheetService",
    "SpreadsheetResult",
    "SpreadsheetQuery",
]

# Tool lists for different use cases
ALL_TOOLS = [
    retrieve_relevant_context,
    store_conversation_context,
    get_rag_performance_stats,
    query_spreadsheet_data,
    list_available_spreadsheets,
    get_spreadsheet_performance_stats,
]

CORE_TOOLS = [
    retrieve_relevant_context,
    query_spreadsheet_data,
]

ADMIN_TOOLS = [
    get_rag_performance_stats,
    get_spreadsheet_performance_stats,
    list_available_spreadsheets,
]