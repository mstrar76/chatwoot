#!/usr/bin/env python3
"""
Test script for Chatwoot Agent MVP tools.

This script tests the RAG and Spreadsheet tools to ensure they work correctly
with the LangChain integration and follow production-ready patterns.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tools import (
    retrieve_relevant_context,
    query_spreadsheet_data,
    list_available_spreadsheets,
    get_rag_performance_stats,
    get_spreadsheet_performance_stats
)


async def test_rag_tools():
    """Test RAG tools functionality."""
    print("🧠 Testing RAG Tools...")
    
    try:
        # Test RAG performance stats (should work without database)
        print("\n📊 Testing RAG performance stats...")
        stats = await get_rag_performance_stats()
        print(f"RAG Stats: {stats[:200]}...")
        
        # Test RAG context retrieval (will fail without database but should handle gracefully)
        print("\n🔍 Testing RAG context retrieval...")
        context = await retrieve_relevant_context(
            query="test query",
            contact_phone="+5511999999999",
            top_k=3
        )
        print(f"RAG Context Result: {context[:200]}...")
        
        print("✅ RAG tools tested successfully (graceful error handling)")
        
    except Exception as e:
        print(f"❌ RAG tools test failed: {e}")
        return False
    
    return True


async def test_spreadsheet_tools():
    """Test Spreadsheet tools functionality."""
    print("\n📊 Testing Spreadsheet Tools...")
    
    try:
        # Test listing available spreadsheets
        print("\n📋 Testing list available spreadsheets...")
        sheets_info = await list_available_spreadsheets()
        print(f"Sheets Info: {sheets_info[:300]}...")
        
        # Test spreadsheet performance stats
        print("\n📈 Testing spreadsheet performance stats...")
        stats = await get_spreadsheet_performance_stats()
        print(f"Spreadsheet Stats: {stats[:200]}...")
        
        # Test spreadsheet query (will return no data message if no CSV files)
        print("\n🔍 Testing spreadsheet query...")
        query_result = await query_spreadsheet_data(
            query="show me all data",
            limit=5
        )
        print(f"Query Result: {query_result[:300]}...")
        
        print("✅ Spreadsheet tools tested successfully")
        
    except Exception as e:
        print(f"❌ Spreadsheet tools test failed: {e}")
        return False
    
    return True


async def test_tool_schemas():
    """Test that tools have proper LangChain schemas."""
    print("\n🔧 Testing Tool Schemas...")
    
    tools_to_test = [
        retrieve_relevant_context,
        query_spreadsheet_data,
        list_available_spreadsheets,
        get_rag_performance_stats,
        get_spreadsheet_performance_stats
    ]
    
    try:
        for tool in tools_to_test:
            # Check that tool has required attributes
            assert hasattr(tool, 'name'), f"Tool {tool} missing 'name' attribute"
            assert hasattr(tool, 'description'), f"Tool {tool} missing 'description' attribute"
            assert hasattr(tool, 'args_schema'), f"Tool {tool} missing 'args_schema' attribute"
            
            print(f"✅ {tool.name}: {tool.description[:100]}...")
        
        print("✅ All tools have proper LangChain schemas")
        return True
        
    except Exception as e:
        print(f"❌ Tool schema test failed: {e}")
        return False


async def create_sample_data():
    """Create sample CSV data for testing."""
    print("\n📝 Creating sample data for testing...")
    
    try:
        # Create data directory
        data_dir = Path(__file__).parent / "data" / "sheets"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample service orders CSV
        service_orders_file = data_dir / "service_orders.csv"
        sample_orders = """order_id,customer_name,service_type,amount,status,date
1001,João Silva,Manutenção,250.00,completed,2024-01-15
1002,Maria Santos,Instalação,450.00,in_progress,2024-01-16
1003,Pedro Costa,Reparo,180.00,completed,2024-01-17
1004,Ana Oliveira,Manutenção,320.00,pending,2024-01-18
1005,Carlos Lima,Instalação,600.00,completed,2024-01-19"""
        
        with open(service_orders_file, 'w', encoding='utf-8') as f:
            f.write(sample_orders)
        
        # Create sample appointments CSV
        appointments_file = data_dir / "appointments.csv"
        sample_appointments = """appointment_id,customer_phone,technician,service_date,time_slot,status
A001,+5511999999999,Tech1,2024-01-20,09:00-12:00,scheduled
A002,+5511888888888,Tech2,2024-01-20,14:00-17:00,completed
A003,+5511777777777,Tech1,2024-01-21,09:00-12:00,scheduled
A004,+5511666666666,Tech3,2024-01-21,14:00-17:00,cancelled
A005,+5511555555555,Tech2,2024-01-22,09:00-12:00,scheduled"""
        
        with open(appointments_file, 'w', encoding='utf-8') as f:
            f.write(sample_appointments)
        
        print(f"✅ Created sample data in {data_dir}")
        print(f"   - {service_orders_file.name}")
        print(f"   - {appointments_file.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False


async def test_with_sample_data():
    """Test tools with actual sample data."""
    print("\n🧪 Testing with sample data...")
    
    try:
        # Test listing spreadsheets with data
        print("\n📋 Testing spreadsheet listing with sample data...")
        sheets_info = await list_available_spreadsheets()
        print(f"Sheets with data: {sheets_info}")
        
        # Test querying the service orders
        print("\n🔍 Testing service orders query...")
        orders_query = await query_spreadsheet_data(
            query="show me all completed orders",
            sheet_name="service_orders",
            limit=10
        )
        print(f"Orders Query Result:\n{orders_query}")
        
        # Test querying appointments
        print("\n📅 Testing appointments query...")
        appointments_query = await query_spreadsheet_data(
            query="count appointments by status",
            sheet_name="appointments",
            format_output="summary"
        )
        print(f"Appointments Summary:\n{appointments_query}")
        
        # Test aggregation query
        print("\n💰 Testing aggregation query...")
        revenue_query = await query_spreadsheet_data(
            query="sum of amount for completed orders",
            sheet_name="service_orders"
        )
        print(f"Revenue Query:\n{revenue_query}")
        
        print("✅ Sample data tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Sample data test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Chatwoot Agent MVP Tools Test\n")
    
    results = []
    
    # Test tool schemas first
    results.append(await test_tool_schemas())
    
    # Test RAG tools
    results.append(await test_rag_tools())
    
    # Test spreadsheet tools (basic)
    results.append(await test_spreadsheet_tools())
    
    # Create sample data
    if await create_sample_data():
        # Test with sample data
        results.append(await test_with_sample_data())
    
    # Summary
    print(f"\n{'='*60}")
    print("🏁 Test Summary:")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✅ All tests passed! Tools are ready for production.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))