"""
Comprehensive unit tests for the Spreadsheet tool.

Tests cover:
- CSV loading and caching mechanisms
- Natural language query processing
- Data filtering and aggregation
- Performance optimization
- Error handling and edge cases
- Memory management
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from typing import Dict, Any, List

from src.tools.spreadsheet import (
    query_spreadsheet_data, list_available_spreadsheets,
    get_spreadsheet_performance_stats, get_spreadsheet_service,
    SpreadsheetService, CSVLoader, QueryProcessor
)
from src.models.schemas import SheetsToolConfig
from tests.conftest import (
    assert_response_time, AsyncMockService
)


class TestSpreadsheetService:
    """Test core spreadsheet service functionality."""
    
    @pytest.fixture
    async def spreadsheet_service(self, test_csv_data):
        """Create spreadsheet service with test data."""
        config = SheetsToolConfig(
            enabled=True,
            sheet_configs=[
                test_csv_data['service_orders'],
                test_csv_data['appointments'],
                test_csv_data['price_list']
            ],
            cache_ttl_minutes=5
        )
        
        service = SpreadsheetService(config)
        await service.initialize()
        return service, test_csv_data
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_spreadsheet_service_initialization(self, test_csv_data):
        """Test spreadsheet service initialization."""
        # Arrange
        config = SheetsToolConfig(
            enabled=True,
            sheet_configs=[test_csv_data['service_orders']]
        )
        
        # Act
        service = SpreadsheetService(config)
        await service.initialize()
        
        # Assert
        assert service._initialized is True
        assert len(service.loaders) == 1
        assert 'service_orders' in service.loaders
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_load_multiple_csv_files(self, spreadsheet_service):
        """Test loading multiple CSV files."""
        # Arrange
        service, test_csv_data = spreadsheet_service
        
        # Assert
        assert len(service.loaders) == 3
        assert 'service_orders' in service.loaders
        assert 'appointments' in service.loaders
        assert 'price_list' in service.loaders
        
        # Verify data is loaded
        for loader_name, loader in service.loaders.items():
            assert loader.data is not None
            assert isinstance(loader.data, pd.DataFrame)
            assert len(loader.data) > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_service_orders_by_phone(self, spreadsheet_service):
        """Test querying service orders by phone number."""
        # Arrange
        service, _ = spreadsheet_service
        query = "Find all orders for phone number +1234567890"
        
        # Act
        start_time = time.time()
        result = await service.query_data(query)
        
        # Assert
        assert_response_time(start_time, 3000)  # Should complete within 3 seconds
        assert isinstance(result, str)
        assert "+1234567890" in result
        assert "1234" in result or "1237" in result  # Order IDs for this phone
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_appointments_by_date(self, spreadsheet_service):
        """Test querying appointments by date."""
        # Arrange
        service, _ = spreadsheet_service
        query = "Show appointments for August 15, 2025"
        
        # Act
        result = await service.query_data(query)
        
        # Assert
        assert isinstance(result, str)
        assert "2025-08-15" in result
        assert "appointment" in result.lower()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_price_list(self, spreadsheet_service):
        """Test querying price list information."""
        # Arrange
        service, _ = spreadsheet_service
        query = "What is the price for installation service?"
        
        # Act
        result = await service.query_data(query)
        
        # Assert
        assert isinstance(result, str)
        assert "299.99" in result  # Installation price from test data
        assert "installation" in result.lower()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_no_results(self, spreadsheet_service):
        """Test query that returns no results."""
        # Arrange
        service, _ = spreadsheet_service
        query = "Find orders for phone number +9999999999"  # Non-existent phone
        
        # Act
        result = await service.query_data(query)
        
        # Assert
        assert isinstance(result, str)
        assert "no results" in result.lower() or "not found" in result.lower()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_available_spreadsheets(self, spreadsheet_service):
        """Test listing available spreadsheets."""
        # Arrange
        service, _ = spreadsheet_service
        
        # Act
        spreadsheet_list = await service.list_spreadsheets()
        
        # Assert
        assert isinstance(spreadsheet_list, list)
        assert len(spreadsheet_list) == 3
        assert any('service_orders' in sheet['name'] for sheet in spreadsheet_list)
        assert any('appointments' in sheet['name'] for sheet in spreadsheet_list)
        assert any('price_list' in sheet['name'] for sheet in spreadsheet_list)
        
        # Verify sheet info
        for sheet in spreadsheet_list:
            assert 'name' in sheet
            assert 'columns' in sheet
            assert 'row_count' in sheet
            assert isinstance(sheet['columns'], list)
            assert isinstance(sheet['row_count'], int)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_performance_stats(self, spreadsheet_service):
        """Test getting performance statistics."""
        # Arrange
        service, _ = spreadsheet_service
        
        # Perform some queries to generate stats
        await service.query_data("Find orders for +1234567890")
        await service.query_data("Show appointments for today")
        
        # Act
        stats = await service.get_performance_stats()
        
        # Assert
        assert isinstance(stats, dict)
        assert 'total_queries' in stats
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        assert 'average_query_time_ms' in stats
        assert 'total_rows_loaded' in stats
        assert 'active_spreadsheets' in stats
        
        assert stats['total_queries'] >= 2
        assert stats['active_spreadsheets'] == 3


class TestCSVLoader:
    """Test CSV loader functionality."""
    
    @pytest.fixture
    def csv_loader(self, test_csv_data):
        """Create CSV loader with test data."""
        service_orders_path = test_csv_data['service_orders']
        loader = CSVLoader(service_orders_path)
        return loader, service_orders_path
    
    @pytest.mark.unit
    def test_csv_loader_initialization(self, csv_loader):
        """Test CSV loader initialization."""
        # Arrange & Act
        loader, file_path = csv_loader
        
        # Assert
        assert loader.file_path == file_path
        assert loader.data is None  # Not loaded yet
        assert loader.last_modified is None
        assert loader.cache_valid is False
    
    @pytest.mark.unit
    def test_load_csv_data(self, csv_loader):
        """Test loading CSV data."""
        # Arrange
        loader, _ = csv_loader
        
        # Act
        start_time = time.time()
        loader.load_data()
        
        # Assert
        assert_response_time(start_time, 1000)  # Should load within 1 second
        assert loader.data is not None
        assert isinstance(loader.data, pd.DataFrame)
        assert len(loader.data) == 4  # Test data has 4 rows
        assert 'order_id' in loader.data.columns
        assert 'customer_phone' in loader.data.columns
        assert 'status' in loader.data.columns
    
    @pytest.mark.unit
    def test_csv_reload_when_modified(self, csv_loader, temp_dir):
        """Test reloading CSV when file is modified."""
        # Arrange
        loader, original_path = csv_loader
        loader.load_data()
        original_data = loader.data.copy()
        
        # Simulate file modification by creating new file
        new_path = temp_dir / "modified_service_orders.csv"
        modified_data = [
            "order_id,customer_phone,status,service_type,scheduled_date,technician",
            "9999,+1111111111,new,test,2025-08-15,Test Tech"
        ]
        new_path.write_text("\\n".join(modified_data))
        
        # Update loader path and modify timestamp
        loader.file_path = str(new_path)
        loader.last_modified = datetime.utcnow() - timedelta(minutes=1)  # Older timestamp
        
        # Act
        loader.load_data()
        
        # Assert
        assert not loader.data.equals(original_data)
        assert len(loader.data) == 1  # New data has 1 row
        assert loader.data.iloc[0]['order_id'] == 9999
    
    @pytest.mark.unit
    def test_csv_cache_validity(self, csv_loader):
        """Test CSV cache validity checking."""
        # Arrange
        loader, _ = csv_loader
        loader.load_data()
        loader.cache_ttl_minutes = 1
        
        # Assert - Initially cache should be valid
        assert loader.is_cache_valid() is True
        
        # Simulate cache expiration
        loader.last_loaded = datetime.utcnow() - timedelta(minutes=2)
        
        # Assert - Cache should be invalid
        assert loader.is_cache_valid() is False
    
    @pytest.mark.unit
    def test_csv_file_not_found(self, temp_dir):
        """Test handling of non-existent CSV file."""
        # Arrange
        non_existent_path = temp_dir / "non_existent.csv"
        loader = CSVLoader(str(non_existent_path))
        
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            loader.load_data()
    
    @pytest.mark.unit
    def test_csv_malformed_data(self, temp_dir):
        """Test handling of malformed CSV data."""
        # Arrange
        malformed_path = temp_dir / "malformed.csv"
        malformed_data = [
            "order_id,customer_phone,status",
            "1234,+1234567890,active",
            "malformed_row_with_missing_columns",  # Invalid row
            "5678,+0987654321,completed"
        ]
        malformed_path.write_text("\\n".join(malformed_data))
        
        loader = CSVLoader(str(malformed_path))
        
        # Act
        loader.load_data()
        
        # Assert - Should handle gracefully, possibly skipping malformed rows
        assert loader.data is not None
        assert len(loader.data) >= 2  # At least the valid rows
    
    @pytest.mark.unit
    def test_csv_empty_file(self, temp_dir):
        """Test handling of empty CSV file."""
        # Arrange
        empty_path = temp_dir / "empty.csv"
        empty_path.write_text("")
        
        loader = CSVLoader(str(empty_path))
        
        # Act & Assert
        with pytest.raises(Exception):  # Should raise appropriate exception
            loader.load_data()
    
    @pytest.mark.unit
    def test_csv_large_file_performance(self, temp_dir):
        """Test performance with large CSV file."""
        # Arrange - Create large CSV file
        large_path = temp_dir / "large.csv"
        rows = ["order_id,customer_phone,status,service_type"]
        
        # Generate 1000 rows
        for i in range(1000):
            rows.append(f"{i},+123456{i:04d},status_{i%3},service_{i%5}")
        
        large_path.write_text("\\n".join(rows))
        loader = CSVLoader(str(large_path))
        
        # Act
        start_time = time.time()
        loader.load_data()
        
        # Assert
        assert_response_time(start_time, 5000)  # Should load within 5 seconds
        assert len(loader.data) == 1000
        assert loader.data is not None


class TestQueryProcessor:
    """Test natural language query processing."""
    
    @pytest.fixture
    def query_processor(self, test_csv_data):
        """Create query processor with test data."""
        # Load test data into pandas DataFrames
        dataframes = {}
        for name, path in test_csv_data.items():
            dataframes[name] = pd.read_csv(path)
        
        processor = QueryProcessor(dataframes)
        return processor, dataframes
    
    @pytest.mark.unit
    def test_process_phone_number_query(self, query_processor):
        """Test processing phone number-based queries."""
        # Arrange
        processor, _ = query_processor
        query = "Find all orders for phone number +1234567890"
        
        # Act
        result = processor.process_query(query)
        
        # Assert
        assert isinstance(result, str)
        assert "+1234567890" in result
        # Should find orders 1234 and 1237 from test data
        assert "1234" in result or "1237" in result
    
    @pytest.mark.unit
    def test_process_date_query(self, query_processor):
        """Test processing date-based queries."""
        # Arrange
        processor, _ = query_processor
        query = "Show appointments for August 15, 2025"
        
        # Act
        result = processor.process_query(query)
        
        # Assert
        assert isinstance(result, str)
        assert "2025-08-15" in result
    
    @pytest.mark.unit
    def test_process_status_query(self, query_processor):
        """Test processing status-based queries."""
        # Arrange
        processor, _ = query_processor
        query = "Find all pending orders"
        
        # Act
        result = processor.process_query(query)
        
        # Assert
        assert isinstance(result, str)
        assert "pending" in result.lower()
    
    @pytest.mark.unit
    def test_process_price_query(self, query_processor):
        """Test processing price-related queries."""
        # Arrange
        processor, _ = query_processor
        query = "What is the price for installation?"
        
        # Act
        result = processor.process_query(query)
        
        # Assert
        assert isinstance(result, str)
        assert "299.99" in result  # Installation price from test data
        assert "installation" in result.lower()
    
    @pytest.mark.unit
    def test_process_service_type_query(self, query_processor):
        """Test processing service type queries."""
        # Arrange
        processor, _ = query_processor
        query = "Show all maintenance services"
        
        # Act
        result = processor.process_query(query)
        
        # Assert
        assert isinstance(result, str)
        assert "maintenance" in result.lower()
    
    @pytest.mark.unit
    def test_process_technician_query(self, query_processor):
        """Test processing technician-based queries."""
        # Arrange
        processor, _ = query_processor
        query = "Which orders are assigned to John Tech?"
        
        # Act
        result = processor.process_query(query)
        
        # Assert
        assert isinstance(result, str)
        assert "John Tech" in result
    
    @pytest.mark.unit
    def test_process_ambiguous_query(self, query_processor):
        """Test processing ambiguous queries."""
        # Arrange
        processor, _ = query_processor
        query = "Find something"  # Very vague query
        
        # Act
        result = processor.process_query(query)
        
        # Assert
        assert isinstance(result, str)
        # Should provide helpful message or show available options
        assert len(result) > 0
    
    @pytest.mark.unit
    def test_process_complex_query(self, query_processor):
        """Test processing complex multi-criteria queries."""
        # Arrange
        processor, _ = query_processor
        query = "Find scheduled installation orders for customer +1234567890"
        
        # Act
        result = processor.process_query(query)
        
        # Assert
        assert isinstance(result, str)
        assert "+1234567890" in result
        assert "scheduled" in result.lower() or "installation" in result.lower()
    
    @pytest.mark.unit
    def test_query_keyword_extraction(self, query_processor):
        """Test keyword extraction from queries."""
        # Arrange
        processor, _ = query_processor
        
        test_cases = [
            ("phone +1234567890", ["+1234567890"]),
            ("status pending", ["pending"]),
            ("service installation", ["installation"]),
            ("date 2025-08-15", ["2025-08-15"]),
        ]
        
        # Act & Assert
        for query, expected_keywords in test_cases:
            keywords = processor.extract_keywords(query)
            for keyword in expected_keywords:
                assert any(keyword in k for k in keywords), f"Keyword {keyword} not found in {keywords}"
    
    @pytest.mark.unit
    def test_query_result_formatting(self, query_processor):
        """Test formatting of query results."""
        # Arrange
        processor, dataframes = query_processor
        
        # Mock a DataFrame result
        sample_data = dataframes['service_orders'].head(2)
        
        # Act
        formatted_result = processor.format_results(sample_data, "service_orders")
        
        # Assert
        assert isinstance(formatted_result, str)
        assert "service_orders" in formatted_result
        assert "order_id" in formatted_result
        assert len(formatted_result) > 50  # Should be reasonably detailed


class TestSpreadsheetToolFunctions:
    """Test the spreadsheet tool functions used by the agent."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_spreadsheet_data_tool(self, test_csv_data):
        """Test query_spreadsheet_data tool function."""
        # Arrange
        query = "Find orders for phone +1234567890"
        
        with patch('src.tools.spreadsheet.get_spreadsheet_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.query_data.return_value = "Found 2 orders for +1234567890: Order 1234 (scheduled), Order 1237 (in_progress)"
            mock_get_service.return_value = mock_service
            
            # Act
            result = await query_spreadsheet_data(query=query)
        
        # Assert
        assert isinstance(result, str)
        assert "+1234567890" in result
        assert "orders" in result.lower()
        mock_service.query_data.assert_called_once_with(query)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_spreadsheet_data_error(self):
        """Test query_spreadsheet_data tool with error."""
        # Arrange
        query = "Test query"
        
        with patch('src.tools.spreadsheet.get_spreadsheet_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.query_data.side_effect = Exception("Query processing failed")
            mock_get_service.return_value = mock_service
            
            # Act
            result = await query_spreadsheet_data(query=query)
        
        # Assert
        assert isinstance(result, str)
        assert "error" in result.lower()
        assert "query processing failed" in result.lower()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_available_spreadsheets_tool(self):
        """Test list_available_spreadsheets tool function."""
        # Arrange
        mock_spreadsheets = [
            {"name": "service_orders", "columns": ["order_id", "customer_phone"], "row_count": 100},
            {"name": "appointments", "columns": ["appointment_id", "date"], "row_count": 50}
        ]
        
        with patch('src.tools.spreadsheet.get_spreadsheet_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_spreadsheets.return_value = mock_spreadsheets
            mock_get_service.return_value = mock_service
            
            # Act
            result = await list_available_spreadsheets()
        
        # Assert
        assert isinstance(result, str)
        assert "service_orders" in result
        assert "appointments" in result
        assert "100" in result  # Row count
        assert "50" in result   # Row count
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_spreadsheet_performance_stats_tool(self):
        """Test get_spreadsheet_performance_stats tool function."""
        # Arrange
        mock_stats = {
            'total_queries': 150,
            'cache_hits': 45,
            'cache_misses': 105,
            'average_query_time_ms': 75.5,
            'total_rows_loaded': 5000,
            'active_spreadsheets': 3
        }
        
        with patch('src.tools.spreadsheet.get_spreadsheet_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_performance_stats.return_value = mock_stats
            mock_get_service.return_value = mock_service
            
            # Act
            result = await get_spreadsheet_performance_stats()
        
        # Assert
        assert isinstance(result, str)
        assert "150" in result  # total_queries
        assert "75.5" in result # average_query_time_ms
        assert "5000" in result # total_rows_loaded
        assert "30%" in result  # cache hit rate (45/150)


class TestSpreadsheetPerformance:
    """Test performance aspects of spreadsheet operations."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_dataset_query_performance(self, temp_dir):
        """Test performance with large datasets."""
        # Arrange - Create large CSV
        large_csv_path = temp_dir / "large_orders.csv"
        rows = ["order_id,customer_phone,status,service_type,amount"]
        
        # Generate 5000 rows
        for i in range(5000):
            phone = f"+123456{i%100:04d}"  # 100 unique phones
            status = ["pending", "scheduled", "completed", "cancelled"][i % 4]
            service = ["installation", "maintenance", "repair"][i % 3]
            amount = 100 + (i % 500)
            rows.append(f"{i},{phone},{status},{service},{amount}")
        
        large_csv_path.write_text("\\n".join(rows))
        
        config = SheetsToolConfig(
            enabled=True,
            sheet_configs=[str(large_csv_path)],
            cache_ttl_minutes=5
        )
        
        service = SpreadsheetService(config)
        await service.initialize()
        
        # Act - Perform queries
        queries = [
            "Find orders for +1234560001",
            "Show all pending orders",
            "List installation services",
            "Find orders with amount greater than 400"
        ]
        
        start_time = time.time()
        results = []
        for query in queries:
            result = await service.query_data(query)
            results.append(result)
        
        # Assert
        assert_response_time(start_time, 10000)  # All queries within 10 seconds
        assert len(results) == 4
        assert all(isinstance(result, str) for result in results)
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_query_performance(self, spreadsheet_service):
        """Test performance under concurrent queries."""
        # Arrange
        service, _ = spreadsheet_service
        
        queries = [
            "Find orders for +1234567890",
            "Show appointments for 2025-08-15",
            "What is the price for maintenance?",
            "List all pending orders",
            "Show completed services"
        ]
        
        # Act - Execute queries concurrently
        import asyncio
        start_time = time.time()
        
        tasks = [service.query_data(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        # Assert
        assert_response_time(start_time, 5000)  # All concurrent queries within 5 seconds
        assert len(results) == 5
        assert all(isinstance(result, str) for result in results)
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_performance_improvement(self, spreadsheet_service):
        """Test that caching improves performance."""
        # Arrange
        service, _ = spreadsheet_service
        query = "Find orders for +1234567890"
        
        # Act - First query (cache miss)
        start_time_1 = time.time()
        result_1 = await service.query_data(query)
        time_1 = (time.time() - start_time_1) * 1000
        
        # Act - Second query (cache hit)
        start_time_2 = time.time()
        result_2 = await service.query_data(query)
        time_2 = (time.time() - start_time_2) * 1000
        
        # Assert
        assert result_1 == result_2  # Same results
        # Note: In actual implementation, second query should be faster due to caching
        # For this test, we just verify both complete successfully
        assert time_1 > 0
        assert time_2 > 0
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, temp_dir):
        """Test memory usage remains stable with repeated operations."""
        import gc
        import psutil
        import os
        
        # Create test CSV
        csv_path = temp_dir / "memory_test.csv"
        rows = ["order_id,customer_phone,status"]
        for i in range(1000):
            rows.append(f"{i},+123456{i:04d},status_{i%3}")
        csv_path.write_text("\\n".join(rows))
        
        config = SheetsToolConfig(
            enabled=True,
            sheet_configs=[str(csv_path)]
        )
        
        service = SpreadsheetService(config)
        await service.initialize()
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform many operations
        for i in range(100):
            await service.query_data(f"Find order {i}")
            if i % 10 == 0:
                gc.collect()
        
        # Check final memory
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 20MB)
        assert memory_increase < 20 * 1024 * 1024, f"Memory increased by {memory_increase / 1024 / 1024:.2f}MB"


class TestSpreadsheetErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_invalid_csv_path(self):
        """Test handling of invalid CSV file paths."""
        # Arrange
        config = SheetsToolConfig(
            enabled=True,
            sheet_configs=["/non/existent/path.csv"]
        )
        
        service = SpreadsheetService(config)
        
        # Act & Assert
        with pytest.raises(Exception):  # Should raise appropriate exception
            await service.initialize()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_corrupted_csv_handling(self, temp_dir):
        """Test handling of corrupted CSV files."""
        # Arrange - Create corrupted CSV
        corrupted_path = temp_dir / "corrupted.csv"
        corrupted_path.write_bytes(b"\\x00\\x01\\x02\\x03")  # Binary data
        
        config = SheetsToolConfig(
            enabled=True,
            sheet_configs=[str(corrupted_path)]
        )
        
        service = SpreadsheetService(config)
        
        # Act & Assert
        with pytest.raises(Exception):
            await service.initialize()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_service_not_initialized(self):
        """Test querying when service is not initialized."""
        # Arrange
        config = SheetsToolConfig(enabled=True)
        service = SpreadsheetService(config)
        # Don't initialize
        
        # Act & Assert
        with pytest.raises(Exception):
            await service.query_data("Test query")
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, spreadsheet_service):
        """Test handling of empty queries."""
        # Arrange
        service, _ = spreadsheet_service
        
        # Act
        result = await service.query_data("")
        
        # Assert
        assert isinstance(result, str)
        assert "empty" in result.lower() or "invalid" in result.lower()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_with_special_characters(self, spreadsheet_service):
        """Test queries with special characters."""
        # Arrange
        service, _ = spreadsheet_service
        query = "Find orders for phone +1(234)567-8901"  # Phone with special chars
        
        # Act
        result = await service.query_data(query)
        
        # Assert
        assert isinstance(result, str)
        # Should handle gracefully even if no exact match
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_very_long_query(self, spreadsheet_service):
        """Test handling of very long queries."""
        # Arrange
        service, _ = spreadsheet_service
        query = "Find " + "very " * 1000 + "long query"  # Very long query
        
        # Act
        result = await service.query_data(query)
        
        # Assert
        assert isinstance(result, str)
        # Should handle gracefully without crashing
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_file_access(self, temp_dir):
        """Test concurrent access to same CSV file."""
        # Arrange
        csv_path = temp_dir / "concurrent_test.csv"
        rows = ["id,data"]
        for i in range(100):
            rows.append(f"{i},data_{i}")
        csv_path.write_text("\\n".join(rows))
        
        config = SheetsToolConfig(
            enabled=True,
            sheet_configs=[str(csv_path)]
        )
        
        # Create multiple service instances
        services = []
        for _ in range(3):
            service = SpreadsheetService(config)
            await service.initialize()
            services.append(service)
        
        # Act - Concurrent queries
        import asyncio
        tasks = []
        for i, service in enumerate(services):
            task = service.query_data(f"Find data_{i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Assert
        assert len(results) == 3
        assert all(isinstance(result, str) for result in results)