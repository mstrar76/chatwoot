"""
Spreadsheet Tool for Chatwoot Agent MVP.

This tool provides production-ready spreadsheet querying functionality with:
- LangChain tool integration using @tool decorator
- In-memory CSV data loader with TTL caching
- Natural language to data lookup interface
- Multiple sheet support (service_orders, appointments, etc.)
- Query optimization and caching strategies
- Async operations for performance
- Comprehensive error handling and fallback strategies

Technical Requirements:
- In-memory CSV processing with pandas
- TTL-based caching for data freshness
- Natural language query interpretation
- Support for multiple data sources/sheets
- Performance optimization with query caching
- Integration with existing Chatwoot Agent services
"""

import asyncio
import csv
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from langchain_core.tools import tool
from pydantic import BaseModel, Field, validator

from ..models.schemas import SheetsToolConfig
from ..utils.config import get_config
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Custom exceptions for spreadsheet operations
class SpreadsheetError(Exception):
    """Base exception for spreadsheet operations."""
    pass

class DataLoadError(SpreadsheetError):
    """Exception for data loading issues."""
    pass

class QueryError(SpreadsheetError):
    """Exception for query processing issues."""
    pass

class ConfigurationError(SpreadsheetError):
    """Exception for spreadsheet configuration issues."""
    pass


class SpreadsheetQuery(BaseModel):
    """Input schema for spreadsheet queries."""
    query: str = Field(..., description="Natural language query to search the spreadsheet data")
    sheet_name: Optional[str] = Field(None, description="Specific sheet/file to search (optional)")
    limit: int = Field(10, description="Maximum number of results to return")
    include_summary: bool = Field(True, description="Include summary statistics in results")
    format_output: str = Field("table", description="Output format: 'table', 'json', 'summary'")
    
    @validator('query')
    def validate_query(cls, v):
        """Ensure query is not empty and within reasonable length."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > 1000:
            raise ValueError("Query too long (max 1000 characters)")
        return v.strip()
    
    @validator('limit')
    def validate_limit(cls, v):
        """Ensure limit is within reasonable bounds."""
        if v < 1 or v > 100:
            raise ValueError("Limit must be between 1 and 100")
        return v
    
    @validator('format_output')
    def validate_format(cls, v):
        """Ensure output format is valid."""
        valid_formats = ['table', 'json', 'summary']
        if v not in valid_formats:
            raise ValueError(f"Format must be one of: {valid_formats}")
        return v


class SpreadsheetResult(BaseModel):
    """Schema for spreadsheet query results."""
    data: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    total_rows: int = Field(0, description="Total rows found")
    query_time_ms: int = Field(0, description="Time taken for query in milliseconds")
    sheet_name: Optional[str] = Field(None, description="Sheet that was queried")
    columns_found: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(None, description="Error message if query failed")
    
    def format_as_table(self, max_width: int = 80) -> str:
        """
        Format results as a readable table string.
        
        Args:
            max_width: Maximum width for the table
            
        Returns:
            Formatted table string
        """
        if not self.data:
            return "No data found for the query."
        
        # Create DataFrame for better formatting
        df = pd.DataFrame(self.data)
        
        # Truncate wide tables
        if len(df.columns) > 8:
            df = df.iloc[:, :8]
            truncated_cols = len(self.data[0]) - 8 if self.data else 0
            note = f"\n... and {truncated_cols} more columns"
        else:
            note = ""
        
        # Format the table
        table_str = df.to_string(index=False, max_rows=self.total_rows)
        
        # Add summary if available
        summary_str = ""
        if self.summary:
            summary_parts = []
            for key, value in self.summary.items():
                summary_parts.append(f"{key}: {value}")
            summary_str = f"\nSummary: {', '.join(summary_parts)}"
        
        return f"{table_str}{note}{summary_str}"
    
    def format_as_json(self) -> str:
        """Format results as JSON string."""
        import json
        result_dict = {
            "data": self.data,
            "summary": self.summary,
            "total_rows": self.total_rows,
            "sheet_name": self.sheet_name,
            "columns_found": self.columns_found
        }
        return json.dumps(result_dict, indent=2, default=str)


@dataclass
class CachedSheet:
    """Represents a cached spreadsheet with metadata."""
    name: str
    data: pd.DataFrame
    loaded_at: datetime
    file_path: Optional[str] = None
    row_count: int = 0
    column_count: int = 0
    last_modified: Optional[datetime] = None
    
    def __post_init__(self):
        """Calculate metadata after initialization."""
        self.row_count = len(self.data)
        self.column_count = len(self.data.columns)
    
    def is_expired(self, ttl_minutes: int) -> bool:
        """Check if the cached data has expired."""
        expiry_time = self.loaded_at + timedelta(minutes=ttl_minutes)
        return datetime.utcnow() > expiry_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the sheet."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        summary = {
            "rows": self.row_count,
            "columns": self.column_count,
            "loaded_at": self.loaded_at.isoformat(),
            "columns_list": list(self.data.columns),
        }
        
        if len(numeric_cols) > 0:
            summary["numeric_columns"] = len(numeric_cols)
            summary["total_numeric_values"] = int(self.data[numeric_cols].count().sum())
        
        return summary


class SpreadsheetService:
    """
    Production-ready spreadsheet service with caching and natural language querying.
    
    Provides in-memory CSV processing with TTL caching, natural language query
    interpretation, and support for multiple data sources.
    """
    
    def __init__(self, config: Optional[SheetsToolConfig] = None):
        """
        Initialize spreadsheet service with configuration.
        
        Args:
            config: Spreadsheet configuration, uses defaults if None
        """
        self.config = config or SheetsToolConfig()
        self._cached_sheets: Dict[str, CachedSheet] = {}
        self._query_cache: Dict[str, Tuple[SpreadsheetResult, datetime]] = {}
        self._initialized = False
        
        # Performance metrics
        self._query_count = 0
        self._cache_hits = 0
        self._load_times: List[float] = []
        
        # Data directory - can be configured
        app_config = get_config()
        self._data_dir = Path(app_config.chatwoot.base_url).parent / "data" / "sheets"
        if not self._data_dir.exists():
            self._data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Spreadsheet service initialized", 
                   config=self.config.dict(),
                   data_dir=str(self._data_dir))
    
    def _get_cache_key(self, query: str, sheet_name: Optional[str], limit: int) -> str:
        """Generate cache key for query results."""
        content = f"{query}:{sheet_name}:{limit}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _detect_file_encoding(self, file_path: Path) -> str:
        """Detect file encoding for proper CSV reading."""
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8')
        except ImportError:
            # Fallback if chardet is not available
            return 'utf-8'
    
    async def _load_csv_file(self, file_path: Path, sheet_name: str) -> CachedSheet:
        """
        Load CSV file into memory with error handling.
        
        Args:
            file_path: Path to the CSV file
            sheet_name: Name to assign to the sheet
            
        Returns:
            CachedSheet with loaded data
            
        Raises:
            DataLoadError: If file loading fails
        """
        start_time = time.time()
        
        try:
            logger.info("Loading CSV file", file_path=str(file_path), sheet_name=sheet_name)
            
            # Check if file exists
            if not file_path.exists():
                raise DataLoadError(f"File not found: {file_path}")
            
            # Get file metadata
            file_stats = file_path.stat()
            last_modified = datetime.fromtimestamp(file_stats.st_mtime)
            
            # Detect encoding
            encoding = self._detect_file_encoding(file_path)
            
            # Load CSV with pandas
            # Try different separators and handle common issues
            load_kwargs = {
                'encoding': encoding,
                'low_memory': False,
                'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'None'],
                'keep_default_na': True
            }
            
            # Try comma separator first
            try:
                df = pd.read_csv(file_path, sep=',', **load_kwargs)
            except pd.errors.ParserError:
                # Try semicolon separator
                try:
                    df = pd.read_csv(file_path, sep=';', **load_kwargs)
                except pd.errors.ParserError:
                    # Try tab separator
                    df = pd.read_csv(file_path, sep='\t', **load_kwargs)
            
            # Basic data cleaning
            df = df.dropna(how='all')  # Remove completely empty rows
            df.columns = [str(col).strip() for col in df.columns]  # Clean column names
            
            # Convert string columns that look like numbers
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to convert to numeric if it looks like numbers
                    try:
                        numeric_values = pd.to_numeric(df[col], errors='coerce')
                        if not numeric_values.isna().all():
                            df[col] = numeric_values
                    except:
                        pass
            
            load_time = time.time() - start_time
            self._load_times.append(load_time)
            
            cached_sheet = CachedSheet(
                name=sheet_name,
                data=df,
                loaded_at=datetime.utcnow(),
                file_path=str(file_path),
                last_modified=last_modified
            )
            
            logger.info("CSV file loaded successfully",
                       sheet_name=sheet_name,
                       rows=cached_sheet.row_count,
                       columns=cached_sheet.column_count,
                       load_time_ms=int(load_time * 1000))
            
            return cached_sheet
            
        except Exception as e:
            logger.error("Failed to load CSV file",
                        file_path=str(file_path),
                        sheet_name=sheet_name,
                        error=str(e))
            raise DataLoadError(f"Failed to load {file_path}: {e}")
    
    async def _discover_sheets(self) -> List[str]:
        """
        Discover available CSV files in the data directory.
        
        Returns:
            List of available sheet names
        """
        try:
            sheets = []
            
            # Add configured sheets
            if self.config.sheet_configs:
                sheets.extend(self.config.sheet_configs)
            
            # Discover CSV files in data directory
            if self._data_dir.exists():
                for file_path in self._data_dir.glob("*.csv"):
                    sheet_name = file_path.stem
                    if sheet_name not in sheets:
                        sheets.append(sheet_name)
            
            logger.debug("Discovered sheets", sheets=sheets)
            return sheets
            
        except Exception as e:
            logger.error("Failed to discover sheets", error=str(e))
            return []
    
    async def _get_or_load_sheet(self, sheet_name: str) -> CachedSheet:
        """
        Get sheet from cache or load it if needed.
        
        Args:
            sheet_name: Name of the sheet to load
            
        Returns:
            CachedSheet instance
            
        Raises:
            DataLoadError: If sheet cannot be loaded
        """
        # Check if we have a cached version
        if sheet_name in self._cached_sheets:
            cached_sheet = self._cached_sheets[sheet_name]
            
            # Check if cache is still valid
            if not cached_sheet.is_expired(self.config.cache_ttl_minutes):
                logger.debug("Using cached sheet", sheet_name=sheet_name)
                return cached_sheet
            else:
                logger.info("Sheet cache expired, reloading", sheet_name=sheet_name)
                del self._cached_sheets[sheet_name]
        
        # Load the sheet
        file_path = self._data_dir / f"{sheet_name}.csv"
        
        # Check if we have a configured path
        if self.config.csv_path:
            csv_dir = Path(self.config.csv_path)
            if csv_dir.exists():
                file_path = csv_dir / f"{sheet_name}.csv"
        
        cached_sheet = await self._load_csv_file(file_path, sheet_name)
        self._cached_sheets[sheet_name] = cached_sheet
        
        return cached_sheet
    
    def _parse_natural_language_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Parse natural language query into pandas operations.
        
        Args:
            query: Natural language query
            df: DataFrame to query
            
        Returns:
            Dictionary with query parameters
        """
        query_lower = query.lower()
        params = {
            'filters': [],
            'sort_by': None,
            'sort_ascending': True,
            'columns': None,
            'aggregation': None,
            'group_by': None
        }
        
        # Column names for matching
        columns = [col.lower() for col in df.columns]
        column_map = {col.lower(): col for col in df.columns}
        
        # Extract column references
        mentioned_columns = []
        for col_lower, col_original in column_map.items():
            if col_lower in query_lower:
                mentioned_columns.append(col_original)
        
        # Sort keywords
        if any(keyword in query_lower for keyword in ['sort', 'order', 'arrange']):
            if 'desc' in query_lower or 'descending' in query_lower or 'highest' in query_lower:
                params['sort_ascending'] = False
            
            # Try to find column to sort by
            for col in mentioned_columns:
                if col.lower() in query_lower:
                    params['sort_by'] = col
                    break
        
        # Filter keywords
        filter_patterns = [
            ('=', ['equal', 'equals', 'is']),
            ('>', ['greater than', 'more than', 'above']),
            ('<', ['less than', 'below', 'under']),
            ('>=', ['at least', 'minimum']),
            ('<=', ['at most', 'maximum']),
            ('contains', ['contains', 'includes', 'has'])
        ]
        
        # Aggregation keywords
        if any(keyword in query_lower for keyword in ['count', 'total', 'sum', 'average', 'mean', 'max', 'min']):
            if 'count' in query_lower:
                params['aggregation'] = 'count'
            elif any(k in query_lower for k in ['sum', 'total']):
                params['aggregation'] = 'sum'
            elif any(k in query_lower for k in ['average', 'mean']):
                params['aggregation'] = 'mean'
            elif 'max' in query_lower:
                params['aggregation'] = 'max'
            elif 'min' in query_lower:
                params['aggregation'] = 'min'
        
        # Group by keywords
        if any(keyword in query_lower for keyword in ['by', 'group', 'per']):
            for col in mentioned_columns:
                if f"by {col.lower()}" in query_lower or f"per {col.lower()}" in query_lower:
                    params['group_by'] = col
                    break
        
        # Column selection
        if mentioned_columns:
            params['columns'] = mentioned_columns
        
        logger.debug("Parsed query parameters", 
                    query=query,
                    params=params,
                    mentioned_columns=mentioned_columns)
        
        return params
    
    def _apply_query_operations(
        self, 
        df: pd.DataFrame, 
        params: Dict[str, Any], 
        limit: int
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply parsed query operations to DataFrame.
        
        Args:
            df: Source DataFrame
            params: Parsed query parameters
            limit: Maximum number of results
            
        Returns:
            Tuple of (filtered DataFrame, summary statistics)
        """
        result_df = df.copy()
        summary = {}
        
        try:
            # Apply column selection
            if params['columns']:
                available_cols = [col for col in params['columns'] if col in df.columns]
                if available_cols:
                    result_df = result_df[available_cols]
                    summary['selected_columns'] = available_cols
            
            # Apply grouping and aggregation
            if params['group_by'] and params['group_by'] in result_df.columns:
                if params['aggregation']:
                    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        agg_func = params['aggregation']
                        if agg_func == 'count':
                            result_df = result_df.groupby(params['group_by']).size().reset_index(name='count')
                        else:
                            result_df = result_df.groupby(params['group_by'])[numeric_cols].agg(agg_func).reset_index()
                        summary['grouped_by'] = params['group_by']
                        summary['aggregation'] = params['aggregation']
            
            # Apply sorting
            if params['sort_by'] and params['sort_by'] in result_df.columns:
                result_df = result_df.sort_values(
                    by=params['sort_by'],
                    ascending=params['sort_ascending']
                )
                summary['sorted_by'] = f"{params['sort_by']} ({'asc' if params['sort_ascending'] else 'desc'})"
            
            # Apply limit
            if len(result_df) > limit:
                result_df = result_df.head(limit)
                summary['limited_to'] = limit
                summary['total_available'] = len(df)
            
            # Add basic statistics for numeric columns
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary['numeric_summary'] = {}
                for col in numeric_cols:
                    if col in result_df.columns:
                        col_summary = {
                            'mean': float(result_df[col].mean()) if not result_df[col].isna().all() else None,
                            'count': int(result_df[col].count())
                        }
                        summary['numeric_summary'][col] = col_summary
            
        except Exception as e:
            logger.warning("Error applying query operations", error=str(e))
            # Return original data if operations fail
            result_df = df.head(limit)
            summary['error'] = f"Query processing error: {str(e)}"
        
        return result_df, summary
    
    async def query_sheet(
        self,
        query: str,
        sheet_name: Optional[str] = None,
        limit: int = 10,
        include_summary: bool = True,
        format_output: str = "table"
    ) -> SpreadsheetResult:
        """
        Query spreadsheet data with natural language.
        
        Args:
            query: Natural language query
            sheet_name: Specific sheet to query (optional)
            limit: Maximum results to return
            include_summary: Include summary statistics
            format_output: Output format preference
            
        Returns:
            SpreadsheetResult with query results
        """
        start_time = time.time()
        self._query_count += 1
        
        try:
            # Input validation
            spreadsheet_query = SpreadsheetQuery(
                query=query,
                sheet_name=sheet_name,
                limit=limit,
                include_summary=include_summary,
                format_output=format_output
            )
            
            # Check query cache
            cache_key = self._get_cache_key(query, sheet_name, limit)
            if cache_key in self._query_cache:
                cached_result, cached_time = self._query_cache[cache_key]
                cache_age = datetime.utcnow() - cached_time
                
                if cache_age.total_seconds() < (self.config.cache_ttl_minutes * 60):
                    self._cache_hits += 1
                    logger.debug("Query cache hit", cache_key=cache_key)
                    return cached_result
                else:
                    del self._query_cache[cache_key]
            
            # Determine which sheet(s) to query
            available_sheets = await self._discover_sheets()
            
            if not available_sheets:
                return SpreadsheetResult(
                    error="No spreadsheet data available. Please ensure CSV files are loaded."
                )
            
            sheets_to_query = []
            if sheet_name:
                if sheet_name in available_sheets:
                    sheets_to_query = [sheet_name]
                else:
                    return SpreadsheetResult(
                        error=f"Sheet '{sheet_name}' not found. Available: {', '.join(available_sheets)}"
                    )
            else:
                # Query all available sheets and combine results
                sheets_to_query = available_sheets[:3]  # Limit to first 3 sheets for performance
            
            all_results = []
            all_summaries = {}
            
            for sheet in sheets_to_query:
                try:
                    # Load sheet data
                    cached_sheet = await self._get_or_load_sheet(sheet)
                    df = cached_sheet.data
                    
                    # Parse natural language query
                    query_params = self._parse_natural_language_query(query, df)
                    
                    # Apply query operations
                    result_df, summary = self._apply_query_operations(df, query_params, limit)
                    
                    # Convert to records
                    records = result_df.to_dict('records')
                    
                    # Clean up NaN values for JSON serialization
                    for record in records:
                        for key, value in record.items():
                            if pd.isna(value):
                                record[key] = None
                    
                    all_results.extend(records)
                    all_summaries[sheet] = summary
                    
                    # If we found results and only querying one sheet, break
                    if records and sheet_name:
                        break
                        
                except Exception as e:
                    logger.error("Error querying sheet", sheet=sheet, error=str(e))
                    all_summaries[sheet] = {"error": str(e)}
                    continue
            
            # Combine results and create final response
            query_time_ms = int((time.time() - start_time) * 1000)
            
            result = SpreadsheetResult(
                data=all_results[:limit],  # Respect overall limit
                summary=all_summaries,
                total_rows=len(all_results),
                query_time_ms=query_time_ms,
                sheet_name=sheet_name or (sheets_to_query[0] if sheets_to_query else None),
                columns_found=list(all_results[0].keys()) if all_results else []
            )
            
            # Cache the result
            self._query_cache[cache_key] = (result, datetime.utcnow())
            
            # Limit cache size
            if len(self._query_cache) > 100:
                oldest_key = min(self._query_cache.keys(), 
                               key=lambda k: self._query_cache[k][1])
                del self._query_cache[oldest_key]
            
            logger.info("Spreadsheet query completed",
                       query=query[:100],
                       sheets_queried=len(sheets_to_query),
                       results_found=len(all_results),
                       query_time_ms=query_time_ms)
            
            return result
            
        except Exception as e:
            logger.error("Spreadsheet query failed",
                        query=query[:100],
                        sheet_name=sheet_name,
                        error=str(e))
            return SpreadsheetResult(error=f"Query failed: {str(e)}")
    
    async def list_available_sheets(self) -> Dict[str, Any]:
        """
        Get information about available spreadsheets.
        
        Returns:
            Dictionary with sheet information
        """
        try:
            sheets_info = {}
            available_sheets = await self._discover_sheets()
            
            for sheet_name in available_sheets:
                try:
                    cached_sheet = await self._get_or_load_sheet(sheet_name)
                    sheets_info[sheet_name] = cached_sheet.get_summary()
                except Exception as e:
                    sheets_info[sheet_name] = {"error": str(e)}
            
            return {
                "available_sheets": sheets_info,
                "total_sheets": len(available_sheets),
                "cache_stats": {
                    "cached_sheets": len(self._cached_sheets),
                    "query_cache_size": len(self._query_cache),
                    "total_queries": self._query_count,
                    "cache_hit_rate": self._cache_hits / max(self._query_count, 1)
                }
            }
            
        except Exception as e:
            logger.error("Failed to list available sheets", error=str(e))
            return {"error": f"Failed to list sheets: {str(e)}"}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the spreadsheet service.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_load_time = sum(self._load_times) / len(self._load_times) if self._load_times else 0
        cache_hit_rate = self._cache_hits / max(self._query_count, 1)
        
        return {
            "total_queries": self._query_count,
            "cache_hit_rate": cache_hit_rate,
            "cached_sheets": len(self._cached_sheets),
            "query_cache_size": len(self._query_cache),
            "average_load_time_seconds": avg_load_time,
            "total_data_loads": len(self._load_times),
            "cache_ttl_minutes": self.config.cache_ttl_minutes,
            "data_directory": str(self._data_dir)
        }


# Global spreadsheet service instance
_spreadsheet_service: Optional[SpreadsheetService] = None

async def get_spreadsheet_service() -> SpreadsheetService:
    """Get the global spreadsheet service instance."""
    global _spreadsheet_service
    if _spreadsheet_service is None:
        _spreadsheet_service = SpreadsheetService()
    return _spreadsheet_service


# LangChain Tool Implementation
@tool
async def query_spreadsheet_data(
    query: str,
    sheet_name: Optional[str] = None,
    limit: int = 10,
    format_output: str = "table"
) -> str:
    """
    Query spreadsheet data using natural language.
    
    This tool allows you to search and analyze CSV/spreadsheet data using natural language
    queries. It supports filtering, sorting, aggregation, and basic data analysis operations.
    The data is cached in memory for fast querying and supports multiple spreadsheets.
    
    Args:
        query: Natural language query to search the data (e.g., "show me orders from last month", 
               "count customers by city", "find highest sales amounts")
        sheet_name: Specific spreadsheet/file to search (optional, searches all if not specified)
        limit: Maximum number of results to return (1-100, default: 10)
        format_output: Output format - 'table' for formatted table, 'json' for structured data,
                      'summary' for just summary statistics (default: 'table')
        
    Returns:
        Formatted string with query results, including data and summary information.
        Returns error message if query fails.
        
    Examples:
        results = await query_spreadsheet_data(
            query="show me all orders with amount greater than 1000",
            sheet_name="service_orders",
            limit=5
        )
        
        summary = await query_spreadsheet_data(
            query="count customers by status",
            format_output="summary"
        )
    """
    try:
        # Input validation
        if not query or not query.strip():
            return "Error: Query cannot be empty"
        
        # Validate limit
        limit = max(1, min(100, limit))
        
        # Validate format
        valid_formats = ['table', 'json', 'summary']
        if format_output not in valid_formats:
            format_output = 'table'
        
        # Get spreadsheet service and perform query
        service = await get_spreadsheet_service()
        result = await service.query_sheet(
            query=query.strip(),
            sheet_name=sheet_name.strip() if sheet_name else None,
            limit=limit,
            include_summary=True,
            format_output=format_output
        )
        
        # Handle query errors
        if result.error:
            logger.error("Spreadsheet query failed", 
                        query=query,
                        error=result.error)
            return f"Query failed: {result.error}"
        
        # Handle no results
        if not result.data and format_output != 'summary':
            return f"No data found for query: '{query}'"
        
        # Format results based on requested format
        if format_output == 'json':
            return result.format_as_json()
        elif format_output == 'summary':
            if result.summary:
                summary_parts = []
                for sheet, summary_data in result.summary.items():
                    if isinstance(summary_data, dict) and 'error' not in summary_data:
                        parts = [f"Sheet: {sheet}"]
                        for key, value in summary_data.items():
                            parts.append(f"  {key}: {value}")
                        summary_parts.append('\n'.join(parts))
                return '\n\n'.join(summary_parts) if summary_parts else "No summary available"
            else:
                return "No summary data available"
        else:
            # Default table format
            table_result = result.format_as_table()
            
            # Add metadata
            metadata_parts = []
            if result.total_rows > 0:
                metadata_parts.append(f"Found {result.total_rows} results")
            if result.sheet_name:
                metadata_parts.append(f"from sheet '{result.sheet_name}'")
            if result.query_time_ms > 0:
                metadata_parts.append(f"in {result.query_time_ms}ms")
            
            metadata_str = f"[{', '.join(metadata_parts)}]" if metadata_parts else ""
            
            return f"{metadata_str}\n\n{table_result}" if metadata_str else table_result
            
    except Exception as e:
        logger.error("Unexpected error in spreadsheet tool",
                    query=query[:100] if query else "None",
                    sheet_name=sheet_name,
                    error=str(e))
        return f"Error processing query: {str(e)}"


@tool
async def list_available_spreadsheets() -> str:
    """
    List all available spreadsheets and their information.
    
    This tool shows you what spreadsheet data is available for querying,
    including sheet names, number of rows/columns, and when the data was last loaded.
    Use this to understand what data sources you can query.
    
    Returns:
        Formatted string with information about available spreadsheets,
        including names, sizes, and metadata.
        
    Example:
        sheets_info = await list_available_spreadsheets()
        print(sheets_info)  # Shows all available data sources
    """
    try:
        service = await get_spreadsheet_service()
        sheets_info = await service.list_available_sheets()
        
        if "error" in sheets_info:
            return f"Error: {sheets_info['error']}"
        
        if not sheets_info.get("available_sheets"):
            return "No spreadsheet data available. Please ensure CSV files are loaded in the data directory."
        
        # Format the information
        result_parts = [f"Available Spreadsheets ({sheets_info['total_sheets']} total):"]
        
        for sheet_name, info in sheets_info["available_sheets"].items():
            if "error" in info:
                result_parts.append(f"\n❌ {sheet_name}: {info['error']}")
            else:
                result_parts.append(f"\n✅ {sheet_name}:")
                result_parts.append(f"   - Rows: {info.get('rows', 'Unknown')}")
                result_parts.append(f"   - Columns: {info.get('columns', 'Unknown')}")
                if info.get('columns_list'):
                    cols_preview = ', '.join(info['columns_list'][:5])
                    if len(info['columns_list']) > 5:
                        cols_preview += f" ... (+{len(info['columns_list'])-5} more)"
                    result_parts.append(f"   - Column names: {cols_preview}")
                if info.get('loaded_at'):
                    result_parts.append(f"   - Last loaded: {info['loaded_at'][:19]}")
        
        # Add cache statistics
        cache_stats = sheets_info.get("cache_stats", {})
        if cache_stats:
            result_parts.append(f"\nCache Statistics:")
            result_parts.append(f"- Cached sheets: {cache_stats.get('cached_sheets', 0)}")
            result_parts.append(f"- Total queries: {cache_stats.get('total_queries', 0)}")
            result_parts.append(f"- Cache hit rate: {cache_stats.get('cache_hit_rate', 0):.1%}")
        
        return '\n'.join(result_parts)
        
    except Exception as e:
        logger.error("Error listing available spreadsheets", error=str(e))
        return f"Error retrieving spreadsheet information: {str(e)}"


@tool
async def get_spreadsheet_performance_stats() -> str:
    """
    Get spreadsheet system performance statistics and health information.
    
    Returns performance metrics, cache statistics, and configuration information
    for monitoring and debugging the spreadsheet system.
    
    Returns:
        Formatted string with spreadsheet performance statistics
        
    Example:
        stats = await get_spreadsheet_performance_stats()
        print(stats)  # Shows query performance, cache usage, etc.
    """
    try:
        service = await get_spreadsheet_service()
        stats = service.get_performance_stats()
        
        formatted_stats = [
            f"Spreadsheet Performance Statistics:",
            f"- Total queries processed: {stats['total_queries']}",
            f"- Cache hit rate: {stats['cache_hit_rate']:.2%}",
            f"- Cached sheets: {stats['cached_sheets']}",
            f"- Query cache size: {stats['query_cache_size']} entries",
            f"- Average load time: {stats['average_load_time_seconds']:.3f}s",
            f"- Total data loads: {stats['total_data_loads']}",
            f"- Cache TTL: {stats['cache_ttl_minutes']} minutes",
            f"- Data directory: {stats['data_directory']}"
        ]
        
        return "\n".join(formatted_stats)
        
    except Exception as e:
        logger.error("Error getting spreadsheet performance stats", error=str(e))
        return f"Error retrieving performance stats: {str(e)}"