# Chatwoot Agent MVP - Tools Documentation

This document provides comprehensive documentation for the RAG and Spreadsheet tools implemented for the Chatwoot Agent MVP.

## Overview

The Chatwoot Agent MVP includes two production-ready tools that follow modern LangChain patterns:

1. **RAG Tool** (`src/tools/rag.py`) - Retrieval Augmented Generation for conversation context
2. **Spreadsheet Tool** (`src/tools/spreadsheet.py`) - CSV data querying with natural language

Both tools use:
- LangChain `@tool` decorator for proper integration
- Async/await for performance
- Comprehensive error handling and logging
- Production-ready caching strategies
- Type hints and Pydantic validation

## RAG Tool

### Features

- **Vector Similarity Search**: Uses pgvector with OpenAI embeddings (text-embedding-3-small)
- **Contact Isolation**: Filters results by contact_phone for privacy
- **Performance Optimized**: <2s retrieval time with caching
- **Configurable**: Top-K results, similarity threshold, time windows
- **Error Handling**: Graceful fallbacks and comprehensive logging

### Available Functions

#### `retrieve_relevant_context`

```python
@tool
async def retrieve_relevant_context(
    query: str,
    contact_phone: str, 
    top_k: int = 4,
    similarity_threshold: float = 0.7,
    max_context_length: int = 4000
) -> str:
```

**Purpose**: Retrieve relevant conversation context using RAG

**Parameters**:
- `query`: User's current query to find relevant context for
- `contact_phone`: Contact's phone number (required for privacy filtering)
- `top_k`: Number of most similar messages to retrieve (1-20)
- `similarity_threshold`: Minimum similarity score (0.0-1.0)
- `max_context_length`: Maximum character length of returned context

**Returns**: Formatted string with relevant conversation context

**Example**:
```python
context = await retrieve_relevant_context(
    query="What was my last order status?",
    contact_phone="+5511999999999",
    top_k=3
)
```

#### `store_conversation_context`

```python
@tool
async def store_conversation_context(
    content: str,
    contact_phone: str,
    conversation_id: int,
    message_type: str = "text"
) -> str:
```

**Purpose**: Store conversation content for future RAG retrieval

**Parameters**:
- `content`: Message or content to store
- `contact_phone`: Contact's phone number
- `conversation_id`: Chatwoot conversation ID
- `message_type`: Type of content (text, image, audio, file)

**Returns**: Success or error message

#### `get_rag_performance_stats`

```python
@tool
async def get_rag_performance_stats() -> str:
```

**Purpose**: Get RAG system performance statistics

**Returns**: Formatted string with performance metrics

### Database Schema

The RAG tool uses these database tables (defined in `/migrations/001_create_agent_tables.sql`):

- **messages**: Raw chat transcripts
- **embeddings**: Vector embeddings (1536 dimensions)
- **agent_configs**: Configuration settings
- **agent_memory**: Per-contact memory
- **agent_logs**: Execution logs

### Configuration

RAG configuration is managed through `RAGConfig` in `src/models/schemas.py`:

```python
class RAGConfig(BaseModel):
    enabled: bool = True
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 4
    time_window_days: int = 90
    vector_table: str = "public.embeddings"
    similarity_threshold: float = 0.7
```

## Spreadsheet Tool

### Features

- **Natural Language Queries**: Parse human language into data operations
- **Multiple Sheet Support**: Handle multiple CSV files simultaneously
- **TTL Caching**: In-memory caching with configurable TTL
- **Data Operations**: Filtering, sorting, aggregation, grouping
- **Format Options**: Table, JSON, or summary output formats
- **Performance Optimized**: Query caching and efficient data loading

### Available Functions

#### `query_spreadsheet_data`

```python
@tool
async def query_spreadsheet_data(
    query: str,
    sheet_name: Optional[str] = None,
    limit: int = 10,
    format_output: str = "table"
) -> str:
```

**Purpose**: Query spreadsheet data using natural language

**Parameters**:
- `query`: Natural language query (e.g., "show me orders from last month")
- `sheet_name`: Specific spreadsheet to search (optional)
- `limit`: Maximum number of results (1-100)
- `format_output`: Output format - 'table', 'json', or 'summary'

**Returns**: Formatted string with query results

**Example Queries**:
```python
# Basic filtering
results = await query_spreadsheet_data(
    query="show me all orders with amount greater than 1000",
    sheet_name="service_orders"
)

# Aggregation
summary = await query_spreadsheet_data(
    query="count customers by status",
    format_output="summary"
)

# Sorting
sorted_data = await query_spreadsheet_data(
    query="show highest sales amounts",
    limit=5
)
```

#### `list_available_spreadsheets`

```python
@tool
async def list_available_spreadsheets() -> str:
```

**Purpose**: List all available spreadsheets and their metadata

**Returns**: Formatted string with spreadsheet information

#### `get_spreadsheet_performance_stats`

```python
@tool
async def get_spreadsheet_performance_stats() -> str:
```

**Purpose**: Get spreadsheet system performance statistics

**Returns**: Formatted string with performance metrics

### Data Directory Structure

Spreadsheet data should be placed in CSV files in the data directory:

```
agent_service/
├── data/
│   └── sheets/
│       ├── service_orders.csv
│       ├── appointments.csv
│       └── customers.csv
```

### Natural Language Query Examples

The spreadsheet tool can understand various types of natural language queries:

**Filtering**:
- "show me orders with status completed"
- "find customers from São Paulo"
- "get appointments for today"

**Aggregation**:
- "count orders by status"
- "sum total sales amount"
- "average order value"

**Sorting**:
- "show highest revenue customers"
- "order by date descending"
- "sort by amount"

**Column Selection**:
- "show only name and phone"
- "get customer details"

### Configuration

Spreadsheet configuration is managed through `SheetsToolConfig`:

```python
class SheetsToolConfig(BaseModel):
    enabled: bool = True
    sheet_configs: List[str] = Field(default_factory=list)
    cache_ttl_minutes: int = 5
    csv_path: Optional[str] = None
```

## Integration with Chatwoot Agent

### Tool Lists

The tools are organized into different lists for various use cases:

```python
from src.tools import ALL_TOOLS, CORE_TOOLS, ADMIN_TOOLS

# All available tools
ALL_TOOLS = [
    retrieve_relevant_context,
    store_conversation_context,
    get_rag_performance_stats,
    query_spreadsheet_data,
    list_available_spreadsheets,
    get_spreadsheet_performance_stats,
]

# Core tools for general agent use
CORE_TOOLS = [
    retrieve_relevant_context,
    query_spreadsheet_data,
]

# Administrative/monitoring tools
ADMIN_TOOLS = [
    get_rag_performance_stats,
    get_spreadsheet_performance_stats,
    list_available_spreadsheets,
]
```

### Using with LangChain Agents

```python
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from src.tools import CORE_TOOLS

# Create agent with tools
llm = ChatOpenAI(model="gpt-4")
agent = create_openai_functions_agent(
    llm=llm,
    tools=CORE_TOOLS,
    prompt=agent_prompt
)

# Execute agent
response = await agent.arun("Find my recent orders and summarize them")
```

## Error Handling

Both tools implement comprehensive error handling:

### RAG Tool Errors

- **EmbeddingError**: OpenAI embedding generation failures
- **RetrievalError**: Database query failures
- **ConfigurationError**: Invalid configuration

### Spreadsheet Tool Errors

- **DataLoadError**: CSV file loading failures
- **QueryError**: Query processing failures
- **ConfigurationError**: Invalid configuration

All errors are logged with structured information and return user-friendly error messages.

## Performance Considerations

### RAG Tool

- **Embedding Caching**: Query embeddings are cached to avoid regeneration
- **Database Indexing**: Uses HNSW index on vector column for fast similarity search
- **Connection Pooling**: AsyncPG connection pooling for database efficiency
- **Query Optimization**: Filters by contact_phone and time windows

### Spreadsheet Tool

- **Memory Caching**: CSV data cached in memory with TTL
- **Query Caching**: Query results cached to avoid recomputation
- **Lazy Loading**: Data loaded only when needed
- **Size Limits**: Configurable limits to prevent memory issues

## Testing

Run the test script to verify tool functionality:

```bash
cd agent_service
python test_tools.py
```

The test script:
1. Tests tool schemas and LangChain integration
2. Tests RAG functionality (with graceful database error handling)
3. Tests spreadsheet functionality
4. Creates sample data for realistic testing
5. Runs end-to-end queries with sample data

## Monitoring and Observability

Both tools provide comprehensive metrics:

### RAG Metrics

- Total queries processed
- Average retrieval time
- Cache hit rate
- Embedding cache size
- Database performance

### Spreadsheet Metrics

- Query count and performance
- Cache statistics
- Data load times
- Sheet information

Access metrics through the performance stats tools:

```python
# Get RAG performance statistics
rag_stats = await get_rag_performance_stats()

# Get spreadsheet performance statistics
sheet_stats = await get_spreadsheet_performance_stats()
```

## Production Deployment

### Prerequisites

1. **Database**: PostgreSQL with pgvector extension
2. **Dependencies**: All packages from `requirements.txt`
3. **Environment Variables**: OpenAI API key and database credentials
4. **Data Files**: CSV files in the configured data directory

### Configuration

Set environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export DATABASE_HOST="your-postgres-host"
export DATABASE_PASSWORD="your-password"
```

### Monitoring

Monitor tool performance through:
- Structured logs (JSON format)
- Performance statistics endpoints
- Database query performance
- Cache hit rates

## Security Considerations

### RAG Tool

- **Contact Isolation**: All queries filtered by contact_phone
- **Data Privacy**: No cross-contact data leakage
- **Input Validation**: Query length and parameter validation
- **SQL Injection Prevention**: Parameterized queries only

### Spreadsheet Tool

- **File Access Control**: Limited to configured data directories
- **Input Validation**: Query and parameter validation
- **Memory Limits**: Configurable limits to prevent DoS
- **Error Information**: Sensitive paths not exposed in errors

## Troubleshooting

### Common Issues

1. **RAG queries failing**: Check database connection and pgvector extension
2. **Spreadsheet no data**: Verify CSV files in data directory
3. **Performance issues**: Check cache settings and data sizes
4. **Encoding errors**: Ensure CSV files are UTF-8 encoded

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger("src.tools").setLevel(logging.DEBUG)
```

### Health Checks

Both tools provide health check capabilities through their performance stats functions that can be integrated into application health endpoints.