# Chatwoot Agent Service

Intelligent conversational AI service for Chatwoot with RAG capabilities, spreadsheet tools, and governance controls.

## Features

- 🤖 **LangChain ReAct Agent** - Modern tool-calling agent with OpenAI GPT-4o
- 📚 **RAG System** - pgvector-powered retrieval for contextual responses  
- 📊 **Spreadsheet Tool** - In-memory CSV queries for business data
- 🛡️ **Governance Controls** - Human-in-the-loop, pause/resume, pre-send confirmation
- 🔗 **Chatwoot Integration** - Webhook processing and API message sending
- 🚀 **Production Ready** - Docker, async operations, comprehensive logging

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key
- Chatwoot instance with API access
- PostgreSQL with pgvector extension

### Environment Setup

1. Copy environment template:
```bash
cp .env.example .env
```

2. Configure required variables in `.env`:
```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
CHATWOOT_API_TOKEN=your-chatwoot-api-token-here
```

3. Start the service:
```bash
# From the main chatwoot directory
docker-compose -f docker-compose.agent.yml up -d
```

### Database Setup

Run the database migrations:
```bash
psql $DATABASE_URL -f migrations/001_create_agent_tables.sql
```

## Configuration

### Global Configuration
Edit `config/agent_config.json` for global settings:
- LLM model and parameters
- RAG configuration (top-k, similarity threshold)
- Cost limits and budgets
- Tool configurations

### Inbox-Specific Configuration
Create files in `config/inbox_configs/` for per-inbox settings:
- `inbox_{id}.json` - Override global settings per inbox
- Custom personas and language preferences
- Inbox-specific tool configurations

## API Endpoints

### Health Check
```bash
curl http://localhost:8082/health
```

### Webhook Endpoint
Configure in Chatwoot:
```
URL: http://omnineural_agent:8082/webhook
Events: message_created
```

### Governance Controls
```bash
# Pause agent for conversation
curl -X POST http://localhost:8082/agent/pause/12345

# Resume agent for conversation
curl -X POST http://localhost:8082/agent/resume/12345

# Confirm pending response
curl -X POST http://localhost:8082/agent/confirm/12345 \
  -H "Content-Type: application/json" \
  -d '{"response": "approved"}'
```

### Configuration Management
```bash
# Get global configuration
curl http://localhost:8082/config/global

# Get inbox-specific configuration
curl http://localhost:8082/config/inbox/42
```

### Metrics and Monitoring
```bash
# Get performance metrics
curl http://localhost:8082/metrics

# Get conversation logs
curl http://localhost:8082/logs/12345?limit=50
```

## Data Management

### CSV Data Files
Place CSV files in `data/` directory:
- `service_orders.csv` - Service order information
- `appointments.csv` - Appointment scheduling data
- Custom CSV files as needed

### RAG Content
The agent automatically:
- Stores conversation context in pgvector
- Retrieves relevant historical context
- Filters by contact for privacy
- Supports multilingual content (Portuguese/English)

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Chatwoot      │───▶│  Agent Service   │───▶│   PostgreSQL    │
│   (Webhooks)    │    │  (FastAPI)       │    │   (pgvector)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌────────▼────────┐
                       │   LangChain     │
                       │   ReAct Agent   │
                       └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
            ┌──────────┐ ┌─────────────┐ ┌──────────┐
            │ RAG Tool │ │Spreadsheet  │ │OpenAI API│
            │          │ │    Tool     │ │          │
            └──────────┘ └─────────────┘ └──────────┘
```

## Development

### Running Tests
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v --cov=src/

# Run specific test file
pytest tests/test_agent.py -v
```

### Local Development
```bash
# Set environment for development
export ENVIRONMENT=development
export LOG_LEVEL=debug

# Run with reload
uvicorn src.main:app --reload --host 0.0.0.0 --port 8082
```

### Code Quality
```bash
# Lint and format
ruff check src/ --fix
black src/
mypy src/
```

## Monitoring

### Logs
- Structured JSON logging
- Correlation IDs for request tracing
- Performance metrics (latency, tokens, cost)
- Error tracking and alerting

### Health Checks
- Database connectivity
- External service availability
- Agent system health
- Memory and performance metrics

### Metrics
- Response times and success rates
- Token usage and costs
- Tool usage statistics
- Daily/monthly budget tracking

## Security

### Authentication
- Chatwoot API token authentication
- Optional webhook signature verification
- Rate limiting and request validation

### Data Privacy
- Contact-specific data isolation
- No PII in logs (configurable)
- Secure configuration management
- Input sanitization and validation

### Production Deployment
- Non-root container user
- Read-only configuration mounts
- Health checks and restart policies
- Resource limits and monitoring

## Troubleshooting

### Common Issues

**Agent not responding to messages:**
1. Check webhook configuration in Chatwoot
2. Verify API token and permissions
3. Check agent logs: `docker logs omnineural_agent`
4. Ensure database connectivity

**RAG not returning relevant context:**
1. Verify pgvector extension is installed
2. Check embedding generation logs
3. Adjust similarity threshold in config
4. Verify contact data isolation

**High costs/token usage:**
1. Review daily budget settings
2. Check token counting in logs
3. Adjust max_tokens in configuration
4. Monitor tool usage patterns

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=debug

# Check specific component health
curl http://localhost:8082/health

# View recent errors
docker logs omnineural_agent --tail 100
```

## License

This project is part of the OmniNeural Chatwoot fork and follows the same licensing terms.

## Support

For issues and questions:
1. Check the logs for error messages
2. Review configuration settings
3. Verify external service connectivity
4. Consult the troubleshooting guide above