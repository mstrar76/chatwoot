#!/bin/bash
set -e

# Docker entrypoint script for Chatwoot Agent Service
# Handles initialization, health checks, and graceful startup

echo "=== Chatwoot Agent Service Startup ==="

# Function to check if a service is ready
check_service() {
    local service_name=$1
    local host=$2
    local port=$3
    local max_attempts=30
    local attempt=1
    
    echo "Checking $service_name connectivity at $host:$port..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            echo "✅ $service_name is ready"
            return 0
        fi
        
        echo "⏳ Waiting for $service_name ($attempt/$max_attempts)..."
        sleep 2
        ((attempt++))
    done
    
    echo "❌ Failed to connect to $service_name after $max_attempts attempts"
    return 1
}

# Function to check database connectivity and run migrations if needed
check_database() {
    echo "🗄️ Checking database connectivity..."
    
    # Extract database components from DATABASE_URL
    DB_HOST=$(echo "$DATABASE_URL" | sed 's/.*@\([^:]*\):.*/\1/')
    DB_PORT=$(echo "$DATABASE_URL" | sed 's/.*:\([0-9]*\)\/.*/\1/')
    
    if check_service "PostgreSQL" "$DB_HOST" "$DB_PORT"; then
        echo "✅ Database connectivity confirmed"
        
        # Check if we need to run migrations (in production, this should be done separately)
        if [ "$ENVIRONMENT" = "development" ]; then
            echo "🔄 Development mode: Checking database schema..."
            python -c "
import asyncio
import asyncpg
import os
import sys

async def check_tables():
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        
        # Check if agent tables exist
        result = await conn.fetchval('''
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'agent_configs'
            );
        ''')
        
        await conn.close()
        
        if not result:
            print('⚠️ Agent tables not found. Please run migrations manually.')
            print('📝 Run: psql \$DATABASE_URL -f /app/../migrations/001_create_agent_tables.sql')
        else:
            print('✅ Database schema validated')
            
    except Exception as e:
        print(f'❌ Database check failed: {e}')
        sys.exit(1)

asyncio.run(check_tables())
"
        fi
    else
        echo "❌ Database not available"
        exit 1
    fi
}

# Function to check Redis connectivity
check_redis() {
    if [ -n "$REDIS_URL" ]; then
        echo "🔴 Checking Redis connectivity..."
        
        # Extract Redis components
        REDIS_HOST=$(echo "$REDIS_URL" | sed 's/.*@\([^:]*\):.*/\1/')
        REDIS_PORT=$(echo "$REDIS_URL" | sed 's/.*:\([0-9]*\).*/\1/')
        
        if check_service "Redis" "$REDIS_HOST" "$REDIS_PORT"; then
            echo "✅ Redis connectivity confirmed"
        else
            echo "⚠️ Redis not available - caching will be disabled"
        fi
    else
        echo "ℹ️ Redis URL not configured - caching disabled"
    fi
}

# Function to validate configuration
validate_config() {
    echo "⚙️ Validating configuration..."
    
    # Check required environment variables
    required_vars=(
        "OPENAI_API_KEY"
        "CHATWOOT_API_TOKEN" 
        "DATABASE_URL"
    )
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            echo "❌ Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Validate configuration file
    if [ ! -f "$AGENT_CONFIG_PATH" ]; then
        echo "⚠️ Agent config file not found at $AGENT_CONFIG_PATH"
        echo "📝 Using default configuration"
    else
        echo "✅ Agent configuration loaded from $AGENT_CONFIG_PATH"
    fi
    
    # Check CSV data directory
    if [ -d "/app/config/data" ]; then
        csv_count=$(find /app/config/data -name "*.csv" | wc -l)
        echo "📊 Found $csv_count CSV files for spreadsheet tool"
    else
        echo "ℹ️ CSV data directory not found - spreadsheet tool will be limited"
    fi
    
    echo "✅ Configuration validation complete"
}

# Function to setup logging directory
setup_logging() {
    echo "📝 Setting up logging..."
    
    # Create logs directory if it doesn't exist
    mkdir -p /app/logs
    
    # Ensure proper permissions
    chown -R appuser:appuser /app/logs
    
    echo "✅ Logging configured"
}

# Function to run health check
health_check() {
    echo "🏥 Running initial health check..."
    
    # Simple Python import test
    python -c "
import sys
try:
    from src.main import app
    from src.agent import ChatwootAgent
    from src.tools.rag import retrieve_relevant_context
    from src.tools.spreadsheet import query_spreadsheet_data
    print('✅ All imports successful')
except Exception as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
" || exit 1
    
    echo "✅ Health check passed"
}

# Main execution flow
main() {
    echo "🚀 Starting Chatwoot Agent Service initialization..."
    
    # Set timezone
    export TZ=${TZ:-UTC}
    
    # Run all checks
    validate_config
    setup_logging
    check_database
    check_redis
    health_check
    
    echo "✅ All checks passed - starting application..."
    echo "🎯 Agent service will be available at http://localhost:8082"
    echo "📊 Health endpoint: http://localhost:8082/health"
    echo "📋 Webhook endpoint: http://localhost:8082/webhook"
    echo ""
    
    # Start the application
    exec "$@"
}

# Handle signals for graceful shutdown
trap 'echo "🛑 Received shutdown signal"; exit 0' SIGTERM SIGINT

# Run main function with all arguments
main "$@"