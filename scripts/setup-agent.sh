#!/bin/bash

# Chatwoot Agent Service Setup Script
# Automates the deployment and configuration of the agent service

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AGENT_DIR="$PROJECT_ROOT/agent_service"

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required commands exist
check_dependencies() {
    print_status "Checking dependencies..."
    
    local deps=("docker" "docker-compose" "psql" "curl")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        print_error "Please install them before running this script."
        exit 1
    fi
    
    print_success "All dependencies found"
}

# Check if environment variables are set
check_environment() {
    print_status "Checking environment variables..."
    
    local required_vars=(
        "OPENAI_API_KEY"
        "CHATWOOT_API_TOKEN"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        print_error "Missing required environment variables: ${missing_vars[*]}"
        print_error "Please set them in your environment or .env file."
        exit 1
    fi
    
    print_success "Environment variables configured"
}

# Check if .env file exists, create from template if not
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f "$AGENT_DIR/.env" ]; then
        if [ -f "$AGENT_DIR/.env.example" ]; then
            print_warning ".env file not found, creating from template..."
            cp "$AGENT_DIR/.env.example" "$AGENT_DIR/.env"
            print_warning "Please edit $AGENT_DIR/.env with your actual values"
            print_warning "Required variables: OPENAI_API_KEY, CHATWOOT_API_TOKEN"
        else
            print_error ".env.example template not found"
            exit 1
        fi
    else
        print_success "Environment file exists"
    fi
}

# Check database connectivity and run migrations
setup_database() {
    print_status "Setting up database..."
    
    # Extract database URL from environment or use default
    DATABASE_URL=${DATABASE_URL:-"postgres://omniadmin:omni4518pgdb@omnineural_postgres:5432/omnicore"}
    
    # Check database connectivity
    if ! psql "$DATABASE_URL" -c "SELECT 1;" &> /dev/null; then
        print_error "Cannot connect to database at $DATABASE_URL"
        print_error "Please ensure PostgreSQL is running and accessible"
        exit 1
    fi
    
    print_success "Database connection established"
    
    # Check if pgvector extension is installed
    if ! psql "$DATABASE_URL" -c "SELECT 1 FROM pg_extension WHERE extname = 'vector';" | grep -q "1"; then
        print_warning "pgvector extension not found, attempting to install..."
        if ! psql "$DATABASE_URL" -c "CREATE EXTENSION IF NOT EXISTS vector;" &> /dev/null; then
            print_error "Failed to install pgvector extension"
            print_error "Please install pgvector manually or contact your database administrator"
            exit 1
        fi
        print_success "pgvector extension installed"
    else
        print_success "pgvector extension is available"
    fi
    
    # Run database migrations
    print_status "Running database migrations..."
    MIGRATION_FILE="$PROJECT_ROOT/migrations/001_create_agent_tables.sql"
    
    if [ ! -f "$MIGRATION_FILE" ]; then
        print_error "Migration file not found: $MIGRATION_FILE"
        exit 1
    fi
    
    if psql "$DATABASE_URL" -f "$MIGRATION_FILE" &> /dev/null; then
        print_success "Database migrations completed"
    else
        print_warning "Migration may have already been applied (this is usually okay)"
    fi
    
    # Verify tables exist
    if psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM agent_configs;" &> /dev/null; then
        print_success "Agent tables verified"
    else
        print_error "Agent tables not found after migration"
        exit 1
    fi
}

# Build and start the agent service
build_and_start() {
    print_status "Building and starting agent service..."
    
    cd "$PROJECT_ROOT"
    
    # Build the agent service
    print_status "Building Docker image..."
    if docker-compose -f docker-compose.agent.yml build; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
    
    # Start the service
    print_status "Starting agent service..."
    if docker-compose -f docker-compose.agent.yml up -d; then
        print_success "Agent service started"
    else
        print_error "Failed to start agent service"
        exit 1
    fi
    
    # Wait for service to be ready
    print_status "Waiting for service to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:8082/health > /dev/null 2>&1; then
            print_success "Agent service is healthy"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            print_error "Agent service failed to become healthy after $max_attempts attempts"
            print_error "Check logs with: docker logs omnineural_agent"
            exit 1
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
}

# Test the agent service
test_service() {
    print_status "Testing agent service..."
    
    # Test health endpoint
    if curl -s http://localhost:8082/health | grep -q '"status":"healthy"'; then
        print_success "Health check passed"
    else
        print_error "Health check failed"
        return 1
    fi
    
    # Test webhook endpoint
    print_status "Testing webhook endpoint..."
    local test_payload='{"event": "test", "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}'
    
    if curl -s -X POST http://localhost:8082/webhook \
        -H "Content-Type: application/json" \
        -d "$test_payload" | grep -q "processed"; then
        print_success "Webhook endpoint responding"
    else
        print_warning "Webhook endpoint test failed (this may be expected for test payload)"
    fi
    
    # Test configuration endpoints
    if curl -s http://localhost:8082/config/global > /dev/null; then
        print_success "Configuration endpoint accessible"
    else
        print_warning "Configuration endpoint not responding"
    fi
    
    # Test metrics endpoint
    if curl -s http://localhost:8082/metrics > /dev/null; then
        print_success "Metrics endpoint accessible"
    else
        print_warning "Metrics endpoint not responding"
    fi
}

# Display service information
show_service_info() {
    print_success "Agent service setup completed!"
    echo
    echo "Service Information:"
    echo "  - Health Check:    http://localhost:8082/health"
    echo "  - Webhook URL:     http://localhost:8082/webhook"
    echo "  - Metrics:         http://localhost:8082/metrics"
    echo "  - Configuration:   http://localhost:8082/config/global"
    echo
    echo "Docker Information:"
    echo "  - Container:       omnineural_agent"
    echo "  - Network:         omnineural-network"
    echo "  - Logs:           docker logs omnineural_agent"
    echo
    echo "Next Steps:"
    echo "  1. Configure Chatwoot webhook to point to: http://omnineural_agent:8082/webhook"
    echo "  2. Set up API Channel inbox in Chatwoot"
    echo "  3. Create automation rules for agent processing"
    echo "  4. Test with a sample message"
    echo
    echo "For detailed configuration instructions, see:"
    echo "  docs/Chatwoot-Agent-Integration-Guide.md"
}

# Show help message
show_help() {
    echo "Chatwoot Agent Service Setup Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --help, -h        Show this help message"
    echo "  --check-only      Only check dependencies and environment"
    echo "  --db-only         Only setup database"
    echo "  --build-only      Only build and start service"
    echo "  --test-only       Only test the service"
    echo "  --restart         Restart the agent service"
    echo "  --stop            Stop the agent service"
    echo "  --logs            Show agent service logs"
    echo "  --status          Show agent service status"
    echo
    echo "Environment Variables:"
    echo "  OPENAI_API_KEY       Required: OpenAI API key"
    echo "  CHATWOOT_API_TOKEN   Required: Chatwoot API access token"
    echo "  DATABASE_URL         Optional: PostgreSQL connection string"
    echo
    echo "Examples:"
    echo "  $0                   # Full setup"
    echo "  $0 --check-only     # Just check dependencies"
    echo "  $0 --restart        # Restart the service"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --check-only)
        check_dependencies
        check_environment
        setup_environment
        exit 0
        ;;
    --db-only)
        check_dependencies
        setup_database
        exit 0
        ;;
    --build-only)
        check_dependencies
        build_and_start
        exit 0
        ;;
    --test-only)
        test_service
        exit 0
        ;;
    --restart)
        print_status "Restarting agent service..."
        cd "$PROJECT_ROOT"
        docker-compose -f docker-compose.agent.yml restart
        print_success "Agent service restarted"
        exit 0
        ;;
    --stop)
        print_status "Stopping agent service..."
        cd "$PROJECT_ROOT"
        docker-compose -f docker-compose.agent.yml down
        print_success "Agent service stopped"
        exit 0
        ;;
    --logs)
        print_status "Showing agent service logs..."
        docker logs omnineural_agent --tail 100 --follow
        exit 0
        ;;
    --status)
        print_status "Checking agent service status..."
        docker-compose -f "$PROJECT_ROOT/docker-compose.agent.yml" ps
        echo
        if curl -s http://localhost:8082/health > /dev/null 2>&1; then
            print_success "Service is healthy"
        else
            print_error "Service is not responding"
        fi
        exit 0
        ;;
    "")
        # Full setup - continue with main flow
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac

# Main execution flow
main() {
    echo "====================================="
    echo "Chatwoot Agent Service Setup"
    echo "====================================="
    echo
    
    check_dependencies
    check_environment
    setup_environment
    setup_database
    build_and_start
    
    # Give service a moment to fully initialize
    sleep 5
    
    test_service
    show_service_info
    
    print_success "Setup completed successfully!"
}

# Run main function
main "$@"