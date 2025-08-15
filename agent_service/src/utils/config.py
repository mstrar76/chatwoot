"""
Configuration management for the Chatwoot Agent Service.

Handles loading configuration from environment variables, files, and database,
with validation and type safety through Pydantic models.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from functools import lru_cache

from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

from ..models.schemas import AgentConfig, InboxConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    host: str = Field(default="omnineural_postgres")
    port: int = Field(default=5432)
    database: str = Field(default="omnicore")
    username: str = Field(default="omniadmin")
    password: str = Field(default="omni4518pgdb")
    pool_min_size: int = Field(default=5)
    pool_max_size: int = Field(default=20)
    pool_max_queries: int = Field(default=50000)
    pool_max_inactive_connection_lifetime: float = Field(default=300.0)
    
    @property
    def connection_string(self) -> str:
        """Get the full PostgreSQL connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def asyncpg_dsn(self) -> str:
        """Get AsyncPG-compatible DSN."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class ChatwootConfig(BaseModel):
    """Chatwoot API configuration."""
    base_url: str = Field(default="http://omnineural_chatwoot:3000")
    api_token: str = Field(default="", description="API access token")
    webhook_secret: Optional[str] = Field(default=None, description="Webhook validation secret")
    rate_limit_per_minute: int = Field(default=60, description="API rate limit")
    timeout_seconds: int = Field(default=30, description="Request timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    backoff_factor: float = Field(default=0.3, description="Exponential backoff factor")
    
    @validator('api_token')
    def validate_api_token(cls, v):
        """Ensure API token is provided."""
        if not v:
            raise ValueError("Chatwoot API token is required")
        return v
    
    @validator('base_url')
    def validate_base_url(cls, v):
        """Ensure base URL is properly formatted."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Chatwoot base URL must start with http:// or https://")
        return v.rstrip('/')


class RedisConfig(BaseModel):
    """Redis configuration for caching and rate limiting."""
    host: str = Field(default="omnineural_redis")
    port: int = Field(default=6379)
    database: int = Field(default=0)
    password: Optional[str] = Field(default=None)
    pool_max_connections: int = Field(default=10)
    
    @property
    def connection_string(self) -> str:
        """Get Redis connection string."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.database}"


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str = Field(default="", description="OpenAI API key")
    base_url: Optional[str] = Field(default=None, description="Custom base URL")
    organization: Optional[str] = Field(default=None, description="OpenAI organization")
    timeout_seconds: int = Field(default=60, description="Request timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    @validator('api_key')
    def validate_api_key(cls, v):
        """Ensure API key is provided."""
        if not v:
            raise ValueError("OpenAI API key is required")
        return v


class AppConfig(BaseModel):
    """Main application configuration."""
    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)
    
    # Service configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    chatwoot: ChatwootConfig = Field(default_factory=ChatwootConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    
    # Agent settings
    agent_enabled: bool = Field(default=True)
    webhook_path: str = Field(default="/webhook/chatwoot")
    health_check_path: str = Field(default="/health")
    
    # Performance settings
    max_concurrent_requests: int = Field(default=100)
    request_timeout_seconds: int = Field(default=300)
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment value."""
        if v not in ['development', 'staging', 'production']:
            raise ValueError("Environment must be development, staging, or production")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        if v.upper() not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ValueError("Invalid log level")
        return v.upper()


class ConfigManager:
    """
    Configuration manager that loads and validates configuration from multiple sources.
    
    Priority order:
    1. Environment variables
    2. Configuration files
    3. Default values
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent.parent / "config"
        self._config: Optional[AppConfig] = None
        self._agent_config: Optional[AgentConfig] = None
    
    def load_config(self, force_reload: bool = False) -> AppConfig:
        """
        Load application configuration from all sources.
        
        Args:
            force_reload: Force reload configuration from sources
            
        Returns:
            Validated AppConfig instance
        """
        if self._config is not None and not force_reload:
            return self._config
        
        # Load environment variables from .env file if it exists
        env_file = self.config_dir.parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        
        # Load configuration from environment and files
        config_data = self._load_from_environment()
        config_data.update(self._load_from_files())
        
        # Validate and create configuration
        try:
            self._config = AppConfig(**config_data)
            logger.info("Configuration loaded successfully", 
                       environment=self._config.environment,
                       log_level=self._config.log_level)
            return self._config
        except Exception as e:
            logger.error("Failed to load configuration", error=str(e))
            raise ValueError(f"Configuration validation failed: {e}")
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # Application settings
        if os.getenv('ENVIRONMENT'):
            config['environment'] = os.getenv('ENVIRONMENT')
        if os.getenv('DEBUG'):
            config['debug'] = os.getenv('DEBUG').lower() in ('true', '1', 'yes')
        if os.getenv('LOG_LEVEL'):
            config['log_level'] = os.getenv('LOG_LEVEL')
        if os.getenv('HOST'):
            config['host'] = os.getenv('HOST')
        if os.getenv('PORT'):
            config['port'] = int(os.getenv('PORT'))
        
        # Database configuration
        db_config = {}
        if os.getenv('DATABASE_HOST'):
            db_config['host'] = os.getenv('DATABASE_HOST')
        if os.getenv('DATABASE_PORT'):
            db_config['port'] = int(os.getenv('DATABASE_PORT'))
        if os.getenv('DATABASE_NAME'):
            db_config['database'] = os.getenv('DATABASE_NAME')
        if os.getenv('DATABASE_USER'):
            db_config['username'] = os.getenv('DATABASE_USER')
        if os.getenv('DATABASE_PASSWORD'):
            db_config['password'] = os.getenv('DATABASE_PASSWORD')
        if db_config:
            config['database'] = db_config
        
        # Chatwoot configuration
        chatwoot_config = {}
        if os.getenv('CHATWOOT_BASE_URL'):
            chatwoot_config['base_url'] = os.getenv('CHATWOOT_BASE_URL')
        if os.getenv('CHATWOOT_API_TOKEN'):
            chatwoot_config['api_token'] = os.getenv('CHATWOOT_API_TOKEN')
        if os.getenv('CHATWOOT_WEBHOOK_SECRET'):
            chatwoot_config['webhook_secret'] = os.getenv('CHATWOOT_WEBHOOK_SECRET')
        if chatwoot_config:
            config['chatwoot'] = chatwoot_config
        
        # Redis configuration
        redis_config = {}
        if os.getenv('REDIS_HOST'):
            redis_config['host'] = os.getenv('REDIS_HOST')
        if os.getenv('REDIS_PORT'):
            redis_config['port'] = int(os.getenv('REDIS_PORT'))
        if os.getenv('REDIS_PASSWORD'):
            redis_config['password'] = os.getenv('REDIS_PASSWORD')
        if redis_config:
            config['redis'] = redis_config
        
        # OpenAI configuration
        openai_config = {}
        if os.getenv('OPENAI_API_KEY'):
            openai_config['api_key'] = os.getenv('OPENAI_API_KEY')
        if os.getenv('OPENAI_BASE_URL'):
            openai_config['base_url'] = os.getenv('OPENAI_BASE_URL')
        if os.getenv('OPENAI_ORGANIZATION'):
            openai_config['organization'] = os.getenv('OPENAI_ORGANIZATION')
        if openai_config:
            config['openai'] = openai_config
        
        return config
    
    def _load_from_files(self) -> Dict[str, Any]:
        """Load configuration from JSON files."""
        config = {}
        
        # Load agent configuration file
        agent_config_file = self.config_dir / "agent_config.json"
        if agent_config_file.exists():
            try:
                with open(agent_config_file, 'r') as f:
                    agent_data = json.load(f)
                    # Store agent config separately
                    self._agent_config = AgentConfig(**agent_data)
                    logger.debug("Agent configuration loaded from file")
            except Exception as e:
                logger.warning("Failed to load agent configuration file", 
                             file=str(agent_config_file), error=str(e))
        
        # Load application configuration file
        app_config_file = self.config_dir / "app_config.json"
        if app_config_file.exists():
            try:
                with open(app_config_file, 'r') as f:
                    config.update(json.load(f))
                    logger.debug("Application configuration loaded from file")
            except Exception as e:
                logger.warning("Failed to load application configuration file",
                             file=str(app_config_file), error=str(e))
        
        return config
    
    def get_agent_config(self) -> Optional[AgentConfig]:
        """Get agent configuration if loaded from file."""
        return self._agent_config
    
    def load_inbox_config(self, inbox_id: int) -> Optional[InboxConfig]:
        """
        Load inbox-specific configuration.
        
        Args:
            inbox_id: Chatwoot inbox ID
            
        Returns:
            InboxConfig if found, None otherwise
        """
        config_file = self.config_dir / "inbox_configs" / f"inbox_{inbox_id}.json"
        
        if not config_file.exists():
            logger.debug("Inbox configuration file not found", 
                        inbox_id=inbox_id, file=str(config_file))
            return None
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                config_data['inbox_id'] = inbox_id
                return InboxConfig(**config_data)
        except Exception as e:
            logger.error("Failed to load inbox configuration",
                        inbox_id=inbox_id, file=str(config_file), error=str(e))
            return None
    
    @property
    def config(self) -> AppConfig:
        """Get the current configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Get the application configuration (cached).
    
    Returns:
        Validated AppConfig instance
    """
    return get_config_manager().load_config()


def reload_config() -> AppConfig:
    """
    Force reload the configuration from all sources.
    
    Returns:
        Newly loaded AppConfig instance
    """
    # Clear cache
    get_config.cache_clear()
    
    # Force reload
    return get_config_manager().load_config(force_reload=True)


def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return get_config().database


def get_chatwoot_config() -> ChatwootConfig:
    """Get Chatwoot API configuration."""
    return get_config().chatwoot


def get_redis_config() -> RedisConfig:
    """Get Redis configuration."""
    return get_config().redis


def get_openai_config() -> OpenAIConfig:
    """Get OpenAI API configuration."""
    return get_config().openai