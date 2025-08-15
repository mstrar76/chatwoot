"""
Tests for the configuration management system.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import pytest
from pydantic import ValidationError

from src.utils.config import (
    DatabaseConfig, ChatwootConfig, RedisConfig, OpenAIConfig, AppConfig,
    ConfigManager, get_config_manager, get_config, reload_config
)
from src.models.schemas import AgentConfig, InboxConfig


class TestDatabaseConfig:
    """Tests for DatabaseConfig model."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DatabaseConfig()
        assert config.host == "omnineural_postgres"
        assert config.port == 5432
        assert config.database == "omnicore"
        assert config.username == "omniadmin"
        assert config.password == "omni4518pgdb"
        assert config.pool_min_size == 5
        assert config.pool_max_size == 20
    
    def test_connection_string_property(self):
        """Test connection string generation."""
        config = DatabaseConfig(
            host="localhost",
            port=5433,
            database="testdb",
            username="testuser",
            password="testpass"
        )
        expected = "postgresql://testuser:testpass@localhost:5433/testdb"
        assert config.connection_string == expected
    
    def test_asyncpg_dsn_property(self):
        """Test AsyncPG DSN generation."""
        config = DatabaseConfig(
            host="localhost",
            port=5433,
            database="testdb",
            username="testuser",
            password="testpass"
        )
        expected = "postgresql://testuser:testpass@localhost:5433/testdb"
        assert config.asyncpg_dsn == expected
    
    def test_custom_pool_settings(self):
        """Test custom pool configuration."""
        config = DatabaseConfig(
            pool_min_size=2,
            pool_max_size=10,
            pool_max_queries=1000,
            pool_max_inactive_connection_lifetime=60.0
        )
        assert config.pool_min_size == 2
        assert config.pool_max_size == 10
        assert config.pool_max_queries == 1000
        assert config.pool_max_inactive_connection_lifetime == 60.0


class TestChatwootConfig:
    """Tests for ChatwootConfig model."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ChatwootConfig(api_token="test_token")
        assert config.base_url == "http://omnineural_chatwoot:3000"
        assert config.api_token == "test_token"
        assert config.webhook_secret is None
        assert config.rate_limit_per_minute == 60
        assert config.timeout_seconds == 30
        assert config.max_retries == 3
        assert config.backoff_factor == 0.3
    
    def test_api_token_validation(self):
        """Test API token validation."""
        # Should raise error for empty token
        with pytest.raises(ValidationError, match="Chatwoot API token is required"):
            ChatwootConfig(api_token="")
        
        # Should raise error for missing token
        with pytest.raises(ValidationError):
            ChatwootConfig()
    
    def test_base_url_validation(self):
        """Test base URL validation."""
        # Should raise error for invalid URL
        with pytest.raises(ValidationError, match="Chatwoot base URL must start with"):
            ChatwootConfig(api_token="test", base_url="invalid-url")
        
        # Should work with http
        config = ChatwootConfig(api_token="test", base_url="http://example.com")
        assert config.base_url == "http://example.com"
        
        # Should work with https
        config = ChatwootConfig(api_token="test", base_url="https://example.com")
        assert config.base_url == "https://example.com"
    
    def test_base_url_trailing_slash_removal(self):
        """Test trailing slash removal from base URL."""
        config = ChatwootConfig(
            api_token="test",
            base_url="http://example.com/"
        )
        assert config.base_url == "http://example.com"
        
        config = ChatwootConfig(
            api_token="test",
            base_url="https://example.com//"
        )
        assert config.base_url == "https://example.com"


class TestRedisConfig:
    """Tests for RedisConfig model."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = RedisConfig()
        assert config.host == "omnineural_redis"
        assert config.port == 6379
        assert config.database == 0
        assert config.password is None
        assert config.pool_max_connections == 10
    
    def test_connection_string_without_password(self):
        """Test connection string without password."""
        config = RedisConfig(
            host="localhost",
            port=6380,
            database=1
        )
        expected = "redis://localhost:6380/1"
        assert config.connection_string == expected
    
    def test_connection_string_with_password(self):
        """Test connection string with password."""
        config = RedisConfig(
            host="localhost",
            port=6380,
            database=1,
            password="secret123"
        )
        expected = "redis://:secret123@localhost:6380/1"
        assert config.connection_string == expected


class TestOpenAIConfig:
    """Tests for OpenAIConfig model."""
    
    def test_api_key_validation(self):
        """Test API key validation."""
        # Should raise error for empty key
        with pytest.raises(ValidationError, match="OpenAI API key is required"):
            OpenAIConfig(api_key="")
        
        # Should raise error for missing key
        with pytest.raises(ValidationError):
            OpenAIConfig()
    
    def test_valid_configuration(self):
        """Test valid configuration."""
        config = OpenAIConfig(
            api_key="sk-test123",
            base_url="https://custom-openai.com",
            organization="org-123",
            timeout_seconds=120,
            max_retries=5
        )
        
        assert config.api_key == "sk-test123"
        assert config.base_url == "https://custom-openai.com"
        assert config.organization == "org-123"
        assert config.timeout_seconds == 120
        assert config.max_retries == 5


class TestAppConfig:
    """Tests for AppConfig model."""
    
    def test_default_values(self):
        """Test default configuration values."""
        # Mock the required API keys for sub-configs
        with patch.dict(os.environ, {
            'CHATWOOT_API_TOKEN': 'test_token',
            'OPENAI_API_KEY': 'test_key'
        }):
            config = AppConfig()
            assert config.environment == "development"
            assert config.debug is False
            assert config.log_level == "INFO"
            assert config.host == "0.0.0.0"
            assert config.port == 8000
            assert config.workers == 1
            assert config.agent_enabled is True
    
    def test_environment_validation(self):
        """Test environment validation."""
        with patch.dict(os.environ, {
            'CHATWOOT_API_TOKEN': 'test_token',
            'OPENAI_API_KEY': 'test_key'
        }):
            # Should raise error for invalid environment
            with pytest.raises(ValidationError):
                AppConfig(environment="invalid")
            
            # Should work with valid environments
            for env in ['development', 'staging', 'production']:
                config = AppConfig(environment=env)
                assert config.environment == env
    
    def test_log_level_validation(self):
        """Test log level validation."""
        with patch.dict(os.environ, {
            'CHATWOOT_API_TOKEN': 'test_token',
            'OPENAI_API_KEY': 'test_key'
        }):
            # Should raise error for invalid log level
            with pytest.raises(ValidationError, match="Invalid log level"):
                AppConfig(log_level="INVALID")
            
            # Should work with valid log levels and normalize case
            config = AppConfig(log_level="debug")
            assert config.log_level == "DEBUG"


class TestConfigManager:
    """Tests for ConfigManager class."""
    
    def test_initialization(self, temp_dir):
        """Test config manager initialization."""
        manager = ConfigManager(config_dir=temp_dir)
        assert manager.config_dir == temp_dir
        assert manager._config is None
        assert manager._agent_config is None
    
    @patch.dict(os.environ, {
        'ENVIRONMENT': 'test',
        'DEBUG': 'true',
        'LOG_LEVEL': 'DEBUG',
        'DATABASE_HOST': 'test-db',
        'CHATWOOT_API_TOKEN': 'test_token',
        'OPENAI_API_KEY': 'test_key'
    })
    def test_load_from_environment(self, temp_dir):
        """Test loading configuration from environment variables."""
        manager = ConfigManager(config_dir=temp_dir)
        config = manager.load_config()
        
        assert config.environment == 'test'
        assert config.debug is True
        assert config.log_level == 'DEBUG'
        assert config.database.host == 'test-db'
        assert config.chatwoot.api_token == 'test_token'
        assert config.openai.api_key == 'test_key'
    
    def test_load_from_files(self, temp_dir):
        """Test loading configuration from files."""
        # Create agent config file
        agent_config_file = temp_dir / "agent_config.json"
        agent_config_data = {
            "enabled": True,
            "llm_provider": "openai",
            "model": "gpt-4",
            "max_tokens": 2048
        }
        
        with open(agent_config_file, 'w') as f:
            json.dump(agent_config_data, f)
        
        # Create app config file
        app_config_file = temp_dir / "app_config.json"
        app_config_data = {
            "environment": "production",
            "debug": False,
            "log_level": "WARNING"
        }
        
        with open(app_config_file, 'w') as f:
            json.dump(app_config_data, f)
        
        with patch.dict(os.environ, {
            'CHATWOOT_API_TOKEN': 'test_token',
            'OPENAI_API_KEY': 'test_key'
        }):
            manager = ConfigManager(config_dir=temp_dir)
            config = manager.load_config()
            
            # App config should be loaded from file
            assert config.environment == "production"
            assert config.debug is False
            assert config.log_level == "WARNING"
            
            # Agent config should be loaded separately
            agent_config = manager.get_agent_config()
            assert agent_config is not None
            assert agent_config.model == "gpt-4"
            assert agent_config.max_tokens == 2048
    
    def test_load_inbox_config(self, temp_dir):
        """Test loading inbox-specific configuration."""
        # Create inbox configs directory
        inbox_configs_dir = temp_dir / "inbox_configs"
        inbox_configs_dir.mkdir()
        
        # Create inbox config file
        inbox_config_file = inbox_configs_dir / "inbox_42.json"
        inbox_config_data = {
            "enabled": True,
            "language": "pt",
            "persona": "helpful assistant"
        }
        
        with open(inbox_config_file, 'w') as f:
            json.dump(inbox_config_data, f)
        
        manager = ConfigManager(config_dir=temp_dir)
        inbox_config = manager.load_inbox_config(42)
        
        assert inbox_config is not None
        assert inbox_config.inbox_id == 42
        assert inbox_config.language == "pt"
        assert inbox_config.persona == "helpful assistant"
    
    def test_load_inbox_config_not_found(self, temp_dir):
        """Test loading non-existent inbox configuration."""
        manager = ConfigManager(config_dir=temp_dir)
        inbox_config = manager.load_inbox_config(999)
        
        assert inbox_config is None
    
    def test_config_caching(self, temp_dir):
        """Test configuration caching."""
        with patch.dict(os.environ, {
            'CHATWOOT_API_TOKEN': 'test_token',
            'OPENAI_API_KEY': 'test_key'
        }):
            manager = ConfigManager(config_dir=temp_dir)
            
            # First load
            config1 = manager.load_config()
            
            # Second load should return cached config
            config2 = manager.load_config()
            assert config1 is config2
            
            # Force reload should return new config
            config3 = manager.load_config(force_reload=True)
            assert config1 is not config3
    
    def test_invalid_config_file(self, temp_dir):
        """Test handling of invalid configuration files."""
        # Create invalid JSON file
        agent_config_file = temp_dir / "agent_config.json"
        
        with open(agent_config_file, 'w') as f:
            f.write("invalid json content")
        
        with patch.dict(os.environ, {
            'CHATWOOT_API_TOKEN': 'test_token',
            'OPENAI_API_KEY': 'test_key'
        }):
            manager = ConfigManager(config_dir=temp_dir)
            
            # Should not crash, but log warning
            with patch('src.utils.logging.get_logger') as mock_logger:
                mock_log = MagicMock()
                mock_logger.return_value = mock_log
                
                config = manager.load_config()
                
                # Should have logged warning about invalid file
                mock_log.warning.assert_called()


class TestGlobalConfigFunctions:
    """Tests for global configuration functions."""
    
    @patch('src.utils.config.ConfigManager')
    def test_get_config_manager(self, mock_manager_class):
        """Test get_config_manager function."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        
        # Clear global variable
        import src.utils.config
        src.utils.config._config_manager = None
        
        manager = get_config_manager()
        assert manager is mock_manager
        mock_manager_class.assert_called_once()
        
        # Second call should return same instance
        manager2 = get_config_manager()
        assert manager2 is mock_manager
        assert mock_manager_class.call_count == 1
    
    @patch('src.utils.config.get_config_manager')
    def test_get_config(self, mock_get_manager):
        """Test get_config function."""
        mock_manager = MagicMock()
        mock_config = MagicMock()
        mock_manager.load_config.return_value = mock_config
        mock_get_manager.return_value = mock_manager
        
        # Clear cache
        get_config.cache_clear()
        
        config = get_config()
        assert config is mock_config
        mock_manager.load_config.assert_called_once()
    
    @patch('src.utils.config.get_config_manager')
    def test_reload_config(self, mock_get_manager):
        """Test reload_config function."""
        mock_manager = MagicMock()
        mock_config = MagicMock()
        mock_manager.load_config.return_value = mock_config
        mock_get_manager.return_value = mock_manager
        
        config = reload_config()
        assert config is mock_config
        mock_manager.load_config.assert_called_with(force_reload=True)


class TestConfigurationIntegration:
    """Integration tests for configuration system."""
    
    @patch.dict(os.environ, {
        'ENVIRONMENT': 'production',
        'DEBUG': 'false',
        'DATABASE_HOST': 'prod-db.example.com',
        'DATABASE_PORT': '5432',
        'DATABASE_NAME': 'chatwoot_prod',
        'DATABASE_USER': 'prod_user',
        'DATABASE_PASSWORD': 'prod_password',
        'CHATWOOT_BASE_URL': 'https://chatwoot.example.com',
        'CHATWOOT_API_TOKEN': 'prod_api_token_123',
        'CHATWOOT_WEBHOOK_SECRET': 'prod_webhook_secret',
        'REDIS_HOST': 'redis.example.com',
        'REDIS_PASSWORD': 'redis_password',
        'OPENAI_API_KEY': 'sk-prod123456789',
        'OPENAI_ORGANIZATION': 'org-prod123'
    })
    def test_full_configuration_load(self, temp_dir):
        """Test loading complete configuration from environment."""
        manager = ConfigManager(config_dir=temp_dir)
        config = manager.load_config()
        
        # App config
        assert config.environment == 'production'
        assert config.debug is False
        
        # Database config
        assert config.database.host == 'prod-db.example.com'
        assert config.database.port == 5432
        assert config.database.database == 'chatwoot_prod'
        assert config.database.username == 'prod_user'
        assert config.database.password == 'prod_password'
        
        # Chatwoot config
        assert config.chatwoot.base_url == 'https://chatwoot.example.com'
        assert config.chatwoot.api_token == 'prod_api_token_123'
        assert config.chatwoot.webhook_secret == 'prod_webhook_secret'
        
        # Redis config
        assert config.redis.host == 'redis.example.com'
        assert config.redis.password == 'redis_password'
        
        # OpenAI config
        assert config.openai.api_key == 'sk-prod123456789'
        assert config.openai.organization == 'org-prod123'
    
    def test_dotenv_loading(self, temp_dir):
        """Test loading from .env file."""
        # Create .env file
        env_file = temp_dir.parent / ".env"
        env_content = """
ENVIRONMENT=staging
DATABASE_HOST=staging-db.example.com
CHATWOOT_API_TOKEN=staging_token_123
OPENAI_API_KEY=sk-staging123
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        # Mock load_dotenv to simulate file loading
        with patch('src.utils.config.load_dotenv') as mock_load_dotenv:
            with patch.dict(os.environ, {
                'ENVIRONMENT': 'staging',
                'DATABASE_HOST': 'staging-db.example.com',
                'CHATWOOT_API_TOKEN': 'staging_token_123',
                'OPENAI_API_KEY': 'sk-staging123'
            }):
                manager = ConfigManager(config_dir=temp_dir)
                config = manager.load_config()
                
                mock_load_dotenv.assert_called_once()
                assert config.environment == 'staging'
                assert config.database.host == 'staging-db.example.com'


class TestErrorHandling:
    """Tests for configuration error handling."""
    
    def test_missing_required_config(self):
        """Test error handling for missing required configuration."""
        with patch.dict(os.environ, {}, clear=True):
            manager = ConfigManager()
            
            with pytest.raises(ValueError, match="Configuration validation failed"):
                manager.load_config()
    
    def test_invalid_environment_variable(self):
        """Test handling of invalid environment variable values."""
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'invalid_env',
            'CHATWOOT_API_TOKEN': 'test_token',
            'OPENAI_API_KEY': 'test_key'
        }):
            manager = ConfigManager()
            
            with pytest.raises(ValueError, match="Configuration validation failed"):
                manager.load_config()
    
    def test_file_permission_error(self, temp_dir):
        """Test handling of file permission errors."""
        # Create a file with restricted permissions (simulation)
        config_file = temp_dir / "agent_config.json"
        config_file.write_text("{}")
        
        # Mock open to raise PermissionError
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with patch.dict(os.environ, {
                'CHATWOOT_API_TOKEN': 'test_token',
                'OPENAI_API_KEY': 'test_key'
            }):
                manager = ConfigManager(config_dir=temp_dir)
                
                # Should not crash, but log error
                with patch('src.utils.logging.get_logger') as mock_logger:
                    mock_log = MagicMock()
                    mock_logger.return_value = mock_log
                    
                    config = manager.load_config()
                    
                    # Should have logged warning about file access
                    mock_log.warning.assert_called()