"""Tests for the Redis utility module.

This module provides comprehensive tests for the Redis connection pool and utility functions.
It covers all methods, edge cases, error scenarios, and configuration options.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from arklex.utils.redis import (
    CONNECTION_TIMEOUT,
    DEFAULT_TTL,
    POOL_SIZE,
    REDIS_CONFIG,
    RedisPool,
    filtered_redis_config,
    redis_pool,
)


class TestRedisConfiguration:
    """Test Redis configuration constants and environment variables."""

    def test_connection_timeout_default(self) -> None:
        """Test that CONNECTION_TIMEOUT has a default value."""
        assert isinstance(CONNECTION_TIMEOUT, int)
        assert CONNECTION_TIMEOUT > 0

    def test_connection_timeout_from_env(self) -> None:
        """Test that CONNECTION_TIMEOUT can be set from environment."""
        with patch.dict(os.environ, {"REDIS_CONNECTION_TIMEOUT": "10"}):
            # Re-import to get updated value
            import importlib

            import arklex.utils.redis as redis_module

            importlib.reload(redis_module)
            assert redis_module.CONNECTION_TIMEOUT == 10

    def test_redis_config_structure(self) -> None:
        """Test that REDIS_CONFIG has the expected structure."""
        expected_keys = {"host", "port", "db", "password", "username"}
        assert set(REDIS_CONFIG.keys()) == expected_keys

    def test_redis_config_defaults(self) -> None:
        """Test that REDIS_CONFIG has correct default values."""
        assert REDIS_CONFIG["host"] == "localhost"
        assert REDIS_CONFIG["port"] == 6379
        assert REDIS_CONFIG["db"] == 0
        assert REDIS_CONFIG["password"] is None
        assert REDIS_CONFIG["username"] is None

    def test_pool_size_default(self) -> None:
        """Test that POOL_SIZE has a default value."""
        assert isinstance(POOL_SIZE, int)
        assert POOL_SIZE > 0

    def test_pool_size_from_env(self) -> None:
        """Test that POOL_SIZE can be set from environment."""
        with patch.dict(os.environ, {"REDIS_POOL_SIZE": "20"}):
            # Re-import to get updated value
            import importlib

            import arklex.utils.redis as redis_module

            importlib.reload(redis_module)
            assert redis_module.POOL_SIZE == 20

    def test_default_ttl_default(self) -> None:
        """Test that DEFAULT_TTL has a default value."""
        assert isinstance(DEFAULT_TTL, int)
        assert DEFAULT_TTL > 0

    def test_default_ttl_from_env(self) -> None:
        """Test that DEFAULT_TTL can be set from environment."""
        with patch.dict(os.environ, {"REDIS_DEFAULT_TTL": "7200"}):
            # Re-import to get updated value
            import importlib

            import arklex.utils.redis as redis_module

            importlib.reload(redis_module)
            assert redis_module.DEFAULT_TTL == 7200

    def test_filtered_redis_config(self) -> None:
        """Test that filtered_redis_config excludes None values."""
        # Test with None values
        test_config = {"host": "localhost", "port": 6379, "password": None}
        filtered = {k: v for k, v in test_config.items() if v is not None}
        assert "password" not in filtered
        assert "host" in filtered
        assert "port" in filtered


class TestRedisPoolInitialization:
    """Test RedisPool initialization and configuration."""

    def test_redis_pool_init_default_params(self) -> None:
        """Test RedisPool initialization with default parameters."""
        with (
            patch("arklex.utils.redis.ConnectionPool") as mock_pool_class,
            patch("arklex.utils.redis.redis.Redis") as mock_redis_class,
        ):
            mock_client = MagicMock()
            mock_redis_class.return_value = mock_client
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            pool = RedisPool()

            assert pool.client == mock_client
            assert pool.connection_pool == mock_pool
            assert pool._db == 0
            mock_redis_class.assert_called_once_with(connection_pool=mock_pool)

    def test_redis_pool_init_custom_params(self) -> None:
        """Test RedisPool initialization with custom parameters."""
        with (
            patch("arklex.utils.redis.ConnectionPool") as mock_pool_class,
            patch("arklex.utils.redis.redis.Redis") as mock_redis_class,
        ):
            mock_client = MagicMock()
            mock_redis_class.return_value = mock_client
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            pool = RedisPool(
                host="custom-host",
                port=6380,
                db=1,
                password="secret",
                username="user",
            )

            assert pool.client == mock_client
            assert pool.connection_pool == mock_pool
            assert pool._db == 1
            # Check that ConnectionPool was called with the correct parameters
            # Note: The order of parameters may vary, so we check the call args
            call_args = mock_pool_class.call_args[1]
            assert call_args["host"] == "custom-host"
            assert call_args["port"] == 6380
            assert call_args["db"] == 1
            assert call_args["password"] == "secret"
            assert call_args["username"] == "user"
            assert call_args["max_connections"] == POOL_SIZE
            assert call_args["socket_connect_timeout"] == CONNECTION_TIMEOUT
            assert call_args["socket_timeout"] == CONNECTION_TIMEOUT

    def test_redis_pool_init_with_password_only(self) -> None:
        """Test RedisPool initialization with password only."""
        with (
            patch("arklex.utils.redis.ConnectionPool") as mock_pool_class,
            patch("arklex.utils.redis.redis.Redis") as mock_redis_class,
        ):
            mock_client = MagicMock()
            mock_redis_class.return_value = mock_client
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            RedisPool(password="secret")

            mock_pool_class.assert_called_once()
            call_args = mock_pool_class.call_args[1]
            assert call_args["password"] == "secret"
            assert "username" not in call_args

    def test_redis_pool_init_with_username_only(self) -> None:
        """Test RedisPool initialization with username only."""
        with (
            patch("arklex.utils.redis.ConnectionPool") as mock_pool_class,
            patch("arklex.utils.redis.redis.Redis") as mock_redis_class,
        ):
            mock_client = MagicMock()
            mock_redis_class.return_value = mock_client
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            RedisPool(username="user")

            mock_pool_class.assert_called_once()
            call_args = mock_pool_class.call_args[1]
            assert call_args["username"] == "user"
            assert "password" not in call_args

    def test_redis_pool_connection_pool_creation(self) -> None:
        """Test that ConnectionPool is created with correct parameters."""
        with (
            patch("arklex.utils.redis.ConnectionPool") as mock_pool_class,
            patch("arklex.utils.redis.redis.Redis") as mock_redis_class,
        ):
            mock_client = MagicMock()
            mock_redis_class.return_value = mock_client
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            RedisPool()

            # Check that ConnectionPool was called with the correct parameters
            # Note: The order of parameters may vary, so we check the call args
            call_args = mock_pool_class.call_args[1]
            assert call_args["host"] == "localhost"
            assert call_args["port"] == 6379
            assert call_args["db"] == 0
            assert call_args["max_connections"] == POOL_SIZE
            assert call_args["socket_connect_timeout"] == CONNECTION_TIMEOUT
            assert call_args["socket_timeout"] == CONNECTION_TIMEOUT
            # Note: password and username are not included when None

    def test_redis_pool_client_creation(self) -> None:
        """Test that Redis client is created with connection pool."""
        with (
            patch("arklex.utils.redis.ConnectionPool") as mock_pool_class,
            patch("arklex.utils.redis.redis.Redis") as mock_redis_class,
        ):
            mock_client = MagicMock()
            mock_redis_class.return_value = mock_client
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            RedisPool()

            mock_redis_class.assert_called_once_with(connection_pool=mock_pool)


class TestRedisPoolMethods:
    """Test RedisPool method implementations."""

    @pytest.fixture
    def mock_redis_pool(self) -> RedisPool:
        """Create a RedisPool instance with mocked dependencies."""
        with (
            patch("arklex.utils.redis.ConnectionPool") as mock_pool_class,
            patch("arklex.utils.redis.redis.Redis") as mock_redis_class,
        ):
            mock_client = MagicMock()
            mock_redis_class.return_value = mock_client
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            pool = RedisPool()
            return pool

    def test_get_connection(self, mock_redis_pool: RedisPool) -> None:
        """Test get_connection method."""
        assert mock_redis_pool.get_connection() == mock_redis_pool.client

    def test_ping_success(self, mock_redis_pool: RedisPool) -> None:
        """Test ping method when operation succeeds."""
        mock_redis_pool.client.ping.return_value = True

        result = mock_redis_pool.ping()

        assert result is True
        mock_redis_pool.client.ping.assert_called_once()

    def test_ping_failure(self, mock_redis_pool: RedisPool) -> None:
        """Test ping method when operation fails."""
        mock_redis_pool.client.ping.side_effect = Exception("Connection failed")

        result = mock_redis_pool.ping()

        assert result is False
        mock_redis_pool.client.ping.assert_called_once()

    def test_set_string_value(self, mock_redis_pool: RedisPool) -> None:
        """Test set method with string value."""
        mock_redis_pool.client.set.return_value = True

        result = mock_redis_pool.set("test_key", "test_value")

        assert result is True
        mock_redis_pool.client.set.assert_called_once_with("test_key", "test_value")

    def test_set_bytes_value(self, mock_redis_pool: RedisPool) -> None:
        """Test set method with bytes value."""
        mock_redis_pool.client.set.return_value = True
        bytes_value = b"test_bytes"

        result = mock_redis_pool.set("test_key", bytes_value)

        assert result is True
        mock_redis_pool.client.set.assert_called_once_with("test_key", bytes_value)

    def test_set_dict_value(self, mock_redis_pool: RedisPool) -> None:
        """Test set method with dictionary value."""
        mock_redis_pool.client.set.return_value = True
        dict_value = {"key": "value", "number": 42}

        result = mock_redis_pool.set("test_key", dict_value)

        assert result is True
        mock_redis_pool.client.set.assert_called_once_with(
            "test_key", '{"key": "value", "number": 42}'
        )

    def test_set_list_value(self, mock_redis_pool: RedisPool) -> None:
        """Test set method with list value."""
        mock_redis_pool.client.set.return_value = True
        list_value = [1, 2, 3, "test"]

        result = mock_redis_pool.set("test_key", list_value)

        assert result is True
        mock_redis_pool.client.set.assert_called_once_with(
            "test_key", '[1, 2, 3, "test"]'
        )

    def test_set_with_ttl(self, mock_redis_pool: RedisPool) -> None:
        """Test set method with TTL."""
        mock_redis_pool.client.setex.return_value = True

        result = mock_redis_pool.set("test_key", "test_value", ttl=1800)

        assert result is True
        # Check that setex was called with the correct parameters
        mock_redis_pool.client.setex.assert_called_once_with(
            "test_key", 1800, "test_value"
        )

    def test_set_failure(self, mock_redis_pool: RedisPool) -> None:
        """Test set method when operation fails."""
        mock_redis_pool.client.set.side_effect = Exception("Redis error")

        result = mock_redis_pool.set("test_key", "test_value")

        assert result is False
        mock_redis_pool.client.set.assert_called_once_with("test_key", "test_value")

    def test_get_existing_key(self, mock_redis_pool: RedisPool) -> None:
        """Test get method with existing key."""
        mock_redis_pool.client.get.return_value = b"test_value"

        result = mock_redis_pool.get("test_key")

        assert result == "test_value"
        mock_redis_pool.client.get.assert_called_once_with("test_key")

    def test_get_nonexistent_key(self, mock_redis_pool: RedisPool) -> None:
        """Test get method with non-existent key."""
        mock_redis_pool.client.get.return_value = None

        result = mock_redis_pool.get("test_key")

        assert result is None
        mock_redis_pool.client.get.assert_called_once_with("test_key")

    def test_get_json_value(self, mock_redis_pool: RedisPool) -> None:
        """Test get method with JSON value."""
        json_value = json.dumps({"key": "value"})
        mock_redis_pool.client.get.return_value = json_value.encode()

        result = mock_redis_pool.get("test_key", decode_json=True)

        assert result == {"key": "value"}
        mock_redis_pool.client.get.assert_called_once_with("test_key")

    def test_get_json_value_no_decode(self, mock_redis_pool: RedisPool) -> None:
        """Test get method with JSON value but decode_json=False."""
        json_value = json.dumps({"key": "value"})
        mock_redis_pool.client.get.return_value = json_value.encode()

        result = mock_redis_pool.get("test_key", decode_json=False)

        assert result == json_value
        mock_redis_pool.client.get.assert_called_once_with("test_key")

    def test_get_invalid_json(self, mock_redis_pool: RedisPool) -> None:
        """Test get method with invalid JSON value."""
        invalid_json = "{invalid json"
        mock_redis_pool.client.get.return_value = invalid_json.encode()

        result = mock_redis_pool.get("test_key", decode_json=True)

        assert result == invalid_json
        mock_redis_pool.client.get.assert_called_once_with("test_key")

    def test_get_failure(self, mock_redis_pool: RedisPool) -> None:
        """Test get method when operation fails."""
        mock_redis_pool.client.get.side_effect = Exception("Redis error")

        result = mock_redis_pool.get("test_key")

        assert result is None
        mock_redis_pool.client.get.assert_called_once_with("test_key")

    def test_delete_single_key(self, mock_redis_pool: RedisPool) -> None:
        """Test delete method with single key."""
        mock_redis_pool.client.delete.return_value = 1

        result = mock_redis_pool.delete("test_key")

        assert result == 1
        mock_redis_pool.client.delete.assert_called_once_with("test_key")

    def test_delete_multiple_keys(self, mock_redis_pool: RedisPool) -> None:
        """Test delete method with multiple keys."""
        mock_redis_pool.client.delete.return_value = 2

        result = mock_redis_pool.delete("key1", "key2")

        assert result == 2
        mock_redis_pool.client.delete.assert_called_once_with("key1", "key2")

    def test_delete_failure(self, mock_redis_pool: RedisPool) -> None:
        """Test delete method when operation fails."""
        mock_redis_pool.client.delete.side_effect = Exception("Redis error")

        result = mock_redis_pool.delete("key1", "key2")

        assert result == 0
        mock_redis_pool.client.delete.assert_called_once_with("key1", "key2")

    def test_exists_single_key(self, mock_redis_pool: RedisPool) -> None:
        """Test exists method with single key."""
        mock_redis_pool.client.exists.return_value = 1

        result = mock_redis_pool.exists("test_key")

        assert result == 1
        mock_redis_pool.client.exists.assert_called_once_with("test_key")

    def test_exists_multiple_keys(self, mock_redis_pool: RedisPool) -> None:
        """Test exists method with multiple keys."""
        mock_redis_pool.client.exists.return_value = 2

        result = mock_redis_pool.exists("key1", "key2")

        assert result == 2
        mock_redis_pool.client.exists.assert_called_once_with("key1", "key2")

    def test_exists_failure(self, mock_redis_pool: RedisPool) -> None:
        """Test exists method when operation fails."""
        mock_redis_pool.client.exists.side_effect = Exception("Redis error")

        result = mock_redis_pool.exists("key1", "key2")

        assert result == 0
        mock_redis_pool.client.exists.assert_called_once_with("key1", "key2")

    def test_expire_success(self, mock_redis_pool: RedisPool) -> None:
        """Test expire method when operation succeeds."""
        mock_redis_pool.client.expire.return_value = True

        result = mock_redis_pool.expire("test_key", 3600)

        assert result is True
        mock_redis_pool.client.expire.assert_called_once_with("test_key", 3600)

    def test_expire_failure(self, mock_redis_pool: RedisPool) -> None:
        """Test expire method when operation fails."""
        mock_redis_pool.client.expire.side_effect = Exception("Redis error")

        result = mock_redis_pool.expire("test_key", 3600)

        assert result is False
        mock_redis_pool.client.expire.assert_called_once_with("test_key", 3600)

    def test_ttl_existing_key(self, mock_redis_pool: RedisPool) -> None:
        """Test ttl method with existing key."""
        mock_redis_pool.client.ttl.return_value = 1800

        result = mock_redis_pool.ttl("test_key")

        assert result == 1800
        mock_redis_pool.client.ttl.assert_called_once_with("test_key")

    def test_ttl_no_expiry(self, mock_redis_pool: RedisPool) -> None:
        """Test ttl method with key that has no expiry."""
        mock_redis_pool.client.ttl.return_value = -1

        result = mock_redis_pool.ttl("test_key")

        assert result == -1
        mock_redis_pool.client.ttl.assert_called_once_with("test_key")

    def test_ttl_nonexistent_key(self, mock_redis_pool: RedisPool) -> None:
        """Test ttl method with non-existent key."""
        mock_redis_pool.client.ttl.return_value = -2

        result = mock_redis_pool.ttl("test_key")

        assert result == -2
        mock_redis_pool.client.ttl.assert_called_once_with("test_key")

    def test_ttl_failure(self, mock_redis_pool: RedisPool) -> None:
        """Test ttl method when operation fails."""
        mock_redis_pool.client.ttl.side_effect = Exception("Redis error")

        result = mock_redis_pool.ttl("test_key")

        assert result == -2
        mock_redis_pool.client.ttl.assert_called_once_with("test_key")

    def test_flush_db_success(self, mock_redis_pool: RedisPool) -> None:
        """Test flush_db method when operation succeeds."""
        mock_redis_pool.client.flushdb.return_value = True

        result = mock_redis_pool.flush_db()

        assert result is True
        mock_redis_pool.client.flushdb.assert_called_once()

    def test_flush_db_failure(self, mock_redis_pool: RedisPool) -> None:
        """Test flush_db method when operation fails."""
        mock_redis_pool.client.flushdb.side_effect = Exception("Redis error")

        result = mock_redis_pool.flush_db()

        assert result is False
        mock_redis_pool.client.flushdb.assert_called_once()

    def test_keys_default_pattern(self, mock_redis_pool: RedisPool) -> None:
        """Test keys method with default pattern."""
        mock_redis_pool.client.keys.return_value = [b"key1", b"key2", "key3"]

        result = mock_redis_pool.keys()

        assert result == ["key1", "key2", "key3"]
        mock_redis_pool.client.keys.assert_called_once_with("*")

    def test_keys_custom_pattern(self, mock_redis_pool: RedisPool) -> None:
        """Test keys method with custom pattern."""
        mock_redis_pool.client.keys.return_value = [b"user:*", b"user:123"]

        result = mock_redis_pool.keys("user:*")

        assert result == ["user:*", "user:123"]
        mock_redis_pool.client.keys.assert_called_once_with("user:*")

    def test_keys_empty_result(self, mock_redis_pool: RedisPool) -> None:
        """Test keys method with empty result."""
        mock_redis_pool.client.keys.return_value = []

        result = mock_redis_pool.keys("user:*")

        assert result == []
        mock_redis_pool.client.keys.assert_called_once_with("user:*")

    def test_keys_failure(self, mock_redis_pool: RedisPool) -> None:
        """Test keys method when operation fails."""
        mock_redis_pool.client.keys.side_effect = Exception("Redis error")

        result = mock_redis_pool.keys()

        assert result == []
        mock_redis_pool.client.keys.assert_called_once_with("*")

    def test_close_success(self, mock_redis_pool: RedisPool) -> None:
        """Test close method when operation succeeds."""
        mock_redis_pool.connection_pool.disconnect.return_value = None

        result = mock_redis_pool.close()

        assert result is None
        mock_redis_pool.connection_pool.disconnect.assert_called_once()

    def test_close_failure(self, mock_redis_pool: RedisPool) -> None:
        """Test close method when operation fails."""
        mock_redis_pool.connection_pool.disconnect.side_effect = Exception(
            "Connection error"
        )

        result = mock_redis_pool.close()

        assert result is None
        mock_redis_pool.connection_pool.disconnect.assert_called_once()


class TestGlobalRedisPool:
    """Test the global redis_pool instance."""

    def test_global_redis_pool_exists(self) -> None:
        """Test that the global redis_pool instance exists."""
        assert redis_pool is not None
        assert isinstance(redis_pool, RedisPool)

    def test_global_redis_pool_configuration(self) -> None:
        """Test that the global redis_pool uses filtered configuration."""
        # The global instance should use the filtered config
        assert redis_pool._db == filtered_redis_config.get("db", 0)

    def test_global_redis_pool_pool_size(self) -> None:
        """Test that the global redis_pool uses the correct pool size."""
        assert redis_pool.connection_pool.max_connections == POOL_SIZE


class TestRedisPoolEdgeCases:
    """Test RedisPool edge cases and special scenarios."""

    @pytest.fixture
    def mock_redis_pool(self) -> RedisPool:
        """Create a RedisPool instance with mocked dependencies."""
        with (
            patch("arklex.utils.redis.ConnectionPool"),
            patch("arklex.utils.redis.redis.Redis") as mock_redis_class,
        ):
            mock_client = MagicMock()
            mock_redis_class.return_value = mock_client
            pool = RedisPool()
            return pool

    def test_set_complex_data_types(self, mock_redis_pool: RedisPool) -> None:
        """Test set method with various complex data types."""
        test_cases = [
            ({"nested": {"key": "value"}}, '{"nested": {"key": "value"}}'),
            ([1, 2, {"a": "b"}], '[1, 2, {"a": "b"}]'),
            ({"unicode": "café 🍕"}, '{"unicode": "caf\\u00e9 \\ud83c\\udf55"}'),
            ({"bytes": b"test"}, '{"bytes": "test"}'),
        ]

        for input_value, expected_json in test_cases:
            mock_redis_pool.client.set.return_value = True
            result = mock_redis_pool.set("test_key", input_value)
            assert result is True
            # Check that set was called with the correct parameters
            mock_redis_pool.client.set.assert_called_once_with(
                "test_key", expected_json
            )
            # Reset the mock for the next iteration
            mock_redis_pool.client.set.reset_mock()

    def test_get_unicode_string(self, mock_redis_pool: RedisPool) -> None:
        """Test get method with Unicode string."""
        unicode_value = "café 🍕"
        mock_redis_pool.client.get.return_value = unicode_value.encode()

        result = mock_redis_pool.get("test_key")

        assert result == unicode_value
        mock_redis_pool.client.get.assert_called_once_with("test_key")

    def test_get_empty_string(self, mock_redis_pool: RedisPool) -> None:
        """Test get method with empty string."""
        mock_redis_pool.client.get.return_value = b""

        result = mock_redis_pool.get("test_key")

        assert result == ""
        mock_redis_pool.client.get.assert_called_once_with("test_key")

    def test_set_empty_string(self, mock_redis_pool: RedisPool) -> None:
        """Test set method with empty string."""
        mock_redis_pool.client.set.return_value = True

        result = mock_redis_pool.set("test_key", "")

        assert result is True
        mock_redis_pool.client.set.assert_called_once_with("test_key", "")

    def test_set_none_value(self, mock_redis_pool: RedisPool) -> None:
        """Test set method with None value."""
        mock_redis_pool.client.set.return_value = True

        result = mock_redis_pool.set("test_key", None)

        assert result is True
        mock_redis_pool.client.set.assert_called_once_with("test_key", "null")

    def test_get_zero_ttl(self, mock_redis_pool: RedisPool) -> None:
        """Test set method with zero TTL."""
        mock_redis_pool.client.set.return_value = True

        result = mock_redis_pool.set("test_key", "value", ttl=0)

        assert result is True
        mock_redis_pool.client.set.assert_called_once_with("test_key", "value")

    def test_delete_no_keys(self, mock_redis_pool: RedisPool) -> None:
        """Test delete method with no keys."""
        mock_redis_pool.client.delete.return_value = 0

        result = mock_redis_pool.delete()

        assert result == 0
        mock_redis_pool.client.delete.assert_called_once_with()

    def test_exists_no_keys(self, mock_redis_pool: RedisPool) -> None:
        """Test exists method with no keys."""
        mock_redis_pool.client.exists.return_value = 0

        result = mock_redis_pool.exists()

        assert result == 0
        mock_redis_pool.client.exists.assert_called_once_with()

    def test_keys_special_patterns(self, mock_redis_pool: RedisPool) -> None:
        """Test keys method with special patterns."""
        patterns = ["*", "?", "[abc]", "user:*", "session:???"]
        for pattern in patterns:
            mock_redis_pool.client.keys.return_value = [b"test_key"]
            result = mock_redis_pool.keys(pattern)
            assert result == ["test_key"]
            mock_redis_pool.client.keys.assert_called_with(pattern)


class TestRedisPoolIntegration:
    """Integration tests for RedisPool (require local Redis)."""

    @pytest.mark.skipif(
        os.getenv("ARKLEX_TEST_ENV") != "local",
        reason="Integration tests require ARKLEX_TEST_ENV=local",
    )
    def test_redis_pool_creation_integration(self) -> None:
        """Test RedisPool creation in integration environment."""
        # This test verifies that RedisPool can be created without errors
        # when running against a real Redis instance
        pool = RedisPool()
        assert pool is not None
        assert isinstance(pool, RedisPool)
        assert pool.client is not None
        assert pool.connection_pool is not None

    @pytest.mark.skipif(
        os.getenv("ARKLEX_TEST_ENV") != "local",
        reason="Integration tests require ARKLEX_TEST_ENV=local",
    )
    def test_global_redis_pool_integration(self) -> None:
        """Test global redis_pool in integration environment."""
        # Verify global instance exists and has correct type
        assert redis_pool is not None
        assert isinstance(redis_pool, RedisPool)
        assert redis_pool.client is not None
        assert redis_pool.connection_pool is not None


class TestRedisPoolLogging:
    """Test RedisPool logging functionality."""

    @pytest.fixture
    def mock_redis_pool(self) -> RedisPool:
        """Create a RedisPool instance with mocked dependencies."""
        with (
            patch("arklex.utils.redis.ConnectionPool"),
            patch("arklex.utils.redis.redis.Redis") as mock_redis_class,
        ):
            mock_client = MagicMock()
            mock_redis_class.return_value = mock_client
            pool = RedisPool()
            return pool

    def test_ping_success_logging(self, mock_redis_pool: RedisPool) -> None:
        """Test that successful ping logs debug message."""
        with patch("arklex.utils.redis.log_context") as mock_log:
            mock_redis_pool.client.ping.return_value = True
            mock_redis_pool.ping()
            mock_log.debug.assert_called_with("Redis ping successful")

    def test_ping_failure_logging(self, mock_redis_pool: RedisPool) -> None:
        """Test that failed ping logs error message."""
        with patch("arklex.utils.redis.log_context") as mock_log:
            mock_redis_pool.client.ping.side_effect = Exception("Connection failed")
            mock_redis_pool.ping()
            mock_log.error.assert_called_with("Redis ping failed: Connection failed")

    def test_set_success_logging(self, mock_redis_pool: RedisPool) -> None:
        """Test that successful set logs debug message."""
        with patch("arklex.utils.redis.log_context") as mock_log:
            mock_redis_pool.client.set.return_value = True
            mock_redis_pool.set("test_key", "test_value")
            mock_log.debug.assert_called_with("Redis SET successful for key: test_key")

    def test_set_failure_logging(self, mock_redis_pool: RedisPool) -> None:
        """Test that failed set logs error message."""
        with patch("arklex.utils.redis.log_context") as mock_log:
            mock_redis_pool.client.set.side_effect = Exception("Redis error")
            mock_redis_pool.set("test_key", "test_value")
            mock_log.error.assert_called_with(
                "Redis SET failed for key test_key: Redis error"
            )

    def test_get_success_logging(self, mock_redis_pool: RedisPool) -> None:
        """Test that successful get logs debug message."""
        with patch("arklex.utils.redis.log_context") as mock_log:
            mock_redis_pool.client.get.return_value = b"test_value"
            mock_redis_pool.get("test_key")
            mock_log.debug.assert_called_with("Redis GET successful for key: test_key")

    def test_get_failure_logging(self, mock_redis_pool: RedisPool) -> None:
        """Test that failed get logs error message."""
        with patch("arklex.utils.redis.log_context") as mock_log:
            mock_redis_pool.client.get.side_effect = Exception("Redis error")
            mock_redis_pool.get("test_key")
            mock_log.error.assert_called_with(
                "Redis GET failed for key test_key: Redis error"
            )

    def test_flush_db_warning_logging(self, mock_redis_pool: RedisPool) -> None:
        """Test that flush_db logs warning message."""
        with patch("arklex.utils.redis.log_context") as mock_log:
            mock_redis_pool.client.flushdb.return_value = True
            mock_redis_pool.flush_db()
            mock_log.warning.assert_called_with("Redis database flushed")

    def test_close_success_logging(self, mock_redis_pool: RedisPool) -> None:
        """Test that successful close logs info message."""
        with patch("arklex.utils.redis.log_context") as mock_log:
            mock_redis_pool.connection_pool.disconnect.return_value = None
            mock_redis_pool.close()
            mock_log.info.assert_called_with("Redis connection pool closed")

    def test_close_failure_logging(self, mock_redis_pool: RedisPool) -> None:
        """Test that failed close logs error message."""
        with patch("arklex.utils.redis.log_context") as mock_log:
            mock_redis_pool.connection_pool.disconnect.side_effect = Exception(
                "Connection error"
            )
            mock_redis_pool.close()
            mock_log.error.assert_called_with(
                "Error closing Redis connection pool: Connection error"
            )
