"""Redis connection and pool management for the Arklex framework.

This module provides functionality for managing Redis connections using connection pooling.
It includes configuration settings for Redis connections, connection pool management,
and utility methods for executing Redis commands and managing cache operations.
The module implements connection pooling to efficiently handle multiple Redis operations
while maintaining connection limits and proper resource management.

The module is organized into several key components:
1. Configuration: Environment-based Redis configuration settings
2. Connection Pool: Management of Redis connection pools
3. Command Execution: Methods for executing Redis commands
4. Cache Operations: Utility methods for common caching patterns
5. Resource Management: Proper handling of Redis connections and resources

Key Features:
- Connection pooling for efficient resource management
- Environment-based configuration
- Automatic connection timeout handling
- Comprehensive error handling and logging
- Support for various Redis data types
- Cache utility methods with TTL support
- Resource cleanup

Usage:
    from arklex.utils.redis import RedisPool, redis_pool

    # Use the global instance
    redis_pool.set("key", "value", ttl=3600)
    value = redis_pool.get("key")

    # Or create a custom instance
    custom_pool = RedisPool(
        host="localhost",
        port=6379,
        db=0,
        password="password",
        max_connections=10
    )
"""

import json
import os
from typing import Any

import redis
from redis.connection import ConnectionPool

from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

# Timeout in seconds for establishing Redis connections
CONNECTION_TIMEOUT: int = int(os.getenv("REDIS_CONNECTION_TIMEOUT", 5))

# Redis configuration loaded from environment variables
REDIS_CONFIG: dict[str, str | int | None] = {
    "host": os.getenv("REDIS_HOST", "localhost"),  # Redis host address
    "port": int(os.getenv("REDIS_PORT", 6379)),  # Redis port number
    "db": int(os.getenv("REDIS_DB", 0)),  # Redis database number
    "password": os.getenv("REDIS_PASSWORD"),  # Redis password (optional)
    "username": os.getenv("REDIS_USERNAME"),  # Redis username (optional)
}

# Maximum number of connections in the connection pool
POOL_SIZE: int = int(os.getenv("REDIS_POOL_SIZE", 10))

# Default TTL for cache operations (in seconds)
DEFAULT_TTL: int = int(os.getenv("REDIS_DEFAULT_TTL", 3600))


class RedisPool:
    """A connection pool manager for Redis connections.

    This class manages a pool of Redis connections, providing methods for
    obtaining connections, executing commands, and managing cache operations. It implements
    connection pooling to efficiently handle multiple Redis operations while
    maintaining connection limits and proper resource management.

    The class provides the following key features:
    1. Connection Pool Management: Efficient handling of Redis connections
    2. Command Execution: Methods for executing Redis commands with proper resource cleanup
    3. Cache Operations: High-level methods for common caching patterns
    4. Error Handling: Comprehensive error handling and logging
    5. Resource Cleanup: Automatic cleanup of Redis resources

    Attributes:
        _host (str): The Redis host address
        _port (int): The Redis port number
        _db (int): The Redis database number
        _password (Optional[str]): The Redis password
        _username (Optional[str]): The Redis username
        redis_config (Dict[str, Any]): Complete Redis configuration dictionary
        connection_pool (ConnectionPool): The Redis connection pool instance
        client (redis.Redis): The Redis client instance
    """

    def __init__(
        self, max_connections: int = POOL_SIZE, **kwargs: dict[str, Any]
    ) -> None:
        """Initialize the Redis connection pool.

        This method sets up a connection pool with the specified size and configuration.
        It initializes the Redis connection parameters and creates the connection pool.

        Args:
            max_connections (int): The maximum number of connections in the pool
            **kwargs (Dict[str, Any]): Redis configuration parameters including:
                - host: Redis host address (default: "localhost")
                - port: Redis port number (default: 6379)
                - db: Redis database number (default: 0)
                - password: Redis password (default: None)
                - username: Redis username (default: None)
        """
        self._host: str = kwargs.get("host", "localhost")
        self._port: int = kwargs.get("port", 6379)
        self._db: int = kwargs.get("db", 0)
        self._password: str | None = kwargs.get("password")
        self._username: str | None = kwargs.get("username")

        # Build Redis configuration
        self.redis_config: dict[str, Any] = {
            "host": self._host,
            "port": self._port,
            "db": self._db,
            "socket_timeout": CONNECTION_TIMEOUT,
            "socket_connect_timeout": CONNECTION_TIMEOUT,
            "decode_responses": False,  # Keep as bytes for flexibility
        }

        # Add optional authentication parameters
        if self._password:
            self.redis_config["password"] = self._password
        if self._username:
            self.redis_config["username"] = self._username

        # Create connection pool
        self.connection_pool = ConnectionPool(
            max_connections=max_connections, **self.redis_config
        )

        # Create Redis client
        self.client = redis.Redis(connection_pool=self.connection_pool)

        log_context.info(
            f"Redis pool initialized with {max_connections} max connections"
        )

    def get_connection(self) -> redis.Redis:
        """Get a Redis client instance.

        This method returns the Redis client which manages connections automatically
        through the connection pool.

        Returns:
            redis.Redis: A Redis client instance
        """
        return self.client

    def ping(self) -> bool:
        """Test the Redis connection.

        This method tests the connection to Redis by sending a PING command.

        Returns:
            bool: True if the connection is successful, False otherwise
        """
        try:
            response = self.client.ping()
            log_context.debug("Redis ping successful")
            return response
        except Exception as e:
            log_context.error(f"Redis ping failed: {e}")
            return False

    def set(
        self,
        key: str,
        value: str | bytes | dict | list | int | float | bool,
        ttl: int | None = None,
    ) -> bool:
        """Set a key-value pair in Redis with optional TTL.

        This method stores a value in Redis with the given key. If the value is not
        a string or bytes, it will be JSON-serialized.

        Args:
            key (str): The Redis key
            value (str | bytes | dict | list | int | float | bool): The value to store
            ttl (Optional[int]): Time to live in seconds (default: None for no expiration)

        Returns:
            bool: True if the operation was successful, False otherwise
        """
        try:
            # Serialize non-string/bytes values as JSON
            if isinstance(value, str | bytes):
                stored_value = value
            else:
                stored_value = json.dumps(value)

            if ttl:
                result = self.client.setex(key, ttl, stored_value)
            else:
                result = self.client.set(key, stored_value)

            log_context.debug(f"Redis SET successful for key: {key}")
            return bool(result)
        except Exception as e:
            log_context.error(f"Redis SET failed for key {key}: {e}")
            return False

    def get(self, key: str, decode_json: bool = True) -> str | None:
        """Get a value from Redis by key.

        This method retrieves a value from Redis. If decode_json is True and the value
        appears to be JSON, it will be deserialized.

        Args:
            key (str): The Redis key
            decode_json (bool): Whether to attempt JSON deserialization (default: True)

        Returns:
            str | None: The retrieved value as a string, or None if the key doesn't exist
        """
        try:
            value = self.client.get(key)
            if value is None:
                return None

            # Decode bytes to string
            if isinstance(value, bytes):
                value = value.decode("utf-8")

            # Attempt JSON deserialization if requested
            if decode_json and isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    # Return as string if not valid JSON
                    pass

            log_context.debug(f"Redis GET successful for key: {key}")
            return value
        except Exception as e:
            log_context.error(f"Redis GET failed for key {key}: {e}")
            return None

    def delete(self, *keys: str) -> int:
        """Delete one or more keys from Redis.

        Args:
            *keys (str): One or more Redis keys to delete

        Returns:
            int: Number of keys that were deleted
        """
        try:
            result = self.client.delete(*keys)
            log_context.debug(f"Redis DELETE successful for keys: {keys}")
            return result
        except Exception as e:
            log_context.error(f"Redis DELETE failed for keys {keys}: {e}")
            return 0

    def exists(self, *keys: str) -> int:
        """Check if one or more keys exist in Redis.

        Args:
            *keys (str): One or more Redis keys to check

        Returns:
            int: Number of keys that exist
        """
        try:
            result = self.client.exists(*keys)
            log_context.debug(f"Redis EXISTS check for keys: {keys}")
            return result
        except Exception as e:
            log_context.error(f"Redis EXISTS failed for keys {keys}: {e}")
            return 0

    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for an existing key.

        Args:
            key (str): The Redis key
            ttl (int): Time to live in seconds

        Returns:
            bool: True if the operation was successful, False otherwise
        """
        try:
            result = self.client.expire(key, ttl)
            log_context.debug(f"Redis EXPIRE successful for key: {key}")
            return bool(result)
        except Exception as e:
            log_context.error(f"Redis EXPIRE failed for key {key}: {e}")
            return False

    def ttl(self, key: str) -> int:
        """Get the TTL of a key.

        Args:
            key (str): The Redis key

        Returns:
            int: TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        try:
            result = self.client.ttl(key)
            log_context.debug(f"Redis TTL check for key: {key}")
            return result
        except Exception as e:
            log_context.error(f"Redis TTL failed for key {key}: {e}")
            return -2

    def flush_db(self) -> bool:
        """Flush the current Redis database.

        Returns:
            bool: True if the operation was successful, False otherwise
        """
        try:
            result = self.client.flushdb()
            log_context.warning("Redis database flushed")
            return bool(result)
        except Exception as e:
            log_context.error(f"Redis FLUSHDB failed: {e}")
            return False

    def keys(self, pattern: str = "*") -> list[str]:
        """Get all keys matching a pattern.

        Args:
            pattern (str): The pattern to match (default: "*" for all keys)

        Returns:
            List[str]: List of matching keys
        """
        try:
            keys = self.client.keys(pattern)
            # Decode bytes keys to strings
            result = [
                key.decode("utf-8") if isinstance(key, bytes) else key for key in keys
            ]
            log_context.debug(
                f"Redis KEYS found {len(result)} keys matching pattern: {pattern}"
            )
            return result
        except Exception as e:
            log_context.error(f"Redis KEYS failed for pattern {pattern}: {e}")
            return []

    def close(self) -> None:
        """Close all connections in the pool.

        This method closes the connection pool and cleans up resources.
        """
        try:
            self.connection_pool.disconnect()
            log_context.info("Redis connection pool closed")
        except Exception as e:
            log_context.error(f"Error closing Redis connection pool: {e}")


# Filter out None values from Redis config
filtered_redis_config = {k: v for k, v in REDIS_CONFIG.items() if v is not None}

# Create a global instance of the Redis connection pool
redis_pool = RedisPool(POOL_SIZE, **filtered_redis_config)
