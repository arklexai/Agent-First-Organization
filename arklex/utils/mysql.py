"""MySQL database connection and pool management for the Arklex framework.

This module provides functionality for managing MySQL database connections using connection pooling.
It includes configuration settings for database connections, connection pool management,
and utility methods for executing queries and managing database transactions.
The module implements connection pooling to efficiently handle multiple database operations
while maintaining connection limits and proper resource management.

The module is organized into several key components:
1. Configuration: Environment-based database configuration settings
2. Connection Pool: Management of database connection pools
3. Query Execution: Methods for executing SQL queries and transactions
4. Resource Management: Proper handling of database connections and resources

Key Features:
- Connection pooling for efficient resource management
- Environment-based configuration
- Automatic connection timeout handling
- Comprehensive error handling and logging
- Support for parameterized queries
- Transaction management
- Resource cleanup

Usage:
    from arklex.utils.mysql import MySQLPool

    # Initialize connection pool
    pool = MySQLPool(
        pool_size=10,
        host="localhost",
        port=3306,
        user="root",
        password="password",
        database="mydb"
    )

    # Execute queries
    results = pool.fetchall("SELECT * FROM users WHERE age > %s", (18,))

    # Execute updates
    pool.execute("UPDATE users SET status = %s WHERE id = %s", ("active", 1))
"""

import os
import time
from typing import Any

import mysql.connector

from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

# Timeout in seconds for establishing database connections
CONNECTION_TIMEOUT: int = int(os.getenv("MYSQL_CONNECTION_TIMEOUT", 10))

# Database configuration loaded from environment variables
MYSQL_CONFIG: dict[str, str | None] = {
    "user": os.getenv("MYSQL_USERNAME"),  # Database username
    "password": os.getenv("MYSQL_PASSWORD"),  # Database password
    "host": os.getenv("MYSQL_HOSTNAME"),  # Database host address
    "port": os.getenv("MYSQL_PORT"),  # Database port number
    "database": os.getenv("MYSQL_DB_NAME"),  # Database name
}

# Maximum number of connections in the connection pool
POOL_SIZE: int = int(os.getenv("MYSQL_POOL_SIZE", 10))

# Configure the maximum pool size for MySQL connector
mysql.connector.pooling.CNX_POOL_MAXSIZE = POOL_SIZE


class MySQLPool:
    """A connection pool manager for MySQL database connections.

    This class manages a pool of MySQL database connections, providing methods for
    obtaining connections, executing queries, and managing transactions. It implements
    connection pooling to efficiently handle multiple database operations while
    maintaining connection limits and proper resource management.

    The class provides the following key features:
    1. Connection Pool Management: Efficient handling of database connections
    2. Query Execution: Methods for executing SQL queries with proper resource cleanup
    3. Transaction Support: Methods for managing database transactions
    4. Error Handling: Comprehensive error handling and logging
    5. Resource Cleanup: Automatic cleanup of database resources

    Attributes:
        _host (str): The database host address
        _port (int): The database port number
        _user (str): The database username
        _password (str): The database password
        _database (str): The database name
        dbconfig (Dict[str, Any]): Complete database configuration dictionary
        pool (mysql.connector.pooling.MySQLConnectionPool): The connection pool instance
    """

    def __init__(self, pool_size: int, **kwargs: dict[str, Any]) -> None:
        """Initialize the MySQL connection pool.

        This method sets up a connection pool with the specified size and configuration.
        It initializes the database connection parameters and creates the connection pool.

        Args:
            pool_size (int): The maximum number of connections in the pool
            **kwargs (Dict[str, Any]): Database configuration parameters including:
                - host: Database host address (default: "localhost")
                - port: Database port number (default: 3306)
                - user: Database username (default: "root")
                - password: Database password (default: "")
                - database: Database name (default: "test")
        """
        self._host: str = kwargs.get("host", "localhost")
        self._port: int = kwargs.get("port", 3306)
        self._user: str = kwargs.get("user", "root")
        self._password: str = kwargs.get("password", "")
        self._database: str = kwargs.get("database", "test")
        self.dbconfig: dict[str, Any] = {
            "host": self._host,
            "port": self._port,
            "user": self._user,
            "password": self._password,
            "database": self._database,
        }
        self.pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="mypool",
            pool_size=pool_size,
            pool_reset_session=True,
            **self.dbconfig,
        )

    def get_connection(self) -> mysql.connector.pooling.PooledMySQLConnection:
        """Get a connection from the pool.

        This method attempts to obtain a connection from the pool within the configured
        timeout period. If the pool is exhausted, it will retry until the timeout is reached.

        The method implements the following behavior:
        1. Attempts to get a connection from the pool
        2. If the pool is exhausted, waits briefly and retries
        3. Continues retrying until the timeout is reached
        4. Logs connection establishment time

        Returns:
            mysql.connector.pooling.PooledMySQLConnection: A pooled database connection

        Raises:
            Exception: If unable to obtain a connection within the timeout period
        """
        t0 = time.time()
        while time.time() - t0 < CONNECTION_TIMEOUT:
            try:
                conn = self.pool.get_connection()
                log_context.info(
                    "mysql connection established", extra={"time": time.time() - t0}
                )
                return conn
            except mysql.connector.pooling.PoolError as e:
                if "pool exhausted" in str(e):
                    time.sleep(0.05)
                    continue
                raise e
            except Exception as e:
                raise e
        raise Exception(f"Pool exhausted; retried for {CONNECTION_TIMEOUT} seconds")

    def close(self, sql_conns: mysql.connector.pooling.PooledMySQLConnection) -> None:
        """Close a database connection and return it to the pool.

        This method properly closes a database connection and returns it to the pool
        for reuse. It ensures proper cleanup of database resources.

        Args:
            sql_conns (mysql.connector.pooling.PooledMySQLConnection): The connection to close
        """
        sql_conns.close()

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        """Execute a SQL query without returning results.

        This method executes the given SQL query with optional parameters and commits
        the transaction. The connection is automatically returned to the pool after execution.

        The method handles:
        1. Connection acquisition from the pool
        2. Query execution with parameters
        3. Transaction commitment
        4. Resource cleanup
        5. Error handling

        Args:
            sql (str): The SQL query to execute
            params (Optional[Tuple[Any, ...]]): Optional parameters for the query

        Raises:
            Exception: If an error occurs during query execution
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                conn.commit()
        except Exception as e:
            raise e
        finally:
            conn.close()
        return

    def fetchall(
        self, sql: str, params: tuple[Any, ...] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return all results.

        This method executes the given SQL query with optional parameters and returns
        all matching rows as a list of dictionaries. The connection is automatically
        returned to the pool after execution.

        The method handles:
        1. Connection acquisition from the pool
        2. Query execution with parameters
        3. Result fetching and formatting
        4. Resource cleanup
        5. Error handling

        Args:
            sql (str): The SQL query to execute
            params (Optional[Tuple[Any, ...]]): Optional parameters for the query

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing the query results

        Raises:
            Exception: If an error occurs during query execution
        """
        conn = self.get_connection()
        try:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(sql, params)
                return cursor.fetchall()
        except Exception as e:
            raise e
        finally:
            conn.close()

    def fetchone(
        self, sql: str, params: tuple[Any, ...] | None = None
    ) -> dict[str, Any] | None:
        """Execute a SQL query and return a single result.

        This method executes the given SQL query with optional parameters and returns
        the first matching row as a dictionary. The connection is automatically returned
        to the pool after execution.

        The method handles:
        1. Connection acquisition from the pool
        2. Query execution with parameters
        3. Single result fetching and formatting
        4. Resource cleanup
        5. Error handling

        Args:
            sql (str): The SQL query to execute
            params (Optional[Tuple[Any, ...]]): Optional parameters for the query

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing the first matching row, or None if no matches

        Raises:
            Exception: If an error occurs during query execution
        """
        conn = self.get_connection()
        try:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(sql, params)
                return cursor.fetchone()
        except Exception as e:
            raise e
        finally:
            if conn.is_connected():
                conn.close()


# Create a global instance of the MySQL connection pool
mysql_pool = MySQLPool(POOL_SIZE, **MYSQL_CONFIG)
