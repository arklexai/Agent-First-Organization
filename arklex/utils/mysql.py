"""MySQL database connection and pool management for the Arklex framework.

This module provides functionality for managing MySQL database connections using connection pooling.
It includes configuration settings for database connections, connection pool management,
and utility methods for executing queries and managing database transactions.
The module implements connection pooling to efficiently handle multiple database operations
while maintaining connection limits and proper resource management.
"""

import os
import mysql.connector
import time
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

CONNECTION_TIMEOUT: int = int(os.getenv("MYSQL_CONNECTION_TIMEOUT", 10))

MYSQL_CONFIG: Dict[str, Optional[str]] = {
    "user": os.getenv("MYSQL_USERNAME"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "host": os.getenv("MYSQL_HOSTNAME"),
    "port": os.getenv("MYSQL_PORT"),
    "database": os.getenv("MYSQL_DB_NAME"),
}

POOL_SIZE: int = int(os.getenv("MYSQL_POOL_SIZE", 10))

mysql.connector.pooling.CNX_POOL_MAXSIZE = POOL_SIZE


class MySQLPool:
    def __init__(self, pool_size: int, **kwargs: Dict[str, Any]) -> None:
        self._host: str = kwargs.get("host", "localhost")
        self._port: int = kwargs.get("port", 3306)
        self._user: str = kwargs.get("user", "root")
        self._password: str = kwargs.get("password", "")
        self._database: str = kwargs.get("database", "test")
        self.dbconfig: Dict[str, Any] = {
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
        t0 = time.time()
        while time.time() - t0 < CONNECTION_TIMEOUT:
            try:
                conn = self.pool.get_connection()
                logger.info(
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
        sql_conns.close()

    def execute(self, sql: str, params: Optional[Tuple[Any, ...]] = None) -> None:
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
        self, sql: str, params: Optional[Tuple[Any, ...]] = None
    ) -> List[Dict[str, Any]]:
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
        self, sql: str, params: Optional[Tuple[Any, ...]] = None
    ) -> Optional[Dict[str, Any]]:
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


mysql_pool = MySQLPool(POOL_SIZE, **MYSQL_CONFIG)
