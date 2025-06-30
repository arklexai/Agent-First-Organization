"""Logging configuration for the Arklex framework.

This module provides centralized logging configuration, including:
- Log level management
- Log formatting
- Log rotation
- Request ID tracking
- Context-aware logging
- JSON logging support
"""

import contextlib
import json
import logging
import logging.handlers
import platform
import socket
from datetime import datetime
from pathlib import Path
from typing import Any

# Constants
DEFAULT_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
DEFAULT_LOG_LEVEL = logging.INFO
MAX_BYTES = 1024 * 1024  # 1MB for production, but tests will override this
BACKUP_COUNT = 5

# Standard log levels for different types of events
LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,  # System is unusable
    "ERROR": logging.ERROR,  # Error events that might still allow the application to continue
    "WARNING": logging.WARNING,  # Warning events that might indicate a problem
    "INFO": logging.INFO,  # General information about program execution
    "DEBUG": logging.DEBUG,  # Detailed information for debugging
}

# Module-specific log levels with more granular control
MODULE_LOG_LEVELS = {
    # Third-party modules
    "urllib3": logging.WARNING,
    "requests": logging.WARNING,
    "fastapi": logging.INFO,
    "uvicorn": logging.INFO,
    "sqlalchemy": logging.WARNING,
    "httpx": logging.WARNING,
    "websockets": logging.WARNING,
    # Application modules
    "arklex.api": logging.INFO,
    "arklex.api.routes": logging.INFO,
    "arklex.api.middleware": logging.INFO,
    "arklex.db": logging.WARNING,
    "arklex.db.models": logging.INFO,
    "arklex.db.migrations": logging.INFO,
    "arklex.cache": logging.DEBUG,
    "arklex.cache.redis": logging.DEBUG,
    "arklex.utils": logging.INFO,
    "arklex.utils.logging": logging.INFO,
    "arklex.utils.exceptions": logging.INFO,
    "arklex.services": logging.INFO,
    "arklex.services.agents": logging.INFO,
    "arklex.services.tasks": logging.INFO,
}


class RequestIdFilter(logging.Filter):
    """Filter to add request_id to log records."""

    def __init__(self, request_id: str = "N/A") -> None:
        """Initialize the filter.

        Args:
            request_id: The request ID to add to log records
        """
        super().__init__()
        self.request_id = request_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request_id to the log record.

        Args:
            record: The log record to modify.

        Returns:
            True to allow the record to be processed.
        """
        record.request_id = self.request_id
        return True


class ContextFilter(logging.Filter):
    """Filter to add context information to log records."""

    def __init__(self, context: dict[str, Any] | None = None) -> None:
        """Initialize the filter.

        Args:
            context: Dictionary of context information to add to log records
        """
        super().__init__()
        self.context = context or {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context information to the log record.

        Args:
            record: The log record to modify.

        Returns:
            True to allow the record to be processed.
        """
        record.context = self.context
        return True


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON strings after parsing the LogRecord."""

    def __init__(self, include_hostname: bool = True) -> None:
        """Initialize the JSON formatter.

        Args:
            include_hostname: Whether to include hostname in log records
        """
        super().__init__()
        self.include_hostname = include_hostname
        self.hostname = None
        if include_hostname:
            with contextlib.suppress(Exception):
                self.hostname = socket.gethostname()

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON string representation of the log record.
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread,
            "thread_name": record.threadName,
        }

        # Add system information
        if self.include_hostname:
            log_data.update(
                {
                    "hostname": self.hostname,
                    "platform": platform.platform(),
                    "python_version": platform.python_version(),
                }
            )

        # Add request tracking information
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        # Add context information
        if hasattr(record, "context"):
            log_data["context"] = record.context

        # Add exception information
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_data)


def setup_logging(
    log_dir: str | None = None,
    log_level: str | int = DEFAULT_LOG_LEVEL,
    log_format: str = DEFAULT_LOG_FORMAT,
    app_name: str = "arklex",
    use_json: bool = False,
    max_bytes: int = MAX_BYTES,
    include_hostname: bool = True,
) -> None:
    """Set up logging configuration for the application.

    Args:
        log_dir: Directory to store log files. If None, logs will be stored in 'logs' directory.
        log_level: Logging level (default: INFO).
        log_format: Format string for log messages.
        app_name: Name of the application for log file naming.
        use_json: Whether to use JSON formatting for logs.
        max_bytes: Maximum bytes for log rotation.
        include_hostname: Whether to include hostname in log records.
    """
    root_log_context = logging.getLogger()
    # Remove all existing handlers
    for handler in list(root_log_context.handlers):
        root_log_context.removeHandler(handler)

    # Convert string log level to integer if needed
    if isinstance(log_level, str):
        log_level = LOG_LEVELS.get(log_level.upper(), DEFAULT_LOG_LEVEL)

    # Set root log_context level
    root_log_context.setLevel(log_level)

    # Create formatter
    if use_json:
        formatter = JSONFormatter(include_hostname=include_hostname)
    else:
        formatter = logging.Formatter(log_format)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestIdFilter())
    root_log_context.addHandler(console_handler)

    # Create file handler if log directory is provided
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / f"{app_name}.log",
            maxBytes=max_bytes,
            backupCount=BACKUP_COUNT,
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RequestIdFilter())
        root_log_context.addHandler(file_handler)

    # Set module-specific log levels
    for module_name, module_level in MODULE_LOG_LEVELS.items():
        module_log_context = logging.getLogger(module_name)
        module_log_context.setLevel(module_level)
