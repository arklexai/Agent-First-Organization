"""Request logging middleware for the Arklex framework.

This module provides middleware for logging HTTP requests and responses,
including request tracking, timing, and error handling with retry mechanisms.
"""

import time
import traceback
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from tenacity import retry, stop_after_attempt, wait_exponential

from arklex.utils.exceptions import (
    NetworkError,
    RetryableError,
    ServiceUnavailableError,
    TimeoutError,
)
from arklex.utils.logging_utils import LogContext


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses.

    This middleware provides:
    - Request ID generation and tracking
    - Request/response logging with context
    - Error handling with retry mechanisms
    - Performance timing
    """

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the middleware.

        Args:
            app: The ASGI application.
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and log relevant information.

        Args:
            request: The incoming request.
            call_next: The next middleware in the chain.

        Returns:
            The response from the application.

        Raises:
            RetryableError: If the request fails with a retryable error.
            Exception: For other types of errors.
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Create log context with enhanced request information
        log_context = LogContext(
            __name__,
            level="INFO",
            base_context={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_host": request.client.host if request.client else None,
                "client_port": request.client.port if request.client else None,
                "headers": dict(request.headers),
                "query_params": dict(request.query_params),
                "path_params": dict(request.path_params)
                if hasattr(request, "path_params")
                else {},
                "cookies": dict(request.cookies),
                "content_type": request.headers.get("content-type"),
                "user_agent": request.headers.get("user-agent"),
            },
        )

        # Log request start with standardized message
        log_context.info(
            f"Request started: {request_id} | method={request.method} | url={request.url} | client_host={request.client.host if request.client else None} | client_port={request.client.port if request.client else None}",
            method=request.method,
            url=str(request.url),
            client_host=request.client.host if request.client else None,
            client_port=request.client.port if request.client else None,
        )

        start_time = time.time()
        try:
            # Process the request with retry mechanism for retryable errors
            response, process_time = await self._process_request_with_retry(
                request, call_next, start_time
            )

            # Log request completion with standardized message
            log_context.info(
                f"Request completed: {request_id} | status_code={response.status_code} | process_time={process_time:.4f}s | content_type={response.headers.get('content-type')} | response_size={len(response.body) if hasattr(response, 'body') else 0}",
                status_code=response.status_code,
                process_time=process_time,
                response_headers=dict(response.headers),
                response_size=len(response.body) if hasattr(response, "body") else 0,
                content_type=response.headers.get("content-type"),
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            return response

        except RetryableError as e:
            # Log retryable error with standardized message
            log_context.error(
                f"Request failed: {str(e)} | error_type={type(e).__name__} | process_time={time.time() - start_time:.4f}s",
                error_type=type(e).__name__,
                error_details=getattr(e, "details", None),
                max_retries=e.max_retries,
                stack_trace=traceback.format_exc(),
                process_time=time.time() - start_time,
                exc_info=e,
            )
            raise

        except Exception as e:
            # Log unexpected error with standardized message
            log_context.error(
                f"Request failed: {str(e)} | error_type={type(e).__name__} | process_time={time.time() - start_time:.4f}s",
                error_type=type(e).__name__,
                error_details=getattr(e, "details", None),
                stack_trace=traceback.format_exc(),
                process_time=time.time() - start_time,
                exc_info=e,
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            lambda e: isinstance(
                e, NetworkError | TimeoutError | ServiceUnavailableError
            )
        ),
    )
    async def _process_request_with_retry(
        self, request: Request, call_next: Callable, start_time: float
    ) -> tuple[Response, float]:
        """Process the request with retry mechanism for retryable errors.

        Args:
            request: The incoming request.
            call_next: The next middleware in the chain.
            start_time: The time when the request started.

        Returns:
            Tuple[Response, float]: The response and process time.

        Raises:
            RetryableError: If the request fails with a retryable error.
            Exception: For other types of errors.
        """
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            return response, process_time
        except Exception as e:
            # Convert certain exceptions to retryable errors with enhanced context
            if isinstance(e, ConnectionError):
                raise NetworkError(
                    str(e),
                    details={
                        "error_type": "connection_error",
                        "original_error": str(e),
                        "request_info": {
                            "method": request.method,
                            "url": str(request.url),
                            "headers": dict(request.headers),
                        },
                    },
                ) from e
            elif isinstance(e, TimeoutError):
                raise TimeoutError(
                    str(e),
                    details={
                        "error_type": "timeout_error",
                        "original_error": str(e),
                        "request_info": {
                            "method": request.method,
                            "url": str(request.url),
                            "headers": dict(request.headers),
                        },
                    },
                ) from e
            elif isinstance(e, ServiceUnavailableError):
                raise ServiceUnavailableError(
                    str(e),
                    details={
                        "error_type": "service_unavailable",
                        "original_error": str(e),
                        "request_info": {
                            "method": request.method,
                            "url": str(request.url),
                            "headers": dict(request.headers),
                        },
                    },
                ) from e
            raise
