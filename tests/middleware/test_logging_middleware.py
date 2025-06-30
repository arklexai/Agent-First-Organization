from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI, Request
from pytest import LogCaptureFixture
from starlette.testclient import TestClient

from arklex.middleware.logging_middleware import RequestLoggingMiddleware
from arklex.utils.exceptions import NetworkError, ServiceUnavailableError, TimeoutError
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


@pytest.fixture
def app_with_middleware() -> FastAPI:
    """Create a FastAPI app with the logging middleware."""
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/test")
    async def test_endpoint(request: Request) -> dict[str, str]:
        return {"message": "test"}

    @app.get("/error")
    async def error_endpoint() -> None:
        raise ValueError("Test error")

    @app.get("/{param1}/{param2}")
    async def path_params_endpoint(
        param1: str, param2: str, request: Request
    ) -> dict[str, str]:
        return {"param1": param1, "param2": param2}

    return app


def test_middleware_adds_request_id(app_with_middleware: FastAPI) -> None:
    """Test that middleware adds request ID to response headers."""
    client = TestClient(app_with_middleware)
    response = client.get("/test")
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"] is not None


def test_middleware_logs_request_start(
    app_with_middleware: FastAPI, caplog: LogCaptureFixture
) -> None:
    """Test that middleware logs request start."""
    with caplog.at_level("INFO"):
        client = TestClient(app_with_middleware)
        client.get("/test")
        assert "Request started" in caplog.text
        assert "method" in caplog.text
        assert "url" in caplog.text


def test_middleware_logs_request_completion(
    app_with_middleware: FastAPI, caplog: LogCaptureFixture
) -> None:
    """Test that middleware logs request completion."""
    with caplog.at_level("INFO"):
        client = TestClient(app_with_middleware)
        client.get("/test")
        assert "Request completed" in caplog.text
        assert "status_code" in caplog.text
        assert "process_time" in caplog.text


def test_middleware_logs_errors(
    app_with_middleware: FastAPI, caplog: LogCaptureFixture
) -> None:
    """Test that middleware logs errors."""
    with caplog.at_level("ERROR"):
        client = TestClient(app_with_middleware)
        with pytest.raises(ValueError):
            client.get("/error")
        assert "Request failed" in caplog.text
        assert "error" in caplog.text
        assert "error_type" in caplog.text


def test_middleware_preserves_request_id(app_with_middleware: FastAPI) -> None:
    """Test that middleware preserves request ID across the request lifecycle."""
    client = TestClient(app_with_middleware)
    response = client.get("/test")
    request_id = response.headers["X-Request-ID"]

    # Make another request to ensure different request IDs
    response2 = client.get("/test")
    request_id2 = response2.headers["X-Request-ID"]

    assert request_id != request_id2


def test_middleware_handles_missing_client(
    app_with_middleware: FastAPI, caplog: LogCaptureFixture
) -> None:
    """Test that middleware handles requests without client information."""
    with caplog.at_level("INFO"):
        client = TestClient(app_with_middleware)
        # Simulate request without client info
        client.get("/test", headers={"X-Forwarded-For": "127.0.0.1"})
        assert "Request started" in caplog.text
        assert "client_host" in caplog.text


def test_middleware_handles_path_params(
    app_with_middleware: FastAPI, caplog: LogCaptureFixture
) -> None:
    """Test that middleware handles path parameters correctly (lines 114-123)."""
    with caplog.at_level("INFO"):
        client = TestClient(app_with_middleware)
        response = client.get("/value1/value2")
        assert response.status_code == 200
        assert "Request started" in caplog.text
        assert "Request completed" in caplog.text


def test_middleware_handles_request_without_path_params(
    app_with_middleware: FastAPI, caplog: LogCaptureFixture
) -> None:
    """Test that middleware handles requests without path_params attribute."""
    with caplog.at_level("INFO"):
        client = TestClient(app_with_middleware)
        response = client.get("/test")
        assert response.status_code == 200
        assert "Request started" in caplog.text
        assert "Request completed" in caplog.text


@pytest.mark.asyncio
async def test_process_request_with_retry_connection_error() -> None:
    """Test _process_request_with_retry handles ConnectionError (line 170)."""
    middleware = RequestLoggingMiddleware(Mock())
    request = Mock()
    request.method = "GET"
    request.url = "http://test.com"
    request.headers = {}

    call_next = AsyncMock(side_effect=ConnectionError("Connection failed"))
    start_time = 0.0

    with pytest.raises(NetworkError) as exc_info:
        await middleware._process_request_with_retry(request, call_next, start_time)

    assert "Connection failed" in str(exc_info.value)
    assert exc_info.value.details["error_type"] == "connection_error"


@pytest.mark.asyncio
async def test_process_request_with_retry_timeout_error() -> None:
    """Test _process_request_with_retry handles TimeoutError (line 183)."""
    middleware = RequestLoggingMiddleware(Mock())
    request = Mock()
    request.method = "GET"
    request.url = "http://test.com"
    request.headers = {}

    call_next = AsyncMock(side_effect=TimeoutError("Request timeout"))
    start_time = 0.0

    with pytest.raises(TimeoutError) as exc_info:
        await middleware._process_request_with_retry(request, call_next, start_time)

    assert "Request timeout" in str(exc_info.value)
    assert exc_info.value.details["error_type"] == "timeout_error"


@pytest.mark.asyncio
async def test_process_request_with_retry_service_unavailable_error() -> None:
    """Test _process_request_with_retry handles ServiceUnavailableError (line 196)."""
    middleware = RequestLoggingMiddleware(Mock())
    request = Mock()
    request.method = "GET"
    request.url = "http://test.com"
    request.headers = {}

    call_next = AsyncMock(side_effect=ServiceUnavailableError("Service unavailable"))
    start_time = 0.0

    with pytest.raises(ServiceUnavailableError) as exc_info:
        await middleware._process_request_with_retry(request, call_next, start_time)

    assert "Service unavailable" in str(exc_info.value)
    assert exc_info.value.details["error_type"] == "service_unavailable"


@pytest.mark.asyncio
async def test_process_request_with_retry_other_exception() -> None:
    """Test _process_request_with_retry handles other exceptions (line 196)."""
    middleware = RequestLoggingMiddleware(Mock())
    request = Mock()
    request.method = "GET"
    request.url = "http://test.com"
    request.headers = {}

    call_next = AsyncMock(side_effect=ValueError("Some other error"))
    start_time = 0.0

    with pytest.raises(ValueError) as exc_info:
        await middleware._process_request_with_retry(request, call_next, start_time)

    assert "Some other error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_process_request_with_retry_success() -> None:
    """Test _process_request_with_retry handles successful requests."""
    middleware = RequestLoggingMiddleware(Mock())
    request = Mock()
    response = Mock()
    response.status_code = 200

    call_next = AsyncMock(return_value=response)
    start_time = 0.0

    result_response, process_time = await middleware._process_request_with_retry(
        request, call_next, start_time
    )

    assert result_response == response
    assert process_time >= 0.0


def test_middleware_retryable_error_handling(
    app_with_middleware: FastAPI, caplog: LogCaptureFixture
) -> None:
    """Test that middleware properly handles RetryableError exceptions."""
    with caplog.at_level("ERROR"):
        client = TestClient(app_with_middleware)

        # Mock the _process_request_with_retry to raise a RetryableError
        with patch.object(
            RequestLoggingMiddleware,
            "_process_request_with_retry",
            side_effect=NetworkError("Network error"),
        ):
            with pytest.raises(NetworkError):
                client.get("/test")

            assert "Request failed" in caplog.text
            assert "NetworkError" in caplog.text
