import pytest
from fastapi import FastAPI, Request
from starlette.testclient import TestClient
from arklex.middleware.logging_middleware import RequestLoggingMiddleware
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


@pytest.fixture
def app_with_middleware():
    """Create a FastAPI app with the logging middleware."""
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/test")
    async def test_endpoint(request: Request):
        return {"message": "test"}

    @app.get("/error")
    async def error_endpoint():
        raise ValueError("Test error")

    return app


def test_middleware_adds_request_id(app_with_middleware):
    """Test that middleware adds request ID to response headers."""
    client = TestClient(app_with_middleware)
    response = client.get("/test")
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"] is not None


def test_middleware_logs_request_start(app_with_middleware, caplog):
    """Test that middleware logs request start."""
    with caplog.at_level("INFO"):
        client = TestClient(app_with_middleware)
        client.get("/test")
        assert "Request started" in caplog.text
        assert "method" in caplog.text
        assert "url" in caplog.text


def test_middleware_logs_request_completion(app_with_middleware, caplog):
    """Test that middleware logs request completion."""
    with caplog.at_level("INFO"):
        client = TestClient(app_with_middleware)
        client.get("/test")
        assert "Request completed" in caplog.text
        assert "status_code" in caplog.text
        assert "process_time" in caplog.text


def test_middleware_logs_errors(app_with_middleware, caplog):
    """Test that middleware logs errors."""
    with caplog.at_level("ERROR"):
        client = TestClient(app_with_middleware)
        with pytest.raises(ValueError):
            client.get("/error")
        assert "Request failed" in caplog.text
        assert "error" in caplog.text
        assert "error_type" in caplog.text


def test_middleware_preserves_request_id(app_with_middleware):
    """Test that middleware preserves request ID across the request lifecycle."""
    client = TestClient(app_with_middleware)
    response = client.get("/test")
    request_id = response.headers["X-Request-ID"]

    # Make another request to ensure different request IDs
    response2 = client.get("/test")
    request_id2 = response2.headers["X-Request-ID"]

    assert request_id != request_id2


def test_middleware_handles_missing_client(app_with_middleware, caplog):
    """Test that middleware handles requests without client information."""
    with caplog.at_level("INFO"):
        client = TestClient(app_with_middleware)
        # Simulate request without client info
        response = client.get("/test", headers={"X-Forwarded-For": "127.0.0.1"})
        assert "Request started" in caplog.text
        assert "client_host" in caplog.text
