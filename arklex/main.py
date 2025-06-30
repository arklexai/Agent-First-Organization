"""Main application module for the Arklex framework.

This module initializes the FastAPI application and configures:
- Logging
- Middleware
- Error handling
- CORS
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from arklex.middleware.logging_middleware import RequestLoggingMiddleware
from arklex.orchestrator.NLU.api.routes import router as nlu_router
from arklex.utils.exceptions import (
    ArklexError,
    AuthenticationError,
    RateLimitError,
    ResourceNotFoundError,
    RetryableError,
)
from arklex.utils.logging_utils import LogContext

# Initialize logging with JSON formatting
log_context = LogContext(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI application."""
    # Startup
    log_context.info("Application startup")
    yield
    # Shutdown
    log_context.info("Application shutdown")


# Create FastAPI app
app = FastAPI(
    title="Arklex API",
    description="Arklex API Service",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)


@app.exception_handler(ArklexError)
async def arklex_exception_handler(request: Request, exc: ArklexError) -> JSONResponse:
    """Handle Arklex-specific exceptions.

    Args:
        request: The incoming request.
        exc: The exception that was raised.

    Returns:
        JSONResponse with error details and appropriate status code.
    """
    error_context = {
        "error_code": exc.error_code,
        "details": exc.details,
        "request_id": getattr(request.state, "request_id", "N/A"),
        "path": request.url.path,
        "method": request.method,
    }

    # Log the error with context
    log_context.error(
        f"Arklex error occurred: {str(exc)}", context=error_context, exc_info=exc
    )

    # Prepare error response
    error_response = {
        "error": {
            "code": exc.error_code,
            "message": str(exc),
            "details": exc.details,
        }
    }

    # Add extra message for user-facing errors
    if hasattr(exc, "extra_message") and exc.extra_message:
        error_response["error"]["guidance"] = exc.extra_message

    # Set appropriate status code based on error type
    status_code = 400  # Default status code
    if isinstance(exc, AuthenticationError):
        status_code = 401
    elif isinstance(exc, ResourceNotFoundError):
        status_code = 404
    elif isinstance(exc, RateLimitError):
        status_code = 429
    elif isinstance(exc, RetryableError):
        status_code = 503

    return JSONResponse(status_code=status_code, content=error_response)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions.

    Args:
        request: The incoming request.
        exc: The exception that was raised.

    Returns:
        JSONResponse with generic error details and 500 status code.
    """
    error_context = {
        "error_type": type(exc).__name__,
        "request_id": getattr(request.state, "request_id", "N/A"),
        "path": request.url.path,
        "method": request.method,
    }

    # Log the unexpected error with context
    log_context.error(
        f"Unexpected error occurred: {str(exc)}", context=error_context, exc_info=exc
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"type": type(exc).__name__},
            }
        },
    )


# Health check endpoint
@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        dict: A dictionary with the health status
    """
    log_context.info("Health check requested")
    return {"status": "healthy"}


# Include routers
app.include_router(nlu_router, prefix="/api/nlu", tags=["NLU"])
