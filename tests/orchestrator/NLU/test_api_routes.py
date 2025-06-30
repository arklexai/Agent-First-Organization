from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import arklex.orchestrator.NLU.api.routes as routes
from arklex.utils.exceptions import ValidationError


def test_import_routes_module() -> None:
    assert hasattr(routes, "app")
    assert hasattr(routes, "router")


# Create a FastAPI app and include the router for integration tests
def create_test_app() -> FastAPI:
    # Use the app directly from routes module
    return routes.app


# Example: test POST /nlu/predict with minimal valid data
def test_predict_intent_route() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {
        "text": "hello",
        "intents": {
            "greet": [{"definition": "Say hello", "sample_utterances": ["hello"]}]
        },
        "chat_history_str": "",
        "model": {"model_name": "test", "model_type_or_path": "test"},
    }
    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        # Mock the tuple (prompt, idx2intents_mapping)
        mock_service.format_intent_input.return_value = ("prompt", {"greet": "greet"})
        mock_service.get_model_response.return_value = "greet"
        response = client.post("/nlu/predict", json=data)
        assert response.status_code == 200
        assert response.json()["intent"] == "greet"


# Example: test POST /nlu/predict with missing required field (error case)
def test_predict_intent_route_missing_field() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {"intents": {}, "chat_history_str": "", "model": {}}
    with pytest.raises(ValidationError):
        client.post("/nlu/predict", json=data)


# Add similar basic tests for /nlu/slotfill/predict and /nlu/slotfill/verify endpoints
def test_predict_slots_route() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {
        "slots": [{"name": "user_name", "type": "str"}],
        "context": "my name is John",
        "model": {"model_name": "test", "model_type_or_path": "test"},
    }
    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        mock_service.format_slot_input.return_value = "prompt"
        mock_service.get_model_response.return_value = "response"
        mock_service.process_slot_response.return_value = [
            {"name": "user_name", "value": "John"}
        ]
        response = client.post("/slotfill/predict", json=data)
        assert response.status_code == 200
        assert response.json()[0]["name"] == "user_name"
        assert response.json()[0]["value"] == "John"


def test_verify_slots_route() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {
        "slot": {"name": "user_name", "type": "str", "value": "John"},
        "chat_history_str": "my name is John",
        "model": {"model_name": "test", "model_type_or_path": "test"},
    }
    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        mock_service.format_verification_input.return_value = "prompt"
        mock_service.get_model_response.return_value = "response"
        # Return a dict with the correct fields and types
        mock_service.process_verification_response.return_value = {
            "verification_needed": True,
            "thought": "Looks valid",
        }
        response = client.post("/slotfill/verify", json=data)
        assert response.status_code == 200
        assert response.json()["verification_needed"] is True
        assert response.json()["thought"] == "Looks valid"


# Error case for /nlu/slotfill/predict
def test_predict_slots_route_missing_field() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {"slots": [], "model": {}}
    with pytest.raises(ValidationError):
        client.post("/slotfill/predict", json=data)


# Error case for /nlu/slotfill/verify
def test_verify_slots_route_missing_field() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {"slots": [], "model": {}}
    with pytest.raises(ValidationError):
        client.post("/slotfill/verify", json=data)


@pytest.mark.asyncio
async def test_get_model_service_success() -> None:
    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        MockModelService.side_effect = None
        from arklex.orchestrator.NLU.api.routes import get_model_service

        result = get_model_service()
        assert result == mock_service
        MockModelService.assert_called_once()


@pytest.mark.asyncio
async def test_get_model_service_exception() -> None:
    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        MockModelService.side_effect = RuntimeError("Model initialization failed")
        from arklex.orchestrator.NLU.api.routes import get_model_service
        from arklex.utils.exceptions import ModelError

        with pytest.raises(ModelError):
            get_model_service()


def test_predict_intent_router_success() -> None:
    app = create_test_app()
    routes_info = [route.path for route in app.routes]
    print(f"Available routes: {routes_info}")
    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        mock_service.predict_intent.return_value = {
            "intent": "greet",
            "confidence": 0.9,
        }
        from arklex.orchestrator.NLU.api.routes import (
            predict_intent as router_predict_intent,
        )

        assert callable(router_predict_intent)


def test_fill_slots_router_success() -> None:
    app = create_test_app()
    routes_info = [route.path for route in app.routes]
    print(f"Available routes: {routes_info}")
    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        mock_service.fill_slots.return_value = {
            "slots": [{"name": "user", "value": "John"}]
        }
        from arklex.orchestrator.NLU.api.routes import (
            fill_slots as router_fill_slots,
        )

        assert callable(router_fill_slots)


def test_verify_slots_router_success() -> None:
    app = create_test_app()
    routes_info = [route.path for route in app.routes]
    print(f"Available routes: {routes_info}")
    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        mock_service.verify_slots.return_value = {
            "verified": True,
            "confidence": 0.8,
        }
        from arklex.orchestrator.NLU.api.routes import (
            verify_slots as router_verify_slots,
        )

        assert callable(router_verify_slots)


def test_predict_intent_app_missing_text_field() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {
        "intents": {"greet": []},
        "chat_history_str": "",
        "model": {"model_name": "test"},
    }
    with pytest.raises(ValidationError) as exc_info:
        client.post("/nlu/predict", json=data)
    assert "Missing required field in request" in str(exc_info.value)


def test_predict_intent_app_missing_intents_field() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {
        "text": "hello",
        "chat_history_str": "",
        "model": {"model_name": "test"},
    }
    with pytest.raises(ValidationError) as exc_info:
        client.post("/nlu/predict", json=data)
    assert "Missing required field in request" in str(exc_info.value)


def test_predict_intent_app_missing_chat_history_field() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {
        "text": "hello",
        "intents": {"greet": []},
        "model": {"model_name": "test"},
    }
    with pytest.raises(ValidationError) as exc_info:
        client.post("/nlu/predict", json=data)
    assert "Missing required field in request" in str(exc_info.value)


def test_predict_intent_app_missing_model_field() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {"text": "hello", "intents": {"greet": []}, "chat_history_str": ""}
    with pytest.raises(ValidationError) as exc_info:
        client.post("/nlu/predict", json=data)
    assert "Missing required field in request" in str(exc_info.value)


def test_predict_intent_app_general_exception() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {
        "text": "hello",
        "intents": {"greet": []},
        "chat_history_str": "",
        "model": {"model_name": "test"},
    }
    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        mock_service.format_intent_input.side_effect = RuntimeError("Model error")
        from arklex.utils.exceptions import ModelError

        with pytest.raises(ModelError):
            client.post("/nlu/predict", json=data)


def test_predict_slots_app_missing_slots_field() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {"context": "my name is John", "model": {"model_name": "test"}}
    with pytest.raises(ValidationError) as exc_info:
        client.post("/slotfill/predict", json=data)
    assert "Missing required field in request" in str(exc_info.value)


def test_predict_slots_app_missing_context_field() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {
        "slots": [{"name": "user_name", "type": "str"}],
        "model": {"model_name": "test"},
    }
    with pytest.raises(ValidationError) as exc_info:
        client.post("/slotfill/predict", json=data)
    assert "Missing required field in request" in str(exc_info.value)


def test_predict_slots_app_missing_model_field() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {
        "slots": [{"name": "user_name", "type": "str"}],
        "context": "my name is John",
    }
    with pytest.raises(ValidationError) as exc_info:
        client.post("/slotfill/predict", json=data)
    assert "Missing required field in request" in str(exc_info.value)


def test_predict_slots_app_general_exception() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {
        "slots": [{"name": "user_name", "type": "str"}],
        "context": "my name is John",
        "model": {"model_name": "test"},
    }
    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        mock_service.format_slot_input.side_effect = RuntimeError("Model error")
        from arklex.utils.exceptions import ModelError

        with pytest.raises(ModelError):
            client.post("/slotfill/predict", json=data)


def test_verify_slot_app_missing_slot_field() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {"chat_history_str": "my name is John", "model": {"model_name": "test"}}
    with pytest.raises(ValidationError) as exc_info:
        client.post("/slotfill/verify", json=data)
    assert "Missing required field in request" in str(exc_info.value)


def test_verify_slot_app_missing_chat_history_field() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {
        "slot": {"name": "user_name", "type": "str", "value": "John"},
        "model": {"model_name": "test"},
    }
    with pytest.raises(ValidationError) as exc_info:
        client.post("/slotfill/verify", json=data)
    assert "Missing required field in request" in str(exc_info.value)


def test_verify_slot_app_missing_model_field() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {
        "slot": {"name": "user_name", "type": "str", "value": "John"},
        "chat_history_str": "my name is John",
    }
    with pytest.raises(ValidationError) as exc_info:
        client.post("/slotfill/verify", json=data)
    assert "Missing required field in request" in str(exc_info.value)


def test_verify_slot_app_general_exception() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {
        "slot": {"name": "user_name", "type": "str", "value": "John"},
        "chat_history_str": "my name is John",
        "model": {"model_name": "test"},
    }
    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        mock_service.format_verification_input.side_effect = RuntimeError("Model error")
        from arklex.utils.exceptions import ModelError

        with pytest.raises(ModelError):
            client.post("/slotfill/verify", json=data)


def test_predict_intent_app_with_type_parameter() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {
        "slots": [{"name": "user_name", "type": "str"}],
        "context": "my name is John",
        "model": {"model_name": "test"},
        "type": "custom_type",
    }
    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        mock_service.format_slot_input.return_value = "prompt"
        mock_service.get_model_response.return_value = "response"
        mock_service.process_slot_response.return_value = [
            {"name": "user_name", "value": "John"}
        ]
        response = client.post("/slotfill/predict", json=data)
        assert response.status_code == 200
        mock_service.format_slot_input.assert_called_once()
        call_args = mock_service.format_slot_input.call_args
        assert call_args[0][2] == "custom_type"


def test_predict_intent_app_with_others_fallback() -> None:
    app = create_test_app()
    client = TestClient(app)
    data = {
        "text": "hello",
        "intents": {
            "greet": [{"definition": "Say hello", "sample_utterances": ["hello"]}]
        },
        "chat_history_str": "",
        "model": {"model_name": "test", "model_type_or_path": "test"},
    }
    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        mock_service.format_intent_input.return_value = (
            "prompt",
            {"greet": "greet"},
        )
        mock_service.get_model_response.return_value = "unknown_intent"
        response = client.post("/nlu/predict", json=data)
        assert response.status_code == 200
        assert response.json()["intent"] == "others"


def test_router_fill_slots_with_client() -> None:
    """Test the router /fill_slots endpoint using TestClient."""
    app = create_test_app()
    app.include_router(routes.router)
    client = TestClient(app)
    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        # SlotResponse expects 'slot' and 'value' fields
        mock_service.fill_slots = AsyncMock(
            return_value={"slot": "user", "value": "John", "confidence": 0.8}
        )
        response = client.post(
            "/fill_slots", params={"text": "my name is John", "intent": "greet"}
        )
        assert response.status_code == 200
        assert response.json()["slot"] == "user"
        assert response.json()["value"] == "John"


def test_router_verify_slots_with_client() -> None:
    """Test the router /verify_slots endpoint using TestClient."""
    app = create_test_app()
    app.include_router(routes.router)
    client = TestClient(app)
    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        # VerificationResponse expects 'slot' and 'reason' fields
        mock_service.verify_slots = AsyncMock(
            return_value={
                "slot": "user",
                "reason": "valid",
                "verified": True,
                "confidence": 0.8,
            }
        )
        response = client.post(
            "/verify_slots",
            params={"text": "my name is John"},
            json={"slots": {"user": "John"}},
        )
        assert response.status_code == 200
        assert response.json()["slot"] == "user"
        assert response.json()["reason"] == "valid"
        assert response.json()["verified"] is True


def test_router_predict_intent_with_client() -> None:
    """Test the router /predict_intent endpoint using TestClient."""
    app = create_test_app()
    app.include_router(routes.router)
    client = TestClient(app)
    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        # IntentResponse expects 'intent' and 'confidence' fields
        mock_service.predict_intent = AsyncMock(
            return_value={"intent": "greet", "confidence": 0.9}
        )
        response = client.post("/predict_intent", params={"text": "hello"})
        assert response.status_code == 200
        assert response.json()["intent"] == "greet"
        assert response.json()["confidence"] == 0.9


@pytest.mark.asyncio
async def test_get_model_service_exception_with_model_error() -> None:
    """Test get_model_service when ModelService raises an exception, ensuring ModelError is raised."""
    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        MockModelService.side_effect = Exception("Model initialization failed")
        # Mock the LOG_MESSAGES to avoid KeyError
        with patch(
            "arklex.orchestrator.NLU.api.routes.LOG_MESSAGES",
            {"ERROR": {"INITIALIZATION_ERROR": "Model initialization error"}},
        ):
            from arklex.orchestrator.NLU.api.routes import get_model_service
            from arklex.utils.exceptions import ModelError

            with pytest.raises(ModelError) as exc_info:
                get_model_service()

            assert "Failed to initialize model service" in str(exc_info.value)
            assert exc_info.value.details["error"] == "Model initialization failed"
            assert exc_info.value.details["operation"] == "initialization"


# Test router endpoints with exceptions
@pytest.mark.asyncio
async def test_router_predict_intent_exception() -> None:
    """Test the router /predict_intent endpoint when ModelService raises an exception."""
    from arklex.orchestrator.NLU.api.routes import (
        predict_intent as router_predict_intent,
    )
    from arklex.utils.exceptions import ArklexError

    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        mock_service.predict_intent = AsyncMock(side_effect=RuntimeError("Model error"))

        with pytest.raises(ArklexError):
            await router_predict_intent("hello", mock_service)


@pytest.mark.asyncio
async def test_router_fill_slots_exception() -> None:
    """Test the router /fill_slots endpoint when ModelService raises an exception."""
    from arklex.orchestrator.NLU.api.routes import fill_slots as router_fill_slots
    from arklex.utils.exceptions import ArklexError

    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        mock_service.fill_slots = AsyncMock(side_effect=RuntimeError("Model error"))

        with pytest.raises(ArklexError):
            await router_fill_slots("my name is John", "greet", mock_service)


@pytest.mark.asyncio
async def test_router_verify_slots_exception() -> None:
    """Test the router /verify_slots endpoint when ModelService raises an exception."""
    from arklex.orchestrator.NLU.api.routes import verify_slots as router_verify_slots
    from arklex.utils.exceptions import ArklexError

    with patch("arklex.orchestrator.NLU.api.routes.ModelService") as MockModelService:
        mock_service = MockModelService.return_value
        mock_service.verify_slots = AsyncMock(side_effect=RuntimeError("Model error"))

        slots = {"user": "John"}
        with pytest.raises(ArklexError):
            await router_verify_slots("my name is John", slots, mock_service)
