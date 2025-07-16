"""Tests for slot filling functionality.

This module contains comprehensive tests for the slot filling implementation,
covering both local model-based and remote API-based approaches.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from arklex.orchestrator.NLU.core.slot import SlotFiller, create_slot_filler
from arklex.orchestrator.NLU.entities.slot_entities import Slot
from arklex.orchestrator.NLU.services.api_service import APIClientService
from arklex.orchestrator.NLU.services.model_service import ModelService
from arklex.utils.exceptions import APIError, ModelError, ValidationError


@pytest.fixture
def mock_model_service() -> ModelService:
    """Create a mock model service for testing.

    Returns:
        ModelService: A mocked model service instance.
    """
    mock = Mock(spec=ModelService)
    # Add missing methods that are used in the tests
    mock.format_verification_input = Mock()
    mock.process_verification_response = Mock()
    return mock


@pytest.fixture
def mock_api_service() -> APIClientService:
    """Create a mock API service for testing.

    Returns:
        APIClientService: A mocked API service instance.
    """
    mock = Mock(spec=APIClientService)
    # Add missing methods that are used in the tests
    mock.predict_slots = Mock()
    mock.verify_slots = Mock()
    return mock


@pytest.fixture
def sample_slots() -> list[Slot]:
    """Create sample slot objects for testing.

    Returns:
        List of sample Slot objects.
    """
    return [
        Slot(
            name="user_name",
            type="str",
            value=None,
            enum=[],
            description="User's name",
            prompt="",
            required=False,
            verified=False,
            items=None,
        ),
        Slot(
            name="user_email",
            type="str",
            value=None,
            enum=[],
            description="User's email",
            prompt="",
            required=False,
            verified=False,
            items=None,
        ),
    ]


@pytest.fixture
def model_config() -> dict[str, Any]:
    """Create sample model configuration.

    Returns:
        Dictionary containing model configuration.
    """
    return {
        "max_tokens": 1000,
        "model_name": "test-model",
        "temperature": 0.1,
    }


class TestSlotFillerInitialization:
    """Test SlotFiller initialization."""

    def test_init_with_model_service(self) -> None:
        """Test initialization with model service only.

        Should create a SlotFiller instance with local mode.
        """
        mock_model_service = Mock(spec=ModelService)
        slot_filler = SlotFiller(mock_model_service)

        assert slot_filler.model_service == mock_model_service
        assert slot_filler.api_service is None

    def test_init_with_model_and_api_service(self) -> None:
        """Test initialization with both model and API service.

        Should create a SlotFiller instance with remote mode.
        """
        mock_model_service = Mock(spec=ModelService)
        mock_api_service = Mock(spec=APIClientService)
        slot_filler = SlotFiller(mock_model_service, mock_api_service)

        assert slot_filler.model_service == mock_model_service
        assert slot_filler.api_service == mock_api_service

    def test_init_without_model_service(self) -> None:
        """Test initialization without model service.

        Should raise ValidationError.
        """
        with pytest.raises(ValidationError) as exc_info:
            SlotFiller(None)

        assert "Model service is required" in str(exc_info.value)


class TestCreateSlotFiller:
    """Test create_slot_filler function."""

    def test_create_slot_filler_with_model_service(self) -> None:
        """Test creating slot filler with model service only."""
        mock_model_service = Mock(spec=ModelService)
        slot_filler = create_slot_filler(mock_model_service)

        assert isinstance(slot_filler, SlotFiller)
        assert slot_filler.model_service == mock_model_service
        assert slot_filler.api_service is None

    def test_create_slot_filler_with_both_services(self) -> None:
        """Test creating slot filler with both services."""
        mock_model_service = Mock(spec=ModelService)
        mock_api_service = Mock(spec=APIClientService)
        slot_filler = create_slot_filler(mock_model_service, mock_api_service)

        assert isinstance(slot_filler, SlotFiller)
        assert slot_filler.model_service == mock_model_service
        assert slot_filler.api_service == mock_api_service


class TestSlotFillerFillSlotsLocal:
    """Test local slot filling functionality."""

    def test_fill_slots_local_success(
        self, mock_model_service: ModelService, model_config: dict[str, Any]
    ) -> None:
        """Test successful local slot filling.

        Args:
            mock_model_service: Mock model service instance.
            model_config: Model configuration.
        """
        slot_filler = SlotFiller(mock_model_service)

        slots = [
            Slot(
                name="user_name",
                type="str",
                value=None,
                enum=[],
                description="User's name",
                prompt="",
                required=False,
                verified=False,
                items=None,
            ),
            Slot(
                name="user_email",
                type="str",
                value=None,
                enum=[],
                description="User's email",
                prompt="",
                required=False,
                verified=False,
                items=None,
            ),
        ]
        context = "Hello, my name is John and my email is john@example.com"

        # Mock the model service methods
        mock_model_service.format_slot_input.return_value = (
            "formatted_prompt",
            "system_prompt",
        )
        mock_model_service.get_response.return_value = (
            '{"user_name": "John", "user_email": "john@example.com"}'
        )
        mock_model_service.process_slot_response.return_value = [
            Slot(
                name="user_name",
                type="str",
                value="John",
                enum=[],
                description="User's name",
                prompt="",
                required=False,
                verified=False,
                items=None,
            ),
            Slot(
                name="user_email",
                type="str",
                value="john@example.com",
                enum=[],
                description="User's email",
                prompt="",
                required=False,
                verified=False,
                items=None,
            ),
        ]

        result = slot_filler._fill_slots_local(slots, context, model_config)

        assert len(result) == 2
        assert result[0].name == "user_name"
        assert result[0].value == "John"
        assert result[1].name == "user_email"
        assert result[1].value == "john@example.com"
        mock_model_service.format_slot_input.assert_called_once_with(
            slots, context, "chat"
        )
        mock_model_service.get_response.assert_called_once_with(
            "formatted_prompt", model_config, "system_prompt"
        )
        mock_model_service.process_slot_response.assert_called_once()

    def test_fill_slots_local_process_response_error(
        self, mock_model_service: ModelService, model_config: dict[str, Any]
    ) -> None:
        """Test local slot filling with processing error.

        Args:
            mock_model_service: Mock model service instance.
            model_config: Model configuration.
        """
        slot_filler = SlotFiller(mock_model_service)

        slots = [
            Slot(
                name="user_name",
                type="str",
                value=None,
                enum=[],
                description="User's name",
                prompt="",
                required=False,
                verified=False,
                items=None,
            ),
        ]
        context = "Hello, my name is John"

        # Mock the model service methods
        mock_model_service.format_slot_input.return_value = (
            "formatted_prompt",
            "system_prompt",
        )
        mock_model_service.get_response.return_value = "invalid_response"
        mock_model_service.process_slot_response.side_effect = Exception(
            "Processing error"
        )

        with pytest.raises(ModelError) as exc_info:
            slot_filler._fill_slots_local(slots, context, model_config)

        assert "Failed to process slot filling response" in str(exc_info.value)


class TestSlotFillerFillSlotsRemote:
    """Test remote slot filling functionality."""

    def test_fill_slots_remote_success(
        self,
        mock_model_service: ModelService,
        mock_api_service: APIClientService,
        sample_slots: list[Slot],
        model_config: dict[str, Any],
    ) -> None:
        """Test successful remote slot filling.

        Args:
            mock_model_service: Mock model service instance.
            mock_api_service: Mock API service instance.
            sample_slots: Sample slot objects.
            model_config: Model configuration.
        """
        slot_filler = SlotFiller(mock_model_service, mock_api_service)

        # Mock the API service method
        mock_api_service.predict_slots.return_value = sample_slots

        result = slot_filler._fill_slots_remote(
            sample_slots, "Hello, my name is John", model_config
        )

        assert result == sample_slots
        mock_api_service.predict_slots.assert_called_once()

    def test_fill_slots_remote_with_custom_type(
        self,
        mock_model_service: ModelService,
        mock_api_service: APIClientService,
        sample_slots: list[Slot],
        model_config: dict[str, Any],
    ) -> None:
        """Test remote slot filling with custom type.

        Args:
            mock_model_service: Mock model service instance.
            mock_api_service: Mock API service instance.
            sample_slots: Sample slot objects.
            model_config: Model configuration.
        """
        slot_filler = SlotFiller(mock_model_service, mock_api_service)

        mock_api_service.predict_slots.return_value = sample_slots

        result = slot_filler._fill_slots_remote(
            sample_slots, "Hello, my name is John", model_config, "custom"
        )

        assert result == sample_slots
        mock_api_service.predict_slots.assert_called_once()

    def test_fill_slots_remote_api_error(
        self,
        mock_model_service: ModelService,
        mock_api_service: APIClientService,
        sample_slots: list[Slot],
        model_config: dict[str, Any],
    ) -> None:
        """Test remote slot filling with API error.

        Args:
            mock_model_service: Mock model service instance.
            mock_api_service: Mock API service instance.
            sample_slots: Sample slot objects.
            model_config: Model configuration.
        """
        slot_filler = SlotFiller(mock_model_service, mock_api_service)

        mock_api_service.predict_slots.side_effect = APIError("API error")

        with pytest.raises(APIError) as exc_info:
            slot_filler._fill_slots_remote(
                sample_slots, "Hello, my name is John", model_config
            )

        assert "Failed to fill slots via API" in str(exc_info.value)


class TestSlotFillerVerifySlotLocal:
    """Test local slot verification functionality."""

    def test_verify_slot_local_success(
        self, mock_model_service: ModelService, model_config: dict[str, Any]
    ) -> None:
        """Test successful local slot verification.

        Args:
            mock_model_service: Mock model service instance.
            model_config: Model configuration.
        """
        slot_filler = SlotFiller(mock_model_service)

        slot = {"name": "user_name", "value": "John"}
        chat_history = "User: Hello, my name is John"

        # Mock the model service methods
        mock_model_service.format_verification_input.return_value = "formatted_prompt"
        mock_model_service.get_response.return_value = (
            '{"is_valid": true, "reason": "Valid name"}'
        )
        mock_model_service.process_verification_response.return_value = (
            True,
            "Valid name",
        )

        result = slot_filler._verify_slot_local(slot, chat_history, model_config)

        assert result == (True, "Valid name")
        mock_model_service.format_verification_input.assert_called_once_with(
            slot, chat_history
        )
        mock_model_service.get_response.assert_called_once_with(
            "formatted_prompt", model_config
        )
        mock_model_service.process_verification_response.assert_called_once()

    def test_verify_slot_local_process_response_error(
        self, mock_model_service: ModelService, model_config: dict[str, Any]
    ) -> None:
        """Test local slot verification with processing error.

        Args:
            mock_model_service: Mock model service instance.
            model_config: Model configuration.
        """
        slot_filler = SlotFiller(mock_model_service)

        slot = {"name": "user_name", "value": "John"}
        chat_history = "User: Hello, my name is John"

        # Mock the model service methods
        mock_model_service.format_verification_input.return_value = "formatted_prompt"
        mock_model_service.get_response.return_value = "invalid_response"
        mock_model_service.process_verification_response.side_effect = Exception(
            "Processing error"
        )

        with pytest.raises(ModelError) as exc_info:
            slot_filler._verify_slot_local(slot, chat_history, model_config)

        assert "Failed to process slot verification response" in str(exc_info.value)


class TestSlotFillerVerifySlotRemote:
    """Test remote slot verification functionality."""

    def test_verify_slot_remote_success(
        self,
        mock_model_service: ModelService,
        mock_api_service: APIClientService,
        model_config: dict[str, Any],
    ) -> None:
        """Test successful remote slot verification.

        Args:
            mock_model_service: Mock model service instance.
            mock_api_service: Mock API service instance.
            model_config: Model configuration.
        """
        slot_filler = SlotFiller(mock_model_service, mock_api_service)

        slot = {"name": "user_name", "value": "John"}
        chat_history = "User: Hello, my name is John"

        mock_api_service.verify_slots.return_value = (True, "Valid name")

        result = slot_filler._verify_slot_remote(slot, chat_history, model_config)

        assert result == (True, "Valid name")
        mock_api_service.verify_slots.assert_called_once()

    def test_verify_slot_remote_api_error(
        self,
        mock_model_service: ModelService,
        mock_api_service: APIClientService,
        model_config: dict[str, Any],
    ) -> None:
        """Test remote slot verification with API error.

        Args:
            mock_model_service: Mock model service instance.
            mock_api_service: Mock API service instance.
            model_config: Model configuration.
        """
        slot_filler = SlotFiller(mock_model_service, mock_api_service)

        slot = {"name": "user_name", "value": "John"}
        chat_history = "User: Hello, my name is John"

        mock_api_service.verify_slots.side_effect = APIError("API error")

        with pytest.raises(APIError) as exc_info:
            slot_filler._verify_slot_remote(slot, chat_history, model_config)

        assert "Failed to verify slot via API" in str(exc_info.value)


class TestSlotFillerVerifySlot:
    """Test slot verification functionality."""

    def test_verify_slot_with_api_service(
        self,
        mock_model_service: ModelService,
        mock_api_service: APIClientService,
        model_config: dict[str, Any],
    ) -> None:
        """Test verify_slot with API service available.

        Args:
            mock_model_service: Mock model service instance.
            mock_api_service: Mock API service instance.
            model_config: Model configuration.
        """
        slot_filler = SlotFiller(mock_model_service, mock_api_service)

        slot = {"name": "user_name", "value": "John"}
        chat_history = "User: Hello, my name is John"

        mock_api_service.verify_slots.return_value = (True, "Valid name")

        result = slot_filler.verify_slot(slot, chat_history, model_config)

        assert result == (True, "Valid name")
        mock_api_service.verify_slots.assert_called_once()

    def test_verify_slot_without_api_service(
        self, mock_model_service: ModelService, model_config: dict[str, Any]
    ) -> None:
        """Test verify_slot without API service (uses local).

        Args:
            mock_model_service: Mock model service instance.
            model_config: Model configuration.
        """
        slot_filler = SlotFiller(mock_model_service)

        slot = {"name": "user_name", "value": "John"}
        chat_history = "User: Hello, my name is John"

        # Mock the model service methods
        mock_model_service.format_verification_input.return_value = "formatted_prompt"
        mock_model_service.get_response.return_value = (
            '{"is_valid": true, "reason": "Valid name"}'
        )
        mock_model_service.process_verification_response.return_value = (
            True,
            "Valid name",
        )

        result = slot_filler.verify_slot(slot, chat_history, model_config)

        assert result == (True, "Valid name")
        mock_model_service.format_verification_input.assert_called_once_with(
            slot, chat_history
        )
        mock_model_service.get_response.assert_called_once_with(
            "formatted_prompt", model_config
        )
        mock_model_service.process_verification_response.assert_called_once()


class TestSlotFillerFillSlots:
    """Test slot filling functionality."""

    def test_fill_slots_with_api_service(
        self,
        mock_model_service: ModelService,
        mock_api_service: APIClientService,
        sample_slots: list[Slot],
        model_config: dict[str, Any],
    ) -> None:
        """Test fill_slots with API service available.

        Args:
            mock_model_service: Mock model service instance.
            mock_api_service: Mock API service instance.
            sample_slots: Sample slot objects.
            model_config: Model configuration.
        """
        slot_filler = SlotFiller(mock_model_service, mock_api_service)

        mock_api_service.predict_slots.return_value = sample_slots

        result = slot_filler.fill_slots(
            sample_slots, "Hello, my name is John", model_config
        )

        assert result == sample_slots
        mock_api_service.predict_slots.assert_called_once()

    def test_fill_slots_without_api_service(
        self, mock_model_service: ModelService, model_config: dict[str, Any]
    ) -> None:
        """Test fill_slots without API service (uses local).

        Args:
            mock_model_service: Mock model service instance.
            model_config: Model configuration.
        """
        slot_filler = SlotFiller(mock_model_service)

        slots = [
            Slot(
                name="user_name",
                type="str",
                value=None,
                enum=[],
                description="User's name",
                prompt="",
                required=False,
                verified=False,
                items=None,
            ),
            Slot(
                name="user_email",
                type="str",
                value=None,
                enum=[],
                description="User's email",
                prompt="",
                required=False,
                verified=False,
                items=None,
            ),
        ]
        context = "Hello, my name is John and my email is john@example.com"

        # Mock the model service methods
        mock_model_service.format_slot_input.return_value = (
            "formatted_prompt",
            "system_prompt",
        )
        mock_model_service.get_response.return_value = (
            '{"user_name": "John", "user_email": "john@example.com"}'
        )
        mock_model_service.process_slot_response.return_value = [
            Slot(
                name="user_name",
                type="str",
                value="John",
                enum=[],
                description="User's name",
                prompt="",
                required=False,
                verified=False,
                items=None,
            ),
            Slot(
                name="user_email",
                type="str",
                value="john@example.com",
                enum=[],
                description="User's email",
                prompt="",
                required=False,
                verified=False,
                items=None,
            ),
        ]

        result = slot_filler.fill_slots(slots, context, model_config)

        assert len(result) == 2
        assert result[0].name == "user_name"
        assert result[0].value == "John"
        assert result[1].name == "user_email"
        assert result[1].value == "john@example.com"
        mock_model_service.format_slot_input.assert_called_once_with(
            slots, context, "chat"
        )
        mock_model_service.get_response.assert_called_once_with(
            "formatted_prompt", model_config, "system_prompt"
        )
        mock_model_service.process_slot_response.assert_called_once()

    def test_fill_slots_with_custom_type(
        self, mock_model_service: ModelService, model_config: dict[str, Any]
    ) -> None:
        """Test fill_slots with custom type parameter.

        Args:
            mock_model_service: Mock model service instance.
            model_config: Model configuration.
        """
        slot_filler = SlotFiller(mock_model_service)

        slots = [
            Slot(
                name="user_name",
                type="str",
                value=None,
                enum=[],
                description="User's name",
                prompt="",
                required=False,
                verified=False,
                items=None,
            )
        ]
        context = "Hello, my name is John"

        # Mock the model service methods
        mock_model_service.format_slot_input.return_value = (
            "formatted_prompt",
            "system_prompt",
        )
        mock_model_service.get_response.return_value = '{"user_name": "John"}'
        mock_model_service.process_slot_response.return_value = [
            Slot(
                name="user_name",
                type="str",
                value="John",
                enum=[],
                description="User's name",
                prompt="",
                required=False,
                verified=False,
                items=None,
            )
        ]

        result = slot_filler.fill_slots(slots, context, model_config, type="custom")

        assert len(result) == 1
        assert result[0].name == "user_name"
        assert result[0].value == "John"

        # Verify the custom type was passed to format_slot_input
        mock_model_service.format_slot_input.assert_called_once_with(
            slots, context, "custom"
        )

    def test_fill_slots_remote_without_api_service(
        self, mock_model_service: ModelService, model_config: dict[str, Any]
    ) -> None:
        """Test _fill_slots_remote when API service is not configured.

        Args:
            mock_model_service: Mock model service instance.
            model_config: Model configuration.
        """
        slot_filler = SlotFiller(mock_model_service)  # No API service

        slots = [
            Slot(
                name="user_name",
                type="str",
                value=None,
                enum=[],
                description="User's name",
                prompt="",
                required=False,
                verified=False,
                items=None,
            )
        ]
        context = "Hello, my name is John"

        with pytest.raises(ValidationError) as exc_info:
            slot_filler._fill_slots_remote(slots, context, model_config)

        assert "API service not configured" in str(exc_info.value)

    def test_verify_slot_remote_without_api_service(
        self, mock_model_service: ModelService, model_config: dict[str, Any]
    ) -> None:
        """Test _verify_slot_remote when API service is not configured.

        Args:
            mock_model_service: Mock model service instance.
            model_config: Model configuration.
        """
        slot_filler = SlotFiller(mock_model_service)  # No API service

        slot = {
            "name": "user_name",
            "type": "str",
            "value": "John",
            "description": "User's name",
        }
        chat_history_str = "Hello, my name is John"

        with pytest.raises(ValidationError) as exc_info:
            slot_filler._verify_slot_remote(slot, chat_history_str, model_config)

        assert "API service not configured" in str(exc_info.value)

    def test_verify_slot_exception_handling(
        self, mock_model_service: ModelService, model_config: dict[str, Any]
    ) -> None:
        """Test verify_slot exception handling.

        Args:
            mock_model_service: Mock model service instance.
            model_config: Model configuration.
        """
        from arklex.utils.exceptions import ArklexError

        slot_filler = SlotFiller(mock_model_service)

        slot = {
            "name": "user_name",
            "type": "str",
            "value": "John",
            "description": "User's name",
        }
        chat_history_str = "Hello, my name is John"

        # Mock _verify_slot_local to raise an exception
        with patch.object(slot_filler, "_verify_slot_local") as mock_verify:
            mock_verify.side_effect = Exception("Test error")

            with pytest.raises(ArklexError) as exc_info:
                slot_filler.verify_slot(slot, chat_history_str, model_config)

            assert "Operation failed in verify_slot" in str(exc_info.value)

    def test_fill_slots_exception_handling(
        self, mock_model_service: ModelService, model_config: dict[str, Any]
    ) -> None:
        """Test fill_slots exception handling.

        Args:
            mock_model_service: Mock model service instance.
            model_config: Model configuration.
        """
        from arklex.utils.exceptions import ArklexError

        slot_filler = SlotFiller(mock_model_service)

        slots = [
            Slot(
                name="user_name",
                type="str",
                value=None,
                enum=[],
                description="User's name",
                prompt="",
                required=False,
                verified=False,
                items=None,
            )
        ]
        context = "Hello, my name is John"

        # Mock _fill_slots_local to raise an exception
        with patch.object(slot_filler, "_fill_slots_local") as mock_fill:
            mock_fill.side_effect = Exception("Test error")

            with pytest.raises(ArklexError) as exc_info:
                slot_filler.fill_slots(slots, context, model_config)

            assert "Operation failed in fill_slots" in str(exc_info.value)
