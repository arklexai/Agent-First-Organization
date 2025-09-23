"""Tests for NLU base module."""

import pytest
from pydantic import ValidationError
from arklex.orchestrator.NLU.core.base import (
    IntentResponse,
    SlotResponse,
    VerificationResponse,
    BaseNLU,
    BaseSlotFilling,
)


class TestIntentResponse:
    """Test cases for IntentResponse class."""

    def test_intent_response_creation(self):
        """Test creating IntentResponse with required fields."""
        response = IntentResponse(
            intent="test_intent",
            confidence=0.95,
            metadata={"key": "value"}
        )
        
        assert response.intent == "test_intent"
        assert response.confidence == 0.95
        assert response.metadata == {"key": "value"}

    def test_intent_response_default_metadata(self):
        """Test creating IntentResponse with default metadata."""
        response = IntentResponse(
            intent="test_intent",
            confidence=0.95
        )
        
        assert response.intent == "test_intent"
        assert response.confidence == 0.95
        assert response.metadata == {}

    def test_intent_response_validation(self):
        """Test IntentResponse validation."""
        # Test with valid values - should not raise
        response = IntentResponse(intent="test_intent", confidence=0.95)
        assert response.intent == "test_intent"
        assert response.confidence == 0.95


class TestSlotResponse:
    """Test cases for SlotResponse class."""

    def test_slot_response_creation(self):
        """Test creating SlotResponse with required fields."""
        response = SlotResponse(
            slot="test_slot",
            value="test_value",
            confidence=0.95,
            metadata={"key": "value"}
        )
        
        assert response.slot == "test_slot"
        assert response.value == "test_value"
        assert response.confidence == 0.95
        assert response.metadata == {"key": "value"}

    def test_slot_response_default_metadata(self):
        """Test creating SlotResponse with default metadata."""
        response = SlotResponse(
            slot="test_slot",
            value="test_value",
            confidence=0.95
        )
        
        assert response.slot == "test_slot"
        assert response.value == "test_value"
        assert response.confidence == 0.95
        assert response.metadata == {}

    def test_slot_response_validation(self):
        """Test SlotResponse validation."""
        # Test that empty slot is allowed (no validation constraint)
        response = SlotResponse(slot="", value="test_value", confidence=0.95)
        assert response.slot == ""
        assert response.value == "test_value"
        assert response.confidence == 0.95
        
        # Test that negative confidence is allowed (no validation constraint)
        response = SlotResponse(slot="test_slot", value="test_value", confidence=-0.1)
        assert response.slot == "test_slot"
        assert response.value == "test_value"
        assert response.confidence == -0.1


class TestVerificationResponse:
    """Test cases for VerificationResponse class."""

    def test_verification_response_creation(self):
        """Test creating VerificationResponse with required fields."""
        response = VerificationResponse(
            slot="test_slot",
            verified=True,
            reason="Valid value",
            metadata={"key": "value"}
        )
        
        assert response.slot == "test_slot"
        assert response.verified is True
        assert response.reason == "Valid value"
        assert response.metadata == {"key": "value"}

    def test_verification_response_default_metadata(self):
        """Test creating VerificationResponse with default metadata."""
        response = VerificationResponse(
            slot="test_slot",
            verified=True,
            reason="Valid value"
        )
        
        assert response.slot == "test_slot"
        assert response.verified is True
        assert response.reason == "Valid value"
        assert response.metadata == {}

    def test_verification_response_validation(self):
        """Test VerificationResponse validation."""
        # Test that empty slot is allowed (no validation constraint)
        response = VerificationResponse(
            slot="", 
            verified=True, 
            reason="Valid value"
        )
        assert response.slot == ""
        assert response.verified is True
        assert response.reason == "Valid value"


class TestBaseNLU:
    """Test cases for BaseNLU abstract class."""

    def test_base_nlu_is_abstract(self):
        """Test that BaseNLU cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseNLU()

    def test_base_nlu_has_predict_intent_method(self):
        """Test that BaseNLU has predict_intent method."""
        assert hasattr(BaseNLU, 'predict_intent')
        assert callable(getattr(BaseNLU, 'predict_intent'))


class TestBaseSlotFilling:
    """Test cases for BaseSlotFilling abstract class."""

    def test_base_slot_filling_is_abstract(self):
        """Test that BaseSlotFilling cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSlotFilling()

    def test_base_slot_filling_has_required_methods(self):
        """Test that BaseSlotFilling has required methods."""
        assert hasattr(BaseSlotFilling, 'verify_slot')
        assert hasattr(BaseSlotFilling, 'fill_slots')
        assert callable(getattr(BaseSlotFilling, 'verify_slot'))
        assert callable(getattr(BaseSlotFilling, 'fill_slots'))
