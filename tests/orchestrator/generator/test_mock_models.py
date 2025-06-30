"""Mock language models for testing LLM-dependent functionality in the generator module.

This module provides mock implementations of language models that can be used
in tests to simulate LLM responses without requiring actual API calls.
"""

import random
from typing import Any
from unittest.mock import Mock

# --- Mock Classes ---


class MockLanguageModel:
    """Mock language model for testing LLM-dependent functionality.

    This mock simulates the behavior of a real language model by returning
    predefined responses based on the input prompts.
    """

    def __init__(self, responses: dict | None = None) -> None:
        """Initialize the mock model with predefined responses.

        Args:
            responses (Optional[dict]): Dictionary mapping prompt patterns to responses
        """
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = None

    def invoke(self, messages: list[Any]) -> Mock:
        """Mock the invoke method of a language model.

        Args:
            messages (List[Any]): List of messages to process or a string prompt

        Returns:
            Mock: Mock response object with content attribute
        """
        self.call_count += 1

        # Handle both string prompts and message lists
        if isinstance(messages, str):
            self.last_prompt = messages
        elif messages and len(messages) > 0:
            if hasattr(messages[0], "content"):
                self.last_prompt = messages[0].content
            elif isinstance(messages[0], dict):
                self.last_prompt = messages[0].get("content", "")
            else:
                self.last_prompt = str(messages[0])
        else:
            self.last_prompt = ""

        # Create a mock response
        response = Mock()

        # Determine response based on prompt content
        if "intent" in self.last_prompt.lower():
            response.content = "User inquires about product information"
        elif (
            "task" in self.last_prompt.lower()
            and "generate" in self.last_prompt.lower()
        ):
            response.content = """[
                {
                    "task": "Customer Support",
                    "intent": "Provide customer assistance",
                    "description": "Handle customer inquiries and support requests"
                },
                {
                    "task": "Product Information",
                    "intent": "Provide product details",
                    "description": "Answer questions about product specifications and availability"
                }
            ]"""
        elif (
            "check" in self.last_prompt.lower()
            and "breakdown" in self.last_prompt.lower()
        ):
            response.content = '{"answer": "no"}'
        elif (
            "generate" in self.last_prompt.lower()
            and "practice" in self.last_prompt.lower()
        ):
            response.content = """[
                {
                    "task": "Execute Customer Support",
                    "description": "Handle customer inquiries professionally"
                },
                {
                    "task": "Provide Product Information",
                    "description": "Share accurate product details"
                }
            ]"""
        else:
            response.content = "Default response for task generation"

        return response

    def generate(self, messages: list[Any]) -> Mock:
        """Mock the generate method of a language model.

        Args:
            messages (List[Any]): List of messages to process (can be nested)

        Returns:
            Mock: Mock response object with generations attribute
        """
        self.call_count += 1

        # Handle nested list structure (messages might be [[message1, message2]])
        if messages and isinstance(messages[0], list):
            messages = messages[0]

        # Extract prompt from messages
        if messages and len(messages) > 0:
            if hasattr(messages[0], "content"):
                self.last_prompt = messages[0].content
            elif isinstance(messages[0], dict):
                self.last_prompt = messages[0].get("content", "")
            else:
                self.last_prompt = str(messages[0])
        else:
            self.last_prompt = ""

        # Create a mock response with generations attribute
        response = Mock()
        response.generations = [[Mock()]]  # Nested structure: [[generation]]
        response.generations[0][0].text = self.invoke(messages).content
        return response


class MockLanguageModelWithErrors:
    """Mock language model that simulates various error conditions.

    This mock can be configured to raise different types of exceptions
    to test error handling in the generator components.
    """

    def __init__(self, error_type: str = "none", error_rate: float = 0.0) -> None:
        """Initialize the mock model with error configuration.

        Args:
            error_type (str): Type of error to simulate ("none", "timeout", "invalid_response", "api_error")
            error_rate (float): Probability of error occurring (0.0 to 1.0)
        """
        self.error_type = error_type
        self.error_rate = error_rate
        self.call_count = 0
        self.base_model = MockLanguageModel()

    def invoke(self, messages: list[Any]) -> Mock:
        """Mock the invoke method with potential errors.

        Args:
            messages (List[Any]): List of messages to process or a string prompt

        Returns:
            Mock: Mock response object or raises exception

        Raises:
            Exception: Various types of exceptions based on error_type
        """
        self.call_count += 1

        # Simulate error based on error_rate
        if random.random() < self.error_rate:
            if self.error_type == "timeout":
                raise TimeoutError("Request timed out")
            elif self.error_type == "invalid_response":
                raise ValueError("Invalid response format")
            elif self.error_type == "api_error":
                raise Exception("API service unavailable")
            elif self.error_type == "rate_limit":
                raise Exception("Rate limit exceeded")
            else:
                raise Exception(f"Unknown error: {self.error_type}")

        # Return normal response if no error
        return self.base_model.invoke(messages)

    def generate(self, messages: list[Any]) -> Mock:
        """Mock the generate method with potential errors.

        Args:
            messages (List[Any]): List of messages to process

        Returns:
            Mock: Mock response object or raises exception

        Raises:
            Exception: Various types of exceptions based on error_type
        """
        self.call_count += 1

        # Simulate error based on error_rate
        if random.random() < self.error_rate:
            if self.error_type == "timeout":
                raise TimeoutError("Request timed out")
            elif self.error_type == "invalid_response":
                raise ValueError("Invalid response format")
            elif self.error_type == "api_error":
                raise Exception("API service unavailable")
            elif self.error_type == "rate_limit":
                raise Exception("Rate limit exceeded")

        # Return normal response if no error
        return self.base_model.generate(messages)


class MockLanguageModelWithCustomResponses:
    """Mock language model that returns custom responses based on input patterns.

    This mock allows for fine-grained control over responses for specific test scenarios.
    """

    def __init__(self) -> None:
        """Initialize the mock model."""
        self.responses = {}
        self.call_count = 0

    def add_response(self, prompt_pattern: str, response: str) -> None:
        """Add a custom response for a specific prompt pattern.

        Args:
            prompt_pattern (str): Pattern to match in prompts
            response (str): Response to return for matching prompts
        """
        self.responses[prompt_pattern] = response

    def invoke(self, messages: list[Any]) -> Mock:
        """Mock the invoke method with custom responses.

        Args:
            messages (List[Any]): List of messages to process or a string prompt

        Returns:
            Mock: Mock response object with content attribute
        """
        self.call_count += 1

        # Handle both string prompts and message lists
        if isinstance(messages, str):
            prompt = messages
        elif messages and len(messages) > 0:
            if hasattr(messages[0], "content"):
                prompt = messages[0].content
            elif isinstance(messages[0], dict):
                prompt = messages[0].get("content", "")
            else:
                prompt = str(messages[0])
        else:
            prompt = ""

        # Check for custom responses
        for pattern, response in self.responses.items():
            if pattern.lower() in prompt.lower():
                mock_response = Mock()
                mock_response.content = response
                return mock_response

        # Default response
        mock_response = Mock()
        mock_response.content = "Default custom response"
        return mock_response

    def generate(self, messages: list[Any]) -> Mock:
        """Mock the generate method with custom responses.

        Args:
            messages (List[Any]): List of messages to process (can be nested)

        Returns:
            Mock: Mock response object with generations attribute
        """
        self.call_count += 1

        # Handle nested list structure (messages might be [[message1, message2]])
        if messages and isinstance(messages[0], list):
            messages = messages[0]

        # Extract prompt from messages
        if messages and len(messages) > 0:
            if hasattr(messages[0], "content"):
                prompt = messages[0].content
            elif isinstance(messages[0], dict):
                prompt = messages[0].get("content", "")
            else:
                prompt = str(messages[0])
        else:
            prompt = ""

        # Check for custom responses
        for pattern, response in self.responses.items():
            if pattern.lower() in prompt.lower():
                mock_response = Mock()
                mock_response.generations = [
                    [Mock()]
                ]  # Nested structure: [[generation]]
                mock_response.generations[0][0].text = response
                return mock_response

        # Default response
        mock_response = Mock()
        mock_response.generations = [[Mock()]]  # Nested structure: [[generation]]
        mock_response.generations[0][0].text = "Default custom response"
        return mock_response


# --- Factory Functions ---


def create_mock_model_for_task_generation() -> MockLanguageModelWithCustomResponses:
    """Create a mock model specifically for task generation tests."""
    model = MockLanguageModelWithCustomResponses()

    # Add responses for task generation scenarios
    model.add_response(
        "generate tasks",
        """[
            {
                "intent": "User wants to search for products",
                "task": "Product Search and Discovery"
            },
            {
                "intent": "User needs customer support",
                "task": "Customer Support Assistance"
            },
            {
                "intent": "User wants to place an order",
                "task": "Order Processing"
            }
        ]""",
    )

    model.add_response("breakdown", "true")

    model.add_response(
        "generate steps",
        """[
            {
                "description": "Identify user's search criteria",
                "step_id": "step_1"
            },
            {
                "description": "Search product database",
                "step_id": "step_2"
            },
            {
                "description": "Present results to user",
                "step_id": "step_3"
            }
        ]""",
    )

    return model


def create_mock_model_for_intent_generation() -> MockLanguageModelWithCustomResponses:
    """Create a mock model specifically for intent generation tests."""
    model = MockLanguageModelWithCustomResponses()

    # Add responses for intent generation scenarios
    model.add_response(
        "task name: product search", "User inquires about product search and discovery"
    )

    model.add_response(
        "task name: customer support", "User needs customer support assistance"
    )

    model.add_response("task name: order processing", "User wants to place an order")

    return model


def create_mock_model_for_best_practices() -> MockLanguageModelWithCustomResponses:
    """Create a mock model specifically for best practice generation tests."""
    model = MockLanguageModelWithCustomResponses()

    # Add responses for best practice scenarios
    model.add_response(
        "best practice",
        """{
            "practice_id": "bp_001",
            "name": "Efficient Task Processing",
            "description": "Process tasks efficiently with proper resource allocation",
            "steps": [
                {"description": "Analyze task requirements", "step_id": "step_1"},
                {"description": "Allocate appropriate resources", "step_id": "step_2"},
                {"description": "Execute task with monitoring", "step_id": "step_3"}
            ],
            "rationale": "Ensures optimal resource utilization",
            "examples": ["Customer inquiry processing", "Order management"],
            "priority": 4,
            "category": "efficiency"
        }""",
    )

    return model
