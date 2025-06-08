"""Tool test utilities for the Arklex framework.

This module provides test utilities for validating tool behavior in the Arklex
framework. It includes test orchestrators for Shopify tools, with validation
methods to ensure correct task graph paths and node status.
"""

import os
import json
from typing import Any, Dict, List

from tests.utils.utils import MockOrchestrator, MockResourceInitializer
from arklex.orchestrator.NLU.core.slot import SlotFiller
import logging

logger = logging.getLogger(__name__)


class ShopifyToolOrchestrator(MockOrchestrator):
    def __init__(self, config_file_path: str) -> None:
        """Initialize the Shopify tool orchestrator.

        Args:
            config_file_path (str): Path to the configuration file.
        """
        fixed_args: str = os.environ.get("SHOPIFY_FIXED_ARGS", "{}")
        self.fixed_args: Dict[str, Any] = json.loads(fixed_args)
        super().__init__(config_file_path, self.fixed_args)
        self.resource_initializer = MockResourceInitializer()

    def _validate_result(
        self,
        test_case: Dict[str, Any],
        history: List[Dict[str, str]],
        params: Dict[str, Any],
    ) -> None:
        """Validate the test results for Shopify tools.

        This function only checks that the assistant's response is non-empty.
        """
        assistant_records: List[Dict[str, str]] = [
            message for message in history if message["role"] == "assistant"
        ]
        for record in assistant_records:
            assert record["content"] != ""

    def initialize_slotfillapi(self, slotsfillapi: str) -> SlotFiller:
        """Initialize the slot filling API.

        Args:
            slotsfillapi: API endpoint for slot filling

        Returns:
            Initialized SlotFiller instance
        """
        if not isinstance(slotsfillapi, str):
            logger.error("slotsfillapi must be a string")
            return None
        if not slotsfillapi:
            logger.warning(
                "slotsfillapi is empty, using local model-based slot filling"
            )
            return SlotFiller(None)
        logger.info(f"Initializing SlotFiller with API URL: {slotsfillapi}")
        return SlotFiller(slotsfillapi)
