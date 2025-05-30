"""Natural Language Understanding (NLU) implementation for the Arklex framework.

This module provides the core NLU functionality for intent detection and slot filling.
It includes classes for handling NLU requests either through a remote API or local implementation,
and managing slot filling operations for extracting structured information from user input.
"""

import requests
import logging
from typing import Dict, List, Any, Tuple, Optional
from dotenv import load_dotenv

from arklex.utils.slot import Slot
from arklex.orchestrator.NLU.api import nlu_api, slotfilling_api

load_dotenv()
logger = logging.getLogger(__name__)


class NLU:
    """Natural Language Understanding (NLU) class for intent detection.

    This class provides functionality for detecting intents from user input, either through
    a remote API or local implementation. It handles the communication with the NLU service
    and processes the responses.
    """

    def __init__(self, url: Optional[str]) -> None:
        """Initialize the NLU instance.

        Args:
            url (Optional[str]): URL of the remote NLU API. If None, local implementation will be used.
        """
        self.url: Optional[str] = url

    def execute(
        self,
        text: str,
        intents: Dict[str, List[Dict[str, Any]]],
        chat_history_str: str,
        llm_config: Dict[str, Any],
    ) -> str:
        """Execute intent detection on the given text.

        This method processes the input text and chat history to detect the most likely intent
        from the provided list of intents. It can use either a remote API or local implementation
        based on the configuration.

        Args:
            text (str): The input text to analyze.
            intents (Dict[str, List[Dict[str, Any]]]): Dictionary of intents with their attributes.
            chat_history_str (str): Formatted chat history string.
            llm_config (Dict[str, Any]): Configuration for the language model.

        Returns:
            str: The predicted intent.

        Note:
            If using remote API and the request fails, returns "others" as the default intent.
        """
        logger.info(f"candidates intents of NLU: {intents}")
        data: Dict[str, Any] = {
            "text": text,
            "intents": intents,
            "chat_history_str": chat_history_str,
            "model": llm_config,
        }
        if self.url:
            logger.info("Using NLU API to predict the intent")
            response: requests.Response = requests.post(
                self.url + "/predict", json=data
            )
            if response.status_code == 200:
                results: Dict[str, str] = response.json()
                pred_intent: str = results["intent"]
                logger.info(f"pred_intent is {pred_intent}")
            else:
                pred_intent: str = "others"
                logger.error("Remote Server Error when predicting NLU")
        else:
            logger.info("Using NLU function to predict the intent")
            pred_intent: str = nlu_api.predict(**data)
            logger.info(f"pred_intent is {pred_intent}")

        return pred_intent


class SlotFilling:
    """Slot filling class for extracting structured information from user input.

    This class provides functionality for verifying and extracting slot values from user input,
    either through a remote API or local implementation. It handles the communication with the
    slot filling service and processes the responses.
    """

    def __init__(self, url: Optional[str]) -> None:
        """Initialize the SlotFilling instance.

        Args:
            url (Optional[str]): URL of the remote slot filling API. If None, local implementation will be used.
        """
        self.url: Optional[str] = url

    def verify_needed(
        self, slot: Slot, chat_history_str: str, llm_config: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Verify if a slot value needs to be confirmed with the user.

        This method checks if the extracted slot value needs verification based on the chat history
        and slot configuration. It can use either a remote API or local implementation.

        Args:
            slot (Slot): The slot to verify.
            chat_history_str (str): Formatted chat history string.
            llm_config (Dict[str, Any]): Configuration for the language model.

        Returns:
            Tuple[bool, str]: A tuple containing:
                - bool: Whether verification is needed
                - str: The reasoning for the verification decision

        Note:
            If using remote API and the request fails, returns (False, "No need to verify").
        """
        logger.info(f"verify slot: {slot}")
        data: Dict[str, Any] = {
            "slot": slot.model_dump(),
            "chat_history_str": chat_history_str,
            "model": llm_config,
        }
        if self.url:
            logger.info("Using Slot Filling API to verify the slot")
            response: requests.Response = requests.post(self.url + "/verify", json=data)
            if response.status_code == 200:
                verification_needed: bool = response.json().get("verification_needed")
                thought: str = response.json().get("thought")
                logger.info(f"verify_needed is {verification_needed}")
            else:
                verification_needed: bool = False
                thought: str = "No need to verify"
                logger.error("Remote Server Error when verifying Slot Filling")
        else:
            logger.info("Using Slot Filling function to verify the slot")
            verification: Any = slotfilling_api.verify(**data)
            verification_needed: bool = verification.verification_needed
            thought: str = verification.thought
            logger.info(f"verify_needed is {verification_needed}")

        return verification_needed, thought

    def execute(
        self,
        slots: List[Slot],
        context: str,
        llm_config: Dict[str, Any],
        type: str = "chat",
    ) -> List[Slot]:
        """Execute slot filling on the given context.

        This method processes the input context to extract values for the specified slots.
        It can use either a remote API or local implementation based on the configuration.

        Args:
            slots (List[Slot]): List of slots to fill.
            context (str): The input context to analyze.
            llm_config (Dict[str, Any]): Configuration for the language model.
            type (str, optional): Type of slot filling task. Defaults to "chat".

        Returns:
            List[Slot]: List of slots with their extracted values.

        Note:
            If using remote API and the request fails, returns the original slots unchanged.
        """
        logger.info(f"extracted slots: {slots}")
        if not slots:
            return []

        data: Dict[str, Any] = {
            "slots": slots,
            "input": context,
            "type": type,
            "model": llm_config,
        }
        if self.url:
            logger.info("Using Slot Filling API to predict the slots")
            response: requests.Response = requests.post(
                self.url + "/predict", json=data
            )
            if response.status_code == 200:
                pred_slots: List[Slot] = response.json()
                logger.info(f"pred_slots is {pred_slots}")
            else:
                pred_slots: List[Slot] = slots
                logger.error("Remote Server Error when predicting Slot Filling")
        else:
            logger.info("Using Slot Filling function to predict the slots")
            pred_slots: List[Slot] = slotfilling_api.predict(**data)
            logger.info(f"pred_slots is {pred_slots}")
        return pred_slots
