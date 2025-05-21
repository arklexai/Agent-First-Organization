import requests
import logging
from typing import Dict, List, Any, Tuple, Optional
from dotenv import load_dotenv

from arklex.utils.slot import Slot
from arklex.orchestrator.NLU.api import nlu_api, slotfilling_api

load_dotenv()
logger = logging.getLogger(__name__)


class NLU:
    def __init__(self, url: Optional[str]) -> None:
        self.url: Optional[str] = url

    def execute(
        self,
        text: str,
        intents: Dict[str, List[Dict[str, Any]]],
        chat_history_str: str,
        llm_config: Dict[str, Any],
    ) -> str:
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
    def __init__(self, url: Optional[str]) -> None:
        self.url: Optional[str] = url

    def verify_needed(
        self, slot: Slot, chat_history_str: str, llm_config: Dict[str, Any]
    ) -> Tuple[bool, str]:
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
