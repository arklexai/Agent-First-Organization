import requests
import logging
from dotenv import load_dotenv

from arklex.utils.model_config import MODEL
from arklex.utils.slot import Slot
from arklex.orchestrator.NLU.api import nlu_api, slotfilling_api

load_dotenv()
logger = logging.getLogger(__name__)


class NLU:
    def __init__(self, url):
        self.url = url

    def execute(self, text:str, intents:dict, chat_history_str:str) -> str:
        logger.info(f"candidates intents of NLU: {intents}")
        data = {
            "text": text,
            "intents": intents,
            "chat_history_str": chat_history_str,
            "model":MODEL
        }
        if self.url:
            logger.info(f"Using NLU API to predict the intent")
            response = requests.post(self.url + "/predict", json=data)
            if response.status_code == 200:
                results = response.json()
                pred_intent = results['intent']
                logger.info(f"pred_intent is {pred_intent}")
            else:
                pred_intent = "others"
                logger.error('Remote Server Error when predicting NLU')
        else:
            logger.info(f"Using NLU function to predict the intent")
            pred_intent = nlu_api.predict(**data)
            logger.info(f"pred_intent is {pred_intent}")

        return pred_intent
    

class SlotFilling:
    def __init__(self, url):
        self.url = url

    def verify_needed(self, slot: Slot, chat_history_str:str) -> Slot:
        logger.info(f"verify slot: {slot}")
        data = {
            "slot": slot.model_dump(),
            "chat_history_str": chat_history_str
        }
        if self.url:
            logger.info(f"Using Slot Filling API to verify the slot")
            response = requests.post(self.url + "/verify", json=data)
            if response.status_code == 200:
                verification_needed = response.json().get("verification_needed")
                thought = response.json().get("thought")
                logger.info(f"verify_needed is {verification_needed}")
            else:
                verification_needed = False
                thought = "No need to verify"
                logger.error('Remote Server Error when verifying Slot Filling')
        else:
            logger.info(f"Using Slot Filling function to verify the slot")
            verification = slotfilling_api.verify(**data)
            verification_needed = verification.verification_needed
            thought = verification.thought
            logger.info(f"verify_needed is {verification_needed}")

        return verification_needed, thought

    def execute(self, slots:list, chat_history_str:str) -> dict:
        logger.info(f"extracted slots: {slots}")
        if not slots: return []
        
        data = {
            "slots": slots,
            "chat_history_str": chat_history_str
        }
        if self.url:
            logger.info(f"Using Slot Filling API to predict the slots")
            response = requests.post(self.url + "/predict", json=data)
            if response.status_code == 200:
                pred_slots = response.json()
                logger.info(f"pred_slots is {pred_slots}")
            else:
                pred_slots = slots
                logger.error('Remote Server Error when predicting Slot Filling')
        else:
            logger.info(f"Using Slot Filling function to predict the slots")
            pred_slots = slotfilling_api.predict(**data)
            logger.info(f"pred_slots is {pred_slots}")
        return pred_slots
