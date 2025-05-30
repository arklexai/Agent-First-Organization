"""API implementation for Natural Language Understanding (NLU) services.

This module provides the API layer for NLU services, including intent detection and slot filling.
It includes classes for handling model interactions, formatting inputs and outputs,
and providing FastAPI endpoints for remote access to NLU functionality.
The module supports both local model execution and remote API calls.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import logging
import string
from typing import Dict, List, Any, Tuple

from fastapi import FastAPI, Response

from arklex.utils.slot import (
    Verification,
    SlotInputList,
    structured_input_output,
    format_slotfilling_output,
    Slot,
)
from dotenv import load_dotenv

load_dotenv()

from arklex.utils.model_provider_config import PROVIDER_MAP
from langchain_openai import ChatOpenAI
from pydantic_ai import Agent
from pydantic import ValidationError


logger = logging.getLogger(__name__)


class NLUModelAPI:
    def __init__(self) -> None:
        """Initialize the NLU model API.

        This function initializes the NLU model API with default user and assistant prefixes.
        """
        self.user_prefix: str = "user"
        self.assistant_prefix: str = "assistant"

    def get_response(
        self,
        sys_prompt: str,
        model: Dict[str, Any],
        response_format: str = "text",
        note: str = "intent detection",
    ) -> str:
        """Get a response from the language model.

        This function sends a system prompt to the language model and returns its response.
        It supports different response formats and model providers.

        Args:
            sys_prompt (str): The system prompt to send to the model.
            model (Dict[str, Any]): Model configuration including provider and type.
            response_format (str, optional): Format of the response. Defaults to "text".
            note (str, optional): Note for logging purposes. Defaults to "intent detection".

        Returns:
            str: The model's response.
        """
        logger.info(f"Prompt for {note}: {sys_prompt}")
        dialog_history: List[Dict[str, str]] = [
            {"role": "system", "content": sys_prompt}
        ]
        kwargs: Dict[str, Any] = {
            "model": model["model_type_or_path"],
            "temperature": 0.1,
        }

        if model["llm_provider"] != "anthropic":
            kwargs["n"] = 1
        llm: Any = PROVIDER_MAP.get(model["llm_provider"], ChatOpenAI)(**kwargs)

        if model["llm_provider"] == "openai":
            llm = llm.bind(
                response_format={"type": "json_object"}
                if response_format == "json"
                else {"type": "text"}
            )
            res: Any = llm.invoke(dialog_history)
        else:
            messages: List[Tuple[str, str]] = [
                (
                    "user",
                    f"{dialog_history[0]['content']} Only choose the option letter, no explanation.",
                )
            ]
            res: Any = llm.invoke(messages)

        return res.content

    def format_input(
        self, intents: Dict[str, List[Dict[str, Any]]], chat_history_str: str
    ) -> Tuple[str, Dict[str, str]]:
        """Format input text for intent detection.

        This function formats the input text by combining intents, definitions, and chat history
        into a structured prompt for the language model. It creates a multiple-choice format
        for intent selection and maintains a mapping between indices and intent names.

        Args:
            intents (Dict[str, List[Dict[str, Any]]]): Dictionary of intents with their attributes,
                including definitions and sample utterances.
            chat_history_str (str): Formatted chat history string.

        Returns:
            Tuple[str, Dict[str, str]]: A tuple containing the formatted system prompt and
                a mapping from indices to intent names.
        """
        intents_choice: str = ""
        definition_str: str = ""
        exemplars_str: str = ""
        idx2intents_mapping: Dict[str, str] = {}
        multiple_choice_index: Dict[int, str] = dict(enumerate(string.ascii_lowercase))
        count: int = 0
        for intent_k, intent_v in intents.items():
            if len(intent_v) == 1:
                intent_name: str = intent_k
                idx2intents_mapping[multiple_choice_index[count]] = intent_name
                definition: str = intent_v[0].get("attribute", {}).get("definition", "")
                sample_utterances: List[str] = (
                    intent_v[0].get("attribute", {}).get("sample_utterances", [])
                )

                if definition:
                    definition_str += (
                        f"{multiple_choice_index[count]}) {intent_name}: {definition}\n"
                    )
                if sample_utterances:
                    exemplars: str = "\n".join(sample_utterances)
                    exemplars_str += f"{multiple_choice_index[count]}) {intent_name}: \n{exemplars}\n"
                intents_choice += f"{multiple_choice_index[count]}) {intent_name}\n"

                count += 1

            else:
                for idx, intent in enumerate(intent_v):
                    intent_name: str = f"{intent_k}__<{idx}>"
                    idx2intents_mapping[multiple_choice_index[count]] = intent_name
                    definition: str = intent.get("attribute", {}).get("definition", "")
                    sample_utterances: List[str] = intent.get("attribute", {}).get(
                        "sample_utterances", []
                    )

                    if definition:
                        definition_str += f"{multiple_choice_index[count]}) {intent_name}: {definition}\n"
                    if sample_utterances:
                        exemplars: str = "\n".join(sample_utterances)
                        exemplars_str += f"{multiple_choice_index[count]}) {intent_name}: \n{exemplars}\n"
                    intents_choice += f"{multiple_choice_index[count]}) {intent_name}\n"

                    count += 1
        # Base prompt without conditional sections
        system_prompt_nlu: str = """According to the conversation, decide what is the user's intent in the last turn? \n"""

        # Conditionally add definitions if available
        if definition_str.strip():
            system_prompt_nlu += (
                """Here are the definitions for each intent:\n{definition}\n"""
            )

        # Conditionally add exemplars if available
        if exemplars_str.strip():
            system_prompt_nlu += """Here are some sample utterances from user that indicate each intent:\n{exemplars}\n"""

        # Add the rest of the prompt
        system_prompt_nlu += """Conversation:\n{formatted_chat}\n\nOnly choose from the following options.\n{intents_choice}\n\nAnswer:"""

        system_prompt: str = system_prompt_nlu.format(
            definition=definition_str,
            exemplars=exemplars_str,
            intents_choice=intents_choice,
            formatted_chat=chat_history_str,
        )
        return system_prompt, idx2intents_mapping

    def predict(
        self,
        text: str,
        intents: Dict[str, List[Dict[str, Any]]],
        chat_history_str: str,
        model: Dict[str, Any],
    ) -> str:
        """Predict the intent from the input text.

        This function processes the input text and chat history to predict the user's intent
        using the language model. It formats the input, gets a response, and processes the
        result to return the predicted intent.

        Args:
            text (str): The input text to analyze.
            intents (Dict[str, List[Dict[str, Any]]]): Dictionary of possible intents.
            chat_history_str (str): Formatted chat history.
            model (Dict[str, Any]): Model configuration.

        Returns:
            str: The predicted intent.
        """
        system_prompt: str
        idx2intents_mapping: Dict[str, str]
        system_prompt, idx2intents_mapping = self.format_input(
            intents, chat_history_str
        )
        response: str = self.get_response(system_prompt, model, note="intent detection")
        logger.info(f"postprocessed intent response: {response}")
        try:
            pred_intent_idx: str = response.split(")")[0]
            pred_intent: str = idx2intents_mapping[pred_intent_idx]
        except:
            pred_intent: str = response.strip().lower()
        logger.info(f"postprocessed intent response: {pred_intent}")
        return pred_intent


class SlotFillModelAPI:
    def __init__(self) -> None:
        """Initialize the slot filling model API.

        This function initializes the slot filling model API with default user and assistant prefixes.
        """
        self.user_prefix: str = "user"
        self.assistant_prefix: str = "assistant"

    def format_input(self, slots: SlotInputList, input: str, type: str = "chat") -> str:
        """Format input text for slot filling.

        This function formats the input text by combining slot definitions and input text
        into a structured prompt for the language model.

        Args:
            slots (SlotInputList): List of slots to fill.
            input (str): The input text to analyze.
            type (str, optional): Type of slot filling task. Defaults to "chat".

        Returns:
            str: The formatted system prompt.
        """
        if type == "chat":
            system_prompt: str = f"Given the conversation and definition of each dialog state, update the value of following dialogue states.\nConversation:\n{input}\n\nDialogue Statues:\n{slots}\n"
        elif type == "user_simulator":
            system_prompt: str = f"Given a user profile, extract the values for each defined slot type. Only extract values that are explicitly mentioned in the profile. If a value is not found, leave it empty.\n\nSlot definitions:\n{slots}\n\nUser profile:\n{input}\n\nFor each slot:\n1. Look for an exact match in the profile\n2. Only extract values that are clearly stated\n3. Do not make assumptions or infer values\n4. If a slot has enum values, the extracted value must match one of them exactly\n\nExtract the values:\n"
        return system_prompt

    def get_response(
        self,
        sys_prompt: str,
        format: Any,
        model: Dict[str, Any],
        note: str = "slot filling",
    ) -> Any:
        """Get a response from the language model for slot filling.

        This function sends a system prompt to the language model and returns its response
        in the specified format. It supports different model providers and response formats.

        Args:
            sys_prompt (str): The system prompt to send to the model.
            format (Any): The expected response format.
            model (Dict[str, Any]): Model configuration.
            note (str, optional): Note for logging purposes. Defaults to "slot filling".

        Returns:
            Any: The model's response in the specified format.
        """
        logger.info(f"Prompt for {note}: {sys_prompt}")
        dialog_history: List[Dict[str, str]] = [
            {"role": "system", "content": sys_prompt}
        ]
        kwargs: Dict[str, Any] = {
            "model": model["model_type_or_path"],
            "temperature": 0.7,
        }
        # set number of chat completions to generate, isn't supported by Anthropic
        if model["llm_provider"] != "anthropic":
            kwargs["n"] = 1
        llm: Any = PROVIDER_MAP.get(model["llm_provider"], ChatOpenAI)(**kwargs)

        if model["llm_provider"] == "openai":
            llm = llm.with_structured_output(schema=format)
            response: Any = llm.invoke(dialog_history)

        # TODO: fix slotfilling for huggingface
        elif model["llm_provider"] == "huggingface":
            # llm = llm.bind_tools([format])
            # chain =  llm | JsonOutputToolsParser()
            # response = chain.invoke(dialog_history)
            raise NotImplementedError("Slotfilling for Huggingface is not implemented")

        elif model["llm_provider"] == "gemini":
            agent: Agent = Agent(
                f"google-gla:{model['model_type_or_path']}", result_type=format
            )
            result: Any = agent.run_sync(dialog_history[0]["content"])
            response: Any = result.data

        # for claude
        else:
            messages: List[Dict[str, str]] = [
                {"role": "user", "content": dialog_history[0]["content"]}
            ]
            llm = llm.bind_tools([format])
            res: Any = llm.invoke(messages)
            response: Any = format(**res.tool_calls[0]["args"])
        return response

    def predict(
        self, slots: List[Slot], input: str, model: Dict[str, Any], type: str = "chat"
    ) -> List[Slot]:
        """Predict slot values from the input text.

        This function processes the input text to extract values for the specified slots
        using the language model. It handles different types of slot filling tasks and
        formats the output appropriately.

        Args:
            slots (List[Slot]): List of slots to fill.
            input (str): The input text to analyze.
            model (Dict[str, Any]): Model configuration.
            type (str, optional): Type of slot filling task. Defaults to "chat".

        Returns:
            List[Slot]: List of slots with their predicted values.
        """
        try:
            input_slots: SlotInputList
            output_slots: Any
            input_slots, output_slots = structured_input_output(slots)
            system_prompt: str = self.format_input(input_slots, input, type)
            response: Any = self.get_response(
                system_prompt, output_slots, model, note="slot filling"
            )
            filled_slots: List[Slot] = format_slotfilling_output(slots, response)
            logger.info(f"Updated dialogue states: {filled_slots}")
            return filled_slots
        except ValidationError as e:
            logger.warning(
                f"SlotFilling failed. The error is {e}. Returning slots without filling."
            )
            return slots

    def verify(
        self,
        slot: Dict[str, Any],
        chat_history_str: str,
        model: Dict[str, Any],
    ) -> Verification:
        """Verify a slot value against the chat history.

        This function checks if a slot value is valid based on the chat history and slot
        definition. It uses the language model to verify the value and returns a verification
        result.

        Args:
            slot (Dict[str, Any]): The slot to verify.
            chat_history_str (str): Formatted chat history.
            model (Dict[str, Any]): Model configuration.

        Returns:
            Verification: The verification result.
        """
        reformat_slot: Dict[str, Any] = {
            key: value
            for key, value in slot.items()
            if key in ["name", "type", "value", "enum", "description", "required"]
        }
        system_prompt: str = f"Given the conversation, definition and extracted value of each dialog state, decide whether the following dialog states values need further verification from the user. Verification is needed for expressions which may cause confusion. If it is an accurate information extracted, no verification is needed. If there is a list of enum value, which means the value has to be chosen from the enum list. If the user has given the affrimative answer, no need to verify. Only Return boolean value: True or False. \nDialogue Statues:\n{reformat_slot}\nConversation:\n{chat_history_str}\n\n"
        response: Verification = self.get_response(
            system_prompt, format=Verification, model=model, note="slot verification"
        )
        if not response:  # no need to verification, we want to make sure it is really confident that we need to ask the question again
            logger.info("Failed to verify dialogue states")
            return Verification(verification_needed=False, thought="No need to verify")
        logger.info(f"Verified dialogue states: {response}")
        return response


app: FastAPI = FastAPI()
nlu_api: NLUModelAPI = NLUModelAPI()
slotfilling_api: SlotFillModelAPI = SlotFillModelAPI()


@app.post("/nlu/predict")
def predict(data: Dict[str, Any], res: Response) -> Dict[str, str]:
    """Predict intent from the input data.

    This endpoint processes the input data to predict the user's intent using the NLU model.

    Args:
        data (Dict[str, Any]): Input data containing text, intents, and model configuration.
        res (Response): FastAPI response object.

    Returns:
        Dict[str, str]: Dictionary containing the predicted intent.
    """
    logger.info(f"Received data: {data}")
    pred_intent: str = nlu_api.predict(**data)

    logger.info(f"pred_intent: {pred_intent}")
    return {"intent": pred_intent}


@app.post("/slotfill/predict")
def predict(data: Dict[str, Any], res: Response) -> List[Slot]:
    """Predict slot values from the input data.

    This endpoint processes the input data to predict slot values using the slot filling model.

    Args:
        data (Dict[str, Any]): Input data containing slots, text, and model configuration.
        res (Response): FastAPI response object.

    Returns:
        List[Slot]: List of predicted slot values.
    """
    logger.info(f"Received data: {data}")
    results: List[Slot] = slotfilling_api.predict(**data)
    logger.info(f"pred_slots: {results}")
    return results


@app.post("/slotfill/verify")
def verify(data: Dict[str, Any], res: Response) -> Verification:
    """Verify slot values against the input data.

    This endpoint verifies slot values using the slot filling model.

    Args:
        data (Dict[str, Any]): Input data containing slot, chat history, and model configuration.
        res (Response): FastAPI response object.

    Returns:
        Verification: The verification result.
    """
    logger.info(f"Received data: {data}")
    verify_needed: Verification = slotfilling_api.verify(**data)

    logger.info(f"verify_needed: {verify_needed}")
    return verify_needed
