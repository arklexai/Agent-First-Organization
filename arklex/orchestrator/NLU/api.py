import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import logging
import string

from fastapi import FastAPI, Response

from arklex.utils.slot import Verification, SlotInputList, structured_input_output, format_slotfilling_output, Slot
from dotenv import load_dotenv
load_dotenv()

from arklex.utils.model_provider_config import PROVIDER_MAP
from langchain_openai import ChatOpenAI
from pydantic_ai import Agent


logger = logging.getLogger(__name__)


class NLUModelAPI ():
    def __init__(self):
        self.user_prefix = "user"
        self.assistant_prefix = "assistant"

    def get_response(self, sys_prompt, model, response_format="text", note="intent detection"):
        logger.info(f"Prompt for {note}: {sys_prompt}")
        dialog_history = [{"role": "system", "content": sys_prompt}]
        kwargs = {'model': model["model_type_or_path"], 'temperature': 0.1}
        
        if model['llm_provider'] != 'anthropic': kwargs['n'] = 1
        llm = PROVIDER_MAP.get(model['llm_provider'], ChatOpenAI)(**kwargs)

        if model['llm_provider'] == 'openai':
            llm = llm.bind(response_format={"type": "json_object"} if response_format == "json" else {"type": "text"})
            res = llm.invoke(dialog_history)
        else:
            messages = [("user", f"{dialog_history[0]['content']} Only choose the option letter, no explanation.")]
            res = llm.invoke(messages)
        
        return res.content

    def format_input(self, intents, chat_history_str) -> str:
        """Format input text before feeding it to the model."""
        intents_choice, definition_str, exemplars_str = "", "", ""
        idx2intents_mapping = {}
        multiple_choice_index = dict(enumerate(string.ascii_lowercase))
        count = 0
        for intent_k, intent_v in intents.items():
            if len(intent_v) == 1:
                intent_name = intent_k
                idx2intents_mapping[multiple_choice_index[count]] = intent_name
                definition = intent_v[0].get("attribute", {}).get("definition", "")
                sample_utterances = intent_v[0].get("attribute", {}).get("sample_utterances", [])

                if definition:
                    definition_str += (
                        f"{multiple_choice_index[count]}) {intent_name}: {definition}\n"
                    )
                if sample_utterances:
                    exemplars = "\n".join(sample_utterances)
                    exemplars_str += (
                        f"{multiple_choice_index[count]}) {intent_name}: \n{exemplars}\n"
                    )
                intents_choice += f"{multiple_choice_index[count]}) {intent_name}\n"

                count += 1

            else:
                for idx, intent in enumerate(intent_v):
                    intent_name = f'{intent_k}__<{idx}>'
                    idx2intents_mapping[multiple_choice_index[count]] = intent_name
                    definition = intent.get("attribute", {}).get("definition", "")
                    sample_utterances = intent.get("attribute", {}).get("sample_utterances", [])

                    if definition:
                        definition_str += (
                            f"{multiple_choice_index[count]}) {intent_name}: {definition}\n"
                        )
                    if sample_utterances:
                        exemplars = "\n".join(sample_utterances)
                        exemplars_str += (
                            f"{multiple_choice_index[count]}) {intent_name}: \n{exemplars}\n"
                        )
                    intents_choice += f"{multiple_choice_index[count]}) {intent_name}\n"

                    count += 1
        # Base prompt without conditional sections
        system_prompt_nlu = """According to the conversation, decide what is the user's intent in the last turn? \n"""
        
        # Conditionally add definitions if available
        if definition_str.strip():
            system_prompt_nlu += """Here are the definitions for each intent:\n{definition}\n"""
            
        # Conditionally add exemplars if available
        if exemplars_str.strip():
            system_prompt_nlu += """Here are some sample utterances from user that indicate each intent:\n{exemplars}\n"""
            
        # Add the rest of the prompt
        system_prompt_nlu += """Conversation:\n{formatted_chat}\n\nOnly choose from the following options.\n{intents_choice}\n\nAnswer:"""
        
        system_prompt = system_prompt_nlu.format(
            definition=definition_str,
            exemplars=exemplars_str,
            intents_choice=intents_choice,
            formatted_chat=chat_history_str,
        )
        return system_prompt, idx2intents_mapping

    def predict(
        self,
        text,
        intents,
        chat_history_str,
        model
    ) -> str:

        system_prompt, idx2intents_mapping = self.format_input(
            intents, chat_history_str
        )
        response = self.get_response(
            system_prompt, model, note="intent detection"
        )
        logger.info(f"postprocessed intent response: {response}")
        try:
            pred_intent_idx = response.split(")")[0]
            pred_intent = idx2intents_mapping[pred_intent_idx]
        except:
            pred_intent = response.strip().lower()
        logger.info(f"postprocessed intent response: {pred_intent}")
        return pred_intent


class SlotFillModelAPI():
    def __init__(self):
        self.user_prefix = "user"
        self.assistant_prefix = "assistant"
    
    # System prompt for slot filling
    def format_input(self, slots: SlotInputList, input, type: str = "chat") -> str:
        """Format input text before feeding it to the model."""
        if type == "chat":
            system_prompt = f"Given the conversation and definition of each dialog state, update the value of following dialogue states.\nConversation:\n{input}\n\nDialogue Statues:\n{slots}\n"
        elif type == "user_simulator":
            system_prompt = f"Given a user profile, extract the values for each defined slot type. Only extract values that are explicitly mentioned in the profile. If a value is not found, leave it empty.\n\nSlot definitions:\n{slots}\n\nUser profile:\n{input}\n\nFor each slot:\n1. Look for an exact match in the profile\n2. Only extract values that are clearly stated\n3. Do not make assumptions or infer values\n4. If a slot has enum values, the extracted value must match one of them exactly\n\nExtract the values:\n"
        return system_prompt

    # get response from model
    def get_response(self, sys_prompt, format, model, note="slot filling"):
        logger.info(f"Prompt for {note}: {sys_prompt}")
        dialog_history = [{"role": "system", "content": sys_prompt}]
        kwargs = {'model': model["model_type_or_path"], 'temperature': 0.7}
        # set number of chat completions to generate, isn't supported by Anthropic
        if model['llm_provider'] != 'anthropic': kwargs['n'] = 1
        llm = PROVIDER_MAP.get(model['llm_provider'], ChatOpenAI)(**kwargs)
        
        if model['llm_provider'] == 'openai':
            llm = llm.with_structured_output(schema=format)
            response = llm.invoke(dialog_history)
    
        # TODO: fix slotfilling for huggingface
        elif model['llm_provider']=='huggingface':
            # llm = llm.bind_tools([format])
            # chain =  llm | JsonOutputToolsParser()
            # response = chain.invoke(dialog_history)
            raise NotImplementedError("Slotfilling for Huggingface is not implemented")

        elif model['llm_provider'] == 'gemini':
            agent = Agent(f"google-gla:{model['model_type_or_path']}", result_type=format)
            result = agent.run_sync(dialog_history[0]['content'])
            response = result.data

        #for claude 
        else:
            messages = [{"role": "user", "content": dialog_history[0]['content']}]
            llm = llm.bind_tools([format])
            res= llm.invoke(messages)
            response = format(**res.tool_calls[0]['args'])
        return response

    # endpoint for slot filling
    def predict(
        self,
        slots: list[Slot],
        input: str,
        model: dict,
        type: str = "chat"
    ):
        input_slots, output_slots = structured_input_output(slots)
        system_prompt = self.format_input(input_slots, input, type)
        response = self.get_response(system_prompt, output_slots, model, note="slot filling")
        filled_slots = format_slotfilling_output(slots, response)
        logger.info(f"Updated dialogue states: {filled_slots}")
        return filled_slots
    
    # endpoint for slot verification
    def verify(
        self,
        slot: dict,
        chat_history_str,
        model: dict,
    ) -> Verification:
        reformat_slot = {key: value for key, value in slot.items() if key in ["name", "type", "value", "enum", "description", "required"]}
        system_prompt = f"Given the conversation, definition and extracted value of each dialog state, decide whether the following dialog states values need further verification from the user. Verification is needed for expressions which may cause confusion. If it is an accurate information extracted, no verification is needed. If there is a list of enum value, which means the value has to be chosen from the enum list. Only Return boolean value: True or False. \nDialogue Statues:\n{reformat_slot}\nConversation:\n{chat_history_str}\n\n"
        response = self.get_response(
            system_prompt, format=Verification, model=model, note="slot verification"
        )
        if not response: # no need to verification, we want to make sure it is really confident that we need to ask the question again
            logger.info(f"Failed to verify dialogue states")
            return Verification(verification_needed=False, thought="No need to verify")
        logger.info(f"Verified dialogue states: {response}")
        return response


app = FastAPI()
nlu_api = NLUModelAPI()
slotfilling_api = SlotFillModelAPI()


@app.post("/nlu/predict")
def predict(data: dict, res: Response):
    logger.info(f"Received data: {data}")
    pred_intent = nlu_api.predict(**data)

    logger.info(f"pred_intent: {pred_intent}")
    return {"intent": pred_intent}

@app.post("/slotfill/predict")
def predict(data: dict, res: Response):
    logger.info(f"Received data: {data}")
    results = slotfilling_api.predict(**data)
    logger.info(f"pred_slots: {results}")
    return results

@app.post("/slotfill/verify")
def verify(data: dict, res: Response):
    logger.info(f"Received data: {data}")
    verify_needed = slotfilling_api.verify(**data)

    logger.info(f"verify_needed: {verify_needed}")
    return verify_needed