import logging
import json
import random
import string
from typing import Dict, Any, Optional
from datetime import datetime

from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.utils.slot import Slot


logger = logging.getLogger(__name__)


@register_worker
class NegotiationSingleIssueWorker(BaseWorker):
    """This must run first. This worker helps process the user's message and generate a response that moves the negotiation forward."""
    
    description = "This must run first. This worker helps process the user's message and generate a response that moves the negotiation forward."
    
    def __init__(self):
        super().__init__()
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000, 
            temperature = 0.0
        )
        self.action_graph = self._create_action_graph()
        self.unit_index = "unit1"
        self.all_slots_present = False
        self.static_prompt = ""
        self.dynamic_prompt = ""
        logger.info("PersuasionWorker initialized successfully")

    def read_json(file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    #Used to determine initial target price 
    def random_in_last_third(max_percieved_marketPrice, max_market_price, reservationPrice):

        if(max_market_price == reservationPrice and max_percieved_marketPrice > reservationPrice):
            max_market_price = max_percieved_marketPrice
        # Calculate the range within the first third
        first_third_range = (max_market_price - reservationPrice) // 3

        # Generate a random integer within that range
        random_num = random.randint(reservationPrice + 2 * first_third_range, 
                                    max_market_price)

        return random_num
    
    round_num = lambda num, base: base * round(num/base)

    def load_prompt(self,current_target, path):
            with open(path) as f:
                instructions = f.read().format(current_target, params["current_target"])
            return instructions
        
    def get_prompts(self, unit_index, current_target):
        if unit_index == 'unit1':
            static_path = "./prompts/seller_system_prompt_static.txt"
            dynamic_path = "./prompts/seller_system_prompt_dynamic_floor.txt"
        elif unit_index == 'unit2':
            static_path = "./prompts/apt_seller_system_prompt_static.txt"
            dynamic_path = "./prompts/apt_seller_system_prompt_dynamic_floor.txt"
        elif unit_index == 'unit3': 
            static_path = "./prompts/jeep_seller_system_prompt_static.txt"
            dynamic_path = "./prompts/jeep_seller_system_prompt_dynamic_floor.txt"
        elif unit_index == 'unit4': 
            static_path = "./prompts/ford_seller_system_prompt_static.txt"
            dynamic_path = "./prompts/ford_seller_system_prompt_dynamic_floor.txt"
        static_prompt = self.load_prompt(current_target=current_target, path=static_path)
        dynamic_prompt = self.load_prompt( current_target=current_target, path=dynamic_path)
        return static_prompt, dynamic_prompt
    
    def check_and_initialize_slots(self, state: MessageState):
        configs = self.read_json("./env/workers/negotiation_config/seller_config.json")
        required_slots = ["turn", "episode_done", "max_percieved_marketPrice", 
                         "reservation_price", "max_market_price", "current_target"]
        # Check if any required slots are missing
        if "slots" not in state:
            state["slots"] = {}
            
        for slot_name in required_slots:
            if slot_name not in state["slots"]:
                if slot_name == "turn":
                    state["slots"]["turn"] = [Slot(
                        name="turn",
                        type="string",
                        value=0,
                        enum=[],
                        description="This tracks the current turn number in the negotiation.",
                        prompt="",
                        required=False,
                        verified=True)]
                
                elif slot_name == "episode_done":
                    state["slots"]["episode_done"] = [Slot(
                        name="episode_done",
                        type="string",
                        value=False,
                        enum=[],
                        description="This indicates whether the negotiation episode is complete.",
                        prompt="",
                        required=False,
                        verified=True)]
                
                elif slot_name == "max_percieved_marketPrice":
                    max_percieved_marketPrice = 0
                    if "max_percieved_marketPrice" in configs['units'][self.unit_index]['parameters']:
                        max_percieved_marketPrice = configs['units'][self.unit_index]['parameters']['max_percieved_marketPrice'][0]
                    
                    state["slots"]["max_percieved_marketPrice"] = [Slot(
                        name="max_percieved_marketPrice",
                        type="string",
                        value=max_percieved_marketPrice,
                        enum=[],
                        description="This is the maximum perceived market price.",
                        prompt="",
                        required=False,
                        verified=True)]
                
                elif slot_name == "reservation_price":
                    state["slots"]["reservation_price"] = [Slot(
                        name="reservation_price",
                        type="string",
                        value=configs['units'][self.unit_index]['parameters']['reservationPrice'][0],
                        enum=[],
                        description="This is the reservation price for the negotiation.",
                        prompt="",
                        required=False,
                        verified=True)]
                
                elif slot_name == "max_market_price":
                    state["slots"]["max_market_price"] = [Slot(
                        name="max_market_price",
                        type="string",
                        value=configs['units'][self.unit_index]['parameters']['max_marketPrice'][0],
                        enum=[],
                        description="This is the maximum market price.",
                        prompt="",
                        required=False,
                        verified=True)]
                    
        self.get_current_target(state)
        
        self.all_slots_present = True
    
    def get_current_target(self, state: MessageState):
        targ = self.round_num(self.random_in_last_third(
                state["slots"]["max_percieved_marketPrice"], state["slots"]["max_marketPrice"], state["slots"]["reservationPrice"]
            ))
        state["slots"]["current_target"] = [Slot(
                name = "current_target", 
                type = "string", 
                value = targ, 
                enum = [],
                description = "This is the value that holds the classification of the user's argument.", 
                prompt = "", 
                required = False, 
                verified = True)] 
        
    def get_response(self, state: MessageState):
        if not self.all_slots_present:
            self.check_and_initialize_slots(state)

        if(state["slots"]["turn"] == 0): 
           print(f"Initial target for seller: {state['slots']['current_target']}") 
           static_prompt, dynamic_prompt = self.get_prompts(
                        state["slots"]["current_target"],
                        self.unit_index
                    )
           state["response"] = self.llm.invoke(dynamic_prompt).content.strip()

        elif(state["slots"]["turn"] == 1):
            state["slots"]["turn"] += 1
            generic_prompt = '\nRespond to the user as the seller. Do not give a counteroffer or mention a price in your response.'
            state["response"] = self.llm.invoke(generic_prompt).content.strip()
        elif(state["slots"]["turn"] > 4): 
            state["slots"]["turn"] += 1
            static_prompt, dynamic_prompt = self.get_prompts(
                        state["slots"]["current_target"],
                        self.unit_index
                    )
            state["response"] = self.llm.invoke(static_prompt).content.strip()

        else: 
            state["slots"]["turn"] += 1
            state["current_target"] = (
                state["slots"]["current_target"] + state["slots"]["reservation_price"]) / 2
            state["slots"]["current_target"] = self.round_num(state["current_target"], int(1*10**2))
            static_prompt, dynamic_prompt = self.get_prompts(
                        state["slots"]["current_target"],
                        self.unit_index
                    )
            state["response"] = self.llm.invoke(dynamic_prompt).content.strip()

        if(state["slots"]["turn"] >= 7):
            state["slots"]["episode_done"] = True
        else:
            state["slots"]["episode_done"] = False
    
    def _create_action_graph(self):
        """Create a processing flow for persuasion strategy."""
        workflow = StateGraph(MessageState)
        
        # Add persuasion strategy node
        workflow.add_node("negotiation_resposne", self.get_response)
        
        # Add edges
        workflow.add_edge(START, "negotiation_resposne")
        
        return workflow
        
    def execute(self, msg_state: MessageState):
        
        graph = self._create_action_graph().compile()
        result = graph.invoke(msg_state)
        return result 
    