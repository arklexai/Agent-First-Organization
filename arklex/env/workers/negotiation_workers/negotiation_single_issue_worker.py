import logging
import json
import random
import string
import os
from tkinter import END
from typing import Dict, Any, Optional
from datetime import datetime
from functools import partial
from langchain_core.language_models import BaseChatModel

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
    """This worker handles single-issue negotiations after the initial ice breaker.
    
    This worker processes user messages and generates responses that move the negotiation forward.
    It maintains the negotiation state, tracks turns, and manages price targets throughout the conversation.
    """
    
    description = "This worker should then be the only worker running for the rest of the conversation. This worker helps process the user's message and generate a response that moves the negotiation forward."
    
    def __init__(self):
        """Initialize the NegotiationSingleIssueWorker with necessary attributes."""
        super().__init__()
        self.llm: Optional[BaseChatModel] = None
        self.action_graph = None  # Will be created in _execute with tags
        self.unit_index = 0
        self.tags = {}
        self.static_prompt = ""
        self.dynamic_prompt = ""
        # Get absolute path to the directory containing this file
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Current directory: {self.current_dir}")
        logger.info("NegotiationSingleIssueWorkerSeller initialized successfully")

    def read_json(self, file_path: str) -> Dict:
        """Read and parse a JSON file.
        
        Args:
            file_path: Path to the JSON file to read
            
        Returns:
            Dict: Parsed JSON content
        """
        with open(file_path, "r") as f:
            return json.load(f)

    def random_in_last_third(self, max_percieved_marketPrice: float, max_market_price: float, reservationPrice: float) -> float:
        """Generate a random price in the last third of the price range.
        
        Args:
            max_percieved_marketPrice: Maximum perceived market price
            max_market_price: Maximum market price
            reservationPrice: Minimum acceptable price
            
        Returns:
            float: Random price within the last third of the range
        """
        if(max_market_price == reservationPrice and max_percieved_marketPrice > reservationPrice):
            max_market_price = max_percieved_marketPrice
        # Calculate the range within the first third
        first_third_range = (max_market_price - reservationPrice) // 3

        # Generate a random integer within that range
        random_num = random.randint(reservationPrice + 2 * first_third_range, 
                                    max_market_price)

        return random_num
    
    def round_num(self, num: float, base: int = 100) -> int:
        """Round a number to the nearest multiple of base.
        
        Args:
            num: Number to round
            base: Base to round to (default: 100)
            
        Returns:
            int: Rounded number
        """
        return base * round(num/base)

    def load_prompt(self, current_target: float, message: str, turn: int, path: str, is_dynamic: bool = False, history: str = "") -> str:
        """Load and format a prompt template from a file.
        
        Args:
            current_target: Current target price
            message: User's message
            turn: Current turn number
            path: Path to prompt template file
            is_dynamic: Whether the prompt is dynamic (default: False)
            history: Conversation history (default: "")
            
        Returns:
            str: Formatted prompt
        """
        with open(path) as f:
            if is_dynamic:
                instructions = f.read().format(current_target, current_target, history, turn)
            else:
                instructions = f.read().format(history, turn)
        return instructions

    def get_prompts(self, unit_index: int, current_target: float, message: str, turn: int, history: str = "") -> tuple[str, str]:
        """Get appropriate static and dynamic prompts based on the unit index.
        
        Args:
            unit_index: Index of the current negotiation unit
            current_target: Current target price
            message: User's message
            turn: Current turn number
            history: Conversation history (default: "")
            
        Returns:
            tuple[str, str]: Static and dynamic prompts
        """
        # Base directory for prompts
        prompts_base_dir = os.path.join(self.current_dir, "negotiation_prompts")
        
        if unit_index == 0:
            logger.info("Getting prompts for unit1")
            static_path = os.path.join(prompts_base_dir, "seller_system_prompt_static.txt")
            dynamic_path = os.path.join(prompts_base_dir, "seller_system_prompt_dynamic_floor.txt")
        elif unit_index == 1:
            static_path = os.path.join(prompts_base_dir, "apt_seller_system_prompt_static.txt")
            dynamic_path = os.path.join(prompts_base_dir, "apt_seller_system_prompt_dynamic_floor.txt")
        elif unit_index == 2: 
            static_path = os.path.join(prompts_base_dir, "jeep_seller_system_prompt_static.txt")
            dynamic_path = os.path.join(prompts_base_dir, "jeep_seller_system_prompt_dynamic_floor.txt")
        elif unit_index == 3: 
            static_path = os.path.join(prompts_base_dir, "ford_seller_system_prompt_static.txt")
            dynamic_path = os.path.join(prompts_base_dir, "ford_seller_system_prompt_dynamic_floor.txt")
        
        logger.info(f"Looking for prompts at: {static_path} and {dynamic_path}")
    
        static_prompt = self.load_prompt(current_target=current_target, message=message, turn=turn, path=static_path, is_dynamic=False, history=history)
        dynamic_prompt = self.load_prompt(current_target=current_target, message=message, turn=turn, path=dynamic_path, is_dynamic=True, history=history)
        logger.info(f"Static prompt: {static_prompt}")
        return static_prompt, dynamic_prompt
    
    def check_and_initialize_slots(self, state: MessageState) -> None:
        """Initialize required negotiation slots if they don't exist.
        
        Args:
            state: Current message state
        """
        logger.info("checking and initializing slots")
        config_path = os.path.join(self.current_dir, "negotiation_config", "seller_config.json")
        configs = self.read_json(config_path)
        required_slots = ["turn", "episode_done", "max_percieved_marketPrice", 
                         "reservation_price", "max_market_price", "current_target"]
        # Check if any required slots are missing
        if not hasattr(state, 'slots'):
            state.slots = {}
            
        for slot_name in required_slots:
            if slot_name not in state.slots:
                if slot_name == "turn":
                    state.slots["turn"] = [Slot(
                        name="turn",
                        type="string",
                        value=0,
                        enum=[],
                        description="This tracks the current turn number in the negotiation.",
                        prompt="",
                        required=False,
                        verified=True)]
                
                elif slot_name == "episode_done":
                    state.slots["episode_done"] = [Slot(
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
                    
                    state.slots["max_percieved_marketPrice"] = [Slot(
                        name="max_percieved_marketPrice",
                        type="string",
                        value=max_percieved_marketPrice,
                        enum=[],
                        description="This is the maximum perceived market price.",
                        prompt="",
                        required=False,
                        verified=True)]
                
                elif slot_name == "reservation_price":
                    state.slots["reservation_price"] = [Slot(
                        name="reservation_price",
                        type="string",
                        value=configs['units'][self.unit_index]['parameters']['reservationPrice'][0],
                        enum=[],
                        description="This is the reservation price for the negotiation.",
                        prompt="",
                        required=False,
                        verified=True)]
                
                elif slot_name == "max_market_price":
                    state.slots["max_market_price"] = [Slot(
                        name="max_market_price",
                        type="string",
                        value=configs['units'][self.unit_index]['parameters']['max_marketPrice'][0],
                        enum=[],
                        description="This is the maximum market price.",
                        prompt="",
                        required=False,
                        verified=True)]
                    
        self.get_current_target(state)
        
    def get_current_target(self, state: MessageState) -> None:
        """Calculate and set the current target price.
        
        Args:
            state: Current message state
        """
        targ = self.round_num(self.random_in_last_third(
                state.slots["max_percieved_marketPrice"][0].value, 
                state.slots["max_market_price"][0].value, 
                state.slots["reservation_price"][0].value
            ))
        state.slots["current_target"] = [Slot(
                name = "current_target", 
                type = "string", 
                value = targ, 
                enum = [],
                description = "This is the value that holds the classification of the user's argument.", 
                prompt = "", 
                required = False, 
                verified = True)] 
        
    def get_response(self, state: MessageState, tags: Dict[str, Any] = {}) -> MessageState:
        """Generate a response based on the current negotiation state.
        
        Args:
            state: Current message state
            
        Returns:
            MessageState: Updated message state with response
        """
        current_turn = state.slots["turn"][0].value
        logger.info(f"Current turn: {current_turn}")
        
        if current_turn == 0: 
           logger.info(f"FIRST TURN FOR SELLER {current_turn}") 
           state.slots["turn"][0].value += 1
           logger.info(f"Initial target for seller: {state.slots['current_target'][0].value}") 
           logger.info(f"Initial target for seller: {state.slots['current_target'][0].value}") 
           static_prompt, dynamic_prompt = self.get_prompts(
                        self.unit_index,
                        state.slots["current_target"][0].value,
                        state.user_message.message, 
                        state.slots["turn"][0].value,
                        history=state.user_message.history
                    )
           logger.info(f"Dynamic prompt: {dynamic_prompt}")
           state.response = self.llm.invoke(dynamic_prompt).content.strip()
           logger.info(f"Response: {state.response}")

        elif current_turn == 1:
            state.slots["turn"][0].value += 1
            logger.info("Responding to user as the seller")
            generic_prompt = '\nRespond to the user as the seller. Do not give a counteroffer or mention a price in your response.'
            state.response = self.llm.invoke(generic_prompt).content.strip()
        elif current_turn > 4: 
            state.slots["turn"][0].value += 1
            static_prompt, dynamic_prompt = self.get_prompts(
                        self.unit_index,
                        state.slots["current_target"][0].value,
                        state.user_message.message,
                        state.slots["turn"][0].value,
                        history=state.user_message.history
                    )
            state.response = self.llm.invoke(static_prompt).content.strip()

        else: 
            state.slots["turn"][0].value += 1
            logger.info("Generating response based on current target")
            new_target = (
                state.slots["current_target"][0].value + state.slots["reservation_price"][0].value
            ) / 2
            state.slots["current_target"][0].value = self.round_num(new_target, int(1*10**2))
            static_prompt, dynamic_prompt = self.get_prompts(
                        self.unit_index,
                        state.slots["current_target"][0].value,
                        state.user_message.message,
                        state.slots["turn"][0].value,
                        history=state.user_message.history
                    )
            state.response = self.llm.invoke(dynamic_prompt).content.strip()

        if(state.slots["turn"][0].value >= 7):
            logger.info("Episode done")
            state.slots["episode_done"][0].value = True
        else:
            state.slots["episode_done"][0].value = False
            
        return state
    
    def _create_action_graph(self,tags: Dict[str, Any]) -> StateGraph:
        """Create a processing flow for the negotiation strategy.
        
        Returns:
            StateGraph: Graph defining the negotiation workflow
        """
        workflow = StateGraph(MessageState)
        # Create a partial function with the tags bound
        negotiation_response_with_tags = partial(self.get_response, tags=tags)
        # Add persuasion strategy node
        workflow.add_node("negotiation_response", negotiation_response_with_tags)
        
        # Add edges
        workflow.add_edge(START, "negotiation_response")
        return workflow
    
    def _execute(self, msg_state: MessageState, **kwargs: Any) -> Dict[str, Any]:
        """Execute the negotiation worker.
        
        Args:
            msg_state: Current message state
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: Updated message state as dictionary
        """
        logger.info("Executing negotiation response worker")
        
        self.llm = PROVIDER_MAP.get(msg_state.bot_config.llm_config.llm_provider, ChatOpenAI )(model=msg_state.bot_config.llm_config.model_type_or_path)
        # Debug the incoming state
        self.check_and_initialize_slots(msg_state)
        self.tags = kwargs.get("tags", {})
        self.action_graph = self._create_action_graph(self.tags)
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        
        # Convert the result back to a MessageState and preserve the response
        response_state = MessageState.model_validate(result)
        logger.info(f"State after graph execution - slots: {response_state.slots}")
        logger.info(f"State after graph execution - slots: {self.tags.items()}")
            
        return response_state.model_dump()
    