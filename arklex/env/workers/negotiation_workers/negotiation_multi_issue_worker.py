import json
import logging
import os
from os.path import dirname, abspath
import time
import re
from itertools import product
from typing import Any, Dict, List, Optional, Tuple
from langgraph.graph import StateGraph, START

from langchain_openai import ChatOpenAI

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState, Slot
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP

import numpy as np
from sklearn.neighbors import KernelDensity

logger = logging.getLogger(__name__)

@register_worker
class NegotiationMultiIssueWorker(BaseWorker):
    """This must run after the first ice breaker from the MessageWorker. This worker should then be the only worker running for the rest of the conversation. This worker helps process the user's message and generate a response that moves the negotiation forward."""
    
    description = "This worker should then be the only worker running for the rest of the conversation. This worker helps process the user's message and generate a response that moves the negotiation forward."
    def __init__(self):
        super().__init__()
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], 
            timeout=30000, 
            temperature=0.0
        )
        self.action_graph = self._create_action_graph()
        
        logger.info("NegotiationMultiIssueWorkerSeller initialized successfully")
        logger.info("NegotiationMultiIssueWorkerSeller initialized successfully")

        self.MONITOR_PROMPT = """
        You are a negotiation monitor overseeing a multi-issue negotiation.

        Based on the negotiation history, please assess how many issues from {issues} have been resolved. 
        Give the answer in **exactly** this format at the end:
        NUMBER OF RESOLVED ISSUES=X where X is the number of resolved issues. 
        When X=4 and all the issues have been agreed upon, logger.info the final outcomes for ALL 4 ISSUES 
        (FINANCING, TAX, COLOR, PRICE, STEREO, DELIVERY DATE, NUMBER OF EXTRAS, WARRANTY) - **exactly** in this format:
        ISSUE NAME IN CAPITAL=Agreement alternative.
        """

        # Utility dictionaries for buyer and seller
        self.buyer_utilities = {
            "FINANCING": {"10%": 0, "8%": 400, "6%": 800, "4%": 1200, "2%": 1600},
            "WARRANTY": {"6 months": 0, "12 months": 1000, "18 months": 2000, "24 months": 3000, "30 months": 4000},
            "PRICE": {"$24000": -6000, "$23000": -4500, "$22000": -3000, "$21000": -1500, "$20000": 0},
            "COLOR": {"Black": 0, "Red": 300, "Blue": 600, "Green": 900, "Yellow": 1200}
        }

        self.seller_utilities = {
            "FINANCING": {"10%": 4000, "8%": 3000, "6%": 2000, "4%": 1000, "2%": 0},
            "WARRANTY": {"6 months": 1600, "12 months": 1200, "18 months": 800, "24 months": 400, "30 months": 0},
            "PRICE": {"$24000": 0, "$23000": -1500, "$22000": -3000, "$21000": -4500, "$20000": -6000},
            "COLOR": {"Yellow": 1200, "Green": 900, "Blue": 600, "Red": 300, "Black": 0}
        }

        
        self.kde_models = {
            'FINANCING': KernelDensity(kernel='gaussian', bandwidth=0.5),
            'WARRANTY': KernelDensity(kernel='gaussian', bandwidth=0.5),
            'PRICE': KernelDensity(kernel='gaussian', bandwidth=0.5),
        }

        # Data structure to hold observations for each issue
        self.observations = {
            'FINANCING': [],
            'WARRANTY': [],
            'PRICE': []
        }


    def read_json(self,file_path):
        with open(file_path, "r") as f:
            return json.load(f)


    def delay_response(self, response: str, time_elapsed: float = 0) -> None:
        """Add a delay based on response length to simulate natural typing.
        
        Args:
            response: The response text
            time_elapsed: Time already elapsed in processing
        """
        word_count = len(response.split())
        if word_count < 30:
            sleep_time = 2 * word_count
        elif word_count < 50:
            sleep_time = 1.5 * word_count
        else:
            sleep_time = word_count
        
        # Uncomment to enable delay
        # time.sleep(max(0, sleep_time - time_elapsed))

    def common_sense_importance(self):
        utilities = {
            "FINANCING": {10: 4000, 8: 3000, 6: 2000, 4: 1000, 2: 0},
            "WARRANTY": {6: 1600, 12: 1200, 18: 800, 24: 400, 30: 0},
            "PRICE": {24000: 0, 23000: -1500, 22000: -3000, 21000: -4500, 20000: -6000},
            "COLOR": {"Yellow": 1200, "Green": 900, "Blue": 600, "Red": 300, "Black": 0}
        }

        prompt = f"""
        Based on the following utilities for each issue, approximate the order of importance of these issues for the buyer. 
        Use common sense and the seller's perspective.

        Utilities:
        {utilities}

        Provide a list ranking these issues from most to least important for the buyer based on common sense assumptions and give justification.
        """

        return self.llm.invoke(prompt).content.strip()

    # Importance Estimation Using KDE
    def importance_estimation_kde(self, conversation_history):
        prompt = f"""
        As the negotiation progresses, you will use Kernel Density Estimation (KDE) to estimate how much the buyer cares about each issue based on their offer changes. Pay attention to the size of their concessions:
        - If they make only small changes early in the negotiation, that issue may be very important to them.
        - Conversely, if they make a significant concession later, it could indicate a final push to save the deal on that issue.

        Given the conversation history: {conversation_history}, please analyze each issue and provide an estimated importance score between 1 and 10 (10 being highly important and 1 being least important) and justify it. 
        """

        return self.llm.invoke(prompt).content.strip()

    # Determining Buyer Personality (Cooperative/Aggressive)
    def determine_user_personality(self, conversation_history):
        prompt = f"""
        Judge if the buyer is being aggressive or cooperative regarding the current issue in negotiation based on these indicators:
        - COOPERATIVE: returning concessions, openly sharing information, actively seeking information about the opponent's preferences.
        - AGGRESSIVE: withholding or misleading information, rigidly committing to their position, issuing ultimatums (e.g., 'take it or leave it').

        Given the conversation history: {conversation_history}, determine if the buyer's behavior on the current issue is cooperative or aggressive.
        Please return either 'COOPERATIVE' or 'AGGRESSIVE', along with justification.
        """

        return self.llm.invoke(prompt).content.strip()

    def extract_issue_and_offers(self, conversation_history):
        """
        Extract the current issue and the buyer's most recent two offers from the conversation history.

        Parameters:
            conversation_history (str): The negotiation history as a string.

        Returns:
            str: The extracted issue, current buyer offer, and previous buyer offer in a numerical format.
        """
        prompt = f"""
        From the following negotiation conversation history, identify the current issue being discussed 
        and extract the buyer's two most recent offers for each issue where applicable. 
        Note that if the buyer is sticking to their previous offer, their current and previous offer will be the SAME VALUE.

        Return the response **STRICTLY**in this format only for the current issue(DON'T DISPLAY PREVIOUS ISSUES):
        - ISSUE: [CURRENT Issue Name]
        CURRENT OFFER: [Most recent buyer offer as numerical value]
        PREVIOUS OFFER: [Second most recent buyer offer as numerical value]

        For issues that involve numerical values, return the value itself:
        - FINANCING: {{"10%", "8%", "6%", "4%", "2%"}} → return 10, 8, 6, 4, 2
        - WARRANTY: {{"6 months", "12 months", "18 months", "24 months", "30 months"}} → return 6, 12, 18, 24, 30
        - PRICE: {{"$20000", "$21000", "$22000", "$23000", "$24000"}} → return 20000, 21000, 22000, 23000, 24000
        - COLOR: {{"Yellow", "Green", "Blue", "Red", "Black"}} → return Yellow, Green, Blue, Red, Black

        If a particular issue does not have at least two buyer offers in the history, return "NO OFFERS YET".
        Conversation history:
        {conversation_history}

        Your output should only include issues with at least two buyer offers, and should follow the required numerical format.
        """

        return self.llm.invoke(prompt).content.strip()

    payoff_schedule = {
            "FINANCING": {10: 4000, 8: 3000, 6: 2000, 4: 1000, 2: 0},
            "WARRANTY": {6: 1600, 12: 1200, 18: 800, 24: 400, 30: 0},
            "PRICE": {24000: 0, 23000: -1500, 22000: -3000, 21000: -4500, 20000: -6000},
            "COLOR": {"Yellow": 1200, "Green": 900, "Blue": 600, "Red": 300, "Black": 0},
        }
    walk_away_point = 2400

    def generate_combos(self):
        """
        Generate all combinations of options that give the seller a total payoff
        of at least the walk-away point.
        
        Args:
            walk_away_point (int): The minimum acceptable payoff for the seller.
            payoff_schedule (dict): A dictionary where keys are issue names, and values
                                    are dictionaries mapping options to payoffs.
            
        Returns:
            str: A string containing all valid combinations, one per line.
        """
        # Generate all combinations of options
        options = list(self.payoff_schedule.values())
        combinations = list(product(*[list(opt.items()) for opt in options]))

        # Filter combinations that meet the walk-away point
        valid_combinations = [
            {issue: choice[0] for issue, choice in zip(self.payoff_schedule.keys(), combination)}
            for combination in combinations
            if sum(choice[1] for choice in combination) >= self.walk_away_point
        ]
        
        # Convert valid combinations to a string
        result_str = f"Total valid combinations: {len(valid_combinations)}\n"
        result_str += "\n".join(str(combination) for combination in valid_combinations)
        return result_str

    



    # Function to update offers and KDE
    def update_offers_and_kde(self,issue, current_offer, previous_offer):
        """
        Update the observations for the specified issue and fit the KDE model with new offers.
        """
        concession_size = abs(current_offer - previous_offer)
        
        # Add the concession size to the observations for this issue
        self.observations[issue].append(concession_size)
        
        # Step 2: Fit the KDE model with the updated observations
        if len(self.observations[issue]) > 1:  # Fit KDE only if there are at least two observations
            self.kde_models[issue].fit(np.array(self.observations[issue]).reshape(-1, 1))
        else:
            return -1  # Not enough data yet to calculate the peak
        
        # Step 3: Evaluate the density over a range of possible concession sizes
        concession_range = np.linspace(min(self.observations[issue]), max(self.observations[issue]), 1000).reshape(-1, 1)
        density = np.exp(self.kde_models[issue].score_samples(concession_range))
        
        # Step 4: Find the peak density and the corresponding concession size
        peak_concession_size = concession_range[np.argmax(density)]
        
        # Return the peak concession size and the current concession size
        return peak_concession_size[0]

    def check_and_initialize_slots(self, state: MessageState):
        """Initialize negotiation slots if they don't exist"""
        logger.info("checking and initializing slots")
        required_slots = [
            "turn", 
            "episode_done", 
            "current_issue",
            "resolved_issues",
            "financing",
            "warranty",
            "color",
            "price",
            "current_target"
        ]
        
        if not hasattr(state, 'slots'):
            state.slots = {}
            
        for slot_name in required_slots:
            if slot_name not in state.slots:
                logger.info(f"Initializing slot: {slot_name}")
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
                
                elif slot_name == "current_issue":
                    state.slots["current_issue"] = [Slot(
                        name="current_issue",
                        type="string",
                        value="",
                        enum=[],
                        description="This tracks the current issue being negotiated.",
                        prompt="",
                        required=False,
                        verified=True)]
                
                elif slot_name == "resolved_issues":
                    state.slots["resolved_issues"] = [Slot(
                        name="resolved_issues",
                        type="list",
                        value=[],
                        enum=[],
                        description="This tracks which issues have been resolved.",
                        prompt="",
                        required=False,
                        verified=True)]
                
                elif slot_name in ["financing", "warranty", "color", "price"]:
                    state.slots[slot_name] = [Slot(
                        name=slot_name,
                        type="string",
                        value=json.dumps({"current": None, "previous": None}),
                        enum=[],
                        description=f"This tracks the current and previous offers for {slot_name}.",
                        prompt="",
                        required=False,
                        verified=True)]
                
                elif slot_name == "current_target":
                    state.slots["current_target"] = [Slot(
                        name="current_target",
                        type="string",
                        value=json.dumps({}),
                        enum=[],
                        description="This tracks the target values for each issue.",
                        prompt="",
                        required=False,
                        verified=True)]

    def get_response(self, state: MessageState) -> MessageState:
        """Generate response based on negotiation state"""
        self.check_and_initialize_slots(state)
        start_time = time.time()
        
        # Update turn counter
        current_turn = state.slots["turn"][0].value
        
        if current_turn == 0:
            # First turn initialization
            state.slots["turn"][0].value += 1
            
            # Load the core negotiation prompt
            base_dir = dirname(abspath(__file__))
            prompt_path = dirname(base_dir) + "/negotiation_prompts/fourissues.txt"
            with open(prompt_path) as f:
                core_prompt = f.read()
            
            # Generate initial combinations and importance ranking
            comb = self.generate_combos()
            common_sense_ranking = self.common_sense_importance()
            
            # Add system messages to message flow
            state.message_flow += f"\nSystem Prompt: {core_prompt}"
            state.message_flow += f"\nValid Combinations: {comb}"
            state.message_flow += f"\nCommon Sense Importance Estimate: {common_sense_ranking}"
            
            # Build the first turn prompt
            prompt = f"""
            {core_prompt}
            
            Available combinations: {comb}
            Issue importance ranking: {common_sense_ranking}
            
            The user has already responded to an ice breaker with: "{state.user_message.message}"
            Continue the conversation naturally and transition into discussing the car sale negotiation.
            Do not repeat an ice breaker since one was already done.
            """
            
            response = self.llm.invoke(prompt).content.strip()
            self.delay_response(response, time.time() - start_time)
            state.response = response
            
        else:
            # Update turn counter
            state.slots["turn"][0].value += 1
            
            # Extract current issue and offers using full history
            current_status = self.extract_issue_and_offers(state.user_message.history)
            
            if current_status:
                pattern = r"ISSUE:\s*(.*?)\s*CURRENT OFFER:\s*(\d+)\s*PREVIOUS OFFER:\s*(\d+)"
                match = re.search(pattern, current_status)
                
                if match:
                    issue = match.group(1)
                    current_offer = match.group(2)
                    previous_offer = match.group(3)
                    
                    # Update issue slots
                    issue_lower = issue.lower()
                    issue_data = json.loads(state.slots[issue_lower][0].value)
                    issue_data["current"] = current_offer
                    issue_data["previous"] = previous_offer
                    state.slots[issue_lower][0].value = json.dumps(issue_data)
                    
                    # Update KDE for numerical issues
                    if issue != 'COLOR':
                        kde_peak = self.update_offers_and_kde(issue, float(current_offer), float(previous_offer))
                        if kde_peak != -1:
                            # Update current_target as JSON string
                            current_target = json.loads(state.slots["current_target"][0].value)
                            current_target[issue] = kde_peak
                            state.slots["current_target"][0].value = json.dumps(current_target)
                            state.message_flow += f"\nKDE Peak Density Offer Difference: {kde_peak}"
                
                # Add current status to message flow
                state.message_flow += f"\nCurrent Issue and Offer Details: {current_status}"
            
            # Analyze user personality using full history
            personality = self.determine_user_personality(state.user_message.history)
            state.message_flow += f"\nUser Personality: {personality}"
            
            # Load the core prompt for context
            base_dir = dirname(abspath(__file__))
            prompt_path = dirname(base_dir) + "/negotiation_prompts/fourissues.txt"
            with open(prompt_path) as f:
                core_prompt = f.read()
            
            # Generate response based on current state
            prompt = f"""
            {core_prompt}
            
            Current turn: {state.slots['turn'][0].value}
            Current issue: {state.slots['current_issue'][0].value}
            Resolved issues: {state.slots['resolved_issues'][0].value}
            User personality: {personality}
            
            Current offers:
            Financing: {state.slots['financing'][0].value}
            Warranty: {state.slots['warranty'][0].value}
            Color: {state.slots['color'][0].value}
            Price: {state.slots['price'][0].value}
            
            Current Issue and Offer Details: {current_status}
            
            Previous conversation:
            {state.user_message.history}
            
            User's message: {state.user_message.message}
            """
            
            response = self.llm.invoke(prompt).content.strip()
            self.delay_response(response, time.time() - start_time)
            state.response = response
        
        # Check if episode should end
        if state.slots["turn"][0].value >= 8:
            state.slots["episode_done"][0].value = True
        
        return state


    def calculate_scores(self, final_outcomes: Dict[str, str]) -> Tuple[int, int]:
        """Calculate buyer and seller scores based on final outcomes.
        
        Args:
            final_outcomes: Dictionary mapping issues to their agreed values
            
        Returns:
            Tuple[int, int]: Buyer score and seller score
        """
        buyer_score = 0
        seller_score = 0

        # Loop through each issue and update scores
        for issue, outcome in final_outcomes.items():
            buyer_score += self.buyer_utilities[issue][outcome]
            seller_score += self.seller_utilities[issue][outcome]

        return buyer_score, seller_score


    # Monitor instance to track resolved issues and store final outcomes
    def monitor_instance(self, state: MessageState, issues: List[str]) -> Optional[Dict[str, str]]:
        """Monitor the negotiation progress and track resolved issues.
        
        Args:
            state: Current message state containing conversation history
            issues: List of issues to track
            
        Returns:
            Optional[Dict[str, str]]: Final outcomes if all issues are resolved, None otherwise
        """
        # Format the monitor prompt with the issues
        monitor_prompt = self.MONITOR_PROMPT.format(issues=issues)
        
        # Create the prompt using the full conversation history
        prompt = f"""
        {monitor_prompt}
        
        Conversation history:
        {state.user_message.history}
        """
        
        # Call the LLM for monitoring
        monitor_response = self.llm.invoke(prompt).content.strip()
        
        # Check for the final result when all issues are resolved
        if "NUMBER OF RESOLVED ISSUES=4" in monitor_response:
            # Parse and store the outcomes in the specified format
            final_outcome_start = monitor_response.index("NUMBER OF RESOLVED ISSUES=4") + len("NUMBER OF RESOLVED ISSUES=4")
            final_outcomes = monitor_response[final_outcome_start:].strip().splitlines()
            final_outcomes_dict = {}
            
            for outcome in final_outcomes:
                if '=' not in outcome:
                    continue
                    
                issue, agreement = outcome.split('=')
                agreement = agreement.strip()
                
                # Remove commas from the PRICE if present
                if issue.strip() == "PRICE":
                    agreement = agreement.replace(',', '')
                
                final_outcomes_dict[issue.strip()] = agreement
            
            # Add the outcomes to message flow for tracking
            state.message_flow += f"\nFinal Outcomes: {final_outcomes_dict}"
            return final_outcomes_dict
        
        return None

    def _create_action_graph(self):
        """Create a processing flow for persuasion strategy."""
        workflow =StateGraph(MessageState)
        
        # Add persuasion strategy node
        workflow.add_node("negotiation_response", self.get_response)
        
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
        
        # Debug the incoming state
        self.check_and_initialize_slots(msg_state)
        
        # Run the action graph
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        
        # Convert the result back to a MessageState
        response_state = MessageState.model_validate(result)
        
        # Check for final outcomes
        issues = ["FINANCING", "WARRANTY", "COLOR", "PRICE"]
        final_outcomes = self.monitor_instance(response_state, issues)
        
        if final_outcomes:
            # Calculate scores based on final outcomes
            buyer_score, seller_score = self.calculate_scores(final_outcomes)
            
            # Add scores to message flow
            response_state.message_flow += f"\nBuyer Score: {buyer_score}, Seller Score: {seller_score}"
            
            # Mark episode as done if we have final outcomes
            response_state.slots["episode_done"][0].value = True
        
        logger.info(f"State after graph execution - slots: {response_state.slots}")
        return response_state.model_dump()