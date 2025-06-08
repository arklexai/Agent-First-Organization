import json
import logging
import os
from os.path import dirname, abspath
import random
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

        self.MONITOR_PROMPT = """
        You are a negotiation monitor overseeing a multi-issue negotiation.

        Based on the negotiation history, please assess how many issues from {issues} have been resolved. 
        Give the answer in **exactly** this format at the end:
        NUMBER OF RESOLVED ISSUES=X where X is the number of resolved issues. 
        When X=6 and all the issues have been agreed upon, print the final outcomes for ALL 6 ISSUES 
        (LOCATION, SALARY, HEALTHCARE, VACATION, MOVING EXPENSES, JOB ASSIGNMENT) - **exactly** in this format:
        ISSUE NAME IN CAPITAL=Agreement alternative.
        """

        # Utility dictionaries for buyer and seller
        self.buyer_utilities = {
            "LOCATION": {"Boston": 3200, "Philadelphia": 1600, "New York": 1600, "San Francisco": 1600, "Chicago": 0},
            "SALARY": {"$85000": 0, "$90000": 1000, "$95000": 2000, "$100000": 3000, "$105000": 4000},
            "HEALTHCARE": {"Red Shield HMO": 0, "Ulster HMO": 1000, "Ulster + Dental": 2000, "Ajax POS": 3000, "Ajax + Dental": 4000},
            "VACATION": {"8 days": 0, "10 days": 250, "12 days": 500, "15 days": 750, "18 days": 1000},
            "MOVING EXPENSES": {"20% covered": 0, "40% covered": 250, "60% covered": 500, "80% covered": 750, "100% covered": 1000},
            "JOB ASSIGNMENT": {"Retail": 1600, "Technology": 1200, "Manufacturing": 800, "Financial Services": 400, "Pharmaceuticals": 0}
        }

        self.seller_utilities = {
            "LOCATION": {"Boston": 3200, "Philadelphia": 3200, "New York": 1600, "San Francisco": 0, "Chicago": 1600},
            "SALARY": {"$85000": 4000, "$90000": 3000, "$95000": 2000, "$100000": 1000, "$105000": 0},
            "HEALTHCARE": {"Red Shield HMO": 2000, "Ulster HMO": 1500, "Ulster + Dental": 1000, "Ajax POS": 500, "Ajax + Dental": 0},
            "VACATION": {"8 days": 2000, "10 days": 1500, "12 days": 1000, "15 days": 500, "18 days": 0},
            "MOVING EXPENSES": {"20% covered": 2000, "40% covered": 1500, "60% covered": 1000, "80% covered": 500, "100% covered": 0},
            "JOB ASSIGNMENT": {"Retail": 1600, "Technology": 400, "Manufacturing": 800, "Financial Services": 1200, "Pharmaceuticals": 0}
        }

        # Initialize the KDE models for each issue
        self.kde_models = {
            'SALARY': KernelDensity(kernel='gaussian', bandwidth=0.5),
            'HEALTHCARE': KernelDensity(kernel='gaussian', bandwidth=0.5),
            'MOVING EXPENSES': KernelDensity(kernel='gaussian', bandwidth=0.5),
        }

        # Data structure to hold observations for each issue
        self.observations = {
            'SALARY': [],
            'HEALTHCARE': [],
            'MOVING EXPENSES': [],
        }
        
        self.payoff_schedule = {
            "LOCATION": {"Boston": 3200, "Philadelphia": 3200, "New York": 1600, "San Francisco": 0, "Chicago": 1600},
            "SALARY": {"$85000": 4000, "$90000": 3000, "$95000": 2000, "$100000": 1000, "$105000": 0},
            "HEALTHCARE": {"Red Shield HMO": 2000, "Ulster HMO": 1500, "Ulster + Dental": 1000, "Ajax POS": 500, "Ajax + Dental": 0},
            "VACATION": {"8 days": 2000, "10 days": 1500, "12 days": 1000, "15 days": 500, "18 days": 0},
            "MOVING EXPENSES": {"20% covered": 2000, "40% covered": 1500, "60% covered": 1000, "80% covered": 500, "100% covered": 0},
            "JOB ASSIGNMENT": {"Retail": 1600, "Technology": 400, "Manufacturing": 800, "Financial Services": 1200, "Pharmaceuticals": 0}
        }

        # Walk-away point for valid combinations
        self.walk_away_point = 10000

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
            "LOCATION": {"Boston": 3200, "Philadelphia": 3200, "New York": 1600, "Chicago": 1600, "San Francisco": 0},
            "SALARY": {"$85000": 4000, "$90000": 3000, "$95000": 2000, "$100000": 1000, "$105000": 0},
            "HEALTHCARE": {"Red Shield HMO": 2000, "Ulster HMO": 1500, "Ulster + Dental": 1000, "Ajax POS": 500, "Ajax + Dental": 0},
            "VACATION": {"8 days": 2000, "10 days": 1500, "12 days": 1000, "15 days": 500, "18 days": 0},
            "MOVING EXPENSES": {"20% covered": 2000, "40% covered": 1500, "60% covered": 1000, "80% covered": 500, "100% covered": 0},
            "JOB ASSIGNMENT": {"Retail": 1600, "Financial Services": 1200, "Manufacturing": 800, "Technology": 400, "Pharmaceuticals": 0}
        }

        prompt = f"""
        Based on the following utilities for each issue, approximate the order of importance of these issues for J. Roberts. 
        Use common sense and Global Consulting's perspective.

        Utilities:
        {utilities}

        Provide a list ranking these issues from most to least important for J. Roberts based on common sense assumptions and give justification.
        """

        return self.llm.invoke(prompt).content.strip()

    # Importance Estimation Using KDE
    def importance_estimation_kde(self, conversation_history):
        prompt = f"""
            As the negotiation progresses, you will use Kernel Density Estimation (KDE) to estimate how much J. Roberts cares about each issue based on their offer changes. Pay attention to the size of their concessions:
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
    and extract J. Roberts' two most recent offers for each issue where applicable. 
    Note that if J. Roberts is sticking to their previous offer, their current and previous offer will be the SAME VALUE.

    Return the response **STRICTLY**in this format only for the current issue(DON'T DISPLAY PREVIOUS ISSUES):
    - ISSUE: [CURRENT Issue Name]
      CURRENT OFFER: [Most recent buyer offer as numerical value]
      PREVIOUS OFFER: [Second most recent buyer offer as numerical value]

    For issues that involve numerical values, return the value itself:
    - LOCATION: {{"Boston", "Philadelphia", "New York", "Chicago", "San Francisco"}} → return Boston, Philadelphia, New York, Chicago, San Francisco
    - SALARY: {{"$85000", "$90000", "$95000", "$100000", "$105000"}} → return 85000, 90000, 95000, 100000, 105000
    - HEALTHCARE: {{"Red Shield HMO", "Ulster HMO", "Ulster + Dental", "Ajax POS", "Ajax + Dental"}} → return Red Shield HMO, Ulster HMO, Ulster + Dental, Ajax POS, Ajax + Dental
    - VACATION: {{"8 days", "10 days", "12 days", "15 days", "18 days"}} → return 8, 10, 12, 15, 18
    - MOVING EXPENSES: {{"20% covered", "40% covered", "60% covered", "80% covered", "100% covered"}} → return 20, 40, 60, 80, 100
    - JOB ASSIGNMENT: {{"Retail", "Financial Services", "Manufacturing", "Technology", "Pharmaceuticals"}} → return Retail, Financial Services, Manufacturing, Technology, Pharmaceuticals

    If a particular issue does not have at least two offers from J. Roberts in the history, return "NO OFFERS YET".
    Conversation history:
    {conversation_history}

    Your output should only include issues with at least two offers from J. Roberts, and should follow the required numerical format.
    """

        return self.llm.invoke(prompt).content.strip()

    def generate_combos(self, sample_size=30):
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
        options = list(self.payoff_schedule.values())
        combinations = list(product(*[list(opt.items()) for opt in options]))

        # Filter combinations that meet the walk-away point
        valid_combinations = [
            {issue: choice[0] for issue, choice in zip(self.payoff_schedule.keys(), combination)}
            for combination in combinations
            if sum(choice[1] for choice in combination) >= self.walk_away_point
        ]
        
        # Shuffle and sample the combinations
        random.shuffle(valid_combinations)
        top_combos = valid_combinations[:sample_size]

        # Convert the combinations to a string
        result_str = f"Valid Combinations:\n"
        for i, combo in enumerate(top_combos, 1):
            result_str += f"Combination {i}: {combo}\n"
            
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
            "location",
            "salary",
            "healthcare",
            "vacation",
            "moving_expenses",
            "job_assignment",
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
                
                elif slot_name in ["location", "salary", "healthcare", "vacation", "moving_expenses", "job_assignment"]:
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
            prompt_path = dirname(base_dir) + "/negotiation_prompts/adaptedscenario.txt"
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
            Continue the conversation naturally and transition into discussing the job offer negotiation.
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
                    if issue not in ['LOCATION', 'JOB_ASSIGNMENT'] and current_offer and previous_offer:
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
            prompt_path = dirname(base_dir) + "/negotiation_prompts/adaptedscenario.txt"
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
            Location: {state.slots['location'][0].value}
            Salary: {state.slots['salary'][0].value}
            Healthcare: {state.slots['healthcare'][0].value}
            Vacation: {state.slots['vacation'][0].value}
            Moving Expenses: {state.slots['moving_expenses'][0].value}
            Job Assignment: {state.slots['job_assignment'][0].value}
            
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
        if "NUMBER OF RESOLVED ISSUES=6" in monitor_response:
            # Parse and store the outcomes in the specified format
            final_outcome_start = monitor_response.index("NUMBER OF RESOLVED ISSUES=6") + len("NUMBER OF RESOLVED ISSUES=6")
            final_outcomes = monitor_response[final_outcome_start:].strip().splitlines()
            final_outcomes_dict = {}
            
            for outcome in final_outcomes:
                if '=' not in outcome:
                    continue
                    
                issue, agreement = outcome.split('=')
                agreement = agreement.strip()
                
                # Remove commas from SALARY if present
                if issue.strip() == "SALARY":
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
        issues = ["LOCATION", "SALARY", "HEALTHCARE", "VACATION", "MOVING EXPENSES", "JOB ASSIGNMENT"]
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