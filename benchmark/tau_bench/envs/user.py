# Copyright Sierra

import abc
import enum
from litellm import completion
from typing import Optional, List, Dict, Any, Union, ClassVar


class BaseUserSimulationEnv(abc.ABC):
    """
    Abstract base class for user simulation environments.

    This class defines the interface that all user simulation implementations must follow.
    """

    metadata: ClassVar[Dict[str, Any]] = {}

    @abc.abstractmethod
    def reset(self, instruction: Optional[str] = None) -> str:
        """
        Reset the user simulation environment.

        Args:
            instruction (Optional[str]): Optional instruction for the user.

        Returns:
            str: The initial user response.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, content: str) -> str:
        """
        Take a step in the user simulation environment.

        Args:
            content (str): The content to respond to.

        Returns:
            str: The user's response.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_total_cost(self) -> float:
        """
        Get the total cost of the simulation.

        Returns:
            float: The total cost.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


class HumanUserSimulationEnv(BaseUserSimulationEnv):
    """
    User simulation environment that uses human input.
    """

    def reset(self, instruction: str) -> str:
        """
        Reset the environment and get initial human input.

        Args:
            instruction (str): The instruction to display.

        Returns:
            str: The human's response.
        """
        return input(f"{instruction}\n")

    def step(self, content: str) -> str:
        """
        Get human input for the next step.

        Args:
            content (str): The content to respond to.

        Returns:
            str: The human's response.
        """
        return input(f"{content}\n")

    def get_total_cost(self) -> float:
        """
        Get the total cost (always 0 for human simulation).

        Returns:
            float: Always returns 0.
        """
        return 0


class LLMUserSimulationEnv(BaseUserSimulationEnv):
    """
    User simulation environment that uses a language model.
    """

    def __init__(self, model: str, provider: str) -> None:
        """
        Initialize the LLM user simulation environment.

        Args:
            model (str): The model to use.
            provider (str): The provider of the model.
        """
        super().__init__()
        self.messages: List[Dict[str, Any]] = []
        self.model: str = model
        self.provider: str = provider
        self.total_cost: float = 0.0
        self.reset()

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate the next message using the language model.

        Args:
            messages (List[Dict[str, Any]]): The conversation history.

        Returns:
            str: The generated message.
        """
        res = completion(
            model=self.model, custom_llm_provider=self.provider, messages=messages
        )
        message = res.choices[0].message
        self.messages.append(message.model_dump())
        self.total_cost = res._hidden_params["response_cost"]
        return message.content

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        """
        Build the system prompt for the language model.

        Args:
            instruction (Optional[str]): Optional instruction for the user.

        Returns:
            str: The system prompt.
        """
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""

    def reset(self, instruction: Optional[str] = None) -> str:
        """
        Reset the environment and get initial response.

        Args:
            instruction (Optional[str]): Optional instruction for the user.

        Returns:
            str: The initial response.
        """
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        """
        Take a step in the environment.

        Args:
            content (str): The content to respond to.

        Returns:
            str: The user's response.
        """
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        """
        Get the total cost of the simulation.

        Returns:
            float: The total cost.
        """
        return self.total_cost


class ReactUserSimulationEnv(LLMUserSimulationEnv):
    """
    User simulation environment that uses a language model with ReAct prompting.
    """

    def __init__(self, model: str, provider: str) -> None:
        """
        Initialize the ReAct user simulation environment.

        Args:
            model (str): The model to use.
            provider (str): The provider of the model.
        """
        super().__init__(model=model, provider=provider)
        self.reset()

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        """
        Build the system prompt for the language model.

        Args:
            instruction (Optional[str]): Optional instruction for the user.

        Returns:
            str: The system prompt.
        """
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- First, generate a Thought about what to do next (this message will not be sent to the agent).
- Then, generate a one line User Response to simulate the user's message (this message will be sent to the agent).
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as the User Response without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction.

Format:

Thought:
<the thought>

User Response:
<the user response (this will be parsed and sent to the agent)>"""

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate the next message using the language model.

        Args:
            messages (List[Dict[str, Any]]): The conversation history.

        Returns:
            str: The generated message.
        """
        res = completion(
            model=self.model, custom_llm_provider=self.provider, messages=messages
        )
        message = res.choices[0].message
        self.messages.append(message.model_dump())
        self.total_cost = res._hidden_params["response_cost"]
        return self.parse_response(message.content)

    def reset(self, instruction: Optional[str] = None) -> str:
        """
        Reset the environment and get initial response.

        Args:
            instruction (Optional[str]): Optional instruction for the user.

        Returns:
            str: The initial response.
        """
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def parse_response(self, response: str) -> str:
        """
        Parse the response from the language model.

        Args:
            response (str): The response to parse.

        Returns:
            str: The parsed response.

        Raises:
            ValueError: If the response format is invalid.
        """
        if "###STOP###" in response:
            return "###STOP###"
        elif "Thought:" in response:
            _, user_response = response.split("Thought:")
            return user_response.strip()
        elif "User Response:" in response:
            _, user_response = response.split("User Response:")
            return user_response.strip()
        else:
            raise ValueError(f"Invalid response format: {response}")

    def step(self, content: str) -> str:
        """
        Take a step in the environment.

        Args:
            content (str): The content to respond to.

        Returns:
            str: The user's response.
        """
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        """
        Get the total cost of the simulation.

        Returns:
            float: The total cost.
        """
        return self.total_cost


class VerifyUserSimulationEnv(LLMUserSimulationEnv):
    """
    User simulation environment that uses a language model with verification.
    """

    def __init__(self, model: str, provider: str, max_attempts: int = 3) -> None:
        """
        Initialize the verify user simulation environment.

        Args:
            model (str): The model to use.
            provider (str): The provider of the model.
            max_attempts (int): Maximum number of verification attempts.
        """
        self.model: str = model
        self.provider: str = provider
        self.max_attempts: int = max_attempts
        self.reset()

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate the next message using the language model with verification.

        Args:
            messages (List[Dict[str, Any]]): The conversation history.

        Returns:
            str: The generated message.
        """
        attempts: int = 0
        cur_message = None
        while attempts < self.max_attempts:
            res = completion(
                model=self.model, custom_llm_provider=self.provider, messages=messages
            )
            cur_message = res.choices[0].message
            self.total_cost = res._hidden_params["response_cost"]
            if verify(self.model, self.provider, cur_message, messages):
                self.messages.append(cur_message.model_dump())
                return cur_message.content
            attempts += 1
        assert cur_message is not None
        return cur_message.content

    def reset(self, instruction: Optional[str] = None) -> str:
        """
        Reset the environment and get initial response.

        Args:
            instruction (Optional[str]): Optional instruction for the user.

        Returns:
            str: The initial response.
        """
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        """
        Take a step in the environment.

        Args:
            content (str): The content to respond to.

        Returns:
            str: The user's response.
        """
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        """
        Get the total cost of the simulation.

        Returns:
            float: The total cost.
        """
        return self.total_cost


def map_role_label(role: str) -> str:
    """
    Map a role to a display label.

    Args:
        role (str): The role to map.

    Returns:
        str: The display label.
    """
    if role == "user":
        return "Customer"
    elif role == "assistant":
        return "Agent"
    else:
        return role.capitalize()


def verify(
    model: str, provider: str, response: str, messages: List[Dict[str, Any]]
) -> bool:
    """
    Verify if a response is satisfactory.

    Args:
        model (str): The model to use for verification.
        provider (str): The provider of the model.
        response (str): The response to verify.
        messages (List[Dict[str, Any]]): The conversation history.

    Returns:
        bool: True if the response is satisfactory, False otherwise.
    """
    transcript: str = "\n".join(
        [
            f"{map_role_label(message['role'])}: {message['content']}"
            for message in messages
        ]
    )
    prompt: str = f"""You are a supervisor of the Agent in the conversation. You are given a Transcript of a conversation between a Customer and an Agent. The Customer has generated a Response, and you need to verify if it is satisfactory (true) or not (false).
Your answer will be parsed, so do not include any other text than the classification (true or false).
    
# Transcript:
{transcript}

# Response:
{response}

-----

Classification:"""
    res = completion(
        model=model,
        custom_llm_provider=provider,
        messages=[{"role": "user", "content": prompt}],
    )
    return "true" in res.choices[0].message.content.lower()


def reflect(
    model: str, provider: str, response: str, messages: List[Dict[str, Any]]
) -> str:
    """
    Generate a reflection on an unsatisfactory response.

    Args:
        model (str): The model to use for reflection.
        provider (str): The provider of the model.
        response (str): The response to reflect on.
        messages (List[Dict[str, Any]]): The conversation history.

    Returns:
        str: The reflection.
    """
    transcript: str = "\n".join(
        [
            f"{map_role_label(message['role'])}: {message['content']}"
            for message in messages
        ]
    )
    prompt: str = f"""You are a supervisor of the Agent in the conversation. You are given a Transcript of a conversation between a (simulated) Customer and an Agent. The Customer generated a Response that was marked as unsatisfactory by you.
You need to generate a Reflection on what went wrong in the conversation, and propose a new Response that should fix the issues.
Your answer will be parsed, so do not include any other text than the classification (true or false).
    
# Transcript:
{transcript}

# Response:
{response}

-----

Reflection:"""
    res = completion(
        model=model,
        custom_llm_provider=provider,
        messages=[{"role": "user", "content": prompt}],
    )
    return res.choices[0].message.content


class ReflectionUserSimulationEnv(LLMUserSimulationEnv):
    """
    User simulation environment that uses a language model with reflection.
    """

    def __init__(self, model: str, provider: str, max_attempts: int = 2) -> None:
        """
        Initialize the reflection user simulation environment.

        Args:
            model (str): The model to use.
            provider (str): The provider of the model.
            max_attempts (int): Maximum number of reflection attempts.
        """
        self.model: str = model
        self.provider: str = provider
        self.max_attempts: int = max_attempts
        self.reset()

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate the next message using the language model with reflection.

        Args:
            messages (List[Dict[str, Any]]): The conversation history.

        Returns:
            str: The generated message.
        """
        attempts: int = 0
        cur_message = None
        while attempts < self.max_attempts:
            res = completion(
                model=self.model, custom_llm_provider=self.provider, messages=messages
            )
            cur_message = res.choices[0].message
            self.total_cost = res._hidden_params["response_cost"]
            if verify(self.model, self.provider, cur_message, messages):
                self.messages.append(cur_message.model_dump())
                return cur_message.content
            reflection: str = reflect(self.model, self.provider, cur_message, messages)
            messages.append({"role": "user", "content": reflection})
            attempts += 1
        assert cur_message is not None
        return cur_message.content

    def reset(self, instruction: Optional[str] = None) -> str:
        """
        Reset the environment and get initial response.

        Args:
            instruction (Optional[str]): Optional instruction for the user.

        Returns:
            str: The initial response.
        """
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        """
        Take a step in the environment.

        Args:
            content (str): The content to respond to.

        Returns:
            str: The user's response.
        """
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        """
        Get the total cost of the simulation.

        Returns:
            float: The total cost.
        """
        return self.total_cost


class UserStrategy(enum.Enum):
    """
    Enumeration of available user simulation strategies.
    """

    HUMAN = "human"
    LLM = "llm"
    REACT = "react"
    VERIFY = "verify"
    REFLECTION = "reflection"


def load_user(
    user_strategy: Union[str, UserStrategy],
    model: Optional[str] = "gpt-4o",
    provider: Optional[str] = None,
) -> BaseUserSimulationEnv:
    """
    Load a user simulation environment.

    Args:
        user_strategy (Union[str, UserStrategy]): The strategy to use.
        model (Optional[str]): The model to use.
        provider (Optional[str]): The provider of the model.

    Returns:
        BaseUserSimulationEnv: The loaded user simulation environment.
    """
    if isinstance(user_strategy, str):
        user_strategy = UserStrategy(user_strategy)
    if user_strategy == UserStrategy.HUMAN:
        return HumanUserSimulationEnv()
    elif user_strategy == UserStrategy.LLM:
        return LLMUserSimulationEnv(model=model, provider=provider)
    elif user_strategy == UserStrategy.REACT:
        return ReactUserSimulationEnv(model=model, provider=provider)
    elif user_strategy == UserStrategy.VERIFY:
        return VerifyUserSimulationEnv(model=model, provider=provider)
    elif user_strategy == UserStrategy.REFLECTION:
        return ReflectionUserSimulationEnv(model=model, provider=provider)
    else:
        raise ValueError(f"Unknown user strategy: {user_strategy}")
