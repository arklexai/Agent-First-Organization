"""Human-in-the-loop worker implementation for the Arklex framework.

This module provides functionality for human-in-the-loop interactions, including
slot filling and verification with human oversight.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from langgraph.graph import StateGraph, START

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.orchestrator.NLU.core.slot import SlotFiller
from arklex.utils.graph_state import MessageState, StatusEnum
from arklex.env.workers.utils.chat_client import ChatClient
from arklex.utils.slot import Slot

logger = logging.getLogger(__name__)


# @register_worker
class HITLWorker(BaseWorker):
    """Human-in-the-loop worker for slot filling and verification.

    This class provides functionality for human-in-the-loop interactions,
    allowing human oversight of slot filling and verification processes.

    Attributes:
        slot_fill_api (Optional[SlotFiller]): Slot filling API instance
        // ... rest of attributes ...
    """

    description: str = "This is a template for a HITL worker."
    mode: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    verifier: List[str] = []

    slot_fill_api: Optional[SlotFiller] = None

    # def __init__(self, server_ip: str, server_port: int, name: str):
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the HITL worker.

        Args:
            **kwargs: Keyword arguments for initializing the worker
        """
        super().__init__(**kwargs)

        self.action_graph: StateGraph = self._create_action_graph()

    def verify_literal(self, state: MessageState) -> Tuple[bool, str]:
        """Override this method to allow verification on the message, either orchestrator's message or user's message
        Case: user's message
        Before the bot generate the response for the user's query, the framework decide whether it need to call human for the help because the user directly request so
        Case: orchestrator's message
        After the bot generate the response for the user's query, the framework decide whether it need to call human for the help because of the low confidence of the bot's response

        Args:
            state (MessageState): The current message state

        Returns:
            tuple[bool, str]: A tuple containing a boolean indicating whether verification is needed and a string message
        """
        return True, ""

    def verify_slots(self, message: Any) -> Tuple[bool, str]:
        """Override this method to allow verification on the slots"""
        return True, ""

    def verify(self, state: MessageState) -> Tuple[bool, str]:
        """Override this method to allow advanced verification on MessageState object"""
        need_hitl: bool
        message_literal: str
        need_hitl, message_literal = self.verify_literal(state)
        if need_hitl:
            return True, message_literal

        need_hitl, message_slot = self.verify_slots(state.slots)
        if need_hitl:
            return True, message_slot

        return False, ""

    def init_slot_filler(self, slot_fill_api: Any) -> None:
        """Initialize the slot filling API.

        Args:
            slot_fill_api: API endpoint for slot filling
        """
        self.slot_fill_api = SlotFiller(slot_fill_api)

    def create_prompt(self) -> str:
        """Create a prompt for the HITL mc worker"""
        return (
            self.params["intro"]
            + "\n"
            + "\n".join(
                f"({key}) {item}" for key, item in self.params["choices"].items()
            )
        )

    def chat(self, state: MessageState) -> MessageState:
        """Connects to chat with the human in the loop"""
        client: ChatClient = ChatClient(
            self.server_ip, self.server_port, name=self.name, mode="c"
        )
        return client.sync_main()

        # arklex pseudocode
        # chat_history = await server_chat() # BACKEND CHATS WITH USER HERE'''
        # state.messageFlow = to_message_flow(chat_history)
        # state.messageFlow['result'] = chat_history[-1]
        return state

    def multiple_choice(self, state: MessageState) -> MessageState:
        """Connects to give human in the loop multiple choice"""
        client: ChatClient = ChatClient(
            self.server_ip, self.server_port, name=self.name, mode="ro"
        )
        return client.sync_main(message=self.create_prompt())

    def hitl(self, state: MessageState) -> str:
        """Human in the loop function"""
        result: Optional[str] = None
        match self.mode:
            case "chat":
                chat_result: MessageState = self.chat(state)
                state.user_message.history += "\n" + chat_result
                state.user_message.message = chat_result.split(f"{self.name}: ")[
                    -1
                ].split(":")[0]
                result = "Live Chat Completed"

            case "mc":
                attempts: int = self.params["max_retries"]

                for _ in range(attempts):
                    selection: MessageState = self.multiple_choice(state)

                    result = self.params["choices"].get(selection)

                    if result:
                        break
                else:
                    result = self.params["default"]

            case _:
                return self.error(state)

        state.response = result
        return state

    def fallback(self, state: MessageState) -> MessageState:
        """The message of the fallback

        Args:
            state (MessageState): The current message state

        Returns:
            MessageState: The updated state
        """
        state.message_flow = "The user don't need human help"
        state.status = StatusEnum.COMPLETE
        return state

    def _create_action_graph(self) -> StateGraph:
        workflow: StateGraph = StateGraph(MessageState)
        # Add nodes for each worker
        workflow.add_node("hitl", self.hitl)
        # Add edges
        workflow.add_edge(START, "hitl")
        return workflow

    def _execute(self, state: MessageState, **kwargs: Any) -> MessageState:
        if not self.verify(state):
            return self.error(state)

        graph = self.action_graph.compile()
        result: MessageState = graph.invoke(state)
        return result


@register_worker
class HITLWorkerTestChat(HITLWorker):
    """This worker is designed to start live chat locally
    Status: Not in use (as of 2025-02-20)

    Args:
        HITLWorker (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    description: str = "Chat with a real end user"
    mode: str = "chat"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_ip: Optional[str] = kwargs.get("server_ip")
        self.server_port: Optional[int] = kwargs.get("server_port")
        self.name: Optional[str] = kwargs.get("name")

        if not self.server_ip or not self.server_port:
            raise ValueError("Server IP and Port are required")

    def verify_literal(self, message: str) -> bool:
        return "chat" in message


@register_worker
class HITLWorkerTestMC(HITLWorker):
    """This worker is designed to start multiple choice human-in-the-loop worker locally
    Status: Not in use (as of 2025-02-20)

    Args:
        HITLWorker (_type_): _description_

    Returns:
        _type_: _description_
    """

    description: str = "Get confirmation from a real end user in purchasing"
    mode: str = "mc"
    params: Dict[str, Any] = {
        "intro": "Should the user continue with this purchase? (Y/N)",
        "max_retries": 5,
        "default": "User is not allowed to continue with the purchase",
        "choices": {
            "Y": "User is allowed to continue with the purchase",
            "N": "User is not allowed to continue with the purchase",
        },
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_ip: Optional[str] = kwargs.get("server_ip")
        self.server_port: Optional[int] = kwargs.get("server_port")
        self.name: Optional[str] = kwargs.get("name")

    def verify_literal(self, message: str) -> bool:
        return "buy" in message


@register_worker
class HITLWorkerChatFlag(HITLWorker):
    """This worker is designed to start live chat with another built server.
    So it will return the indicator of what type of human help needed.

    Args:
        HITLWorker (_type_): _description_

    Returns:
        MessageState: with hitl value in the MessageState[metadata]
    """

    description: str = "Human in the loop worker"
    mode: str = "chat"

    def verify_literal(self, state: MessageState) -> Tuple[bool, str]:
        """[TODO] Need to implement for orchestrator message as well
        (as of 2025-02-20)
        This method is to check the message from the user, since in the NLU, we already determine that the user wants to chat with the human in the loop.

        Args:
            message (str): _description_

        Returns:
            bool: _description_
        """
        message: str = "I'll connect you to a representative!"

        return True, message

    def _execute(self, state: MessageState, **kwargs: Any) -> MessageState:
        if not state.metadata.hitl:
            need_hitl: bool
            message: str
            need_hitl, message = self.verify(state)
            if not need_hitl:
                return self.fallback(state)

            state.response = (
                "[[sending confirmation : this should not show up for user]]"
            )
            state.metadata.hitl = "live"
            state.status = StatusEnum.STAY

        else:
            state.message_flow = "Live chat completed"
            state.metadata.hitl = None
            state.status = StatusEnum.COMPLETE

        logger.info(state.message_flow)
        return state


@register_worker
class HITLWorkerMCFlag(HITLWorker):
    """This worker is designed to start live chat with another built server.
    So it will return the indicator of what type of human help needed.
    Status: Not in use (as of 2025-02-20)

    Args:
        HITLWorker (_type_): _description_

    Returns:
        MessageState: with hitl value in the MessageState[metadata]
    """

    description: str = "Get confirmation from a real end user in purchasing"
    mode: str = "mc"
    params: Dict[str, Any] = {
        "intro": "Should the user continue with this purchase? (Y/N)",
        "max_retries": 5,
        "default": "User is not allowed to continue with the purchase",
        "choices": {
            "Y": "User is allowed to continue with the purchase",
            "N": "User is not allowed to continue with the purchase",
        },
    }

    def verify_literal(self, message: str) -> bool:
        return "buy" in message

    def _execute(self, state: MessageState, **kwargs: Any) -> MessageState:
        if not state.metadata.hitl:
            need_hitl: bool
            _: str
            need_hitl, _ = self.verify(state)
            if not need_hitl:
                return self.fallback(state)

            state.response = (
                "[[sending confirmation : this should not show up for user]]"
            )
            state.metadata.hitl = "mc"
            state.metadata.attempts = self.params.get("max_retries", 3)
            state.status = StatusEnum.STAY

        else:
            result: Optional[str] = self.params["choices"].get(
                state.user_message.message
            )  # not actually user message but system confirmation

            if result:
                state.response = result
                state.metadata.hitl = None
                state.status = StatusEnum.COMPLETE
                return state

            state.metadata.attempts -= 1
            if state.metadata.attempts <= 0:
                state.response = self.params["default"]
                state.metadata.hitl = None
                state.status = StatusEnum.INCOMPLETE
                return state

            state.response = (
                "[[sending confirmation : this should not show up for user]]"
            )
            state.metadata.hitl = "mc"
            state.status = StatusEnum.STAY

        return state
