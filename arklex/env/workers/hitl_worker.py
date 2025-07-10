"""Human-in-the-loop worker implementation for the Arklex framework.

This module provides functionality for human-in-the-loop interactions, including
slot filling and verification with human oversight. It supports both chat-based
and multiple choice interaction modes.

Key Components:
- HITLWorker: Base human-in-the-loop worker class
- HITLWorkerTestChat: Test implementation for local chat functionality
- HITLWorkerTestMC: Test implementation for multiple choice functionality
- HITLWorkerChatFlag: Production chat worker with flag-based state management
- HITLWorkerMCFlag: Production multiple choice worker with flag-based state management

Features:
- Human verification of bot responses
- Slot filling with human oversight
- Multiple interaction modes (chat, multiple choice)
- State management for conversation flow
- Error handling and fallback mechanisms
- Integration with slot filling APIs

Usage:
    from arklex.env.workers.hitl_worker import HITLWorker

    # Initialize HITL worker
    worker = HITLWorker(
        name="verification_worker",
        server_ip="localhost",
        server_port=8080,
        mode="chat"
    )

    # Execute verification
    result = worker._execute(message_state)
"""

from typing import Any, TypedDict

from langgraph.graph import START, StateGraph

from arklex.env.workers.utils.chat_client import ChatClient
from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.orchestrator.entities.msg_state_entities import MessageState, StatusEnum
from arklex.orchestrator.NLU.core.slot import SlotFiller
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class HITLWorkerKwargs(TypedDict, total=False):
    """Type definition for kwargs used in HITLWorker.__init__ method.

    Attributes:
        name: Name identifier for the worker
        server_ip: IP address of the HITL server
        server_port: Port number of the HITL server
        mode: Interaction mode ("chat" or "mc")
        params: Configuration parameters for the worker
        verifier: List of verification criteria
        slot_fill_api: Slot filling API instance
    """

    name: str
    server_ip: str
    server_port: int
    mode: str
    params: dict[str, Any]
    verifier: list[str]
    slot_fill_api: SlotFiller


class HITLWorkerExecuteKwargs(TypedDict, total=False):
    """Type definition for kwargs used in HITLWorker._execute method.

    This class defines the structure for execution parameters that can be passed
    to the HITL worker's execute method.
    """

    # Add specific execution parameters as needed
    pass


# @register_worker
class HITLWorker(BaseWorker):
    """Human-in-the-loop worker for slot filling and verification.

    This class provides functionality for human-in-the-loop interactions,
    allowing human oversight of slot filling and verification processes.
    It supports both chat-based and multiple choice interaction modes.

    Attributes:
        name: Name identifier for the worker
        server_ip: IP address of the HITL server
        server_port: Port number of the HITL server
        mode: Interaction mode ("chat" or "mc")
        params: Configuration parameters for the worker
        verifier: List of verification criteria
        slot_fill_api: Slot filling API instance
        action_graph: StateGraph for managing worker workflow
    """

    description: str = "This is a template for a HITL worker."
    mode: str | None = None
    params: dict[str, Any] | None = None
    verifier: list[str] = []

    slot_fill_api: SlotFiller | None = None

    def __init__(self, **kwargs: HITLWorkerKwargs) -> None:
        """Initialize the HITL worker.

        Args:
            **kwargs: Keyword arguments containing worker configuration
        """
        # Initialize attributes from kwargs
        self.name = kwargs.get("name")
        self.server_ip = kwargs.get("server_ip")
        self.server_port = kwargs.get("server_port")
        self.mode = kwargs.get("mode", self.mode)
        self.params = kwargs.get("params", self.params)
        self.verifier = kwargs.get("verifier", self.verifier)
        self.slot_fill_api = kwargs.get("slot_fill_api", self.slot_fill_api)
        self.action_graph: StateGraph = self._create_action_graph()

    def verify_literal(self, state: MessageState) -> tuple[bool, str]:
        """Verify if human intervention is needed based on the message content.

        This method determines whether human verification is required for the
        current message state. It can be overridden to implement custom
        verification logic.

        Args:
            state: The current message state

        Returns:
            Tuple containing a boolean indicating whether verification is needed
            and a string message explaining the reason
        """
        return True, ""

    def verify_slots(self, message: dict[str, Any]) -> tuple[bool, str]:
        """Verify if human intervention is needed based on slot information.

        This method determines whether human verification is required for the
        slot filling process. It can be overridden to implement custom
        slot verification logic.

        Args:
            message: Dictionary containing slot information

        Returns:
            Tuple containing a boolean indicating whether verification is needed
            and a string message explaining the reason
        """
        return True, ""

    def verify(self, state: MessageState) -> tuple[bool, str]:
        """Perform comprehensive verification to determine if human intervention is needed.

        This method combines literal and slot verification to determine if
        human-in-the-loop processing is required.

        Args:
            state: The current message state

        Returns:
            Tuple containing a boolean indicating whether verification is needed
            and a string message explaining the reason
        """
        need_hitl: bool
        message_literal: str
        need_hitl, message_literal = self.verify_literal(state)
        if need_hitl:
            return True, message_literal

        need_hitl, message_slot = self.verify_slots(state.slots)
        if need_hitl:
            return True, message_slot

        return False, ""

    def init_slot_filler(self, slot_fill_api: SlotFiller) -> None:
        """Initialize the slot filling API.

        Args:
            slot_fill_api: API endpoint for slot filling
        """
        self.slot_fill_api = SlotFiller(slot_fill_api)

    def create_prompt(self) -> str:
        """Create a prompt for the HITL multiple choice worker.

        Returns:
            Formatted prompt string for multiple choice interactions
        """
        return (
            self.params["intro"]
            + "\n"
            + "\n".join(
                f"({key}) {item}" for key, item in self.params["choices"].items()
            )
        )

    def chat(self, state: MessageState) -> MessageState:
        """Connect to chat with the human in the loop.

        This method establishes a connection to the HITL chat server and
        handles the conversation flow.

        Args:
            state: Current message state

        Returns:
            Updated message state after chat interaction
        """
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
        """Connect to give human in the loop multiple choice options.

        This method establishes a connection to the HITL server for multiple
        choice interactions.

        Args:
            state: Current message state

        Returns:
            Updated message state after multiple choice interaction
        """
        client: ChatClient = ChatClient(
            self.server_ip, self.server_port, name=self.name, mode="ro"
        )
        return client.sync_main(message=self.create_prompt())

    def hitl(self, state: MessageState) -> str:
        """Execute human-in-the-loop interaction based on the configured mode.

        This method handles the main HITL logic, routing to either chat or
        multiple choice mode based on the worker configuration.

        Args:
            state: Current message state

        Returns:
            Result string from the HITL interaction
        """
        result: str | None = None
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
        """Handle fallback when human intervention is not needed.

        Args:
            state: The current message state

        Returns:
            MessageState: The updated state with fallback message
        """
        state.message_flow = "The user don't need human help"
        state.status = StatusEnum.COMPLETE
        return state

    def _create_action_graph(self) -> StateGraph:
        """Create the action graph for the HITL worker workflow.

        Returns:
            StateGraph representing the worker's execution flow
        """
        workflow: StateGraph = StateGraph(MessageState)
        # Add nodes for each worker
        workflow.add_node("hitl", self.hitl)
        # Add edges
        workflow.add_edge(START, "hitl")
        return workflow

    def _execute(
        self, state: MessageState, **kwargs: HITLWorkerExecuteKwargs
    ) -> MessageState:
        """Execute the HITL worker with the given state.

        This method performs verification to determine if human intervention
        is needed, and either executes the HITL workflow or returns an error.

        Args:
            state: Current message state
            **kwargs: Additional execution parameters

        Returns:
            Updated message state after execution
        """
        need_hitl, _ = self.verify(state)
        if not need_hitl:
            return self.error(state)

        graph = self.action_graph.compile()
        result: MessageState = graph.invoke(state)
        return result

    def error(self, state: MessageState) -> MessageState:
        """Handle error state when HITL processing fails.

        Args:
            state: Current message state

        Returns:
            MessageState with error status
        """
        state.status = StatusEnum.INCOMPLETE
        return state


@register_worker
class HITLWorkerTestChat(HITLWorker):
    """Test implementation for local chat functionality.

    This worker is designed to start live chat locally for testing purposes.
    It provides a simple chat interface for development and debugging.

    Status: Not in use (as of 2025-02-20)

    Attributes:
        description: Description of the worker functionality
        mode: Interaction mode set to "chat"
    """

    description: str = "This worker is designed to start live chat locally"
    mode: str = "chat"

    def __init__(self, **kwargs: HITLWorkerKwargs) -> None:
        """Initialize the test chat worker.

        Args:
            **kwargs: Keyword arguments containing worker configuration

        Raises:
            ValueError: If server IP or port are not provided
        """
        # Initialize attributes from kwargs
        self.name = kwargs.get("name")
        self.server_ip = kwargs.get("server_ip")
        self.server_port = kwargs.get("server_port")
        self.mode = kwargs.get("mode", self.mode)
        self.params = kwargs.get("params", self.params)
        self.verifier = kwargs.get("verifier", self.verifier)
        self.slot_fill_api = kwargs.get("slot_fill_api", self.slot_fill_api)
        if not self.server_ip or not self.server_port:
            raise ValueError("Server IP and Port are required")
        self.action_graph: StateGraph = self._create_action_graph()

    def verify_literal(self, message: str) -> bool:
        """Verify if chat interaction is needed based on message content.

        Args:
            message: Message content to check

        Returns:
            True if message contains "chat", False otherwise
        """
        return "chat" in message


@register_worker
class HITLWorkerTestMC(HITLWorker):
    """Test implementation for multiple choice functionality.

    This worker is designed to start multiple choice human-in-the-loop
    interactions locally for testing purposes. It provides a simple
    Y/N confirmation interface.

    Status: Not in use (as of 2025-02-20)

    Attributes:
        description: Description of the worker functionality
        mode: Interaction mode set to "mc"
        params: Configuration for multiple choice interaction
    """

    description: str = "Get confirmation from a real end user in purchasing"
    mode: str = "mc"
    params: dict[str, Any] = {
        "intro": "Should the user continue with this purchase? (Y/N)",
        "max_retries": 5,
        "default": "User is not allowed to continue with the purchase",
        "choices": {
            "Y": "User is allowed to continue with the purchase",
            "N": "User is not allowed to continue with the purchase",
        },
    }

    def __init__(self, **kwargs: HITLWorkerKwargs) -> None:
        """Initialize the test multiple choice worker.

        Args:
            **kwargs: Keyword arguments containing worker configuration

        Raises:
            ValueError: If server IP or port are not provided
        """
        # Initialize attributes from kwargs
        self.name = kwargs.get("name")
        self.server_ip = kwargs.get("server_ip")
        self.server_port = kwargs.get("server_port")
        self.mode = kwargs.get("mode", self.mode)
        self.params = kwargs.get("params", self.params)
        self.verifier = kwargs.get("verifier", self.verifier)
        self.slot_fill_api = kwargs.get("slot_fill_api", self.slot_fill_api)
        if not self.server_ip or not self.server_port:
            raise ValueError("Server IP and Port are required")
        self.action_graph: StateGraph = self._create_action_graph()

    def verify_literal(self, message: str) -> bool:
        """Verify if multiple choice interaction is needed based on message content.

        Args:
            message: Message content to check

        Returns:
            True if message contains "buy", False otherwise
        """
        return "buy" in message


@register_worker
class HITLWorkerChatFlag(HITLWorker):
    """Production chat worker with flag-based state management.

    This worker is designed to start live chat with another built server.
    It uses flag-based state management to track conversation flow and
    returns indicators of what type of human help is needed.

    Attributes:
        description: Description of the worker functionality
        mode: Interaction mode set to "chat"
    """

    description: str = "Human in the loop worker"
    mode: str = "chat"

    def verify_literal(self, state: MessageState) -> tuple[bool, str]:
        """Verify if chat interaction is needed based on message state.

        This method checks the message from the user. Since the NLU has already
        determined that the user wants to chat with a human, this method
        returns a positive response with a connection message.

        Args:
            state: Current message state

        Returns:
            Tuple containing True and connection message
        """
        message: str = "I'll connect you to a representative!"
        return True, message

    def _execute(
        self, state: MessageState, **kwargs: HITLWorkerExecuteKwargs
    ) -> MessageState:
        """Execute the chat worker with flag-based state management.

        This method manages the conversation flow using metadata flags to
        track the state of the HITL interaction.

        Args:
            state: Current message state
            **kwargs: Additional execution parameters

        Returns:
            Updated message state after execution
        """
        if not state.metadata.hitl:
            need_hitl: bool
            message: str
            need_hitl, message = self.verify(state)
            if not need_hitl:
                return self.fallback(state)

            state.message_flow = message
            state.metadata.hitl = "live"
            state.status = StatusEnum.STAY

        else:
            state.message_flow = "Live chat completed"
            state.metadata.hitl = None
            state.status = StatusEnum.COMPLETE

        log_context.info(state.message_flow)
        return state


@register_worker
class HITLWorkerMCFlag(HITLWorker):
    """Production multiple choice worker with flag-based state management.

    This worker is designed to handle multiple choice interactions with
    flag-based state management. It manages confirmation flows and
    tracks interaction attempts.

    Status: Not in use (as of 2025-02-20)

    Attributes:
        description: Description of the worker functionality
        mode: Interaction mode set to "mc"
        params: Configuration for multiple choice interaction
    """

    description: str = "Get confirmation from a real end user in purchasing"
    mode: str = "mc"
    params: dict[str, Any] = {
        "intro": "Should the user continue with this purchase? (Y/N)",
        "max_retries": 5,
        "default": "User is not allowed to continue with the purchase",
        "choices": {
            "Y": "User is allowed to continue with the purchase",
            "N": "User is not allowed to continue with the purchase",
        },
    }

    def verify_literal(self, message: str) -> bool:
        """Verify if multiple choice interaction is needed based on message content.

        Args:
            message: Message content to check

        Returns:
            True if message contains "buy", False otherwise
        """
        return "buy" in message

    def _execute(
        self, state: MessageState, **kwargs: HITLWorkerExecuteKwargs
    ) -> MessageState:
        """Execute the multiple choice worker with flag-based state management.

        This method manages the multiple choice interaction flow using metadata
        flags to track the state and attempts of the HITL interaction.

        Args:
            state: Current message state
            **kwargs: Additional execution parameters

        Returns:
            Updated message state after execution
        """
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
            result: str | None = self.params["choices"].get(
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
