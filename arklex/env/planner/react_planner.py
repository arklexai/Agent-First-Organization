"""ReAct (Reasoning and Acting) planner implementation for the Arklex framework.

This module provides a planner that uses the ReAct paradigm to choose tools and workers
based on task requirements and chat history. It includes functionality for resource
retrieval, trajectory planning, and action execution.

Key Components:
- ReactPlanner: Main planner class implementing ReAct methodology
- DefaultPlanner: Base planner class for simple pass-through behavior
- Action: Data model for planner actions
- EnvResponse: Data model for environment responses
- PlannerResource: Data model for planner resources

Features:
- Tool and worker selection based on task analysis
- RAG-based resource retrieval
- Trajectory planning and summarization
- Multi-step action planning
- Resource library management
- Intent-based action selection

Usage:
    from arklex.env.planner.react_planner import ReactPlanner
    from arklex.utils.graph_state import MessageState, LLMConfig

    # Initialize planner
    planner = ReactPlanner(tools_map, workers_map, name2id)
    planner.set_llm_config_and_build_resource_library(llm_config)

    # Execute planning
    action, state, history = planner.execute(msg_state, msg_history)
"""

import json
import traceback
import uuid
from typing import Any, Literal

from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel

from arklex.orchestrator.entities.msg_state_entities import LLMConfig, MessageState
from arklex.orchestrator.prompts import (
    PLANNER_REACT_INSTRUCTION_FEW_SHOT,
    PLANNER_REACT_INSTRUCTION_ZERO_SHOT,
    PLANNER_SUMMARIZE_TRAJECTORY_PROMPT,
    RESPOND_ACTION_NAME,
)
from arklex.utils.logging_utils import LogContext
from arklex.utils.model_provider_config import (
    PROVIDER_EMBEDDING_MODELS,
    PROVIDER_EMBEDDINGS,
)
from arklex.utils.provider_utils import validate_and_get_model_class

log_context = LogContext(__name__)

# Configuration flag for ReAct prompt type
USE_FEW_SHOT_REACT_PROMPT: bool = True

# Global constants for resource retrieval based on planning trajectory step count
MIN_NUM_RETRIEVALS: int = 3
MAX_NUM_RETRIEVALS: int = 15


def NUM_STEPS_TO_NUM_RETRIEVALS(n_steps: int) -> int:
    """Convert number of planning steps to number of resources to retrieve.

    The number of retrievals should be >= number of steps to account for cases
    where each planning step corresponds to a distinct tool/worker call and to
    increase tool selection robustness.

    Args:
        n_steps: Number of planning steps in the trajectory

    Returns:
        Number of resources to retrieve (n_steps + 3)
    """
    return n_steps + 3


class Action(BaseModel):
    """Data model for planner actions.

    Attributes:
        name: Name of the action to execute
        kwargs: Keyword arguments for the action
    """

    name: str
    kwargs: dict[str, Any]


class EnvResponse(BaseModel):
    """Data model for environment responses.

    Attributes:
        observation: Observation result from the environment
    """

    observation: Any


class PlannerResource(BaseModel):
    """Data model for planner resources (tools and workers).

    Attributes:
        name: Name of the resource
        type: Type of resource ("tool" or "worker")
        description: Description of the resource functionality
        parameters: List of parameter specifications
        required: List of required parameter names
        returns: Return value specification
    """

    name: str
    type: Literal["tool", "worker"]
    description: str
    parameters: list[dict[str, dict[str, str]]]
    required: list[str]
    returns: dict[str, Any]


# Standard respond action resource for when user requests are satisfied
RESPOND_ACTION_RESOURCE: PlannerResource = PlannerResource(
    name=RESPOND_ACTION_NAME,
    type="worker",
    description="Respond to the user if the user's request has been satisfied or if there is not enough information to do so.",
    parameters=[
        {
            "content": {
                "type": "string",
                "description": "The message to return to the user.",
            }
        }
    ],
    required=["content"],
    returns={},
)

# Default LLM configuration used on planner initialization
# This will be updated by the orchestrator with actual model configuration
DEFAULT_LLM_CONFIG: LLMConfig = LLMConfig(
    model_type_or_path="placeholder", llm_provider="placeholder"
)


class DefaultPlanner:
    """Default planner that returns unaltered MessageState on execute().

    This is a simple pass-through planner that doesn't perform any planning
    logic and just returns the input state unchanged.

    Attributes:
        tools_map: Mapping of tool names to tool instances
        workers_map: Mapping of worker names to worker instances
        name2id: Mapping of names to IDs
        all_resources_info: Information about all available resources
        llm_config: Language model configuration
    """

    description: str = (
        "Default planner that returns unaltered MessageState on execute()"
    )

    def __init__(
        self,
        tools_map: dict[str, Any],
        workers_map: dict[str, Any],
        name2id: dict[str, int],
    ) -> None:
        """Initialize the default planner.

        Args:
            tools_map: Mapping of tool names to tool instances
            workers_map: Mapping of worker names to worker instances
            name2id: Mapping of names to IDs
        """
        self.tools_map: dict[str, Any] = tools_map
        self.workers_map: dict[str, Any] = workers_map
        self.name2id: dict[str, int] = name2id
        self.all_resources_info: dict[str, Any] = {}
        self.llm_config: LLMConfig = DEFAULT_LLM_CONFIG

    def set_llm_config_and_build_resource_library(self, llm_config: LLMConfig) -> None:
        """Update planner LLM model and provider info from default.

        Note that in most cases, this must be invoked (again) after __init__(), because the LLMConfig info
        may be updated after planner is initialized, which may change the embedding model(s) used.

        The DefaultPlanner does nothing and has no need for retrieval steps, so it will not create RAG
        documents.

        Args:
            llm_config: Updated language model configuration
        """
        self.llm_config = llm_config

    def execute(
        self, msg_state: MessageState, msg_history: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], MessageState, list[dict[str, Any]]]:
        """Execute the planner (pass-through implementation).

        Args:
            msg_state: Current message state
            msg_history: History of messages

        Returns:
            Tuple containing empty action, unaltered message state, and message history
        """
        # Return empty action alongside unaltered msg_state and msg_history
        empty_action: dict[str, Any] = {
            "name": RESPOND_ACTION_NAME,
            "kwargs": {"content": ""},
        }
        return empty_action, msg_state, msg_history


class ReactPlanner(DefaultPlanner):
    """ReAct planner that chooses tools/workers based on task analysis and chat history.

    This planner implements the ReAct (Reasoning and Acting) paradigm to intelligently
    select appropriate tools and workers based on the current task and conversation
    context. It uses RAG-based resource retrieval and trajectory planning to make
    informed decisions about which actions to take.

    Attributes:
        llm: Language model instance for planning
        llm_provider: Provider of the language model
        model_name: Name of the language model
        system_role: Role identifier for system messages
        all_resources_info: Information about all available resources
        resource_rag_docs_created: Flag indicating if RAG documents have been created
    """

    description: str = "Choose tools/workers based on task and chat records if there is no specific worker/node for the user's query"

    def __init__(
        self,
        tools_map: dict[str, Any],
        workers_map: dict[str, Any],
        name2id: dict[str, int],
    ) -> None:
        """Initialize the ReAct planner.

        Args:
            tools_map: Mapping of tool names to tool instances
            workers_map: Mapping of worker names to worker instances
            name2id: Mapping of names to IDs
        """
        super().__init__(tools_map, workers_map, name2id)
        self.tools_map: dict[str, Any] = tools_map
        self.workers_map: dict[str, Any] = workers_map
        self.name2id: dict[str, int] = name2id

        # Assume default model and model provider from model_config are used
        # until model and provider info is set explicitly by orchestrator
        # with set_llm_config(llm_config: LLMConfig)
        self.llm_config: LLMConfig = DEFAULT_LLM_CONFIG

        # Set initial model and provider info
        self.llm_provider: str = self.llm_config.llm_provider
        self.model_name: str = self.llm_config.model_type_or_path
        # Initialize LLM lazily to avoid API key issues during initialization
        self.llm: Any = None
        self.system_role: str = (
            "system"  # Default to system, will be updated when LLM is initialized
        )

        # Store worker and tool info in single resources dict with standardized formatting
        formatted_worker_info: dict[str, PlannerResource] = self._format_worker_info(
            self.workers_map
        )
        formatted_tool_info: dict[str, PlannerResource] = self._format_tool_info(
            self.tools_map
        )
        # all_resources_info is Dict[str, PlannerResource]
        self.all_resources_info: dict[str, PlannerResource] = {
            **formatted_worker_info,
            **formatted_tool_info,
        }

        # Add RESPOND_ACTION to resource library
        self.all_resources_info[RESPOND_ACTION_NAME] = RESPOND_ACTION_RESOURCE

        # Track whether or not RAG documents for planner resources have already been created;
        # these will be created in AgentOrg.__init__() once model info is provided
        self.resource_rag_docs_created: bool = False

    def _initialize_llm(self) -> None:
        """Initialize the LLM if it hasn't been initialized yet."""
        if self.llm is None:
            # Create a temporary config object for validation
            temp_config = type("TempConfig", (), {"llm_provider": self.llm_provider})()
            model_class = validate_and_get_model_class(temp_config)

            self.llm = model_class(
                model=self.model_name,
                temperature=0.0,
            )
            self.system_role = "user" if self.llm_provider == "google" else "system"

    def set_llm_config_and_build_resource_library(self, llm_config: LLMConfig) -> None:
        """Update planner LLM model and provider info from default, and create RAG vector store for planner
        resource documents.

        Note that in most cases, this must be invoked (again) after __init__(), because the LLMConfig info
        may be updated after planner is initialized, which may change the embedding model(s) used.

        Args:
            llm_config: Updated language model configuration
        """
        self.llm_config = llm_config

        # Update model provider info
        self.llm_provider: str = self.llm_config.llm_provider
        self.model_name: str = self.llm_config.model_type_or_path
        # Initialize LLM with updated configuration
        self._initialize_llm()

        # Create documents containing tool/worker info
        resource_docs: list[Document] = self._create_resource_rag_docs(
            self.all_resources_info
        )

        # Init embedding model and FAISS retriever for RAG resource signature retrieval
        self.embedding_model_name: str = PROVIDER_EMBEDDING_MODELS[self.llm_provider]
        self.embedding_model: Any = PROVIDER_EMBEDDINGS.get(
            self.llm_provider, OpenAIEmbeddings
        )(
            **{"model": self.embedding_model_name}
            if self.llm_provider != "anthropic"
            else {"model_name": self.embedding_model_name}
        )
        docsearch: FAISS = FAISS.from_documents(resource_docs, self.embedding_model)
        self.retriever: Any = docsearch.as_retriever()

        # Ensure RESPOND_ACTION will always be included in list of retrieved resources if RAG is used for selecting
        # relevant tools/workers during planning (RESPOND_ACTION should always be available as a planner action, independent
        # of the expected planning trajectory)
        respond_action_doc: list[Document] = [
            d
            for d in resource_docs
            if d.metadata["resource_name"] == RESPOND_ACTION_NAME
        ]
        self.guaranteed_retrieval_docs: list[Document] = [respond_action_doc[0]]

        self.resource_rag_docs_created = True

    def _format_worker_info(
        self, workers_map: dict[str, Any]
    ) -> dict[str, PlannerResource]:
        """Convert worker information to standardized format for planner ReAct prompt.

        This method transforms the workers map into a standardized format that can be
        used by the planner to understand available workers and their capabilities.

        Args:
            workers_map: Mapping of worker names to worker configurations

        Returns:
            Dictionary mapping worker names to PlannerResource objects

        Note:
            MessageWorker is removed from the available resources to avoid conflicts
            with RESPOND_ACTION, as both return natural language responses.
        """
        formatted_worker_info: dict[str, PlannerResource] = {
            worker_name: PlannerResource(
                name=worker_name,
                type="worker",
                description=workers_map[worker_name].get("description", ""),
                parameters=[],
                required=[],
                returns={},
            )
            for worker_name in workers_map
        }

        # NOTE: MessageWorker will be removed from list of resource available to planner to avoid
        # conflicts with RESPOND_ACTION; both return a str representing model's natural language
        # response, but RESPOND_ACTION is necessary to return message to user and break planning loop
        if "MessageWorker" in formatted_worker_info:
            formatted_worker_info.pop("MessageWorker", None)

        return formatted_worker_info

    def _format_tool_info(
        self, tools_map: dict[str, Any]
    ) -> dict[str, PlannerResource]:
        """Convert tool information to standardized format for planner ReAct prompt.

        This method transforms the tools map into a standardized format that can be
        used by the planner to understand available tools and their parameters.

        Args:
            tools_map: Mapping of tool IDs to tool configurations

        Returns:
            Dictionary mapping tool names to PlannerResource objects
        """
        formatted_tools_info: dict[str, PlannerResource] = {}
        for _tool_id, tool in tools_map.items():
            # Handle both MockTool instances and regular tool dictionaries
            if hasattr(tool, "info"):
                tool_info = tool.info
                tool_outputs = tool.output
            else:
                tool_info = tool.get("info", {})
                tool_outputs = tool.get("output", [])

            if tool_info.get("type") == "function":
                tool_name: str = tool_info["function"]["name"]
                tool_description: str = tool_info["function"]["description"]

                # Get tool call parameters info and required parameters list
                parameters: list[dict[str, dict[str, str]]] = []
                _params: dict[str, dict[str, str]] = tool_info["function"][
                    "parameters"
                ]["properties"]
                for param_name in _params:
                    param: dict[str, dict[str, str]] = {
                        param_name: {
                            k: v
                            for k, v in _params[param_name].items()
                            if k != "prompt"
                        }
                    }
                    parameters.append(param)

                required: list[str] = tool_info["function"]["parameters"]["required"]

                # Get tool call return values/types
                return_values: dict[str, str] = {}
                for return_value in tool_outputs:
                    output_name: str = return_value["name"]
                    return_values[output_name] = return_value["description"]

                formatted_tools_info[tool_name] = PlannerResource(
                    name=tool_name,
                    type="tool",
                    description=tool_description,
                    parameters=parameters,
                    required=required,
                    returns=return_values,
                )

        return formatted_tools_info

    def _create_resource_rag_docs(
        self, all_resources_info: dict[str, PlannerResource]
    ) -> list[Document]:
        """Create LangChain Documents for RAG retrieval from resource information.

        Given a dictionary containing available tools and workers, this method creates
        a list of LangChain Documents (one per tool/worker) to be used for vector store
        RAG retrieval during planning.

        Args:
            all_resources_info: Dictionary mapping resource names to PlannerResource objects

        Returns:
            List of Document objects containing resource information for RAG
        """
        resource_docs: list[Document] = []

        for resource_name in all_resources_info:
            resource: PlannerResource = all_resources_info[resource_name]
            resource_type: str = resource.type
            json_signature: dict[str, Any] = resource.model_dump(mode="json")

            resource_metadata: dict[str, Any] = {
                "resource_name": resource_name,
                "type": resource_type,
                "json_signature": json_signature,
            }

            resource_page_content: str = str(json_signature)

            resource_doc: Document = Document(
                metadata=resource_metadata, page_content=resource_page_content
            )
            resource_docs.append(resource_doc)

        return resource_docs

    def _get_planning_trajectory_summary(
        self, state: MessageState, msg_history: list[dict[str, Any]]
    ) -> str:
        """Generate a natural language summary of the expected planning trajectory.

        This method invokes the language model to create a summary of the expected
        planning steps based on the current state and message history. The response
        is used as a query to retrieve relevant resource descriptions.

        Args:
            state: Current message state containing user message and task
            msg_history: History of previous messages

        Returns:
            Natural language summary of the expected planning trajectory
        """
        user_message: str = state.user_message.message
        task: str = state.orchestrator_message.attribute.get("task", "")

        # Get list of (brief) descriptions of available resources to guide planning steps summary generation
        resource_descriptions: str = ""
        for resource_name in self.all_resources_info:
            resource: PlannerResource = self.all_resources_info[resource_name]
            description: str = resource.description
            if description:
                resource_descriptions += f"\n- {description}"

        # Format planning trajectory summarization prompt with user message, task, and resource descriptions
        prompt: PromptTemplate = PromptTemplate.from_template(
            PLANNER_SUMMARIZE_TRAJECTORY_PROMPT
        )
        system_prompt: Any = prompt.invoke(
            {
                "user_message": user_message,
                "resource_descriptions": resource_descriptions,
                "task": task,
            }
        )
        log_context.info(
            f"Planner trajectory summarization system prompt: {system_prompt.text}"
        )

        # If model provider is OpenAI, messages can contain a single system message.
        # If model provider is Google, messages cannot contain system messages, and must
        # contain a single user message with system instructions.
        # If model provider is Anthropic, messages can contain a system prompt, but must also
        # contain at least one user message.
        messages: list[dict[str, Any]] = []
        if self.llm_provider.lower() in ["openai", "gemini"]:
            messages = [{"role": self.system_role, "content": system_prompt.text}]
        elif self.llm_provider.lower() == "anthropic":
            messages = [
                {"role": self.system_role, "content": system_prompt.text},
                {"role": "user", "content": user_message},
            ]
        else:
            # Default case for unknown providers
            messages = [{"role": self.system_role, "content": system_prompt.text}]

        # Initialize LLM if needed
        self._initialize_llm()
        # Invoke model to get response to ReAct instruction
        res: Any = self.llm.invoke(messages)
        message: dict[str, Any] = aimessage_to_dict(res)
        response_text: str = message["content"]

        return response_text

    def _parse_trajectory_summary_to_steps(self, summary: str) -> list[str]:
        """Parse a bulleted list summary into individual planning steps.

        Given a bulleted list representing the expected planning trajectory summary,
        this method removes the list formatting and returns a list of individual steps.

        Args:
            summary: Bulleted list string representing planning trajectory

        Returns:
            List of individual planning steps
        """
        steps: list[str] = [step.strip() for step in summary.split("- ")]
        steps = [step for step in steps if len(step) > 0]
        return steps

    def _get_num_resource_retrievals(self, summary: str) -> int:
        """Determine the number of resource signatures to retrieve based on planning trajectory.

        Given a string representing the model's summarization of the expected planning
        trajectory, this method determines the number of planning steps and uses that
        to calculate how many resource signatures to retrieve via RAG for the planner
        ReAct loop.

        Args:
            summary: String representing the planning trajectory summary

        Returns:
            Number of resource signature documents to retrieve (between MIN_NUM_RETRIEVALS
            and MAX_NUM_RETRIEVALS)
        """
        # Attempt to parse planning trajectory summary into bulleted list of steps and use
        # step count to determine num. retrievals
        valid_summary: bool = True
        try:
            steps: list[str] = self._parse_trajectory_summary_to_steps(summary)
            n_steps: int = len(steps)

            if n_steps == 0:
                valid_summary = False
            else:
                n_retrievals: int = NUM_STEPS_TO_NUM_RETRIEVALS(n_steps)

        except Exception:
            valid_summary = False

        if not valid_summary:
            log_context.info(
                f"Failed to parse planning trajectory summary into valid list of steps: '{summary}'..."
                + f" Using MIN_NUM_RETRIEVALS = {MIN_NUM_RETRIEVALS}"
            )
            n_retrievals = MIN_NUM_RETRIEVALS

        # Ensure n_retrievals is in range [MIN_NUM_RETRIEVALS, MAX_NUM_RETRIEVALS]
        n_retrievals = min(max(n_retrievals, MIN_NUM_RETRIEVALS), MAX_NUM_RETRIEVALS)

        return n_retrievals

    def _retrieve_resource_signatures(
        self,
        n_retrievals: int,
        trajectory_summary: str,
        user_message: str | None = None,
        task: str | None = None,
    ) -> list[Document]:
        """Retrieve relevant resource signature documents using RAG.

        Given the number of resource signature documents to retrieve, a summary of the
        expected planning trajectory, and optionally a user message and task description,
        this method retrieves the most relevant resource signature documents for use
        during the planning ReAct loop.

        Args:
            n_retrievals: Number of resource signature documents to retrieve
            trajectory_summary: Summary of the expected planning trajectory
            user_message: Optional user message for context
            task: Optional task description for context

        Returns:
            List of Document objects, each corresponding to a single resource/action
        """
        # Return early if no retrievals requested
        if n_retrievals <= 0:
            return []

        # Format RAG query
        query: str = ""
        if user_message:
            query += f"User Message: {user_message}\n"
        if task:
            query += f"Task: {task}\n"
        # Remove list formatting from trajectory summary (if summary is valid list)
        planning_steps: str = trajectory_summary
        steps: list[str] = self._parse_trajectory_summary_to_steps(trajectory_summary)
        if len(steps) > 0:
            planning_steps = " ".join(steps)
        query += f"Steps: {planning_steps}"

        # Retrieve relevant resource signatures
        docs_and_scores: list[tuple[Document, float]] = (
            self.retriever.vectorstore.similarity_search_with_score(
                query, k=n_retrievals
            )
        )
        signature_docs: list[Document] = [doc[0] for doc in docs_and_scores]

        # Ensure any and all resource signature docs in self.guaranteed_retrievals are included in list
        # of retrieved documents (e.g., document corresponding to RESPOND_ACTION since RESPOND_ACTION
        # should always be available)
        # NOTE: This assumes all resource names are unique identifiers!
        for doc in self.guaranteed_retrieval_docs:
            guaranteed_resource_name: str = doc.metadata["resource_name"]
            retrieved_resource_names: list[str] = [
                d.metadata["resource_name"] for d in signature_docs
            ]
            if guaranteed_resource_name not in retrieved_resource_names:
                signature_docs.append(doc)

        return signature_docs

    def _parse_response_action_to_json(self, response: str) -> dict[str, Any]:
        """Parse model response to planner ReAct instruction to extract tool/worker info as JSON.

        This method extracts the action information from the model's response to the
        ReAct instruction and attempts to parse it as JSON. If parsing fails, it
        returns a default respond action.

        Args:
            response: Raw response text from the language model

        Returns:
            Dictionary containing the parsed action information or default respond action
        """
        action_str: str = response.split("Action:\n")[-1]
        log_context.info(f"planner action_str: {action_str}")

        # Attempt to parse action as JSON object
        try:
            return json.loads(action_str)
        except json.JSONDecodeError:
            log_context.info(
                f'Failed to parse action in planner ReAct response as JSON object: "{action_str}"...'
                + " Returning response text as respond action."
            )
            return {"name": RESPOND_ACTION_NAME, "arguments": {"content": action_str}}

    def message_to_actions(
        self,
        message: dict[str, Any],
    ) -> list[Action]:
        """Convert a message to a list of Action objects.

        This method extracts the resource name and arguments from a planner action
        message and validates that the selected resource is a valid worker or tool.
        If validation fails, it returns a respond action with the message content.

        Args:
            message: Dictionary containing action information

        Returns:
            List of Action objects to be executed
        """
        # Extract resource name and arguments from planner action
        resource_name: str | None = message.get("name")
        resource_id: int | None = self.name2id.get(resource_name, None)
        arguments: dict[str, Any] | None = message.get("arguments")
        if arguments is None:
            arguments = {}

        # Ensure selected resource is a valid worker or tool
        if resource_id is not None and (
            resource_name in self.workers_map or resource_id in self.tools_map
        ):
            return [Action(name=resource_name, kwargs=arguments)]

        else:
            # Extract response message content from message["arguments"]["content"] or message["content"]
            # if former is unavailable (response is malformed) - content defaults to empty str
            args: dict[str, Any] | None = message.get("arguments")
            content: str = message.get("content", "")
            if args:
                content = args.get("content", content)

            return [Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})]

    def plan(
        self,
        state: MessageState,
        msg_history: list[dict[str, Any]],
        max_num_steps: int = 3,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Execute the ReAct planning process.

        This method implements the core ReAct planning loop. It generates a planning
        trajectory summary, retrieves relevant resources using RAG, and executes
        actions until completion or maximum steps reached.

        Args:
            state: Current message state
            msg_history: History of previous messages
            max_num_steps: Maximum number of planning steps to execute

        Returns:
            Tuple containing message history, final action name, and observation
        """
        # Invoke model to get summary of planning trajectory to determine relevant resources
        # for which to retrieve more detailed info (from RAG documents)
        trajectory_summary: str = self._get_planning_trajectory_summary(
            state, msg_history
        )
        log_context.info(
            f"planning trajectory summary response in planner:\n{trajectory_summary}"
        )
        n_retrievals: int = self._get_num_resource_retrievals(trajectory_summary)

        # Retrieve signature documents for desired number of resources using trajectory summary as RAG query
        signature_docs: list[Document] = self._retrieve_resource_signatures(
            n_retrievals, trajectory_summary
        )
        actual_n_retrievals: int = len(signature_docs)
        resource_names: list[str] = [
            doc.metadata["resource_name"] for doc in signature_docs
        ]
        log_context.info(
            f"Planner retrieved {actual_n_retrievals} signatures for the following resources (tools/workers): {resource_names}"
        )

        # Format signatures of retrieved resources to insert into ReAct instruction
        formatted_actions_str: str = "None"
        if len(signature_docs) > 0:
            retrieved_actions: dict[str, dict[str, Any]] = {
                doc.metadata["resource_name"]: doc.metadata["json_signature"]
                for doc in signature_docs
            }
            formatted_actions_str = str(retrieved_actions)

        user_message: str = state.user_message.message
        task: str = state.orchestrator_message.attribute.get("task", "")

        # Format planner ReAct system prompt
        if USE_FEW_SHOT_REACT_PROMPT:
            prompt: PromptTemplate = PromptTemplate.from_template(
                PLANNER_REACT_INSTRUCTION_FEW_SHOT
            )
        else:
            prompt = PromptTemplate.from_template(PLANNER_REACT_INSTRUCTION_ZERO_SHOT)

        input_prompt: Any = prompt.invoke(
            {
                "user_message": user_message,
                "available_actions": formatted_actions_str,
                "respond_action_name": RESPOND_ACTION_NAME,
                "task": task,
            }
        )

        messages: list[dict[str, Any]] = [
            {"role": self.system_role, "content": input_prompt.text}
        ]
        messages.extend(msg_history)

        for _ in range(max_num_steps):
            # Initialize LLM if needed
            self._initialize_llm()
            # Invoke model to get response to ReAct instruction
            res: Any = self.llm.invoke(messages)
            message: dict[str, Any] = aimessage_to_dict(res)
            response_text: str = message["content"]
            log_context.info(f"response text in planner: {response_text}")
            json_response: dict[str, Any] = self._parse_response_action_to_json(
                response_text
            )
            log_context.info(f"JSON response in planner: {json_response}")
            actions: list[Action] = self.message_to_actions(json_response)

            log_context.info(f"actions in planner: {actions}")

            # Execute actions
            for action in actions:
                env_response: EnvResponse = self.step(action, state)

                # Exit loop if planner action is RESPOND_ACTION
                if action.name == RESPOND_ACTION_NAME:
                    return msg_history, action.name, env_response.observation

                else:
                    # Mimic tool call(s) in msg_history in absence of tools input parameter
                    call_id: str = str(uuid.uuid4())
                    assistant_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": response_text,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": action.name,
                                    "arguments": json.dumps(action.kwargs),
                                },
                                "id": call_id,
                                "type": "function",
                            }
                        ],
                        "function_call": None,
                    }
                    resource_response: dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": action.name,
                        "content": env_response.observation,
                    }
                    new_messages: list[dict[str, Any]] = [
                        assistant_message,
                        resource_response,
                    ]
                    messages.extend(new_messages)
                    msg_history.extend(new_messages)

        # If we reach here, we've exhausted max_num_steps without finding RESPOND_ACTION
        # Return the last action and response
        return msg_history, action.name, env_response.observation

    def step(self, action: Action, msg_state: MessageState) -> EnvResponse:
        """Execute a single action step.

        This method executes a single action by calling the appropriate tool or worker.
        It handles different types of actions: respond actions, tool calls, and worker
        executions. Error handling is included for each action type.

        Args:
            action: Action object containing the action to execute
            msg_state: Current message state

        Returns:
            EnvResponse containing the observation from the action execution
        """
        if action.name == RESPOND_ACTION_NAME:
            response: str = action.kwargs["content"]
            observation: str = response

        # tools_map indexed by tool ID
        elif self.name2id.get(action.name, None) in self.tools_map:
            try:
                resource_id: int = self.name2id[action.name]
                calling_tool: dict[str, Any] = self.tools_map[resource_id]
                kwargs: dict[str, Any] = action.kwargs
                combined_kwargs: dict[str, Any] = {
                    **kwargs,
                    **calling_tool["fixed_args"],
                }
                observation: Any = calling_tool["execute"]().func(**combined_kwargs)
                log_context.info(
                    f"planner calling tool {action.name} with kwargs {combined_kwargs}"
                )
                observation: str = str(observation)
                log_context.info(f"tool call response: {str(observation)}")

            except Exception as e:
                log_context.error(traceback.format_exc())
                observation = f"Error: {e}"

        # workers_map indexed by worker name
        elif action.name in self.workers_map:
            try:
                observation: str = str(
                    self.workers_map[action.name]["execute"]().execute(msg_state)
                )
            except Exception as e:
                log_context.error(traceback.format_exc())
                observation = f"Error: {e}"

        else:
            observation = f"Unknown action {action.name}"

        return EnvResponse(observation=observation)

    def execute(
        self, msg_state: MessageState, msg_history: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], MessageState, list[dict[str, Any]]]:
        """Execute the planner with the given message state and history.

        This method is the main entry point for the planner. It calls the plan method
        to execute the ReAct planning process and updates the message state with the
        final response.

        Args:
            msg_state: Current message state
            msg_history: History of previous messages

        Returns:
            Tuple containing the final action, updated message state, and message history
        """
        msg_history, action, response = self.plan(msg_state, msg_history)
        # msg_state["response"] = response
        msg_state.response = response
        return action, msg_state, msg_history


def aimessage_to_dict(ai_message: AIMessage | dict[str, Any]) -> dict[str, Any]:
    """Convert an AIMessage or dictionary to a standardized dictionary format.

    This utility function converts either an AIMessage object or a dictionary
    to a standardized dictionary format for consistent message handling.

    Args:
        ai_message: Either an AIMessage object or a dictionary containing message data

    Returns:
        Dictionary with standardized message format containing content, role,
        function_call, and tool_calls fields
    """
    if isinstance(ai_message, dict):
        message_dict: dict[str, Any] = {
            "content": ai_message.get("content", ""),
            "role": ai_message.get("role", "user"),
            "function_call": ai_message.get("function_call", None),
            "tool_calls": ai_message.get("tool_calls", None),
        }
    else:
        message_dict: dict[str, Any] = {
            "content": ai_message.content,
            "role": "assistant",
            "function_call": None,
            "tool_calls": None,
        }
    return message_dict
