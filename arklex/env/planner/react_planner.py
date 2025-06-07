import logging
import json
from typing import Any, Dict, List, Optional, Literal, Tuple
from pydantic import BaseModel
import traceback
import uuid

from langchain.schema import AIMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings

from arklex.utils.graph_state import MessageState, LLMConfig
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import (
    PROVIDER_MAP,
    PROVIDER_EMBEDDINGS,
    PROVIDER_EMBEDDING_MODELS,
)
from arklex.orchestrator.prompts import (
    RESPOND_ACTION_NAME,
    PLANNER_REACT_INSTRUCTION_ZERO_SHOT,
    PLANNER_REACT_INSTRUCTION_FEW_SHOT,
    PLANNER_SUMMARIZE_TRAJECTORY_PROMPT,
)


logger: logging.Logger = logging.getLogger(__name__)

# If False, use shorter version of planner ReAct instruction without few-shot example(s)
USE_FEW_SHOT_REACT_PROMPT: bool = True

# Globals determining number of resources to retrieve based on step count of model
# planning trajectory summary
MIN_NUM_RETRIEVALS: int = 3
MAX_NUM_RETRIEVALS: int = 15


# Determines how to translate trajectory step count into number of relevant resources to retrieve;
# note that num. retrievals should be >= num. steps to account for cases where each planning step
# corresponds to a distinct tool/worker call and to increase tool selection robustness
def NUM_STEPS_TO_NUM_RETRIEVALS(n_steps: int) -> int:
    return n_steps + 3


class Action(BaseModel):
    name: str
    kwargs: Dict[str, Any]


class EnvResponse(BaseModel):
    observation: Any


class PlannerResource(BaseModel):
    name: str
    type: Literal["tool", "worker"]
    description: str
    parameters: List[Dict[str, Dict[str, str]]]
    required: List[str]
    returns: Dict[str, Any]


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

logger: logging.Logger = logging.getLogger(__name__)

# Default LLM Config used on planner initialization, overwritten by
# updated llm config info with planner.set_llm_config (invoked in
# AgentOrg init)
DEFAULT_LLM_CONFIG: LLMConfig = LLMConfig(
    model_type_or_path=MODEL["model_type_or_path"], llm_provider=MODEL["llm_provider"]
)


class DefaultPlanner:
    description: str = (
        "Default planner that returns unaltered MessageState on execute()"
    )

    def __init__(
        self,
        tools_map: Dict[str, Any],
        workers_map: Dict[str, Any],
        name2id: Dict[str, int],
    ) -> None:
        self.tools_map: Dict[str, Any] = tools_map
        self.workers_map: Dict[str, Any] = workers_map
        self.name2id: Dict[str, int] = name2id
        self.all_resources_info: Dict[str, Any] = {}
        self.llm_config: LLMConfig = DEFAULT_LLM_CONFIG

    def set_llm_config_and_build_resource_library(self, llm_config: LLMConfig) -> None:
        """
        Update planner LLM model and provider info from default.

        Note that in most cases, this must be invoked (again) after __init__(), because the LLMConfig info
        may be updated after planner is initialized, which may change the embedding model(s) used.

        The DefaultPlanner does nothing and has no need for retrieval steps, so it will not create RAG
        documents.
        """
        self.llm_config = llm_config

    def execute(
        self, msg_state: MessageState, msg_history: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], MessageState, List[Dict[str, Any]]]:
        # Return empty action alongside unaltered msg_state and msg_history
        empty_action: Dict[str, Any] = {
            "name": RESPOND_ACTION_NAME,
            "kwargs": {"content": ""},
        }
        return empty_action, msg_state, msg_history


class ReactPlanner(DefaultPlanner):
    description: str = "Choose tools/workers based on task and chat records if there is no specific worker/node for the user's query"

    def __init__(
        self,
        tools_map: Dict[str, Any],
        workers_map: Dict[str, Any],
        name2id: Dict[str, int],
    ) -> None:
        super().__init__(tools_map, workers_map, name2id)
        self.tools_map: Dict[str, Any] = tools_map
        self.workers_map: Dict[str, Any] = workers_map
        self.name2id: Dict[str, int] = name2id

        # Assume default model and model provider from model_config are used
        # until model and provider info is set explicitly by orchestrator
        # with set_llm_config(llm_config: LLMConfig)
        self.llm_config: LLMConfig = DEFAULT_LLM_CONFIG

        # Set initial model and provider info
        self.llm_provider: str = self.llm_config.llm_provider
        self.model_name: str = self.llm_config.model_type_or_path
        self.llm: Any = PROVIDER_MAP.get(self.llm_provider, ChatOpenAI)(
            model=self.model_name,
            temperature=0.0,
        )
        self.system_role: str = "user" if self.llm_provider == "gemini" else "system"

        # Store worker and tool info in single resources dict with standardized formatting
        formatted_worker_info: Dict[str, PlannerResource] = self._format_worker_info(
            self.workers_map
        )
        formatted_tool_info: Dict[str, PlannerResource] = self._format_tool_info(
            self.tools_map
        )
        # all_resources_info is Dict[str, PlannerResource]
        self.all_resources_info: Dict[str, PlannerResource] = {
            **formatted_worker_info,
            **formatted_tool_info,
        }

        # Add RESPOND_ACTION to resource library
        self.all_resources_info[RESPOND_ACTION_NAME] = RESPOND_ACTION_RESOURCE

        # Track whether or not RAG documents for planner resources have already been created;
        # these will be created in AgentOrg.__init__() once model info is provided
        self.resource_rag_docs_created: bool = False

    def set_llm_config_and_build_resource_library(self, llm_config: LLMConfig) -> None:
        """
        Update planner LLM model and provider info from default, and create RAG vector store for planner
        resource documents.

        Note that in most cases, this must be invoked (again) after __init__(), because the LLMConfig info
        may be updated after planner is initialized, which may change the embedding model(s) used.
        """
        self.llm_config = llm_config

        # Update model provider info
        self.llm_provider: str = self.llm_config.llm_provider
        self.model_name: str = self.llm_config.model_type_or_path
        self.llm: Any = PROVIDER_MAP.get(self.llm_provider, ChatOpenAI)(
            model=self.model_name,
            temperature=0.0,
        )
        self.system_role: str = "user" if self.llm_provider == "gemini" else "system"

        # Create documents containing tool/worker info
        resource_docs: List[Document] = self._create_resource_rag_docs(
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
        respond_action_doc: List[Document] = [
            d
            for d in resource_docs
            if d.metadata["resource_name"] == RESPOND_ACTION_NAME
        ]
        self.guaranteed_retrieval_docs: List[Document] = [respond_action_doc[0]]

        self.resource_rag_docs_created = True

    def _format_worker_info(
        self, workers_map: Dict[str, Any]
    ) -> Dict[str, PlannerResource]:
        """
        Convert info on available workers to standardized format for planner ReAct prompt.
        """
        formatted_worker_info: Dict[str, PlannerResource] = {
            worker_name: PlannerResource(
                name=worker_name,
                type="worker",
                description=workers_map[worker_name].get("description", ""),
                parameters=[],
                required=[],
                returns={},
            )
            for worker_name in workers_map.keys()
        }

        # NOTE: MessageWorker will be removed from list of resource available to planner to avoid
        # conflicts with RESPOND_ACTION; both return a str representing model's natural language
        # response, but RESPOND_ACTION is necessary to return message to user and break planning loop
        if "MessageWorker" in formatted_worker_info:
            formatted_worker_info.pop("MessageWorker", None)

        return formatted_worker_info

    def _format_tool_info(
        self, tools_map: Dict[str, Any]
    ) -> Dict[str, PlannerResource]:
        """
        Convert info on available tools to standardized format for planner ReAct prompt.
        """
        formatted_tools_info: Dict[str, PlannerResource] = {}
        for tool_id, tool in tools_map.items():
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
                parameters: List[Dict[str, Dict[str, str]]] = []
                _params: Dict[str, Dict[str, str]] = tool_info["function"][
                    "parameters"
                ]["properties"]
                for param_name in _params.keys():
                    param: Dict[str, Dict[str, str]] = {
                        param_name: {
                            k: v
                            for k, v in _params[param_name].items()
                            if k != "prompt"
                        }
                    }
                    parameters.append(param)

                required: List[str] = tool_info["function"]["parameters"]["required"]

                # Get tool call return values/types
                return_values: Dict[str, str] = {}
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
        self, all_resources_info: Dict[str, PlannerResource]
    ) -> List[Document]:
        """
        Given dict all_resources_info containing available tools and workers, return list of LangChain Documents
        containing resource info (one tool/worker per document) to save as vector store for RAG retrieval.
        """
        resource_docs: List[Document] = []

        for resource_name in all_resources_info:
            resource: PlannerResource = all_resources_info[resource_name]
            resource_type: str = resource.type
            json_signature: Dict[str, Any] = resource.model_dump(mode="json")

            resource_metadata: Dict[str, Any] = {
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
        self, state: MessageState, msg_history: List[Dict[str, Any]]
    ) -> str:
        """
        Invoke model to get natural language summary of expected planning trajectory.

        Response will be used as query to retrieve more detailed descriptions of relevant resources
        (available tools/workers).
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
        logger.info(
            f"Planner trajectory summarization system prompt: {system_prompt.text}"
        )

        # If model provider is OpenAI, messages can contain a single system message.
        # If model provider is Google, messages cannot contain system messages, and must
        # contain a single user message with system instructions.
        # If model provider is Anthropic, messages can contain a system prompt, but must also
        # contain at least one user message.
        messages: List[Dict[str, Any]]
        if self.llm_provider.lower() in ["openai", "gemini"]:
            messages = [{"role": self.system_role, "content": system_prompt.text}]
        elif self.llm_provider.lower() == "anthropic":
            messages = [
                {"role": self.system_role, "content": system_prompt.text},
                {"role": "user", "content": user_message},
            ]

        # Invoke model to get response to ReAct instruction
        res: Any = self.llm.invoke(messages)
        message: Dict[str, Any] = aimessage_to_dict(res)
        response_text: str = message["content"]

        return response_text

    def _parse_trajectory_summary_to_steps(self, summary: str) -> List[str]:
        """
        Given bulleted list representing expected planning trajectory summary, remove list
        formatting and return list of steps.
        """
        steps: List[str] = [step.strip() for step in summary.split("- ")]
        steps = [step for step in steps if len(step) > 0]
        return steps

    def _get_num_resource_retrievals(self, summary: str) -> int:
        """
        Given a str representing model summarization of expected planning trajectory,
        determine number of planning trajectory steps and use step count to determine
        number of resource signatures to retrieve (via RAG) for planner ReAct loop.

        Return value (number of resource signature docs to retrieve) will be in the range
        [MIN_NUM_RETRIEVALS, MAX_NUM_RETRIEVALS].
        """
        # Attempt to parse planning trajectoy summary into bulleted list of steps and use
        # step count to determine num. retrievals
        valid_summary: bool = True
        try:
            steps: List[str] = self._parse_trajectory_summary_to_steps(summary)
            n_steps: int = len(steps)

            if n_steps == 0:
                valid_summary = False
            else:
                n_retrievals: int = NUM_STEPS_TO_NUM_RETRIEVALS(n_steps)

        except Exception as e:
            valid_summary = False

        if not valid_summary:
            logger.info(
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
        user_message: Optional[str] = None,
        task: Optional[str] = None,
    ) -> List[Document]:
        """
        Given an int representing number of resource signature docs to retrieve, a summary of the expected
        planning trajectory, and optionally a user message/query and a task description, retrieve the desired
        number of resource signature docs most relevant to the planning trajectory to use with RAG during
        planning ReAct loop.

        Returns:
            A list of Documents, each corresponding to a single resource/action (tool or worker to be called).
        """
        # Format RAG query
        query: str = ""
        if user_message:
            query += f"User Message: {user_message}\n"
        if task:
            query += f"Task: {task}\n"
        # Remove list formatting from trajectory summary (if summary is valid list)
        planning_steps: str = trajectory_summary
        steps: List[str] = self._parse_trajectory_summary_to_steps(trajectory_summary)
        if len(steps) > 0:
            planning_steps = " ".join(steps)
        query += f"Steps: {planning_steps}"

        # Retrieve relevant resource signatures
        docs_and_scores: List[Tuple[Document, float]] = (
            self.retriever.vectorstore.similarity_search_with_score(
                query, k=n_retrievals
            )
        )
        signature_docs: List[Document] = [doc[0] for doc in docs_and_scores]

        # Ensure any and all resource signature docs in self.guaranteed_retrievals are included in list
        # of retrieved documents (e.g., document corresponding to RESPOND_ACTION since RESPOND_ACTION
        # should always be available)
        # NOTE: This assumes all resource names are unique identifiers!
        for doc in self.guaranteed_retrieval_docs:
            guaranteed_resource_name: str = doc.metadata["resource_name"]
            retrieved_resource_names: List[str] = [
                d.metadata["resource_name"] for d in signature_docs
            ]
            if guaranteed_resource_name not in retrieved_resource_names:
                signature_docs.append(doc)

        return signature_docs

    def _parse_response_action_to_json(self, response: str) -> Dict[str, Any]:
        """
        Parse model response to planner ReAct instruction to extract tool/worker info as JSON.
        """
        action_str: str = response.split("Action:\n")[-1]
        logger.info(f"planner action_str: {action_str}")

        # Attempt to parse action as JSON object
        try:
            return json.loads(action_str)
        except json.JSONDecodeError as e:
            logger.info(
                f'Failed to parse action in planner ReAct response as JSON object: "{action_str}"...'
                + " Returning response text as respond action."
            )
            return {"name": RESPOND_ACTION_NAME, "arguments": {"content": action_str}}

    def message_to_actions(
        self,
        message: Dict[str, Any],
    ) -> List[Action]:
        # Extract resource name and arguments from planner action
        resource_name: Optional[str] = message.get("name", None)
        action_args: Optional[Dict[str, Any]] = message.get("arguments", None)
        resource_id: Optional[int] = self.name2id.get(resource_name, None)
        arguments: Optional[Dict[str, Any]] = message.get("arguments", None)

        # Ensure selected resource is a valid worker or tool
        if resource_id is not None and (
            resource_name in self.workers_map.keys()
            or resource_id in self.tools_map.keys()
        ):
            return [Action(name=resource_name, kwargs=arguments)]

        else:
            # Extract response message content from message["arguments"]["content"] or message["content"]
            # if former is unavailable (response is malformed) - content defaults to empty str
            args: Optional[Dict[str, Any]] = message.get("arguments", None)
            content: str = message.get("content", "")
            if args:
                content = args.get("content", content)

            return [Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})]

    def plan(
        self,
        state: MessageState,
        msg_history: List[Dict[str, Any]],
        max_num_steps: int = 3,
    ) -> Tuple[List[Dict[str, Any]], str, Any]:
        # Invoke model to get summary of planning trajectory to determine relevant resources
        # for which to retrieve more detailed info (from RAG documents)
        trajectory_summary: str = self._get_planning_trajectory_summary(
            state, msg_history
        )
        logger.info(
            f"planning trajectory summary response in planner:\n{trajectory_summary}"
        )
        n_retrievals: int = self._get_num_resource_retrievals(trajectory_summary)

        # Retrieve signature documents for desired number of resources using trajectory summary as RAG query
        signature_docs: List[Document] = self._retrieve_resource_signatures(
            n_retrievals, trajectory_summary
        )
        actual_n_retrievals: int = len(signature_docs)
        resource_names: List[str] = [
            doc.metadata["resource_name"] for doc in signature_docs
        ]
        logger.info(
            f"Planner retrieved {actual_n_retrievals} signatures for the following resources (tools/workers): {resource_names}"
        )

        # Format signatures of retrieved resources to insert into ReAct instruction
        formatted_actions_str: str = "None"
        if len(signature_docs) > 0:
            retrieved_actions: Dict[str, Dict[str, Any]] = {
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

        messages: List[Dict[str, Any]] = [
            {"role": self.system_role, "content": input_prompt.text}
        ]
        messages.extend(msg_history)

        for _ in range(max_num_steps):
            # Invoke model to get response to ReAct instruction
            res: Any = self.llm.invoke(messages)
            message: Dict[str, Any] = aimessage_to_dict(res)
            response_text: str = message["content"]
            logger.info(f"response text in planner: {response_text}")
            json_response: Dict[str, Any] = self._parse_response_action_to_json(
                response_text
            )
            logger.info(f"JSON response in planner: {json_response}")
            actions: List[Action] = self.message_to_actions(json_response)

            logger.info(f"actions in planner: {actions}")

            # Execute actions
            for action in actions:
                env_response: EnvResponse = self.step(action, state)

                # Exit loop if planner action is RESPOND_ACTION
                if action.name == RESPOND_ACTION_NAME:
                    return msg_history, action.name, env_response.observation

                else:
                    # Mimic tool call(s) in msg_history in absence of tools input parameter
                    call_id: str = str(uuid.uuid4())
                    assistant_message: Dict[str, Any] = {
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
                    resource_response: Dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": action.name,
                        "content": env_response.observation,
                    }
                    new_messages: List[Dict[str, Any]] = [
                        assistant_message,
                        resource_response,
                    ]
                    messages.extend(new_messages)
                    msg_history.extend(new_messages)

        return msg_history, action.name, env_response.observation

    def step(self, action: Action, msg_state: MessageState) -> EnvResponse:
        if action.name == RESPOND_ACTION_NAME:
            response: str = action.kwargs["content"]
            observation: str = response

        # tools_map indexed by tool ID
        elif self.name2id.get(action.name, None) in self.tools_map:
            try:
                resource_id: int = self.name2id[action.name]
                calling_tool: Dict[str, Any] = self.tools_map[resource_id]
                kwargs: Dict[str, Any] = action.kwargs
                combined_kwargs: Dict[str, Any] = {
                    **kwargs,
                    **calling_tool["fixed_args"],
                }
                observation: Any = calling_tool["execute"]().func(**combined_kwargs)
                logger.info(
                    f"planner calling tool {action.name} with kwargs {combined_kwargs}"
                )
                if not isinstance(observation, str):
                    # Convert to string if not already
                    observation = str(observation)
                logger.info(f"tool call response: {str(observation)}")

            except Exception as e:
                logger.error(traceback.format_exc())
                observation = f"Error: {e}"

        # workers_map indexed by worker name
        elif action.name in self.workers_map:
            try:
                worker: Any = self.workers_map[action.name]["execute"]()
                observation: Any = worker.execute(msg_state)
                if not isinstance(observation, str):
                    # Convert to string if not already
                    observation = str(observation)

            except Exception as e:
                logger.error(traceback.format_exc())
                observation = f"Error: {e}"

        else:
            observation = f"Unknown action {action.name}"

        return EnvResponse(observation=observation)

    def execute(
        self, msg_state: MessageState, msg_history: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], MessageState, List[Dict[str, Any]]]:
        msg_history, action, response = self.plan(msg_state, msg_history)
        # msg_state["response"] = response
        msg_state.response = response
        return action, msg_state, msg_history


def aimessage_to_dict(ai_message: Any) -> Dict[str, Any]:
    message_dict: Dict[str, Any] = {
        "content": ai_message.content,
        "role": "assistant" if isinstance(ai_message, AIMessage) else "user",
        "function_call": None,
        "tool_calls": None,
    }
    return message_dict
