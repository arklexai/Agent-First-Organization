from typing import Any

from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, StateGraph

from arklex.env.prompts import load_prompts
from arklex.env.tools.database.utils import DatabaseActions
from arklex.env.tools.utils import ToolGenerator
from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.orchestrator.entities.msg_state_entities import MessageState
from arklex.utils.logging_utils import LogContext
from arklex.utils.model_config import MODEL
from arklex.utils.utils import chunk_string

log_context = LogContext(__name__)


@register_worker
class DataBaseWorker(BaseWorker):
    description: str = "Help the user with actions related to customer support like a booking system with structured data, always involving search, insert, update, and delete operations."

    def __init__(self, model_config: dict[str, Any] | None = None) -> None:
        """Initialize the database worker.

        Args:
            model_config: Model configuration dictionary. If None, will use default model.
        """
        if model_config is None:
            # Use default model configuration if none provided
            from arklex.utils.model_config import MODEL

            model_config = MODEL

        # Store the model_config for access by tests
        self.model_config = model_config

        # Initialize LLM with provided configuration
        from arklex.utils.model_provider_config import PROVIDER_MAP

        provider = model_config.get("llm_provider")
        model_name = model_config.get("model_type_or_path")

        if not provider:
            raise ValueError(
                "llm_provider must be explicitly specified in model_config"
            )
        if not model_name:
            raise ValueError("Model name must be specified in model_config")

        model_class = PROVIDER_MAP.get(provider)
        if not model_class:
            raise ValueError(f"Unsupported provider: {provider}")

        if provider == "google":
            # Google models use google_api_key parameter
            self.llm: BaseChatModel = model_class(
                model=model_name,
                google_api_key=model_config.get("api_key"),
                timeout=30000,
            )
        else:
            # Other providers use api_key parameter
            self.llm: BaseChatModel = model_class(
                model=model_name, api_key=model_config.get("api_key"), timeout=30000
            )

        self.actions: dict[str, str] = {
            "SearchShow": "Search for shows",
            "BookShow": "Book a show",
            "CheckBooking": "Check details of booked show(s)",
            "CancelBooking": "Cancel a booking",
            "Others": "Other actions not mentioned above",
        }
        self.DBActions: DatabaseActions = DatabaseActions()
        self.action_graph: StateGraph = self._create_action_graph()

    def search_show(self, state: MessageState) -> MessageState:
        return self.DBActions.search_show(state)

    def book_show(self, state: MessageState) -> MessageState:
        return self.DBActions.book_show(state)

    def check_booking(self, state: MessageState) -> MessageState:
        return self.DBActions.check_booking(state)

    def cancel_booking(self, state: MessageState) -> MessageState:
        return self.DBActions.cancel_booking(state)

    def verify_action(self, msg_state: MessageState) -> str:
        user_intent: str = msg_state.orchestrator_message.attribute.get("task", "")
        actions_info: str = "\n".join(
            [f"{name}: {description}" for name, description in self.actions.items()]
        )
        actions_name: str = ", ".join(self.actions.keys())

        prompts: dict[str, str] = load_prompts(msg_state.bot_config)
        prompt: PromptTemplate = PromptTemplate.from_template(
            prompts["database_action_prompt"]
        )
        input_prompt = prompt.invoke(
            {
                "user_intent": user_intent,
                "actions_info": actions_info,
                "actions_name": actions_name,
            }
        )
        chunked_prompt: str = chunk_string(
            input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"]
        )
        log_context.info(
            f"Chunked prompt for deciding choosing DB action: {chunked_prompt}"
        )
        final_chain = self.llm | StrOutputParser()
        try:
            answer: str = final_chain.invoke(chunked_prompt)
            for action_name in self.actions:
                if action_name in answer:
                    log_context.info(
                        f"Chosen action in the database worker: {action_name}"
                    )
                    return action_name
            log_context.info("Base action chosen in the database worker: Others")
            return "Others"
        except Exception as e:
            log_context.error(
                f"Error occurred while choosing action in the database worker: {e}"
            )
            return "Others"

    def _create_action_graph(self) -> StateGraph:
        workflow: StateGraph = StateGraph(MessageState)
        # Add nodes for each worker
        workflow.add_node("SearchShow", self.search_show)
        workflow.add_node("BookShow", self.book_show)
        workflow.add_node("CheckBooking", self.check_booking)
        workflow.add_node("CancelBooking", self.cancel_booking)
        workflow.add_node("Others", ToolGenerator.generate)
        workflow.add_node("tool_generator", ToolGenerator.context_generate)
        workflow.add_conditional_edges(START, self.verify_action)
        workflow.add_edge("SearchShow", "tool_generator")
        workflow.add_edge("BookShow", "tool_generator")
        workflow.add_edge("CheckBooking", "tool_generator")
        workflow.add_edge("CancelBooking", "tool_generator")
        return workflow

    def _execute(self, msg_state: MessageState) -> MessageState:
        self.DBActions.log_in()
        msg_state.slots = self.DBActions.init_slots(
            msg_state.slots, msg_state.bot_config
        )
        graph = self.action_graph.compile()
        result: MessageState = graph.invoke(msg_state)
        return result
