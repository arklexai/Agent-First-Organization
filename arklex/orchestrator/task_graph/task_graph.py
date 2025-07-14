"""Task graph management and execution for the Arklex framework.

This module provides functionality for managing and executing task graphs, including
node management, state tracking, and integration with intent detection and slot filling systems.

Key Components:
- TaskGraphBase: Base class for task graph functionality
- TaskGraph: Main class for managing conversation flow and intent handling
- NodeInfo: Information about graph nodes
- Params: Parameters for graph traversal
- PathNode: Node information for path tracking

Features:
- Directed graph structure for task flow
- Intent-based node navigation
- Global and local intent handling
- Node state management
- Path tracking and history
- Resource management
- Slot filling integration
- Intent detection integration

Usage:
    from arklex.orchestrator.task_graph import TaskGraph
    from arklex.utils.graph_state import LLMConfig, Params

    # Initialize task graph
    config = {
        "nodes": [...],
        "edges": [...],
        "intent_api": {...},
        "slotfillapi": {...}
    }

    llm_config = LLMConfig(...)
    task_graph = TaskGraph("conversation", config, llm_config)

    # Process input
    params = Params(...)
    node_info, updated_params = task_graph.get_node({"input": "user message"})
"""

import collections
import copy
from typing import Any

import networkx as nx
import numpy as np

from arklex.env.nested_graph.nested_graph import NestedGraph
from arklex.orchestrator.entities.msg_state_entities import LLMConfig, StatusEnum
from arklex.orchestrator.entities.orchestrator_params_entities import OrchestratorParams
from arklex.orchestrator.entities.taskgraph_entities import NodeInfo, PathNode
from arklex.orchestrator.NLU.core.intent import IntentDetector
from arklex.orchestrator.NLU.core.slot import SlotFiller
from arklex.orchestrator.NLU.services.model_service import (
    DummyModelService,
    ModelService,
)
from arklex.utils.exceptions import TaskGraphError
from arklex.utils.logging_utils import LogContext
from arklex.utils.utils import normalize, str_similarity

log_context = LogContext(__name__)


class TaskGraphBase:
    """Base class for task graph functionality.

    This class provides the fundamental structure and methods for managing task graphs.
    It handles graph creation, intent management, and node traversal.

    Attributes:
        graph (nx.DiGraph): The directed graph representing the task flow
        product_kwargs (Dict[str, Any]): Configuration settings for the graph
        intents (DefaultDict[str, List[Dict[str, Any]]]): Global intents for node navigation
        start_node (Optional[str]): The initial node in the graph

    Methods:
        create_graph(): Creates the graph structure
        get_pred_intents(): Gets predicted intents from graph edges
        get_start_node(): Gets the starting node of the graph
    """

    def __init__(self, name: str, product_kwargs: dict[str, Any]) -> None:
        self.graph: nx.DiGraph = nx.DiGraph(name=name)
        self.product_kwargs: dict[str, Any] = product_kwargs
        self.create_graph()
        self.intents: collections.defaultdict[str, list[dict[str, Any]]] = (
            self.get_pred_intents()
        )  # global intents
        self.start_node: str | None = self.get_start_node()

    def create_graph(self) -> None:
        raise NotImplementedError

    def get_pred_intents(self) -> collections.defaultdict[str, list[dict[str, Any]]]:
        intents: collections.defaultdict[str, list[dict[str, Any]]] = (
            collections.defaultdict(list)
        )
        for edge in self.graph.edges.data():
            if edge[2].get("attribute", {}).get("pred", False):
                edge_info: dict[str, Any] = copy.deepcopy(edge[2])
                edge_info["source_node"] = edge[0]
                edge_info["target_node"] = edge[1]
                intents[edge[2].get("intent")].append(edge_info)
        return intents

    def get_start_node(self) -> str | None:
        for node in self.graph.nodes.data():
            if node[1].get("type", "") == "start" or node[1].get("attribute", {}).get(
                "start", False
            ):
                return node[0]
        return None


class TaskGraph(TaskGraphBase):
    """Task graph management and execution.

    This class manages the execution of task graphs, including node management,
    state tracking, and integration with intent detection and slot filling systems.

    Attributes:
        unsure_intent (Dict[str, Any]): Default intent for unknown inputs
        initial_node (Optional[str]): Initial node for conversation flow
        llm_config (LLMConfig): Configuration for language model
        intent_detector (IntentDetector): Intent detection API
        slotfillapi (SlotFiller): Slot filling API

    Methods:
        create_graph(): Creates the conversation graph
        get_initial_flow(): Gets the initial flow for conversation
        jump_to_node(): Jumps to a node based on intent
        get_current_node(): Gets the current node in conversation
        get_available_global_intents(): Gets available global intents
        update_node_limit(): Updates node visit limits
        get_local_intent(): Gets local intents for current node
        handle_multi_step_node(): Handles multi-step node processing
        handle_incomplete_node(): Handles incomplete node processing
        global_intent_prediction(): Predicts global intents
        local_intent_prediction(): Predicts local intents
        handle_unknown_intent(): Handles unknown intents
        handle_leaf_node(): Handles leaf node processing
        get_node(): Gets next node based on input
        postprocess_node(): Post-processes node information
    """

    def __init__(
        self,
        name: str,
        product_kwargs: dict[str, Any],
        llm_config: LLMConfig,
        slotfillapi: str = "",
        model_service: ModelService | None = None,
    ) -> None:
        """Initialize the task graph.

        Args:
            name: Name of the task graph
            product_kwargs: Configuration settings for the graph
            llm_config: Configuration for language model
            slotfillapi: API endpoint for slot filling
            model_service: Model service for intent detection (required)
        """
        super().__init__(name, product_kwargs)
        self.unsure_intent: dict[str, Any] = {
            "intent": "others",
            "source_node": None,
            "target_node": None,
            "attribute": {
                "weight": 1,
                "pred": False,
                "definition": "",
                "sample_utterances": [],
            },
        }
        self.initial_node: str | None = self.get_initial_flow()
        self.llm_config: LLMConfig = llm_config
        if model_service is None:
            raise ValueError(
                "model_service is required for TaskGraph and cannot be None."
            )
        self.intent_detector: IntentDetector = IntentDetector(model_service)
        # Ensure slotfillapi is a valid model service for SlotFiller
        if isinstance(slotfillapi, str) or not slotfillapi:
            dummy_config = {
                "model_name": "dummy",
                "api_key": "dummy",
                "endpoint": "http://dummy",
                "model_type_or_path": "dummy-path",
                "llm_provider": "dummy",
            }
            self.slotfillapi: SlotFiller = SlotFiller(DummyModelService(dummy_config))
        else:
            self.slotfillapi: SlotFiller = SlotFiller(slotfillapi)

    def get_initial_flow(self) -> str | None:
        services_nodes: dict[str, str] | None = self.product_kwargs.get(
            "services_nodes", None
        )
        node: str | None = None
        if services_nodes:
            candidates_nodes: list[str] = [v for k, v in services_nodes.items()]
            candidates_nodes_weights: list[float] = [
                list(self.graph.in_edges(n, data="attribute"))[0][2]["weight"]
                for n in candidates_nodes
            ]
            node = np.random.choice(
                candidates_nodes, p=normalize(candidates_nodes_weights)
            )
        return node

    def jump_to_node(
        self, pred_intent: str, intent_idx: int, curr_node: str
    ) -> tuple[str, str]:
        """
        Jump to a node based on the intent
        """
        log_context.info(f"pred_intent in jump_to_node is {pred_intent}")
        try:
            candidates_nodes: list[dict[str, Any]] = [
                self.intents[pred_intent][intent_idx]
            ]
            candidates_nodes_weights: list[float] = [
                node["attribute"]["weight"] for node in candidates_nodes
            ]
            next_node: str = np.random.choice(
                [node["target_node"] for node in candidates_nodes],
                p=normalize(candidates_nodes_weights),
            )
            next_intent: str = pred_intent
        except Exception as e:
            log_context.error(f"Error in jump_to_node: {e}")
            next_node: str = curr_node
            next_intent: str = list(self.graph.in_edges(curr_node, data="intent"))[0][2]
        return next_node, next_intent

    def _build_neighbor_node_info(self, node_id: str) -> NodeInfo:
        n = self.graph.nodes[node_id]
        return NodeInfo(
            node_id=node_id,
            type=n.get("type", ""),
            resource_id=n["resource"]["id"],
            resource_name=n["resource"]["name"],
            can_skipped=True,
            is_leaf=len(list(self.graph.successors(node_id))) == 0,
            attributes=n["attribute"],
            add_flow_stack=False,
            additional_args={
                "tags": n["attribute"].get("tags", {}),
                **{
                    k2: v2
                    for k, v in n["attribute"].get("node_specific_data", {}).items()
                    if isinstance(v, dict)
                    for k2, v2 in v.items()
                },
                **{
                    k: v
                    for k, v in n["attribute"].get("node_specific_data", {}).items()
                    if not isinstance(v, dict)
                },
            },
        )

    def _get_node(
        self, sample_node: str, params: OrchestratorParams, intent: str | None = None
    ) -> tuple[NodeInfo, OrchestratorParams]:
        """
        Get the output format (NodeInfo, Params) that get_node should return
        """
        log_context.info(
            f"available_intents in _get_node: {params.taskgraph.available_global_intents}"
        )
        log_context.info(f"intent in _get_node: {intent}")
        node_info: dict[str, Any] = self.graph.nodes[sample_node]
        # Handle missing resource gracefully
        resource_name: str = node_info.get("resource", {}).get(
            "name", "default_resource"
        )
        resource_id: str = node_info.get("resource", {}).get("id", "default_id")
        if intent and intent in params.taskgraph.available_global_intents:
            # delete the corresponding node item from the intent list
            for item in params.taskgraph.available_global_intents.get(intent, []):
                if item["target_node"] == sample_node:
                    params.taskgraph.available_global_intents[intent].remove(item)
            if not params.taskgraph.available_global_intents[intent]:
                params.taskgraph.available_global_intents.pop(intent)

        params.taskgraph.curr_node = sample_node

        node_info = NodeInfo(
            node_id=sample_node,
            type=node_info.get("type", ""),
            resource_id=resource_id,
            resource_name=resource_name,
            can_skipped=node_info.get("attribute", {}).get("can_skipped", False),
            is_leaf=len(list(self.graph.successors(sample_node))) == 0,
            attributes=node_info["attribute"],
            add_flow_stack=False,
            additional_args={
                "successors": [
                    self._build_neighbor_node_info(succ)
                    for succ in self.graph.successors(sample_node)
                ],
                "predecessors": [
                    self._build_neighbor_node_info(pred)
                    for pred in self.graph.predecessors(sample_node)
                ],
                "prompt": node_info["attribute"].get("prompt", ""),
                "tags": node_info["attribute"].get("tags", {}),
                **{
                    k2: v2
                    for k, v in node_info["attribute"]
                    .get("node_specific_data", {})
                    .items()
                    if isinstance(v, dict)
                    for k2, v2 in v.items()
                },
                **{
                    k: v
                    for k, v in node_info["attribute"]
                    .get("node_specific_data", {})
                    .items()
                    if not isinstance(v, dict)
                },
            },
        )

        return node_info, params

    def _postprocess_intent(
        self,
        pred_intent: str,
        available_global_intents: list[str] | dict[str, list[dict[str, Any]]],
    ) -> tuple[bool, str, int]:
        found_pred_in_avil: bool = False
        real_intent: str = pred_intent
        idx: int = 0

        # Handle format like "1) test_intent" from IntentDetector
        if ") " in pred_intent:
            try:
                parts = pred_intent.split(") ", 1)
                if len(parts) == 2:
                    idx = int(parts[0])
                    real_intent = parts[1].strip()
            except (ValueError, IndexError):
                # If parsing fails, use the original pred_intent
                pass

        # check whether there are __<{idx}> in the pred_intent
        elif "__<" in pred_intent:
            real_intent = pred_intent.split("__<")[0]
            # get the idx
            idx = int(pred_intent.split("__<")[1].split(">")[0])

        # Convert dict to list of keys if needed
        intent_list = available_global_intents
        if isinstance(available_global_intents, dict):
            intent_list = list(available_global_intents.keys())

        for item in intent_list:
            if str_similarity(real_intent, item) > 0.9:
                found_pred_in_avil = True
                real_intent = item
                break
        # Fallback: if predicted intent is 'others' and 'others' is in available intents, treat as found
        if (
            not found_pred_in_avil
            and real_intent == "others"
            and "others" in intent_list
        ):
            found_pred_in_avil = True
        return found_pred_in_avil, real_intent, idx

    def get_current_node(
        self, params: OrchestratorParams
    ) -> tuple[str, OrchestratorParams]:
        """
        Get current node
        If current node is unknown, use start node
        """
        curr_node: str | None = params.taskgraph.curr_node
        if not curr_node:
            curr_node = self.start_node
        else:
            curr_node = str(curr_node)
            # Only fallback to start_node if the node is not in the graph
            if curr_node not in self.graph.nodes:
                curr_node = self.start_node
        params.taskgraph.curr_node = curr_node
        return curr_node, params

    def get_available_global_intents(
        self, params: OrchestratorParams
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Get available global intents
        """
        available_global_intents: dict[str, list[dict[str, Any]]] = (
            params.taskgraph.available_global_intents
        )
        if not available_global_intents:
            available_global_intents = copy.deepcopy(self.intents)

        # Always ensure unsure_intent is present
        if self.unsure_intent.get("intent") not in available_global_intents:
            available_global_intents[self.unsure_intent.get("intent")] = [
                self.unsure_intent
            ]
        log_context.info(f"Available global intents: {available_global_intents}")
        return available_global_intents

    def update_node_limit(self, params: OrchestratorParams) -> OrchestratorParams:
        """
        Update the node_limit in params which will be used to check if we can skip the node or not
        """
        old_node_limit: dict[str, int] = params.taskgraph.node_limit
        node_limit: dict[str, int] = {}
        for node in self.graph.nodes.data():
            limit: int | None = old_node_limit.get(node[0], node[1].get("limit"))
            if limit is not None:
                node_limit[node[0]] = limit
        params.taskgraph.node_limit = node_limit
        return params

    def get_local_intent(
        self, curr_node: str, params: OrchestratorParams
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Get the local intent of a current node
        """
        candidates_intents: collections.defaultdict[str, list[dict[str, Any]]] = (
            collections.defaultdict(list)
        )
        for u, v, data in self.graph.out_edges(curr_node, data=True):
            intent: str = data.get("intent")
            if intent != "none" and data.get("intent"):
                edge_info: dict[str, Any] = copy.deepcopy(data)
                edge_info["source_node"] = u
                edge_info["target_node"] = v
                candidates_intents[intent].append(edge_info)
        log_context.info(f"Current local intent: {candidates_intents}")
        return dict(candidates_intents)

    def get_last_flow_stack_node(self, params: OrchestratorParams) -> PathNode | None:
        """
        Get the last flow stack node from path
        """
        path: list[PathNode] = params.taskgraph.path
        for i in range(len(path) - 1, -1, -1):
            if path[i].in_flow_stack:
                path[i].in_flow_stack = False
                return path[i]
        return None

    def handle_multi_step_node(
        self, curr_node: str, params: OrchestratorParams
    ) -> tuple[bool, NodeInfo, OrchestratorParams]:
        """
        In case of a node having status == STAY, returned directly the same node
        """
        node_status: dict[str, StatusEnum] = params.taskgraph.node_status
        log_context.info(f"node_status: {node_status}")
        status: StatusEnum = node_status.get(curr_node, StatusEnum.COMPLETE)
        if status == StatusEnum.STAY:
            node_info: dict[str, Any] = self.graph.nodes[curr_node]
            # Handle missing resource gracefully
            resource_name: str = node_info.get("resource", {}).get(
                "name", "default_resource"
            )
            resource_id: str = node_info.get("resource", {}).get("id", "default_id")
            node_info = NodeInfo(
                type=node_info.get("type", ""),
                node_id=curr_node,
                resource_id=resource_id,
                resource_name=resource_name,
                can_skipped=node_info.get("attribute", {}).get("can_skipped", False),
                is_leaf=len(list(self.graph.successors(curr_node))) == 0,
                attributes=node_info["attribute"],
                additional_args={
                    "successors": [
                        self._build_neighbor_node_info(succ)
                        for succ in self.graph.successors(curr_node)
                    ],
                    "predecessors": [
                        self._build_neighbor_node_info(pred)
                        for pred in self.graph.predecessors(curr_node)
                    ],
                    "prompt": node_info["attribute"].get("prompt", ""),
                    "tags": node_info["attribute"].get("tags", {}),
                    **{
                        k2: v2
                        for k, v in node_info["attribute"]
                        .get("node_specific_data", {})
                        .items()
                        if isinstance(v, dict)
                        for k2, v2 in v.items()
                    },
                    **{
                        k: v
                        for k, v in node_info["attribute"]
                        .get("node_specific_data", {})
                        .items()
                        if not isinstance(v, dict)
                    },
                },
            )
            return True, node_info, params
        return False, NodeInfo(), params

    def handle_incomplete_node(
        self, curr_node: str, params: OrchestratorParams
    ) -> tuple[bool, dict[str, Any], OrchestratorParams]:
        """
        If node is incomplete, return directly the node
        """
        node_status: dict[str, StatusEnum] = params.taskgraph.node_status
        status: StatusEnum = node_status.get(curr_node, StatusEnum.COMPLETE)

        if status == StatusEnum.INCOMPLETE:
            log_context.info(
                "no local or global intent found, the current node is not complete"
            )
            node_info: NodeInfo
            node_info, params = self._get_node(curr_node, params)
            return True, node_info, params

        return False, {}, params

    def global_intent_prediction(
        self,
        curr_node: str,
        params: OrchestratorParams,
        available_global_intents: dict[str, list[dict[str, Any]]],
        excluded_intents: dict[str, Any],
    ) -> tuple[bool, str | None, dict[str, Any], OrchestratorParams]:
        """
        Do global intent prediction
        """
        candidate_intents: dict[str, list[dict[str, Any]]] = copy.deepcopy(
            available_global_intents
        )
        candidate_intents = {
            k: v for k, v in candidate_intents.items() if k not in excluded_intents
        }
        pred_intent: str | None = None
        # if only unsure_intent is available -> no meaningful intent prediction
        if (
            len(candidate_intents) == 1
            and self.unsure_intent.get("intent") in candidate_intents
        ):
            pred_intent = self.unsure_intent.get("intent")
            # Add NLU record for unsure intent
            params.taskgraph.nlu_records.append(
                {
                    "candidate_intents": candidate_intents,
                    "pred_intent": pred_intent,
                    "no_intent": False,
                    "global_intent": True,
                }
            )
            return False, pred_intent, {}, params
        else:  # global intent prediction
            # if match other intent, add flow, jump over
            candidate_intents[self.unsure_intent.get("intent")] = candidate_intents.get(
                self.unsure_intent.get("intent"), [self.unsure_intent]
            )
            log_context.info(
                f"Available global intents with unsure intent: {candidate_intents}"
            )

            pred_intent = self.intent_detector.execute(
                self.text,
                candidate_intents,
                self.chat_history_str,
                self.llm_config.model_dump(),
            )
            params.taskgraph.nlu_records.append(
                {
                    "candidate_intents": candidate_intents,
                    "pred_intent": pred_intent,
                    "no_intent": False,
                    "global_intent": True,
                }
            )
            found_pred_in_avil: bool
            intent_idx: int
            found_pred_in_avil, pred_intent, intent_idx = self._postprocess_intent(
                pred_intent, candidate_intents
            )
            # if found prediction and prediction is not unsure intent and current intent
            if found_pred_in_avil and pred_intent != self.unsure_intent.get("intent"):
                # If the prediction is the same as the current global intent and the current node is not a leaf node, continue the current global intent
                if (
                    pred_intent == params.taskgraph.curr_global_intent
                    and len(list(self.graph.successors(curr_node))) != 0
                    and params.taskgraph.node_status.get(
                        curr_node, StatusEnum.INCOMPLETE
                    )
                    == StatusEnum.INCOMPLETE
                ):
                    return False, pred_intent, {}, params
                next_node: str
                next_intent: str
                next_node, next_intent = self.jump_to_node(
                    pred_intent, intent_idx, curr_node
                )
                log_context.info(f"curr_node: {next_node}")
                node_info: NodeInfo
                node_info, params = self._get_node(
                    next_node, params, intent=next_intent
                )
                # if current node is not a leaf node and jump to another node, then add it onto stack
                if next_node != curr_node and list(self.graph.successors(curr_node)):
                    node_info.add_flow_stack = True
                params.taskgraph.curr_global_intent = pred_intent
                params.taskgraph.intent = pred_intent
                return True, pred_intent, node_info, params
        return False, pred_intent, {}, params

    def handle_random_next_node(
        self, curr_node: str, params: OrchestratorParams
    ) -> tuple[bool, dict[str, Any], OrchestratorParams]:
        candidate_samples: list[str] = []
        candidates_nodes_weights: list[float] = []
        for out_edge in self.graph.out_edges(curr_node, data=True):
            if out_edge[2]["intent"] == "none":
                candidate_samples.append(out_edge[1])
                candidates_nodes_weights.append(out_edge[2]["attribute"]["weight"])
        if candidate_samples:
            # randomly choose one sample from candidate samples
            next_node: str = np.random.choice(
                candidate_samples, p=normalize(candidates_nodes_weights)
            )
        else:  # leaf node + the node without None intents
            next_node: str = curr_node

        if (
            next_node != curr_node
        ):  # continue if curr_node is not leaf node, i.e. there is a actual next_node
            log_context.info(f"curr_node: {next_node}")
            node_info: NodeInfo
            node_info, params = self._get_node(next_node, params)
            if params.taskgraph.nlu_records:
                params.taskgraph.nlu_records[-1]["no_intent"] = (
                    True  # move on to the next node
                )
            else:  # only others available
                params.taskgraph.nlu_records = [
                    {
                        "candidate_intents": [],
                        "pred_intent": "",
                        "no_intent": True,
                        "global_intent": False,
                    }
                ]
            return True, node_info, params
        return False, {}, params

    def local_intent_prediction(
        self,
        curr_node: str,
        params: OrchestratorParams,
        curr_local_intents: dict[str, list[dict[str, Any]]],
    ) -> tuple[bool, dict[str, Any], OrchestratorParams]:
        """
        Do local intent prediction
        """
        curr_local_intents_w_unsure: dict[str, list[dict[str, Any]]] = copy.deepcopy(
            curr_local_intents
        )
        curr_local_intents_w_unsure[self.unsure_intent.get("intent")] = (
            curr_local_intents_w_unsure.get(
                self.unsure_intent.get("intent"), [self.unsure_intent]
            )
        )
        log_context.info(
            f"Check intent under current node: {curr_local_intents_w_unsure}"
        )
        pred_intent: str = self.intent_detector.execute(
            self.text,
            curr_local_intents_w_unsure,
            self.chat_history_str,
            self.llm_config.model_dump(),
        )
        params.taskgraph.nlu_records.append(
            {
                "candidate_intents": curr_local_intents_w_unsure,
                "pred_intent": pred_intent,
                "no_intent": False,
                "global_intent": False,
            }
        )
        found_pred_in_avil: bool
        intent_idx: int
        found_pred_in_avil, pred_intent, intent_idx = self._postprocess_intent(
            pred_intent, curr_local_intents_w_unsure
        )
        log_context.info(
            f"Local intent predition -> found_pred_in_avil: {found_pred_in_avil}, pred_intent: {pred_intent}"
        )
        if found_pred_in_avil:
            params.taskgraph.intent = pred_intent
            next_node: str = curr_node
            for edge in self.graph.out_edges(curr_node, data="intent"):
                if edge[2] == pred_intent:
                    next_node = edge[1]  # found intent under the current node
                    break
            log_context.info(f"curr_node: {next_node}")
            node_info: NodeInfo
            node_info, params = self._get_node(next_node, params, intent=pred_intent)
            if curr_node == self.start_node:
                params.taskgraph.curr_global_intent = pred_intent
            return True, node_info, params
        return False, {}, params

    def handle_unknown_intent(
        self, curr_node: str, params: OrchestratorParams
    ) -> tuple[NodeInfo, OrchestratorParams]:
        """
        If unknown intent, call planner
        """
        # if none of the available intents can represent user's utterance, transfer to the planner to let it decide for the next step
        params.taskgraph.intent = self.unsure_intent.get("intent")
        params.taskgraph.curr_global_intent = self.unsure_intent.get("intent")
        if params.taskgraph.nlu_records:
            params.taskgraph.nlu_records[-1]["no_intent"] = True  # no intent found
        else:
            params.taskgraph.nlu_records.append(
                {
                    "candidate_intents": [],
                    "pred_intent": "",
                    "no_intent": True,
                    "global_intent": False,
                }
            )
        params.taskgraph.curr_node = curr_node
        node_info: NodeInfo = NodeInfo(
            node_id=None,
            type="",
            resource_id="planner",
            resource_name="planner",
            can_skipped=False,
            is_leaf=len(list(self.graph.successors(curr_node))) == 0,
            attributes={"value": "", "direct": False},
            additional_args={"tags": {}},
        )
        return node_info, params

    def handle_leaf_node(
        self, curr_node: str, params: OrchestratorParams
    ) -> tuple[str, OrchestratorParams]:
        """
        if leaf node, first check if it's in a nested graph
        if not in nested graph, check if we have flow stack
        """

        def is_leaf(node: str) -> bool:
            if node not in self.graph.nodes:
                return True  # Consider non-existent nodes as leaf nodes
            return len(list(self.graph.successors(node))) == 0

        # if not leaf, return directly current node
        if not is_leaf(curr_node):
            return curr_node, params

        nested_graph_next_node: NodeInfo | None
        nested_graph_next_node, params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf
        )
        if nested_graph_next_node is not None:
            curr_node = nested_graph_next_node.node_id
            params.taskgraph.curr_node = curr_node
            if not is_leaf(nested_graph_next_node.node_id):
                return curr_node, params

        last_flow_stack_node: PathNode | None = self.get_last_flow_stack_node(params)
        if last_flow_stack_node:
            curr_node = last_flow_stack_node.node_id
            params.taskgraph.curr_global_intent = last_flow_stack_node.global_intent
        if self.initial_node:
            curr_node = self.initial_node

        return curr_node, params

    def get_node(self, inputs: dict[str, Any]) -> tuple[NodeInfo, OrchestratorParams]:
        """
        Get the next node
        """
        self.text: str = inputs["text"]
        self.chat_history_str: str = inputs["chat_history_str"]
        params: OrchestratorParams = inputs["parameters"]
        # boolean to check if we allow global intent switch or not.
        allow_global_intent_switch: bool = inputs["allow_global_intent_switch"]
        params.taskgraph.nlu_records = []

        if self.text == "<start>":
            curr_node: str = self.start_node
            params.taskgraph.curr_node = curr_node
            node_info: NodeInfo
            node_info, params = self._get_node(curr_node, params)
            return node_info, params

        curr_node: str
        curr_node, params = self.get_current_node(params)
        log_context.info(f"Intial curr_node: {curr_node}")

        # For the multi-step nodes, directly stay at that node instead of moving to other nodes
        is_multi_step_node: bool
        node_output: NodeInfo
        is_multi_step_node, node_output, params = self.handle_multi_step_node(
            curr_node, params
        )
        if is_multi_step_node:
            return node_output, params

        curr_node, params = self.handle_leaf_node(curr_node, params)

        # store current node
        params.taskgraph.curr_node = curr_node
        log_context.info(f"curr_node: {curr_node}")

        # available global intents
        available_global_intents: dict[str, list[dict[str, Any]]] = (
            self.get_available_global_intents(params)
        )

        # update limit
        params = self.update_node_limit(params)

        # Get local intents of the curr_node
        curr_local_intents: dict[str, list[dict[str, Any]]] = self.get_local_intent(
            curr_node, params
        )

        if (
            not curr_local_intents and allow_global_intent_switch
        ):  # no local intent under the current node
            log_context.info("no local intent under the current node")
            is_global_intent_found: bool
            pred_intent: str | None
            node_output: NodeInfo
            is_global_intent_found, pred_intent, node_output, params = (
                self.global_intent_prediction(
                    curr_node, params, available_global_intents, {}
                )
            )
            if is_global_intent_found:
                return node_output, params
            # If global intent prediction failed but we have a pred_intent that's not unsure,
            # try random next node
            if pred_intent and pred_intent != self.unsure_intent.get("intent"):
                has_random_next_node: bool
                node_output: dict[str, Any]
                has_random_next_node, node_output, params = (
                    self.handle_random_next_node(curr_node, params)
                )
                if has_random_next_node:
                    return node_output, params

        # if current node is incompleted -> return current node
        is_incomplete_node: bool
        node_output: dict[str, Any]
        is_incomplete_node, node_output, params = self.handle_incomplete_node(
            curr_node, params
        )
        if is_incomplete_node:
            return node_output, params

        # if completed and no local intents -> randomly choose one of the next connected nodes (edges with intent = None)
        if not curr_local_intents:
            log_context.info(
                "no local or global intent found, move to the next connected node(s)"
            )
            has_random_next_node: bool
            node_output: dict[str, Any]
            has_random_next_node, node_output, params = self.handle_random_next_node(
                curr_node, params
            )
            if has_random_next_node:
                return node_output, params

        log_context.info("Finish global condition, start local intent prediction")
        is_local_intent_found: bool
        node_output: dict[str, Any]
        is_local_intent_found, node_output, params = self.local_intent_prediction(
            curr_node, params, curr_local_intents
        )
        if is_local_intent_found:
            return node_output, params

        pred_intent: str | None = None
        if allow_global_intent_switch:
            is_global_intent_found: bool
            node_output: dict[str, Any]
            is_global_intent_found, pred_intent, node_output, params = (
                self.global_intent_prediction(
                    curr_node,
                    params,
                    available_global_intents,
                    {**curr_local_intents, **{"none": None}},
                )
            )
            if is_global_intent_found:
                return node_output, params
        if pred_intent and pred_intent != self.unsure_intent.get(
            "intent"
        ):  # if not unsure intent
            # If user didn't indicate all the intent of children nodes under the current node,
            # then we could randomly choose one of Nones to continue the dialog flow
            has_random_next_node: bool
            node_output: dict[str, Any]
            has_random_next_node, node_output, params = self.handle_random_next_node(
                curr_node, params
            )
            if has_random_next_node:
                return node_output, params

        # if none of the available intents can represent user's utterance or it is an unsure intents,
        # transfer to the planner to let it decide for the next step
        node_output: NodeInfo
        node_output, params = self.handle_unknown_intent(curr_node, params)
        return node_output, params

    def postprocess_node(
        self, node: tuple[NodeInfo, OrchestratorParams]
    ) -> tuple[NodeInfo, OrchestratorParams]:
        node_info: NodeInfo = node[0]
        params: OrchestratorParams = node[1]
        # TODO: future node postprocessing
        return node_info, params

    def _validate_node(self, node: dict[str, Any]) -> None:
        """Validate a node in the task graph.

        Args:
            node: Node to validate

        Raises:
            TaskGraphError: If node is invalid
        """
        if not isinstance(node, dict):
            log_context.error(
                "Node must be a dictionary",
                extra={"node": node},
            )
            raise TaskGraphError("Node must be a dictionary")

        if "id" not in node:
            log_context.error(
                "Node must have an id",
                extra={"node": node},
            )
            raise TaskGraphError("Node must have an id")

        if "type" not in node:
            log_context.error(
                "Node must have a type",
                extra={"node": node},
            )
            raise TaskGraphError("Node must have a type")

        if "next" in node and not isinstance(node["next"], list):
            log_context.error(
                "Node next must be a list",
                extra={"node": node},
            )
            raise TaskGraphError("Node next must be a list")

    def create_graph(self) -> None:
        nodes: list[dict[str, Any]] = self.product_kwargs["nodes"]
        edges: list[tuple[str, str, dict[str, Any]]] = self.product_kwargs["edges"]
        for edge in edges:
            edge[2]["intent"] = (
                edge[2]["intent"].lower() if edge[2]["intent"] else "none"
            )
        formatted_nodes = []
        for node in nodes:
            if isinstance(node, list | tuple) and len(node) == 2:
                formatted_nodes.append(node)
            elif isinstance(node, dict) and "id" in node:
                formatted_nodes.append((node["id"], node))
            else:
                formatted_nodes.append(node)
        self.graph.add_nodes_from(formatted_nodes)
        self.graph.add_edges_from(edges)
