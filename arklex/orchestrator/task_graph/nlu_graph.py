import collections
import copy
from typing import Any

import numpy as np

from arklex.models.llm_config import LLMConfig
from arklex.models.model_service import (
    ModelService,
)
from arklex.orchestrator.entities.orchestrator_state_entities import (
    StatusEnum,
)
from arklex.orchestrator.entities.taskgraph_entities import (
    NLUGraphParams,
    NodeInfo,
)
from arklex.orchestrator.nlu.core.intent import IntentDetector
from arklex.orchestrator.task_graph.base import GraphBase
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


class NLUGraph(GraphBase):
    def __init__(
        self,
        name: str,
        graph_config: dict[str, Any],
        llm_config: LLMConfig,
    ) -> None:
        """Initialize the NLU graph.

        Args:
            name: Name of the nlu graph
            graph_config: Configuration settings for the graph
            llm_config: Language model configuration
        """
        super().__init__(name, graph_config)
        self.intents: collections.defaultdict[str, list[dict[str, Any]]] = (
            self.get_pred_intents()
        )  # global intents
        self.agent_node: NodeInfo = self.get_agent_node()
        log_context.info(f"agent_node: {self.agent_node}")
        self.start_node: str = self.get_start_node()
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
        model_service = ModelService(llm_config)
        self.intent_detector: IntentDetector = IntentDetector(model_service)

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

    def get_agent_node(self) -> NodeInfo:
        for node in self.graph.nodes.data():
            if node[1].get("attribute", {}).get("start", False):
                node_info = self.graph.nodes[node[0]]
                return NodeInfo(
                    node_id=node[0],
                    data=node_info["data"],
                )
        raise ValueError("No agent node found in the graph")

    def get_start_node(self) -> str:
        agent_node_id = self.agent_node.node_id
        candidate_samples: list[str] = []
        for out_edge in self.graph.out_edges(agent_node_id, data=True):
            if out_edge[2]["intent"] == "none":
                candidate_samples.append(out_edge[1])
        if candidate_samples:
            # randomly choose one sample from candidate samples
            start_node: str = np.random.choice(candidate_samples)
            return start_node
        log_context.warning("No random next node found for NLU agent node")
        return agent_node_id

    def jump_to_node(self, pred_intent: str, curr_node: str) -> tuple[str, str]:
        """
        Jump to a node based on the intent
        """
        log_context.info(f"pred_intent in jump_to_node is {pred_intent}")
        # One global intent can have multiple nodes, choose the first one by default
        intent_idx = 0
        try:
            candidates_nodes: list[dict[str, Any]] = [
                self.intents[pred_intent][intent_idx]
            ]
            # Use equal weights instead of node attribute weights
            next_node: str = np.random.choice(
                [node["target_node"] for node in candidates_nodes]
            )
            next_intent: str = pred_intent
        except Exception as e:
            log_context.error(f"Error in jump_to_node: {e}")
            next_node: str = curr_node
            next_intent: str = list(self.graph.in_edges(curr_node, data="intent"))[0][2]
        return next_node, next_intent

    def _get_node(
        self, sample_node: str, params: NLUGraphParams, intent: str | None = None
    ) -> tuple[NodeInfo, NLUGraphParams]:
        """
        Get the output format (NodeInfo, Params) that get_node should return
        """
        log_context.info(
            f"available_intents in _get_node: {params.available_global_intents}"
        )
        log_context.info(f"intent in _get_node: {intent}")
        node_info: dict[str, Any] = self.graph.nodes[sample_node]
        if intent and intent in params.available_global_intents:
            # delete the corresponding node item from the intent list
            for item in params.available_global_intents.get(intent, []):
                if item["target_node"] == sample_node:
                    params.available_global_intents[intent].remove(item)
            if not params.available_global_intents[intent]:
                params.available_global_intents.pop(intent)

        params.curr_node = sample_node
        node_info = NodeInfo(
            node_id=sample_node,
            resource=node_info["resource"],
            attribute=node_info["attribute"],
            data=node_info["data"],
            is_leaf=len(list(self.graph.successors(sample_node))) == 0,
        )

        return node_info, params

    def get_current_node(self, params: NLUGraphParams) -> tuple[str, NLUGraphParams]:
        """
        Get current node
        If current node is unknown, use start node
        """
        curr_node: str | None = params.curr_node
        if not curr_node:
            curr_node = self.start_node
        else:
            curr_node = str(curr_node)
            # Only fallback to start_node if the node is not in the graph
            if curr_node not in self.graph.nodes:
                curr_node = self.start_node
        params.curr_node = curr_node
        return curr_node, params

    def get_available_global_intents(
        self, params: NLUGraphParams
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Get available global intents
        """
        available_global_intents: dict[str, list[dict[str, Any]]] = (
            params.available_global_intents
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

    def get_local_intent(
        self, curr_node: str, params: NLUGraphParams
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

    def handle_multi_step_node(
        self, curr_node: str, params: NLUGraphParams
    ) -> tuple[bool, NodeInfo, NLUGraphParams]:
        """
        In case of a node having status == STAY, returned directly the same node
        """
        node_status: dict[str, StatusEnum] = params.node_status
        log_context.info(f"node_status: {node_status}")
        status: StatusEnum = node_status.get(curr_node, StatusEnum.COMPLETE)
        if status == StatusEnum.STAY:
            node_info: dict[str, Any] = self.graph.nodes[curr_node]
            node_info = NodeInfo(
                node_id=curr_node,
                resource=node_info["resource"],
                attribute=node_info["attribute"],
                data=node_info["data"],
                is_leaf=len(list(self.graph.successors(curr_node))) == 0,
            )
            return True, node_info, params
        return False, NodeInfo(), params

    def handle_incomplete_node(
        self, curr_node: str, params: NLUGraphParams
    ) -> tuple[bool, dict[str, Any], NLUGraphParams]:
        """
        If node is incomplete, return directly the node
        """
        node_status: dict[str, StatusEnum] = params.node_status
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
        params: NLUGraphParams,
        available_global_intents: dict[str, list[dict[str, Any]]],
        excluded_intents: dict[str, Any],
    ) -> tuple[bool, str | None, dict[str, Any], NLUGraphParams]:
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
            params.nlu_records.append(
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
                candidate_intents,
                self.chat_history_str,
            )
            params.nlu_records.append(
                {
                    "candidate_intents": candidate_intents,
                    "pred_intent": pred_intent,
                    "no_intent": False,
                    "global_intent": True,
                }
            )
            # if found prediction and prediction is not unsure intent and current intent
            if pred_intent != self.unsure_intent.get("intent"):
                # If the prediction is the same as the current global intent and the current node is not a leaf node, continue the current global intent
                if pred_intent == params.curr_global_intent and (
                    len(list(self.graph.successors(curr_node))) != 0
                    or params.node_status.get(curr_node, StatusEnum.COMPLETE)
                    == StatusEnum.INCOMPLETE
                ):
                    return False, pred_intent, {}, params
                next_node: str
                next_intent: str
                next_node, next_intent = self.jump_to_node(pred_intent, curr_node)
                log_context.info(f"curr_node: {next_node}")
                node_info: NodeInfo
                node_info, params = self._get_node(
                    next_node, params, intent=next_intent
                )
                params.curr_global_intent = pred_intent
                params.intent = pred_intent
                return True, pred_intent, node_info, params
        return False, pred_intent, {}, params

    def handle_random_next_node(
        self, curr_node: str, params: NLUGraphParams
    ) -> tuple[bool, dict[str, Any], NLUGraphParams]:
        candidate_samples: list[str] = []
        for out_edge in self.graph.out_edges(curr_node, data=True):
            if out_edge[2]["intent"] == "none":
                candidate_samples.append(out_edge[1])
        if candidate_samples:
            # randomly choose one sample from candidate samples
            next_node: str = np.random.choice(candidate_samples)
        else:  # leaf node + the node without None intents
            next_node: str = curr_node

        if (
            next_node != curr_node
        ):  # continue if curr_node is not leaf node, i.e. there is a actual next_node
            log_context.info(f"curr_node: {next_node}")
            node_info: NodeInfo
            node_info, params = self._get_node(next_node, params)
            if params.nlu_records:
                params.nlu_records[-1]["no_intent"] = True  # move on to the next node
            else:  # only others available
                params.nlu_records = [
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
        params: NLUGraphParams,
        curr_local_intents: dict[str, list[dict[str, Any]]],
    ) -> tuple[bool, dict[str, Any], NLUGraphParams]:
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
        # if only unsure_intent is available -> no meaningful intent prediction
        if (
            len(curr_local_intents_w_unsure) == 1
            and self.unsure_intent.get("intent") in curr_local_intents_w_unsure
        ):
            pred_intent = self.unsure_intent.get("intent")
            params.nlu_records.append(
                {
                    "candidate_intents": curr_local_intents_w_unsure,
                    "pred_intent": pred_intent,
                    "no_intent": False,
                    "global_intent": False,
                }
            )
            return False, pred_intent, params

        pred_intent: str = self.intent_detector.execute(
            curr_local_intents_w_unsure,
            self.chat_history_str,
        )
        params.nlu_records.append(
            {
                "candidate_intents": curr_local_intents_w_unsure,
                "pred_intent": pred_intent,
                "no_intent": False,
                "global_intent": False,
            }
        )
        log_context.info(f"Local intent prediction: pred_intent: {pred_intent}")
        if pred_intent != self.unsure_intent.get("intent"):
            params.intent = pred_intent
            next_node: str = curr_node
            for edge in self.graph.out_edges(curr_node, data="intent"):
                if edge[2] == pred_intent:
                    next_node = edge[1]  # found intent under the current node
                    break
            log_context.info(f"curr_node: {next_node}")
            node_info: NodeInfo
            node_info, params = self._get_node(next_node, params, intent=pred_intent)
            if curr_node == self.start_node:
                params.curr_global_intent = pred_intent
            return True, node_info, params
        return False, {}, params

    def handle_unknown_intent(
        self, curr_node: str, params: NLUGraphParams
    ) -> tuple[NodeInfo, NLUGraphParams]:
        """
        If unknown intent, call planner
        """
        # if none of the available intents can represent user's utterance, transfer to the planner to let it decide for the next step
        params.intent = self.unsure_intent.get("intent")
        params.curr_global_intent = self.unsure_intent.get("intent")
        if params.nlu_records:
            # no intent found
            params.nlu_records[-1]["no_intent"] = True
        else:
            params.nlu_records.append(
                {
                    "candidate_intents": [],
                    "pred_intent": "",
                    "no_intent": True,
                    "global_intent": False,
                }
            )
        params.curr_node = curr_node
        node_info: NodeInfo = NodeInfo(
            node_id="",
            resource={"id": "planner", "name": "planner"},
            attribute={"value": "", "direct": False},
            data={},
            is_leaf=len(list(self.graph.successors(curr_node))) == 0,
        )
        return node_info, params

    def handle_leaf_node(
        self, curr_node: str, params: NLUGraphParams
    ) -> tuple[str, NLUGraphParams]:
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

        return curr_node, params

    def get_node(self, inputs: dict[str, Any]) -> tuple[NodeInfo, NLUGraphParams]:
        """
        Get the next node
        """
        self.text: str = inputs["text"]
        self.chat_history_str: str = inputs["chat_history_str"]
        params: NLUGraphParams = inputs["nlu_params"]
        # boolean to check if we allow global intent switch or not.
        allow_global_intent_switch: bool = inputs["allow_global_intent_switch"]
        params.nlu_records = []

        if self.text == "<start>":
            curr_node: str = self.start_node
            params.curr_node = curr_node
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
        params.curr_node = curr_node
        log_context.info(f"curr_node: {curr_node}")

        # available global intents
        available_global_intents: dict[str, list[dict[str, Any]]] = (
            self.get_available_global_intents(params)
        )

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

    def create_graph(self) -> None:
        nodes: list[dict[str, Any]] = self.graph_config["nodes"]
        edges: list[tuple[str, str, dict[str, Any]]] = self.graph_config["edges"]
        for edge in edges:
            edge[2]["intent"] = (
                edge[2]["intent"].lower() if edge[2]["intent"] else "none"
            )
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)
