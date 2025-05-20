import copy
import logging
import collections
from typing import Tuple, Dict, List, Any, Optional, DefaultDict

import networkx as nx
import numpy as np

from arklex.utils.utils import normalize, str_similarity
from arklex.utils.graph_state import NodeInfo, Params, PathNode, StatusEnum, LLMConfig
from arklex.orchestrator.NLU.nlu import NLU, SlotFilling

logger = logging.getLogger(__name__)


class TaskGraphBase:
    def __init__(self, name: str, product_kwargs: Dict[str, Any]) -> None:
        self.graph: nx.DiGraph = nx.DiGraph(name=name)
        self.product_kwargs: Dict[str, Any] = product_kwargs
        self.create_graph()
        self.intents: DefaultDict[str, List[Dict[str, Any]]] = (
            self.get_pred_intents()
        )  # global intents
        self.start_node: Optional[str] = self.get_start_node()

    def create_graph(self) -> None:
        raise NotImplementedError

    def get_pred_intents(self) -> DefaultDict[str, List[Dict[str, Any]]]:
        intents: DefaultDict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for edge in self.graph.edges.data():
            if edge[2].get("attribute", {}).get("pred", False):
                edge_info: Dict[str, Any] = copy.deepcopy(edge[2])
                edge_info["source_node"] = edge[0]
                edge_info["target_node"] = edge[1]
                intents[edge[2].get("intent")].append(edge_info)
        return intents

    def get_start_node(self) -> Optional[str]:
        for node in self.graph.nodes.data():
            if node[1].get("type", "") == "start":
                return node[0]
        return None


class TaskGraph(TaskGraphBase):
    def __init__(
        self, name: str, product_kwargs: Dict[str, Any], llm_config: LLMConfig
    ) -> None:
        super().__init__(name, product_kwargs)
        self.unsure_intent: Dict[str, Any] = {
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
        self.initial_node: Optional[str] = self.get_initial_flow()
        self.llm_config: LLMConfig = llm_config
        self.nluapi: NLU = NLU(self.product_kwargs.get("nluapi"))
        self.slotfillapi: SlotFilling = SlotFilling(
            self.product_kwargs.get("slotfillapi")
        )

    def create_graph(self) -> None:
        nodes: List[Dict[str, Any]] = self.product_kwargs["nodes"]
        edges: List[Tuple[str, str, Dict[str, Any]]] = self.product_kwargs["edges"]
        # convert the intent into lowercase
        for edge in edges:
            edge[2]["intent"] = edge[2]["intent"].lower()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)

    def get_initial_flow(self) -> Optional[str]:
        services_nodes: Optional[Dict[str, str]] = self.product_kwargs.get(
            "services_nodes", None
        )
        node: Optional[str] = None
        if services_nodes:
            candidates_nodes: List[str] = [v for k, v in services_nodes.items()]
            candidates_nodes_weights: List[float] = [
                list(self.graph.in_edges(n, data="attribute"))[0][2]["weight"]
                for n in candidates_nodes
            ]
            node = np.random.choice(
                candidates_nodes, p=normalize(candidates_nodes_weights)
            )
        return node

    def jump_to_node(
        self, pred_intent: str, intent_idx: int, curr_node: str
    ) -> Tuple[str, str]:
        """Jump to a node based on the intent.

        Args:
            pred_intent (str): The predicted intent.
            intent_idx (int): The index of the intent.
            curr_node (str): The current node.

        Returns:
            Tuple[str, str]: A tuple containing the next node and intent.
        """
        logger.info(f"pred_intent in jump_to_node is {pred_intent}")
        try:
            candidates_nodes: List[Dict[str, Any]] = [
                self.intents[pred_intent][intent_idx]
            ]
            candidates_nodes_weights: List[float] = [
                node["attribute"]["weight"] for node in candidates_nodes
            ]
            if candidates_nodes:
                next_node: str = np.random.choice(
                    [node["target_node"] for node in candidates_nodes],
                    p=normalize(candidates_nodes_weights),
                )
                next_intent: str = pred_intent
            else:  # This is for protection, logically shouldn't enter this branch
                next_node: str = curr_node
                next_intent: str = list(self.graph.in_edges(curr_node, data="intent"))[
                    0
                ][2]
        except Exception as e:
            logger.error(f"Error in jump_to_node: {e}")
            next_node: str = curr_node
            next_intent: str = list(self.graph.in_edges(curr_node, data="intent"))[0][2]
        return next_node, next_intent

    def _get_node(
        self, sample_node: str, params: Params, intent: Optional[str] = None
    ) -> Tuple[NodeInfo, Params]:
        """Get the output format (NodeInfo, Params) that get_node should return.

        Args:
            sample_node (str): The sample node.
            params (Params): The current parameters.
            intent (Optional[str], optional): The intent. Defaults to None.

        Returns:
            Tuple[NodeInfo, Params]: A tuple containing the node info and updated parameters.
        """
        logger.info(
            f"available_intents in _get_node: {params.taskgraph.available_global_intents}"
        )
        logger.info(f"intent in _get_node: {intent}")
        node_info: Dict[str, Any] = self.graph.nodes[sample_node]
        resource_name: str = node_info["resource"]["name"]
        resource_id: str = node_info["resource"]["id"]
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
            can_skipped=True,
            is_leaf=len(list(self.graph.successors(sample_node))) == 0,
            attributes=node_info["attribute"],
            add_flow_stack=False,
            additional_args={"tags": node_info["attribute"].get("tags", {})},
        )

        return node_info, params

    def _postprocess_intent(
        self, pred_intent: str, available_global_intents: List[str]
    ) -> Tuple[bool, str, int]:
        """Post-process the intent.

        Args:
            pred_intent (str): The predicted intent.
            available_global_intents (List[str]): The available global intents.

        Returns:
            Tuple[bool, str, int]: A tuple containing whether the intent was found,
                                 the real intent, and the index.
        """
        found_pred_in_avil: bool = False
        real_intent: str = pred_intent
        idx: int = 0
        # check whether there are __<{idx}> in the pred_intent
        if "__<" in pred_intent:
            real_intent = pred_intent.split("__<")[0]
            # get the idx
            idx = int(pred_intent.split("__<")[1].split(">")[0])
        for item in available_global_intents:
            if str_similarity(real_intent, item) > 0.9:
                found_pred_in_avil = True
                real_intent = item
                break
        return found_pred_in_avil, real_intent, idx

    def get_current_node(self, params: Params) -> Tuple[str, Params]:
        """Get current node.
        If current node is unknown, use start node.

        Args:
            params (Params): The current parameters.

        Returns:
            Tuple[str, Params]: A tuple containing the current node and updated parameters.
        """
        curr_node: Optional[str] = params.taskgraph.curr_node
        if not curr_node or curr_node not in self.graph.nodes:
            curr_node = self.start_node
        else:
            curr_node = str(curr_node)
        params.taskgraph.curr_node = curr_node
        return curr_node, params

    def get_available_global_intents(
        self, params: Params
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get available global intents.

        Args:
            params (Params): The current parameters.

        Returns:
            Dict[str, List[Dict[str, Any]]]: The available global intents.
        """
        available_global_intents: Dict[str, List[Dict[str, Any]]] = (
            params.taskgraph.available_global_intents
        )
        if not available_global_intents:
            available_global_intents = copy.deepcopy(self.intents)
            if self.unsure_intent.get("intent") not in available_global_intents.keys():
                available_global_intents[self.unsure_intent.get("intent")].append(
                    self.unsure_intent
                )
        logger.info(f"Available global intents: {available_global_intents}")
        return available_global_intents

    def update_node_limit(self, params: Params) -> Params:
        """Update the node_limit in params which will be used to check if we can skip the node or not.

        Args:
            params (Params): The current parameters.

        Returns:
            Params: The updated parameters.
        """
        old_node_limit: Dict[str, int] = params.taskgraph.node_limit
        node_limit: Dict[str, int] = {}
        for node in self.graph.nodes.data():
            limit: Optional[int] = old_node_limit.get(node[0], node[1].get("limit"))
            if limit is not None:
                node_limit[node[0]] = limit
        params.taskgraph.node_limit = node_limit
        return params

    def get_local_intent(
        self, curr_node: str, params: Params
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get the local intent of a current node.

        Args:
            curr_node (str): The current node.
            params (Params): The current parameters.

        Returns:
            Dict[str, List[Dict[str, Any]]]: The local intents.
        """
        candidates_intents: DefaultDict[str, List[Dict[str, Any]]] = (
            collections.defaultdict(list)
        )
        for u, v, data in self.graph.out_edges(curr_node, data=True):
            intent: str = data.get("intent")
            if intent != "none" and data.get("intent"):
                edge_info: Dict[str, Any] = copy.deepcopy(data)
                edge_info["source_node"] = u
                edge_info["target_node"] = v
                candidates_intents[intent].append(edge_info)
        logger.info(f"Current local intent: {candidates_intents}")
        return dict(candidates_intents)

    def get_last_flow_stack_node(self, params: Params) -> Optional[PathNode]:
        """Get the last flow stack node from path.

        Args:
            params (Params): The current parameters.

        Returns:
            Optional[PathNode]: The last flow stack node, or None if not found.
        """
        path: List[PathNode] = params.taskgraph.path
        for i in range(len(path) - 1, -1, -1):
            if path[i].in_flow_stack:
                path[i].in_flow_stack = False
                return path[i]
        return None

    def handle_multi_step_node(
        self, curr_node: str, params: Params
    ) -> Tuple[bool, NodeInfo, Params]:
        """Handle a multi-step node.
        In case of a node having status == STAY, returned directly the same node.

        Args:
            curr_node (str): The current node.
            params (Params): The current parameters.

        Returns:
            Tuple[bool, NodeInfo, Params]: A tuple containing whether the node was handled,
                                        the node info, and updated parameters.
        """
        node_status: Dict[str, StatusEnum] = params.taskgraph.node_status
        logger.info(f"node_status: {node_status}")
        status: StatusEnum = node_status.get(curr_node, StatusEnum.COMPLETE)
        if status == StatusEnum.STAY:
            node_info: Dict[str, Any] = self.graph.nodes[curr_node]
            resource_name: str = node_info["resource"]["name"]
            resource_id: str = node_info["resource"]["id"]
            node_info = NodeInfo(
                type=node_info.get("type", ""),
                node_id=curr_node,
                resource_id=resource_id,
                resource_name=resource_name,
                can_skipped=False,
                is_leaf=len(list(self.graph.successors(curr_node))) == 0,
                attributes=node_info["attribute"],
                add_flow_stack=False,
                additional_args={"tags": node_info["attribute"].get("tags", {})},
            )
            return True, node_info, params
        return False, None, params

    def handle_incomplete_node(
        self, curr_node: str, params: Params
    ) -> Tuple[bool, Dict[str, Any], Params]:
        """Handle an incomplete node.

        Args:
            curr_node (str): The current node.
            params (Params): The current parameters.

        Returns:
            Tuple[bool, Dict[str, Any], Params]: A tuple containing whether the node was handled,
                                              the node info, and updated parameters.
        """
        node_status: Dict[str, StatusEnum] = params.taskgraph.node_status
        status: StatusEnum = node_status.get(curr_node, StatusEnum.COMPLETE)
        if status == StatusEnum.INCOMPLETE:
            logger.info(
                f"no local or global intent found, the current node is not complete"
            )
            node_info, params = self._get_node(curr_node, params)
            return True, node_info, params
        return False, {}, params

    def global_intent_prediction(
        self,
        curr_node: str,
        params: Params,
        available_global_intents: Dict[str, List[Dict[str, Any]]],
        excluded_intents: List[str],
    ) -> Tuple[bool, str, Dict[str, Any], Params]:
        """Predict the global intent.

        Args:
            curr_node (str): The current node.
            params (Params): The current parameters.
            available_global_intents (Dict[str, List[Dict[str, Any]]]): The available global intents.
            excluded_intents (List[str]): The excluded intents.

        Returns:
            Tuple[bool, str, Dict[str, Any], Params]: A tuple containing whether the intent was predicted,
                                                    the intent, the node info, and updated parameters.
        """
        if not available_global_intents:
            return False, "", {}, params

        # Get the text from the last message
        text: str = params.memory.function_calling_trajectory[-1]["content"]
        # Get the intent from the NLU API
        intent: str = self.nluapi.get_intent(
            text, available_global_intents.keys(), excluded_intents
        )
        if intent:
            # Get the node info from the intent
            node_info: Dict[str, Any] = available_global_intents[intent][0]
            return True, intent, node_info, params
        return False, "", {}, params

    def handle_random_next_node(
        self, curr_node: str, params: Params
    ) -> Tuple[bool, Dict[str, Any], Params]:
        """Handle a random next node.

        Args:
            curr_node (str): The current node.
            params (Params): The current parameters.

        Returns:
            Tuple[bool, Dict[str, Any], Params]: A tuple containing whether the node was handled,
                                              the node info, and updated parameters.
        """
        # Get the successors of the current node
        successors: List[str] = list(self.graph.successors(curr_node))
        if not successors:
            return False, {}, params

        # Get the weights of the successors
        weights: List[float] = [
            self.graph[curr_node][succ]["attribute"]["weight"] for succ in successors
        ]
        # Choose a random successor based on the weights
        next_node: str = np.random.choice(successors, p=normalize(weights))
        node_info: Dict[str, Any] = self.graph.nodes[next_node]
        return True, node_info, params

    def local_intent_prediction(
        self,
        curr_node: str,
        params: Params,
        curr_local_intents: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[bool, Dict[str, Any], Params]:
        """Predict the local intent.

        Args:
            curr_node (str): The current node.
            params (Params): The current parameters.
            curr_local_intents (Dict[str, List[Dict[str, Any]]]): The current local intents.

        Returns:
            Tuple[bool, Dict[str, Any], Params]: A tuple containing whether the intent was predicted,
                                              the node info, and updated parameters.
        """
        if not curr_local_intents:
            return False, {}, params

        # Get the text from the last message
        text: str = params.memory.function_calling_trajectory[-1]["content"]
        # Get the intent from the NLU API
        intent: str = self.nluapi.get_intent(text, curr_local_intents.keys(), [])
        if intent:
            # Get the node info from the intent
            node_info: Dict[str, Any] = curr_local_intents[intent][0]
            return True, node_info, params
        return False, {}, params

    def handle_unknown_intent(
        self, curr_node: str, params: Params
    ) -> Tuple[Dict[str, Any], Params]:
        """Handle an unknown intent.

        Args:
            curr_node (str): The current node.
            params (Params): The current parameters.

        Returns:
            Tuple[Dict[str, Any], Params]: A tuple containing the node info and updated parameters.
        """
        # Get the node info from the unsure intent
        node_info: Dict[str, Any] = self.unsure_intent
        return node_info, params

    def handle_leaf_node(
        self, curr_node: str, params: Params
    ) -> Tuple[bool, Dict[str, Any], Params]:
        """Handle a leaf node.

        Args:
            curr_node (str): The current node.
            params (Params): The current parameters.

        Returns:
            Tuple[bool, Dict[str, Any], Params]: A tuple containing whether the node was handled,
                                              the node info, and updated parameters.
        """

        def is_leaf(node: str) -> bool:
            """Check if a node is a leaf node.

            Args:
                node (str): The node to check.

            Returns:
                bool: True if the node is a leaf node, False otherwise.
            """
            return len(list(self.graph.successors(node))) == 0

        if is_leaf(curr_node):
            node_info: Dict[str, Any] = self.graph.nodes[curr_node]
            return True, node_info, params
        return False, {}, params

    def get_node(self, inputs: Dict[str, Any]) -> Tuple[NodeInfo, Params]:
        """Get a node.

        Args:
            inputs (Dict[str, Any]): The input parameters.

        Returns:
            Tuple[NodeInfo, Params]: A tuple containing the node info and updated parameters.
        """
        text: str = inputs["text"]
        chat_history_str: str = inputs["chat_history_str"]
        params: Params = inputs["parameters"]
        allow_global_intent_switch: bool = inputs.get(
            "allow_global_intent_switch", False
        )

        # Get the current node
        curr_node: str
        curr_node, params = self.get_current_node(params)
        logger.info(f"Intial curr_node: {curr_node}")

        # Handle multi-step node
        handled: bool
        node_info: NodeInfo
        handled, node_info, params = self.handle_multi_step_node(curr_node, params)
        if handled:
            return node_info, params

        curr_node, params = self.handle_leaf_node(curr_node, params)

        # store current node
        params.taskgraph.curr_node = curr_node
        logger.info(f"curr_node: {curr_node}")

        # available global intents
        available_global_intents: Dict[str, List[Dict[str, Any]]] = (
            self.get_available_global_intents(params)
        )

        # update limit
        params = self.update_node_limit(params)

        # Get local intents of the curr_node
        curr_local_intents: Dict[str, List[Dict[str, Any]]] = self.get_local_intent(
            curr_node, params
        )

        # Get available global intents
        excluded_intents: List[str] = []
        if not allow_global_intent_switch:
            excluded_intents = list(available_global_intents.keys())

        # Predict global intent
        handled: bool
        intent: str
        node_info: Dict[str, Any]
        handled, intent, node_info, params = self.global_intent_prediction(
            curr_node, params, available_global_intents, excluded_intents
        )
        if handled:
            return self.postprocess_node(node_info)

        # if current node is incompleted -> return current node
        is_incomplete_node, node_info, params = self.handle_incomplete_node(
            curr_node, params
        )
        if is_incomplete_node:
            return self.postprocess_node(node_info)

        # if completed and no local intents -> randomly choose one of the next connected nodes (edges with intent = None)
        if not curr_local_intents:
            logger.info(
                f"no local or global intent found, move to the next connected node(s)"
            )
            has_random_next_node, node_info, params = self.handle_random_next_node(
                curr_node, params
            )
            if has_random_next_node:
                return self.postprocess_node(node_info)

        logger.info("Finish global condition, start local intent prediction")
        is_local_intent_found, node_info, params = self.local_intent_prediction(
            curr_node, params, curr_local_intents
        )
        if is_local_intent_found:
            return self.postprocess_node(node_info)

        # if none of the available intents can represent user's utterance or it is an unsure intents,
        # transfer to the planner to let it decide for the next step
        node_info, params = self.handle_unknown_intent(curr_node, params)
        return self.postprocess_node(node_info)

    def postprocess_node(self, node: Dict[str, Any]) -> Tuple[NodeInfo, Params]:
        """Post-process a node.

        Args:
            node (Dict[str, Any]): The node to post-process.

        Returns:
            Tuple[NodeInfo, Params]: A tuple containing the node info and updated parameters.
        """
        node_info: NodeInfo = NodeInfo(
            node_id=node["target_node"],
            type=node.get("type", ""),
            resource_id=node["resource"]["id"],
            resource_name=node["resource"]["name"],
            can_skipped=node.get("can_skipped", True),
            is_leaf=len(list(self.graph.successors(node["target_node"]))) == 0,
            attributes=node["attribute"],
            add_flow_stack=node.get("add_flow_stack", False),
            additional_args={"tags": node["attribute"].get("tags", {})},
        )
        return node_info, Params()
