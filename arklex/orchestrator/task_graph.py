import copy
import logging
import collections
from typing import Tuple

import networkx as nx
import numpy as np

from arklex.env.nested_graph.nested_graph import NestedGraph
from arklex.utils.utils import normalize, str_similarity
from arklex.utils.graph_state import NodeInfo, Params, PathNode, StatusEnum, LLMConfig
from arklex.orchestrator.NLU.nlu import NLU, SlotFilling

logger = logging.getLogger(__name__)

class TaskGraphBase:
    def __init__(self, name, product_kwargs):
        self.graph = nx.DiGraph(name=name)
        self.product_kwargs = product_kwargs
        self.create_graph()
        self.intents = self.get_pred_intents() # global intents
        self.start_node = self.get_start_node()

    def create_graph(self):
        raise NotImplementedError

    def get_pred_intents(self):
        intents = collections.defaultdict(list)
        for edge in self.graph.edges.data():
            if edge[2].get("attribute", {}).get("pred", False):
                edge_info = copy.deepcopy(edge[2])
                edge_info["source_node"] = edge[0]
                edge_info["target_node"] = edge[1]
                intents[edge[2].get("intent")].append(edge_info)
        return intents
    
    def get_start_node(self):
        for node in self.graph.nodes.data():
            if node[1].get("type", "") == "start":
                return node[0]
        return None


class TaskGraph(TaskGraphBase):
    def __init__(self, name: str, product_kwargs: dict, llm_config: LLMConfig):
        super().__init__(name, product_kwargs)
        self.unsure_intent = {
                "intent": "others",
                "source_node": None,
                "target_node": None,
                "attribute": {
                    "weight": 1,
                    "pred": False,
                    "definition": "",
                    "sample_utterances": []
                }
            }
        self.initial_node = self.get_initial_flow()
        self.llm_config = llm_config
        self.nluapi = NLU(self.product_kwargs.get("nluapi"))
        self.slotfillapi = SlotFilling(self.product_kwargs.get("slotfillapi"))

    def create_graph(self):
        nodes = self.product_kwargs["nodes"]
        edges = self.product_kwargs["edges"]
        # convert the intent into lowercase
        for edge in edges:
            edge[2]['intent'] = edge[2]['intent'].lower()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)

    def get_initial_flow(self):
        services_nodes = self.product_kwargs.get("services_nodes", None)
        node = None
        if services_nodes:
            candidates_nodes = [v for k, v in services_nodes.items()]
            candidates_nodes_weights = [list(self.graph.in_edges(n, data="attribute"))[0][2]["weight"] for n in candidates_nodes]
            node = np.random.choice(candidates_nodes, p=normalize(candidates_nodes_weights))
        return node

    def jump_to_node(self, pred_intent, intent_idx, curr_node):
        """
        Jump to a node based on the intent
        """
        logger.info(f"pred_intent in jump_to_node is {pred_intent}")
        try:
            candidates_nodes = [self.intents[pred_intent][intent_idx]]
            candidates_nodes_weights = [node["attribute"]["weight"] for node in candidates_nodes]
            if candidates_nodes:
                next_node = np.random.choice([node["target_node"] for node in candidates_nodes], p=normalize(candidates_nodes_weights))
                next_intent = pred_intent
            else:  # This is for protection, logically shouldn't enter this branch
                next_node = curr_node
                next_intent = list(self.graph.in_edges(curr_node, data="intent"))[0][2]
        except Exception as e:
            logger.error(f"Error in jump_to_node: {e}")
            next_node = curr_node
            next_intent = list(self.graph.in_edges(curr_node, data="intent"))[0][2]
        return next_node, next_intent
    
    def _get_node(self, sample_node, params: Params, intent=None) -> Tuple[NodeInfo, Params]:
        """
        Get the output format (NodeInfo, Params) that get_node should return
        """
        logger.info(f"available_intents in _get_node: {params.taskgraph.available_global_intents}")
        logger.info(f"intent in _get_node: {intent}")
        node_info = self.graph.nodes[sample_node]
        resource_name = node_info["resource"]["name"]
        resource_id = node_info["resource"]["id"]
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
            additional_args={"tags": node_info["attribute"].get("tags", {})}
        )
        
        return node_info, params

    def _postprocess_intent(self, pred_intent, available_global_intents):
        found_pred_in_avil = False
        real_intent = pred_intent
        idx = 0
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

    def get_current_node(self, params: Params):
        """
        Get current node
        If current node is unknown, use start node
        """
        curr_node = params.taskgraph.curr_node
        if not curr_node or curr_node not in self.graph.nodes:
            curr_node = self.start_node
        else:
            curr_node = str(curr_node)
        params.taskgraph.curr_node = curr_node
        return curr_node, params
    
    def get_available_global_intents(self, params: Params):
        """
        Get available global intents
        """
        available_global_intents = params.taskgraph.available_global_intents
        if not available_global_intents:
            available_global_intents = copy.deepcopy(self.intents)
            if self.unsure_intent.get("intent") not in available_global_intents.keys():
                available_global_intents[self.unsure_intent.get("intent")].append(self.unsure_intent)
        logger.info(f"Available global intents: {available_global_intents}")
        return available_global_intents
    
    def update_node_limit(self, params: Params):
        """
        Update the node_limit in params which will be used to check if we can skip the node or not
        """
        old_node_limit = params.taskgraph.node_limit
        node_limit = {}
        for node in self.graph.nodes.data():
            limit = old_node_limit.get(node[0], node[1].get("limit")) 
            if limit is not None:
                node_limit[node[0]] = limit
        params.taskgraph.node_limit = node_limit
        return params

    def get_local_intent(self, curr_node, params: Params):
        """
        Get the local intent of a current node
        """
        candidates_intents = collections.defaultdict(list)
        for u, v, data in self.graph.out_edges(curr_node, data=True):
            intent = data.get("intent")
            if intent != "none" and data.get("intent"):
                edge_info = copy.deepcopy(data)
                edge_info["source_node"] = u
                edge_info["target_node"] = v
                candidates_intents[intent].append(edge_info)
        logger.info(f"Current local intent: {candidates_intents}")
        return dict(candidates_intents)

    def get_last_flow_stack_node(self, params: Params) -> PathNode:
        """
        Get the last flow stack node from path
        """
        path = params.taskgraph.path
        for i in range(len(path) - 1, -1, -1):
            if path[i].in_flow_stack:
                path[i].in_flow_stack = False
                return path[i]
        return None
    
    def handle_multi_step_node(self, curr_node, params: Params) -> Tuple[bool, NodeInfo, Params]:
        """
        In case of a node having status == STAY, returned directly the same node
        """
        node_status = params.taskgraph.node_status
        logger.info(f"node_status: {node_status}")
        status = node_status.get(curr_node, StatusEnum.COMPLETE)
        if status == StatusEnum.STAY:
            node_info = self.graph.nodes[curr_node]
            resource_name = node_info["resource"]["name"]
            resource_id = node_info["resource"]["id"]
            node_info = NodeInfo(
                type=node_info.get("type", ""),
                node_id=curr_node,
                resource_id = resource_id,
                resource_name = resource_name,
                can_skipped=False,
                is_leaf=len(list(self.graph.successors(curr_node))) == 0,
                attributes=node_info["attribute"],
                additional_args={"tags": node_info["attribute"].get("tags", {})}
            )
            return True, node_info, params
        return False, NodeInfo(), params
    
    def handle_incomplete_node(self, curr_node: str, params: Params) -> Tuple[bool, dict, Params]:
        """
        If node is incomplete, return directly the node
        """
        node_status = params.taskgraph.node_status
        status = node_status.get(curr_node, StatusEnum.COMPLETE)
        if status == StatusEnum.INCOMPLETE:
            logger.info(f"no local or global intent found, the current node is not complete")
            node_info, params = self._get_node(curr_node, params)
            return True, node_info, params
        return False, {}, params
    
    def global_intent_prediction(self, curr_node, params: Params, available_global_intents, excluded_intents) -> Tuple[bool, str, dict, Params]:
        """
        Do global intent prediction
        """
        candidate_intents = copy.deepcopy(available_global_intents)
        candidate_intents = {k: v for k, v in candidate_intents.items() if k not in excluded_intents}
        pred_intent = None
        # if only unsure_intent is available -> move directly to this intent
        if len(candidate_intents) == 1 and self.unsure_intent.get("intent") in candidate_intents.keys():
            pred_intent = self.unsure_intent.get("intent")
        else: # global intent prediction
            # if match other intent, add flow, jump over
            candidate_intents[self.unsure_intent.get("intent")] = \
                candidate_intents.get(self.unsure_intent.get("intent"), [self.unsure_intent])
            logger.info(f"Available global intents with unsure intent: {candidate_intents}")
            
            pred_intent = self.nluapi.execute(self.text, candidate_intents, self.chat_history_str, self.llm_config.model_dump())
            params.taskgraph.nlu_records.append({"candidate_intents": candidate_intents, 
                                "pred_intent": pred_intent, "no_intent": False, "global_intent": True})
            found_pred_in_avil, pred_intent, intent_idx = self._postprocess_intent(pred_intent, available_global_intents)
            # if found prediction and prediction is not unsure intent and current intent
            if found_pred_in_avil and \
                pred_intent != self.unsure_intent.get("intent"):
                # If the prediction is the same as the current global intent and the current node is not a leaf node, continue the current global intent
                if pred_intent == params.taskgraph.curr_global_intent and len(list(self.graph.successors(curr_node))) != 0:
                    return False, pred_intent, {}, params
                next_node, next_intent = self.jump_to_node(pred_intent, intent_idx, curr_node)
                logger.info(f"curr_node: {next_node}")
                node_info, params = self._get_node(next_node, params, intent=next_intent)
                # if current node is not a leaf node and jump to another node, then add it onto stack
                if next_node != curr_node and list(self.graph.successors(curr_node)):
                    node_info.add_flow_stack = True
                params.taskgraph.curr_global_intent = pred_intent
                return True, pred_intent, node_info, params
        return False, pred_intent, {}, params
 
    def handle_random_next_node(self, curr_node, params: Params) -> Tuple[bool, dict, Params]:
        candidate_samples = []
        candidates_nodes_weights = []
        for out_edge in self.graph.out_edges(curr_node, data=True):
            if out_edge[2]["intent"] == "none":
                candidate_samples.append(out_edge[1])
                candidates_nodes_weights.append(out_edge[2]["attribute"]["weight"])
        if candidate_samples:
            # randomly choose one sample from candidate samples
            next_node = np.random.choice(candidate_samples, p=normalize(candidates_nodes_weights))
        else:  # leaf node + the node without None intents
            next_node = curr_node

        if next_node != curr_node:  # continue if curr_node is not leaf node, i.e. there is a actual next_node
            logger.info(f"curr_node: {next_node}")
            node_info, params = self._get_node(next_node, params)
            if params.taskgraph.nlu_records:
                params.taskgraph.nlu_records[-1]["no_intent"] = True  # move on to the next node
            else: # only others available
                params.taskgraph.nlu_records = [{"candidate_intents": [], "pred_intent": "", "no_intent": True, "global_intent": False}]
            return True, node_info, params
        return False, {}, params
    
    def local_intent_prediction(self, curr_node, params: Params, curr_local_intents) -> Tuple[bool, dict, Params]:
        """
        Do local intent prediction
        """
        curr_local_intents_w_unsure = copy.deepcopy(curr_local_intents)
        curr_local_intents_w_unsure[self.unsure_intent.get("intent")] = \
            curr_local_intents_w_unsure.get(self.unsure_intent.get("intent"), [self.unsure_intent])
        logger.info(f"Check intent under current node: {curr_local_intents_w_unsure}")
        pred_intent = self.nluapi.execute(self.text, curr_local_intents_w_unsure, self.chat_history_str, self.llm_config.model_dump())
        params.taskgraph.nlu_records.append({"candidate_intents": curr_local_intents_w_unsure, 
                                "pred_intent": pred_intent, "no_intent": False, "global_intent": False})
        found_pred_in_avil, pred_intent, intent_idx = self._postprocess_intent(pred_intent, curr_local_intents)
        logger.info(f"Local intent predition -> found_pred_in_avil: {found_pred_in_avil}, pred_intent: {pred_intent}")
        if found_pred_in_avil:
            params.taskgraph.intent = pred_intent
            for edge in self.graph.out_edges(curr_node, data="intent"):
                if edge[2] == pred_intent:
                    next_node = edge[1]  # found intent under the current node
                    break
            logger.info(f"curr_node: {next_node}")
            node_info, params = self._get_node(next_node, params, intent=pred_intent)
            if curr_node == self.start_node:
                params.taskgraph.curr_global_intent = pred_intent
            return True, node_info, params
        return False, {}, params
    
    def handle_unknown_intent(self, curr_node, params: Params) -> Tuple[dict, Params]:
        """
        If unknown intent, call planner
        """
        # if none of the available intents can represent user's utterance, transfer to the planner to let it decide for the next step
        params.taskgraph.intent = self.unsure_intent.get("intent")
        params.taskgraph.curr_global_intent = ""
        if params.taskgraph.nlu_records:
            params.taskgraph.nlu_records[-1]["no_intent"] = True  # no intent found
        else:
            params.taskgraph.nlu_records.append({"candidate_intents": [], "pred_intent": "", "no_intent": True, "global_intent": False})
        params.taskgraph.curr_node = curr_node
        node_info = NodeInfo(
            node_id=None,
            type="",
            resource_id = "planner",
            resource_name = "planner",
            can_skipped=False,
            is_leaf=len(list(self.graph.successors(curr_node))) == 0,
            attributes = {"value": "", "direct": False},
            additional_args = {"tags": {}}
        )
        return node_info, params
        
    def handle_leaf_node(self, curr_node, params: Params):
        '''
        if leaf node, first check if it's in a nested graph
        if not in nested graph, check if we have flow stack
        '''
        def is_leaf(node):
            return len(list(self.graph.successors(node))) == 0
        
        # if not leaf, return directly current node
        if not is_leaf(curr_node):
            return curr_node, params
        
        
        nested_graph_next_node, params = NestedGraph.get_nested_graph_component_node(params, is_leaf)
        if nested_graph_next_node is not None:
            curr_node = nested_graph_next_node.node_id
            params.taskgraph.curr_node = curr_node
            if not is_leaf(nested_graph_next_node.node_id):
                return curr_node, params
        
        last_flow_stack_node: PathNode = self.get_last_flow_stack_node(params)
        if last_flow_stack_node:
            curr_node = last_flow_stack_node.node_id
            params.taskgraph.curr_global_intent = last_flow_stack_node.global_intent
        if self.initial_node:
            curr_node = self.initial_node
        
        return curr_node, params

    def get_node(self, inputs):
        """
        Get the next node
        """
        self.text = inputs["text"]
        self.chat_history_str = inputs["chat_history_str"]
        params: Params = inputs["parameters"]
        # boolean to check if we allow global intent switch or not.
        allow_global_intent_switch = inputs["allow_global_intent_switch"]
        params.taskgraph.nlu_records = []

        curr_node, params = self.get_current_node(params)
        logger.info(f"Intial curr_node: {curr_node}")

        # For the multi-step nodes, directly stay at that node instead of moving to other nodes
        is_multi_step_node, node_output, params = self.handle_multi_step_node(curr_node, params)
        if is_multi_step_node:
            return node_output, params
        
        curr_node, params = self.handle_leaf_node(curr_node, params)
        
        
        # store current node
        params.taskgraph.curr_node = curr_node
        logger.info(f"curr_node: {curr_node}")

        # available global intents
        available_global_intents = self.get_available_global_intents(params)
        
        # update limit
        params = self.update_node_limit(params)

        # Get local intents of the curr_node
        curr_local_intents = self.get_local_intent(curr_node, params)

        if not curr_local_intents and allow_global_intent_switch:  # no local intent under the current node
            logger.info(f"no local intent under the current node")
            is_global_intent_found, _, node_output, params = \
                self.global_intent_prediction(
                    curr_node,
                    params,
                    available_global_intents,
                    {}
                )
            if is_global_intent_found:
                return node_output, params

        # if current node is incompleted -> return current node
        is_incomplete_node, node_output, params = self.handle_incomplete_node(curr_node, params)
        if is_incomplete_node:
            return node_output, params
        
        # if completed and no local intents -> randomly choose one of the next connected nodes (edges with intent = None)
        if not curr_local_intents:
            logger.info(f"no local or global intent found, move to the next connected node(s)")
            has_random_next_node, node_output, params = self.handle_random_next_node(curr_node, params)
            if has_random_next_node:
                return node_output, params

        logger.info("Finish global condition, start local intent prediction")
        is_local_intent_found, node_output, params = self.local_intent_prediction(curr_node, params, curr_local_intents)
        if is_local_intent_found:
            return node_output, params
        
        pred_intent = None
        if allow_global_intent_switch:
            is_global_intent_found, pred_intent, node_output, params = \
                    self.global_intent_prediction(
                        curr_node,
                        params,
                        available_global_intents,
                        {**curr_local_intents, **{"none": None}}
                    )
            if is_global_intent_found: 
                return node_output, params
        if pred_intent and pred_intent != self.unsure_intent.get("intent"): # if not unsure intent
            # If user didn't indicate all the intent of children nodes under the current node, 
            # then we could randomly choose one of Nones to continue the dialog flow
            has_random_next_node, node_output, params = self.handle_random_next_node(curr_node, params)
            if has_random_next_node:
                return node_output, params
            
        # if none of the available intents can represent user's utterance or it is an unsure intents,
        # transfer to the planner to let it decide for the next step
        node_output, params = self.handle_unknown_intent(curr_node, params)
        return node_output, params
        

    def postprocess_node(self, node) -> Tuple[NodeInfo, Params]:
        node_info: NodeInfo = node[0]
        params: Params = node[1]
        # TODO: future node postprocessing
        return node_info, params