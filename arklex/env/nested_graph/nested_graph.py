from typing import List, Tuple, Callable, Any, Optional
from arklex.utils.graph_state import NodeInfo, Params, PathNode, StatusEnum


NESTED_GRAPH_ID: str = "nested_graph"


class NestedGraph:
    def __init__(self, node_info: NodeInfo) -> None:
        """
        Initialize a NestedGraph instance with nested graph resource node information.

        Args:
            node_info (NodeInfo): The node information containing attributes relevant to the graph.
        """
        self.node_info: NodeInfo = node_info

    def get_nested_graph_start_node_id(self) -> str:
        """
        Retrieve the starting node identifier for the nested graph.

        Returns:
            str: The start node ID derived from the 'value' attribute of node_info.
        """
        return str(self.node_info.attributes["value"])

    @staticmethod
    def get_nested_graph_component_node(
        params: Params, is_leaf_func: Callable[[Any], bool]
    ) -> Tuple[Optional[PathNode], Params]:
        """
        If in nested subgraph, locate and return the nested graph resource node
        If leaf in main graph, return current node
        1) call _get_nested_graph_component_node to get nested graph resource node given node
        2) if in nested graph component not found (i.e. _get_nested_graph_component_node returns -1)
            current node is in main graph -> return current node
        3) if found nested graph component is not a leaf -> return this node
        4) found nested graph component is a leaf (possibly its also in a nested graph component) -> redo from step 1

        Args:
            params (Params): The parameters object that contains the current task graph, including its node path.
            is_leaf_func (Callable[[Any], bool]): A function that takes a node ID and returns True if the node
                                                    is a leaf node in the nested graph structure, otherwise False.

        Returns:
            Tuple[PathNode | None, Params]: A tuple where:
                - The first element is a PathNode instance representing the nested graph component node,
                  or None if no such node is identified.
                - The second element is the updated params object, which may include modified task graph state.
        """

        def _get_nested_graph_component_node(
            node_i: int, params: Params
        ) -> Tuple[int, Params]:
            """
            if node is in nested graph, return path index of nested graph resource node, params and update path
            if node in main graph, return -1, params
            """
            path: List[PathNode] = params.taskgraph.path
            cur_node_i: int = node_i
            prev_node_id: Optional[str] = None
            while cur_node_i >= 0:
                if path[cur_node_i].nested_graph_leaf_jump is not None:
                    cur_node_i = path[cur_node_i].nested_graph_leaf_jump

                cur_node: PathNode = path[cur_node_i]
                cur_node_id: str = cur_node.node_id
                to_node_id: Optional[str] = cur_node.nested_graph_node_value
                if to_node_id is not None and to_node_id == prev_node_id:
                    params.taskgraph.node_status[cur_node_id] = StatusEnum.COMPLETE
                    params.taskgraph.path[node_i].nested_graph_leaf_jump = cur_node_i
                    return cur_node_i, params
                prev_node_id = cur_node_id
                cur_node_i -= 1

            return -1, params

        path: List[PathNode] = params.taskgraph.path
        cur_node_i: int = len(path) - 1
        while cur_node_i >= 0:
            nested_graph_next_node_path_i: int
            nested_graph_next_node_path_i, params = _get_nested_graph_component_node(
                cur_node_i, params
            )
            if nested_graph_next_node_path_i == -1:
                path_node: PathNode = path[cur_node_i]
                return path_node, params
            path_node: PathNode = path[nested_graph_next_node_path_i]
            node_id: str = path_node.node_id
            if not is_leaf_func(node_id):
                return path_node, params
            cur_node_i = nested_graph_next_node_path_i
        # None should never be returned
        # if in main graph, current node should be returned as _get_nested_graph_component_node will return -1
        return None, params
