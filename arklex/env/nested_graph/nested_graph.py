from typing import List, Tuple, Optional, Callable, Any
from arklex.utils.graph_state import NodeInfo, Params, PathNode, StatusEnum


NESTED_GRAPH_ID = "nested_graph"


class NestedGraph:
    def __init__(self, node_info: NodeInfo):
        """
        Initialize a NestedGraph instance with nested graph resource node information.

        Args:
            node_info (NodeInfo): The node information containing attributes relevant to the graph.
        """
        self.node_info: NodeInfo = (
            node_info  # Stores the node information for the nested graph
        )

    def get_nested_graph_start_node_id(self) -> str:
        """
        Retrieve the starting node identifier for the nested graph.

        Returns:
            str: The start node ID derived from the 'value' attribute of node_info.
        """
        return str(
            self.node_info.attributes["value"]
        )  # Returns the start node ID as a string

    @staticmethod
    def get_nested_graph_component_node(
        params: Params, is_leaf_func: Callable[[Any], bool]
    ) -> Tuple[PathNode | None, Params]:
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
            path: List[PathNode] = (
                params.taskgraph.path
            )  # List of PathNode objects representing the current path
            cur_node_i: int = node_i  # Current node index in the path
            prev_node_id: Optional[str] = None  # Previous node ID, initialized to None
            while cur_node_i >= 0:
                if path[cur_node_i].nested_graph_leaf_jump is not None:
                    cur_node_i = (
                        path[cur_node_i].nested_graph_leaf_jump
                    )  # Jump to the nested graph leaf node if specified

                cur_node: PathNode = path[cur_node_i]  # Current node in the path
                cur_node_id: str = cur_node.node_id  # ID of the current node
                to_node_id: Optional[str] = (
                    cur_node.nested_graph_node_value
                )  # Value indicating the nested graph node
                if to_node_id is not None and to_node_id == prev_node_id:
                    params.taskgraph.node_status[cur_node_id] = (
                        StatusEnum.COMPLETE
                    )  # Mark the node as complete
                    params.taskgraph.path[
                        node_i
                    ].nested_graph_leaf_jump = cur_node_i  # Update the leaf jump
                    return (
                        cur_node_i,
                        params,
                    )  # Return the current node index and updated params
                prev_node_id = cur_node_id  # Update the previous node ID
                cur_node_i -= 1  # Move to the previous node

            return -1, params  # Return -1 if no nested graph component is found

        path: List[PathNode] = (
            params.taskgraph.path
        )  # List of PathNode objects representing the current path
        cur_node_i: int = len(path) - 1  # Start from the last node in the path
        while cur_node_i >= 0:
            nested_graph_next_node_path_i: int
            nested_graph_next_node_path_i, params = _get_nested_graph_component_node(
                cur_node_i, params
            )  # Get the next nested graph component node
            if nested_graph_next_node_path_i == -1:
                path_node: PathNode = path[cur_node_i]  # Current node in the path
                return (
                    path_node,
                    params,
                )  # Return the current node and params if no nested graph component is found
            path_node: PathNode = path[
                nested_graph_next_node_path_i
            ]  # Nested graph component node
            node_id: str = path_node.node_id  # ID of the nested graph component node
            if not is_leaf_func(node_id):
                return (
                    path_node,
                    params,
                )  # Return the nested graph component node if it's not a leaf
            cur_node_i = nested_graph_next_node_path_i  # Update the current node index
        # None should never be returned
        # if in main graph, current node should be returned as _get_nested_graph_component_node will return -1
        return None, params  # Return None and params if no valid node is found
