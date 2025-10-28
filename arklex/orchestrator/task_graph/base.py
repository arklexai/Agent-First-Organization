from typing import Any

import networkx as nx


class GraphBase:
    """Base class for task graph functionality.

    This class provides the fundamental structure and methods for managing task graphs.
    It handles graph creation, intent management, and node traversal.

    Attributes:
        graph (nx.DiGraph): The directed graph representing the task flow
        graph_config (Dict[str, Any]): Configuration settings for the graph
        intents (DefaultDict[str, List[Dict[str, Any]]]): Global intents for node navigation
        start_node (Optional[str]): The initial node in the graph

    Methods:
        create_graph(): Creates the graph structure
        get_pred_intents(): Gets predicted intents from graph edges
        get_start_node(): Gets the starting node of the graph
    """

    def __init__(self, name: str, graph_config: dict[str, Any]) -> None:
        self.graph: nx.DiGraph = nx.DiGraph(name=name)
        self.graph_config: dict[str, Any] = graph_config
        self.create_graph()

    def create_graph(self) -> None:
        raise NotImplementedError
