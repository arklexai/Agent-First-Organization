"""Task graph formatting component for the Arklex framework.

This module provides the TaskGraphFormatter class that handles formatting and
structuring of task graphs. It ensures consistent graph representation and
proper visualization of task relationships.

Key Features:
- Graph structure formatting
- Node formatting
- Edge formatting
- Error handling
"""

from typing import Dict, Any, List, Optional

from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class TaskGraphFormatter:
    """Formats task graphs.

    This class handles formatting of task graphs, ensuring they are properly
    structured and contain all required components.

    Attributes:
        _role (str): Role of the assistant
        _user_objective (str): User's objective
        _builder_objective (str): Builder's objective
        _domain (str): Domain of the task
        _intro (str): Introduction text
        _task_docs (List[Dict[str, Any]]): Task documentation
        _rag_docs (List[Dict[str, Any]]): RAG documentation
        _workers (List[Dict[str, Any]]): List of workers
    """

    def __init__(
        self,
        role: str = "",
        user_objective: str = "",
        builder_objective: str = "",
        domain: str = "",
        intro: str = "",
        task_docs: Optional[List[Dict[str, Any]]] = None,
        rag_docs: Optional[List[Dict[str, Any]]] = None,
        workers: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        nluapi: str = "",
        slotfillapi: str = "",
        default_intent: str = "depends_on",
        default_weight: int = 1,
        nodes: Optional[List[Any]] = None,
        edges: Optional[List[Any]] = None,
        allow_nested_graph: bool = True,
    ) -> None:
        """Initialize the TaskGraphFormatter.

        Args:
            role (str): Role of the assistant
            user_objective (str): User's objective
            builder_objective (str): Builder's objective
            domain (str): Domain of the task
            intro (str): Introduction text
            task_docs (Optional[List[Dict[str, Any]]]): Task documentation
            rag_docs (Optional[List[Dict[str, Any]]]): RAG documentation
            workers (Optional[List[Dict[str, Any]]]): List of workers
            tools (Optional[List[Dict[str, Any]]]): List of tools
            nluapi (str): NLU API
            slotfillapi (str): Slotfill API
            default_intent (str): Default intent for edges
            default_weight (int): Default weight for edges
            nodes (Optional[List[Any]]): List of nodes
            edges (Optional[List[Any]]): List of edges
            allow_nested_graph (bool): Whether to allow nested graph generation
        """
        self._role = role
        self._user_objective = user_objective
        self._builder_objective = builder_objective
        self._domain = domain
        self._intro = intro
        self._task_docs = task_docs or []
        self._rag_docs = rag_docs or []
        self._workers = workers or []
        self._tools = tools or []
        self._nluapi = nluapi
        self._slotfillapi = slotfillapi
        self._default_intent = default_intent
        self._default_weight = default_weight
        self._nodes = nodes
        self._edges = edges
        self._allow_nested_graph = allow_nested_graph

    def _find_worker_id_by_name(self, worker_name: str) -> str:
        """Find the actual worker ID by name from the config.

        Args:
            worker_name (str): The name of the worker to find

        Returns:
            str: The worker ID if found, otherwise the worker name as fallback
        """
        if worker_name == "NestedGraph":
            return "nested_graph"

        if self._workers:
            for worker in self._workers:
                if isinstance(worker, dict) and worker.get("name") == worker_name:
                    return worker.get("id", worker_name)
        return worker_name

    def format_task_graph(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format a task graph using config nodes/edges if present, otherwise generate from tasks."""
        if self._nodes is not None and self._edges is not None:
            graph = {
                "nodes": self._nodes,
                "edges": self._edges,
                "role": self._role,
                "user_objective": self._user_objective,
                "builder_objective": self._builder_objective,
                "domain": self._domain,
                "intro": self._intro,
                "task_docs": self._task_docs,
                "rag_docs": self._rag_docs,
                "tasks": tasks,
                "workers": self._workers,
                "tools": self._tools,
                "nluapi": self._nluapi,
                "slotfillapi": self._slotfillapi,
            }
            return graph

        nodes = []
        edges = []
        node_id_counter = 0
        node_lookup = {}  # task_id/step_id -> node_id (str)
        step_parent_lookup = {}  # step node_id -> parent task node_id
        step_nodes = []
        task_node_mapping = {}  # task index -> node_id

        # Create start node first
        message_worker_id = self._find_worker_id_by_name("MessageWorker")
        start_node = {
            "resource": {
                "id": message_worker_id,
                "name": "MessageWorker",
            },
            "attribute": {
                "value": "Hello! I'm here to assist you with any customer service inquiries you may have. Whether you need information about our products, services, or policies, or if you need help resolving an issue or completing a transaction, feel free to ask. How can I assist you today?",
                "task": "start message",
                "directed": False,
            },
            "type": "start",
        }
        nodes.append([str(node_id_counter), start_node])
        start_node_id = str(node_id_counter)
        node_id_counter += 1

        # Create all task nodes and collect their IDs
        task_node_mapping = {}
        node_lookup = {}
        step_parent_lookup = {}
        step_nodes = []
        all_task_node_ids = []
        nested_graph_node_id = None

        if tasks:
            for task_idx, task in enumerate(tasks):
                resource = task.get("resource", {})
                resource_name = resource.get("name", "MessageWorker")
                resource_id = self._find_worker_id_by_name(resource_name)
                node = {
                    "resource": {
                        "id": resource_id,
                        "name": resource_name,
                    },
                    "attribute": {
                        "value": task.get("description", ""),
                        "task": task.get("name", ""),
                        "directed": False,
                    },
                }
                if "limit" in task:
                    node["limit"] = task["limit"]
                if "type" in task:
                    node["type"] = task["type"]
                nodes.append([str(node_id_counter), node])
                all_task_node_ids.append(str(node_id_counter))
                task_identifier = task.get("task_id", f"task_{task_idx}")
                node_lookup[task_identifier] = str(node_id_counter)
                task_node_mapping[task_idx] = str(node_id_counter)
                node_id_counter += 1

            # Only create nested graph node if allow_nested_graph is True
            if self._allow_nested_graph:
                # Find all nodes that are the target of an edge from the main graph start node
                main_graph_targets = set()
                for task_idx, task in enumerate(tasks):
                    dependencies = task.get("dependencies", [])
                    if not dependencies:
                        main_graph_targets.add(task_node_mapping[task_idx])

                # Subgraph start nodes: task nodes that are not the main graph start node and not directly targeted by the main graph start node
                subgraph_start_nodes = [
                    node_id
                    for node_id in all_task_node_ids
                    if node_id != start_node_id and node_id not in main_graph_targets
                ]
                # Fallback: if all task nodes are main graph targets, just use the first task node that is not the start node
                if not subgraph_start_nodes:
                    subgraph_start_nodes = [
                        node_id
                        for node_id in all_task_node_ids
                        if node_id != start_node_id
                    ]
                # Use the first valid subgraph start node
                nested_graph_value = (
                    subgraph_start_nodes[0]
                    if subgraph_start_nodes
                    else all_task_node_ids[0]
                )

                # Now create the nested graph node, value set to a true subgraph start node
                nested_graph_node = {
                    "resource": {
                        "id": "nested_graph",
                        "name": "NestedGraph",
                    },
                    "attribute": {
                        "value": nested_graph_value,
                        "task": "TBD",
                        "directed": True,
                    },
                    "limit": 1,
                }
                nested_graph_node_id = str(node_id_counter)
                nodes.append([nested_graph_node_id, nested_graph_node])
                node_id_counter += 1
        else:
            start_node_id = None
            nested_graph_node_id = None
            node_id_counter = 0

        # Create nodes for steps
        for task_idx, task in enumerate(tasks):
            steps = task.get("steps", [])
            task_identifier = task.get("task_id", f"task_{task_idx}")

            for idx, step in enumerate(steps):
                step_id = f"{task_identifier}_step{idx}"
                resource = step.get("resource", {})

                if isinstance(resource, str):
                    resource_id = self._find_worker_id_by_name(resource)
                    resource_name = resource
                else:
                    resource_name = resource.get("name", "MessageWorker")
                    resource_id = self._find_worker_id_by_name(resource_name)

                step_node = {
                    "resource": {
                        "id": resource_id,
                        "name": resource_name,
                    },
                    "attribute": {
                        "value": step.get("description", step.get("value", "")),
                        "task": step.get("name", step.get("task", "")),
                        "directed": False,
                    },
                }
                nodes.append([str(node_id_counter), step_node])
                node_lookup[step_id] = str(node_id_counter)
                step_parent_lookup[str(node_id_counter)] = task_node_mapping[task_idx]
                step_nodes.append(
                    (str(node_id_counter), step_id, task_node_mapping[task_idx])
                )
                node_id_counter += 1

        # Create edges for dependencies (task-to-task)
        for task_idx, task in enumerate(tasks):
            this_node_id = task_node_mapping[task_idx]
            task_identifier = task.get("task_id", f"task_{task_idx}")
            dependencies = task.get("dependencies", [])
            if dependencies:
                for dep in dependencies:
                    if dep in node_lookup:
                        edge_data = {
                            "intent": "depends_on",
                            "attribute": {
                                "weight": 1,
                                "pred": True,
                                "definition": f"{task.get('name', '')} depends on {dep}",
                                "sample_utterances": [],
                            },
                        }
                        edges.append([node_lookup[dep], this_node_id, edge_data])
            elif start_node_id is not None:
                edge_data = {
                    "intent": f"User inquires about {task.get('name', '').lower()}",
                    "attribute": {
                        "weight": 1,
                        "pred": True,
                        "definition": "",
                        "sample_utterances": [],
                    },
                }
                edges.append([start_node_id, this_node_id, edge_data])

        # Create edges for task-to-step
        for task_idx, task in enumerate(tasks):
            steps = task.get("steps", [])
            if not steps:
                continue

            task_identifier = task.get("task_id", f"task_{task_idx}")

            first_step_node_id = node_lookup[f"{task_identifier}_step0"]
            edge_data = {
                "intent": None,
                "attribute": {
                    "weight": 1,
                    "pred": False,
                    "definition": "",
                    "sample_utterances": [],
                },
            }
            edges.append([task_node_mapping[task_idx], first_step_node_id, edge_data])

            for i in range(len(steps) - 1):
                current_step_id = f"{task_identifier}_step{i}"
                next_step_id = f"{task_identifier}_step{i + 1}"
                current_step_node_id = node_lookup[current_step_id]
                next_step_node_id = node_lookup[next_step_id]

                edge_data = {
                    "intent": None,
                    "attribute": {
                        "weight": 1,
                        "pred": False,
                        "definition": "",
                        "sample_utterances": [],
                    },
                }
                edges.append([current_step_node_id, next_step_node_id, edge_data])

        # Add edges from nested_graph to leaf nodes
        if tasks:
            all_node_ids = set(str(n[0]) for n in nodes)
            source_node_ids = set(str(e[0]) for e in edges)
            leaf_node_ids = [
                nid
                for nid in all_node_ids
                if nid not in source_node_ids and nid != nested_graph_node_id
            ]

            if not leaf_node_ids:
                task_node_ids = [
                    nid
                    for nid in all_node_ids
                    if nid not in (start_node_id, nested_graph_node_id)
                ]
                if len(task_node_ids) == 1:
                    leaf_node_ids = task_node_ids

            for leaf_id in leaf_node_ids:
                nested_graph_to_leaf_edge = [
                    nested_graph_node_id,
                    leaf_id,
                    {
                        "intent": None,
                        "attribute": {
                            "weight": 1,
                            "pred": False,
                            "definition": "",
                            "sample_utterances": [],
                        },
                    },
                ]
                edges.append(nested_graph_to_leaf_edge)

        graph = {
            "nodes": nodes,
            "edges": edges,
            "role": self._role,
            "user_objective": self._user_objective,
            "builder_objective": self._builder_objective,
            "domain": self._domain,
            "intro": self._intro,
            "task_docs": self._task_docs,
            "rag_docs": self._rag_docs,
            "tasks": tasks,
            "workers": self._workers,
            "tools": self._tools,
            "nluapi": self._nluapi,
            "slotfillapi": self._slotfillapi,
        }
        return graph

    def link_main_graph_to_nested_graph(
        self, main_graph: Dict[str, Any], nested_graph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Link the main graph to the nested graph following the original codebase logic.
        This function:
        1. Links the main graph to the nested graph
        2. Removes any edge from the nested graph to node 0 (main graph start node)
        3. Adds edges from the nested graph to each subgraph start node (excluding main graph start node)
        """
        # Create a copy of the main graph to avoid modifying the original
        combined_graph = {
            "nodes": main_graph.get("nodes", []).copy(),
            "edges": main_graph.get("edges", []).copy(),
            "role": main_graph.get("role", ""),
            "user_objective": main_graph.get("user_objective", ""),
            "builder_objective": main_graph.get("builder_objective", ""),
            "domain": main_graph.get("domain", ""),
            "intro": main_graph.get("intro", ""),
            "task_docs": main_graph.get("task_docs", []),
            "rag_docs": main_graph.get("rag_docs", []),
            "tasks": main_graph.get("tasks", []),
            "workers": main_graph.get("workers", []),
            "tools": main_graph.get("tools", []),
            "nluapi": main_graph.get("nluapi", ""),
            "slotfillapi": main_graph.get("slotfillapi", ""),
        }

        # Find the main graph's start node id (usually node '0')
        start_node_id = None
        for node_id, node_data in combined_graph["nodes"]:
            if node_data.get("resource", {}).get("id") == "MessageWorker":
                start_node_id = node_id
                break

        # Find the nested graph node in the main graph
        nested_graph_node_id = None
        for node_id, node_data in combined_graph["nodes"]:
            if node_data.get("resource", {}).get("id") == "nested_graph":
                nested_graph_node_id = node_id
                break

        if nested_graph_node_id is None:
            # If no nested graph node exists, create one
            node_id_counter = len(combined_graph["nodes"])
            nested_graph_node = {
                "resource": {
                    "id": "nested_graph",
                    "name": "NestedGraph",
                },
                "attribute": {
                    "value": "placeholder",  # Will be updated to first task node
                    "task": "TBD",
                    "directed": True,
                },
                "limit": 1,
            }
            combined_graph["nodes"].append([str(node_id_counter), nested_graph_node])
            nested_graph_node_id = str(node_id_counter)

        # Get nested graph nodes and edges
        nested_nodes = nested_graph.get("nodes", [])
        nested_edges = nested_graph.get("edges", [])

        # Find the start node of the nested graph (node 0 or the first node)
        nested_start_node_id = None
        for node_id, node_data in nested_nodes:
            if node_id == "0":
                nested_start_node_id = node_id
                break
        if nested_start_node_id is None and nested_nodes:
            nested_start_node_id = nested_nodes[0][0]

        # Update the nested graph node's value to point to the nested graph start
        # But only if it's not pointing to the main graph start node
        if nested_start_node_id and nested_start_node_id != start_node_id:
            for node_id, node_data in combined_graph["nodes"]:
                if node_id == nested_graph_node_id:
                    node_data["attribute"]["value"] = nested_start_node_id
                    node_data["attribute"]["task"] = "TBD"
                    node_data["attribute"]["directed"] = True
                    break
        else:
            # If no valid nested start found, point to first task node
            task_node_ids = [
                node_id
                for node_id, node_data in combined_graph["nodes"]
                if node_data.get("resource", {}).get("id") == "MessageWorker"
                and node_id not in (start_node_id, nested_graph_node_id)
            ]
            if task_node_ids:
                first_task_id = task_node_ids[0]
                for node_id, node_data in combined_graph["nodes"]:
                    if node_id == nested_graph_node_id:
                        node_data["attribute"]["value"] = first_task_id
                        node_data["attribute"]["task"] = "TBD"
                        node_data["attribute"]["directed"] = True
                        break

        # Remove any existing edges from nested_graph to node 0 (main graph start node)
        combined_graph["edges"] = [
            edge
            for edge in combined_graph["edges"]
            if not (edge[0] == nested_graph_node_id and edge[1] == start_node_id)
        ]

        # Find subgraph start nodes (nodes with no incoming edges in the nested graph)
        nested_node_ids = {node_id for node_id, _ in nested_nodes}
        nested_target_node_ids = {edge[1] for edge in nested_edges}
        subgraph_start_nodes = [
            node_id
            for node_id in nested_node_ids
            if node_id not in nested_target_node_ids
        ]

        # Exclude the main graph's start node from subgraph start nodes
        subgraph_start_nodes = [
            node_id for node_id in subgraph_start_nodes if node_id != start_node_id
        ]

        # If no clear start nodes found, use all nodes that are not the nested graph node or main graph start node
        if not subgraph_start_nodes:
            subgraph_start_nodes = [
                node_id
                for node_id in nested_node_ids
                if node_id != nested_graph_node_id and node_id != start_node_id
            ]

        # Add edges from the nested graph to each subgraph start node (excluding main graph start node)
        for start_id in subgraph_start_nodes:
            nested_graph_to_start_edge = [
                nested_graph_node_id,
                start_id,
                {
                    "intent": None,
                    "attribute": {
                        "weight": 1,
                        "pred": False,
                        "definition": f"Nested graph connects to subgraph start node {start_id}",
                        "sample_utterances": [],
                    },
                },
            ]
            combined_graph["edges"].append(nested_graph_to_start_edge)

        return combined_graph

    def ensure_nested_graph_connectivity(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure the nested graph is properly connected following the original codebase logic.

        This function ensures:
        1. The nested graph has incoming edges from the main graph
        2. The nested graph has outgoing edges to subgraph start nodes
        3. No orphaned edges to node 0
        4. Proper connectivity for graph traversal

        Args:
            graph (Dict[str, Any]): The graph structure to validate and fix

        Returns:
            Dict[str, Any]: The graph with proper nested graph connectivity
        """
        # Create a copy to avoid modifying the original
        fixed_graph = {
            "nodes": graph.get("nodes", []).copy(),
            "edges": graph.get("edges", []).copy(),
            "role": graph.get("role", ""),
            "user_objective": graph.get("user_objective", ""),
            "builder_objective": graph.get("builder_objective", ""),
            "domain": graph.get("domain", ""),
            "intro": graph.get("intro", ""),
            "task_docs": graph.get("task_docs", []),
            "rag_docs": graph.get("rag_docs", []),
            "tasks": graph.get("tasks", []),
            "workers": graph.get("workers", []),
            "tools": graph.get("tools", []),
            "nluapi": graph.get("nluapi", ""),
            "slotfillapi": graph.get("slotfillapi", ""),
        }

        # Find the nested graph node
        nested_graph_node_id = None
        start_node_id = None

        for node_id, node_data in fixed_graph["nodes"]:
            if node_data.get("resource", {}).get("id") == "nested_graph":
                nested_graph_node_id = node_id
            elif node_data.get("resource", {}).get("id") == "MessageWorker":
                start_node_id = node_id

        if nested_graph_node_id is None:
            # No nested graph node found, return original graph
            return fixed_graph

        # Get all node IDs and edge information
        all_node_ids = {node_id for node_id, _ in fixed_graph["nodes"]}
        source_node_ids = {edge[0] for edge in fixed_graph["edges"]}
        target_node_ids = {edge[1] for edge in fixed_graph["edges"]}

        # Remove any edges from nested_graph to node 0
        fixed_graph["edges"] = [
            edge
            for edge in fixed_graph["edges"]
            if not (edge[0] == nested_graph_node_id and edge[1] == "0")
        ]

        # Find leaf nodes (nodes with no outgoing edges, excluding nested_graph)
        leaf_node_ids = [
            node_id
            for node_id in all_node_ids
            if node_id not in source_node_ids and node_id != nested_graph_node_id
        ]

        # If no leaf nodes found, use task nodes as fallback
        if not leaf_node_ids:
            task_node_ids = [
                node_id
                for node_id in all_node_ids
                if node_id not in (start_node_id, nested_graph_node_id)
            ]
            if len(task_node_ids) == 1:
                leaf_node_ids = task_node_ids

        # Ensure nested graph has incoming edges from main graph
        nested_graph_has_incoming = any(
            edge[1] == nested_graph_node_id for edge in fixed_graph["edges"]
        )

        if not nested_graph_has_incoming and start_node_id:
            # Add edge from start node to nested graph
            start_to_nested_edge = [
                start_node_id,
                nested_graph_node_id,
                {
                    "intent": "Navigate to nested graph",
                    "attribute": {
                        "weight": 1,
                        "pred": True,
                        "definition": "Start node connects to nested graph",
                        "sample_utterances": [],
                    },
                },
            ]
            fixed_graph["edges"].append(start_to_nested_edge)

        # Ensure nested graph has outgoing edges to leaf nodes
        nested_graph_outgoing = [
            edge for edge in fixed_graph["edges"] if edge[0] == nested_graph_node_id
        ]

        # Remove existing nested graph outgoing edges
        fixed_graph["edges"] = [
            edge for edge in fixed_graph["edges"] if edge[0] != nested_graph_node_id
        ]

        # Add edges from nested graph to all leaf nodes
        for leaf_id in leaf_node_ids:
            nested_graph_to_leaf_edge = [
                nested_graph_node_id,
                leaf_id,
                {
                    "intent": None,
                    "attribute": {
                        "weight": 1,
                        "pred": False,
                        "definition": f"Nested graph connects to leaf node {leaf_id}",
                        "sample_utterances": [],
                    },
                },
            ]
            fixed_graph["edges"].append(nested_graph_to_leaf_edge)

        return fixed_graph
