"""Task graph formatter for Arklex framework.

Formats task definitions into graph structure with nodes, edges, and metadata.
Handles LLM-based intent generation and nested graph connectivity.
"""

from typing import Dict, Any, List, Optional, Tuple

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

    # Default worker names - can be overridden in config
    DEFAULT_MESSAGE_WORKER = "MessageWorker"
    DEFAULT_RAG_WORKER = "FaissRAGWorker"
    DEFAULT_SEARCH_WORKER = "SearchWorker"
    DEFAULT_NESTED_GRAPH = "NestedGraph"

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
        default_intent: Optional[str] = None,
        default_weight: int = 1,
        default_pred: bool = False,
        default_definition: str = "",
        default_sample_utterances: Optional[List[str]] = None,
        nodes: Optional[List[Any]] = None,
        edges: Optional[List[Any]] = None,
        allow_nested_graph: bool = True,
        model: Optional[Any] = None,
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
            default_intent (Optional[str]): Default intent for edges
            default_weight (int): Default weight for edges
            default_pred (bool): Default pred value for edge attributes
            default_definition (str): Default definition for edge attributes
            default_sample_utterances (Optional[List[str]]): Default sample utterances for edge attributes
            nodes (Optional[List[Any]]): List of nodes
            edges (Optional[List[Any]]): List of edges
            allow_nested_graph (bool): Whether to allow nested graph generation
            model (Optional[Any]): Language model for intent generation
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
        self._default_pred = default_pred
        self._default_definition = default_definition
        self._default_sample_utterances = default_sample_utterances or []
        self._nodes = nodes
        self._edges = edges
        self._allow_nested_graph = allow_nested_graph
        self._model = model

    def _find_worker_by_name(self, worker_name: str) -> Dict[str, str]:
        """Get worker info from name, with fallback mappings.

        Args:
            worker_name (str): Name of the worker to find info for

        Returns:
            Dict[str, str]: Worker info with 'id' and 'name' keys
        """
        if self._workers:
            for worker in self._workers:
                if isinstance(worker, dict) and worker.get("name") == worker_name:
                    return {
                        "id": worker.get("id", worker_name.lower()),
                        "name": worker.get("name", worker_name),
                    }

        # Fallback mappings based on config
        fallback_workers = {
            "MessageWorker": {
                "id": "26bb6634-3bee-417d-ad75-23269ac17bc3",
                "name": "MessageWorker",
            },
            "FaissRAGWorker": {"id": "FaissRAGWorker", "name": "FaissRAGWorker"},
            "SearchWorker": {
                "id": "9c15af81-04b3-443e-be04-a3522124b905",
                "name": "SearchWorker",
            },
        }
        return fallback_workers.get(
            worker_name, {"id": worker_name.lower(), "name": worker_name}
        )

    def format_task_graph(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format tasks into complete graph structure with nodes, edges, and metadata.

        Args:
            tasks (List[Dict[str, Any]]): List of task definitions to format

        Returns:
            Dict[str, Any]: Complete task graph with nodes, edges, and metadata
        """
        if self._nodes is not None and self._edges is not None:
            return {"nodes": self._nodes, "edges": self._edges}

        # Format nodes and edges
        nodes, node_lookup, all_task_node_ids = self._format_nodes(tasks)
        start_node_id = "0"  # Start node is always "0" in our current implementation
        edges, nested_graph_nodes = self._format_edges(
            tasks, node_lookup, all_task_node_ids, start_node_id
        )

        # Update NestedGraph node values with their target node IDs
        all_nodes = nodes + nested_graph_nodes
        for node_id, node_data in all_nodes:
            if node_data.get("resource", {}).get("name") == "NestedGraph":
                # Find the target node this NestedGraph connects to by looking at edges
                target_node_id = None
                for edge in edges:
                    if edge[0] == node_id:  # This edge starts from the NestedGraph node
                        potential_target = edge[1]  # The target is the second element
                        # Don't point to node "0" (start node)
                        if potential_target != "0":
                            target_node_id = potential_target
                            break

                if target_node_id:
                    node_data["attribute"]["value"] = target_node_id
                else:
                    # If no valid target found, point to the first non-start task node
                    task_node_ids = [
                        nid
                        for nid, ndata in all_nodes
                        if ndata.get("resource", {}).get("name")
                        in ["MessageWorker", "FaissRAGWorker", "SearchWorker"]
                        and nid != "0"
                    ]
                    if task_node_ids:
                        node_data["attribute"]["value"] = task_node_ids[0]
                    else:
                        # Fallback: point to node "1" if it exists
                        node_data["attribute"]["value"] = "1"
            else:
                # If the value is a dict, replace with its description
                value = node_data.get("attribute", {}).get("value")
                if isinstance(value, dict):
                    desc = value.get("description")
                    if desc:
                        node_data["attribute"]["value"] = desc
                    else:
                        node_data["attribute"]["value"] = str(value)
                # If the value is a list or nested, flatten to string
                elif isinstance(value, list):
                    node_data["attribute"]["value"] = ", ".join(str(v) for v in value)

        # Build the complete task graph with all required sections
        task_graph = {
            "nodes": all_nodes,
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

        return task_graph

    def _format_nodes(self, tasks: List[Dict]) -> Tuple[List, Dict, List]:
        """Create nodes for start, tasks, and steps with worker assignments.

        Args:
            tasks (List[Dict]): List of task definitions to create nodes for

        Returns:
            Tuple[List, Dict, List]: (nodes, node_lookup, all_task_node_ids)
                - nodes: List of formatted node data
                - node_lookup: Mapping of task identifiers to node IDs
                - all_task_node_ids: List of all task node IDs
        """
        nodes = []
        node_id_counter = 0
        node_lookup = {}
        all_task_node_ids = []

        # Create start node
        start_node_id = str(node_id_counter)
        message_worker_id = self._find_worker_by_name(self.DEFAULT_MESSAGE_WORKER)
        nodes.append(
            [
                start_node_id,
                {
                    "resource": {
                        "id": message_worker_id["id"],
                        "name": message_worker_id["name"],
                    },
                    "attribute": {
                        "value": "Hello! I'm here to assist you with any customer service inquiries.",
                        "task": "start message",
                        "directed": False,
                    },
                    "limit": 1,
                    "type": "start",
                },
            ]
        )
        node_id_counter += 1

        # First pass: Create all task and step nodes
        for task_idx, task in enumerate(tasks):
            # Use 'id' if available, otherwise generate one
            task_identifier = task.get("id", f"task_{task_idx}")

            # Ensure task has an id field
            if "id" not in task:
                task["id"] = task_identifier

            # Use the resource from the task if available, otherwise default to MessageWorker
            resource_name = self.DEFAULT_MESSAGE_WORKER
            if task.get("resource") and isinstance(task["resource"], dict):
                resource_name = task["resource"].get(
                    "name", self.DEFAULT_MESSAGE_WORKER
                )
            elif task.get("resource"):
                resource_name = str(task["resource"])

            # Handle nested graph resources with specific names
            if resource_name == self.DEFAULT_NESTED_GRAPH or (
                resource_name
                not in [
                    self.DEFAULT_MESSAGE_WORKER,
                    self.DEFAULT_RAG_WORKER,
                    self.DEFAULT_SEARCH_WORKER,
                ]
                and "workflow" in resource_name.lower()
            ):
                resource_info = {"id": "nested_graph", "name": "NestedGraph"}
            else:
                resource_info = self._find_worker_by_name(resource_name)

            task_node_id = str(node_id_counter)
            node_data = {
                "resource": {"id": resource_info["id"], "name": resource_info["name"]},
                "attribute": {
                    "value": task.get("description", ""),
                    "task": task.get("name", ""),
                    "directed": False,
                },
            }
            if task.get("type"):
                node_data["type"] = task.get("type")
            if task.get("limit"):
                node_data["limit"] = task.get("limit")

            nodes.append([task_node_id, node_data])
            node_lookup[task_identifier] = task_node_id
            all_task_node_ids.append(task_node_id)
            node_id_counter += 1

            # Ensure steps are properly broken down
            steps = task.get("steps", [])
            if not steps and task.get("description"):
                # If no steps defined, create a single step from the task description
                steps = [
                    {"description": task.get("description", ""), "step_id": "step_1"}
                ]
                task["steps"] = steps

            for step_idx, step in enumerate(steps):
                step_id = f"{task_identifier}_step{step_idx}"
                step_node_id = str(node_id_counter)

                # Use the resource from the step if available (from best practice manager)
                step_worker_name = self.DEFAULT_MESSAGE_WORKER
                if (
                    isinstance(step, dict)
                    and step.get("resource")
                    and isinstance(step["resource"], dict)
                ):
                    step_worker_name = step["resource"].get(
                        "name", self.DEFAULT_MESSAGE_WORKER
                    )
                elif isinstance(step, dict) and step.get("resource"):
                    step_worker_name = str(step["resource"])

                # Handle nested graph resources with specific names
                if step_worker_name == self.DEFAULT_NESTED_GRAPH or (
                    step_worker_name
                    not in [
                        self.DEFAULT_MESSAGE_WORKER,
                        self.DEFAULT_RAG_WORKER,
                        self.DEFAULT_SEARCH_WORKER,
                    ]
                    and "workflow" in step_worker_name.lower()
                ):
                    step_worker_info = {"id": "nested_graph", "name": "NestedGraph"}
                else:
                    step_worker_info = self._find_worker_by_name(step_worker_name)

                # Simplify step value to use simple string instead of complex nested structure
                if isinstance(step, dict):
                    # Check if step has a complex structure with task, description, step_id, etc.
                    if "task" in step and "description" in step and "step_id" in step:
                        # This is a complex step structure, extract just the description
                        step_value = step.get("description", "")
                    else:
                        # Simple step structure, use description or task
                        step_value = step.get("description", step.get("task", ""))
                elif isinstance(step, str):
                    step_value = step
                else:
                    step_value = str(step)

                nodes.append(
                    [
                        step_node_id,
                        {
                            "resource": {
                                "id": step_worker_info["id"],
                                "name": step_worker_info["name"],
                            },
                            "attribute": {
                                "value": step_value,
                                "task": task.get("name", ""),
                                "directed": False,
                            },
                        },
                    ]
                )
                node_lookup[step_id] = step_node_id
                node_id_counter += 1

        return nodes, node_lookup, all_task_node_ids

    def _create_edge_attributes(
        self,
        intent: Optional[str] = None,
        weight: int = 1,
        pred: bool = False,
        definition: str = "",
        sample_utterances: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create edge attributes with intent and metadata.

        Args:
            intent (Optional[str]): Intent for the edge, defaults to None
            weight (int): Edge weight, defaults to 1
            pred (bool): Prediction flag, defaults to False
            definition (str): Edge definition, defaults to empty string
            sample_utterances (Optional[List[str]]): Sample utterances for the edge

        Returns:
            Dict[str, Any]: Edge attributes dictionary with intent and metadata
        """
        return {
            "intent": intent,
            "attribute": {
                "weight": weight,
                "pred": pred,
                "definition": definition,
                "sample_utterances": sample_utterances or [],
            },
        }

    def _format_edges(
        self,
        tasks: List[Dict[str, Any]],
        node_lookup: Dict[str, str],
        all_task_node_ids: List[str],
        start_node_id: str,
    ) -> Tuple[List[Any], List[Any]]:
        """Create edges between nodes with LLM-generated intents.

        Args:
            tasks (List[Dict[str, Any]]): List of task definitions
            node_lookup (Dict[str, str]): Mapping of task identifiers to node IDs
            all_task_node_ids (List[str]): List of all task node IDs
            start_node_id (str): ID of the start node

        Returns:
            Tuple[List[Any], List[Any]]: (edges, nested_graph_nodes)
                - edges: List of formatted edge data
                - nested_graph_nodes: List of nested graph nodes
        """
        edges = []
        nested_graph_nodes = []

        # Generate descriptive intents for main task edges using LLM
        for task in tasks:
            task_identifier = task.get("id", f"task_{tasks.index(task)}")
            task_node_id = node_lookup.get(task_identifier)

            if task_node_id:
                dependencies = task.get("dependencies", [])
                if not dependencies:
                    # This is a main task edge from the start node.
                    # Use the intent from the task, or generate a descriptive one
                    intent = task.get("intent", "")
                    if not intent:
                        # Generate a descriptive intent based on task name and description
                        task_name = task.get("name", "")
                        task_description = task.get("description", "")

                        # Create a prompt for intent generation
                        intent_prompt = f"""
                        Given a task with the following details, generate a user-facing intent that describes what the user wants to accomplish.
                        
                        Task Name: {task_name}
                        Task Description: {task_description}
                        
                        Generate a natural, user-facing intent that describes what the user is trying to achieve. 
                        Examples of good intents:
                        - "User inquires about purchasing options"
                        - "User wants to explore rental options"
                        - "User asks about delivery times and logistics"
                        - "User has technical support or troubleshooting queries"
                        - "User seeks information on robot capabilities and features"
                        - "User is interested in customization options"
                        - "User inquires about service and maintenance"
                        - "User wants to learn about new releases and updates"
                        
                        Intent: """

                        try:
                            # Use the model to generate intent if available
                            if self._model:
                                from langchain_core.messages import HumanMessage

                                response = self._model.invoke(
                                    [HumanMessage(content=intent_prompt)]
                                )
                                intent = response.content.strip().strip('"').strip("'")
                                # Clean up the response to ensure it's a valid intent
                                if not intent or intent.lower() in [
                                    "none",
                                    "null",
                                    "n/a",
                                ]:
                                    raise ValueError("Invalid intent generated")
                            else:
                                intent = "User inquires about purchasing options"  # default fallback
                        except Exception as e:
                            log_context.warning(
                                f"Failed to generate intent for task {task_name}: {e}"
                            )
                            intent = (
                                "User inquires about purchasing options"  # fallback
                            )

                    edges.append(
                        [
                            start_node_id,
                            task_node_id,
                            self._create_edge_attributes(intent=intent, pred=True),
                        ]
                    )
                else:
                    # This handles dependencies between tasks.
                    for dep in dependencies:
                        dep_id = dep if isinstance(dep, str) else dep.get("id")
                        target_node_id = node_lookup.get(dep_id)
                        if target_node_id:
                            edges.append(
                                [
                                    target_node_id,
                                    task_node_id,
                                    self._create_edge_attributes(),
                                ]
                            )
                        else:
                            log_context.warning(
                                f"Could not find target node for dependency '{dep_id}'"
                            )

        # Steps
        for task in tasks:
            task_identifier = task.get("id", f"task_{tasks.index(task)}")
            task_node_id = node_lookup.get(task_identifier)

            if task_node_id:
                steps = task.get("steps", [])
                if steps:
                    first_step_id = f"{task_identifier}_step0"
                    if first_step_id in node_lookup:
                        edges.append(
                            [
                                task_node_id,
                                node_lookup[first_step_id],
                                self._create_edge_attributes(intent=None),
                            ]
                        )
                        for i in range(len(steps) - 1):
                            current_step_id = f"{task_identifier}_step{i}"
                            next_step_id = f"{task_identifier}_step{i + 1}"
                            if (
                                current_step_id in node_lookup
                                and next_step_id in node_lookup
                            ):
                                edges.append(
                                    [
                                        node_lookup[current_step_id],
                                        node_lookup[next_step_id],
                                        self._create_edge_attributes(intent=None),
                                    ]
                                )

        # Nested graph node - create only one for the entire graph
        if self._allow_nested_graph and tasks:
            nested_graph_node_id = str(
                len(edges) + len(all_task_node_ids) + 1
            )  # Use a unique ID

            # Find all nodes that are the target of an edge from the main graph start node
            main_graph_targets = {edge[1] for edge in edges if edge[0] == start_node_id}

            # Subgraph start nodes: task nodes that are not the main graph start node and not directly targeted by the main graph start node
            subgraph_start_nodes = [
                node_id
                for node_id in all_task_node_ids
                if node_id != start_node_id and node_id not in main_graph_targets
            ]

            # Fallback: if all task nodes are main graph targets, just use the first task node that is not the start node
            if not subgraph_start_nodes:
                subgraph_start_nodes = [
                    node_id for node_id in all_task_node_ids if node_id != start_node_id
                ]

            # Use the first valid subgraph start node
            if subgraph_start_nodes:
                nested_graph_value = subgraph_start_nodes[0]
            else:
                nested_graph_value = (
                    all_task_node_ids[0] if all_task_node_ids else start_node_id
                )

            # Get a task name for the nested graph (use the first task)
            task_name = tasks[0].get("name", "nested task") if tasks else "nested task"

            # Create nested graph node
            nested_graph_nodes.append(
                [
                    nested_graph_node_id,
                    {
                        "resource": {
                            "id": "nested_graph",
                            "name": self.DEFAULT_NESTED_GRAPH,
                        },
                        "attribute": {
                            "value": nested_graph_value,
                            "task": f"Execute {task_name}",
                            "directed": True,
                        },
                        "limit": 1,
                    },
                ]
            )

            # Add edge from start node to nested graph node
            edges.append(
                [
                    start_node_id,
                    nested_graph_node_id,
                    self._create_edge_attributes(
                        intent=None,
                        weight=1,
                        pred=False,
                        definition="",
                        sample_utterances=[],
                    ),
                ]
            )

        return edges, nested_graph_nodes

    def link_main_graph_to_nested_graph(
        self, main_graph: Dict[str, Any], nested_graph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Link main graph to nested graph and establish proper connectivity.

        Args:
            main_graph (Dict[str, Any]): Main graph structure
            nested_graph (Dict[str, Any]): Nested graph structure to link

        Returns:
            Dict[str, Any]: Combined graph with proper connectivity between main and nested graphs
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
            if node_data.get("resource", {}).get("id") == self.DEFAULT_MESSAGE_WORKER:
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
                    "name": self.DEFAULT_NESTED_GRAPH,
                },
                "attribute": {
                    "value": "placeholder",  # Will be updated to first task node
                    "task": "Execute nested graph task",
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
            if node_id == start_node_id:
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
                    node_data["attribute"]["task"] = "Execute nested graph task"
                    node_data["attribute"]["directed"] = True
                    break
        else:
            # If no valid nested start found, point to first task node
            # Extract task node IDs from main graph nodes
            main_task_node_ids = [
                node_id
                for node_id in all_task_node_ids
                if (
                    node_data.get("resource", {}).get("id")
                    in [
                        self.DEFAULT_MESSAGE_WORKER,
                        self.DEFAULT_RAG_WORKER,
                        self.DEFAULT_SEARCH_WORKER,
                    ]
                    and node_id not in (start_node_id, nested_graph_node_id)
                )
            ]
            if main_task_node_ids:
                first_task_id = main_task_node_ids[0]
                for node_id, node_data in combined_graph["nodes"]:
                    if node_id == nested_graph_node_id:
                        node_data["attribute"]["value"] = first_task_id
                        node_data["attribute"]["task"] = "Execute nested graph task"
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
                self._create_edge_attributes(
                    intent=None,
                    weight=1,
                    pred=False,
                    definition=f"Nested graph connects to subgraph start node {start_id}",
                    sample_utterances=[],
                ),
            ]
            combined_graph["edges"].append(nested_graph_to_start_edge)

        return combined_graph

    def ensure_nested_graph_connectivity(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure nested graph has proper incoming and outgoing connections.

        Args:
            graph (Dict[str, Any]): Graph structure to validate and fix

        Returns:
            Dict[str, Any]: Graph with proper nested graph connectivity
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
            elif node_data.get("resource", {}).get("id") == self.DEFAULT_MESSAGE_WORKER:
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
            if not (edge[0] == nested_graph_node_id and edge[1] == start_node_id)
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
                self._create_edge_attributes(
                    intent=None,
                    weight=1,
                    pred=True,
                    definition="Start node connects to nested graph",
                    sample_utterances=[],
                ),
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
                self._create_edge_attributes(
                    intent=None,
                    weight=1,
                    pred=False,
                    definition=f"Nested graph connects to leaf node {leaf_id}",
                    sample_utterances=[],
                ),
            ]
            fixed_graph["edges"].append(nested_graph_to_leaf_edge)

        return fixed_graph
