"""Task graph formatter for Arklex framework.

Formats task definitions into graph structure with nodes, edges, and metadata.
Handles LLM-based intent generation and nested graph connectivity.
"""

from typing import Any

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
        task_docs: list[dict[str, Any]] | None = None,
        rag_docs: list[dict[str, Any]] | None = None,
        workers: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        nluapi: str = "",
        slotfillapi: str = "",
        default_intent: str | None = None,
        default_weight: int = 1,
        default_pred: bool = False,
        default_definition: str = "",
        default_sample_utterances: list[str] | None = None,
        nodes: list[Any] | None = None,
        edges: list[Any] | None = None,
        allow_nested_graph: bool = True,
        model: object | None = None,
        settings: dict[str, Any] | None = None,
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
            settings (Optional[Dict[str, Any]]): Additional configuration settings
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
        self._settings = settings

    def _find_worker_by_name(self, worker_name: str) -> dict[str, str]:
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

    def format_task_graph(self, tasks: list[dict[str, Any]]) -> dict[str, Any]:
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
                        # The target is the second element
                        potential_target = edge[1]
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

        # Add nested graph nodes to the main nodes list
        nodes.extend(nested_graph_nodes)

        reusable_tasks = {}  # Define reusable_tasks, can be passed in later

        graph = {
            "nodes": nodes,
            "edges": edges,
            "tasks": tasks,  # Include tasks for connectivity logic
            "role": self._role,
            "user_objective": self._user_objective,
            "builder_objective": self._builder_objective,
            "domain": self._domain,
            "intro": self._intro,
            "task_docs": self._task_docs,
            "rag_docs": self._rag_docs,
            "workers": self._workers,
            "tools": self._tools,
            "nluapi": self._nluapi,
            "slotfillapi": self._slotfillapi,
            "reusable_tasks": reusable_tasks,
            "settings": self._settings,
        }

        # Ensure nested graphs are connected correctly as sequential steps
        return self.ensure_nested_graph_connectivity(graph)

    def _format_nodes(self, tasks: list[dict]) -> tuple[list, dict, list]:
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
        intent: str | None = None,
        weight: int = 1,
        pred: bool = False,
        definition: str = "",
        sample_utterances: list[str] | None = None,
    ) -> dict[str, Any]:
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
        tasks: list[dict[str, Any]],
        node_lookup: dict[str, str],
        all_task_node_ids: list[str],
        start_node_id: str,
    ) -> tuple[list[Any], list[Any]]:
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
                        # Handle edge cases where dependency is None or not a string/dict
                        if dep is None:
                            log_context.warning("Skipping None dependency")
                            continue
                        elif isinstance(dep, str):
                            dep_id = dep
                        elif isinstance(dep, dict):
                            dep_id = dep.get("id")
                            if dep_id is None:
                                log_context.warning(
                                    "Skipping dependency dict without 'id' field"
                                )
                                continue
                        else:
                            log_context.warning(
                                f"Skipping invalid dependency type: {type(dep)}"
                            )
                            continue

                        # Find the source task from the list of tasks
                        source_task = next(
                            (t for t in tasks if t.get("id") == dep_id), None
                        )

                        if source_task:
                            source_steps = source_task.get("steps", [])
                            if source_steps:
                                # Dependency should come from the last step of the source task
                                last_step_identifier = (
                                    f"{dep_id}_step{len(source_steps) - 1}"
                                )
                                source_node_id = node_lookup.get(last_step_identifier)
                            else:
                                # If no steps, dependency is from the task node itself
                                source_node_id = node_lookup.get(dep_id)
                        else:
                            # Fallback to original logic if source task not found
                            source_node_id = node_lookup.get(dep_id)

                        if source_node_id:
                            edges.append(
                                [
                                    source_node_id,
                                    task_node_id,
                                    self._create_edge_attributes(),
                                ]
                            )
                        else:
                            log_context.warning(
                                f"Could not find source node for dependency '{dep_id}'"
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

        return edges, nested_graph_nodes

    def ensure_nested_graph_connectivity(self, graph: dict[str, Any]) -> dict[str, Any]:
        """Ensures that all nested graph nodes are properly connected as sequential steps.

        This method finds all 'NestedGraph' nodes and connects them to the
        next step within their own task, preventing incorrect fan-out connections
        to leaf nodes.

        Args:
            graph (Dict[str, Any]): The graph structure.

        Returns:
            Dict[str, Any]: The graph with corrected nested graph connectivity.
        """
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        tasks = graph.get("tasks", [])
        node_lookup = {node_data[0]: node_data[1] for node_data in nodes}
        # Create a reverse lookup from node ID to task_id
        node_to_task_map = {}
        for task in tasks:
            task_id = task.get("id")
            for i in range(len(task.get("steps", []))):
                step_node_id = f"{task_id}_step{i}"
                node_to_task_map[step_node_id] = task_id

        nested_graph_nodes = [
            (node_id, node_data)
            for node_id, node_data in node_lookup.items()
            if node_data.get("resource", {}).get("name") == self.DEFAULT_NESTED_GRAPH
        ]

        for ng_node_id, ng_node_data in nested_graph_nodes:
            # Find which task this nested_graph node belongs to
            task_id = node_to_task_map.get(ng_node_id)
            if not task_id:
                continue

            # Find the corresponding task and the index of the nested_graph step
            task = next((t for t in tasks if t.get("id") == task_id), None)
            if not task:
                continue

            steps = task.get("steps", [])
            step_index = -1
            for i, _step in enumerate(steps):
                step_node_id = f"{task_id}_step{i}"
                if step_node_id == ng_node_id:
                    step_index = i
                    break

            # If it's not the last step, connect it to the next one
            if 0 <= step_index < len(steps) - 1:
                next_step_node_id = f"{task_id}_step{step_index + 1}"

                # Set the value attribute - create attribute dict if it doesn't exist
                ng_node_data.setdefault("attribute", {})["value"] = next_step_node_id

                # Create the edge
                edges.append(
                    [
                        ng_node_id,
                        next_step_node_id,
                        self._create_edge_attributes(
                            definition="Continue to next step from nested graph"
                        ),
                    ]
                )

        graph["edges"] = edges
        return graph
