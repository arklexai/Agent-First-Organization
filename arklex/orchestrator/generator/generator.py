"""Task graph generator implementation for the Arklex framework.

This module provides functionality for generating task graphs and managing task hierarchies.
It includes classes for task editing, task generation, and handling reusable tasks.

Key Components:
- Generator: Main class for creating task graphs based on user objectives and documentation
- TaskEditorApp: Text-based UI for editing tasks and their steps
- InputModal: Modal dialog for editing task and step descriptions

Features:
- Natural language task generation
- Interactive task editing
- Reusable task management
- Best practice integration
- Documentation processing
- Resource initialization
- Task graph formatting
- Configuration management

Usage:
    from arklex.orchestrator.generator import Generator
    from arklex.env.env import DefaultResourceInitializer

    # Initialize generator
    config = {
        "role": "customer_service",
        "user_objective": "Handle customer inquiries",
        "builder_objective": "Create efficient response system",
        "instructions": [...],
        "tasks": [...],
        "workers": [...],
        "tools": [...]
    }

    generator = Generator(
        config=config,
        model=language_model,
        output_dir="output",
        resource_inizializer=DefaultResourceInitializer()
    )

    # Generate task graph
    task_graph = generator.generate()

    # Save task graph
    output_path = generator.save_task_graph(task_graph)
"""

import os
import json
import logging
from datetime import datetime
from tqdm import tqdm as progress_bar
import pickle
from pathlib import Path
import inspect
from typing import Optional
from collections import deque

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from textual.app import App, ComposeResult
from textual.widgets import Tree, Label, Input, Button, Static
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets.tree import TreeNode

from arklex.utils.utils import postprocess_json
from arklex.orchestrator.generator.prompts import *
from arklex.utils.loader import Loader, SourceType
from arklex.env.env import BaseResourceInitializer, DefaultResourceInitializer
from arklex.env.nested_graph.nested_graph import NESTED_GRAPH_ID


logger = logging.getLogger(__name__)


class InputModal(Screen):
    """A simple input modal for editing or adding tasks/steps.

    This class provides a modal dialog interface for editing task and step descriptions.
    It includes input validation and callback handling for user interactions.

    Attributes:
        title (str): The title of the modal dialog
        default (str): Default value for the input field
        result (str): The final result after user interaction
        node (TreeNode): The tree node being edited
        callback (callable): Function to call after user interaction

    Methods:
        compose(): Creates the modal UI components
        on_button_pressed(): Handles button press events
    """

    def __init__(self, title: str, default: str = "", node=None, callback=None):
        super().__init__()
        self.title = title
        self.default = default
        self.result = default
        self.node = node
        self.callback = callback

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(self.title, classes="title"),
            Input(value=self.default, id="input-field"),
            Horizontal(
                Button("Submit", id="submit"),
                Button("Cancel", id="cancel"),
                id="buttons",
            ),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            self.result = self.query_one("#input-field", Input).value
            # logger.debug(f"InputModal result: {self.result}")
        if self.callback:
            self.callback(self.result, self.node)
        logger.debug(f"InputModal result: {self.result}")
        self.app.pop_screen()  # Close modal


class TaskEditorApp(App):
    """A Textual app to edit tasks and steps in a hierarchical structure.

    This class provides a text-based user interface for editing tasks and their steps.
    It supports adding, editing, and deleting tasks and steps in a tree structure.

    Attributes:
        tasks (list): List of task dictionaries containing task names and steps
        task_tree (Tree): The tree widget displaying tasks and steps

    Methods:
        compose(): Creates the main UI components
        on_mount(): Initializes the UI after mounting
        on_tree_node_selected(): Handles node selection events
        on_key(): Processes keyboard input
        action_add_node(): Adds new nodes to the tree
        show_input_modal(): Displays the input modal dialog
        update_tasks(): Updates the tasks list from the tree structure
    """

    def __init__(self, tasks):
        super().__init__()
        self.tasks = tasks
        self.task_tree: Tree[str] = None

    def compose(self) -> ComposeResult:
        self.task_tree = Tree("Tasks")
        self.task_tree.root.expand()

        # Populate the tree with tasks and steps
        for task in self.tasks:
            task_node = self.task_tree.root.add(task["task_name"], expand=True)
            for step in task["steps"]:
                task_node.add_leaf(step)

        yield self.task_tree
        yield Label(
            "Click on a task or step to edit it. Press 'a' to add new item, 'd' to delete, 's' to save and exit."
        )

    def on_mount(self):
        self.task_tree.focus()

    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        selected_node = event.node

        def handle_modal_result(result, node):
            if result is not None:  # Check if the user submitted a valid result
                node.set_label(result)  # Update the tree node's label
                self.call_later(
                    self.update_tasks
                )  # Ensure task sync runs after UI update

        self.push_screen(
            InputModal(
                f"Edit '{selected_node.label}'",
                default=str(selected_node.label),
                node=selected_node,
                callback=handle_modal_result,
            )
        )

    async def on_key(self, event):
        selected_node = self.task_tree.cursor_node
        if event.key == "a" and selected_node and selected_node.parent is not None:
            await self.action_add_node(selected_node)
        elif event.key == "d" and selected_node and selected_node.parent is not None:
            selected_node.remove()
            await self.update_tasks()
        elif event.key == "s":
            self.exit(self.tasks)

    async def action_add_node(self, node: TreeNode):
        # if the node is a step node
        if node.parent.parent is not None:
            leaf = True
            node = node.parent
            title = f"Add new step under '{node.label.plain}'"
        else:  # if the node is a task node
            if node.is_expanded:  # add step
                leaf = True
                node = node
                title = f"Enter new step under '{node.label.plain}'"
            else:
                leaf = False
                node = node.parent
                title = f"Add new task under '{node.label.plain}'"

        def handle_modal_result(result, node):
            if result is not None:  # Check if the user submitted a valid result
                if leaf:
                    node.add_leaf(result)
                else:
                    node.add(result, expand=True)
                self.call_later(
                    self.update_tasks
                )  # Ensure task sync runs after UI update

        self.push_screen(
            InputModal(title, default="", node=node, callback=handle_modal_result)
        )

    def show_input_modal(self, title: str, default: str = "") -> str:
        modal = InputModal(title, default)
        self.push_screen(modal)
        return modal.result

    async def update_tasks(self):
        self.tasks = []
        for task_node in self.task_tree.root.children:
            task_name = task_node.label.plain
            steps = [step.label.plain for step in task_node.children]
            self.tasks.append({"task_name": task_name, "steps": steps})

        log_message = f"Updated Tasks: {self.tasks}"
        logger.debug(log_message)


class Generator:
    """Main class for generating task graphs based on user objectives and documentation.

    This class handles the generation of task graphs, including reusable tasks,
    best practices, and task hierarchy management. It processes user objectives,
    documentation, and configuration to create structured task graphs.

    Attributes:
        product_kwargs (dict): Configuration settings for the generator
        role (str): The role or context for task generation
        u_objective (str): User's objective for the task graph
        b_objective (str): Builder's objective for the task graph
        intro (str): Introduction text for the task graph
        instruction_docs (list): Documentation for instructions
        task_docs (list): Documentation for tasks
        rag_docs (list): Documentation for RAG operations
        user_tasks (list): User-provided tasks
        example_conversations (list): Example conversations for reference
        workers (list): Available workers for task execution
        tools (list): Available tools for task execution
        interactable_with_user (bool): Whether to allow user interaction
        allow_nested_graph (bool): Whether to allow nested graph generation
        model: The language model for task generation
        timestamp (str): Timestamp for output files
        output_dir (str): Directory for saving generated files
        documents (str): Processed task documents
        reusable_tasks (dict): Generated reusable tasks
        tasks (list): Generated tasks

    Methods:
        _generate_reusable_tasks(): Generates reusable tasks
        _generate_tasks(): Generates main tasks
        _add_provided_tasks(): Adds user-provided tasks
        _generate_best_practice(): Generates best practices
        _finetune_best_practice(): Refines best practices
        _format_task_graph(): Formats the final task graph
        _load_docs(): Loads and processes documentation
        _load_instructions(): Loads and processes instructions
        generate(): Main method to generate the task graph
        save_task_graph(): Saves the generated task graph
    """

    def __init__(
        self,
        config: dict,
        model,
        output_dir: Optional[str] = None,
        resource_inizializer: Optional[BaseResourceInitializer] = None,
        interactable_with_user=True,
        allow_nested_graph=True,
    ):
        """Initialize the Generator instance.

        This function initializes the task graph generator with configuration settings,
        model, and other parameters. It sets up the necessary components for generating
        and managing task graphs.

        Args:
            config (dict): Configuration dictionary containing product settings and parameters.
            model: The language model to use for task generation.
            output_dir (Optional[str]): Directory to save generated task graphs. Defaults to None.
            resource_inizializer (Optional[BaseResourceInitializer]): Initializer for resources.
                Defaults to None.
            interactable_with_user (bool): Whether to allow user interaction during generation.
                Defaults to True.
            allow_nested_graph (bool): Whether to allow nested graph generation. Defaults to True.
        """
        if resource_inizializer is None:
            resource_inizializer = DefaultResourceInitializer()
        self.product_kwargs = config
        self.role = self.product_kwargs.get("role")
        self.u_objective = self.product_kwargs.get("user_objective")
        self.b_objective = self.product_kwargs.get("builder_objective")
        self.intro = self.product_kwargs.get("intro")
        self.instruction_docs = self.product_kwargs.get("instructions")
        self.task_docs = self.product_kwargs.get("task_docs")
        self.rag_docs = self.product_kwargs.get("rag_docs")
        self.user_tasks = self.product_kwargs.get("tasks")
        self.example_conversations = self.product_kwargs.get("example_conversations")
        self.workers = resource_inizializer.init_workers(
            self.product_kwargs.get("workers")
        )
        self.tools = resource_inizializer.init_tools(self.product_kwargs.get("tools"))
        self.interactable_with_user = interactable_with_user
        self.allow_nested_graph = allow_nested_graph
        self.model = model
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_dir = output_dir

        # variables that will be used
        self.documents = ""  # task documents
        self.reusable_tasks = {}  # nested graph tasks
        self.tasks = []  # tasks

    def _generate_reusable_tasks(self):
        """Generate reusable tasks based on the configuration.

        This function creates reusable tasks that can be shared across different parts
        of the task graph. It uses the language model to generate task definitions and
        their associated steps.

        Returns:
            List[Dict[str, Any]]: List of generated reusable tasks.
        """
        prompt = PromptTemplate.from_template(generate_reusable_tasks_sys_prompt)
        input_prompt = prompt.invoke(
            {
                "role": self.role,
                "u_objective": self.u_objective,
                "intro": self.intro,
                "tasks": self.tasks,
                "docs": self.documents,
                "instructions": self.instructions,
                "example_conversations": self.example_conversations,
            }
        )
        final_chain = self.model | StrOutputParser()
        answer = final_chain.invoke(input_prompt)
        results = postprocess_json(answer)
        reusable_tasks = {}
        for task in results:
            task_name = task["name"].replace(" ", "_").lower()
            reusable_tasks[task_name] = {
                "nestedgraph_task": task["description"],
                "subgraph": task["steps"],
            }

        resources = {}
        for _, worker_info in self.workers.items():
            worker_name = worker_info["name"]
            worker_desc = worker_info["description"]
            resources[worker_name] = worker_desc

        for _, tool_info in self.tools.items():
            tool_name = tool_info["name"]
            tool_desc = tool_info["description"]
            resources[tool_name] = tool_desc

        # TODO: do I want to allow subgraph in subgraph?
        # for task_name, task_info in reusable_tasks.items():
        #     resources[task_name] = task_info["nestedgraph_task"]

        reusable_task_finetune_prompt = PromptTemplate.from_template(
            embed_reusable_task_resources_sys_prompt
        )
        for task_name in reusable_tasks:
            n_trials = 0
            max_trials = 3
            while n_trials < max_trials:
                input_prompt = reusable_task_finetune_prompt.invoke(
                    {
                        "best_practice": reusable_tasks[task_name]["subgraph"],
                        "resources": resources,
                    }
                )
                final_chain = self.model | StrOutputParser()
                answer = final_chain.invoke(input_prompt)
                task_subgraph = postprocess_json(answer)

                tasks = [task_subgraph]
                has_all_resource = True
                while tasks:
                    task = tasks.pop()
                    if task.get("resource") not in resources:
                        has_all_resource = False
                        break
                    for next_task in task.get("next", []):
                        tasks.append(next_task)
                if has_all_resource:
                    break
                n_trials += 1

            reusable_tasks[task_name]["subgraph"] = task_subgraph

        self.reusable_tasks = reusable_tasks

    def _generate_tasks(self):
        """Generate tasks based on the type and documents.

        This function creates tasks based on the configuration type and available
        documentation. It handles different types of task generation and integrates
        them into the task graph.

        Returns:
            List[Dict[str, Any]]: List of generated tasks.
        """
        # based on the type and documents
        prompt = PromptTemplate.from_template(generate_tasks_sys_prompt)
        input_prompt = prompt.invoke(
            {
                "role": self.role,
                "u_objective": self.u_objective,
                "intro": self.intro,
                "docs": self.documents,
                "instructions": self.instructions,
                "existing_tasks": self.tasks,
            }
        )
        final_chain = self.model | StrOutputParser()
        answer = final_chain.invoke(input_prompt)
        logger.debug(f"Generated tasks with thought: {answer}")
        self.tasks.extend(postprocess_json(answer))

    def _add_provided_tasks(self):
        """Add provided tasks to the task graph.

        This function processes and adds tasks that were explicitly provided in the
        configuration to the task graph.

        Returns:
            List[Dict[str, Any]]: List of added tasks.
        """
        if not self.user_tasks:
            return
        new_format_tasks = []
        for user_task in self.user_tasks:
            task = {}
            # task['intent'] = task_str
            task["task"] = user_task["task"]
            new_format_tasks.append(task)

        # given the provided tasks, predict the intent of tasks
        prompt = PromptTemplate.from_template(task_intents_prediction_prompt)
        input_prompt = prompt.invoke(
            {
                "role": self.role,
                "u_objective": self.u_objective,
                "intro": self.intro,
                "docs": self.documents,
                "instructions": self.instructions,
                "user_tasks": self.user_tasks,
            }
        )
        final_chain = self.model | StrOutputParser()
        answer = final_chain.invoke(input_prompt)
        new_format_tasks = postprocess_json(answer)

        self.tasks.extend(new_format_tasks)

    def _generate_best_practice(self, task):
        """Generate best practices for a given task.

        This function analyzes a task and generates best practices for its execution.
        It uses the language model to identify optimal approaches and strategies.

        Args:
            task (Dict[str, Any]): The task to generate best practices for.

        Returns:
            Dict[str, Any]: Generated best practices for the task.
        """
        # Best practice detection
        resources = {}
        for _, worker_info in self.workers.items():
            worker_name = worker_info["name"]
            worker_desc = worker_info["description"]
            worker_func = worker_info["execute"]
            # Retrieve all methods of the class
            skeleton = {}
            for name, method in inspect.getmembers(
                worker_func, predicate=inspect.isfunction
            ):
                signature = inspect.signature(method)
                skeleton[name] = str(signature)
            worker_resource = worker_desc + "\n"
            worker_resource += (
                "The class skeleton of the worker is as follow: \n"
                + "\n".join(
                    [f"{name}{parameters}" for name, parameters in skeleton.items()]
                )
                + "\n\n"
            )
            logger.debug(f"Code skeleton of the worker: {worker_resource}")

            resources[worker_name] = worker_resource
        for _, tool_info in self.tools.items():
            tool_name = tool_info["name"]
            tool_desc = tool_info["description"]
            resources[tool_name] = tool_desc

        for task_name, task_info in self.reusable_tasks.items():
            resources[task_name] = task_info["nestedgraph_task"]

        resources_str = "\n".join(
            [f"{name}\n: {desc}" for name, desc in resources.items()]
        )
        prompt = PromptTemplate.from_template(check_best_practice_sys_prompt)
        input_prompt = prompt.invoke(
            {"task": task["task"], "level": "1", "resources": resources_str}
        )
        final_chain = self.model | StrOutputParser()
        answer = final_chain.invoke(input_prompt)
        logger.info(f"Best practice detection: {answer}")
        answer = postprocess_json(answer)

        if not answer or answer["answer"].lower() == "no":
            best_practice = [{"step": 1, "task": task["task"]}]
            return best_practice

        # Best practice suggestion
        prompt = PromptTemplate.from_template(generate_best_practice_sys_prompt)
        input_prompt = prompt.invoke(
            {
                "role": self.role,
                "u_objective": self.u_objective,
                "task": task["task"],
                "resources": resources_str,
                "instructions": self.instructions,
                "example_conversations": self.example_conversations,
            }
        )
        final_chain = self.model | StrOutputParser()
        answer = final_chain.invoke(input_prompt)
        logger.debug(f"Generated best practice with thought: {answer}")
        return postprocess_json(answer)

    def _finetune_best_practice(self, best_practice):
        """Finetune best practices based on the builder's objective.

        This function adjusts the generated best practices to align with the builder's
        objectives and requirements.

        Args:
            best_practice (Dict[str, Any]): The best practice to finetune.

        Returns:
            Dict[str, Any]: Finetuned best practice.
        """
        # embed build's objective
        if not self.b_objective:
            prompt = PromptTemplate.from_template(embed_builder_obj_sys_prompt)
            input_prompt = prompt.invoke(
                {"best_practice": best_practice, "b_objective": self.b_objective}
            )
            final_chain = self.model | StrOutputParser()
            best_practice = postprocess_json(final_chain.invoke(input_prompt))
        # mapping resources to the best practice
        prompt = PromptTemplate.from_template(embed_resources_sys_prompt)
        resources = {}
        resource_id_map = {}
        for worker_id, worker_info in self.workers.items():
            worker_name = worker_info["name"]
            worker_desc = worker_info["description"]
            resources[worker_name] = worker_desc
            resource_id_map[worker_name] = worker_id

        for tool_id, tool_info in self.tools.items():
            tool_name = tool_info["name"]
            tool_desc = tool_info["description"]
            resources[tool_name] = tool_desc
            resource_id_map[tool_name] = tool_id

        for task_name, task_info in self.reusable_tasks.items():
            resources[task_name] = task_info["nestedgraph_task"]
            resource_id_map[task_name] = NESTED_GRAPH_ID

        input_prompt = prompt.invoke(
            {"best_practice": best_practice, "resources": resources}
        )
        final_chain = self.model | StrOutputParser()
        answer = final_chain.invoke(input_prompt)
        json_answer = postprocess_json(answer)
        # add resource id
        for i in range(len(json_answer)):
            ans = json_answer[i]
            resource_name = ans["resource"]
            # we need to enforce the answer to be only one worker or tool
            if isinstance(resource_name, list):
                resource_name = resource_name[0]
            if not isinstance(resource_name, str):
                resource_name = str(resource_name)
            resource_name = resource_name.split(" ")[0]
            resource_id = resource_id_map.get(resource_name, None)

            if not resource_id:
                logger.info("Error while retrieving resource id")
            json_answer[i]["resource_id"] = resource_id
        return json_answer

    def _format_task_graph(self, finetuned_best_practices):
        """Format the task graph with finetuned best practices.

        This function structures the task graph, incorporating the finetuned best
        practices and organizing the tasks and their relationships.

        Args:
            finetuned_best_practices (List[Dict[str, Any]]): List of finetuned best practices.

        Returns:
            Dict[str, Any]: Formatted task graph.
        """
        node_id = 1
        nodes = []
        edges = []
        task_ids = {}
        nested_graph_nodes = []

        resource_id_map = {}
        for worker_id, worker_info in self.workers.items():
            worker_name = worker_info["name"]
            resource_id_map[worker_name] = worker_id
        for tool_id, tool_info in self.tools.items():
            tool_name = tool_info["name"]
            resource_id_map[tool_name] = tool_id
        for task_name, _ in self.reusable_tasks.items():
            resource_id_map[task_name] = NESTED_GRAPH_ID

        for best_practice, task in zip(finetuned_best_practices, self.tasks):
            task_ids[node_id] = task
            for idx, step in enumerate(best_practice):
                resource_name = (
                    step.get("resource")
                    if step.get("resource") in resource_id_map
                    else "MessageWorker"
                )
                resource_id = resource_id_map[resource_name]
                node = []
                node.append(str(node_id))
                node.append(
                    {
                        "resource": {
                            "id": resource_id,
                            "name": resource_name,
                        },
                        "attribute": {
                            "value": step.get("example_response", ""),
                            "task": step.get("task", ""),
                            "directed": False,
                        },
                    }
                )
                if step["resource_id"] == NESTED_GRAPH_ID:
                    # store the index of the nested graph == len(nodes) (not the id of the node)
                    nested_graph_nodes.append(len(nodes))
                nodes.append(node)

                if idx == 0:
                    edge = []
                    edge.append("0")
                    edge.append(str(node_id))
                    edge.append(
                        {
                            "intent": task.get("intent"),
                            "attribute": {
                                "weight": 1,
                                "pred": True,
                                "definition": "",
                                "sample_utterances": [],
                            },
                        }
                    )
                else:
                    edge = []
                    edge.append(str(node_id - 1))
                    edge.append(str(node_id))
                    edge.append(
                        {
                            "intent": "None",
                            "attribute": {
                                "weight": 1,
                                "pred": False,
                                "definition": "",
                                "sample_utterances": [],
                            },
                        }
                    )
                edges.append(edge)
                node_id += 1

        # Nested Graph Format Task Graph
        nested_graph_map = {}
        for node_idx in nested_graph_nodes:
            task_name = nodes[node_idx][1]["resource"]["name"]
            if task_name in nested_graph_map:
                continue
            # store node_id which is the value of the nested graph resource node id
            nested_graph_map[task_name] = node_id
            next_tasks = deque()
            next_tasks.append((self.reusable_tasks[task_name]["subgraph"], None))
            while next_tasks:
                cur_task, prev_node_id = next_tasks.popleft()
                resource_name = (
                    cur_task.get("resource")
                    if cur_task.get("resource") in resource_id_map
                    else "MessageWorker"
                )
                resource_id = resource_id_map[resource_name]
                node = []
                node.append(str(node_id))
                node.append(
                    {
                        "resource": {
                            "id": resource_id,
                            "name": resource_name,
                        },
                        "attribute": {
                            "value": cur_task.get("example_response", ""),
                            "task": cur_task.get("task", ""),
                            "directed": False,
                        },
                    }
                )
                nodes.append(node)
                if prev_node_id is not None:
                    edge = []
                    edge.append(str(prev_node_id))
                    edge.append(str(node_id))
                    edge.append(
                        {
                            "intent": "None",
                            "attribute": {
                                "weight": 1,
                                "pred": False,
                                "definition": "",
                                "sample_utterances": [],
                            },
                        }
                    )
                    edges.append(edge)
                for next_task in cur_task.get("next", []):
                    next_tasks.append((next_task, node_id))
                node_id += 1

        for node_idx in nested_graph_nodes:
            nodes[node_idx][1]["attribute"]["value"] = str(
                nested_graph_map[nodes[node_idx][1]["resource"]["name"]]
            )

        # Add the start node
        start_node = []
        start_node.append("0")
        # generate the start message
        prompt = PromptTemplate.from_template(generate_start_msg)
        input_prompt = prompt.invoke(
            {"role": self.role, "u_objective": self.u_objective}
        )
        final_chain = self.model | StrOutputParser()
        answer = final_chain.invoke(input_prompt)
        start_msg = postprocess_json(answer)

        start_node.append(
            {
                "resource": {
                    "id": resource_id_map.get("MessageWorker"),
                    "name": "MessageWorker",
                },
                "attribute": {
                    "value": start_msg.get("message", ""),
                    "task": "start message",
                    "directed": False,
                },
                "limit": 1,
                "type": "start",
            }
        )
        nodes.insert(0, start_node)

        task_graph = {"nodes": nodes, "edges": edges}

        for key, value in self.product_kwargs.items():
            task_graph[key] = value

        return task_graph

    def _load_docs(self):
        """Load documentation for task graph generation.

        This function loads and processes documentation that will be used to inform
        the task graph generation process.

        Returns:
            Dict[str, Any]: Loaded documentation.
        """
        if self.task_docs:
            filepath = os.path.join(self.output_dir, "task_documents.pkl")
            total_num_docs = sum([doc.get("num", 1) for doc in self.task_docs])
            loader = Loader()
            if Path(filepath).exists():
                logger.warning(
                    f"Loading existing documents from {os.path.join(self.output_dir, 'task_documents.pkl')}! If you want to recrawl, please delete the file or specify a new --output-dir when initiate Generator."
                )
                docs = pickle.load(
                    open(os.path.join(self.output_dir, "task_documents.pkl"), "rb")
                )
            else:
                docs = []
                for doc in self.task_docs:
                    source = doc.get("source")
                    if doc.get("type") == "url":
                        num_docs = doc.get("num") if doc.get("num") else 1
                        urls = loader.get_all_urls(source, num_docs)
                        crawled_urls = loader.to_crawled_url_objs(urls)
                        docs.extend(crawled_urls)
                    elif doc.get("type") == "file":
                        file_list = [
                            os.path.join(source, f) for f in os.listdir(source)
                        ]
                        docs.extend(loader.to_crawled_local_objs(file_list))
                    elif doc.get("type") == "text":
                        docs.extend(loader.to_crawled_text([source]))
                    else:
                        # TODO: how to handle when type is not provided
                        raise Exception(
                            "type must be one of [url, file, text] and it must be provided"
                        )

                Loader.save(filepath, docs)

            limit = max(total_num_docs // 5, 10)

            crawled_docs = []
            web_docs = list(filter(lambda x: x.source_type == SourceType.WEB, docs))
            file_docs = list(filter(lambda x: x.source_type == SourceType.FILE, docs))
            text_docs = list(filter(lambda x: x.source_type == SourceType.TEXT, docs))
            crawled_docs.extend(loader.get_candidates_websites(web_docs, limit))
            crawled_docs.extend(file_docs)
            crawled_docs.extend(text_docs)

            logger.debug(f"Loaded {len(crawled_docs)} documents")
            self.documents = "\n\n".join(
                [f"{doc.source}\n{doc.content}" for doc in crawled_docs]
            )
        else:
            self.documents = ""

    def _load_instructions(self):
        """Load instructions for task graph generation.

        This function loads and processes instructions that will guide the task graph
        generation process.

        Returns:
            Dict[str, Any]: Loaded instructions.
        """
        instructions = []
        if not self.instruction_docs:
            self.instructions = ""
            return
        limit = len(self.instruction_docs)
        for doc in self.instruction_docs:
            loader = Loader()
            source = doc.get("source")
            if doc.get("type") == "url":
                num_docs = doc.get("num") if doc.get("num") else 1
                urls = loader.get_all_urls(source, num_docs)
                crawled_urls = loader.to_crawled_url_objs(urls)
                instructions.extend(crawled_urls)
            elif doc.get("type") == "file":
                file_list = [os.path.join(source, f) for f in os.listdir(source)]
                instructions.extend(loader.to_crawled_local_objs(file_list))
            elif doc.get("type") == "text":
                instructions.extend(loader.to_crawled_text([source]))
            else:
                # TODO: how to handle when type is not provided
                raise Exception(
                    "type must be one of [url, file, text] and it must be provided"
                )

        crawled_docs = []
        web_docs = list(filter(lambda x: x.source_type == SourceType.WEB, instructions))
        file_docs = list(
            filter(lambda x: x.source_type == SourceType.FILE, instructions)
        )
        text_docs = list(
            filter(lambda x: x.source_type == SourceType.TEXT, instructions)
        )
        crawled_docs.extend(loader.get_candidates_websites(web_docs, limit))
        crawled_docs.extend(file_docs)
        crawled_docs.extend(text_docs)
        logger.debug(f"Loaded {len(crawled_docs)} instruction documents")
        self.instructions = "\n\n".join([f"{doc.content}" for doc in crawled_docs])

    def generate(self) -> dict:
        """Generate a complete task graph.

        This function orchestrates the task graph generation process, loading
        documentation and instructions, generating tasks and best practices,
        and formatting the final task graph.

        Returns:
            dict: The generated task graph.
        """
        # Load the docs for task graph
        self._load_docs()

        # Load the instructions
        self._load_instructions()

        # Add tasks provided by users
        self._add_provided_tasks()

        # Generate tasks
        self._generate_tasks()

        if self.allow_nested_graph:
            self._generate_reusable_tasks()

        # Step 2: Generate the task planning
        best_practices = []
        for idx, task in progress_bar(enumerate(self.tasks), total=len(self.tasks)):
            logger.info(f"Generating best practice for task {idx}: {task}")
            best_practice = self._generate_best_practice(task)
            logger.info(f"Generated best practice for task {idx}: {best_practice}")
            best_practices.append(best_practice)

        # Step 3: iterate with user
        format_tasks = []
        for best_practice, task in zip(best_practices, self.tasks):
            try:
                task_name = task["task"]
                steps = [bp["task"] for bp in best_practice]
            except Exception as e:
                logger.error(f"Error in format task {task}")
                logger.error(e)
                continue
            format_tasks.append({"task_name": task_name, "steps": steps})

        hitl_result = format_tasks
        if self.interactable_with_user:
            app = TaskEditorApp(hitl_result)
            hitl_result = app.run()
        if self.output_dir:
            task_planning_filepath = os.path.join(self.output_dir, f"taskplanning.json")
            json.dump(hitl_result, open(task_planning_filepath, "w"), indent=4)

        # Step 4: Pair task with worker
        finetuned_best_practices = []
        for idx_t, task in enumerate(hitl_result):
            steps = task["steps"]
            format_steps = []
            for idx_s, step in enumerate(steps):
                format_steps.append({"step": idx_s + 1, "task": step})
            finetuned_best_practice = self._finetune_best_practice(format_steps)
            logger.info(
                f"Finetuned best practice for task {idx_t}: {finetuned_best_practice}"
            )
            finetuned_best_practices.append(finetuned_best_practice)

        # Step 5: Format the task graph
        task_graph = self._format_task_graph(finetuned_best_practices)

        return task_graph

    def save_task_graph(self, task_graph) -> str:
        """Save the task graph to a file.

        This function saves the generated task graph to a JSON file in the output
        directory.

        Args:
            task_graph (dict): The task graph to save.

        Returns:
            str: Path to the saved task graph file.
        """
        taskgraph_filepath = os.path.join(self.output_dir, f"taskgraph.json")
        with open(taskgraph_filepath, "w") as f:
            json.dump(task_graph, f, indent=4)
        return taskgraph_filepath
