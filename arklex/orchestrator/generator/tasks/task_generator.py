"""Task generation component for the Arklex framework.

This module provides the TaskGenerator class that handles generation of tasks
from objectives and documentation. It manages task hierarchy, relationships, and
validation to create well-structured task definitions.

Key Features:
- Task generation from objectives and documentation
- Task hierarchy and relationship management
- Task validation and refinement
- Support for user-provided tasks
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from arklex.orchestrator.generator.prompts import PromptManager
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


@dataclass
class TaskDefinition:
    """Represents a task definition with its properties.

    Attributes:
        task_id (str): Unique identifier for the task
        name (str): Name of the task
        description (str): Detailed description of the task
        steps (List[Dict[str, Any]]): List of steps to complete the task
        dependencies (List[str]): List of task IDs this task depends on
        required_resources (List[str]): List of resources required for the task
        estimated_duration (Optional[str]): Estimated duration as a string (e.g., "1 hour")
        priority (int): Task priority (1-5, where 5 is highest)
    """

    task_id: str
    name: str
    description: str
    steps: list[dict[str, Any]]
    dependencies: list[str]
    required_resources: list[str]
    estimated_duration: str | None
    priority: int


class TaskGenerator:
    """Generates and manages tasks based on objectives and documentation.

    This class handles the generation of tasks from user objectives and documentation,
    ensuring proper task hierarchy, relationships, and validation. It processes input
    data to create well-structured task definitions that can be used by the orchestrator.

    The generator follows a modular design pattern where each major responsibility
    is handled by a dedicated method, promoting maintainability and testability.

    Attributes:
        model: The language model used for task generation
        role (str): The role or context for task generation
        user_objective (str): User's objective for the task graph
        instructions (str): Processed instruction documents
        documents (str): Processed task documents
        _task_definitions (Dict[str, TaskDefinition]): Cache of generated task definitions
        _task_hierarchy (Dict[str, List[str]]): Task hierarchy relationships
        prompt_manager: The prompt manager for generating prompts

    Methods:
        generate_tasks(): Main method to generate tasks
        add_provided_tasks(): Add user-provided tasks
        _process_objective(): Process user objective
        _generate_task_definitions(): Generate task definitions
        _validate_tasks(): Validate generated tasks
        _establish_relationships(): Establish task relationships
        _build_hierarchy(): Build task hierarchy
    """

    def __init__(
        self,
        model: Callable[..., str],
        role: str,
        user_objective: str,
        instructions: str,
        documents: str,
    ) -> None:
        """Initialize the TaskGenerator with required components.

        Args:
            model: The language model to use for task generation
            role (str): The role or context for task generation
            user_objective (str): User's objective for the task graph
            instructions (str): Processed instruction documents
            documents (str): Processed task documents
        """
        self.model = model
        self.role = role
        self.user_objective = user_objective
        self.instructions = instructions
        self.documents = documents
        self._task_definitions: dict[str, TaskDefinition] = {}
        self._task_hierarchy: dict[str, list[str]] = {}
        self.prompt_manager = PromptManager()  # Initialize prompt manager

    def generate_tasks(
        self, intro: str, existing_tasks: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """Generate tasks from the introduction and existing tasks.

        This method implements the original three-step process:
        1. Generate high-level tasks using generate_tasks_sys_prompt
        2. Check if tasks need further breakdown using check_best_practice_sys_prompt
        3. Generate steps for tasks that need breakdown using generate_best_practice_sys_prompt

        Args:
            intro (str): Introduction or documentation string.
            existing_tasks (Optional[List[Dict[str, Any]]]): List of existing tasks, if any.

        Returns:
            List[Dict[str, Any]]: List of validated and structured tasks.
        """
        if existing_tasks is None:
            existing_tasks = []

        log_context.info(
            "    📝 Step 1: Generating high-level tasks using generate_tasks_sys_prompt..."
        )

        # Step 1: Generate high-level tasks using the original generate_tasks_sys_prompt
        high_level_tasks = self._generate_high_level_tasks(intro, existing_tasks)
        log_context.info(f"    📋 Generated {len(high_level_tasks)} high-level tasks")

        log_context.info("    🔍 Step 2: Analyzing tasks for breakdown requirements...")
        tasks_with_steps = []

        for i, task in enumerate(high_level_tasks, 1):
            task_name = task.get("task", "")
            task_intent = task.get("intent", "")

            log_context.info(
                f"      🔍 Analyzing task {i}/{len(high_level_tasks)}: {task_name}"
            )

            # Step 2: Check if this task needs further breakdown using check_best_practice_sys_prompt
            needs_breakdown = self._check_task_breakdown_original(
                task_name, task_intent
            )

            if needs_breakdown:
                log_context.info(f"      📝 Generating detailed steps for: {task_name}")
                # Step 3: Generate steps for tasks that need breakdown using generate_best_practice_sys_prompt
                steps = self._generate_task_steps_original(task_name, task_intent)
                task["steps"] = steps
                log_context.info(
                    f"      ✅ Generated {len(steps)} steps for: {task_name}"
                )
            else:
                log_context.info(f"      ✅ Task is actionable: {task_name}")
                # Task is already actionable, create a simple step
                task["steps"] = [{"task": f"Execute {task_name}"}]

            tasks_with_steps.append(task)

        log_context.info(
            "    🏗️ Step 3: Converting to task definitions and validating..."
        )
        # Convert to task definitions and validate
        task_definitions = self._convert_to_task_definitions(tasks_with_steps)
        tasks = []
        for i, task_def in enumerate(task_definitions):
            # Pass the original task data to preserve intent field
            original_task_data = (
                tasks_with_steps[i] if i < len(tasks_with_steps) else None
            )
            task_dict = self._convert_to_dict(task_def, original_task_data)
            task_dict["id"] = f"task_{i + 1}"
            tasks.append(task_dict)

        log_context.info("    ✅ Validating and building task hierarchy...")
        validated_tasks = self._validate_tasks(tasks)
        self._build_hierarchy(validated_tasks)
        log_context.info(
            f"    🎉 Successfully generated {len(validated_tasks)} validated tasks"
        )
        return validated_tasks

    def add_provided_tasks(
        self, user_tasks: list[dict[str, Any]], intro: str
    ) -> list[dict[str, Any]]:
        """Add user-provided tasks to the task set.

        This method processes and validates user-provided tasks, ensuring they
        meet the required format and quality standards. If tasks don't have steps,
        it will generate steps for them.

        Args:
            user_tasks (List[Dict[str, Any]]): List of user-provided tasks
            intro (str): Introduction text for context

        Returns:
            List[Dict[str, Any]]: List of processed and validated tasks
        """
        processed_tasks: list[dict[str, Any]] = []
        for task in user_tasks:
            try:
                # Check if this is a simple task description without steps
                if "task" in task and not task.get("steps"):
                    # This is a simple task description, need to add steps
                    task_name = task.get("task", "")
                    task_intent = task.get("intent", "")

                    # Check if task needs breakdown
                    needs_breakdown = self._check_task_breakdown_original(
                        task_name, task_intent
                    )

                    if needs_breakdown:
                        # Generate detailed steps
                        steps = self._generate_task_steps_original(
                            task_name, task_intent
                        )
                        task["steps"] = steps
                        log_context.info(
                            f"Generated {len(steps)} steps for task: {task_name}"
                        )
                    else:
                        # Task is actionable, create a simple step
                        task["steps"] = [{"task": f"Execute {task_name}"}]
                        log_context.info(f"Created simple step for task: {task_name}")

                    # Add name field if not present
                    if not task.get("name"):
                        task["name"] = task_name

                # Now create task definition with steps
                task_def = TaskDefinition(
                    task_id=task.get("id", ""),
                    name=task.get("name", ""),
                    description=task.get("description", ""),
                    steps=task.get("steps", []),
                    dependencies=task.get("dependencies", []),
                    required_resources=task.get("required_resources", []),
                    estimated_duration=task.get("estimated_duration"),
                    priority=task.get("priority", 3),
                )
                if self._validate_task_definition(task_def):
                    processed_tasks.append(task)
                    log_context.info(f"Added user-provided task: {task_def.name}")
                else:
                    log_context.warning(f"Invalid user-provided task: {task_def.name}")
            except Exception as e:
                log_context.error(f"Error processing user task: {str(e)}")
                continue
        return processed_tasks

    def _process_objective(
        self,
        objective: str,
        intro: str,
        docs: str,
        existing_tasks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Process objective and documentation to generate tasks.

        Args:
            objective (str): User objective.
            intro (str): Introduction text.
            docs (str): Documentation text.
            existing_tasks (Optional[List[Dict[str, Any]]]): Optional list of existing tasks.

        Returns:
            Dict[str, Any]: Dictionary containing processed objective and tasks.
        """
        existing_tasks_json = ""
        if existing_tasks:
            existing_tasks_json = json.dumps(existing_tasks, indent=2)
        prompt = self.prompt_manager.generate_tasks_sys_prompt.format(
            role=self.role,
            u_objective=objective,
            intro=intro,
            docs=docs,
            instructions=self.instructions,
            existing_tasks=existing_tasks_json,
        )
        try:
            from langchain_core.messages import HumanMessage

            messages = [HumanMessage(content=prompt)]
        except ImportError:
            messages = [{"role": "user", "content": prompt}]

        try:
            response = self.model.generate([messages])
        except Exception as e:
            log_context.error(f"Error generating tasks from objective: {e}")
            return {"tasks": []}

        response_text: str | None = None
        if hasattr(response, "generations"):
            if hasattr(response.generations[0][0], "text"):
                response_text = response.generations[0][0].text
            elif hasattr(response.generations[0][0], "message") and hasattr(
                response.generations[0][0].message, "content"
            ):
                response_text = response.generations[0][0].message.content
        elif isinstance(response, dict) and "text" in response:
            response_text = response["text"]
        elif isinstance(response, dict) and "content" in response:
            response_text = response["content"]
        else:
            response_text = str(response)
        try:
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                tasks_data = json.loads(json_str)
                log_context.info(
                    f"Generated {len(tasks_data)} tasks from documentation"
                )
            else:
                log_context.error("No valid JSON array found in response")
                tasks_data = []
        except json.JSONDecodeError as e:
            log_context.error(f"Failed to parse generated tasks: {e}")
            tasks_data = []
        return {"tasks": tasks_data}

    def _generate_high_level_tasks(
        self, intro: str, existing_tasks: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """Generate high-level tasks using the original generate_tasks_sys_prompt.

        This is Step 1 of the original 3-step process.

        Args:
            intro (str): Introduction or documentation string.
            existing_tasks (Optional[List[Dict[str, Any]]]): List of existing tasks.

        Returns:
            List[Dict[str, Any]]: List of high-level tasks.
        """
        if existing_tasks is None:
            existing_tasks = []

        # Format existing tasks for the prompt
        existing_tasks_str = ""
        if existing_tasks:
            existing_tasks_str = "\n".join(
                [
                    f"- {task.get('task', '')}: {task.get('intent', '')}"
                    for task in existing_tasks
                ]
            )

        # Use the original generate_tasks_sys_prompt
        prompt = self.prompt_manager.generate_tasks_sys_prompt.format(
            role=self.role,
            u_objective=self.user_objective,
            intro=intro,
            docs=self.documents,
            instructions=self.instructions,
            existing_tasks=existing_tasks_str,
        )

        try:
            response = self.model.invoke(prompt)
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                tasks_data = json.loads(json_str)
                return tasks_data
            else:
                log_context.warning("Could not parse high-level tasks response")
                return []
        except Exception as e:
            log_context.error(f"Error generating high-level tasks: {e}")
            return []

    def _check_task_breakdown_original(self, task_name: str, task_intent: str) -> bool:
        """Check if a task needs further breakdown using the original check_best_practice_sys_prompt.

        This is Step 2 of the original 3-step process.

        Args:
            task_name (str): Name of the task
            task_intent (str): Intent of the task

        Returns:
            bool: True if task needs breakdown, False if it's already actionable
        """
        # Create a proper resource list for the check
        prompt = self.prompt_manager.check_best_practice_sys_prompt.format(
            task=task_name,
            level=1,
            resources="MessageWorker: The worker responsible for interacting with the user with predefined responses, RAGWorker: Answer the user's questions based on the company's internal documentations, such as the policies, FAQs, and product information",
        )

        try:
            response = self.model.invoke(prompt)
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                answer = result.get("answer", "No").lower()
                return answer == "yes"
            else:
                log_context.warning(
                    f"Could not parse breakdown check response for task: {task_name}"
                )
                return True  # Default to breakdown if we can't parse

        except Exception as e:
            log_context.error(f"Error checking task breakdown for {task_name}: {e}")
            return True  # Default to breakdown on error

    def _generate_task_steps_original(
        self, task_name: str, task_intent: str
    ) -> list[dict[str, Any]]:
        """Generate steps for a task using the original generate_best_practice_sys_prompt.

        This is Step 3 of the original 3-step process.

        Args:
            task_name (str): Name of the task
            task_intent (str): Intent of the task

        Returns:
            List[Dict[str, Any]]: List of steps for the task
        """
        prompt = self.prompt_manager.generate_best_practice_sys_prompt.format(
            role=self.role,
            u_objective=self.user_objective,
            task=task_name,
            resources="MessageWorker: The worker responsible for interacting with the user with predefined responses, RAGWorker: Answer the user's questions based on the company's internal documentations, such as the policies, FAQs, and product information",
            instructions=self.instructions,
            example_conversations="",
        )

        try:
            response = self.model.invoke(prompt)
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                steps_data = json.loads(json_str)
                # Convert the step format to the expected format
                converted_steps = []
                for step in steps_data:
                    if isinstance(step, dict) and "task" in step:
                        # Ensure description exists, use task as fallback
                        description = step.get(
                            "description", step.get("task", "Execute step")
                        )
                        converted_steps.append(
                            {"task": step["task"], "description": description}
                        )
                    elif isinstance(step, dict) and "step" in step:
                        # Handle alternative step format
                        task = step.get("task", step.get("step", "Execute step"))
                        description = step.get("description", task)
                        converted_steps.append(
                            {"task": task, "description": description}
                        )
                    elif isinstance(step, str):
                        # Handle string steps
                        converted_steps.append(
                            {"task": step, "description": f"Execute: {step}"}
                        )
                return converted_steps
            else:
                log_context.warning(
                    f"Could not parse steps response for task: {task_name}"
                )
                return [
                    {
                        "task": f"Execute {task_name}",
                        "description": f"Execute the task: {task_name}",
                    }
                ]

        except Exception as e:
            log_context.error(f"Error generating steps for {task_name}: {e}")
            return [
                {
                    "task": f"Execute {task_name}",
                    "description": f"Execute the task: {task_name}",
                }
            ]

    def _convert_to_task_definitions(
        self, tasks_with_steps: list[dict[str, Any]]
    ) -> list[TaskDefinition]:
        """Convert tasks with steps to TaskDefinition objects.

        Args:
            tasks_with_steps (List[Dict[str, Any]]): List of tasks with steps

        Returns:
            List[TaskDefinition]: List of TaskDefinition objects
        """
        task_definitions: list[TaskDefinition] = []
        for i, task_data in enumerate(tasks_with_steps):
            steps = task_data.get("steps", [])
            if not steps:
                steps = [
                    {
                        "task": f"Execute {task_data.get('task', '')}",
                        "description": f"Execute the task: {task_data.get('task', '')}",
                    }
                ]
            elif isinstance(steps, list) and steps and isinstance(steps[0], str):
                steps = [
                    {"task": step, "description": f"Execute step: {step}"}
                    for step in steps
                ]

            task_def = TaskDefinition(
                task_id=f"task_{i + 1}",
                name=task_data.get("task", ""),
                description=task_data.get("description", task_data.get("intent", "")),
                steps=steps,
                dependencies=task_data.get("dependencies", []),
                required_resources=task_data.get("required_resources", []),
                estimated_duration=task_data.get("estimated_duration", "1 hour"),
                priority=task_data.get("priority", 3),
            )
            task_definitions.append(task_def)
        return task_definitions

    def _validate_tasks(self, tasks: list[Any]) -> list[dict[str, Any]]:
        """Validate task definitions.

        Args:
            tasks (List[Any]): List of tasks to validate (TaskDefinition or dict).

        Returns:
            List[Dict[str, Any]]: List of validated tasks.
        """
        validated_tasks: list[dict[str, Any]] = []
        for task in tasks:
            if hasattr(task, "__dataclass_fields__"):
                task = self._convert_to_dict(task)
            if not all(field in task for field in ["name", "description", "steps"]):
                log_context.warning(f"Task missing required fields: {task}")
                continue
            if not isinstance(task["steps"], list):
                log_context.warning(f"Task steps must be a list: {task}")
                continue
            if task["steps"] and isinstance(task["steps"][0], str):
                task["steps"] = [
                    {"task": step, "description": f"Execute step: {step}"}
                    for step in task["steps"]
                ]
            if not all(
                isinstance(step, dict) and "task" in step for step in task["steps"]
            ):
                log_context.warning(
                    f"Task steps must be dictionaries with 'task' key: {task}"
                )
                continue
            # Ensure all steps have descriptions
            for step in task["steps"]:
                if "description" not in step or not step["description"].strip():
                    step["description"] = f"Execute: {step.get('task', 'Unknown step')}"
            if "dependencies" not in task or not isinstance(task["dependencies"], list):
                task["dependencies"] = []
            if "required_resources" not in task or not isinstance(
                task["required_resources"], list
            ):
                task["required_resources"] = []
            if "estimated_duration" not in task or not isinstance(
                task["estimated_duration"], str
            ):
                task["estimated_duration"] = "1 hour"
            if (
                "priority" not in task
                or not isinstance(task["priority"], int | float)
                or not 1 <= task["priority"] <= 5
            ):
                task["priority"] = 3
            validated_tasks.append(task)
        return validated_tasks

    def _validate_task_definition(self, task_def: TaskDefinition) -> bool:
        """Validate a single task definition.

        This method checks if a task definition meets all required criteria.

        Args:
            task_def (TaskDefinition): Task definition to validate

        Returns:
            bool: True if task definition is valid, False otherwise
        """
        if not all([task_def.name, task_def.description, task_def.steps]):
            return False
        for step in task_def.steps:
            if not isinstance(step, dict) or "task" not in step:
                return False
        return 1 <= task_def.priority <= 5

    def _establish_relationships(self, tasks: list[dict[str, Any]]) -> None:
        """Establish relationships between tasks.

        This method analyzes task dependencies and establishes proper
        relationships between tasks.

        Args:
            tasks (List[Dict[str, Any]]): List of tasks to process
        """
        # Implementation details...

    def _build_hierarchy(self, tasks: list[dict[str, Any]]) -> None:
        """Build task hierarchy based on dependencies.

        Args:
            tasks (List[Dict[str, Any]]): List of tasks to organize.
        """
        graph: dict[str, Any] = {}
        for task in tasks:
            task_id = task.get("id") or task.get("task_id")
            if not task_id:
                log_context.warning(f"Task missing 'id' or 'task_id': {task}")
                continue
            graph[task_id] = {
                "task": task,
                "dependencies": set(task.get("dependencies", [])),
                "level": 0,
            }
        for _task_id, node in graph.items():
            if not node["dependencies"]:
                node["level"] = 0
            else:
                max_dep_level = 0
                for dep_id in node["dependencies"]:
                    if dep_id in graph:
                        max_dep_level = max(max_dep_level, graph[dep_id]["level"])
                node["level"] = max_dep_level + 1
        for task in tasks:
            task_id = task.get("id") or task.get("task_id")
            if task_id in graph:
                task["level"] = graph[task_id]["level"]
            else:
                task["level"] = 0
        tasks.sort(key=lambda x: x.get("level", 0))

    def _convert_to_dict(
        self,
        task_def: TaskDefinition,
        original_task_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert a TaskDefinition to a dictionary.

        Args:
            task_def (TaskDefinition): Task definition to convert.
            original_task_data (Optional[Dict[str, Any]]): Original task data to preserve additional fields like intent.

        Returns:
            Dict[str, Any]: Dictionary representation of the task.
        """
        result = {
            "id": task_def.task_id,
            "name": task_def.name,
            "description": task_def.description,
            "steps": task_def.steps,
            "dependencies": task_def.dependencies,
            "required_resources": task_def.required_resources,
            "estimated_duration": task_def.estimated_duration,
            "priority": task_def.priority,
        }

        # Preserve intent field from original task data if available
        if original_task_data and "intent" in original_task_data:
            result["intent"] = original_task_data["intent"]

        return result

    def _convert_to_task_dict(
        self, task_definitions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Convert task definitions to task dictionary format."""
        tasks: dict[str, Any] = {}
        for task_def in task_definitions:
            task_id = f"task_{len(tasks) + 1}"
            tasks[task_id] = {
                "name": task_def["task"],
                "description": task_def["intent"],
                "steps": task_def.get("steps", []),
                "dependencies": [],
                "required_resources": [],
                "estimated_duration": "1 hour",
                "priority": "high",
            }
        return tasks
