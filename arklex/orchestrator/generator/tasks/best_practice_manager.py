"""Best practice management component for the Arklex framework.

This module provides the BestPracticeManager class that handles generation and
management of best practices for task execution. It ensures quality and consistency
in task execution through practice refinement and optimization.

Key Features:
- Best practice generation from tasks
- Practice refinement through feedback
- Practice categorization and optimization
- Quality assurance and validation
"""

from dataclasses import dataclass
from typing import Any

from arklex.orchestrator.generator.prompts import PromptManager
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


@dataclass
class BestPractice:
    """Represents a best practice for task execution.

    Attributes:
        practice_id (str): Unique identifier for the practice.
        name (str): Name of the practice.
        description (str): Detailed description of the practice.
        steps (List[Dict[str, Any]]): List of steps to follow.
        rationale (str): Explanation of why this practice is important.
        examples (List[str]): Example applications of the practice.
        priority (int): Practice priority (1-5, where 5 is highest).
        category (str): Category of the practice (e.g., 'efficiency', 'quality').
    """

    practice_id: str
    name: str
    description: str
    steps: list[dict[str, Any]]
    rationale: str
    examples: list[str]
    priority: int
    category: str


class BestPracticeManager:
    """Manages the generation and refinement of best practices for task execution.

    This class handles the creation and management of best practices for task
    execution, ensuring high-quality and consistent task performance. It
    processes task information to create well-structured practice guidelines
    that can be used by the orchestrator.

    The manager follows a modular design pattern where each major responsibility
    is handled by a dedicated method, promoting maintainability and testability.

    Attributes:
        model: The language model used for practice generation
        role (str): The role or context for practice generation
        user_objective (str): User's objective for the task graph
        _practices (Dict[str, BestPractice]): Cache of generated practices
        _practice_categories (Dict[str, List[str]]): Practice categorization
        _workers (List[Dict[str, Any]]): List of available workers from config
        _tools (List[Dict[str, Any]]): List of available tools from config
        _all_resources (List[Dict[str, Any]]): List of all available resources including nested_graph
        prompt_manager: Instance of PromptManager for prompt generation

    Methods:
        generate_best_practices(): Main method to generate practices
        finetune_best_practice(): Refine a practice based on feedback
        _generate_practice_definitions(): Generate practice definitions
        _validate_practices(): Validate generated practices
        _categorize_practices(): Categorize practices by type
        _optimize_practices(): Optimize practices for efficiency
    """

    def __init__(
        self,
        model: object,
        role: str,
        user_objective: str,
        workers: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        all_resources: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the BestPracticeManager with required components.

        Args:
            model: The language model to use for practice generation
            role (str): The role or context for practice generation
            user_objective (str): User's objective for the task graph
            workers (Optional[List[Dict[str, Any]]]): List of available workers from config
            tools (Optional[List[Dict[str, Any]]]): List of available tools from config
            all_resources (Optional[List[Dict[str, Any]]]): List of all available resources including nested_graph
        """
        self.model = model
        self.role = role
        self.user_objective = user_objective
        self._workers = workers or []
        self._tools = tools or []
        self._all_resources = all_resources or []
        self._practices: dict[str, BestPractice] = {}
        self._practice_categories: dict[str, list[str]] = {}
        self.prompt_manager = PromptManager()

    def generate_best_practices(
        self, tasks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate best practices for the given tasks.

        This method orchestrates the practice generation process by:
        1. Generating practice definitions
        2. Validating practices
        3. Categorizing practices
        4. Optimizing practices

        Args:
            tasks (List[Dict[str, Any]]): List of tasks to generate practices for

        Returns:
            List[Dict[str, Any]]: List of generated best practices
        """
        log_context.info(
            f"    📋 Creating practice definitions for {len(tasks)} tasks..."
        )
        practice_definitions = self._generate_practice_definitions(tasks)
        log_context.info(
            f"    ✅ Generated {len(practice_definitions)} practice definitions"
        )

        log_context.info("    🔍 Validating practice definitions...")
        validated_practices = self._validate_practices(practice_definitions)
        log_context.info(f"    ✅ Validated {len(validated_practices)} practices")

        log_context.info("    📂 Categorizing practices by type...")
        self._categorize_practices(validated_practices)
        log_context.info("    ✅ Categorized practices")

        log_context.info("    ⚡ Optimizing practices for efficiency...")
        optimized_practices = self._optimize_practices(validated_practices)
        log_context.info("    ✅ Optimized practices")

        return optimized_practices

    def finetune_best_practice(
        self, practice: dict[str, Any], task: dict[str, Any]
    ) -> dict[str, Any]:
        """Refine a best practice based on feedback and task information.

        This method takes a practice and a task, and refines the practice by
        pairing the steps with available resources (workers, tools, nested_graph) from the config.

        Args:
            practice (Dict[str, Any]): The practice to refine
            task (Dict[str, Any]): The task information to use for refinement

        Returns:
            Dict[str, Any]: Refined practice with resource mappings
        """
        try:
            # Get the steps from the task
            steps = task.get("steps", [])
            if not steps:
                log_context.warning("No steps found in task for resource pairing")
                return practice

            # Create a list of available resources for round-robin assignment
            available_resources = []

            # Add workers
            if self._workers:
                for worker in self._workers:
                    if isinstance(worker, dict) and "name" in worker:
                        available_resources.append(
                            {
                                "name": worker["name"],
                                "description": worker.get(
                                    "description", f"{worker['name']} worker"
                                ),
                                "type": "worker",
                            }
                        )

            # Add tools
            if self._tools:
                for tool in self._tools:
                    if isinstance(tool, dict) and "name" in tool:
                        available_resources.append(
                            {
                                "name": tool["name"],
                                "description": tool.get(
                                    "description", f"{tool['name']} tool"
                                ),
                                "type": "tool",
                            }
                        )

            # Add nested_graph if available in all_resources
            if self._all_resources:
                for resource in self._all_resources:
                    if (
                        isinstance(resource, dict)
                        and resource.get("type") == "nested_graph"
                    ):
                        available_resources.append(resource)

            # If no resources from config, use default ones
            if not available_resources:
                available_resources = [
                    {
                        "name": "MessageWorker",
                        "description": "The worker responsible for interacting with the user with predefined responses",
                        "type": "worker",
                    },
                    {
                        "name": "FaissRAGWorker",
                        "description": "Answer the user's questions based on the company's internal documentations, such as the policies, FAQs, and product information",
                        "type": "worker",
                    },
                    {
                        "name": "SearchWorker",
                        "description": "Search for information from external sources and databases",
                        "type": "worker",
                    },
                ]

            # Use round-robin assignment to ensure all resource types are used
            refined_steps = []

            for resource_index, step in enumerate(steps):
                # Get step description
                step_description = step.get("description", "")
                if isinstance(step, str):
                    step_description = step

                # Assign resource in round-robin fashion
                assigned_resource = available_resources[
                    resource_index % len(available_resources)
                ]

                # Create refined step with resource assignment
                refined_step = {
                    "task": step_description,
                    "description": step_description,
                    "step_id": step.get("step_id", f"step_{len(refined_steps) + 1}"),
                    "required_fields": step.get("required_fields", []),
                    "resource": {
                        "name": assigned_resource["name"],
                        "description": assigned_resource["description"],
                        "type": assigned_resource.get("type", "worker"),
                    },
                }
                refined_steps.append(refined_step)

            # Update the practice with the refined steps
            practice["steps"] = refined_steps
            log_context.info(
                f"Successfully refined practice with resource mappings using {len(available_resources)} resources"
            )
            return practice

        except Exception as e:
            log_context.error(f"Error refining practice: {str(e)}")
            return practice

    def _generate_practice_definitions(
        self, tasks: list[dict[str, Any]]
    ) -> list[BestPractice]:
        """Generate practice definitions from tasks.

        This method creates practice definitions based on the tasks,
        ensuring proper structure and quality.

        Args:
            tasks (List[Dict[str, Any]]): List of tasks to process

        Returns:
            List[BestPractice]: List of generated practice definitions
        """
        practice_definitions: list[BestPractice] = []
        for i, task in enumerate(tasks):
            practice_def = BestPractice(
                practice_id=f"practice_{i + 1}",
                name=f"Best Practice for {task['name']}",
                description=f"Best practices for executing {task['name']}",
                steps=task.get("steps", []),
                rationale="Ensures consistent and high-quality task execution",
                examples=[],
                priority=task.get("priority", 3),
                category=task.get("category", "general"),
            )
            practice_definitions.append(practice_def)
        return practice_definitions

    def _validate_practices(
        self, practice_definitions: list[BestPractice]
    ) -> list[dict[str, Any]]:
        """Validate generated practice definitions.

        This method ensures that all practice definitions meet the required
        format and quality standards.

        Args:
            practice_definitions (List[BestPractice]): List of practice definitions

        Returns:
            List[Dict[str, Any]]: List of validated practices with the following structure:
                {
                    "practice_id": str,
                    "name": str,
                    "description": str,
                    "steps": List[Dict[str, Any]],
                    "rationale": str,
                    "examples": List[str],
                    "priority": int,
                    "category": str
                }
        """
        validated_practices: list[dict[str, Any]] = []
        for practice_def in practice_definitions:
            if self._validate_practice_definition(practice_def):
                validated_practices.append(self._convert_to_dict(practice_def))
        return validated_practices

    def _validate_practice_definition(self, practice_def: BestPractice) -> bool:
        """Validate a practice definition.

        Args:
            practice_def (BestPractice): Practice definition to validate

        Returns:
            bool: True if practice definition is valid
        """
        if not practice_def.practice_id:
            return False
        if not practice_def.name:
            return False
        if not practice_def.description:
            return False
        if not practice_def.steps:
            return False
        if not isinstance(practice_def.steps, list):
            return False
        if not practice_def.rationale:
            return False
        if not isinstance(practice_def.examples, list):
            return False
        if not isinstance(practice_def.priority, int):
            return False
        return bool(practice_def.category)

    def _categorize_practices(self, practices: list[dict[str, Any]]) -> None:
        """Categorize practices by type.

        Args:
            practices (List[Dict[str, Any]]): List of practices to categorize
        """
        self._practice_categories = {}
        for practice in practices:
            category = practice.get("category", "general")
            if category not in self._practice_categories:
                self._practice_categories[category] = []
            self._practice_categories[category].append(practice["practice_id"])

    def _optimize_practices(
        self, practices: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Optimize practices for efficiency and effectiveness.

        This method analyzes and optimizes the practices to ensure they are
        efficient and effective in achieving their goals.

        Args:
            practices (List[Dict[str, Any]]): List of practices to optimize

        Returns:
            List[Dict[str, Any]]: List of optimized practices with the same structure
                as the input practices
        """
        optimized_practices: list[dict[str, Any]] = []
        for practice in practices:
            practice["steps"] = self._optimize_steps(practice["steps"])
            optimized_practices.append(practice)
        return optimized_practices

    def _optimize_steps(self, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Optimize individual steps within a practice.

        This method analyzes and optimizes the steps to ensure they are
        clear, concise, and effective.

        Args:
            steps (List[Dict[str, Any]]): List of steps to optimize

        Returns:
            List[Dict[str, Any]]: List of optimized steps with the same structure
                as the input steps
        """
        optimized_steps: list[dict[str, Any]] = []
        for i, step in enumerate(steps):
            # Ensure step_id exists
            if "step_id" not in step:
                step["step_id"] = f"step_{len(optimized_steps) + 1}"

            # Handle description validation and optimization
            description = step.get("description")

            # Check if description is missing, None, or not a string
            if not description or not isinstance(description, str):
                log_context.warning(
                    f"Step {i + 1} has invalid or missing description: {description}"
                )
                # Set a default description if missing
                if not description:
                    description = f"Step {i + 1}"
                elif not isinstance(description, str):
                    description = str(description)

            # Update step with validated description
            step["description"] = description

            # Ensure required_fields exists
            if "required_fields" not in step:
                step["required_fields"] = []

            optimized_steps.append(step)

        return optimized_steps

    def _convert_to_dict(self, practice_def: BestPractice) -> dict[str, Any]:
        """Convert a BestPractice object to a dictionary.

        Args:
            practice_def (BestPractice): Practice definition to convert

        Returns:
            Dict[str, Any]: Dictionary representation of the practice
        """
        return {
            "practice_id": practice_def.practice_id,
            "name": practice_def.name,
            "description": practice_def.description,
            "steps": practice_def.steps,
            "rationale": practice_def.rationale,
            "examples": practice_def.examples,
            "priority": practice_def.priority,
            "category": practice_def.category,
        }
