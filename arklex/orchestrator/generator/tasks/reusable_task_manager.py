"""Reusable task management component for the Arklex framework.

This module provides the ReusableTaskManager class that handles creation and
management of reusable task components. It promotes code reuse and maintainability
through template-based task generation.

Key Features:
- Reusable task template generation
- Template instantiation and customization
- Parameter validation and management
- Template categorization and organization
"""

from dataclasses import dataclass
from typing import Any

from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


@dataclass
class ReusableTask:
    """Represents a reusable task template for the Arklex framework.

    Attributes:
        template_id (str): Unique identifier for the template.
        name (str): Name of the template.
        description (str): Detailed description of the template.
        steps (List[Dict[str, Any]]): List of template steps.
        parameters (Dict[str, Any]): Template parameters and their types.
        examples (List[Dict[str, Any]]): Example instantiations.
        version (str): Template version.
        category (str): Template category (e.g., 'utility', 'workflow').
    """

    template_id: str
    name: str
    description: str
    steps: list[dict[str, Any]]
    parameters: dict[str, Any]
    examples: list[dict[str, Any]]
    version: str
    category: str


class ReusableTaskManager:
    """Manages the creation and management of reusable task components.

    This class handles the identification and extraction of reusable task
    patterns, creating templates that can be reused across different task
    graphs. It ensures high-quality and maintainable task components.

    The manager follows a modular design pattern where each major responsibility
    is handled by a dedicated method, promoting maintainability and testability.

    Attributes:
        model: The language model used for template generation
        role (str): The role or context for template generation
        user_objective (str): User's objective for the task graph
        _templates (Dict[str, ReusableTask]): Cache of generated templates
        _template_categories (Dict[str, List[str]]): Template categorization

    Methods:
        generate_reusable_tasks(): Main method to generate templates
        instantiate_template(): Create a task instance from a template
        _identify_patterns(): Identify reusable task patterns
        _extract_components(): Extract common components
        _create_templates(): Create reusable templates
        _validate_templates(): Validate generated templates
        _categorize_templates(): Categorize templates by type
    """

    def __init__(
        self,
        model: object,
        role: str,
        user_objective: str,
    ) -> None:
        """Initialize the ReusableTaskManager with required components.

        Args:
            model (Any): The language model to use for template generation.
            role (str): The role or context for template generation.
            user_objective (str): User's objective for the task graph.
        """
        self.model = model
        self.role = role
        self.user_objective = user_objective
        self._templates: dict[str, ReusableTask] = {}
        self._template_categories: dict[str, list[str]] = {}

    def generate_reusable_tasks(
        self, tasks: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Generate reusable task templates from the given tasks.

        This method orchestrates the template generation process by:
        1. Identifying reusable patterns
        2. Extracting common components
        3. Creating templates
        4. Validating templates
        5. Categorizing templates

        Args:
            tasks (List[Dict[str, Any]]): List of tasks to analyze.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of generated templates.
        """
        patterns = self._identify_patterns(tasks)
        if not patterns and tasks:
            patterns = [
                {
                    "name": tasks[0]["name"],
                    "description": tasks[0]["description"],
                    "steps": tasks[0].get("steps", []),
                    "parameters": {},
                }
            ]
        log_context.info(f"Identified {len(patterns)} reusable patterns")
        components = self._extract_components(patterns)
        log_context.info(f"Extracted {len(components)} common components")
        templates = self._create_templates(components)
        log_context.info(f"Created {len(templates)} templates")
        validated_templates = self._validate_templates(templates)
        log_context.info(f"Validated {len(validated_templates)} templates")
        self._categorize_templates(validated_templates)
        log_context.info("Categorized templates")
        return validated_templates

    def instantiate_template(
        self, template_id: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a task instance from a template.

        This method takes a template and parameters to create a concrete
        task instance that can be used in a task graph.

        Args:
            template_id (str): ID of the template to instantiate.
            parameters (Dict[str, Any]): Parameters for instantiation.

        Returns:
            Dict[str, Any]: Instantiated task.

        Raises:
            ValueError: If template_id is not found or parameters are invalid.
        """
        try:
            template = self._templates.get(template_id)
            if not template:
                raise ValueError(f"Template not found: {template_id}")
            if not self._validate_parameters(template, parameters):
                raise ValueError(f"Invalid parameters for template: {template_id}")
            instance = self._create_instance(template, parameters)
            log_context.info(f"Created instance from template: {template_id}")
            return instance
        except Exception as e:
            log_context.error(f"Error instantiating template: {str(e)}")
            raise

    def _identify_patterns(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify reusable task patterns.

        This method analyzes tasks to identify common patterns that could
        be extracted into reusable templates.

        Args:
            tasks (List[Dict[str, Any]]): List of tasks to analyze.

        Returns:
            List[Dict[str, Any]]: List of identified patterns.
        """
        patterns: list[dict[str, Any]] = []
        for task in tasks:
            # Include tasks with any number of steps, not just those with more than 1
            pattern = {
                "name": task["name"],
                "description": task["description"],
                "steps": task.get("steps", []),
                "parameters": {},
            }
            patterns.append(pattern)
        return patterns

    def _extract_components(
        self, patterns: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract common components from patterns.

        This method processes identified patterns to extract reusable
        components that can be used in templates.

        Args:
            patterns (List[Dict[str, Any]]): List of identified patterns.

        Returns:
            List[Dict[str, Any]]: List of extracted components.
        """
        components: list[dict[str, Any]] = []
        for pattern in patterns:
            parameters: dict[str, Any] = {}
            for step in pattern["steps"]:
                for field in step.get("required_fields", []):
                    if field not in parameters:
                        parameters[field] = "string"
            component = {
                "name": pattern["name"],
                "description": pattern["description"],
                "steps": pattern["steps"],
                "parameters": parameters,
            }
            components.append(component)
        return components

    def _create_templates(
        self, components: list[dict[str, Any]]
    ) -> dict[str, ReusableTask]:
        """Create reusable templates from components.

        This method processes components to create reusable task templates
        that can be instantiated with different parameters.

        Args:
            components (List[Dict[str, Any]]): List of components to process.

        Returns:
            Dict[str, ReusableTask]: Dictionary mapping template IDs to ReusableTask objects.
        """
        templates: dict[str, ReusableTask] = {}
        for i, component in enumerate(components):
            template = ReusableTask(
                template_id=f"template_{i + 1}",
                name=component["name"],
                description=component["description"],
                steps=component["steps"],
                parameters=component["parameters"],
                examples=[],
                version="1.0",
                category="general",
            )
            templates[template.template_id] = template
        return templates

    def _validate_templates(
        self, templates: dict[str, ReusableTask]
    ) -> dict[str, dict[str, Any]]:
        """Validate generated templates.

        This method ensures that all templates meet the required format
        and quality standards.

        Args:
            templates (Dict[str, ReusableTask]): Dictionary of templates to validate

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping template IDs to validated templates
                with the following structure:
                {
                    "template_id": str,
                    "name": str,
                    "description": str,
                    "steps": List[Dict[str, Any]],
                    "parameters": Dict[str, Any],
                    "examples": List[Dict[str, Any]],
                    "version": str,
                    "category": str
                }
        """
        validated_templates: dict[str, dict[str, Any]] = {}
        for template_id, template in templates.items():
            if self._validate_template(template):
                validated_templates[template_id] = self._convert_to_dict(template)
        return validated_templates

    def _validate_template(self, template: ReusableTask) -> bool:
        """Validate a template.

        Args:
            template (ReusableTask): Template to validate

        Returns:
            bool: True if template is valid
        """
        if not template.template_id:
            log_context.debug("Template validation failed: missing template_id")
            return False
        if not template.name:
            log_context.debug(
                f"Template validation failed: missing name for {template.template_id}"
            )
            return False
        if not template.description:
            log_context.debug(
                f"Template validation failed: missing description for {template.template_id}"
            )
            return False
        if not template.steps:
            log_context.debug(
                f"Template validation failed: missing steps for {template.template_id}"
            )
            return False
        if not isinstance(template.steps, list):
            log_context.debug(
                f"Template validation failed: steps not a list for {template.template_id}"
            )
            return False
        if not isinstance(template.parameters, dict):
            log_context.debug(
                f"Template validation failed: parameters not a dict for {template.template_id}"
            )
            return False
        if not isinstance(template.examples, list):
            log_context.debug(
                f"Template validation failed: examples not a list for {template.template_id}"
            )
            return False
        if not template.version:
            log_context.debug(
                f"Template validation failed: missing version for {template.template_id}"
            )
            return False
        if not template.category:
            log_context.debug(
                f"Template validation failed: missing category for {template.template_id}"
            )
            return False
        log_context.debug(f"Template validation passed for {template.template_id}")
        return True

    def _validate_parameters(
        self, template: ReusableTask, parameters: dict[str, Any]
    ) -> bool:
        """Validate template parameters.

        Args:
            template (ReusableTask): Template to validate parameters for
            parameters (Dict[str, Any]): Parameters to validate

        Returns:
            bool: True if parameters are valid
        """
        for param_name, param_type in template.parameters.items():
            if param_name not in parameters:
                return False
            if param_type == "string" and not isinstance(parameters[param_name], str):
                return False
            if param_type == "number" and not isinstance(
                parameters[param_name], int | float
            ):
                return False
            if param_type == "boolean" and not isinstance(parameters[param_name], bool):
                return False
        return True

    def _create_instance(
        self, template: ReusableTask, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a task instance from a template.

        This method takes a template and parameters to create a concrete
        task instance that can be used in a task graph.

        Args:
            template (ReusableTask): Template to instantiate
            parameters (Dict[str, Any]): Parameters for instantiation

        Returns:
            Dict[str, Any]: Instantiated task with the following structure:
                {
                    "task_id": str,
                    "name": str,
                    "description": str,
                    "steps": List[Dict[str, Any]],
                    "parameters": Dict[str, Any],
                    "template_id": str,
                    "version": str
                }
        """
        instance: dict[str, Any] = {
            "task_id": f"task_{template.template_id}",
            "name": template.name,
            "description": template.description,
            "steps": template.steps,
            "parameters": parameters,
            "template_id": template.template_id,
            "version": template.version,
        }
        return instance

    def _categorize_templates(self, templates: dict[str, dict[str, Any]]) -> None:
        """Categorize templates by type.

        Args:
            templates (Dict[str, Dict[str, Any]]): Dictionary of templates to categorize
        """
        self._template_categories = {}
        for template_id, template in templates.items():
            category = template.get("category", "general")
            if category not in self._template_categories:
                self._template_categories[category] = []
            self._template_categories[category].append(template_id)

    def _convert_to_dict(self, template: ReusableTask) -> dict[str, Any]:
        """Convert a ReusableTask object to a dictionary.

        Args:
            template (ReusableTask): Template to convert

        Returns:
            Dict[str, Any]: Dictionary representation of the template
        """
        return {
            "template_id": template.template_id,
            "name": template.name,
            "description": template.description,
            "steps": template.steps,
            "parameters": template.parameters,
            "examples": template.examples,
            "version": template.version,
            "category": template.category,
        }
