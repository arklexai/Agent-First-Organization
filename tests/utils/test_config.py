"""Test configuration utilities for the Arklex framework."""

from typing import Any, Type

# Type aliases for better readability
OrchestratorType = Type[Any]


class CaseConfig:
    """Configuration for a test case suite.

    This class holds the configuration for a set of test cases, including
    the orchestrator class to test and the paths to its configuration files.

    Attributes:
        orchestrator_cls: The orchestrator class to test
        config_file: Path to the taskgraph configuration file
        test_cases_file: Path to the test cases file
    """

    def __init__(
        self,
        orchestrator_cls: OrchestratorType,
        config_file: str,
        test_cases_file: str,
    ):
        self.orchestrator_cls = orchestrator_cls
        self.config_file = config_file
        self.test_cases_file = test_cases_file

    @property
    def name(self) -> str:
        """Get the name of the test case configuration."""
        return f"{self.orchestrator_cls.__name__}-{self.config_file}-{self.test_cases_file}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CaseConfig):
            return NotImplemented
        return (
            self.orchestrator_cls == other.orchestrator_cls
            and self.config_file == other.config_file
            and self.test_cases_file == other.test_cases_file
        )

    def __hash__(self) -> int:
        return hash((self.orchestrator_cls, self.config_file, self.test_cases_file))
