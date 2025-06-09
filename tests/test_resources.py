"""Test resources for the Arklex framework.

This module provides test resources for testing the Arklex framework.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Type

import pytest

from tests.utils.utils_workers import MCWorkerOrchestrator, MsgWorkerOrchestrator
from tests.utils.utils_tools import ShopifyToolOrchestrator
from tests.utils.test_config import CaseConfig


# Register custom markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "no_collect: mark a class to be excluded from pytest collection"
    )


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


# Test case configuration for different orchestrator types
TEST_CASES: List[CaseConfig] = [
    CaseConfig(
        MCWorkerOrchestrator,
        "mc_worker_taskgraph.json",
        "mc_worker_testcases.json",
    ),
    CaseConfig(
        MsgWorkerOrchestrator,
        "message_worker_taskgraph.json",
        "message_worker_testcases.json",
    ),
    CaseConfig(
        ShopifyToolOrchestrator,
        "shopify_tool_taskgraph.json",
        "shopify_tool_testcases.json",
    ),
]


def load_test_cases(test_cases_path: Path) -> List[Dict[str, Any]]:
    """Load test cases from a JSON file.

    Args:
        test_cases_path (Path): Path to the test cases file.

    Returns:
        List[Dict[str, Any]]: List of test cases.

    Raises:
        pytest.fail: If the file cannot be loaded or parsed.
    """
    try:
        with open(test_cases_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        pytest.fail(f"Test cases file not found: {test_cases_path}", pytrace=True)
    except json.JSONDecodeError as e:
        pytest.fail(
            f"Invalid JSON in test cases file {test_cases_path}: {str(e)}", pytrace=True
        )


@pytest.mark.parametrize(
    "test_case",
    TEST_CASES,
    ids=lambda tc: tc.name,
)
def test_resources(test_case: CaseConfig) -> None:
    """Run test cases for a specific orchestrator class.

    This function loads test cases from a file and runs them using the specified
    orchestrator class. It handles test failures and provides detailed error
    messages.

    Args:
        test_case (CaseConfig): The test case configuration.

    Raises:
        pytest.fail: If any test case fails, with a detailed error message
            including the test case number and orchestrator class name.
    """
    data_dir = Path(__file__).parent / "data"
    config_path = data_dir / test_case.config_file
    test_cases_path = data_dir / test_case.test_cases_file

    test_resources_instance = test_case.orchestrator_cls(str(config_path))
    test_cases = load_test_cases(test_cases_path)

    for i, test_case_data in enumerate(test_cases):
        try:
            test_resources_instance.run_single_test(test_case_data)
        except Exception as e:
            pytest.fail(
                f"Test case {i} failed for {test_case.orchestrator_cls.__name__} "
                f"from {test_case.test_cases_file}: {str(e)}",
                pytrace=True,
            )
