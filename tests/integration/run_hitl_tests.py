#!/usr/bin/env python3
"""
Test runner for HITL (Human-in-the-Loop) integration tests.

This script provides a convenient way to run the HITL integration tests
with proper configuration and environment setup. It handles dependency
checking, file validation, and test execution with various options.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

# Add the project root to the Python path to ensure imports work correctly
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def setup_environment() -> None:
    """
    Set up the test environment with required environment variables.

    This function ensures that all necessary environment variables are set
    for the integration tests to run properly. The actual environment setup
    is handled in conftest.py, so this function primarily serves as a
    placeholder for any additional setup that might be needed.
    """
    # Import conftest to use its environment setup
    # The environment variables are already set up in conftest.py
    # We just need to ensure we're in the right directory
    pass


def check_test_dependencies() -> bool:
    """
    Check if all required test dependencies are available.

    Returns:
        bool: True if all dependencies are available, False otherwise.

    This function verifies that all necessary Python packages are installed
    before running the integration tests. Missing dependencies will cause
    test failures, so this check helps provide clear error messages.
    """
    # List of required packages for integration tests
    required_modules = [
        "langchain",
        "langchain_community",
        "langchain_openai",
        "openai",
        "pytest",
    ]

    missing_modules = []
    for module_name in required_modules:
        if importlib.util.find_spec(module_name) is None:
            missing_modules.append(module_name)

    if missing_modules:
        print(f"❌ Missing required dependencies: {', '.join(missing_modules)}")
        print(
            "Please install test dependencies with: pip install pytest openai langchain langchain-community langchain-openai"
        )
        return False

    return True


def check_test_files() -> bool:
    """
    Check if required test files exist.

    Returns:
        bool: True if all required files exist, False otherwise.

    This function verifies that all necessary test files and configuration
    files are present before running the integration tests. Missing files
    will cause test failures, so this check helps provide clear error messages.
    """
    # List of required files for integration tests
    required_files = [
        "tests/integration/test_hitl_server.py",
        "examples/hitl_server/taskgraph.json",
        "tests/integration/conftest.py",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required test files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False

    return True


def run_tests(
    test_file: str | None = None, verbose: bool = False, markers: str | None = None
) -> bool:
    """
    Run the HITL integration tests.

    Args:
        test_file: Specific test file to run (optional)
        verbose: Whether to run tests in verbose mode
        markers: Pytest markers to filter tests

    Returns:
        bool: True if tests pass, False otherwise.

    This function sets up the environment, checks dependencies and files,
    then executes the pytest command with appropriate options.
    """
    setup_environment()

    # Verify prerequisites before running tests
    if not check_test_dependencies():
        return False

    if not check_test_files():
        return False

    # Determine which tests to run based on input parameters
    if test_file:
        test_path = f"tests/integration/{test_file}"
    else:
        test_path = "tests/integration/test_hitl_server.py"

    # Build pytest command with appropriate options
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_path,
        "-v" if verbose else "",  # Verbose output if requested
        f"-m {markers}" if markers else "",  # Filter by markers if specified
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker validation
        "--disable-warnings",  # Suppress warnings for cleaner output
        "--no-header",  # Remove pytest header
    ]

    # Remove empty strings from command list
    cmd = [arg for arg in cmd if arg]

    print(f"Running tests with command: {' '.join(cmd)}")
    print(f"Working directory: {Path.cwd()}")
    print("-" * 80)

    try:
        # Execute pytest with the constructed command
        subprocess.run(cmd, check=True, capture_output=False)
        print("-" * 80)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 80)
        print(f"❌ Tests failed with exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Unexpected error running tests: {e}")
        return False


def main() -> None:
    """
    Main entry point for the test runner.

    This function parses command line arguments and orchestrates the test
    execution process. It supports various options for running specific
    tests, checking dependencies, and listing available tests.
    """
    import argparse

    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Run HITL integration tests")
    parser.add_argument(
        "--test-file",
        type=str,
        help="Specific test file to run (e.g., test_hitl_server.py)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Run tests in verbose mode"
    )
    parser.add_argument(
        "-m",
        "--markers",
        type=str,
        help="Run only tests matching the given markers (e.g., 'hitl and not slow')",
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List available tests without running them",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check dependencies and files without running tests",
    )

    args = parser.parse_args()

    # Handle check-only mode - verify environment without running tests
    if args.check_only:
        print("🔍 Checking test environment...")
        setup_environment()

        if check_test_dependencies():
            print("✅ All dependencies are available")
        else:
            print("❌ Missing dependencies")
            sys.exit(1)

        if check_test_files():
            print("✅ All required test files exist")
        else:
            print("❌ Missing test files")
            sys.exit(1)

        print("✅ Test environment is ready")
        return

    # Handle list-tests mode - show available tests without execution
    if args.list_tests:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration/test_hitl_server.py",
            "--collect-only",  # Only collect tests, don't run them
            "-q",  # Quiet mode for cleaner output
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to list tests: {e}")
            sys.exit(1)
        return

    # Run the tests with the specified options
    success = run_tests(
        test_file=args.test_file,
        verbose=args.verbose,
        markers=args.markers,
    )

    # Exit with appropriate code based on test results
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
