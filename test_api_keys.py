"""
Standalone API key validation script for testing create.py/run.py functionality.

This script validates API keys and tests the customer service example with real API calls.
It should be run manually by developers to verify their API keys work correctly.

This script provides a modular test framework that validates:
1. Environment configuration and API key loading
2. API key validation for OpenAI, Anthropic, and Google
3. Provider utility functions with real API keys
4. End-to-end testing of customer service examples
5. Network connectivity and package requirements

The script is designed to be run independently and provides detailed feedback
on each test step, making it easy to identify and resolve issues.

Usage:
    python test_api_keys.py                           # Test all providers
    python test_api_keys.py --providers openai google # Test specific providers
    python test_api_keys.py --providers openai anthropic

Requirements:
    - .env file with API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY)
      NOTE: A .env file is REQUIRED to run this script successfully
    - Internet connection for API calls
    - Required packages: requests, python-dotenv
"""

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import requests

# Add the project root to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arklex.utils.provider_utils import get_api_key_for_provider, get_provider_config


class APITestConfiguration:
    """
    Holds configuration constants for API testing across different providers.
    This class centralizes all static configuration, such as supported API keys, provider mappings,
    default models, API endpoints, and timeouts, to ensure consistency across the test suite.
    """

    # Supported API key environment variables
    SUPPORTED_API_KEYS = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]

    # Mapping from environment variable names to provider identifiers
    ENV_TO_PROVIDER_MAP = {
        "OPENAI_API_KEY": "openai",
        "ANTHROPIC_API_KEY": "anthropic",
        "GOOGLE_API_KEY": "google",
    }

    # Default models for each provider
    DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "google": "gemini-1.5-flash",
    }

    # All supported providers
    ALL_PROVIDERS = ["openai", "anthropic", "google"]

    # API endpoints for connectivity testing
    API_ENDPOINTS = [
        ("https://api.openai.com", "OpenAI API"),
        ("https://api.anthropic.com", "Anthropic API"),
        ("https://generativelanguage.googleapis.com", "Google API"),
    ]

    # Test configuration
    API_TIMEOUT_SECONDS = 10
    PROCESS_TIMEOUT_SECONDS = 600
    CONNECTIVITY_TIMEOUT_SECONDS = 5
    PROGRESS_UPDATE_INTERVAL_SECONDS = 5

    @classmethod
    def get_env_key_for_provider(cls, provider: str) -> str:
        """
        Get the correct environment variable name for a given provider.

        Args:
            provider: The provider name (e.g., 'openai', 'anthropic', 'google')

        Returns:
            The environment variable name for the provider
        """
        # Create reverse mapping
        provider_to_env_map = {v: k for k, v in cls.ENV_TO_PROVIDER_MAP.items()}
        return provider_to_env_map.get(provider, f"{provider.upper()}_API_KEY")


class EnvironmentManager:
    """
    Handles loading and parsing of environment variables from a .env file.
    This class is responsible for extracting API keys and other environment variables needed for testing,
    and provides user-friendly feedback if the .env file or required keys are missing.
    """

    def __init__(self, env_file_path: str = ".env") -> None:
        """
        Initialize the EnvironmentManager.

        Args:
            env_file_path: Path to the .env file to load. Defaults to '.env'.
        """
        self.env_file_path = Path(env_file_path)
        self.env_variables: dict[str, str] = {}

    def load_environment_variables(self) -> dict[str, str]:
        """
        Load environment variables from the .env file, parse them, and print a summary.
        If the file is missing, prints instructions for creating it.

        Returns:
            Dictionary containing environment variables from the .env file.
        """
        if not self.env_file_path.exists():
            self._display_env_file_instructions()
            return {}

        print(f"📁 Found .env file: {self.env_file_path.absolute()}")
        self.env_variables = self._parse_env_file()
        print(f"✅ Loaded {len(self.env_variables)} environment variables")
        self._display_api_key_summary()
        return self.env_variables

    def _display_env_file_instructions(self) -> None:
        """
        Print instructions for creating a .env file with the required API keys.
        """
        print(f"⚠️  No .env file found at {self.env_file_path.absolute()}")
        print("💡 Please create a .env file with your API keys:")
        for key in APITestConfiguration.SUPPORTED_API_KEYS:
            print(f"   {key}=your_{key.lower()}_here")

    def _parse_env_file(self) -> dict[str, str]:
        """
        Parse the .env file and extract key-value pairs, ignoring comments and blank lines.

        Returns:
            Dictionary of environment variables from the .env file.
        """
        env_vars = {}
        with open(self.env_file_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key] = value
        return env_vars

    def _display_api_key_summary(self) -> None:
        """
        Print a summary of found API keys, masking their values for security.
        If no keys are found, prints instructions for adding them.
        """
        found_keys = [
            key
            for key in APITestConfiguration.SUPPORTED_API_KEYS
            if self.env_variables.get(key)
        ]

        if not found_keys:
            self._display_missing_api_keys_instructions()
            return

        print(f"🔑 Found API keys: {', '.join(found_keys)}")
        print("📊 API Key Details:")
        for key in found_keys:
            api_key = self.env_variables[key]
            masked_key = (
                f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else api_key
            )
            print(f"   {key}: {masked_key}")

    def _display_missing_api_keys_instructions(self) -> None:
        """
        Print instructions for adding missing API keys to the .env file.
        """
        print("⚠️  No API keys found in .env file")
        print("💡 Please add your API keys to the .env file:")
        for key in APITestConfiguration.SUPPORTED_API_KEYS:
            print(f"   {key}=your_{key.lower()}_here")


class APIKeyValidator:
    """
    Validates API keys for supported providers by making real API calls.
    This class is responsible for confirming that the provided API keys are valid and functional
    by sending minimal requests to each provider's API and interpreting the responses.
    """

    def __init__(self) -> None:
        """
        Initialize the APIKeyValidator with the default timeout for API requests.
        """
        self.timeout = APITestConfiguration.API_TIMEOUT_SECONDS

    def validate_api_key(self, provider: str, api_key: str) -> bool:
        """
        Validate an API key by making a test API call to the specified provider.
        Prints detailed feedback about the validation process and result.

        Args:
            provider: The provider name (e.g., 'openai', 'anthropic', 'google').
            api_key: The API key to validate.

        Returns:
            True if the API key is valid and accepted by the provider, False otherwise.
        """
        print(f"🔍 Validating {provider} API key...")
        print(f"🔑 API key starts with: {api_key[:10]}...")

        try:
            if provider == "openai":
                return self._validate_openai_api_key(api_key)
            elif provider == "anthropic":
                return self._validate_anthropic_api_key(api_key)
            elif provider == "google":
                return self._validate_google_api_key(api_key)
            else:
                print(f"❌ Unknown provider: {provider}")
                return False

        except requests.exceptions.Timeout:
            print(f"⏰ Timeout validating {provider} API key ({self.timeout}s timeout)")
            return False
        except requests.exceptions.ConnectionError:
            print(f"🌐 Connection error validating {provider} API key")
            print("💡 Check your internet connection and try again")
            return False
        except Exception as e:
            print(f"❌ Error validating {provider} API key: {e}")
            return False

    def _validate_openai_api_key(self, api_key: str) -> bool:
        """
        Validate an OpenAI API key by making a test completion request.
        Prints request and response details for debugging.

        Args:
            api_key: The OpenAI API key to validate.

        Returns:
            True if the API key is valid, False otherwise.
        """
        print("🌐 Making OpenAI API call...")
        print("📡 Request URL: https://api.openai.com/v1/chat/completions")
        print("📡 Request model: gpt-3.5-turbo")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5,
        }

        print("📤 Sending request...")
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=self.timeout,
        )

        return self._process_api_response(response, "OpenAI")

    def _validate_anthropic_api_key(self, api_key: str) -> bool:
        """
        Validate an Anthropic API key by making a test message request.
        Prints request and response details for debugging.

        Args:
            api_key: The Anthropic API key to validate.

        Returns:
            True if the API key is valid, False otherwise.
        """
        print("🌐 Making Anthropic API call...")
        print("📡 Request URL: https://api.anthropic.com/v1/messages")
        print("📡 Request model: claude-3-haiku-20240307")

        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        data = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "Hello"}],
        }

        print("📤 Sending request...")
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=self.timeout,
        )

        return self._process_api_response(response, "Anthropic")

    def _validate_google_api_key(self, api_key: str) -> bool:
        """
        Validate a Google API key by making a test content generation request.
        Prints request and response details for debugging.

        Args:
            api_key: The Google API key to validate.

        Returns:
            True if the API key is valid, False otherwise.
        """
        print("🌐 Making Google API call...")
        print(
            "📡 Request URL: https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        )
        print("📡 Request model: gemini-1.5-flash")

        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": "Hello"}]}],
            "generationConfig": {"maxOutputTokens": 5},
        }

        print("📤 Sending request...")
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
            headers=headers,
            json=data,
            timeout=self.timeout,
        )

        return self._process_api_response(response, "Google")

    def _process_api_response(
        self, response: requests.Response, provider_name: str
    ) -> bool:
        """
        Process the API response and determine if the key is valid.
        Prints status and a preview of the response content.

        Args:
            response: The HTTP response from the API.
            provider_name: Name of the provider for logging and parsing.

        Returns:
            True if the response indicates a valid API key, False otherwise.
        """
        print(f"📡 {provider_name} response status: {response.status_code}")

        if response.status_code != 200:
            print(
                f"❌ {provider_name} API key validation failed: {response.status_code}"
            )
            print(f"📤 Error response: {response.text[:200]}...")
            return False

        print(f"✅ {provider_name} API key is valid")
        self._display_response_preview(response, provider_name)
        return True

    def _display_response_preview(
        self, response: requests.Response, provider_name: str
    ) -> None:
        """
        Print a preview of the API response content for the given provider.
        """
        try:
            response_data = response.json()

            if (
                provider_name == "OpenAI"
                and "choices" in response_data
                and len(response_data["choices"]) > 0
            ):
                content = (
                    response_data["choices"][0].get("message", {}).get("content", "")
                )
                print(f"📤 Response preview: {content[:50]}...")
            elif (
                provider_name == "Anthropic"
                and "content" in response_data
                and len(response_data["content"]) > 0
            ):
                content = response_data["content"][0].get("text", "")
                print(f"📤 Response preview: {content[:50]}...")
            elif (
                provider_name == "Google"
                and "candidates" in response_data
                and len(response_data["candidates"]) > 0
            ):
                content = (
                    response_data["candidates"][0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
                print(f"📤 Response preview: {content[:50]}...")
        except Exception:
            pass


class ProviderUtilityTester:
    """
    Tests provider utility functions (such as key retrieval and config generation) with real API keys.
    Ensures that the utility functions used throughout the codebase work as expected with actual credentials.
    """

    def __init__(self, validator: APIKeyValidator) -> None:
        """
        Initialize the ProviderUtilityTester with a reference to an APIKeyValidator.

        Args:
            validator: APIKeyValidator instance used for key validation.
        """
        self.validator = validator

    def test_provider_utilities_with_real_keys(
        self, env_vars: dict[str, str], providers: list[str] | None = None
    ) -> bool:
        """
        Test that provider utility functions work correctly with real API keys.
        Runs tests for specified providers and prints results.

        Args:
            env_vars: Dictionary of environment variables containing API keys.
            providers: List of providers to test. If None, tests all providers.

        Returns:
            True if all provider utilities work correctly, False otherwise.
        """
        if providers is None:
            providers = APITestConfiguration.ALL_PROVIDERS

        print("\n🔍 Testing provider utilities with real API keys...")
        print(f"📋 Testing providers: {', '.join(providers)}")

        results = []

        for provider in providers:
            env_key = APITestConfiguration.get_env_key_for_provider(provider)
            api_key = env_vars.get(env_key, "")

            if not api_key:
                print(f"⚠️  No {provider} API key found in .env file")
                results.append(False)
                continue

            result = self._test_single_provider_utility(provider, api_key)
            results.append(result)

        return all(results)

    def _test_single_provider_utility(self, provider: str, api_key: str) -> bool:
        """
        Test utility functions for a single provider, including key validation and config retrieval.
        Prints detailed feedback for each step.

        Args:
            provider: The provider name to test.
            api_key: The API key for the provider.

        Returns:
            True if all utility functions work correctly, False otherwise.
        """
        print(f"\n🧪 Testing {provider}...")
        print(f"🔑 Found {provider} API key: {api_key[:10]}...")

        if not self.validator.validate_api_key(provider, api_key):
            print(f"❌ {provider} API key is invalid")
            return False

        print(f"✅ {provider} API key is valid")

        if not self._test_get_api_key_for_provider_function(provider, api_key):
            return False

        return self._test_get_provider_config_function(provider, api_key)

    def _test_get_api_key_for_provider_function(
        self, provider: str, api_key: str
    ) -> bool:
        """
        Test the get_api_key_for_provider function for the given provider.
        Checks that the returned key matches the expected value.

        Args:
            provider: The provider name to test.
            api_key: The expected API key value.

        Returns:
            True if the function works correctly, False otherwise.
        """
        try:
            retrieved_key = get_api_key_for_provider(provider)
            if retrieved_key != api_key:
                print(
                    f"❌ get_api_key_for_provider('{provider}') failed - keys don't match"
                )
                return False
            print(f"✅ get_api_key_for_provider('{provider}') works correctly")
            return True
        except Exception as e:
            print(f"❌ Error testing {provider}: {e}")
            return False

    def _test_get_provider_config_function(self, provider: str, api_key: str) -> bool:
        """
        Test the get_provider_config function for the given provider.
        Checks that the returned config contains the correct API key.

        Args:
            provider: The provider name to test.
            api_key: The expected API key value.

        Returns:
            True if the function works correctly, False otherwise.
        """
        try:
            config = get_provider_config(provider, f"{provider}-test-model")
            if config["api_key"] != api_key:
                print(f"❌ get_provider_config('{provider}') failed - keys don't match")
                return False
            print(f"✅ get_provider_config('{provider}') works correctly")
            return True
        except Exception as e:
            print(f"❌ Error testing {provider} config: {e}")
            return False


class ProcessExecutor:
    """
    Executes subprocess commands with real-time output monitoring and progress reporting.
    Used to run create.py and run.py scripts and capture their output for validation.
    """

    def __init__(
        self, timeout: int = APITestConfiguration.PROCESS_TIMEOUT_SECONDS
    ) -> None:
        """
        Initialize the ProcessExecutor with a default timeout for process execution.

        Args:
            timeout: Timeout in seconds for process execution. Defaults to APITestConfiguration.PROCESS_TIMEOUT_SECONDS.
        """
        self.timeout = timeout

    def execute_command_with_realtime_output(
        self,
        cmd: list[str],
        env: dict[str, str],
        input_text: str | None = None,
        timeout: int | None = None,
    ) -> tuple[int, str, str]:
        """
        Execute a command as a subprocess, capturing and printing output in real time.
        Optionally sends input to the process. Handles timeouts and prints progress.

        Args:
            cmd: Command to execute as a list of strings.
            env: Environment variables for the process.
            input_text: Optional input text to send to the process.
            timeout: Optional timeout override in seconds.

        Returns:
            Tuple of (return_code, stdout, stderr) from the process.
        """
        timeout = timeout or self.timeout
        print(f"🚀 Running: {' '.join(cmd)}")
        print(f"⏱️  Timeout: {timeout} seconds")
        print("=" * 80)

        if input_text:
            return self._execute_process_with_input(cmd, env, input_text, timeout)
        else:
            return self._execute_process_without_input(cmd, env, timeout)

    def _execute_process_with_input(
        self, cmd: list[str], env: dict[str, str], input_text: str, timeout: int
    ) -> tuple[int, str, str]:
        """
        Execute a process that expects input, sending the provided input text.
        Captures and prints output in real time.

        Args:
            cmd: Command to execute.
            env: Environment variables.
            input_text: Input text to send to the process.
            timeout: Timeout in seconds.

        Returns:
            Tuple of (return_code, stdout, stderr) from the process.
        """
        print("📝 Sending input to process...")
        print(f"📤 Input preview: {repr(input_text[:100])}...")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=0,
            universal_newlines=True,
        )

        try:
            start_time = time.time()
            print("🔄 Waiting for process output...")
            print("💡 Output will be displayed in real-time as it comes...")
            print("⏳ Process started, waiting for first output...")

            stdout, stderr = process.communicate(input=input_text, timeout=timeout)
            return self._process_final_output(process, stdout, stderr, start_time)

        except subprocess.TimeoutExpired:
            print(f"⏰ Command timed out after {timeout} seconds")
            process.terminate()
            raise

    def _execute_process_without_input(
        self, cmd: list[str], env: dict[str, str], timeout: int
    ) -> tuple[int, str, str]:
        """
        Execute a process that does not expect input, capturing and printing output in real time.

        Args:
            cmd: Command to execute.
            env: Environment variables.
            timeout: Timeout in seconds.

        Returns:
            Tuple of (return_code, stdout, stderr) from the process.
        """
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=None,
            text=True,
            env=env,
            bufsize=0,
            universal_newlines=True,
        )

        stdout_lines = []
        stderr_lines = []
        output_count = 0

        try:
            start_time = time.time()
            print("🔄 Waiting for process output...")
            print("💡 Output will be displayed in real-time as it comes...")
            print("⏳ Process started, waiting for first output...")

            stdout_lines, stderr_lines, output_count = self._monitor_process_output(
                process, stdout_lines, stderr_lines, start_time, timeout
            )

            remaining_stdout, remaining_stderr = process.communicate()
            stdout_lines, stderr_lines = self._add_remaining_output(
                remaining_stdout, remaining_stderr, stdout_lines, stderr_lines
            )

            return self._process_final_output(
                process, "".join(stdout_lines), "".join(stderr_lines), start_time
            )

        except subprocess.TimeoutExpired:
            print(f"⏰ Command timed out after {timeout} seconds")
            process.terminate()
            raise

    def _monitor_process_output(
        self,
        process: subprocess.Popen,
        stdout_lines: list[str],
        stderr_lines: list[str],
        start_time: float,
        timeout: int,
    ) -> tuple[list[str], list[str], int]:
        """
        Monitor process output in real time, printing progress and handling timeouts.

        Args:
            process: The subprocess to monitor.
            stdout_lines: List to collect stdout lines.
            stderr_lines: List to collect stderr lines.
            start_time: When the process started.
            timeout: Timeout in seconds.

        Returns:
            Tuple of (stdout_lines, stderr_lines, output_count).
        """
        last_output_time = time.time()
        output_count = 0
        last_progress_time = time.time()

        while process.poll() is None:
            if time.time() - start_time > timeout:
                print(f"⏰ Command timed out after {timeout} seconds")
                process.terminate()
                raise subprocess.TimeoutExpired(process.args, timeout)

            stdout_lines, stderr_lines, output_count, last_output_time = (
                self._read_available_output(
                    process, stdout_lines, stderr_lines, output_count, last_output_time
                )
            )

            last_progress_time = self._print_progress_if_needed(
                start_time, last_output_time, output_count, last_progress_time
            )
            time.sleep(0.01)

        return stdout_lines, stderr_lines, output_count

    def _read_available_output(
        self,
        process: subprocess.Popen,
        stdout_lines: list[str],
        stderr_lines: list[str],
        output_count: int,
        last_output_time: float,
    ) -> tuple[list[str], list[str], int, float]:
        """
        Read available output from process streams (stdout and stderr), updating output counters.

        Args:
            process: The subprocess to read from.
            stdout_lines: List to collect stdout lines.
            stderr_lines: List to collect stderr lines.
            output_count: Current output count.
            last_output_time: Last time output was received.

        Returns:
            Tuple of (stdout_lines, stderr_lines, output_count, last_output_time).
        """
        import select

        reads = []
        if process.stdout:
            reads.append(process.stdout)
        if process.stderr:
            reads.append(process.stderr)

        if not reads:
            return stdout_lines, stderr_lines, output_count, last_output_time

        try:
            ready, _, _ = select.select(reads, [], [], 0.1)
            for stream in ready:
                if stream == process.stdout:
                    line = stream.readline()
                    if line:
                        output_count += 1
                        print(f"📤 STDOUT [{output_count}]: {line.rstrip()}")
                        stdout_lines.append(line)
                        last_output_time = time.time()
                elif stream == process.stderr:
                    line = stream.readline()
                    if line:
                        output_count += 1
                        print(f"📤 STDERR [{output_count}]: {line.rstrip()}")
                        stderr_lines.append(line)
                        last_output_time = time.time()
        except (OSError, ValueError):
            pass

        return stdout_lines, stderr_lines, output_count, last_output_time

    def _print_progress_if_needed(
        self,
        start_time: float,
        last_output_time: float,
        output_count: int,
        last_progress_time: float,
    ) -> float:
        """
        Print a progress message if enough time has elapsed since the last output.

        Args:
            start_time: When the process started.
            last_output_time: Last time output was received.
            output_count: Number of outputs received.
            last_progress_time: Last time progress was printed.

        Returns:
            Updated last_progress_time.
        """
        current_time = time.time()
        if (
            current_time - last_output_time > 2
            and current_time - start_time > 3
            and current_time - last_progress_time
            > APITestConfiguration.PROGRESS_UPDATE_INTERVAL_SECONDS
        ):
            elapsed = int(current_time - start_time)
            print(f"⏳ Still running... (elapsed: {elapsed}s, outputs: {output_count})")
            return current_time
        return last_progress_time

    def _add_remaining_output(
        self,
        remaining_stdout: str,
        remaining_stderr: str,
        stdout_lines: list[str],
        stderr_lines: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Add any remaining output from the process to the output lists.

        Args:
            remaining_stdout: Remaining stdout content.
            remaining_stderr: Remaining stderr content.
            stdout_lines: List to add stdout lines to.
            stderr_lines: List to add stderr lines to.

        Returns:
            Tuple of (stdout_lines, stderr_lines).
        """
        if remaining_stdout:
            print(f"📤 FINAL STDOUT: {remaining_stdout}")
            stdout_lines.append(remaining_stdout)
        if remaining_stderr:
            print(f"📤 FINAL STDERR: {remaining_stderr}")
            stderr_lines.append(remaining_stderr)
        return stdout_lines, stderr_lines

    def _process_final_output(
        self, process: subprocess.Popen, stdout: str, stderr: str, start_time: float
    ) -> tuple[int, str, str]:
        """
        Process and print the final output from a completed subprocess, including elapsed time and status.

        Args:
            process: The completed subprocess.
            stdout: Standard output content.
            stderr: Standard error content.
            start_time: When the process started.

        Returns:
            Tuple of (return_code, stdout, stderr).
        """
        output_count = 0
        stdout_lines = []
        stderr_lines = []

        if stdout:
            lines = stdout.split("\n")
            for line in lines:
                if line.strip():
                    output_count += 1
                    print(f"📤 STDOUT [{output_count}]: {line}")
                    stdout_lines.append(line + "\n")

        if stderr:
            lines = stderr.split("\n")
            for line in lines:
                if line.strip():
                    output_count += 1
                    print(f"📤 STDERR [{output_count}]: {line}")
                    stderr_lines.append(line + "\n")

        elapsed_time = int(time.time() - start_time)
        print("=" * 80)
        print(
            f"🏁 Command finished with return code: {process.returncode} (elapsed: {elapsed_time}s, total outputs: {output_count})"
        )

        if process.returncode == 0:
            print("✅ Command completed successfully")
        else:
            print("❌ Command failed")

        return process.returncode, "".join(stdout_lines), "".join(stderr_lines)


class EnvironmentTester:
    """
    Tests environment loading and API key validation by checking .env file parsing and key validity.
    This class ensures that the environment is set up correctly before running further tests.
    """

    def __init__(self, validator: APIKeyValidator) -> None:
        """
        Initialize the EnvironmentTester with a reference to an APIKeyValidator.

        Args:
            validator: APIKeyValidator instance used for key validation.
        """
        self.validator = validator

    def test_environment_loading(self, providers: list[str] | None = None) -> bool:
        """
        Test that .env file loading and API key extraction work correctly.
        Prints a summary of valid and invalid keys.

        Args:
            providers: List of providers to test. If None, tests all providers.

        Returns:
            True if at least one valid API key is found, False otherwise.
        """
        if providers is None:
            providers = APITestConfiguration.ALL_PROVIDERS

        print(
            f"\n🔍 Testing .env file loading for providers: {', '.join(providers)}..."
        )

        try:
            valid_keys, invalid_keys = self._validate_loaded_keys(providers)

            if valid_keys:
                print(f"✅ Valid API keys: {', '.join(valid_keys)}")
                return True
            else:
                print("❌ No valid API keys found")
                return False

        except Exception as e:
            print(f"❌ Error testing .env file loading: {e}")
            return False

    def _validate_loaded_keys(
        self, providers: list[str]
    ) -> tuple[list[str], list[str]]:
        """
        Validate loaded API keys by checking each against its provider.

        Args:
            providers: List of providers to validate.

        Returns:
            Tuple of (valid_keys, invalid_keys), where each is a list of environment variable names.
        """
        from dotenv import load_dotenv

        load_dotenv()

        valid_keys = []
        invalid_keys = []

        # Get the environment variable names for the specified providers
        provider_to_env_map = {
            v: k for k, v in APITestConfiguration.ENV_TO_PROVIDER_MAP.items()
        }

        for provider in providers:
            env_key = provider_to_env_map.get(provider)
            if not env_key:
                continue

            api_key = os.getenv(env_key, "")
            if not api_key:
                continue

            if self.validator.validate_api_key(provider, api_key):
                valid_keys.append(env_key)
                print(f"✅ {env_key} is valid")
            else:
                invalid_keys.append(env_key)
                print(f"❌ {env_key} is invalid")

        return valid_keys, invalid_keys


class CustomerServiceExampleTester:
    """
    Tests the customer service example workflow with real API keys and all supported providers.
    This class runs both create.py and run.py scripts for each provider, validating end-to-end functionality.
    """

    def __init__(self, executor: ProcessExecutor, validator: APIKeyValidator) -> None:
        """
        Initialize the CustomerServiceExampleTester with references to a ProcessExecutor and APIKeyValidator.

        Args:
            executor: ProcessExecutor instance used to run scripts.
            validator: APIKeyValidator instance used for key validation.
        """
        self.executor = executor
        self.validator = validator

    def test_customer_service_example_with_all_providers(
        self, providers: list[str] | None = None
    ) -> bool:
        """
        Test the customer service example with specified providers by running create.py and run.py.
        Prints detailed feedback for each provider.

        Args:
            providers: List of providers to test. If None, tests all providers.

        Returns:
            True if at least one provider works end-to-end, False otherwise.
        """
        if providers is None:
            providers = APITestConfiguration.ALL_PROVIDERS

        print(
            f"\n🔍 Testing customer service example with providers: {', '.join(providers)}..."
        )
        print("🔧 Testing both create.py and run.py for each provider")

        env_manager = EnvironmentManager()
        env_vars = env_manager.load_environment_variables()
        results = []

        for provider in providers:
            env_key = APITestConfiguration.get_env_key_for_provider(provider)
            api_key = env_vars.get(env_key, "")

            if not api_key:
                print(f"⚠️  No {provider} API key found in .env file")
                results.append(False)
                continue

            result = self._test_customer_service_with_provider(
                provider, api_key, env_vars
            )
            results.append(result)

        return any(results)

    def _test_customer_service_with_provider(
        self, provider: str, api_key: str, env_vars: dict[str, str]
    ) -> bool:
        """
        Test the customer service example workflow with a specific provider.
        Runs both create.py and run.py, printing detailed feedback.

        Args:
            provider: The provider to test.
            api_key: The API key for the provider.
            env_vars: Environment variables.

        Returns:
            True if the provider works correctly, False otherwise.
        """
        print(f"\n🧪 Testing customer service example with {provider}...")
        print(f"🔧 Testing both create.py and run.py with {provider}")

        if not self.validator.validate_api_key(provider, api_key):
            print(
                f"❌ Skipping {provider} - API key is invalid or cannot generate responses"
            )
            return False

        env_key = APITestConfiguration.get_env_key_for_provider(provider)
        env = os.environ.copy()
        env[env_key] = api_key

        config_path = "./examples/customer_service/customer_service_config.json"
        if not os.path.exists(config_path):
            print(f"❌ Customer service config not found: {config_path}")
            return False

        try:
            if not self._test_create_script(provider, api_key, env, config_path):
                return False

            if not self._test_run_script(provider, api_key, env):
                return False

            print(f"🎉 Both create.py and run.py work with {provider}!")
            return True

        except subprocess.TimeoutExpired:
            print(f"❌ Test timed out with {provider}")
            return False
        except Exception as e:
            print(f"❌ Error testing {provider}: {e}")
            return False

    def _test_create_script(
        self, provider: str, api_key: str, env: dict[str, str], config_path: str
    ) -> bool:
        """
        Test the create.py script with a specific provider and configuration.
        Prints output and error details.

        Args:
            provider: The provider to test.
            api_key: The API key for the provider.
            env: Environment variables.
            config_path: Path to the config file.

        Returns:
            True if create.py works correctly, False otherwise.
        """
        print(f"📝 Step 1: Testing create.py with {provider}...")

        create_cmd = [
            sys.executable,
            "create.py",
            "--config",
            config_path,
            "--output-dir",
            "./examples/customer_service",
            "--llm_provider",
            provider,
            "--model",
            APITestConfiguration.DEFAULT_MODELS[provider],
            "--no-ui",
        ]

        create_returncode, create_stdout, create_stderr = (
            self.executor.execute_command_with_realtime_output(
                create_cmd, env, timeout=600
            )
        )

        if create_returncode != 0:
            print(f"❌ create.py failed with {provider}: {create_stderr}")
            return False

        print(f"✅ create.py succeeded with {provider}")
        return True

    def _test_run_script(
        self, provider: str, api_key: str, env: dict[str, str]
    ) -> bool:
        """
        Test the run.py script with a specific provider, sending a test input.
        Prints output and error details.

        Args:
            provider: The provider to test.
            api_key: The API key for the provider.
            env: Environment variables.

        Returns:
            True if run.py works correctly, False otherwise.
        """
        print(f"🤖 Step 2: Testing run.py with {provider}...")
        print("💬 Sending test input: 'Tell me about your robots.'")

        run_cmd = [
            sys.executable,
            "run.py",
            "--input-dir",
            "./examples/customer_service",
            "--llm_provider",
            provider,
            "--model",
            APITestConfiguration.DEFAULT_MODELS[provider],
        ]

        test_input = "Tell me about your robots.\nquit\n"
        run_returncode, run_stdout, run_stderr = (
            self.executor.execute_command_with_realtime_output(
                run_cmd, env=env, input_text=test_input, timeout=60
            )
        )

        if run_returncode != 0:
            print(f"❌ run.py failed with {provider}: {run_stderr}")
            return False

        print(f"✅ run.py succeeded with {provider}")
        print(f"📤 Agent response preview: {run_stdout[:500]}...")
        return True


class ConnectivityTester:
    """
    Tests network connectivity to API endpoints and checks for required package installations.
    Ensures that the environment is ready for API key validation and end-to-end tests.
    """

    def __init__(self) -> None:
        """
        Initialize the ConnectivityTester with a default timeout for connectivity checks.
        """
        self.timeout = APITestConfiguration.CONNECTIVITY_TIMEOUT_SECONDS

    def check_connectivity_to_url(
        self, url: str, name: str, timeout: int | None = None
    ) -> bool:
        """
        Check network connectivity to a specific URL, printing the result.

        Args:
            url: URL to test connectivity to.
            name: Name for logging purposes.
            timeout: Optional timeout override in seconds.

        Returns:
            True if connectivity is successful, False otherwise.
        """
        timeout = timeout or self.timeout
        try:
            requests.get(url, timeout=timeout)
            print(f"✅ Network connectivity to {name} is working")
            return True
        except Exception as e:
            print(f"⚠️  {name} connectivity test failed: {e}")
            return False

    def check_package_availability(self) -> list[str]:
        """
        Check if required packages are installed, printing the result for each.

        Returns:
            List of missing package names (empty if all are installed).
        """
        required_packages = ["requests", "python-dotenv"]
        missing_packages = []

        for package in required_packages:
            try:
                import importlib.util

                if package == "requests":
                    if importlib.util.find_spec("requests") is None:
                        raise ImportError
                    print("✅ requests is installed")
                elif package == "python-dotenv":
                    if importlib.util.find_spec("dotenv") is None:
                        raise ImportError
                    print("✅ python-dotenv is installed")
            except ImportError:
                print(f"❌ {package} is not installed")
                missing_packages.append(package)

        return missing_packages

    def test_api_connectivity(self) -> tuple[int, int]:
        """
        Test connectivity to all configured API endpoints, printing the result for each.

        Returns:
            Tuple of (successful_connections, total_connections).
        """
        connectivity_results = []
        for url, name in APITestConfiguration.API_ENDPOINTS:
            result = self.check_connectivity_to_url(url, name)
            connectivity_results.append(result)

        successful_connections = sum(connectivity_results)
        total_connections = len(connectivity_results)

        if successful_connections == total_connections:
            print(f"✅ All {total_connections} API endpoints are reachable")
        elif successful_connections > 0:
            print(
                f"⚠️  {successful_connections}/{total_connections} API endpoints are reachable"
            )
        else:
            print(f"❌ None of the {total_connections} API endpoints are reachable")

        print(
            "💡 If any connectivity tests failed, API key validation might be affected"
        )
        return successful_connections, total_connections

    def check_all_requirements(self) -> bool:
        """
        Check if all requirements (packages and connectivity) are met before running tests.
        Prints detailed feedback for missing requirements.

        Returns:
            True if all requirements are met, False otherwise.
        """
        print("🔍 Checking requirements...")

        missing_packages = self.check_package_availability()
        if missing_packages:
            print(
                f"\n⚠️  Please install missing packages: pip install {' '.join(missing_packages)}"
            )
            return False

        print("✅ All required packages are installed")

        print("🌐 Testing network connectivity...")
        successful, total = self.test_api_connectivity()
        return True


class TestRunner:
    """
    Orchestrates all test components, running the full suite in order and summarizing results.
    This class is the main entry point for running the test script as a whole.
    """

    def __init__(self) -> None:
        """
        Initialize the TestRunner with all necessary test components.
        """
        self.validator = APIKeyValidator()
        self.executor = ProcessExecutor()
        self.connectivity_tester = ConnectivityTester()
        self.environment_tester = EnvironmentTester(self.validator)
        self.provider_utility_tester = ProviderUtilityTester(self.validator)
        self.customer_service_tester = CustomerServiceExampleTester(
            self.executor, self.validator
        )

    def run_all_tests(self, providers: list[str] | None = None) -> bool:
        """
        Run all tests in the test suite in order, printing progress and a final summary.

        Args:
            providers: List of providers to test. If None, tests all providers.

        Returns:
            True if all tests pass, False otherwise.
        """
        if providers is None:
            providers = APITestConfiguration.ALL_PROVIDERS

        print("🚀 Testing create.py and run.py with real API keys...\n")
        print(f"🎯 Testing providers: {', '.join(providers)}")
        print("This script will:")
        print("1. Check required packages")
        print("2. Load API keys from .env file")
        print("3. Validate each API key with actual API calls")
        print(f"4. Test provider utilities with real keys ({' → '.join(providers)})")
        print(f"5. Test customer service example end-to-end ({' → '.join(providers)})")
        print("   - Tests both create.py and run.py for each provider")

        if not self.connectivity_tester.check_all_requirements():
            return False

        env_manager = EnvironmentManager()
        env_vars = env_manager.load_environment_variables()

        test_steps = [
            (
                "Environment Loading",
                self.environment_tester.test_environment_loading,
                providers,
            ),
            (
                "Provider Utilities",
                self.provider_utility_tester.test_provider_utilities_with_real_keys,
                env_vars,
                providers,
            ),
            (
                "Customer Service Example",
                self.customer_service_tester.test_customer_service_example_with_all_providers,
                providers,
            ),
        ]

        results = []
        for step in test_steps:
            step_name = step[0]
            test_func = step[1]
            args = step[2:] if len(step) > 2 else []
            result = self._run_single_test_step(step_name, test_func, *args)
            results.append(result)

        self._display_final_test_summary(results)
        return self._check_all_tests_passed(results)

    def _run_single_test_step(
        self, step_name: str, test_func: Callable, *args: object
    ) -> tuple[str, bool]:
        """
        Run a single test step, printing the step name and result.

        Args:
            step_name: Name of the test step.
            test_func: Function to execute.
            *args: Arguments to pass to the test function.

        Returns:
            Tuple of (step_name, result).
        """
        print(f"\n{'=' * 60}")
        print(f"🔍 STEP: {step_name}")
        print(f"{'=' * 60}")
        result = test_func(*args)
        return step_name, result

    def _display_final_test_summary(self, results: list[tuple[str, bool]]) -> None:
        """
        Print a summary of all test results, including pass/fail status for each step.

        Args:
            results: List of (test_name, result) tuples.
        """
        print(f"\n{'=' * 60}")
        print("📊 FINAL TEST SUMMARY")
        print(f"{'=' * 60}")

        passed = 0
        total = len(results)

        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<30} {status}")
            if result:
                passed += 1

        print(f"{'=' * 60}")
        print(f"Total: {total}, Passed: {passed}, Failed: {total - passed}")

    def _check_all_tests_passed(self, results: list[tuple[str, bool]]) -> bool:
        """
        Check if all tests passed and print a final message with next steps.

        Args:
            results: List of (test_name, result) tuples.

        Returns:
            True if all tests passed, False otherwise.
        """
        passed = sum(1 for _, result in results if result)
        total = len(results)

        if passed == total:
            print("🎉 All tests passed! create.py and run.py work with real API keys.")
            print("✅ API key validation successful")
            print("✅ Provider utilities working correctly")
            print("✅ create.py working with --no-ui flag")
            print("✅ run.py working in interactive mode")
            print("✅ End-to-end customer service example working")
            print("✅ All providers tested in order: OpenAI → Anthropic → Google")
            return True
        else:
            print("⚠️  Some tests failed. Please check:")
            print("   - Your .env file contains valid API keys")
            print("   - API keys have sufficient credits/permissions")
            print("   - Network connectivity for API calls")
            print("   - Required packages are installed")
            return False


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the test script.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Test API keys and customer service example functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_api_keys.py                    # Test all providers
  python test_api_keys.py --providers openai google
  python test_api_keys.py --providers openai anthropic
        """,
    )

    parser.add_argument(
        "--providers",
        nargs="+",
        choices=APITestConfiguration.ALL_PROVIDERS,
        help="Specific providers to test (default: all providers)",
    )

    return parser.parse_args()


def main() -> bool:
    """
    Main entry point for the test suite. Instantiates the TestRunner and runs all tests.

    Returns:
        True if all tests pass, False otherwise.
    """
    args = parse_arguments()
    test_runner = TestRunner()
    return test_runner.run_all_tests(args.providers)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
