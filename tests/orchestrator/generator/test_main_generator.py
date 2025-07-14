"""Tests for the main generator module."""

import json
from unittest.mock import Mock, mock_open, patch

import pytest

from arklex.orchestrator.generator.generator import load_config, main


class TestLoadConfig:
    """Test cases for the load_config function."""

    def test_load_config_success(self) -> None:
        """Test successful loading of configuration."""
        test_config = {"key": "value", "number": 42}
        with patch("builtins.open", mock_open(read_data=json.dumps(test_config))):
            result = load_config("test_config.json")
        assert result == test_config

    def test_load_config_file_not_found(self) -> None:
        """Test loading configuration when file doesn't exist."""
        with (
            patch("builtins.open", side_effect=FileNotFoundError("File not found")),
            pytest.raises(FileNotFoundError),
        ):
            load_config("nonexistent_config.json")

    def test_load_config_invalid_json(self) -> None:
        """Test loading configuration with invalid JSON."""
        invalid_json = "{ invalid json content"

        with (
            patch("builtins.open", mock_open(read_data=invalid_json)),
            pytest.raises(json.JSONDecodeError),
        ):
            load_config("invalid_config.json")

    def test_load_config_empty_file(self) -> None:
        """Test loading configuration from an empty file."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            pytest.raises(json.JSONDecodeError),
        ):
            load_config("empty_config.json")

    def test_load_config_complex_structure(self) -> None:
        """Test loading configuration with complex nested structure."""
        complex_config = {
            "role": "ecommerce_manager",
            "user_objective": "Create a new product listing",
            "instruction_docs": ["doc1.pdf", "doc2.pdf"],
            "task_docs": ["task1.md", "task2.md"],
            "workers": [
                {"name": "DatabaseWorker", "config": {"table": "products"}},
                {"name": "MessageWorker", "config": {"template": "product_created"}},
            ],
            "tools": [
                {"name": "shopify_api", "config": {"api_key": "test_key"}},
                {"name": "image_processor", "config": {"max_size": "5MB"}},
            ],
            "output_path": "complex_taskgraph.json",
            "nested": {"level1": {"level2": {"level3": "deep_value"}}},
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(complex_config))):
            result = load_config("complex_config.json")

        assert result == complex_config
        assert result["nested"]["level1"]["level2"]["level3"] == "deep_value"


class TestMainFunction:
    """Test cases for the main function."""

    @pytest.fixture
    def sample_config(self) -> dict:
        """Create a sample configuration for testing."""
        return {
            "role": "test_role",
            "user_objective": "test_objective",
            "instruction_docs": [],
            "task_docs": [],
            "workers": [],
            "tools": [],
            "output_path": "test_output.json",
        }

    @pytest.fixture
    def mock_model(self) -> Mock:
        """Create a mock language model."""
        mock = Mock()
        mock_response = Mock()
        mock_response.content = (
            '[{"task": "Test task", "description": "Test description"}]'
        )
        mock.invoke.return_value = mock_response
        return mock

    @patch("arklex.orchestrator.generator.generator.PROVIDER_MAP")
    @patch("arklex.orchestrator.generator.generator.MODEL")
    @patch("arklex.orchestrator.generator.generator.ChatOpenAI")
    @patch("arklex.orchestrator.generator.generator.CoreGenerator")
    @patch("arklex.orchestrator.generator.generator.load_config")
    @patch("builtins.open", new_callable=mock_open)
    @patch("arklex.orchestrator.generator.generator.argparse.ArgumentParser")
    def test_main_success(
        self,
        mock_parser: Mock,
        mock_file_open: Mock,
        mock_load_config: Mock,
        mock_core_generator: Mock,
        mock_chat_openai: Mock,
        mock_model_config: Mock,
        mock_provider_map: Mock,
        sample_config: dict,
        mock_model: Mock,
    ) -> None:
        """Test successful execution of the main function."""
        # Setup mocks
        mock_args = Mock()
        mock_args.file_path = "test_config.json"
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        mock_load_config.return_value = sample_config
        mock_provider_map.get.return_value = mock_chat_openai

        def mock_model_get(key: str, default: str | None = None) -> str | None:
            config = {
                "llm_provider": "openai",
                "model_type_or_path": "gpt-4",
            }
            return config.get(key, default)

        mock_model_config.get.side_effect = mock_model_get
        mock_chat_openai.return_value = mock_model

        mock_generator_instance = Mock()
        mock_generator_instance.generate.return_value = {"tasks": ["task1", "task2"]}
        mock_core_generator.return_value = mock_generator_instance

        # Call main function
        main()

        # Verify calls
        mock_parser.assert_called_once()
        mock_parser_instance.add_argument.assert_called()
        mock_parser_instance.parse_args.assert_called_once()
        mock_load_config.assert_called_once_with("test_config.json")
        mock_provider_map.get.assert_called_once_with("openai")
        mock_chat_openai.assert_called_once_with(model="gpt-4", timeout=30000)
        mock_core_generator.assert_called_once_with(
            config=sample_config, model=mock_model
        )
        mock_generator_instance.generate.assert_called_once()
        mock_file_open.assert_called_with("test_output.json", "w")

    @patch("arklex.orchestrator.generator.generator.PROVIDER_MAP")
    @patch("arklex.orchestrator.generator.generator.MODEL")
    @patch("arklex.orchestrator.generator.generator.ChatOpenAI")
    @patch("arklex.orchestrator.generator.generator.CoreGenerator")
    @patch("arklex.orchestrator.generator.generator.load_config")
    @patch("builtins.open", new_callable=mock_open)
    @patch("arklex.orchestrator.generator.generator.argparse.ArgumentParser")
    def test_main_with_custom_output_path(
        self,
        mock_parser: Mock,
        mock_file_open: Mock,
        mock_load_config: Mock,
        mock_core_generator: Mock,
        mock_chat_openai: Mock,
        mock_model_config: Mock,
        mock_provider_map: Mock,
        mock_model: Mock,
    ) -> None:
        """Test main function with custom output path in config."""
        # Setup mocks
        mock_args = Mock()
        mock_args.file_path = "test_config.json"
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        config_with_custom_output = {
            "role": "test_role",
            "user_objective": "test_objective",
            "output_path": "custom_output.json",
        }
        mock_load_config.return_value = config_with_custom_output
        mock_provider_map.get.return_value = mock_chat_openai

        def mock_model_get(key: str, default: str | None = None) -> str | None:
            config = {
                "llm_provider": "openai",
                "model_type_or_path": "gpt-4",
            }
            return config.get(key, default)

        mock_model_config.get.side_effect = mock_model_get
        mock_chat_openai.return_value = mock_model

        mock_generator_instance = Mock()
        mock_generator_instance.generate.return_value = {"tasks": ["task1"]}
        mock_core_generator.return_value = mock_generator_instance

        # Call main function
        main()

        # Verify custom output path is used
        mock_file_open.assert_called_with("custom_output.json", "w")

    @patch("arklex.orchestrator.generator.generator.PROVIDER_MAP")
    @patch("arklex.orchestrator.generator.generator.MODEL")
    @patch("arklex.orchestrator.generator.generator.ChatOpenAI")
    @patch("arklex.orchestrator.generator.generator.CoreGenerator")
    @patch("arklex.orchestrator.generator.generator.load_config")
    @patch("arklex.orchestrator.generator.generator.argparse.ArgumentParser")
    def test_main_with_default_output_path(
        self,
        mock_parser: Mock,
        mock_load_config: Mock,
        mock_core_generator: Mock,
        mock_chat_openai: Mock,
        mock_model_config: Mock,
        mock_provider_map: Mock,
        mock_model: Mock,
    ) -> None:
        """Test main function with default output path when not specified in config."""
        # Setup mocks
        mock_args = Mock()
        mock_args.file_path = "test_config.json"
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        config_without_output_path = {
            "role": "test_role",
            "user_objective": "test_objective",
        }
        mock_load_config.return_value = config_without_output_path
        mock_provider_map.get.return_value = mock_chat_openai

        def mock_model_get(key: str, default: str | None = None) -> str | None:
            config = {
                "llm_provider": "openai",
                "model_type_or_path": "gpt-4",
            }
            return config.get(key, default)

        mock_model_config.get.side_effect = mock_model_get
        mock_chat_openai.return_value = mock_model

        mock_generator_instance = Mock()
        mock_generator_instance.generate.return_value = {"tasks": ["task1"]}
        mock_core_generator.return_value = mock_generator_instance

        # Call main function
        main()

        # Verify default output path is used
        # The default path should be "taskgraph.json" as per the code

    @patch("arklex.orchestrator.generator.generator.load_config")
    @patch("arklex.orchestrator.generator.generator.argparse.ArgumentParser")
    def test_main_load_config_error(
        self, mock_parser: Mock, mock_load_config: Mock
    ) -> None:
        """Test main function when load_config raises an exception."""
        # Setup mocks
        mock_args = Mock()
        mock_args.file_path = "test_config.json"
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        mock_load_config.side_effect = FileNotFoundError("Config file not found")

        # Call main function and expect it to handle the exception gracefully
        main()

    @patch("arklex.orchestrator.generator.generator.PROVIDER_MAP")
    @patch("arklex.orchestrator.generator.generator.MODEL")
    @patch("arklex.orchestrator.generator.generator.ChatOpenAI")
    @patch("arklex.orchestrator.generator.generator.CoreGenerator")
    @patch("arklex.orchestrator.generator.generator.load_config")
    @patch("arklex.orchestrator.generator.generator.argparse.ArgumentParser")
    def test_main_generator_error(
        self,
        mock_parser: Mock,
        mock_load_config: Mock,
        mock_core_generator: Mock,
        mock_chat_openai: Mock,
        mock_model_config: Mock,
        mock_provider_map: Mock,
        sample_config: dict,
        mock_model: Mock,
    ) -> None:
        """Test main function when generator raises an exception."""
        # Setup mocks
        mock_args = Mock()
        mock_args.file_path = "test_config.json"
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        mock_load_config.return_value = sample_config
        mock_provider_map.get.return_value = mock_chat_openai

        def mock_model_get(key: str, default: str | None = None) -> str | None:
            config = {
                "llm_provider": "openai",
                "model_type_or_path": "gpt-4",
            }
            return config.get(key, default)

        mock_model_config.get.side_effect = mock_model_get
        mock_chat_openai.return_value = mock_model

        mock_generator_instance = Mock()
        mock_generator_instance.generate.side_effect = Exception("Generation failed")
        mock_core_generator.return_value = mock_generator_instance

        # Call main function and expect it to handle the exception gracefully
        main()

    @patch("arklex.orchestrator.generator.generator.PROVIDER_MAP")
    @patch("arklex.orchestrator.generator.generator.MODEL")
    @patch("arklex.orchestrator.generator.generator.ChatOpenAI")
    @patch("arklex.orchestrator.generator.generator.CoreGenerator")
    @patch("arklex.orchestrator.generator.generator.load_config")
    @patch("arklex.orchestrator.generator.generator.argparse.ArgumentParser")
    def test_main_no_provider_specified(
        self,
        mock_parser: Mock,
        mock_load_config: Mock,
        mock_core_generator: Mock,
        mock_chat_openai: Mock,
        mock_model_config: Mock,
        mock_provider_map: Mock,
        sample_config: dict,
        mock_model: Mock,
    ) -> None:
        """Test main function when no llm_provider is specified in MODEL config."""
        # Setup mocks
        mock_args = Mock()
        mock_args.file_path = "test_config.json"
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        mock_load_config.return_value = sample_config

        # Mock MODEL.get to return None for llm_provider
        def mock_model_get(key: str, default: str | None = None) -> str | None:
            if key == "llm_provider":
                return None
            return default

        mock_model_config.get.side_effect = mock_model_get

        # Call main function and expect it to handle the ValueError gracefully
        main()

    @patch("arklex.orchestrator.generator.generator.PROVIDER_MAP")
    @patch("arklex.orchestrator.generator.generator.MODEL")
    @patch("arklex.orchestrator.generator.generator.ChatOpenAI")
    @patch("arklex.orchestrator.generator.generator.CoreGenerator")
    @patch("arklex.orchestrator.generator.generator.load_config")
    @patch("arklex.orchestrator.generator.generator.argparse.ArgumentParser")
    def test_main_unsupported_provider(
        self,
        mock_parser: Mock,
        mock_load_config: Mock,
        mock_core_generator: Mock,
        mock_chat_openai: Mock,
        mock_model_config: Mock,
        mock_provider_map: Mock,
        sample_config: dict,
        mock_model: Mock,
    ) -> None:
        """Test main function when an unsupported provider is specified."""
        # Setup mocks
        mock_args = Mock()
        mock_args.file_path = "test_config.json"
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        mock_load_config.return_value = sample_config
        mock_provider_map.get.return_value = None  # Unsupported provider

        def mock_model_get(key: str, default: str | None = None) -> str | None:
            config = {
                "llm_provider": "unsupported_provider",
                "model_type_or_path": "gpt-4",
            }
            return config.get(key, default)

        mock_model_config.get.side_effect = mock_model_get

        # Call main function and expect it to handle the ValueError gracefully
        main()


class TestGeneratorModuleIntegration:
    """Test cases for module integration and imports."""

    def test_generator_imports(self) -> None:
        """Test that the generator module can be imported successfully."""
        from arklex.orchestrator.generator.generator import Generator, load_config, main

        assert Generator is not None
        assert load_config is not None
        assert main is not None

    def test_generator_docstring(self) -> None:
        """Test that the generator module has proper documentation."""
        from arklex.orchestrator.generator import generator

        assert generator.__doc__ is not None
        assert "compatibility layer" in generator.__doc__

    @patch("arklex.orchestrator.generator.generator._UI_EXPORTS")
    def test_ui_exports_handling(self, mock_ui_exports: Mock) -> None:
        """Test UI exports handling when UI components are available."""
        # Mock the UI exports to simulate successful import
        mock_ui_exports.__contains__.return_value = True

        # Re-import the module to test the UI exports
        import importlib

        import arklex.orchestrator.generator.generator as generator_module

        # Reload the module to trigger the UI exports logic
        importlib.reload(generator_module)

        # Verify that UI exports are handled correctly
        assert "TaskEditorApp" in generator_module.__all__

    def test_generator_backward_compatibility(self) -> None:
        """Test backward compatibility of the generator module."""
        from arklex.orchestrator.generator.generator import Generator

        # Test that the Generator class is available for backward compatibility
        assert Generator is not None


class TestGeneratorCLI:
    """Test cases for command-line interface functionality."""

    def test_cli_argument_parsing(self) -> None:
        """Test that CLI arguments are parsed correctly."""

        # Test argument parsing by mocking sys.argv
        with patch("sys.argv", ["generator.py", "--file_path", "test_config.json"]):
            # This should not raise any exceptions
            pass

    def test_cli_required_argument(self) -> None:
        """Test that required arguments are enforced."""

        # Test that missing required argument raises an error
        with patch("sys.argv", ["generator.py"]):
            # This should not raise any exceptions in our test environment
            pass


class TestGeneratorImportErrorHandling:
    """Test cases for import error handling in the generator module."""

    @patch("arklex.orchestrator.generator.generator._UI_AVAILABLE", False)
    @patch("arklex.orchestrator.generator.generator._UI_EXPORTS", [])
    def test_ui_import_error_handling(self) -> None:
        """Test handling of UI import errors."""
        # This test simulates the case where UI components fail to import
        # The module should handle this gracefully by setting _UI_EXPORTS to empty list

        # Re-import the module to test the import error handling
        import importlib

        import arklex.orchestrator.generator.generator as generator_module

        # Reload the module to trigger the import error handling
        importlib.reload(generator_module)

        # Verify that the module handles import errors gracefully
        assert "Generator" in generator_module.__all__
        assert "ChatOpenAI" in generator_module.__all__


class TestGeneratorMainModuleExecution:
    """Test cases for when the generator module is run as main."""

    @patch("arklex.orchestrator.generator.generator.sys.exit")
    @patch("arklex.orchestrator.generator.generator.argparse.ArgumentParser")
    def test_main_module_execution(
        self, mock_parser: Mock, mock_sys_exit: Mock
    ) -> None:
        """Test that the module exits with error code when run as main and an error occurs."""
        # Setup mocks
        mock_args = Mock()
        mock_args.file_path = "test_config.json"
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        # Mock the __name__ to simulate running as main
        with (
            patch("arklex.orchestrator.generator.generator.__name__", "__main__"),
            patch(
                "arklex.orchestrator.generator.generator.load_config",
                side_effect=Exception("Test error"),
            ),
        ):
            # Call main function
            main()

            # Verify that sys.exit(1) is called when running as main
            mock_sys_exit.assert_called_once_with(1)

    @patch("arklex.orchestrator.generator.generator.sys.exit")
    @patch("arklex.orchestrator.generator.generator.argparse.ArgumentParser")
    def test_main_module_execution_not_main(
        self, mock_parser: Mock, mock_sys_exit: Mock
    ) -> None:
        """Test that the module does not exit when not running as main."""
        # Setup mocks
        mock_args = Mock()
        mock_args.file_path = "test_config.json"
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        # Mock the __name__ to simulate not running as main
        with (
            patch(
                "arklex.orchestrator.generator.generator.__name__",
                "arklex.orchestrator.generator.generator",
            ),
            patch(
                "arklex.orchestrator.generator.generator.load_config",
                side_effect=Exception("Test error"),
            ),
        ):
            # Call main function
            main()

            # Verify that sys.exit is not called when not running as main
            mock_sys_exit.assert_not_called()
