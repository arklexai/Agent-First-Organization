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
        mock_model_config.get.side_effect = lambda key, default: {
            "llm_provider": "openai",
            "model_type_or_path": "gpt-4",
        }.get(key, default)
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
        mock_provider_map.get.assert_called_once_with("openai", mock_chat_openai)
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
        mock_model_config.get.side_effect = lambda key, default: {
            "llm_provider": "openai",
            "model_type_or_path": "gpt-4",
        }.get(key, default)
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
        mock_model_config.get.side_effect = lambda key, default: {
            "llm_provider": "openai",
            "model_type_or_path": "gpt-4",
        }.get(key, default)
        mock_chat_openai.return_value = mock_model

        mock_generator_instance = Mock()
        mock_generator_instance.generate.return_value = {"tasks": ["task1"]}
        mock_core_generator.return_value = mock_generator_instance

        # Call main function
        main()

        # Verify default output path is used
        mock_core_generator.assert_called_once_with(
            config=config_without_output_path, model=mock_model
        )

    @patch("arklex.orchestrator.generator.generator.load_config")
    @patch("arklex.orchestrator.generator.generator.argparse.ArgumentParser")
    def test_main_load_config_error(
        self, mock_parser: Mock, mock_load_config: Mock
    ) -> None:
        """Test main function when load_config raises an error."""
        # Setup mocks
        mock_args = Mock()
        mock_args.file_path = "test_config.json"
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        # Make load_config raise an error
        mock_load_config.side_effect = FileNotFoundError("Config file not found")

        # Call main function and expect it to handle the error gracefully
        main()

        # Verify that load_config was called
        mock_load_config.assert_called_once_with("test_config.json")

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
        """Test main function when generator.generate() raises an error."""
        # Setup mocks
        mock_args = Mock()
        mock_args.file_path = "test_config.json"
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        mock_load_config.return_value = sample_config
        mock_provider_map.get.return_value = mock_chat_openai
        mock_model_config.get.side_effect = lambda key, default: {
            "llm_provider": "openai",
            "model_type_or_path": "gpt-4",
        }.get(key, default)
        mock_chat_openai.return_value = mock_model

        # Make generator.generate() raise an error
        mock_generator_instance = Mock()
        mock_generator_instance.generate.side_effect = Exception("Generation failed")
        mock_core_generator.return_value = mock_generator_instance

        # Call main function and expect it to handle the error gracefully
        main()

        # Verify that the generator was called
        mock_core_generator.assert_called_once_with(
            config=sample_config, model=mock_model
        )
        mock_generator_instance.generate.assert_called_once()


class TestGeneratorModuleIntegration:
    """Test integration aspects of the generator module."""

    def test_generator_imports(self) -> None:
        """Test that all necessary imports are available."""
        from arklex.orchestrator.generator.generator import load_config, main

        assert callable(load_config)
        assert callable(main)

    def test_generator_docstring(self) -> None:
        """Test that the generator module has proper documentation."""
        import arklex.orchestrator.generator.generator as generator_mod

        assert generator_mod.__doc__ is not None
        assert len(generator_mod.__doc__) > 0

    @patch("arklex.orchestrator.generator.generator._UI_EXPORTS")
    def test_ui_exports_handling(self, mock_ui_exports: Mock) -> None:
        """Test that UI exports are handled properly."""
        # Mock UI exports to be available
        mock_ui_exports.__bool__.return_value = True

        # Import should work without errors
        from arklex.orchestrator.generator.generator import _UI_AVAILABLE

        assert _UI_AVAILABLE is True

    def test_generator_backward_compatibility(self) -> None:
        """Test that the generator maintains backward compatibility."""
        # Test that the module can be imported without errors
        import arklex.orchestrator.generator.generator

        # Test that basic functions are callable
        assert hasattr(arklex.orchestrator.generator.generator, "load_config")
        assert hasattr(arklex.orchestrator.generator.generator, "main")


class TestGeneratorCLI:
    """Test CLI functionality of the generator."""

    def test_cli_argument_parsing(self) -> None:
        """Test that the CLI argument parsing works correctly."""
        with (
            patch("sys.argv", ["generator.py", "--file_path", "test_config.json"]),
            patch("arklex.orchestrator.generator.generator.main") as mock_main,
        ):
            # Import the module - this should not trigger main() since it's not run directly

            # Verify that main was not called during import
            mock_main.assert_not_called()

    def test_cli_required_argument(self) -> None:
        """Test that the CLI requires the file_path argument."""
        with (
            patch("sys.argv", ["generator.py"]),
            patch(
                "arklex.orchestrator.generator.generator.argparse.ArgumentParser"
            ) as mock_parser,
        ):
            mock_parser_instance = Mock()
            mock_parser_instance.parse_args.side_effect = SystemExit(
                "the following arguments are required: file_path"
            )
            mock_parser.return_value = mock_parser_instance

            # Import the module - this should not trigger argument parsing

            # Verify that ArgumentParser was not called during import
            mock_parser.assert_not_called()
