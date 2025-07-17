"""Tests for the simulate_second_pass_convos module.

This module contains comprehensive test cases for second-pass conversation simulation
functionality, including path sampling, conversation generation, and labeled conversation
creation based on intent paths.
"""

from unittest.mock import Mock, patch

from arklex.evaluation.simulate_second_pass_convos import (
    generate_labeled_convos,
    get_labeled_convos,
    get_paths,
    interact,
    sampling_paths,
)


class TestSimulateSecondPassConvos:
    """Test cases for simulate_second_pass_convos module.

    This class contains comprehensive tests for second-pass conversation simulation,
    including path sampling, conversation generation, and labeled conversation creation.
    """

    @patch("arklex.evaluation.simulate_second_pass_convos.random.choices")
    def test_sampling_paths(self, mock_random_choices: Mock) -> None:
        """Test sampling_paths function generates valid intent paths."""
        # Setup
        start_node = "start"
        graph = Mock()
        path_length = 0
        max_turns = 5
        intents = ["start"]

        # Mock graph behavior
        graph.successors.return_value = ["intent1", "intent2"]
        graph.get_edge_data.return_value = {"weight": 1.0}
        mock_random_choices.return_value = ["intent1"]

        # Execute
        result = sampling_paths(start_node, graph, path_length, max_turns, intents)

        # Assert
        assert "intent1" in result
        assert len(result) > 1

    @patch("arklex.evaluation.simulate_second_pass_convos.random.choices")
    def test_sampling_paths_max_turns_reached(self, mock_random_choices: Mock) -> None:
        """Test sampling_paths function when max_turns limit is reached."""
        # Setup
        start_node = "start"
        graph = Mock()
        path_length = 5
        max_turns = 5
        intents = ["start", "intent1", "intent2"]

        # Mock graph behavior to avoid iteration
        graph.successors.return_value = []

        # Execute
        result = sampling_paths(start_node, graph, path_length, max_turns, intents)

        # Assert
        assert result == intents
        mock_random_choices.assert_not_called()

    @patch("arklex.evaluation.simulate_second_pass_convos.random.choices")
    def test_sampling_paths_no_children(self, mock_random_choices: Mock) -> None:
        """Test sampling_paths function when no child nodes exist."""
        # Setup
        start_node = "start"
        graph = Mock()
        path_length = 0
        max_turns = 5
        intents = ["start"]

        # Mock graph behavior - no children
        graph.successors.return_value = []

        # Execute
        result = sampling_paths(start_node, graph, path_length, max_turns, intents)

        # Assert
        assert result == intents
        mock_random_choices.assert_not_called()

    @patch("arklex.evaluation.simulate_second_pass_convos.sampling_paths")
    def test_get_paths(self, mock_sampling_paths: Mock) -> None:
        """Test get_paths function generates multiple intent paths."""
        # Setup
        graph = Mock()
        num_paths = 3
        max_turns = 5

        mock_sampling_paths.side_effect = [
            ["start", "intent1", "intent2"],
            ["start", "intent3", "intent4"],
            ["start", "intent5"],
        ]

        # Execute
        result = get_paths(graph, num_paths, max_turns)

        # Assert
        assert len(result) == 3
        assert result[0] == ["intent1", "intent2"]
        assert result[1] == ["intent3", "intent4"]
        assert result[2] == ["intent5"]
        assert mock_sampling_paths.call_count == 3

    @patch("arklex.evaluation.simulate_second_pass_convos.query_chatbot")
    @patch("arklex.evaluation.simulate_second_pass_convos.chatgpt_chatbot")
    def test_interact(self, mock_chatgpt: Mock, mock_query_chatbot: Mock) -> None:
        """Test interact function generates conversation from intent path."""
        # Setup
        intent_path = ["intent1", "intent2"]
        summary = "Test company summary"
        model_api = "test_api"
        model_params = {}
        client = Mock()

        mock_chatgpt.return_value = "Assistant response"
        mock_query_chatbot.return_value = {
            "answer": "User response",
            "parameters": {"test": "params"},
        }

        # Execute
        env_config = {"workers": [], "tools": []}
        result = interact(
            intent_path, summary, model_api, model_params, client, env_config
        )

        # Assert
        assert len(result) > 0
        assert any("role" in msg for msg in result)
        assert mock_chatgpt.call_count == 2
        assert mock_query_chatbot.call_count == 2

    @patch("arklex.evaluation.simulate_second_pass_convos.interact")
    @patch("arklex.evaluation.simulate_second_pass_convos.flip_hist")
    @patch("arklex.evaluation.simulate_second_pass_convos.filter_convo")
    def test_generate_labeled_convos(
        self, mock_filter_convo: Mock, mock_flip_hist: Mock, mock_interact: Mock
    ) -> None:
        """Test generate_labeled_convos function creates labeled conversations."""
        # Setup
        intent_paths = [["intent1", "intent2"], ["intent3", "intent4"]]
        summary = "Test company summary"
        model_api = "test_api"
        model_params = {}
        client = Mock()

        mock_interact.return_value = [{"role": "user", "content": "Hello"}]
        mock_filter_convo.return_value = [{"role": "user", "content": "Hello"}]
        mock_flip_hist.return_value = [{"role": "user", "content": "Hello"}]

        # Execute
        env_config = {"workers": [], "tools": []}
        result = generate_labeled_convos(
            intent_paths, summary, model_api, model_params, client, env_config
        )

        # Assert
        assert len(result) == 2
        assert all(isinstance(convo, list) for convo in result)
        assert mock_interact.call_count == 2

    @patch("arklex.evaluation.simulate_second_pass_convos.generate_labeled_convos")
    @patch("arklex.evaluation.simulate_second_pass_convos.get_paths")
    @patch("arklex.evaluation.simulate_second_pass_convos.build_intent_graph")
    def test_get_labeled_convos(
        self, mock_build_graph: Mock, mock_get_paths: Mock, mock_generate_convos: Mock
    ) -> None:
        """Test get_labeled_convos function orchestrates labeled conversation generation."""
        # Setup
        first_pass_data = [{"conversation": [{"role": "user", "content": "Hello"}]}]
        model_api = "test_api"
        synthetic_data_params = {"num_convos": 2, "max_turns": 5}
        model_params = {}
        config = {"intro": "Test company", "client": Mock(), "workers": [], "tools": []}

        mock_graph = Mock()
        mock_build_graph.return_value = mock_graph
        mock_get_paths.return_value = [["intent1", "intent2"], ["intent3", "intent4"]]
        mock_generate_convos.return_value = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Hi"}],
        ]

        # Execute
        result = get_labeled_convos(
            first_pass_data, model_api, synthetic_data_params, model_params, config
        )

        # Assert
        assert len(result) == 2
        assert all(isinstance(convo, list) for convo in result)
        mock_build_graph.assert_called_once_with(first_pass_data)
        mock_get_paths.assert_called_once_with(mock_graph, 2, 5)
        mock_generate_convos.assert_called_once()

    @patch("arklex.evaluation.simulate_second_pass_convos.random.choices")
    def test_sampling_paths_edge_case_empty_intents(
        self, mock_random_choices: Mock
    ) -> None:
        """Test sampling_paths function with empty intents list."""
        # Setup
        start_node = "start"
        graph = Mock()
        path_length = 0
        max_turns = 5
        intents = []  # Empty intents list

        # Mock graph behavior
        graph.successors.return_value = ["intent1", "intent2"]
        graph.get_edge_data.return_value = {"weight": 1.0}
        mock_random_choices.return_value = ["intent1"]

        # Execute
        result = sampling_paths(start_node, graph, path_length, max_turns, intents)

        # Assert
        assert len(result) > 0
        assert "intent1" in result

    @patch("arklex.evaluation.simulate_second_pass_convos.query_chatbot")
    @patch("arklex.evaluation.simulate_second_pass_convos.chatgpt_chatbot")
    def test_interact_single_intent(
        self, mock_chatgpt: Mock, mock_query_chatbot: Mock
    ) -> None:
        """Test interact function with single intent in path."""
        # Setup
        intent_path = ["intent1"]  # Single intent
        summary = "Test company summary"
        model_api = "test_api"
        model_params = {}
        client = Mock()

        mock_chatgpt.return_value = "Assistant response"
        mock_query_chatbot.return_value = {
            "answer": "User response",
            "parameters": {"test": "params"},
        }

        # Execute
        env_config = {"workers": [], "tools": []}
        result = interact(
            intent_path, summary, model_api, model_params, client, env_config
        )

        # Assert
        assert len(result) > 0
        assert mock_chatgpt.call_count == 1
        assert mock_query_chatbot.call_count == 1

    @patch("arklex.evaluation.simulate_second_pass_convos.query_chatbot")
    @patch("arklex.evaluation.simulate_second_pass_convos.chatgpt_chatbot")
    def test_interact_with_parameters_update(
        self, mock_chatgpt: Mock, mock_query_chatbot: Mock
    ) -> None:
        """Test interact function updates parameters during conversation."""
        # Setup
        intent_path = ["intent1", "intent2"]
        summary = "Test company summary"
        model_api = "test_api"
        model_params = {}
        client = Mock()

        mock_chatgpt.return_value = "Assistant response"
        mock_query_chatbot.return_value = {
            "answer": "User response",
            "parameters": {
                "taskgraph": {"curr_global_intent": "updated_intent"},
                "memory": {"trajectory": [["updated_trajectory"]]},
            },
        }

        # Execute
        env_config = {"workers": [], "tools": []}
        result = interact(
            intent_path, summary, model_api, model_params, client, env_config
        )

        # Assert
        assert len(result) > 0
        assert mock_chatgpt.call_count == 2
        assert mock_query_chatbot.call_count == 2

    @patch("builtins.open")
    @patch("json.dump")
    @patch("arklex.evaluation.simulate_second_pass_convos.get_labeled_convos")
    @patch("json.load")
    def test_main_execution_block_with_file_operations(
        self,
        mock_json_load: Mock,
        mock_get_labeled_convos: Mock,
        mock_json_dump: Mock,
        mock_open: Mock,
    ) -> None:
        """Test the main execution block including file read/write operations."""
        # Setup mock data
        mock_data = [{"conversation": [{"role": "user", "content": "Hello"}]}]
        mock_config = {"intro": "Test company", "client": Mock()}
        mock_labeled_convos = [[{"role": "user", "content": "Hello"}]]

        # Mock file operations
        mock_file_context = Mock()
        mock_open.return_value.__enter__.return_value = mock_file_context
        mock_json_load.side_effect = [mock_data, mock_config]
        mock_get_labeled_convos.return_value = mock_labeled_convos

        # Import the module and call main()
        import arklex.evaluation.simulate_second_pass_convos as module

        module.main()

        # Assert file operations were called correctly
        assert mock_open.call_count >= 3  # 2 reads, 1 write
        assert mock_json_load.call_count == 2  # Two files loaded
        assert mock_json_dump.call_count == 1  # One file written
