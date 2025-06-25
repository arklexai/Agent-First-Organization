"""Tests for conversation information extraction module.

This module tests the conversation analysis utilities including edge counting,
intent graph building, goal checking, and task completion metrics extraction.
"""

from unittest.mock import patch, MagicMock
import networkx as nx

from arklex.evaluation import extract_conversation_info


class TestGetEdgesAndCounts:
    """Test cases for get_edges_and_counts function."""

    def test_get_edges_and_counts_basic_conversation(self) -> None:
        """Test edge counting with basic conversation."""
        data = [
            {
                "convo": [
                    {"role": "user", "intent": "greet", "content": "Hello"},
                    {"role": "assistant", "intent": "greet", "content": "Hi there!"},
                    {"role": "user", "intent": "goodbye", "content": "Goodbye"},
                    {"role": "assistant", "intent": "goodbye", "content": "See you!"},
                ]
            }
        ]
        result = extract_conversation_info.get_edges_and_counts(data)

        # The function filters conversations (skips first 2 turns) and only counts user turns
        # So with this data, after filtering, we get: [{"role": "user", "intent": "goodbye", "content": "Goodbye"}]
        # Since there's only one user turn after filtering, it creates an edge from "start" to "goodbye"
        assert len(result) == 1
        assert result[("start", "goodbye")] == 1

    def test_get_edges_and_counts_multiple_conversations(self) -> None:
        """Test edge counting with multiple conversations."""
        data = [
            {
                "convo": [
                    {"role": "user", "intent": "greet", "content": "Hello"},
                    {"role": "assistant", "intent": "greet", "content": "Hi there!"},
                    {"role": "user", "intent": "goodbye", "content": "Goodbye"},
                ]
            },
            {
                "convo": [
                    {"role": "user", "intent": "greet", "content": "Hello"},
                    {"role": "assistant", "intent": "greet", "content": "Hi there!"},
                    {"role": "user", "intent": "help", "content": "Help me"},
                ]
            },
        ]
        result = extract_conversation_info.get_edges_and_counts(data)

        # After filtering, each conversation has only one user turn
        # First conversation: edge from "start" to "goodbye"
        # Second conversation: edge from "start" to "help"
        assert len(result) == 2
        assert result[("start", "goodbye")] == 1
        assert result[("start", "help")] == 1

    def test_get_edges_and_counts_empty_data(self) -> None:
        """Test edge counting with empty data."""
        data = []
        result = extract_conversation_info.get_edges_and_counts(data)
        assert result == {}

    def test_get_edges_and_counts_empty_conversation(self) -> None:
        """Test edge counting with empty conversation."""
        data = [{"convo": []}]
        result = extract_conversation_info.get_edges_and_counts(data)
        assert result == {}

    def test_get_edges_and_counts_assistant_first(self) -> None:
        """Test edge counting when assistant speaks first."""
        data = [
            {
                "convo": [
                    {"role": "assistant", "intent": "greet", "content": "Hello"},
                    {"role": "user", "intent": "help", "content": "Help me"},
                    {"role": "assistant", "intent": "help", "content": "I can help"},
                ]
            }
        ]
        result = extract_conversation_info.get_edges_and_counts(data)

        # After filtering (skips first 2 turns), we get: [{"role": "assistant", "intent": "help", "content": "I can help"}]
        # Only assistant turn remains, so no edges
        assert len(result) == 0

    def test_get_edges_and_counts_single_user_turn(self) -> None:
        """Test edge counting with single user turn."""
        data = [{"convo": [{"role": "user", "intent": "greet", "content": "Hello"}]}]
        result = extract_conversation_info.get_edges_and_counts(data)

        # After filtering (skips first 2 turns), nothing remains
        assert len(result) == 0

    def test_get_edges_and_counts_with_multiple_user_turns_after_filtering(
        self,
    ) -> None:
        """Test edge counting with multiple user turns after filtering."""
        data = [
            {
                "convo": [
                    {"role": "user", "intent": "start", "content": "Start"},
                    {"role": "assistant", "intent": "start", "content": "Welcome"},
                    {"role": "user", "intent": "greet", "content": "Hello"},
                    {"role": "assistant", "intent": "greet", "content": "Hi there!"},
                    {"role": "user", "intent": "goodbye", "content": "Goodbye"},
                ]
            }
        ]
        result = extract_conversation_info.get_edges_and_counts(data)

        # After filtering (skips first 2 turns), we get:
        # [{"role": "user", "intent": "greet", "content": "Hello"}, {"role": "assistant", "intent": "greet", "content": "Hi there!"}, {"role": "user", "intent": "goodbye", "content": "Goodbye"}]
        # For the first user turn (greet), prev_intent = "start" (since i=0)
        # For the second user turn (goodbye), prev_intent = convo[i-2]["intent"] = convo[1]["intent"] = "greet"
        assert result[("start", "greet")] == 1
        assert result[("greet", "goodbye")] == 1

    def test_get_edges_and_counts_no_intent_field(self) -> None:
        """Test edge counting with missing intent field."""
        data = [
            {
                "convo": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ]
            }
        ]
        result = extract_conversation_info.get_edges_and_counts(data)

        # Should handle missing intent gracefully
        assert result == {}


class TestBuildIntentGraph:
    """Test cases for build_intent_graph function."""

    def test_build_intent_graph_basic(self) -> None:
        """Test building intent graph with basic data."""
        data = [
            {
                "convo": [
                    {"role": "user", "intent": "start", "content": "Start"},
                    {"role": "assistant", "intent": "start", "content": "Welcome"},
                    {"role": "user", "intent": "greet", "content": "Hello"},
                    {"role": "assistant", "intent": "greet", "content": "Hi there!"},
                    {"role": "user", "intent": "goodbye", "content": "Goodbye"},
                ]
            }
        ]
        graph = extract_conversation_info.build_intent_graph(data)

        assert isinstance(graph, nx.DiGraph)
        assert graph.has_edge("start", "greet")
        assert graph.has_edge("greet", "goodbye")
        assert graph["start"]["greet"]["weight"] == 1
        assert graph["greet"]["goodbye"]["weight"] == 1

    def test_build_intent_graph_multiple_edges(self) -> None:
        """Test building intent graph with multiple edges."""
        data = [
            {
                "convo": [
                    {"role": "user", "intent": "start", "content": "Start"},
                    {"role": "assistant", "intent": "start", "content": "Welcome"},
                    {"role": "user", "intent": "greet", "content": "Hello"},
                    {"role": "assistant", "intent": "greet", "content": "Hi there!"},
                    {"role": "user", "intent": "help", "content": "Help me"},
                ]
            },
            {
                "convo": [
                    {"role": "user", "intent": "start", "content": "Start"},
                    {"role": "assistant", "intent": "start", "content": "Welcome"},
                    {"role": "user", "intent": "greet", "content": "Hello"},
                    {"role": "assistant", "intent": "greet", "content": "Hi there!"},
                    {"role": "user", "intent": "help", "content": "Help me"},
                ]
            },
        ]
        graph = extract_conversation_info.build_intent_graph(data)

        assert graph["start"]["greet"]["weight"] == 2
        assert graph["greet"]["help"]["weight"] == 2

    def test_build_intent_graph_empty_data(self) -> None:
        """Test building intent graph with empty data."""
        data = []
        graph = extract_conversation_info.build_intent_graph(data)

        assert isinstance(graph, nx.DiGraph)
        assert len(graph.nodes()) == 0
        assert len(graph.edges()) == 0


class TestCheckBotGoal:
    """Test cases for check_bot_goal function."""

    @patch("arklex.evaluation.extract_conversation_info.chatgpt_chatbot")
    def test_check_bot_goal_true_response(self, mock_chatbot) -> None:
        """Test bot goal checking with True response."""
        mock_chatbot.return_value = "True"
        convo = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        bot_goal = "Greet the user"
        client = MagicMock()

        result = extract_conversation_info.check_bot_goal(convo, bot_goal, client)

        assert result is True
        mock_chatbot.assert_called_once()

    @patch("arklex.evaluation.extract_conversation_info.chatgpt_chatbot")
    def test_check_bot_goal_false_response(self, mock_chatbot) -> None:
        """Test bot goal checking with False response."""
        mock_chatbot.return_value = "False"
        convo = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        bot_goal = "Greet the user"
        client = MagicMock()

        result = extract_conversation_info.check_bot_goal(convo, bot_goal, client)

        assert result is False
        mock_chatbot.assert_called_once()

    @patch("arklex.evaluation.extract_conversation_info.chatgpt_chatbot")
    def test_check_bot_goal_case_insensitive(self, mock_chatbot) -> None:
        """Test bot goal checking with case insensitive response."""
        mock_chatbot.return_value = "true"
        convo = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        bot_goal = "Greet the user"
        client = MagicMock()

        result = extract_conversation_info.check_bot_goal(convo, bot_goal, client)

        assert result is False  # Should be case sensitive

    @patch("arklex.evaluation.extract_conversation_info.chatgpt_chatbot")
    def test_check_bot_goal_empty_conversation(self, mock_chatbot) -> None:
        """Test bot goal checking with empty conversation."""
        mock_chatbot.return_value = "False"
        convo = []
        bot_goal = "Greet the user"
        client = MagicMock()

        result = extract_conversation_info.check_bot_goal(convo, bot_goal, client)

        assert result is False
        mock_chatbot.assert_called_once()


class TestNumUserTurns:
    """Test cases for num_user_turns function."""

    def test_num_user_turns_basic(self) -> None:
        """Test counting user turns in basic conversation."""
        convo = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good, thanks!"},
        ]
        result = extract_conversation_info.num_user_turns(convo)
        assert result == 2

    def test_num_user_turns_no_user_messages(self) -> None:
        """Test counting user turns with no user messages."""
        convo = [
            {"role": "assistant", "content": "Hi there!"},
            {"role": "assistant", "content": "I'm good, thanks!"},
        ]
        result = extract_conversation_info.num_user_turns(convo)
        assert result == 0

    def test_num_user_turns_empty_conversation(self) -> None:
        """Test counting user turns in empty conversation."""
        convo = []
        result = extract_conversation_info.num_user_turns(convo)
        assert result == 0

    def test_num_user_turns_missing_role(self) -> None:
        """Test counting user turns with missing role field."""
        convo = [
            {"content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = extract_conversation_info.num_user_turns(convo)
        assert result == 1

    def test_num_user_turns_all_user_messages(self) -> None:
        """Test counting user turns with all user messages."""
        convo = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
            {"role": "user", "content": "Goodbye"},
        ]
        result = extract_conversation_info.num_user_turns(convo)
        assert result == 3


class TestExtractTaskCompletionMetrics:
    """Test cases for extract_task_completion_metrics function."""

    @patch("arklex.evaluation.extract_conversation_info.check_bot_goal")
    def test_extract_task_completion_metrics_basic(self, mock_check_bot_goal) -> None:
        """Test extracting task completion metrics with basic data."""
        mock_check_bot_goal.return_value = True
        data = [
            {
                "convo": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                "goal_completion": True,
            },
            {
                "convo": [
                    {"role": "user", "content": "Help"},
                    {"role": "assistant", "content": "I can help!"},
                ],
                "goal_completion": False,
            },
        ]
        client = MagicMock()
        bot_goal = "Help users"

        result = extract_conversation_info.extract_task_completion_metrics(
            data, client, bot_goal
        )

        assert isinstance(result, dict)
        assert "user_task_completion" in result
        assert "user_task_completion_efficiency" in result
        assert "bot_goal_completion" in result
        assert result["user_task_completion"] == 0.5  # 1 out of 2
        assert (
            result["user_task_completion_efficiency"] == 1.0
        )  # 2 user turns / 2 conversations = 1.0

    @patch("arklex.evaluation.extract_conversation_info.check_bot_goal")
    def test_extract_task_completion_metrics_no_bot_goal(
        self, mock_check_bot_goal
    ) -> None:
        """Test extracting task completion metrics without bot goal."""
        data = [
            {
                "convo": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                "goal_completion": True,
            }
        ]
        client = MagicMock()

        result = extract_conversation_info.extract_task_completion_metrics(data, client)

        assert isinstance(result, dict)
        assert "user_task_completion" in result
        assert "user_task_completion_efficiency" in result
        assert "bot_goal_completion" not in result
        assert result["user_task_completion"] == 1.0
        assert result["user_task_completion_efficiency"] == 1.0

    def test_extract_task_completion_metrics_empty_data(self) -> None:
        """Test extracting task completion metrics with empty data."""
        data = []
        client = MagicMock()

        result = extract_conversation_info.extract_task_completion_metrics(data, client)

        assert isinstance(result, str)
        assert "Error" in result

    @patch("arklex.evaluation.extract_conversation_info.check_bot_goal")
    def test_extract_task_completion_metrics_complex_conversation(
        self, mock_check_bot_goal
    ) -> None:
        """Test extracting task completion metrics with complex conversation."""
        mock_check_bot_goal.return_value = True
        data = [
            {
                "convo": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm good!"},
                    {"role": "user", "content": "Goodbye"},
                ],
                "goal_completion": True,
            }
        ]
        client = MagicMock()
        bot_goal = "Have a conversation"

        result = extract_conversation_info.extract_task_completion_metrics(
            data, client, bot_goal
        )

        assert result["user_task_completion"] == 1.0
        assert (
            result["user_task_completion_efficiency"] == 3.0
        )  # 3 user turns / 1 conversation
        assert result["bot_goal_completion"] == 1.0
