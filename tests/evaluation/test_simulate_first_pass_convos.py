"""Tests for the simulate_first_pass_convos module.

This module contains comprehensive test cases for first-pass conversation simulation
functionality, including profile matching, conversation generation, goal completion
checking, and parallel processing capabilities.
"""

from unittest.mock import Mock, mock_open, patch

from arklex.evaluation.simulate_first_pass_convos import (
    check_goal_completion,
    conversation,
    count_matches,
    create_convo_profile,
    generate_conversations,
    get_example_convo,
    get_relevant_vals,
    join_messages,
    retrieve_convo,
    retrieve_prompts,
    simulate_conversations,
)


class TestSimulateFirstPassConvos:
    """Test cases for simulate_first_pass_convos module.

    This class contains comprehensive tests for first-pass conversation simulation,
    including profile matching, conversation generation, and goal completion checking.
    """

    def test_get_relevant_vals(self) -> None:
        """Test get_relevant_vals function extracts correct attribute values."""
        # Setup
        attr: dict[str, str] = {
            "goal": "test_goal",
            "product_experience_level": "beginner",
            "customer_type": "enterprise",
            "persona": "decision_maker",
            "discovery_type": "self_service",
            "buying_behavior": "analytical",
        }

        # Execute
        result = get_relevant_vals(attr)

        # Assert
        assert len(result) == 6
        assert "test_goal" in result
        assert "beginner" in result
        assert "enterprise" in result
        assert "decision_maker" in result
        assert "self_service" in result
        assert "analytical" in result

    def test_count_matches(self) -> None:
        """Test count_matches function with partial matches."""
        # Setup
        list1 = ["a", "b", "c", "d"]
        list2 = ["a", "x", "c", "y"]

        # Execute
        result = count_matches(list1, list2)

        # Assert
        assert result == 2  # "a" and "c" match

    def test_count_matches_no_matches(self) -> None:
        """Test count_matches function with no matching elements."""
        # Setup
        list1 = ["a", "b", "c"]
        list2 = ["x", "y", "z"]

        # Execute
        result = count_matches(list1, list2)

        # Assert
        assert result == 0

    def test_count_matches_all_matches(self) -> None:
        """Test count_matches function with all elements matching."""
        # Setup
        list1 = ["a", "b", "c"]
        list2 = ["a", "b", "c"]

        # Execute
        result = count_matches(list1, list2)

        # Assert
        assert result == 3

    def test_join_messages(self) -> None:
        """Test join_messages function formats conversation correctly."""
        # Setup
        messages: list[dict[str, str]] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Execute
        result = join_messages(messages)

        # Assert
        assert "user: Hello" in result
        assert "assistant: Hi there!" in result

    def test_join_messages_empty_list(self) -> None:
        """Test join_messages function with empty message list."""
        # Setup
        messages: list[dict[str, str]] = []

        # Execute
        result = join_messages(messages)

        # Assert
        assert result == ""

    def test_join_messages_with_bot_follow_up(self) -> None:
        """Test join_messages function with bot_follow_up messages that should be skipped."""
        # Setup
        messages: list[dict[str, str]] = [
            {"role": "user", "content": "Hello"},
            {"role": "bot_follow_up", "content": "This should be skipped"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "bot_follow_up", "content": "This should also be skipped"},
            {"role": "user", "content": "How are you?"},
        ]

        # Execute
        result = join_messages(messages)

        # Assert
        assert "user: Hello" in result
        assert "bot_follow_up: This should be skipped" not in result
        assert "assistant: Hi there!" in result
        assert "bot_follow_up: This should also be skipped" not in result
        assert "user: How are you?" in result

    @patch(
        "arklex.evaluation.simulate_first_pass_convos.chatgpt_chatbot",
        return_value="dummy",
    )
    def test_create_convo_profile(self, mock_chatgpt: Mock) -> None:
        """Test create_convo_profile function."""
        # Setup
        best_match = [
            "goal1",
            "beginner",
            "enterprise",
            "other",
            "self_service",
            "analytical",
        ]
        attr_vals = [
            "goal1",
            "beginner",
            "enterprise",
            "decision_maker",
            "self_service",
            "analytical",
        ]
        summary = "Test company summary"
        client = Mock()
        mock_chatgpt.return_value = "Generated profile"

        # Execute
        result = create_convo_profile(best_match, attr_vals, summary, client)

        # Assert
        assert result == "Generated profile"
        mock_chatgpt.assert_called_once()

    @patch("arklex.evaluation.simulate_first_pass_convos.random.choice")
    @patch("arklex.evaluation.simulate_first_pass_convos.create_convo_profile")
    def test_retrieve_convo(
        self, mock_create_profile: Mock, mock_random_choice: Mock
    ) -> None:
        """Test retrieve_convo function."""
        # Setup
        attr_vals = [
            "goal1",
            "beginner",
            "enterprise",
            "decision_maker",
            "self_service",
            "analytical",
        ]
        all_profiles = [
            "goal1,beginner,enterprise,decision_maker,self_service,analytical"
        ]
        user_convos = {
            "goal1,beginner,enterprise,decision_maker,self_service,analytical": [
                {"message": [{"role": "user", "content": "Hello"}]}
            ]
        }
        summary = "Test company"
        client = Mock()

        mock_random_choice.return_value = {
            "message": [{"role": "user", "content": "Hello"}]
        }
        mock_create_profile.return_value = "Generated profile"

        # Execute
        convo_messages, convo_profile = retrieve_convo(
            attr_vals, all_profiles, user_convos, summary, client
        )

        # Assert
        assert "user: Hello" in convo_messages
        assert convo_profile == "Generated profile"

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"profile1": [{"message": []}]}',
    )
    @patch("arklex.evaluation.simulate_first_pass_convos.retrieve_convo")
    def test_get_example_convo(
        self, mock_retrieve_convo: Mock, mock_file: Mock
    ) -> None:
        """Test get_example_convo function."""
        # Setup
        attr = {
            "goal": "test_goal",
            "product_experience_level": "beginner",
            "customer_type": "enterprise",
            "persona": "decision_maker",
            "discovery_type": "self_service",
            "buying_behavior": "analytical",
        }
        synthetic_data_params = {"data_file": "test_file.json"}
        summary = "Test company"
        client = Mock()

        mock_retrieve_convo.return_value = ("test_convo", "test_profile")

        # Execute
        convo, matching_profile = get_example_convo(
            attr, synthetic_data_params, summary, client
        )

        # Assert
        assert convo == "test_convo"
        assert matching_profile == "test_profile"

    @patch("arklex.evaluation.simulate_first_pass_convos.get_example_convo")
    def test_retrieve_prompts_with_data_file(
        self, mock_get_example_convo: Mock
    ) -> None:
        """Test retrieve_prompts function with data_file."""
        # Setup
        profile = "Test profile"
        goal = "Test goal"
        attr = {"test": "attr"}
        summary = "Test company"
        synthetic_data_params = {"data_file": "test_file.json"}
        client = Mock()

        mock_get_example_convo.return_value = ("test_convo", "test_profile")

        # Execute
        instructional_prompt, start_text = retrieve_prompts(
            profile, goal, attr, summary, synthetic_data_params, client
        )

        # Assert
        assert "test_convo" in instructional_prompt
        assert "test_profile" in instructional_prompt
        assert "Replicate the writing behavior" in start_text

    def test_retrieve_prompts_without_data_file(self) -> None:
        """Test retrieve_prompts function without data_file."""
        # Setup
        profile = "Test profile"
        goal = "Test goal"
        attr = {"test": "attr"}
        summary = "Test company"
        synthetic_data_params = {"data_file": None}
        client = Mock()

        # Execute
        instructional_prompt, start_text = retrieve_prompts(
            profile, goal, attr, summary, synthetic_data_params, client
        )

        # Assert
        assert "Test company" in instructional_prompt
        assert "Test goal" in instructional_prompt
        assert "Test profile" in instructional_prompt
        assert "Humans write short questions" in start_text

    @patch(
        "arklex.evaluation.simulate_first_pass_convos.chatgpt_chatbot",
        return_value="dummy",
    )
    @patch("arklex.evaluation.simulate_first_pass_convos.format_chat_history_str")
    @patch("arklex.evaluation.simulate_first_pass_convos.flip_hist_content_only")
    def test_check_goal_completion_true(
        self, mock_flip: Mock, mock_format: Mock, mock_chatgpt: Mock
    ) -> None:
        """Test check_goal_completion function returning True."""
        # Setup
        goal = "Test goal"
        convo = [
            {"role": "system", "content": "system message"},
            {"role": "user", "content": "user message"},
            {"role": "assistant", "content": "assistant message"},
        ]
        client = Mock()

        mock_flip.return_value = [{"role": "user", "content": "user message"}]
        mock_format.return_value = "formatted conversation"
        mock_chatgpt.return_value = "True"

        # Execute
        result = check_goal_completion(goal, convo, client)

        # Assert
        assert result is True
        mock_chatgpt.assert_called_once()

    @patch(
        "arklex.evaluation.simulate_first_pass_convos.chatgpt_chatbot",
        return_value="dummy",
    )
    @patch("arklex.evaluation.simulate_first_pass_convos.format_chat_history_str")
    @patch("arklex.evaluation.simulate_first_pass_convos.flip_hist_content_only")
    def test_check_goal_completion_false(
        self, mock_flip: Mock, mock_format: Mock, mock_chatgpt: Mock
    ) -> None:
        """Test check_goal_completion function returning False."""
        # Setup
        goal = "Test goal"
        convo = [
            {"role": "system", "content": "system message"},
            {"role": "user", "content": "user message"},
            {"role": "assistant", "content": "assistant message"},
        ]
        client = Mock()

        mock_flip.return_value = [{"role": "user", "content": "user message"}]
        mock_format.return_value = "formatted conversation"
        mock_chatgpt.return_value = "False"

        # Execute
        result = check_goal_completion(goal, convo, client)

        # Assert
        assert result is False
        mock_chatgpt.assert_called_once()

    @patch(
        "arklex.evaluation.simulate_first_pass_convos.chatgpt_chatbot",
        return_value="dummy",
    )
    @patch("arklex.evaluation.simulate_first_pass_convos.query_chatbot")
    @patch("arklex.evaluation.simulate_first_pass_convos.check_goal_completion")
    def test_conversation(
        self, mock_check_goal: Mock, mock_query_chatbot: Mock, mock_chatgpt: Mock
    ) -> None:
        """Test conversation function."""
        # Setup
        model_api = "test_api"
        profile = "Test profile"
        goal = "Test goal"
        attr = {"test": "attr"}
        sys_input = {"input1": "value1"}
        summary = "Test company"
        model_params = {"param1": "value1"}
        synthetic_data_params = {"max_turns": 2, "data_file": None}  # Keep it simple
        env_config = {"client": Mock()}

        mock_chatgpt.return_value = "Test response"
        mock_query_chatbot.return_value = {
            "answer": "Test answer",
            "parameters": {
                "taskgraph": {
                    "curr_global_intent": "test_intent",
                    "curr_node": "test_node",
                },
                "memory": {"trajectory": ["test_trajectory"]},
            },
        }
        # Goal completion check won't happen with max_turns=2, so it should return False
        mock_check_goal.return_value = False

        # Execute
        history, goal_completion = conversation(
            model_api,
            profile,
            goal,
            attr,
            sys_input,
            summary,
            model_params,
            synthetic_data_params,
            env_config,
        )

        # Assert
        assert (
            goal_completion is False
        )  # Should be False since goal completion check doesn't happen
        assert len(history) > 0
        assert history[0]["role"] == "system"

    @patch("arklex.evaluation.simulate_first_pass_convos.conversation")
    def test_generate_conversations(self, mock_conversation: Mock) -> None:
        """Test generate_conversations function."""
        # Setup
        model_api = "test_api"
        profiles = ["profile1", "profile2"]
        goals = ["goal1", "goal2"]
        attributes_list = [{"attr1": "value1"}, {"attr2": "value2"}]
        system_inputs = [{"input1": "value1"}, {"input2": "value2"}]
        summary = "Test company"
        model_params = {"param1": "value1"}
        synthetic_data_params = {"max_turns": 3}
        env_config = {"client": Mock()}

        mock_conversation.return_value = (
            [{"role": "user", "content": "test"}],
            True,
        )

        # Execute
        result = generate_conversations(
            model_api,
            profiles,
            goals,
            attributes_list,
            system_inputs,
            summary,
            model_params,
            synthetic_data_params,
            env_config,
        )

        # Assert
        assert len(result) == 2
        assert all("id" in conv for conv in result)

    @patch(
        "arklex.evaluation.build_user_profiles.chatgpt_chatbot", return_value="dummy"
    )
    @patch(
        "arklex.evaluation.simulate_first_pass_convos.chatgpt_chatbot",
        return_value="dummy",
    )
    @patch("arklex.evaluation.simulate_first_pass_convos.generate_conversations")
    def test_simulate_conversations(
        self,
        mock_generate_conversations: Mock,
        mock_sim_chatgpt: Mock,
        mock_build_chatgpt: Mock,
    ) -> None:
        """Test simulate_conversations function."""
        # Setup
        model_api = "test_api"
        model_params = {"param1": "value1"}
        synthetic_data_params = {"num_convos": 2, "num_goals": 3}
        config = {
            "task": "first_pass",
            "output_dir": "test_dir",
            "intro": "Test intro",
            "workers": {},
            "tools": {},
            "client": Mock(),
            "documents_dir": "test_docs",
            "custom_profile": False,
            "system_inputs": False,
            "company_summary": "Test company summary",
            "rag_docs": [{"source": "https://example.com", "type": "url", "num": 1}],
            "user_attributes": {
                "goal": {"values": ["goal1", "goal2", "goal3"]},
                "system_attributes": {
                    "attr1": {"values": ["val1", "val2"]},
                    "attr2": {"values": ["val3", "val4"]},
                },
            },
        }

        mock_generate_conversations.return_value = [{"convo": "test"}]

        # Execute
        conversations, goals = simulate_conversations(
            model_api, model_params, synthetic_data_params, config
        )

        # Assert
        assert conversations == [{"convo": "test"}]
        mock_generate_conversations.assert_called_once()

    @patch("arklex.evaluation.simulate_first_pass_convos.conversation")
    @patch("arklex.evaluation.simulate_first_pass_convos.flip_hist")
    @patch("arklex.evaluation.simulate_first_pass_convos.filter_convo")
    def test_generate_conversations_with_thread_pool(
        self, mock_filter_convo: Mock, mock_flip_hist: Mock, mock_conversation: Mock
    ) -> None:
        """Test generate_conversations function with ThreadPoolExecutor logic."""
        # Setup
        model_api = "http://test.api"
        profiles = ["profile1", "profile2"]
        goals = ["goal1", "goal2"]
        attributes_list = [{"attr1": "val1"}, {"attr2": "val2"}]
        system_inputs = [{"sys1": "val1"}, {"sys2": "val2"}]
        summary = "Test summary"
        model_params = {"param1": "val1"}
        synthetic_data_params = {"max_turns": 5}
        env_config = {"client": Mock()}

        # Mock conversation return values - use a deterministic approach
        # Create a mock that returns different values based on the profile argument
        def mock_conversation_side_effect(
            model_api: str,
            profile: str,
            goal: str,
            attr: dict[str, str],
            sys_input: dict[str, str],
            summary: str,
            model_params: dict[str, str],
            synthetic_data_params: dict[str, int],
            env_config: dict[str, Mock],
        ) -> tuple[list[dict[str, str]], bool]:
            if profile == "profile1":
                return ([{"role": "user", "content": "test1"}], True)
            else:  # profile2
                return ([{"role": "user", "content": "test2"}], False)

        mock_conversation.side_effect = mock_conversation_side_effect

        # Mock filter_convo and flip_hist
        mock_filter_convo.return_value = [{"role": "user", "content": "filtered"}]
        mock_flip_hist.return_value = [{"role": "assistant", "content": "flipped"}]

        # Execute
        result = generate_conversations(
            model_api,
            profiles,
            goals,
            attributes_list,
            system_inputs,
            summary,
            model_params,
            synthetic_data_params,
            env_config,
        )

        # Assert
        assert len(result) == 2

        # Sort results by id to make the test deterministic
        result.sort(key=lambda x: x["id"])

        # Find the result with id=0 and id=1
        result_0 = next(r for r in result if r["id"] == 0)
        result_1 = next(r for r in result if r["id"] == 1)

        assert result_0["profile"] == "profile1"
        assert result_0["goal"] == "goal1"
        assert result_0["goal_completion"] is True
        assert result_1["profile"] == "profile2"
        assert result_1["goal"] == "goal2"
        assert result_1["goal_completion"] is False

        # Verify conversation was called for each input combination
        assert mock_conversation.call_count == 2
        assert mock_filter_convo.call_count == 2
        assert mock_flip_hist.call_count == 2

    @patch("arklex.evaluation.simulate_first_pass_convos.build_profile")
    @patch("arklex.evaluation.simulate_first_pass_convos.generate_conversations")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    @patch("json.dump")
    def test_simulate_conversations_first_pass_task(
        self,
        mock_json_dump: Mock,
        mock_makedirs: Mock,
        mock_file: Mock,
        mock_generate_conversations: Mock,
        mock_build_profile: Mock,
    ) -> None:
        """Test simulate_conversations function with first_pass task."""
        # Setup
        model_api = "http://test.api"
        model_params = {"param1": "val1"}
        synthetic_data_params = {"num_convos": 2}
        config = {
            "task": "first_pass",
            "output_dir": "/test/output",
            "intro": "Test intro",
            "workers": ["worker1"],
            "tools": ["tool1"],
            "client": Mock(),
        }

        # Mock build_profile return values
        mock_build_profile.return_value = (
            ["profile1", "profile2"],
            ["goal1", "goal2"],
            [{"attr1": "val1"}, {"attr2": "val2"}],
            [{"sys1": "val1"}, {"sys2": "val2"}],
            ["label1", "label2"],
        )

        # Mock generate_conversations return value
        mock_generate_conversations.return_value = [
            {"id": 0, "convo": [{"role": "user", "content": "test"}]},
            {"id": 1, "convo": [{"role": "assistant", "content": "response"}]},
        ]

        # Execute
        conversations, goals = simulate_conversations(
            model_api, model_params, synthetic_data_params, config
        )

        # Assert
        assert len(conversations) == 2
        assert goals == ["goal1", "goal2"]

        # Verify build_profile was called
        mock_build_profile.assert_called_once_with(synthetic_data_params, config)

        # Verify directory creation and file saving
        mock_makedirs.assert_called_once_with(
            "/test/output/simulate_data", exist_ok=True
        )
        assert mock_json_dump.call_count == 5  # 5 files saved

    @patch("arklex.evaluation.simulate_first_pass_convos.generate_conversations")
    @patch(
        "builtins.open", new_callable=mock_open, read_data='["profile1", "profile2"]'
    )
    @patch("json.load")
    def test_simulate_conversations_simulate_conv_only_task(
        self,
        mock_json_load: Mock,
        mock_file: Mock,
        mock_generate_conversations: Mock,
    ) -> None:
        """Test simulate_conversations with simulate_conv_only task."""
        # Setup
        model_api = "test_api"
        model_params = {"param1": "value1"}
        synthetic_data_params = {"num_convos": 2}
        config = {
            "task": "simulate_conv_only",
            "output_dir": "test_dir",
            "intro": "Test intro",
            "workers": {},
            "tools": {},
            "client": Mock(),
        }

        mock_generate_conversations.return_value = [{"convo": "test"}]
        mock_json_load.side_effect = [
            ["profile1", "profile2"],  # profiles
            ["goal1", "goal2"],  # goals
            [{"attr1": "value1"}],  # attributes_list
            [{"input1": "value1"}],  # system_inputs
            ["label1", "label2"],  # labels_list
        ]

        # Execute
        result_convos, result_goals = simulate_conversations(
            model_api, model_params, synthetic_data_params, config
        )

        # Assert
        assert result_convos == [{"convo": "test"}]
        assert result_goals == ["goal1", "goal2"]
        mock_generate_conversations.assert_called_once()

    @patch("arklex.evaluation.simulate_first_pass_convos.simulate_conversations")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_main_with_default_parameters(
        self, mock_json_dump: Mock, mock_file: Mock, mock_simulate_conversations: Mock
    ) -> None:
        """Test main function with default parameters."""
        # Import the main function
        from arklex.evaluation.simulate_first_pass_convos import main

        # Setup
        mock_simulate_conversations.return_value = ([{"convo": "test"}], ["goal1"])

        # Execute
        result = main()

        # Assert
        assert result == [{"convo": "test"}]
        mock_simulate_conversations.assert_called_once()
        mock_json_dump.assert_called_once()

    @patch("arklex.evaluation.simulate_first_pass_convos.simulate_conversations")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_main_with_custom_parameters(
        self, mock_json_dump: Mock, mock_file: Mock, mock_simulate_conversations: Mock
    ) -> None:
        """Test main function with custom parameters."""
        # Import the main function
        from arklex.evaluation.simulate_first_pass_convos import main

        # Setup
        model_api = "custom_api"
        synthetic_data_params = {"num_convos": 10, "max_turns": 15}
        model_params = {"custom_param": "value"}
        config = {
            "task": "first_pass",
            "output_dir": "custom_dir",
            "intro": "Custom intro",
            "workers": {"worker1": "config1"},
            "tools": {"tool1": "config1"},
            "client": Mock(),
        }
        output_file = "custom_output.json"

        mock_simulate_conversations.return_value = ([{"convo": "custom"}], ["goal1"])

        # Execute
        result = main(
            model_api=model_api,
            synthetic_data_params=synthetic_data_params,
            model_params=model_params,
            config=config,
            output_file=output_file,
        )

        # Assert
        assert result == [{"convo": "custom"}]
        mock_simulate_conversations.assert_called_once_with(
            model_api, model_params, synthetic_data_params, config
        )
        mock_json_dump.assert_called_once()

    @patch("arklex.evaluation.simulate_first_pass_convos.simulate_conversations")
    def test_main_without_output_file(self, mock_simulate_conversations: Mock) -> None:
        """Test main function without saving to file."""
        # Import the main function
        from arklex.evaluation.simulate_first_pass_convos import main

        # Setup
        mock_simulate_conversations.return_value = ([{"convo": "test"}], ["goal1"])

        # Execute
        result = main(output_file=None)

        # Assert
        assert result == [{"convo": "test"}]
        mock_simulate_conversations.assert_called_once()

    @patch("arklex.evaluation.simulate_first_pass_convos.simulate_conversations")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_main_with_none_parameters(
        self, mock_json_dump: Mock, mock_file: Mock, mock_simulate_conversations: Mock
    ) -> None:
        """Test main function with None parameters (should use defaults)."""
        # Import the main function
        from arklex.evaluation.simulate_first_pass_convos import main

        # Setup
        mock_simulate_conversations.return_value = ([{"convo": "test"}], ["goal1"])

        # Execute
        result = main(
            synthetic_data_params=None,
            model_params=None,
            config=None,
        )

        # Assert
        assert result == [{"convo": "test"}]
        mock_simulate_conversations.assert_called_once()
        # Verify default values were used
        call_args = mock_simulate_conversations.call_args
        assert (
            call_args[0][0]
            == "http://adaptation.cs.columbia.edu:55231/qa/richtech/v1alpha1"
        )
        assert call_args[0][1] == {}  # model_params
        assert call_args[0][2] == {
            "num_convos": 5,
            "num_goals": 3,
            "max_turns": 10,
        }  # synthetic_data_params
        assert call_args[0][3]["task"] == "first_pass"  # config
        mock_json_dump.assert_called_once()
