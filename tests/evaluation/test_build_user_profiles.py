"""Tests for the build_user_profiles module.

This module contains comprehensive test cases for user profile building functionality,
including profile generation, attribute conversion, and custom profile handling.
"""

from collections.abc import Generator
from typing import Any, NoReturn
from unittest.mock import Mock, patch

import pytest
import requests

from arklex.evaluation import build_user_profiles as bup
from arklex.evaluation.build_user_profiles import (
    augment_attributes,
    build_profile,
    convert_attributes_to_profile,
    get_custom_profiles,
    select_system_attributes,
)


class TestCoverageGaps:
    def test__select_random_attributes_all_defaults(self) -> None:
        # Each category triggers a different default
        categories = [
            ("budget", "medium"),
            ("location", "United States"),
            ("history", "none"),
            ("job", "professional"),
            ("business", "small business"),
            ("size", "10-50 employees"),
            ("other", "default"),
        ]
        attrs = {cat: [] for cat, _ in categories}
        goals = ["goal1"]
        result, _ = bup._select_random_attributes(attrs, goals)
        for cat, expected in categories:
            assert result[cat] == expected

    def test_augment_attributes_nested_and_flat(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = {"client": object(), "intro": "summary"}
        docs = [{"content": "doc content"}]

        # Nested structure with augment
        monkeypatch.setattr(bup, "chatgpt_chatbot", lambda prompt, client: "aug1, aug2")
        nested = {"parent": {"child": {"values": ["a"], "augment": True}}}
        out = bup.augment_attributes(nested, config, docs)
        assert out["child"] == ["a", "aug1", "aug2"]

        # Flat structure with augment, with docs
        monkeypatch.setattr(bup, "chatgpt_chatbot", lambda prompt, client: "aug1, aug2")
        flat = {"cat": {"values": ["b"], "augment": True}}
        out2 = bup.augment_attributes(flat, config, docs)
        assert out2["cat"] == ["b", "aug1", "aug2"]

        # Flat structure with augment, no docs - use a fresh mock
        monkeypatch.setattr(bup, "chatgpt_chatbot", lambda prompt, client: "aug1, aug2")
        fresh_flat = {"cat": {"values": ["b"], "augment": True}}  # Create fresh data
        out3 = bup.augment_attributes(fresh_flat, config, [])
        assert out3["cat"] == ["b", "aug1", "aug2"]

        # Nested structure with augment, no docs - use a fresh mock
        monkeypatch.setattr(bup, "chatgpt_chatbot", lambda prompt, client: "aug1, aug2")
        fresh_nested = {
            "parent": {"child": {"values": ["a"], "augment": True}}
        }  # Create fresh data
        out4 = bup.augment_attributes(fresh_nested, config, [])
        assert out4["child"] == ["a", "aug1", "aug2"]

        # Flat structure without augment
        flat_no_aug = {"cat": {"values": ["b"]}}
        out5 = bup.augment_attributes(flat_no_aug, config, docs)
        assert out5["cat"] == ["b"]

        # Nested structure without augment
        nested_no_aug = {"parent": {"child": {"values": ["a"]}}}
        out6 = bup.augment_attributes(nested_no_aug, config, docs)
        assert out6["child"] == ["a"]


@pytest.fixture
def mock_chatgpt_chatbot() -> Generator[Mock, None, None]:
    """Mock chatgpt_chatbot function."""
    with patch("arklex.evaluation.build_user_profiles.chatgpt_chatbot") as mock:
        mock.return_value = "mocked_response"
        yield mock


@pytest.fixture
def mock_requests_get() -> Generator[Mock, None, None]:
    """Mock requests.get function."""
    with patch("arklex.evaluation.build_user_profiles.requests.get") as mock:
        mock_response = Mock()
        mock_response.json.return_value = ["api_value1", "api_value2"]
        mock.return_value = mock_response
        yield mock


@pytest.fixture
def mock_environment() -> Generator[Mock, None, None]:
    """Mock Environment class."""
    with patch("arklex.evaluation.build_user_profiles.Environment") as mock:
        mock_instance = Mock()
        mock_tool = Mock()
        mock_tool.slots = [Mock(name="slot1", value="value1")]
        mock_tool.description = "Test tool description"
        mock_tool.output = "Test output"
        mock_tool.name = "Test Tool"
        mock_instance.tools = {"tool1": {"execute": Mock(return_value=mock_tool)}}
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_slot_filler() -> Generator[Mock, None, None]:
    """Mock SlotFiller class."""
    with patch("arklex.evaluation.build_user_profiles.SlotFiller") as mock:
        mock_instance = Mock()
        mock_instance.execute.return_value = [Mock(name="slot1", value="filled_value")]
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_find_matched_attribute() -> Generator[Mock, None, None]:
    """Mock find_matched_attribute function."""
    with patch("arklex.evaluation.build_user_profiles.find_matched_attribute") as mock:
        mock.return_value = {"matched": "data"}
        yield mock


@pytest.fixture
def mock_load_docs() -> Generator[Mock, None, None]:
    """Mock load_docs function."""
    with patch("arklex.evaluation.build_user_profiles.load_docs") as mock:
        mock.return_value = [{"content": "test content"}]
        yield mock


@pytest.fixture
def mock_filter_attributes() -> Generator[Mock, None, None]:
    """Mock filter_attributes function."""
    with patch("arklex.evaluation.build_user_profiles.filter_attributes") as mock:
        mock.return_value = {
            "attr1": {"values": ["val1"]},
            "attr2": {"values": ["val3"]},
        }
        yield mock


@pytest.fixture
def mock_augment_attributes() -> Generator[Mock, None, None]:
    """Mock augment_attributes function."""
    with patch("arklex.evaluation.build_user_profiles.augment_attributes") as mock:
        mock.return_value = {
            "attr1": ["val1", "val2"],
            "attr2": ["val3", "val4"],
        }
        yield mock


@pytest.fixture
def mock_pick_attributes() -> Generator[Mock, None, None]:
    """Mock pick_attributes function."""
    with patch("arklex.evaluation.build_user_profiles.pick_attributes") as mock:
        mock.return_value = (
            {"goal": "test_goal", "attr1": "val1", "attr2": "val3"},
            {"matched": "data"},
        )
        yield mock


@pytest.fixture
def mock_adapt_goal() -> Generator[Mock, None, None]:
    """Mock adapt_goal function."""
    with patch("arklex.evaluation.build_user_profiles.adapt_goal") as mock:
        mock.return_value = "adapted_goal"
        yield mock


@pytest.fixture
def mock_select_system_attributes() -> Generator[Mock, None, None]:
    """Mock select_system_attributes function."""
    with patch(
        "arklex.evaluation.build_user_profiles.select_system_attributes"
    ) as mock:
        mock.return_value = [{"sys_attr": "val"}, {"sys_attr": "val2"}]
        yield mock


@pytest.fixture
def mock_get_custom_profiles() -> Generator[Mock, None, None]:
    """Mock get_custom_profiles function."""
    with patch("arklex.evaluation.build_user_profiles.get_custom_profiles") as mock:
        mock.return_value = (
            {"profile1": ["prof1", "prof2"], "profile2": ["prof3", "prof4"]},
            {"attr1": ["val1", "val2"], "attr2": ["val3", "val4"]},
        )
        yield mock


@pytest.fixture
def mock_convert_attributes_to_profiles() -> Generator[Mock, None, None]:
    """Mock convert_attributes_to_profiles function."""
    with patch(
        "arklex.evaluation.build_user_profiles.convert_attributes_to_profiles"
    ) as mock:
        mock.return_value = (
            ["profile1", "profile1"],
            ["goal1", "goal1"],
            [{"sys": "input"}, {"sys": "input"}],
        )
        yield mock


@pytest.fixture
def always_valid_mock_model() -> Generator[Mock, None, None]:
    """Fixture that provides a mock model that always returns valid responses."""
    with patch("arklex.evaluation.build_user_profiles.chatgpt_chatbot") as mock:
        mock.return_value = "mocked_response"
        yield mock


@pytest.fixture
def patched_sample_config() -> Generator[None, None, None]:
    """Fixture that provides a patched sample configuration for testing."""
    with (
        patch("arklex.evaluation.build_user_profiles.load_docs") as mock_load_docs,
        patch("arklex.evaluation.build_user_profiles.filter_attributes") as mock_filter,
        patch(
            "arklex.evaluation.build_user_profiles.augment_attributes"
        ) as mock_augment,
        patch("arklex.evaluation.build_user_profiles.pick_attributes") as mock_pick,
        patch("arklex.evaluation.build_user_profiles.adapt_goal") as mock_adapt,
        patch(
            "arklex.evaluation.build_user_profiles.select_system_attributes"
        ) as mock_select,
        patch(
            "arklex.evaluation.build_user_profiles.get_custom_profiles"
        ) as mock_custom,
        patch(
            "arklex.evaluation.build_user_profiles.convert_attributes_to_profiles"
        ) as mock_convert,
    ):
        mock_load_docs.return_value = [{"content": "test content"}]
        mock_filter.return_value = {
            "attr1": {"values": ["val1"], "generate_values": False},
        }
        mock_augment.return_value = {
            "attr1": ["val1"],
        }
        mock_pick.return_value = (
            {"goal": "test_goal", "attr1": "val1"},
            {"matched": "data"},
        )
        mock_adapt.return_value = "adapted_goal"
        mock_select.return_value = [{"sys_attr": "val"}, {"sys_attr": "val2"}]
        mock_custom.return_value = (
            {"profile1": ["prof1", "prof2"]},
            {"attr1": ["val1", "val2"]},
        )
        mock_convert.return_value = (
            ["profile1", "profile1"],
            ["goal1", "goal1"],
            [{"sys": "input"}, {"sys": "input"}],
        )
        yield


class TestBuildUserProfiles:
    """Test cases for build_user_profiles module.

    This class contains comprehensive tests for user profile building,
    including predefined and custom profile generation, attribute handling,
    and profile conversion functionality.
    """

    @pytest.fixture
    def mock_config(self) -> dict[str, Any]:
        """Create a mock configuration for testing."""
        return {
            "documents_dir": "/test/documents",
            "custom_profile": False,
            "system_inputs": True,
            "company_summary": "Test company summary",
            "client": Mock(),
            "user_attributes": {
                "goal": {"values": ["goal1", "goal2", "goal3"]},
                "system_attributes": {
                    "attr1": {"values": ["val1", "val2"]},
                    "attr2": {"values": ["val3", "val4"]},
                },
                "user_profiles": {"profile1": {"values": ["p1"]}},
            },
            "tools": [{"id": "tool1", "name": "Test Tool", "path": "mock_path"}],
            "workers": [{"id": "worker1", "name": "Test Worker"}],
        }

    @pytest.fixture
    def mock_synthetic_params(self) -> dict[str, int]:
        """Create mock synthetic data parameters."""
        return {"num_convos": 2, "num_goals": 3}

    @pytest.fixture
    def basic_mock_setup(
        self,
        mock_load_docs: Mock,
        mock_filter_attributes: Mock,
        mock_augment_attributes: Mock,
        mock_pick_attributes: Mock,
        mock_adapt_goal: Mock,
        mock_select_system_attributes: Mock,
        mock_chatgpt_chatbot: Mock,
    ) -> dict[str, Mock]:
        """Create basic mock setup for build_profile tests."""
        return {
            "load_docs": mock_load_docs,
            "filter_attributes": mock_filter_attributes,
            "augment_attributes": mock_augment_attributes,
            "pick_attributes": mock_pick_attributes,
            "adapt_goal": mock_adapt_goal,
            "select_system_attributes": mock_select_system_attributes,
            "chatgpt_chatbot": mock_chatgpt_chatbot,
        }

    @pytest.fixture
    def custom_profile_mock_setup(
        self,
        mock_load_docs: Mock,
        mock_filter_attributes: Mock,
        mock_augment_attributes: Mock,
        mock_get_custom_profiles: Mock,
        mock_pick_attributes: Mock,
        mock_adapt_goal: Mock,
        mock_chatgpt_chatbot: Mock,
    ) -> dict[str, Mock]:
        """Create mock setup for custom profile tests."""
        return {
            "load_docs": mock_load_docs,
            "filter_attributes": mock_filter_attributes,
            "augment_attributes": mock_augment_attributes,
            "get_custom_profiles": mock_get_custom_profiles,
            "pick_attributes": mock_pick_attributes,
            "adapt_goal": mock_adapt_goal,
            "chatgpt_chatbot": mock_chatgpt_chatbot,
        }

    @pytest.mark.parametrize(
        "test_config,test_name",
        [
            ({"system_inputs": False}, "system_inputs_false"),
            (
                {
                    "custom_profile": True,
                    "user_attributes": {
                        "goal": {"values": ["goal1", "goal2"]},
                        "user_profiles": {"profile1": {"values": ["prof1", "prof2"]}},
                        "system_attributes": {"attr1": {"values": ["val1", "val2"]}},
                    },
                },
                "custom_with_binding",
            ),
            (
                {
                    "custom_profile": True,
                    "user_attributes": {
                        "goal": {"values": ["goal1", "goal2"]},
                        "user_profiles": {"profile1": {"values": ["prof1", "prof2"]}},
                        "system_attributes": {"attr1": {"values": ["val1", "val2"]}},
                    },
                },
                "custom_without_documents",
            ),
        ],
    )
    def test_build_profile_variations(
        self,
        patched_sample_config: None,
        mock_config: dict[str, Any],
        mock_synthetic_params: dict[str, int],
        test_config: dict[str, Any],
        test_name: str,
    ) -> None:
        """Test build_profile with various configurations."""
        mock_config.update(test_config)

        profiles, goals, attributes, system_attrs, labels = build_profile(
            mock_synthetic_params, mock_config
        )

        assert len(profiles) == 2
        assert len(goals) == 2
        assert len(attributes) == 2
        assert len(system_attrs) == 2
        assert len(labels) == 2

    def test_convert_attributes_to_profile(
        self, mock_chatgpt_chatbot: Mock, mock_config: dict[str, Any]
    ) -> None:
        """Test convert_attributes_to_profile function."""
        attributes = {"attr1": "value1", "attr2": "value2"}
        result = convert_attributes_to_profile(attributes, mock_config)
        assert result == "mocked_response"
        mock_chatgpt_chatbot.assert_called_once()

    def test_get_custom_profiles(self, mock_config: dict[str, Any]) -> None:
        """Test get_custom_profiles function with API endpoints."""
        mock_config["user_attributes"]["system_attributes"] = {
            "attr1": {"api": "http://test.com/api1"}
        }
        mock_config["user_attributes"]["user_profiles"] = {
            "profile1": {"api": "http://test.com/api2"}
        }

        with patch(
            "arklex.evaluation.build_user_profiles.requests.get"
        ) as mock_requests_get:
            mock_response1 = Mock()
            mock_response1.json.return_value = ["api_value1", "api_value2"]
            mock_response2 = Mock()
            mock_response2.json.return_value = ["profile_value1", "profile_value2"]
            mock_requests_get.side_effect = [mock_response1, mock_response2]

            user_profiles, system_attributes = get_custom_profiles(mock_config)

            assert isinstance(user_profiles, dict)
            assert isinstance(system_attributes, dict)
            assert "profile1" in user_profiles
            assert "attr1" in system_attributes

    def test_get_custom_profiles_without_required_structure(self) -> None:
        """Test get_custom_profiles function when required structure is not present."""
        config = {"user_attributes": {}}

        user_profiles, system_attributes = get_custom_profiles(config)

        assert isinstance(user_profiles, dict)
        assert isinstance(system_attributes, dict)
        assert len(user_profiles) == 0
        assert len(system_attributes) == 0

    def test_build_profile_with_system_inputs_false_and_custom_profile(
        self,
        patched_sample_config: None,
        mock_config: dict[str, Any],
        mock_synthetic_params: dict[str, int],
    ) -> None:
        """Test build_profile function when system_inputs is False and custom_profile is True."""
        config = mock_config.copy()
        config["custom_profile"] = True
        config["system_inputs"] = False
        config["user_attributes"] = {
            "goal": {"values": ["goal1", "goal2"]},
            "user_profiles": {"profile1": {"values": ["prof1", "prof2"]}},
            "system_attributes": {"attr1": {"values": ["val1", "val2"]}},
        }

        profiles, goals, attributes_list, system_inputs, labels_list = build_profile(
            mock_synthetic_params, config
        )

        assert len(profiles) == 2
        assert len(goals) == 2
        assert len(attributes_list) == 2
        assert len(system_inputs) == 2
        assert len(labels_list) == 2

    def test_build_profile_with_binding_index_mismatch(
        self,
        patched_sample_config: None,
        mock_config: dict[str, Any],
        mock_synthetic_params: dict[str, int],
    ) -> None:
        """Test build_profile function when binding_index doesn't contain the expected key."""
        config = mock_config.copy()
        config["custom_profile"] = True
        config["user_attributes"] = {
            "goal": {"values": ["goal1", "goal2"]},
            "user_profiles": {
                "profile1": {"values": ["prof1", "prof2"], "bind_to": "system_attr1"},
            },
            "system_attributes": {
                "attr1": {"values": ["val1", "val2"]},  # No bind_to specified
            },
        }

        profiles, goals, attributes_list, system_inputs, labels_list = build_profile(
            mock_synthetic_params, config
        )

        assert len(profiles) == 2
        assert len(goals) == 2
        assert len(attributes_list) == 2
        assert len(system_inputs) == 2
        assert len(labels_list) == 2

    def test_build_profile_with_commented_get_label(
        self,
        patched_sample_config: None,
        mock_config: dict[str, Any],
        mock_synthetic_params: dict[str, int],
    ) -> None:
        """Test build_profile function with the commented get_label section."""
        config = mock_config.copy()
        config["custom_profile"] = True
        config["user_attributes"] = {
            "goal": {"values": ["goal1", "goal2"]},
            "user_profiles": {"profile1": {"values": ["prof1", "prof2"]}},
            "system_attributes": {"attr1": {"values": ["val1", "val2"]}},
        }

        profiles, goals, attributes_list, system_inputs, labels_list = build_profile(
            mock_synthetic_params, config
        )

        assert len(profiles) == 2
        assert len(goals) == 2
        assert len(attributes_list) == 2
        assert len(system_inputs) == 2
        assert len(labels_list) == 2

    def test_build_profile_with_empty_documents_in_custom_mode(
        self,
        patched_sample_config: None,
        mock_config: dict[str, Any],
        mock_synthetic_params: dict[str, int],
    ) -> None:
        """Test build_profile function with empty documents in custom profile mode."""
        config = mock_config.copy()
        config["custom_profile"] = True
        config["user_attributes"] = {
            "goal": {"values": ["goal1", "goal2"]},
            "user_profiles": {"profile1": {"values": ["prof1", "prof2"]}},
            "system_attributes": {"attr1": {"values": ["val1", "val2"]}},
        }

        # Override load_docs to return empty list using the patched function
        with patch("arklex.evaluation.build_user_profiles.load_docs") as mock_load_docs:
            mock_load_docs.return_value = []

            profiles, goals, attributes_list, system_inputs, labels_list = (
                build_profile(mock_synthetic_params, config)
            )

            assert len(profiles) == 2
            assert len(goals) == 2
            assert len(attributes_list) == 2
            assert len(system_inputs) == 2
            assert len(labels_list) == 2

    def test_get_custom_profiles_with_api_and_exception(self) -> None:
        """Test get_custom_profiles function when API call raises an exception."""
        config = {
            "user_attributes": {
                "system_attributes": {"attr1": {"api": "http://test.com/api1"}},
                "user_profiles": {"profile1": {"api": "http://test.com/api2"}},
            }
        }

        with patch(
            "arklex.evaluation.build_user_profiles.requests.get"
        ) as mock_requests_get:
            mock_response1 = Mock()
            mock_response1.json.return_value = ["api_value1", "api_value2"]
            mock_requests_get.side_effect = [mock_response1, Exception("API Error")]

            with pytest.raises(Exception, match="API Error"):
                get_custom_profiles(config)

    def test_get_custom_profiles_with_api_and_invalid_response(self) -> None:
        """Test get_custom_profiles function when API returns invalid response."""
        config = {
            "user_attributes": {
                "system_attributes": {"attr1": {"api": "http://test.com/api1"}},
                "user_profiles": {"profile1": {"api": "http://test.com/api2"}},
            }
        }

        with patch(
            "arklex.evaluation.build_user_profiles.requests.get"
        ) as mock_requests_get:
            mock_response1 = Mock()
            mock_response1.json.side_effect = ValueError("Invalid JSON")
            mock_response2 = Mock()
            mock_response2.json.return_value = ["api_value1", "api_value2"]
            mock_requests_get.side_effect = [mock_response1, mock_response2]

            with pytest.raises(ValueError, match="Invalid JSON"):
                get_custom_profiles(config)

    def test_select_system_attributes_with_non_dict_values(
        self, mock_synthetic_params: dict[str, int]
    ) -> None:
        """Test select_system_attributes function when system attributes are not dictionaries."""
        config = {
            "user_attributes": {
                "system_attributes": {
                    "attr1": ["val1", "val2"],  # Not a dict
                }
            }
        }

        with pytest.raises(
            TypeError, match="list indices must be integers or slices, not str"
        ):
            select_system_attributes(config, mock_synthetic_params)

    def test_select_system_attributes_with_empty_list(
        self, mock_synthetic_params: dict[str, int]
    ) -> None:
        """Test select_system_attributes function when system attributes list is empty."""
        config = {
            "user_attributes": {
                "system_attributes": {
                    "attr1": [],  # Empty list
                }
            }
        }

        with pytest.raises(
            TypeError, match="list indices must be integers or slices, not str"
        ):
            select_system_attributes(config, mock_synthetic_params)

    def test_augment_attributes_with_empty_values(
        self, mock_config: dict[str, Any]
    ) -> None:
        """Test augment_attributes function when some attributes have empty values."""
        attributes = {
            "attr1": {"values": [], "augment": True},  # Empty values
            "attr2": {"values": ["val3", "val4"], "augment": True},
        }
        documents = [{"content": "test content"}]

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatgpt_chatbot:
            mock_chatgpt_chatbot.return_value = "new_val1, new_val2"

            result = augment_attributes(attributes, mock_config, documents)

            assert (
                "attr1" not in result["attr2"]
            )  # attr1 should not appear in text_attribute
            assert result["attr2"] == ["val3", "val4", "new_val1", "new_val2"]

    def test_augment_attributes_with_mixed_generate_values_and_documents(
        self, mock_config: dict[str, Any]
    ) -> None:
        """Test augment_attributes function with mixed generate_values and documents."""
        attributes = {
            "attr1": {"values": ["val1"], "augment": False},
            "attr2": {"values": ["val2"], "augment": True},
        }
        documents = [{"content": "test content"}]

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatgpt_chatbot:
            mock_chatgpt_chatbot.return_value = "new_val1, new_val2"

            result = augment_attributes(attributes, mock_config, documents)

            assert result["attr1"] == ["val1"]  # Should use original values for attr1
            assert result["attr2"] == [
                "val2",
                "new_val1",
                "new_val2",
            ]  # Should generate new values for attr2
            assert (
                mock_chatgpt_chatbot.call_count == 1
            )  # Should call chatgpt_chatbot only once for attr2

    def test_augment_attributes_with_mixed_generate_values_and_no_documents(
        self, mock_config: dict[str, Any]
    ) -> None:
        """Test augment_attributes function with mixed generate_values and no documents."""
        attributes = {
            "attr1": {"values": ["val1"], "augment": False},
            "attr2": {"values": ["val2"], "augment": True},
        }
        documents = []  # No documents

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatgpt_chatbot:
            mock_chatgpt_chatbot.return_value = "new_val1, new_val2"

            result = augment_attributes(attributes, mock_config, documents)

            assert result["attr1"] == ["val1"]  # Should use original values for attr1
            assert result["attr2"] == [
                "val2",
                "new_val1",
                "new_val2",
            ]  # Should generate new values for attr2 without documents
            assert (
                mock_chatgpt_chatbot.call_count == 1
            )  # Should call chatgpt_chatbot only once for attr2

    def test_select_random_from_list_with_empty_list(self) -> None:
        """Test _select_random_from_list function with empty list."""
        from arklex.evaluation.build_user_profiles import _select_random_from_list

        result, index = _select_random_from_list([], "test_key")
        assert result is None
        assert index == -1

    def test_select_random_from_list_with_non_empty_list(self) -> None:
        """Test _select_random_from_list function with non-empty list."""
        from arklex.evaluation.build_user_profiles import _select_random_from_list

        test_list = ["item1", "item2", "item3"]
        result, index = _select_random_from_list(test_list, "test_key")
        assert result in test_list
        assert 0 <= index < len(test_list)

    def test_process_system_attributes_with_binding(self) -> None:
        """Test _process_system_attributes function with binding logic."""
        from arklex.evaluation.build_user_profiles import _process_system_attributes

        config = {
            "user_attributes": {
                "system_attributes": {
                    "attr1": {"bind_to": "bound_attr"},
                    "attr2": {},  # No binding
                }
            }
        }
        system_attributes = {
            "attr1": ["val1", "val2", "val3"],
            "attr2": ["val4", "val5"],
        }

        result, binding_index = _process_system_attributes(config, system_attributes)

        assert "attr1" in result
        assert "attr2" in result
        assert result["attr1"] in system_attributes["attr1"]
        assert result["attr2"] in system_attributes["attr2"]
        assert "bound_attr" in binding_index
        assert binding_index["bound_attr"] >= 0

    def test_process_user_profiles_with_binding(self) -> None:
        """Test _process_user_profiles function with binding logic."""
        from arklex.evaluation.build_user_profiles import _process_user_profiles

        config = {
            "user_attributes": {
                "user_profiles": {
                    "profile1": {"bind_to": "bound_attr"},
                    "profile2": {},  # No binding
                }
            }
        }
        user_profiles = {
            "profile1": ["prof1", "prof2", "prof3"],
            "profile2": ["prof4", "prof5"],
        }
        binding_index = {"bound_attr": 1}  # Bind to index 1

        result = _process_user_profiles(config, user_profiles, binding_index)

        assert "profile1" in result
        assert "profile2" in result
        assert result["profile1"] == "prof2"  # Should use binding index
        assert result["profile2"] in user_profiles["profile2"]

    def test_process_user_profiles_with_empty_binding_list(self) -> None:
        """Test _process_user_profiles function with empty binding list."""
        from arklex.evaluation.build_user_profiles import _process_user_profiles

        config = {
            "user_attributes": {
                "user_profiles": {
                    "profile1": {"bind_to": "bound_attr"},
                }
            }
        }
        user_profiles = {
            "profile1": [],  # Empty list
        }
        binding_index = {"bound_attr": 1}

        result = _process_user_profiles(config, user_profiles, binding_index)

        assert "profile1" in result
        assert result["profile1"] is None

    def test_pick_goal_with_llm_based_strategy(
        self, mock_config: dict[str, Any]
    ) -> None:
        """Test pick_goal function with llm_based strategy."""
        from arklex.evaluation.build_user_profiles import pick_goal

        attributes = {"attr1": "value1", "attr2": "value2"}
        goals = ["goal1", "goal2", "goal3"]

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatgpt_chatbot:
            mock_chatgpt_chatbot.return_value = "Goal: goal2"

            result = pick_goal(attributes, goals, strategy="llm_based", client=Mock())

            assert result == "goal2"
            mock_chatgpt_chatbot.assert_called_once()

    def test_pick_goal_with_react_strategy(self, mock_config: dict[str, Any]) -> None:
        """Test pick_goal function with react strategy."""
        from arklex.evaluation.build_user_profiles import pick_goal

        attributes = {"attr1": "value1", "attr2": "value2"}
        goals = ["goal1", "goal2", "goal3"]

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatgpt_chatbot:
            mock_chatgpt_chatbot.return_value = (
                "Thought: This is the reasoning\nGoal: goal3"
            )

            result = pick_goal(attributes, goals, strategy="react", client=Mock())

            assert result == "goal3"
            mock_chatgpt_chatbot.assert_called_once()

    def test_pick_goal_with_invalid_strategy(self) -> None:
        """Test pick_goal function with invalid strategy."""
        from arklex.evaluation.build_user_profiles import pick_goal

        attributes = {"attr1": "value1"}
        goals = ["goal1", "goal2"]

        with pytest.raises(ValueError, match="Invalid strategy"):
            pick_goal(attributes, goals, strategy="invalid_strategy")

    def test_find_matched_attribute_with_dict_input(self) -> None:
        """Test find_matched_attribute function with dictionary input."""
        from arklex.evaluation.build_user_profiles import find_matched_attribute

        goal = "test_goal"
        attributes = {"attr1": "value1", "attr2": "value2"}

        result = find_matched_attribute(goal, attributes, strategy="react")

        assert isinstance(result, dict)
        assert result["goal"] == goal
        assert result["matched_attribute"] == attributes

    def test_find_matched_attribute_with_string_input(
        self, mock_config: dict[str, Any]
    ) -> None:
        """Test find_matched_attribute function with string input."""
        from arklex.evaluation.build_user_profiles import find_matched_attribute

        goal = "test_goal"
        user_profile_str = "user_info: test_user; current_webpage: test_page"

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatgpt_chatbot:
            mock_chatgpt_chatbot.return_value = (
                "Thought: This is reasoning\nAttribute: test_attribute"
            )

            result = find_matched_attribute(
                goal, user_profile_str, strategy="react", client=Mock()
            )

            assert result == "test_attribute"
            mock_chatgpt_chatbot.assert_called_once()

    def test_find_matched_attribute_with_invalid_strategy(self) -> None:
        """Test find_matched_attribute function with invalid strategy."""
        from arklex.evaluation.build_user_profiles import find_matched_attribute

        goal = "test_goal"
        user_profile_str = "test_profile"

        with pytest.raises(ValueError, match="Invalid strategy"):
            find_matched_attribute(goal, user_profile_str, strategy="invalid_strategy")

    def test_pick_attributes_react(self, mock_config: dict[str, Any]) -> None:
        """Test pick_attributes_react function."""
        from arklex.evaluation.build_user_profiles import pick_attributes_react

        user_profile = {"profile_attr": "profile_value"}
        attributes = {"attr1": ["val1", "val2"], "attr2": ["val3", "val4"]}
        goals = ["goal1", "goal2"]

        with (
            patch(
                "arklex.evaluation.build_user_profiles._select_random_attributes"
            ) as mock_select,
            patch(
                "arklex.evaluation.build_user_profiles.find_matched_attribute"
            ) as mock_find,
        ):
            mock_select.return_value = ({"goal": "goal1", "attr1": "val1"}, "goal1")
            mock_find.return_value = {
                "goal": "goal1",
                "matched_attribute": "matched_value",
            }

            result_attrs, result_label = pick_attributes_react(
                user_profile, attributes, goals, Mock()
            )

            assert "goal" in result_attrs
            assert result_attrs["goal"] == "goal1"
            assert "attr1" in result_attrs
            assert result_attrs["attr1"] == "val1"
            assert result_label["goal"] == "goal1"
            assert result_label["matched_attribute"] == "matched_value"

    def test_pick_attributes_random(self) -> None:
        """Test pick_attributes_random function."""
        from arklex.evaluation.build_user_profiles import pick_attributes_random

        user_profile = {"profile_attr": "profile_value"}
        attributes = {"attr1": ["val1", "val2"], "attr2": ["val3", "val4"]}
        goals = ["goal1", "goal2"]

        with patch(
            "arklex.evaluation.build_user_profiles._select_random_attributes"
        ) as mock_select:
            mock_select.return_value = ({"goal": "goal2", "attr1": "val2"}, "goal2")

            result_attrs, result_label = pick_attributes_random(
                user_profile, attributes, goals
            )

            assert "goal" in result_attrs
            assert result_attrs["goal"] == "goal2"
            assert "attr1" in result_attrs
            assert result_attrs["attr1"] == "val2"
            assert result_label["goal"] == "goal2"
            assert result_label["matched_attribute"] == {
                "goal": "goal2",
                "attr1": "val2",
            }

    def test_adapt_goal(self, mock_config: dict[str, Any]) -> None:
        """Test adapt_goal function."""
        from arklex.evaluation.build_user_profiles import adapt_goal

        goal = "original_goal"
        doc = "test document content"
        user_profile = "test user profile"

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatgpt_chatbot:
            mock_chatgpt_chatbot.return_value = "adapted_goal"

            result = adapt_goal(goal, mock_config, doc, user_profile)

            assert result == "adapted_goal"
            mock_chatgpt_chatbot.assert_called_once()

    def test_fetch_api_data_with_request_exception(self) -> None:
        """Test _fetch_api_data function with request exception."""
        from arklex.evaluation.build_user_profiles import _fetch_api_data

        with patch("arklex.evaluation.build_user_profiles.requests.get") as mock_get:
            mock_get.side_effect = Exception("Network error")

            with pytest.raises(Exception, match="Network error"):
                _fetch_api_data("http://test.com/api", "test_key")

    def test_augment_attributes_with_llm_augmentation(
        self, mock_config: dict[str, Any]
    ) -> None:
        """Test augment_attributes function with LLM augmentation."""
        from arklex.evaluation.build_user_profiles import augment_attributes

        attributes = {
            "attr1": {"values": ["val1"], "augment": True},
            "attr2": {"values": ["val2"], "augment": False},
        }
        documents = [{"content": "test document content"}]

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatgpt_chatbot:
            mock_chatgpt_chatbot.return_value = "new_val1, new_val2, new_val3"

            result = augment_attributes(attributes, mock_config, documents)

            assert "attr1" in result
            assert "attr2" in result
            assert result["attr1"] == ["val1", "new_val1", "new_val2", "new_val3"]
            assert result["attr2"] == ["val2"]  # Should not be augmented
            assert mock_chatgpt_chatbot.call_count == 1

    def test_augment_attributes_without_documents(
        self, mock_config: dict[str, Any]
    ) -> None:
        """Test augment_attributes function without documents."""
        from arklex.evaluation.build_user_profiles import augment_attributes

        attributes = {
            "attr1": {"values": ["val1"], "augment": True},
        }
        documents = []

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatgpt_chatbot:
            mock_chatgpt_chatbot.return_value = "new_val1, new_val2"

            result = augment_attributes(attributes, mock_config, documents)

            assert "attr1" in result
            assert result["attr1"] == ["val1", "new_val1", "new_val2"]
            mock_chatgpt_chatbot.assert_called_once()

    def test_get_label_successful_tool_selection(
        self, mock_config: dict[str, Any]
    ) -> None:
        """Test get_label function with successful tool selection."""
        from arklex.evaluation.build_user_profiles import get_label

        attribute = {"goal": "test_goal", "attr1": "value1"}

        with (
            patch(
                "arklex.evaluation.build_user_profiles._build_tool_list"
            ) as mock_build_tools,
            patch(
                "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
            ) as mock_chatgpt_chatbot,
            patch(
                "arklex.evaluation.build_user_profiles.SlotFiller"
            ) as mock_slot_filler,
        ):
            mock_tool_list = [
                {
                    "tool_id": "tool1",
                    "tool_description": "Test tool",
                    "tool_input": [],
                    "tool_output": "Test output",
                }
            ]
            mock_env = Mock()
            mock_tool = Mock()
            mock_tool.name = "Test Tool"
            mock_tool.slots = []
            mock_env.tools = {"tool1": {"execute": Mock(return_value=mock_tool)}}
            mock_build_tools.return_value = (mock_tool_list, mock_env)

            mock_chatgpt_chatbot.return_value = "tool1"

            mock_slots = [Mock(name="slot1", value="value1")]
            mock_slot_filler_instance = Mock()
            mock_slot_filler_instance.execute.return_value = mock_slots
            mock_slot_filler.return_value = mock_slot_filler_instance

            result, success = get_label(attribute, mock_config)

            assert success is True
            assert len(result) == 1
            assert result[0]["tool_id"] == "tool1"
            assert result[0]["tool_name"] == "Test Tool"

    def test_get_label_with_tool_id_zero(self, mock_config: dict[str, Any]) -> None:
        """Test get_label function when tool_id is 0."""
        from arklex.evaluation.build_user_profiles import get_label

        attribute = {"goal": "test_goal", "attr1": "value1"}

        with (
            patch(
                "arklex.evaluation.build_user_profiles._build_tool_list"
            ) as mock_build_tools,
            patch(
                "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
            ) as mock_chatgpt_chatbot,
        ):
            mock_tool_list = [
                {
                    "tool_id": "0",
                    "tool_description": "No tool",
                    "tool_input": [],
                    "tool_output": "No tool",
                }
            ]
            mock_env = Mock()
            mock_build_tools.return_value = (mock_tool_list, mock_env)
            mock_chatgpt_chatbot.return_value = "0"

            result, success = get_label(attribute, mock_config)

            assert success is True
            assert len(result) == 1
            assert result[0]["tool_id"] == "0"
            assert result[0]["tool_name"] == "No tool"

    def test_get_label_with_retry_logic(self, mock_config: dict[str, Any]) -> None:
        """Test get_label function with retry logic."""
        from arklex.evaluation.build_user_profiles import get_label

        attribute = {"goal": "test_goal", "attr1": "value1"}

        with (
            patch(
                "arklex.evaluation.build_user_profiles._build_tool_list"
            ) as mock_build_tools,
            patch(
                "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
            ) as mock_chatgpt_chatbot,
            patch(
                "arklex.evaluation.build_user_profiles.SlotFiller"
            ) as mock_slot_filler,
        ):
            mock_tool_list = [
                {
                    "tool_id": "tool1",
                    "tool_description": "Test tool",
                    "tool_input": [],
                    "tool_output": "Test output",
                }
            ]
            mock_env = Mock()
            mock_tool = Mock()
            mock_tool.name = "Test Tool"
            mock_tool.slots = []
            mock_env.tools = {"tool1": {"execute": Mock(return_value=mock_tool)}}
            mock_build_tools.return_value = (mock_tool_list, mock_env)

            # First call raises KeyError, second call succeeds
            mock_chatgpt_chatbot.side_effect = [KeyError("Tool not found"), "tool1"]

            mock_slots = [Mock(name="slot1", value="value1")]
            mock_slot_filler_instance = Mock()
            mock_slot_filler_instance.execute.return_value = mock_slots
            mock_slot_filler.return_value = mock_slot_filler_instance

            result, success = get_label(attribute, mock_config)

            assert success is True
            assert mock_chatgpt_chatbot.call_count == 2

    def test_get_label_with_max_retries_exceeded(
        self, mock_config: dict[str, Any]
    ) -> None:
        """Test get_label function when max retries are exceeded."""
        from arklex.evaluation.build_user_profiles import get_label

        attribute = {"goal": "test_goal", "attr1": "value1"}

        with (
            patch(
                "arklex.evaluation.build_user_profiles._build_tool_list"
            ) as mock_build_tools,
            patch(
                "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
            ) as mock_chatgpt_chatbot,
        ):
            mock_tool_list = [
                {
                    "tool_id": "tool1",
                    "tool_description": "Test tool",
                    "tool_input": [],
                    "tool_output": "Test output",
                }
            ]
            mock_env = Mock()
            mock_build_tools.return_value = (mock_tool_list, mock_env)

            # All calls raise exceptions
            mock_chatgpt_chatbot.side_effect = Exception("Network error")

            result, success = get_label(attribute, mock_config)

            assert success is True
            assert len(result) == 1
            assert result[0]["tool_id"] == "0"  # Should fall back to no tool
            assert mock_chatgpt_chatbot.call_count == 3

    def test_build_user_profiles(self) -> None:
        """Test build_user_profiles function returns empty list."""
        from arklex.evaluation.build_user_profiles import build_user_profiles

        test_data = [{"test": "data"}]
        result = build_user_profiles(test_data)

        assert result == []

    def test_build_user_profiles_with_empty_input(self) -> None:
        """Test build_user_profiles function with empty input."""
        from arklex.evaluation.build_user_profiles import build_user_profiles

        test_data = []
        result = build_user_profiles(test_data)

        assert result == []

    def test_build_user_profiles_with_none_input(self) -> None:
        """Test build_user_profiles function with None input."""
        from arklex.evaluation.build_user_profiles import build_user_profiles

        test_data = None
        result = build_user_profiles(test_data)

        assert result == []

    def test_attributes_to_text(self) -> None:
        """Test attributes_to_text function."""
        from arklex.evaluation.build_user_profiles import attributes_to_text

        attribute_list = [
            {"attr1": "value1", "attr2": "value2"},
            {"attr3": "value3"},
        ]

        result = attributes_to_text(attribute_list)

        assert len(result) == 2
        assert "attr1: value1" in result[0]
        assert "attr2: value2" in result[0]
        assert "attr3: value3" in result[1]
        assert not result[0].endswith("\n")  # Should not have trailing newline

    def test_attributes_to_text_with_empty_list(self) -> None:
        """Test attributes_to_text function with empty list."""
        from arklex.evaluation.build_user_profiles import attributes_to_text

        result = attributes_to_text([])
        assert result == []

    def test_attributes_to_text_with_empty_attributes(self) -> None:
        """Test attributes_to_text function with empty attributes."""
        from arklex.evaluation.build_user_profiles import attributes_to_text

        attribute_list = [{}]
        result = attributes_to_text(attribute_list)
        assert len(result) == 1
        assert result[0] == ""

    def test_pick_attributes_with_react_strategy(
        self, mock_config: dict[str, Any]
    ) -> None:
        """Test pick_attributes function with react strategy."""
        from arklex.evaluation.build_user_profiles import pick_attributes

        user_profile = {"profile_attr": "profile_value"}
        attributes = {"attr1": ["val1", "val2"]}
        goals = ["goal1", "goal2"]

        with patch(
            "arklex.evaluation.build_user_profiles.pick_attributes_react"
        ) as mock_react:
            mock_react.return_value = ({"goal": "goal1"}, {"goal": "goal1"})

            result_attrs, result_label = pick_attributes(
                user_profile, attributes, goals, strategy="react", client=Mock()
            )

            mock_react.assert_called_once()
            assert result_attrs["goal"] == "goal1"
            assert result_label["goal"] == "goal1"

    def test_pick_attributes_with_random_strategy(
        self, mock_config: dict[str, Any]
    ) -> None:
        """Test pick_attributes function with random strategy."""
        from arklex.evaluation.build_user_profiles import pick_attributes

        user_profile = {"profile_attr": "profile_value"}
        attributes = {"attr1": ["val1", "val2"]}
        goals = ["goal1", "goal2"]

        with patch(
            "arklex.evaluation.build_user_profiles.pick_attributes_random"
        ) as mock_random:
            mock_random.return_value = ({"goal": "goal2"}, {"goal": "goal2"})

            result_attrs, result_label = pick_attributes(
                user_profile, attributes, goals, strategy="random"
            )

            mock_random.assert_called_once()
            assert result_attrs["goal"] == "goal2"
            assert result_label["goal"] == "goal2"

    def test_select_random_attributes(self) -> None:
        """Test _select_random_attributes function."""
        from arklex.evaluation.build_user_profiles import _select_random_attributes

        attributes = {"attr1": ["val1", "val2"], "attr2": ["val3", "val4"]}
        goals = ["goal1", "goal2"]

        result_attrs, result_goal = _select_random_attributes(attributes, goals)

        assert "goal" in result_attrs
        assert result_attrs["goal"] in goals
        assert result_goal in goals
        assert "attr1" in result_attrs
        assert result_attrs["attr1"] in attributes["attr1"]
        assert "attr2" in result_attrs
        assert result_attrs["attr2"] in attributes["attr2"]

    def test_build_tool_list(self, mock_config: dict[str, Any]) -> None:
        """Test _build_tool_list function."""
        from arklex.evaluation.build_user_profiles import _build_tool_list

        with patch(
            "arklex.evaluation.build_user_profiles.Environment"
        ) as mock_environment:
            mock_env_instance = Mock()
            mock_tool = Mock()
            mock_tool.slots = [Mock(name="slot1", value="value1")]
            mock_tool.description = "Test tool description"
            mock_tool.output = "Test output"
            mock_env_instance.tools = {
                "tool1": {"execute": Mock(return_value=mock_tool)}
            }
            mock_environment.return_value = mock_env_instance

            tool_list, env = _build_tool_list(mock_config)

            assert len(tool_list) == 2  # Should include the "no tool" option
            assert tool_list[0]["tool_id"] == "tool1"
            assert tool_list[0]["tool_description"] == "Test tool description"
            assert tool_list[1]["tool_id"] == "0"  # No tool option
            assert env == mock_env_instance

    def test_convert_attributes_to_profiles(self, mock_config: dict[str, Any]) -> None:
        """Test convert_attributes_to_profiles function."""
        from arklex.evaluation.build_user_profiles import convert_attributes_to_profiles

        attributes_list = [
            {"goal": "goal1", "attr1": "value1"},
            {"goal": "goal2", "attr2": "value2"},
        ]
        system_attributes_list = [
            {"sys_attr1": "sys_value1"},
            {"sys_attr2": "sys_value2"},
        ]

        with patch(
            "arklex.evaluation.build_user_profiles.convert_attributes_to_profile",
            return_value="Test profile",
        ):
            profiles, goals, system_inputs = convert_attributes_to_profiles(
                attributes_list, system_attributes_list, mock_config
            )
            assert len(profiles) == 2
            assert profiles == ["Test profile", "Test profile"]
            assert len(goals) == 2
            assert goals == ["goal1", "goal2"]
            assert len(system_inputs) == 2
            assert system_inputs == system_attributes_list

    def test_select_random_attributes_empty_goals(self) -> None:
        from arklex.evaluation.build_user_profiles import _select_random_attributes

        attributes = {"attr1": ["a", "b"]}
        goals = []
        with pytest.raises(IndexError):
            _select_random_attributes(attributes, goals)

    def test_get_custom_profiles_binding_logic(
        self, mock_config: dict[str, Any]
    ) -> None:
        from arklex.evaluation.build_user_profiles import get_custom_profiles

        config = mock_config.copy()
        config["user_attributes"]["system_attributes"] = {
            "attr1": {"api": "http://test.com/api1", "bind_to": "user_profiles.attrB"},
            "attrB": {"api": "http://test.com/api2"},
        }
        config["user_attributes"]["user_profiles"] = {
            "profile1": {
                "api": "http://test.com/api2",
                "bind_to": "system_attributes.attrA",
            },
            "attrB": {"api": "http://test.com/api3"},
        }
        with patch("arklex.evaluation.build_user_profiles.requests.get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = ["v1"]
            mock_get.return_value = mock_resp
            user_profiles, system_attributes = get_custom_profiles(config)
            assert "profile1" in user_profiles
            assert "attr1" in system_attributes
            assert "attrB" in system_attributes
            assert "attrB" in user_profiles
            assert user_profiles["attrB"] == system_attributes["attrB"]

    def test_build_tool_list_missing_fields(self, mock_config: dict[str, Any]) -> None:
        from arklex.evaluation.build_user_profiles import _build_tool_list

        # Remove 'id' from a tool to trigger error
        config = mock_config.copy()
        config["tools"] = [{"name": "Tool without id"}]
        config["workers"] = []
        with pytest.raises(KeyError):
            _build_tool_list(config)

    def test_get_label_all_retries_fail(self, mock_config: dict[str, Any]) -> None:
        from arklex.evaluation.build_user_profiles import get_label

        # Patch chatgpt_chatbot to always raise
        # Add 'path' to tool config
        mock_config = mock_config.copy()
        mock_config["tools"] = [
            {"id": "tool1", "name": "Test Tool", "path": "dummy_path"}
        ]
        mock_config["workers"] = []

        # Patch Environment to avoid tool registration issues
        with patch(
            "arklex.evaluation.build_user_profiles.Environment"
        ) as mock_env_class:
            mock_env = Mock()
            # Create a proper mock tool object with slots
            mock_tool = Mock()
            mock_slot = Mock()
            mock_slot.model_dump.return_value = {"name": "test_slot"}
            mock_tool.slots = [mock_slot]
            mock_tool.description = "Test tool description"
            mock_env.tools = {"tool1": {"execute": Mock(return_value=mock_tool)}}
            mock_env_class.return_value = mock_env

            with patch(
                "arklex.evaluation.build_user_profiles.chatgpt_chatbot",
                side_effect=KeyError("fail"),
            ):
                label, success = get_label({"goal": "g"}, mock_config)
                assert label[0]["tool_id"] == "0"
                assert success is True

    def test_get_label_unexpected_exception(self, mock_config: dict[str, Any]) -> None:
        from arklex.evaluation.build_user_profiles import get_label

        # Patch chatgpt_chatbot to raise a generic Exception
        # Add 'path' to tool config
        mock_config = mock_config.copy()
        mock_config["tools"] = [
            {"id": "tool1", "name": "Test Tool", "path": "dummy_path"}
        ]
        mock_config["workers"] = []

        # Patch Environment to avoid tool registration issues
        with patch(
            "arklex.evaluation.build_user_profiles.Environment"
        ) as mock_env_class:
            mock_env = Mock()
            # Create a proper mock tool object with slots
            mock_tool = Mock()
            mock_slot = Mock()
            mock_slot.model_dump.return_value = {"name": "test_slot"}
            mock_tool.slots = [mock_slot]
            mock_tool.description = "Test tool description"
            mock_env.tools = {"tool1": {"execute": Mock(return_value=mock_tool)}}
            mock_env_class.return_value = mock_env

            with patch(
                "arklex.evaluation.build_user_profiles.chatgpt_chatbot",
                side_effect=Exception("fail"),
            ):
                label, success = get_label({"goal": "g"}, mock_config)
                assert label[0]["tool_id"] == "0"
                assert success is True

    def test__select_random_attributes_empty_attributes(self) -> None:
        from arklex.evaluation import build_user_profiles

        with pytest.raises(IndexError):
            build_user_profiles._select_random_attributes({}, [])

    def test__fetch_api_data_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from arklex.evaluation import build_user_profiles

        def raise_value_error(*a: object, **kw: object) -> NoReturn:  # type: ignore[misc]
            raise ValueError("bad json")

        monkeypatch.setattr(
            "arklex.evaluation.build_user_profiles.requests.get",
            lambda *a, **kw: type("R", (), {"json": raise_value_error})(),
        )
        try:
            build_user_profiles._fetch_api_data("http://fake", "key")
        except ValueError:
            pass
        else:
            raise AssertionError("Should raise ValueError")

    def test_get_custom_profiles_no_system_or_user_profiles(self) -> None:
        from arklex.evaluation import build_user_profiles

        config = {"user_attributes": {}}
        user_profiles, system_attributes = build_user_profiles.get_custom_profiles(
            config
        )
        assert user_profiles == {}
        assert system_attributes == {}

    def test_filter_attributes_excludes_goal_and_system(self) -> None:
        from arklex.evaluation import build_user_profiles

        config = {"user_attributes": {"goal": 1, "system_attributes": 2, "foo": 3}}
        result = build_user_profiles.filter_attributes(config)
        assert "goal" not in result and "system_attributes" not in result
        assert "foo" in result

    def test_get_label_unexpected_exception_branch(
        self, monkeypatch: pytest.MonkeyPatch, mock_config: dict[str, Any]
    ) -> None:
        from arklex.evaluation import build_user_profiles

        class DummyEnv:
            def __getitem__(self, k: str) -> object:
                raise Exception("fail")

        monkeypatch.setattr(
            "arklex.evaluation.build_user_profiles.Environment",
            lambda *a, **kw: DummyEnv(),
        )
        try:
            build_user_profiles.get_label({"goal": "foo"}, mock_config)
        except Exception:
            pass
        else:
            raise AssertionError("Should raise Exception")

    def test_chatgpt_chatbot_openai_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from arklex.evaluation import chatgpt_utils

        class DummyChat:
            class completions:
                @staticmethod
                def create(model: str, messages: list, temperature: float) -> object:
                    class Choice:
                        class Message:
                            content = " result "

                        message = Message()

                    class Result:
                        choices = [Choice()]

                    return Result()

        class DummyClient:
            chat = DummyChat()

        monkeypatch.setattr(
            chatgpt_utils,
            "MODEL",
            {"llm_provider": "openai", "model_type_or_path": "gpt"},
        )
        result = chatgpt_utils.chatgpt_chatbot(
            [{"role": "user", "content": "hi"}], DummyClient()
        )
        assert result == "result"

    def test_format_chat_history_str_trailing_space(self) -> None:
        from arklex.evaluation import chatgpt_utils

        chat_history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = chatgpt_utils.format_chat_history_str(chat_history)
        assert result.endswith("hi") and not result.endswith(" ")

    def test__select_random_attributes_goal_and_attributes(self) -> None:
        """Explicitly test _select_random_attributes for coverage (line 579)."""
        from arklex.evaluation.build_user_profiles import _select_random_attributes

        attributes = {"color": ["red", "blue"], "size": ["small", "large"]}
        goals = ["goal1", "goal2"]
        selected, goal = _select_random_attributes(attributes, goals)
        assert goal in goals
        assert set(selected.keys()) == {"goal", "color", "size"}
        assert selected["goal"] == goal
        assert selected["color"] in attributes["color"]
        assert selected["size"] in attributes["size"]

    def test_get_custom_profiles_binding_logic_full(self) -> None:
        """Test get_custom_profiles with full binding logic and API responses."""
        config = {
            "user_attributes": {
                "system_attributes": {
                    "attr1": {
                        "api": "http://api.example.com/system1",
                        "bind_to": "user_profiles.profile1",
                    }
                },
                "user_profiles": {
                    "profile1": {
                        "api": "http://api.example.com/profile1",
                        "bind_to": "system_attributes.attr1",
                    }
                },
            }
        }

        def fake_fetch_api_data(api_url: str, key: str) -> list[str]:
            if "system1" in api_url:
                return ["sys_val1", "sys_val2"]
            elif "profile1" in api_url:
                return ["prof_val1", "prof_val2"]
            return []

        with patch(
            "arklex.evaluation.build_user_profiles._fetch_api_data",
            side_effect=fake_fetch_api_data,
        ):
            user_profiles, system_attributes = get_custom_profiles(config)

        assert system_attributes["attr1"] == ["sys_val1", "sys_val2"]
        assert user_profiles["profile1"] == ["sys_val1", "sys_val2"]

    def test_select_random_attributes_with_goal_category(self) -> None:
        """Test _select_random_attributes when attributes contain a 'goal' category."""
        from arklex.evaluation.build_user_profiles import _select_random_attributes

        attributes = {
            "goal": ["goal1", "goal2"],
            "attr1": ["val1", "val2"],
            "attr2": ["val3", "val4"],
        }
        goals = ["goal1", "goal2"]

        selected_attributes, selected_goal = _select_random_attributes(
            attributes, goals
        )

        # Goal should be selected from goals list, not from attributes
        assert selected_goal in goals
        assert selected_attributes["goal"] == selected_goal
        # Other attributes should be selected from their respective lists
        assert selected_attributes["attr1"] in ["val1", "val2"]
        assert selected_attributes["attr2"] in ["val3", "val4"]

    def test_get_custom_profiles_with_non_dict_system_attributes(self) -> None:
        """Test get_custom_profiles when system_attributes values are not dictionaries."""
        config = {
            "user_attributes": {
                "system_attributes": {
                    "attr1": ["val1", "val2"],  # Not a dict
                    "attr2": {"api": "http://api.example.com/test"},
                },
                "user_profiles": {
                    "profile1": {"api": "http://api.example.com/profile1"}
                },
            }
        }

        with patch(
            "arklex.evaluation.build_user_profiles._fetch_api_data",
            return_value=["api_val1", "api_val2"],
        ):
            user_profiles, system_attributes = get_custom_profiles(config)

        # Non-dict values should be assigned directly
        assert system_attributes["attr1"] == ["val1", "val2"]
        # Dict values should be processed through API
        assert system_attributes["attr2"] == ["api_val1", "api_val2"]

    def test_get_custom_profiles_with_non_dict_user_profiles(self) -> None:
        """Test get_custom_profiles when user_profiles values are not dictionaries."""
        config = {
            "user_attributes": {
                "system_attributes": {
                    "attr1": {"api": "http://api.example.com/system1"}
                },
                "user_profiles": {
                    "profile1": ["prof1", "prof2"],  # Not a dict
                    "profile2": {"api": "http://api.example.com/profile2"},
                },
            }
        }

        with patch(
            "arklex.evaluation.build_user_profiles._fetch_api_data",
            return_value=["api_val1", "api_val2"],
        ):
            user_profiles, system_attributes = get_custom_profiles(config)

        # Non-dict values should be assigned directly
        assert user_profiles["profile1"] == ["prof1", "prof2"]
        # Dict values should be processed through API
        assert user_profiles["profile2"] == ["api_val1", "api_val2"]

    def test_augment_attributes_without_documents_using_wo_doc_prompt(self) -> None:
        """Test augment_attributes when documents are empty, using ADD_ATTRIBUTES_WO_DOC prompt."""
        config = {
            "company_summary": "Test company summary",
            "client": Mock(),
        }
        predefined_attributes = {
            "category1": {
                "values": ["val1", "val2"],
                "augment": True,
            }
        }
        documents = []  # Empty documents list

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatbot:
            mock_chatbot.return_value = "new_val1, new_val2, new_val3"

            result = augment_attributes(predefined_attributes, config, documents)

            # Should use ADD_ATTRIBUTES_WO_DOC prompt (without company_doc)
            assert mock_chatbot.called
            call_args = mock_chatbot.call_args[0][0]
            assert "company_doc" not in call_args
            assert "Here is a page from the company website" not in call_args

            # Should include original and new values
            assert "val1" in result["category1"]
            assert "val2" in result["category1"]
            assert "new_val1" in result["category1"]
            assert "new_val2" in result["category1"]
            assert "new_val3" in result["category1"]

    def test_fetch_api_data_request_exception_returns_empty_list(self) -> None:
        """Test _fetch_api_data when RequestException occurs (line 688)."""
        from arklex.evaluation.build_user_profiles import _fetch_api_data

        with patch("arklex.evaluation.build_user_profiles.requests.get") as mock_get:
            # Mock RequestException (not ValueError)
            mock_get.side_effect = requests.RequestException("Network error")

            result = _fetch_api_data("http://api.example.com/test", "test_key")

            # Should return empty list for RequestException
            assert result == []

    def test_augment_attributes_uses_wo_doc_prompt_when_documents_empty(self) -> None:
        """Test augment_attributes uses ADD_ATTRIBUTES_WO_DOC when documents are empty (lines 914-915)."""
        from arklex.evaluation.build_user_profiles import (
            augment_attributes,
        )

        config = {
            "company_summary": "Test company summary",
            "client": Mock(),
        }
        predefined_attributes = {
            "category1": {
                "values": ["val1", "val2"],
                "augment": True,
            }
        }
        documents = []  # Empty documents list

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatbot:
            mock_chatbot.return_value = "new_val1, new_val2"

            result = augment_attributes(predefined_attributes, config, documents)

            # Verify the prompt used contains ADD_ATTRIBUTES_WO_DOC content
            call_args = mock_chatbot.call_args[0][0]
            # Check that it uses the wo_doc prompt (no company_doc field)
            assert "company_doc" not in str(call_args)
            assert "Here is the summary of the company:" in str(
                call_args
            )  # Part of ADD_ATTRIBUTES_WO_DOC

            # Verify the result includes both original and new values
            assert "val1" in result["category1"]
            assert "val2" in result["category1"]
            assert "new_val1" in result["category1"]
            assert "new_val2" in result["category1"]

    def test_augment_attributes_llm(
        self, monkeypatch: pytest.MonkeyPatch, mock_config: dict[str, Any]
    ) -> None:
        from arklex.evaluation import build_user_profiles

        called = {}

        def fake_chatgpt_chatbot(prompt: str, client: object) -> str:
            called["prompt"] = prompt
            return "aug1, aug2"

        monkeypatch.setattr(
            build_user_profiles, "chatgpt_chatbot", fake_chatgpt_chatbot
        )
        predefined = {"cat": {"values": ["a"], "augment": True}}
        docs = [{"content": "doc content"}]
        out = build_user_profiles.augment_attributes(predefined, mock_config, docs)
        assert "cat" in out and "aug1" in out["cat"] and "aug2" in out["cat"]
        assert called["prompt"]

    def test_build_user_profiles_return_line(self) -> None:
        from arklex.evaluation.build_user_profiles import build_user_profiles

        # Should always return an empty list, even with non-empty input
        result = build_user_profiles([{"foo": "bar"}])
        assert result == []

    def test_attributes_to_text_return_line_edge_case(self) -> None:
        from arklex.evaluation.build_user_profiles import attributes_to_text

        # Edge case: attribute dict with one key-value
        result = attributes_to_text([{"a": 1}])
        assert result == ["a: 1"]
        # Edge case: attribute dict with multiple key-values
        result = attributes_to_text([{"a": 1, "b": 2}])
        assert all(isinstance(s, str) for s in result)

    def test_get_label_unexpected_exception_branch_full(
        self, monkeypatch: pytest.MonkeyPatch, mock_config: dict[str, Any]
    ) -> None:
        """Covers the except Exception branch in get_label (lines 914-915)."""
        from arklex.evaluation.build_user_profiles import get_label

        calls = {"count": 0}

        class DummyEnv:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            @property
            def tools(self) -> dict[str, dict[str, object]]:
                # Return a dict that will raise an exception when accessed
                return {
                    "tool1": {
                        "execute": lambda: (_ for _ in ()).throw(
                            Exception("unexpected error")
                        )
                    }
                }

        def dummy_log_error(msg: str, *args: object, **kwargs: object) -> None:
            calls["count"] += 1

        def mock_build_tool_list(
            config: dict[str, object],
        ) -> tuple[list[dict[str, object]], DummyEnv]:
            # Return a tool list and an environment that will raise an exception
            tool_list = [
                {
                    "tool_id": "tool1",
                    "tool_description": "Test Tool",
                    "tool_input": [],
                    "tool_output": "Test output",
                }
            ]
            return tool_list, DummyEnv()

        monkeypatch.setattr(
            "arklex.evaluation.build_user_profiles._build_tool_list",
            mock_build_tool_list,
        )
        monkeypatch.setattr(
            "arklex.evaluation.build_user_profiles.log_context",
            type(
                "Log", (), {"error": dummy_log_error, "warning": lambda *a, **kw: None}
            )(),
        )
        attr = {"tool_id": "tool1", "goal": "test goal"}
        config = {
            "tools": [{"id": "tool1", "name": "Test Tool", "slots": []}],
            "user_attributes": {},
            "workers": [],
            "client": Mock(),
        }
        # Should trigger the except Exception branch and retry
        label, success = get_label(attr, config)
        # The exception should be caught and logged, but the function should still return successfully
        assert isinstance(label, list)
        assert success is True
