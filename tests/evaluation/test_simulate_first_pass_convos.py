"""Tests for simulate_first_pass_convos module."""

import pytest
from unittest.mock import patch, MagicMock
from arklex.evaluation.simulate_first_pass_convos import (
    get_relevant_vals,
    count_matches,
)


class TestGetRelevantVals:
    """Test cases for get_relevant_vals function."""

    def test_get_relevant_vals_complete_attributes(self):
        """Test with complete attributes."""
        attr = {
            "goal": "test_goal",
            "product_experience_level": "beginner",
            "customer_type": "individual",
            "persona": "explorer",
            "discovery_type": "self_guided",
            "buying_behavior": "analytical",
        }
        result = get_relevant_vals(attr)
        
        expected = [
            "test_goal",
            "beginner",
            "individual",
            "explorer",
            "self_guided",
            "analytical",
        ]
        assert result == expected

    def test_get_relevant_vals_partial_attributes(self):
        """Test with partial attributes."""
        attr = {
            "goal": "test_goal",
            "product_experience_level": "beginner",
            # Missing other keys
        }
        result = get_relevant_vals(attr)
        
        # Should still return values for available keys
        assert len(result) == 6
        assert result[0] == "test_goal"
        assert result[1] == "beginner"

    def test_get_relevant_vals_empty_attributes(self):
        """Test with empty attributes."""
        attr = {}
        result = get_relevant_vals(attr)
        
        # Should return list with None values for missing keys
        assert len(result) == 6
        assert all(val is None for val in result)

    def test_get_relevant_vals_none_values(self):
        """Test with None values."""
        attr = {
            "goal": None,
            "product_experience_level": None,
            "customer_type": None,
            "persona": None,
            "discovery_type": None,
            "buying_behavior": None,
        }
        result = get_relevant_vals(attr)
        
        expected = [None, None, None, None, None, None]
        assert result == expected

    def test_get_relevant_vals_mixed_types(self):
        """Test with mixed value types."""
        attr = {
            "goal": "test_goal",
            "product_experience_level": 1,
            "customer_type": True,
            "persona": "explorer",
            "discovery_type": None,
            "buying_behavior": "analytical",
        }
        result = get_relevant_vals(attr)
        
        expected = ["test_goal", 1, True, "explorer", None, "analytical"]
        assert result == expected


class TestCountMatches:
    """Test cases for count_matches function."""

    def test_count_matches_identical_lists(self):
        """Test with identical lists."""
        l1 = ["a", "b", "c"]
        l2 = ["a", "b", "c"]
        result = count_matches(l1, l2)
        assert result == 3

    def test_count_matches_no_matches(self):
        """Test with no matches."""
        l1 = ["a", "b", "c"]
        l2 = ["d", "e", "f"]
        result = count_matches(l1, l2)
        assert result == 0

    def test_count_matches_partial_matches(self):
        """Test with partial matches."""
        l1 = ["a", "b", "c"]
        l2 = ["a", "e", "c"]
        result = count_matches(l1, l2)
        assert result == 2

    def test_count_matches_empty_lists(self):
        """Test with empty lists."""
        l1 = []
        l2 = []
        result = count_matches(l1, l2)
        assert result == 0

    def test_count_matches_one_empty_list(self):
        """Test with one empty list."""
        l1 = ["a", "b", "c"]
        l2 = []
        result = count_matches(l1, l2)
        assert result == 0

    def test_count_matches_different_lengths(self):
        """Test with different length lists."""
        l1 = ["a", "b", "c"]
        l2 = ["a", "b"]
        result = count_matches(l1, l2)
        assert result == 2

    def test_count_matches_with_none_values(self):
        """Test with None values."""
        l1 = ["a", None, "c"]
        l2 = ["a", "b", "c"]
        result = count_matches(l1, l2)
        assert result == 2

    def test_count_matches_with_duplicates(self):
        """Test with duplicate values."""
        l1 = ["a", "a", "b"]
        l2 = ["a", "b", "c"]
        result = count_matches(l1, l2)
        assert result == 2

    def test_count_matches_mixed_types(self):
        """Test with mixed types."""
        l1 = ["a", 1, True]
        l2 = ["a", 1, False]
        result = count_matches(l1, l2)
        assert result == 2
