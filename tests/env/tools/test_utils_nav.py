"""
Tests for arklex.env.tools.shopify.utils_nav module.

This module tests the pagination utilities used by Shopify tools,
including the cursorify function and related constants.
"""

import unittest

from arklex.env.tools.shopify.utils_nav import (
    NAVIGATE_WITH_NO_CURSOR,
    NO_NEXT_PAGE,
    NO_PREV_PAGE,
    PAGEINFO_OUTPUTS,
    PAGEINFO_SLOTS,
    cursorify,
)


class TestUtilsNavConstants(unittest.TestCase):
    """Test the constant definitions in utils_nav module."""

    def test_pageinfo_slots_structure(self) -> None:
        """Test that PAGEINFO_SLOTS has the correct structure and content."""
        self.assertIsInstance(PAGEINFO_SLOTS, list)
        self.assertEqual(len(PAGEINFO_SLOTS), 3)

        # Test limit slot
        limit_slot = PAGEINFO_SLOTS[0]
        self.assertEqual(limit_slot["name"], "limit")
        self.assertEqual(limit_slot["type"], "string")
        self.assertEqual(
            limit_slot["description"], "Maximum number of entries to show."
        )
        self.assertEqual(limit_slot["prompt"], "")
        self.assertFalse(limit_slot["required"])

        # Test navigate slot
        navigate_slot = PAGEINFO_SLOTS[1]
        self.assertEqual(navigate_slot["name"], "navigate")
        self.assertEqual(navigate_slot["type"], "string")
        self.assertIn(
            "navigate relative to previous view", navigate_slot["description"]
        )
        self.assertEqual(navigate_slot["prompt"], "")
        self.assertFalse(navigate_slot["required"])

        # Test pageInfo slot
        pageinfo_slot = PAGEINFO_SLOTS[2]
        self.assertEqual(pageinfo_slot["name"], "pageInfo")
        self.assertEqual(pageinfo_slot["type"], "string")
        self.assertIn("The previous pageInfo object", pageinfo_slot["description"])
        self.assertEqual(pageinfo_slot["prompt"], "")
        self.assertFalse(pageinfo_slot["required"])

    def test_pageinfo_outputs_structure(self) -> None:
        """Test that PAGEINFO_OUTPUTS has the correct structure and content."""
        self.assertIsInstance(PAGEINFO_OUTPUTS, list)
        self.assertEqual(len(PAGEINFO_OUTPUTS), 1)

        pageinfo_output = PAGEINFO_OUTPUTS[0]
        self.assertEqual(pageinfo_output["name"], "pageInfo")
        self.assertEqual(pageinfo_output["type"], "string")
        self.assertIn("Current pageInfo object", pageinfo_output["description"])

    def test_error_constants(self) -> None:
        """Test that error constants are properly defined."""
        self.assertEqual(
            NAVIGATE_WITH_NO_CURSOR, "error: cannot navigate without reference cursor"
        )
        self.assertEqual(NO_NEXT_PAGE, "error: no more pages after")
        self.assertEqual(NO_PREV_PAGE, "error: no more pages before")


class TestCursorifyFunction(unittest.TestCase):
    """Test the cursorify function with various input scenarios."""

    def test_cursorify_default_parameters(self) -> None:
        """Test cursorify with no parameters (uses defaults)."""
        result, success = cursorify({})

        self.assertTrue(success)
        self.assertEqual(result, "first: 3")

    def test_cursorify_with_limit_only(self) -> None:
        """Test cursorify with only limit parameter."""
        result, success = cursorify({"limit": "5"})

        self.assertTrue(success)
        self.assertEqual(result, "first: 5")

    def test_cursorify_with_navigate_stay(self) -> None:
        """Test cursorify with navigate='stay' (default behavior)."""
        result, success = cursorify({"navigate": "stay"})

        self.assertTrue(success)
        self.assertEqual(result, "first: 3")

    def test_cursorify_with_navigate_none(self) -> None:
        """Test cursorify with navigate=None (should behave like 'stay')."""
        result, success = cursorify({"navigate": None})

        self.assertTrue(success)
        self.assertEqual(result, "first: 3")

    def test_cursorify_navigate_next_without_pageinfo(self) -> None:
        """Test cursorify with navigate='next' but no pageInfo (should fail)."""
        result, success = cursorify({"navigate": "next"})

        self.assertFalse(success)
        self.assertEqual(result, NAVIGATE_WITH_NO_CURSOR)

    def test_cursorify_navigate_prev_without_pageinfo(self) -> None:
        """Test cursorify with navigate='prev' but no pageInfo (should fail)."""
        result, success = cursorify({"navigate": "prev"})

        self.assertFalse(success)
        self.assertEqual(result, NAVIGATE_WITH_NO_CURSOR)

    def test_cursorify_navigate_next_with_pageinfo_no_next_page(self) -> None:
        """Test cursorify with navigate='next' but pageInfo indicates no next page."""
        pageinfo = {
            "endCursor": "cursor123",
            "hasNextPage": False,
            "hasPreviousPage": True,
            "startCursor": "cursor456",
        }
        result, success = cursorify({"navigate": "next", "pageInfo": pageinfo})

        self.assertFalse(success)
        self.assertEqual(result, NO_NEXT_PAGE)

    def test_cursorify_navigate_prev_with_pageinfo_no_prev_page(self) -> None:
        """Test cursorify with navigate='prev' but pageInfo indicates no previous page."""
        pageinfo = {
            "endCursor": "cursor123",
            "hasNextPage": True,
            "hasPreviousPage": False,
            "startCursor": "cursor456",
        }
        result, success = cursorify({"navigate": "prev", "pageInfo": pageinfo})

        self.assertFalse(success)
        self.assertEqual(result, NO_PREV_PAGE)

    def test_cursorify_navigate_next_success(self) -> None:
        """Test cursorify with navigate='next' and valid pageInfo."""
        pageinfo = {
            "endCursor": "cursor123",
            "hasNextPage": True,
            "hasPreviousPage": True,
            "startCursor": "cursor456",
        }
        result, success = cursorify({"navigate": "next", "pageInfo": pageinfo})

        self.assertTrue(success)
        self.assertEqual(result, 'first: 3, after: "cursor123"')

    def test_cursorify_navigate_prev_success(self) -> None:
        """Test cursorify with navigate='prev' and valid pageInfo."""
        pageinfo = {
            "endCursor": "cursor123",
            "hasNextPage": True,
            "hasPreviousPage": True,
            "startCursor": "cursor456",
        }
        result, success = cursorify({"navigate": "prev", "pageInfo": pageinfo})

        self.assertTrue(success)
        self.assertEqual(result, 'last: 3, before: "cursor456"')

    def test_cursorify_with_custom_limit_and_navigation(self) -> None:
        """Test cursorify with custom limit and navigation."""
        pageinfo = {
            "endCursor": "cursor123",
            "hasNextPage": True,
            "hasPreviousPage": True,
            "startCursor": "cursor456",
        }
        result, success = cursorify(
            {"limit": "10", "navigate": "next", "pageInfo": pageinfo}
        )

        self.assertTrue(success)
        self.assertEqual(result, 'first: 10, after: "cursor123"')

    def test_cursorify_with_invalid_limit(self) -> None:
        """Test cursorify with invalid limit (should raise ValueError)."""
        # This should raise a ValueError when trying to convert "invalid" to int
        with self.assertRaises(ValueError):
            cursorify({"limit": "invalid"})

    def test_cursorify_with_zero_limit(self) -> None:
        """Test cursorify with limit=0."""
        result, success = cursorify({"limit": "0"})

        self.assertTrue(success)
        self.assertEqual(result, "first: 0")

    def test_cursorify_with_negative_limit(self) -> None:
        """Test cursorify with negative limit."""
        result, success = cursorify({"limit": "-5"})

        self.assertTrue(success)
        self.assertEqual(result, "first: -5")

    def test_cursorify_with_empty_string_navigate(self) -> None:
        """Test cursorify with empty string navigate (should behave like 'stay')."""
        result, success = cursorify({"navigate": ""})

        self.assertTrue(success)
        self.assertEqual(result, "first: 3")

    def test_cursorify_with_invalid_navigate_value(self) -> None:
        """Test cursorify with invalid navigate value (should fail without pageInfo)."""
        result, success = cursorify({"navigate": "invalid"})

        self.assertFalse(success)
        self.assertEqual(result, NAVIGATE_WITH_NO_CURSOR)

    def test_cursorify_with_complex_pageinfo(self) -> None:
        """Test cursorify with complex pageInfo object."""
        pageinfo = {
            "endCursor": "eyJsYXN0X2lkIjo3Mjk2NTgxODk0MjU3LCJsYXN0X3ZhbHVlIjoiNzI5NjU4MTg5NDI1NyJ9",
            "hasNextPage": True,
            "hasPreviousPage": True,
            "startCursor": "eyJsYXN0X2lkIjo3Mjk2NTgwODQ1NjgxLCJsYXN0X3ZhbHVlIjoiNzI5NjU4MDg0NTY4MSJ9",
        }
        result, success = cursorify(
            {"limit": "5", "navigate": "next", "pageInfo": pageinfo}
        )

        self.assertTrue(success)
        expected_cursor = (
            "eyJsYXN0X2lkIjo3Mjk2NTgxODk0MjU3LCJsYXN0X3ZhbHVlIjoiNzI5NjU4MTg5NDI1NyJ9"
        )
        self.assertEqual(result, f'first: 5, after: "{expected_cursor}"')

    def test_cursorify_with_missing_pageinfo_fields(self) -> None:
        """Test cursorify with pageInfo missing required fields."""
        # Missing hasNextPage
        pageinfo_incomplete = {
            "endCursor": "cursor123",
            "hasPreviousPage": True,
            "startCursor": "cursor456",
        }

        # When hasNextPage is missing, it defaults to False, so navigation should fail
        result, success = cursorify(
            {"navigate": "next", "pageInfo": pageinfo_incomplete}
        )

        self.assertFalse(success)
        self.assertEqual(result, NO_NEXT_PAGE)

    def test_cursorify_with_none_pageinfo(self) -> None:
        """Test cursorify with pageInfo=None."""
        result, success = cursorify({"navigate": "next", "pageInfo": None})

        self.assertFalse(success)
        self.assertEqual(result, NAVIGATE_WITH_NO_CURSOR)

    def test_cursorify_with_empty_pageinfo(self) -> None:
        """Test cursorify with empty pageInfo dict."""
        result, success = cursorify({"navigate": "next", "pageInfo": {}})

        # Empty dict is falsy, so it should fail without reference cursor
        self.assertFalse(success)
        self.assertEqual(result, NAVIGATE_WITH_NO_CURSOR)

    def test_cursorify_return_type(self) -> None:
        """Test that cursorify always returns a tuple of (str, bool)."""
        result, success = cursorify({})

        self.assertIsInstance(result, str)
        self.assertIsInstance(success, bool)

    def test_cursorify_all_parameters(self) -> None:
        """Test cursorify with all parameters provided."""
        pageinfo = {
            "endCursor": "cursor123",
            "hasNextPage": True,
            "hasPreviousPage": True,
            "startCursor": "cursor456",
        }
        result, success = cursorify(
            {"limit": "7", "navigate": "prev", "pageInfo": pageinfo}
        )

        self.assertTrue(success)
        self.assertEqual(result, 'last: 7, before: "cursor456"')


if __name__ == "__main__":
    unittest.main()
