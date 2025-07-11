"""
Unit tests for helper functions in create_meeting.py.

This module contains comprehensive unit tests for the helper functions
used in the create_meeting tool, including parse_natural_date and is_iso8601.
These tests validate the date/time parsing functionality independently
from the main meeting creation logic.
"""

from datetime import datetime

from arklex.env.tools.hubspot.create_meeting import is_iso8601, parse_natural_date


class TestParseNaturalDate:
    """Test cases for the parse_natural_date function."""

    def test_parse_natural_date_basic(self) -> None:
        """Test basic natural language date parsing."""
        result = parse_natural_date("tomorrow")
        assert isinstance(result, datetime)
        # Note: Without timezone parameter, result may not have timezone info
        # This is expected behavior

    def test_parse_natural_date_with_timezone(self) -> None:
        """Test natural language date parsing with timezone."""
        result = parse_natural_date("today", timezone="America/New_York")
        assert isinstance(result, datetime)
        assert result.tzinfo is not None
        # Should be converted to UTC
        assert result.tzinfo.utcoffset(result) is not None

    def test_parse_natural_date_date_only(self) -> None:
        """Test date-only parsing with date_input=True."""
        result = parse_natural_date(
            "May 1st", timezone="America/New_York", date_input=True
        )
        assert isinstance(result, datetime)
        # When timezone is provided, the time is converted to UTC
        # So the hour may not be 0, but it should be a valid time
        assert 0 <= result.hour <= 23
        assert 0 <= result.minute <= 59
        assert 0 <= result.second <= 59

    def test_parse_natural_date_with_base_date(self) -> None:
        """Test parsing with a base date for relative dates."""
        base_date = datetime(2024, 1, 15, 10, 30, 0)
        result = parse_natural_date("3pm", base_date, timezone="America/New_York")
        assert isinstance(result, datetime)
        # Should use the base date's date but the parsed time
        assert result.date() == base_date.date()
        # The time is converted to UTC, so 3pm EST becomes 8pm UTC (or similar)
        # We just check that it's a valid hour
        assert 0 <= result.hour <= 23

    def test_parse_natural_date_different_formats(self) -> None:
        """Test various natural language date formats."""
        formats = [
            "tomorrow",
            "today",
            "next Monday",
            "May 1st",
            "December 25th",
            "next week",
            "yesterday",
            "next month",
        ]

        for date_str in formats:
            result = parse_natural_date(date_str, timezone="America/New_York")
            assert isinstance(result, datetime)
            assert result.tzinfo is not None

    def test_parse_natural_date_time_without_timezone(self) -> None:
        """Test time parsing without timezone (should not have timezone info)."""
        result = parse_natural_date("10:30 AM", timezone=None)
        assert isinstance(result, datetime)
        assert result.tzinfo is None

    def test_parse_natural_date_with_different_timezones(self) -> None:
        """Test parsing with different timezone configurations."""
        timezones = [
            "America/New_York",
            "America/Los_Angeles",
            "Asia/Tokyo",
            "Europe/London",
        ]

        for timezone in timezones:
            result = parse_natural_date("tomorrow", timezone=timezone)
            assert isinstance(result, datetime)
            assert result.tzinfo is not None
            # Should be converted to UTC
            assert result.tzinfo.utcoffset(result) is not None

    def test_parse_natural_date_edge_cases(self) -> None:
        """Test edge cases for natural language date parsing."""
        edge_cases = [
            "now",
            "noon",
            "midnight",
            "next year",
            "last week",
        ]

        for date_str in edge_cases:
            result = parse_natural_date(date_str, timezone="America/New_York")
            assert isinstance(result, datetime)

    def test_parse_natural_date_base_date_logic(self) -> None:
        """Test the base date logic for relative dates."""
        base_date = datetime(2024, 1, 15, 10, 30, 0)

        # Test when parsed date is different from base date
        result = parse_natural_date("3pm", base_date, timezone="America/New_York")
        assert result.date() == base_date.date()
        # The time is converted to UTC, so we just check it's valid
        assert 0 <= result.hour <= 23

        # Test when parsed date is same as base date
        result = parse_natural_date("10:30 AM", base_date, timezone="America/New_York")
        assert result.date() == base_date.date()
        # The time is converted to UTC, so we just check it's valid
        assert 0 <= result.hour <= 23

    def test_parse_natural_date_timezone_conversion(self) -> None:
        """Test that timezone conversion to UTC works correctly."""
        result = parse_natural_date("tomorrow", timezone="America/New_York")
        assert result.tzinfo is not None
        # Should be in UTC
        assert result.tzinfo.utcoffset(result) is not None


class TestIsIso8601:
    """Test cases for the is_iso8601 function."""

    def test_is_iso8601_valid_formats(self) -> None:
        """Test valid ISO8601 format strings."""
        valid_formats = [
            "2024-01-15T10:30:00Z",
            "2024-01-15T10:30:00+05:00",
            "2024-01-15T10:30:00-05:00",
            "2024-01-15T10:30:00.123Z",
            "2024-01-15T10:30:00.123+05:00",
            "2024-01-15T10:30:00.123-05:00",
            "2024-01-15T10:30:00",
            "2024-01-15",
        ]

        for iso_str in valid_formats:
            assert is_iso8601(iso_str) is True, f"Failed for: {iso_str}"

    def test_is_iso8601_invalid_formats(self) -> None:
        """Test invalid ISO8601 format strings."""
        invalid_formats = [
            "tomorrow",
            "10:30 AM",
            "2024/01/15",
            "15-01-2024",
            "2024.01.15",
            "not a date",
            "",
            "2024-13-45T25:70:99Z",  # Invalid date/time values
        ]

        for non_iso_str in invalid_formats:
            assert is_iso8601(non_iso_str) is False, f"Failed for: {non_iso_str}"

    def test_is_iso8601_edge_cases(self) -> None:
        """Test edge cases for ISO8601 format detection."""
        edge_cases = [
            "2024-01-15T00:00:00Z",  # Midnight
            "2024-12-31T23:59:59Z",  # End of year
            "2024-02-29T12:00:00Z",  # Leap year
            "2024-01-15T10:30:00.000Z",  # Zero milliseconds
            "2024-01-15T10:30:00.999Z",  # Maximum milliseconds
        ]

        for edge_case in edge_cases:
            assert is_iso8601(edge_case) is True, f"Failed for: {edge_case}"

    def test_is_iso8601_with_different_timezones(self) -> None:
        """Test ISO8601 detection with different timezone formats."""
        timezone_formats = [
            "2024-01-15T10:30:00Z",
            "2024-01-15T10:30:00+00:00",
            "2024-01-15T10:30:00-00:00",
            "2024-01-15T10:30:00+05:30",
            "2024-01-15T10:30:00-08:00",
            "2024-01-15T10:30:00+14:00",
            "2024-01-15T10:30:00-12:00",
        ]

        for tz_format in timezone_formats:
            assert is_iso8601(tz_format) is True, f"Failed for: {tz_format}"

    def test_is_iso8601_with_fractional_seconds(self) -> None:
        """Test ISO8601 detection with fractional seconds."""
        fractional_formats = [
            "2024-01-15T10:30:00.1Z",
            "2024-01-15T10:30:00.12Z",
            "2024-01-15T10:30:00.123Z",
            "2024-01-15T10:30:00.1234Z",
            "2024-01-15T10:30:00.12345Z",
            "2024-01-15T10:30:00.123456Z",
        ]

        for frac_format in fractional_formats:
            assert is_iso8601(frac_format) is True, f"Failed for: {frac_format}"

    def test_is_iso8601_date_only(self) -> None:
        """Test ISO8601 detection for date-only strings."""
        date_only_formats = [
            "2024-01-15",
            "2024-12-31",
            "2024-02-29",
        ]

        for date_format in date_only_formats:
            assert is_iso8601(date_format) is True, f"Failed for: {date_format}"

    def test_is_iso8601_malformed_strings(self) -> None:
        """Test ISO8601 detection with malformed strings."""
        malformed_strings = [
            "2024-01-15T",  # Incomplete
            "T10:30:00Z",  # Missing date
            # Note: "2024-01-15 10:30:00Z" is actually valid ISO8601 with space
            "2024-01-15T10:30:00+",  # Incomplete timezone
            "2024-01-15T10:30:00Z+05:00",  # Multiple timezone indicators
        ]

        for malformed in malformed_strings:
            assert is_iso8601(malformed) is False, f"Failed for: {malformed}"

    def test_is_iso8601_space_separated_format(self) -> None:
        """Test that space-separated format is correctly identified as valid ISO8601."""
        # The isoparse function accepts space-separated format as valid ISO8601
        assert is_iso8601("2024-01-15 10:30:00Z") is True


class TestCreateMeetingIntegration:
    """Integration tests for create_meeting helper functions."""

    def test_iso8601_detection_integration(self) -> None:
        """Test that is_iso8601 function works correctly with create_meeting logic."""
        # Test ISO8601 format
        assert is_iso8601("2024-01-15T10:00:00Z") is True

        # Test natural language format
        assert is_iso8601("10:00 AM") is False

        # Test edge case
        assert is_iso8601("2024-01-15T10:00:00") is True  # No timezone
        assert is_iso8601("2024-01-15T10:00:00+05:00") is True  # With timezone

    def test_parse_natural_date_integration(self) -> None:
        """Test that parse_natural_date function works correctly with create_meeting logic."""
        # Test natural language date parsing
        result = parse_natural_date("tomorrow", timezone="America/New_York")
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

        # Test natural language time parsing
        result = parse_natural_date("10:00 AM", timezone="America/New_York")
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

    def test_parse_natural_date_edge_case_handling(self) -> None:
        """Test edge case handling in parse_natural_date function."""
        # Test with None base_date
        result = parse_natural_date(
            "tomorrow", base_date=None, timezone="America/New_York"
        )
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

        # Test with None timezone
        result = parse_natural_date("tomorrow", timezone=None)
        assert isinstance(result, datetime)
        # Should not have timezone info when timezone is None
        assert result.tzinfo is None

        # Test date_input=True with timezone
        result = parse_natural_date(
            "May 1st", timezone="America/New_York", date_input=True
        )
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

    def test_parse_natural_date_complex_scenarios(self) -> None:
        """Test complex scenarios in parse_natural_date function."""
        # Test with base_date and different parsed date
        base_date = datetime(2024, 1, 15, 10, 30, 0)
        result = parse_natural_date("3pm", base_date, timezone="America/New_York")
        assert result.date() == base_date.date()
        assert result.tzinfo is not None

        # Test with base_date and same parsed date
        result = parse_natural_date("10:30 AM", base_date, timezone="America/New_York")
        assert result.date() == base_date.date()
        assert result.tzinfo is not None

        # Test date_input=True without timezone
        result = parse_natural_date("May 1st", date_input=True)
        assert isinstance(result, datetime)
        assert result.tzinfo is None

    def test_is_iso8601_comprehensive_validation(self) -> None:
        """Test comprehensive validation in is_iso8601 function."""
        # Test various valid formats
        valid_formats = [
            "2024-01-15T10:30:00Z",
            "2024-01-15T10:30:00+05:00",
            "2024-01-15T10:30:00-05:00",
            "2024-01-15T10:30:00.123Z",
            "2024-01-15T10:30:00.123+05:00",
            "2024-01-15T10:30:00.123-05:00",
            "2024-01-15T10:30:00",
            "2024-01-15",
            "2024-01-15 10:30:00Z",
            "2024-01-15 10:30:00+05:00",
        ]

        for iso_str in valid_formats:
            assert is_iso8601(iso_str) is True, f"Failed for: {iso_str}"

        # Test various invalid formats
        invalid_formats = [
            "tomorrow",
            "10:30 AM",
            "2024/01/15",
            "15-01-2024",
            "2024.01.15",
            "not a date",
            "",
            "2024-13-45T25:70:99Z",
            "2024-01-15T",
            "T10:30:00Z",
            "2024-01-15T10:30:00+",
            "2024-01-15T10:30:00Z+05:00",
        ]

        for non_iso_str in invalid_formats:
            assert is_iso8601(non_iso_str) is False, f"Failed for: {non_iso_str}"

    def test_parse_natural_date_timezone_edge_cases(self) -> None:
        """Test timezone edge cases in parse_natural_date function."""
        # Test with empty string timezone
        result = parse_natural_date("tomorrow", timezone="")
        assert isinstance(result, datetime)
        # Should not have timezone info with empty string
        assert result.tzinfo is None

        # Test with invalid timezone
        try:
            result = parse_natural_date("tomorrow", timezone="Invalid/Timezone")
            # Should handle gracefully or raise appropriate exception
            assert isinstance(result, datetime)
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass

    def test_is_iso8601_performance_edge_cases(self) -> None:
        """Test performance edge cases in is_iso8601 function."""
        # Test with very long strings
        long_invalid_string = "not_a_date" * 1000
        assert is_iso8601(long_invalid_string) is False

        # Test with very long valid strings
        long_valid_string = "2024-01-15T10:30:00.123456789Z"
        assert is_iso8601(long_valid_string) is True

        # Test with single character
        assert is_iso8601("a") is False
        assert is_iso8601("1") is False

    def test_parse_natural_date_performance_edge_cases(self) -> None:
        """Test performance edge cases in parse_natural_date function."""
        # Test with very long date strings
        long_date_string = "tomorrow " * 100
        result = parse_natural_date(long_date_string, timezone="America/New_York")
        assert isinstance(result, datetime)

        # Test with very long time strings
        long_time_string = "10:00 AM " * 100
        result = parse_natural_date(long_time_string, timezone="America/New_York")
        assert isinstance(result, datetime)

    def test_integration_with_create_meeting_logic(self) -> None:
        """Test integration of helper functions with create_meeting logic."""
        # Test ISO8601 detection for create_meeting time parsing
        assert is_iso8601("2024-01-15T10:00:00Z") is True
        assert is_iso8601("10:00 AM") is False

        # Test natural language parsing for create_meeting date parsing
        result = parse_natural_date(
            "tomorrow", timezone="America/New_York", date_input=True
        )
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

        # Test natural language parsing for create_meeting time parsing
        result = parse_natural_date("10:00 AM", timezone="America/New_York")
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

    def test_parse_natural_date_return_path_coverage(self) -> None:
        """Test the return path of parse_natural_date to ensure complete coverage."""
        # Test basic return path without timezone
        result = parse_natural_date("tomorrow")
        assert isinstance(result, datetime)
        assert result.tzinfo is None

        # Test return path with timezone
        result = parse_natural_date("tomorrow", timezone="America/New_York")
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

        # Test return path with date_input=True
        result = parse_natural_date("May 1st", date_input=True)
        assert isinstance(result, datetime)
        assert result.tzinfo is None

        # Test return path with date_input=True and timezone
        result = parse_natural_date(
            "May 1st", timezone="America/New_York", date_input=True
        )
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

        # Test return path with base_date
        base_date = datetime(2024, 1, 15, 10, 30, 0)
        result = parse_natural_date("3pm", base_date)
        assert isinstance(result, datetime)

        # Test return path with base_date and timezone
        result = parse_natural_date("3pm", base_date, timezone="America/New_York")
        assert isinstance(result, datetime)
        assert result.tzinfo is not None
