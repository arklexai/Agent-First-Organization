from datetime import datetime

from arklex.env.tools.types import ResourceAuthGroup, Transcript


class TestTranscript:
    """Test cases for the Transcript class"""

    def test_transcript_creation(self) -> None:
        """Test creating a Transcript instance with all required parameters"""
        test_id = "test_id_123"
        test_text = "This is a test transcript"
        test_origin = "test_origin"
        test_created_at = datetime(2023, 1, 1, 12, 0, 0)

        transcript = Transcript(test_id, test_text, test_origin, test_created_at)

        assert transcript.id == test_id
        assert transcript.text == test_text
        assert transcript.origin == test_origin
        assert transcript.created_at == test_created_at

    def test_transcript_attributes_types(self) -> None:
        """Test that Transcript attributes have correct types"""
        test_id = "test_id_123"
        test_text = "This is a test transcript"
        test_origin = "test_origin"
        test_created_at = datetime(2023, 1, 1, 12, 0, 0)

        transcript = Transcript(test_id, test_text, test_origin, test_created_at)

        assert isinstance(transcript.id, str)
        assert isinstance(transcript.text, str)
        assert isinstance(transcript.origin, str)
        assert isinstance(transcript.created_at, datetime)

    def test_transcript_with_empty_strings(self) -> None:
        """Test Transcript creation with empty strings"""
        transcript = Transcript("", "", "", datetime(2023, 1, 1))

        assert transcript.id == ""
        assert transcript.text == ""
        assert transcript.origin == ""

    def test_transcript_with_special_characters(self) -> None:
        """Test Transcript creation with special characters"""
        test_id = "test_id_123!@#$%"
        test_text = "This is a test transcript with special chars: !@#$%^&*()"
        test_origin = "test_origin_with_special_chars"
        test_created_at = datetime(2023, 1, 1, 12, 0, 0)

        transcript = Transcript(test_id, test_text, test_origin, test_created_at)

        assert transcript.id == test_id
        assert transcript.text == test_text
        assert transcript.origin == test_origin


class TestResourceAuthGroup:
    """Test cases for the ResourceAuthGroup enum"""

    def test_enum_values(self) -> None:
        """Test that all enum values are correctly defined"""
        assert ResourceAuthGroup.PUBLIC == -1
        assert ResourceAuthGroup.GOOGLE_CALENDAR == 0
        assert ResourceAuthGroup.SHOPIFY == 1
        assert ResourceAuthGroup.HUBSPOT == 2
        assert ResourceAuthGroup.TWILIO == 3
        assert ResourceAuthGroup.SALESFORCE == 4

    def test_enum_names(self) -> None:
        """Test that all enum names are correctly defined"""
        assert ResourceAuthGroup.PUBLIC.name == "PUBLIC"
        assert ResourceAuthGroup.GOOGLE_CALENDAR.name == "GOOGLE_CALENDAR"
        assert ResourceAuthGroup.SHOPIFY.name == "SHOPIFY"
        assert ResourceAuthGroup.HUBSPOT.name == "HUBSPOT"
        assert ResourceAuthGroup.TWILIO.name == "TWILIO"
        assert ResourceAuthGroup.SALESFORCE.name == "SALESFORCE"

    def test_enum_inheritance(self) -> None:
        """Test that ResourceAuthGroup inherits from int and Enum"""
        # Test that it's an int
        assert isinstance(ResourceAuthGroup.PUBLIC, int)
        assert isinstance(ResourceAuthGroup.SHOPIFY, int)

        # Test that it's an Enum
        assert isinstance(ResourceAuthGroup.PUBLIC, ResourceAuthGroup)
        assert isinstance(ResourceAuthGroup.SHOPIFY, ResourceAuthGroup)

    def test_enum_comparison(self) -> None:
        """Test enum comparison operations"""
        assert ResourceAuthGroup.PUBLIC < ResourceAuthGroup.GOOGLE_CALENDAR
        assert ResourceAuthGroup.SHOPIFY > ResourceAuthGroup.GOOGLE_CALENDAR
        assert ResourceAuthGroup.HUBSPOT == ResourceAuthGroup.HUBSPOT
        assert ResourceAuthGroup.TWILIO != ResourceAuthGroup.SALESFORCE

    def test_enum_arithmetic(self) -> None:
        """Test enum arithmetic operations since it inherits from int"""
        assert ResourceAuthGroup.SHOPIFY + ResourceAuthGroup.HUBSPOT == 3
        assert ResourceAuthGroup.SALESFORCE - ResourceAuthGroup.TWILIO == 1
        assert ResourceAuthGroup.GOOGLE_CALENDAR * ResourceAuthGroup.SHOPIFY == 0

    def test_enum_iteration(self) -> None:
        """Test that we can iterate through enum values"""
        values = list(ResourceAuthGroup)
        expected_values = [
            ResourceAuthGroup.PUBLIC,
            ResourceAuthGroup.GOOGLE_CALENDAR,
            ResourceAuthGroup.SHOPIFY,
            ResourceAuthGroup.HUBSPOT,
            ResourceAuthGroup.TWILIO,
            ResourceAuthGroup.SALESFORCE,
        ]
        assert values == expected_values

    def test_enum_docstring(self) -> None:
        """Test that the enum has the expected docstring"""
        assert (
            "when adding new auth group, add it also in the backend sql script"
            in ResourceAuthGroup.__doc__
        )

    def test_enum_value_access(self) -> None:
        """Test accessing enum values through .value attribute"""
        assert ResourceAuthGroup.PUBLIC.value == -1
        assert ResourceAuthGroup.GOOGLE_CALENDAR.value == 0
        assert ResourceAuthGroup.SHOPIFY.value == 1
        assert ResourceAuthGroup.HUBSPOT.value == 2
        assert ResourceAuthGroup.TWILIO.value == 3
        assert ResourceAuthGroup.SALESFORCE.value == 4
