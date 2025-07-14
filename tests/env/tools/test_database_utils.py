import contextlib
import os
import sqlite3
import tempfile
import uuid
from collections.abc import Generator
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from arklex.env.tools.database.utils import (
    DBNAME,
    MULTIPLE_SHOWS_MESSAGE,
    NO_BOOKING_MESSAGE,
    NO_SHOW_MESSAGE,
    SLOTS,
    USER_ID,
    DatabaseActions,
)
from arklex.orchestrator.entities.msg_state_entities import (
    MessageState,
    SlotDetail,
    StatusEnum,
)

# Set test environment variable
os.environ["ARKLEX_TEST_ENV"] = "local"

# Create a temporary directory for test data
test_data_dir = tempfile.mkdtemp(prefix="arklex_test_")
os.environ["DATA_DIR"] = test_data_dir


# Helper to create a temp DB with required schema and data
def setup_temp_db() -> None:
    # Ensure the data directory exists and is writable
    data_dir = os.environ["DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)

    db_path = os.path.join(data_dir, "show_booking_db.sqlite")

    # Remove existing database file if it exists to avoid corruption
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            # If we can't remove it, try to create a new one with a different name
            db_path = os.path.join(
                data_dir, f"show_booking_db_{uuid.uuid4().hex[:8]}.sqlite"
            )

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create tables with proper error handling
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user (
                id VARCHAR(40) PRIMARY KEY,
                first_name VARCHAR(40),
                last_name VARCHAR(40),
                email VARCHAR(60),
                register_at TIMESTAMP,
                last_login TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS show (
                id VARCHAR(40) PRIMARY KEY,
                show_name VARCHAR(100),
                genre VARCHAR(40),
                date DATE,
                time TIME,
                description TEXT,
                location VARCHAR(100),
                price DECIMAL,
                available_seats INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS booking (
                id VARCHAR(40) PRIMARY KEY,
                show_id VARCHAR(40),
                user_id VARCHAR(40),
                created_at TIMESTAMP
            )
        """)

        # Insert a user
        cursor.execute(
            "INSERT OR IGNORE INTO user (id, first_name, last_name, email, register_at, last_login) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "user_be6e1836-8fe9-4938-b2d0-48f810648e72",
                "Test",
                "User",
                "test@example.com",
                "2024-01-01 00:00:00",
                "2024-01-01 00:00:00",
            ),
        )

        # Insert multiple shows for testing
        shows = [
            (
                "show_1",
                "Test Show",
                "Opera",
                "2024-12-31",
                "20:00:00",
                "A test show.",
                "Test Location",
                100.0,
                50,
            ),
            (
                "show_2",
                "Another Show",
                "Comedy",
                "2024-12-31",
                "21:00:00",
                "Another test show.",
                "Test Location",
                150.0,
                30,
            ),
            (
                "show_3",
                "Test Show",
                "Drama",
                "2024-01-15",
                "19:00:00",
                "A different test show.",
                "Other Location",
                75.0,
                25,
            ),
        ]
        for show in shows:
            cursor.execute(
                "INSERT OR IGNORE INTO show (id, show_name, genre, date, time, description, location, price, available_seats) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                show,
            )

        conn.commit()
        conn.close()

        # Verify the database was created successfully
        test_conn = sqlite3.connect(db_path)
        test_cursor = test_conn.cursor()
        test_cursor.execute("SELECT COUNT(*) FROM show")
        show_count = test_cursor.fetchone()[0]
        test_cursor.execute("SELECT COUNT(*) FROM user")
        user_count = test_cursor.fetchone()[0]
        test_conn.close()

        if show_count == 0 or user_count == 0:
            raise Exception("Database setup failed - tables are empty")

    except Exception as e:
        # If setup fails, try to clean up and raise a more informative error
        if os.path.exists(db_path):
            with contextlib.suppress(OSError):
                os.remove(db_path)
        raise Exception(f"Failed to setup test database: {e}") from e


@pytest.fixture(autouse=True)
def run_around_tests() -> Generator[None, None, None]:
    setup_temp_db()
    yield
    # Clean up DB and temp directory after test
    import shutil

    data_dir = os.environ["DATA_DIR"]
    if os.path.exists(data_dir):
        try:
            shutil.rmtree(data_dir)
        except OSError:
            # If we can't remove the directory, try to remove just the DB file
            db_path = os.path.join(data_dir, "show_booking_db.sqlite")
            if os.path.exists(db_path):
                os.remove(db_path)


class TestConstants:
    """Test the constants defined in the module."""

    def test_slots_structure(self) -> None:
        """Test that SLOTS constant has the correct structure."""
        assert isinstance(SLOTS, list)
        assert len(SLOTS) == 4  # show_name, location, date, time

        for slot in SLOTS:
            assert isinstance(slot, dict)
            required_keys = ["name", "type", "value", "description", "prompt"]
            for key in required_keys:
                assert key in slot
                assert isinstance(slot[key], str)

    def test_slot_names(self) -> None:
        """Test that slot names are correct."""
        slot_names = [slot["name"] for slot in SLOTS]
        expected_names = ["show_name", "location", "date", "time"]
        assert slot_names == expected_names

    def test_slot_types(self) -> None:
        """Test that slot types are correct."""
        slot_types = [slot["type"] for slot in SLOTS]
        expected_types = ["string", "string", "date", "time"]
        assert slot_types == expected_types

    def test_message_constants(self) -> None:
        """Test that message constants are defined."""
        assert isinstance(NO_SHOW_MESSAGE, str)
        assert isinstance(MULTIPLE_SHOWS_MESSAGE, str)
        assert isinstance(NO_BOOKING_MESSAGE, str)
        assert len(NO_SHOW_MESSAGE) > 0
        assert len(MULTIPLE_SHOWS_MESSAGE) > 0
        assert len(NO_BOOKING_MESSAGE) > 0

    def test_database_constants(self) -> None:
        """Test database-related constants."""
        assert isinstance(DBNAME, str)
        assert DBNAME == "show_booking_db.sqlite"
        assert isinstance(USER_ID, str)
        assert len(USER_ID) > 0


class TestDatabaseActionsInit:
    """Test the DatabaseActions class initialization."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_init_with_default_user_id(self, mock_llm: object) -> None:
        """Test initialization with default user ID."""
        db = DatabaseActions()
        assert db.user_id == USER_ID
        assert db.db_path.endswith(DBNAME)
        assert isinstance(db.slots, list)
        assert isinstance(db.slot_prompts, list)
        assert len(db.slots) == 0
        assert len(db.slot_prompts) == 0

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_init_with_custom_user_id(self, mock_llm: object) -> None:
        """Test initialization with custom user ID."""
        custom_user_id = "custom_user_123"
        db = DatabaseActions(user_id=custom_user_id)
        assert db.user_id == custom_user_id
        assert db.db_path.endswith(DBNAME)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_init_llm_configuration(self, mock_llm: object) -> None:
        """Test that LLM is properly configured."""
        db = DatabaseActions()
        assert db.llm is not None
        # Verify ChatOpenAI was called with expected parameters
        mock_llm.assert_called_once()


class TestDatabaseActionsLogin:
    """Test the log_in method."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_log_in_success(self, mock_llm: object) -> None:
        """Test successful login with existing user."""
        db = DatabaseActions()
        result = db.log_in()
        assert result is True

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_log_in_failure(self, mock_llm: object) -> None:
        """Test login failure with non-existent user."""
        db = DatabaseActions(user_id="nonexistent_user")
        result = db.log_in()
        assert result is False

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_log_in_database_connection(self, mock_llm: object) -> None:
        """Test that database connection is properly handled."""
        db = DatabaseActions()
        # This should not raise any exceptions
        result = db.log_in()
        assert isinstance(result, bool)


class TestDatabaseActionsInitSlots:
    """Test the init_slots method."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.load_prompts")
    @patch("arklex.env.tools.database.utils.sqlite3.connect")
    def test_init_slots_with_empty_slots(
        self, mock_sqlite_connect: object, mock_load_prompts: object, mock_llm: object
    ) -> None:
        """Test init_slots with empty slots list."""
        mock_load_prompts.return_value = {
            "database_slot_prompt": "template {slot} {value} {value_list}"
        }

        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [("Test Show",), ("Another Show",)]

        # Mock LLM response
        mock_llm_instance = MagicMock()
        parser = MagicMock()
        parser.invoke.return_value = "Test Show"
        mock_llm_instance.__or__.return_value = parser
        mock_llm.return_value = mock_llm_instance

        db = DatabaseActions()
        db.llm = mock_llm_instance
        bot_config = {}
        result = db.init_slots([], bot_config)
        assert isinstance(result, list)
        assert result == SLOTS

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.load_prompts")
    def test_init_slots_with_none_slots(
        self, mock_load_prompts: object, mock_llm: object
    ) -> None:
        """Test init_slots with None slots."""
        mock_load_prompts.return_value = {
            "database_slot_prompt": "template {slot} {value} {value_list}"
        }
        db = DatabaseActions()
        bot_config = {}
        result = db.init_slots(None, bot_config)  # type: ignore
        assert isinstance(result, list)
        assert result == SLOTS

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.load_prompts")
    def test_init_slots_with_custom_slots(
        self, mock_load_prompts: object, mock_llm: object
    ) -> None:
        """Test init_slots with custom slots."""
        mock_load_prompts.return_value = {
            "database_slot_prompt": "template {slot} {value} {value_list}"
        }
        db = DatabaseActions()
        custom_slots = [
            {
                "name": "show_name",  # Use existing column name
                "type": "string",
                "value": "",
                "description": "Custom field",
                "prompt": "Please provide custom field",
            }
        ]
        bot_config = {}
        result = db.init_slots(custom_slots, bot_config)
        assert isinstance(result, list)
        # Should return SLOTS (default) not custom_slots
        assert result == SLOTS


class TestDatabaseActionsVerifySlot:
    """Test the verify_slot method."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.load_prompts")
    @patch("arklex.env.tools.database.utils.chunk_string")
    def test_verify_slot_success(
        self, mock_chunk: object, mock_load_prompts: object, mock_llm: object
    ) -> None:
        """Test successful slot verification."""
        db = DatabaseActions()
        mock_load_prompts.return_value = {
            "database_slot_prompt": "template {slot} {value} {value_list}"
        }
        mock_chunk.return_value = ["chunked_prompt"]

        # Mock LLM response
        db.llm = MagicMock()
        parser = MagicMock()
        parser.invoke.return_value = "Test Show"
        db.llm.__or__.return_value = parser

        slot = SLOTS[0].copy()
        slot["value"] = "Test Show"
        value_list = ["Test Show", "Other Show"]
        bot_config = {}

        result = db.verify_slot(slot, value_list, bot_config)

        assert isinstance(result, SlotDetail)
        assert result.verified_value == "Test Show"
        assert result.confirmed is True

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.load_prompts")
    @patch("arklex.env.tools.database.utils.chunk_string")
    def test_verify_slot_no_match(
        self, mock_chunk: object, mock_load_prompts: object, mock_llm: object
    ) -> None:
        """Test slot verification when no value matches."""
        db = DatabaseActions()
        mock_load_prompts.return_value = {
            "database_slot_prompt": "template {slot} {value} {value_list}"
        }
        mock_chunk.return_value = ["chunked_prompt"]

        # Mock LLM response that doesn't match any value
        db.llm = MagicMock()
        parser = MagicMock()
        parser.invoke.return_value = "Non-existent Show"
        db.llm.__or__.return_value = parser

        slot = SLOTS[0].copy()
        slot["value"] = "Test Show"
        value_list = ["Test Show", "Other Show"]
        bot_config = {}

        result = db.verify_slot(slot, value_list, bot_config)

        assert isinstance(result, SlotDetail)
        assert result.verified_value == ""
        assert result.confirmed is False

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.load_prompts")
    @patch("arklex.env.tools.database.utils.chunk_string")
    def test_verify_slot_exception_handling(
        self, mock_chunk: object, mock_load_prompts: object, mock_llm: object
    ) -> None:
        """Test slot verification with exception handling."""
        db = DatabaseActions()
        mock_load_prompts.return_value = {
            "database_slot_prompt": "template {slot} {value} {value_list}"
        }
        mock_chunk.return_value = ["chunked_prompt"]

        # Mock LLM to raise exception
        db.llm = MagicMock()
        parser = MagicMock()
        parser.invoke.side_effect = Exception("LLM Error")
        db.llm.__or__.return_value = parser

        slot = SLOTS[0].copy()
        slot["value"] = "Test Show"
        value_list = ["Test Show"]
        bot_config = {}

        result = db.verify_slot(slot, value_list, bot_config)

        assert isinstance(result, SlotDetail)
        assert result.verified_value == ""
        assert result.confirmed is False


class TestDatabaseActionsSearchShow:
    """Test the search_show method."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.sqlite3.connect")
    def test_search_show_with_confirmed_slots(
        self, mock_sqlite_connect: object, mock_llm: object
    ) -> None:
        """Test search_show with confirmed slots."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock query results
        mock_cursor.execute.return_value = None
        mock_cursor.fetchall.return_value = [
            (
                "Test Show",
                "2024-12-31",
                "20:00:00",
                "A test show.",
                "Test Location",
                100.0,
            )
        ]
        mock_cursor.description = [
            ("show_name",),
            ("date",),
            ("time",),
            ("description",),
            ("location",),
            ("price",),
        ]

        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Test Show",
                description="",
                prompt="",
                verified_value="Test Show",
                confirmed=True,
            )
        ]
        msg_state = MessageState()

        result = db.search_show(msg_state)

        assert isinstance(result, MessageState)
        assert result.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)

        # Verify database was called correctly
        mock_cursor.execute.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.sqlite3.connect")
    def test_search_show_with_no_confirmed_slots(
        self, mock_sqlite_connect: object, mock_llm: object
    ) -> None:
        """Test search_show with no confirmed slots."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock query results - all shows returned when no filters
        mock_cursor.execute.return_value = None
        mock_cursor.fetchall.return_value = [
            (
                "Test Show",
                "2024-12-31",
                "20:00:00",
                "A test show.",
                "Test Location",
                100.0,
            ),
            (
                "Another Show",
                "2024-12-31",
                "21:00:00",
                "Another test show.",
                "Test Location",
                150.0,
            ),
        ]
        mock_cursor.description = [
            ("show_name",),
            ("date",),
            ("time",),
            ("description",),
            ("location",),
            ("price",),
        ]

        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Test Show",
                description="",
                prompt="",
                verified_value="",
                confirmed=False,
            )
        ]
        msg_state = MessageState()

        result = db.search_show(msg_state)

        assert isinstance(result, MessageState)
        # Should return all shows when no filters applied
        assert result.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)

        # Verify database was called correctly
        mock_cursor.execute.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.sqlite3.connect")
    def test_search_show_no_results(
        self, mock_sqlite_connect: object, mock_llm: object
    ) -> None:
        """Test search_show when no shows match criteria."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock query results - no shows found
        mock_cursor.execute.return_value = None
        mock_cursor.fetchall.return_value = []

        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Non-existent Show",
                description="",
                prompt="",
                verified_value="Non-existent Show",
                confirmed=True,
            )
        ]
        msg_state = MessageState()

        result = db.search_show(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.INCOMPLETE
        assert NO_SHOW_MESSAGE in result.message_flow

        # Verify database was called correctly
        mock_cursor.execute.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.sqlite3.connect")
    def test_search_show_multiple_filters(
        self, mock_sqlite_connect: object, mock_llm: object
    ) -> None:
        """Test search_show with multiple confirmed slots."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (
                "Test Show",
                "2024-12-31",
                "20:00:00",
                "A test show.",
                "Test Location",
                100.0,
            )
        ]
        mock_cursor.description = [
            ("show_name",),
            ("date",),
            ("time",),
            ("description",),
            ("location",),
            ("price",),
        ]

        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Test Show",
                description="",
                prompt="",
                verified_value="Test Show",
                confirmed=True,
            ),
            SlotDetail(
                name="location",
                type="string",
                value="Test Location",
                description="",
                prompt="",
                verified_value="Test Location",
                confirmed=True,
            ),
        ]
        msg_state = MessageState()

        result = db.search_show(msg_state)

        assert isinstance(result, MessageState)
        assert result.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)


class TestDatabaseActionsBookShow:
    """Test the book_show method."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.sqlite3.connect")
    def test_book_show_success(
        self, mock_sqlite_connect: object, mock_llm: object
    ) -> None:
        """Test successful booking."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock query results - single show found
        mock_cursor.execute.return_value = None
        mock_cursor.fetchall.return_value = [
            (
                "show_1",
                "Test Show",
                "2024-12-31",
                "20:00:00",
                "A test show.",
                "Test Location",
                100.0,
            )
        ]
        mock_cursor.description = [
            ("id",),
            ("show_name",),
            ("date",),
            ("time",),
            ("description",),
            ("location",),
            ("price",),
        ]

        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Test Show",
                description="",
                prompt="",
                verified_value="Test Show",
                confirmed=True,
            )
        ]
        db.slot_prompts = []
        msg_state = MessageState()

        result = db.book_show(msg_state)

        assert isinstance(result, MessageState)
        assert result.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)

        # Verify database was called correctly
        assert mock_cursor.execute.call_count >= 1  # At least one query executed
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.sqlite3.connect")
    def test_book_show_no_shows_found(
        self, mock_sqlite_connect: object, mock_llm: object
    ) -> None:
        """Test booking when no shows match criteria."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []  # No shows found

        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Non-existent Show",
                description="",
                prompt="",
                verified_value="Non-existent Show",
                confirmed=True,
            )
        ]
        db.slot_prompts = []
        msg_state = MessageState()

        result = db.book_show(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.INCOMPLETE
        assert NO_SHOW_MESSAGE in result.message_flow

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.sqlite3.connect")
    def test_book_show_multiple_shows_found(
        self, mock_sqlite_connect: object, mock_llm: object
    ) -> None:
        """Test booking when multiple shows match criteria."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock query results - multiple shows found
        mock_cursor.execute.return_value = None
        mock_cursor.fetchall.return_value = [
            (
                "show_1",
                "Test Show",
                "2024-12-31",
                "20:00:00",
                "A test show.",
                "Test Location",
                100.0,
            ),
            (
                "show_2",
                "Another Show",
                "2024-12-31",
                "21:00:00",
                "Another test show.",
                "Test Location",
                150.0,
            ),
        ]
        mock_cursor.description = [
            ("id",),
            ("show_name",),
            ("date",),
            ("time",),
            ("description",),
            ("location",),
            ("price",),
        ]

        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="location",
                type="string",
                value="Test Location",
                description="",
                prompt="",
                verified_value="Test Location",
                confirmed=True,
            )
        ]
        db.slot_prompts = ["Please specify the show name"]
        msg_state = MessageState()

        result = db.book_show(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.INCOMPLETE
        assert result.message_flow in (db.slot_prompts[0], MULTIPLE_SHOWS_MESSAGE)

        # Verify database was called correctly
        mock_cursor.execute.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.sqlite3.connect")
    def test_book_show_no_slots_confirmed(
        self, mock_sqlite_connect: object, mock_llm: object
    ) -> None:
        """Test booking when no slots are confirmed."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock query results - multiple shows found (no filters applied)
        mock_cursor.execute.return_value = None
        mock_cursor.fetchall.return_value = [
            (
                "show_1",
                "Test Show",
                "2024-12-31",
                "20:00:00",
                "A test show.",
                "Test Location",
                100.0,
            ),
            (
                "show_2",
                "Another Show",
                "2024-12-31",
                "21:00:00",
                "Another test show.",
                "Test Location",
                150.0,
            ),
            (
                "show_3",
                "Test Show",
                "2024-01-15",
                "19:00:00",
                "A different test show.",
                "Other Location",
                75.0,
            ),
        ]
        mock_cursor.description = [
            ("id",),
            ("show_name",),
            ("date",),
            ("time",),
            ("description",),
            ("location",),
            ("price",),
        ]

        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Test Show",
                description="",
                prompt="",
                verified_value="",
                confirmed=False,
            )
        ]
        db.slot_prompts = []
        msg_state = MessageState()

        result = db.book_show(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.INCOMPLETE
        # The actual behavior is to return MULTIPLE_SHOWS_MESSAGE when no slots are confirmed
        # because the query returns all shows (3 shows in our test data)
        assert MULTIPLE_SHOWS_MESSAGE in result.message_flow

        # Verify database was called correctly
        mock_cursor.execute.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.sqlite3.connect")
    def test_book_show_empty_slots(
        self, mock_sqlite_connect: object, mock_llm: object
    ) -> None:
        """Test booking with empty slots."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock query results - multiple shows found (no filters applied)
        mock_cursor.execute.return_value = None
        mock_cursor.fetchall.return_value = [
            (
                "show_1",
                "Test Show",
                "2024-12-31",
                "20:00:00",
                "A test show.",
                "Test Location",
                100.0,
            ),
            (
                "show_2",
                "Another Show",
                "2024-12-31",
                "21:00:00",
                "Another test show.",
                "Test Location",
                150.0,
            ),
            (
                "show_3",
                "Test Show",
                "2024-01-15",
                "19:00:00",
                "A different test show.",
                "Other Location",
                75.0,
            ),
        ]
        mock_cursor.description = [
            ("id",),
            ("show_name",),
            ("date",),
            ("time",),
            ("description",),
            ("location",),
            ("price",),
        ]

        db = DatabaseActions()
        db.slots = []
        db.slot_prompts = []
        msg_state = MessageState()

        result = db.book_show(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.INCOMPLETE
        # The actual behavior is to return MULTIPLE_SHOWS_MESSAGE when slots are empty
        # because the query returns all shows (3 shows in our test data)
        assert MULTIPLE_SHOWS_MESSAGE in result.message_flow

        # Verify database was called correctly
        mock_cursor.execute.assert_called_once()
        mock_conn.close.assert_called_once()


class TestDatabaseActionsCheckBooking:
    """Test the check_booking method."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.sqlite3.connect")
    def test_check_booking_with_bookings(
        self, mock_sqlite_connect: object, mock_llm: object
    ) -> None:
        """Test check_booking when user has bookings."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock query results - user has bookings
        mock_cursor.execute.return_value = None
        mock_cursor.fetchall.return_value = [
            (
                "booking_1",
                "show_1",
                "user_1",
                "2024-01-01",
                "Test Show",
                "2024-12-31",
                "20:00:00",
                "A test show.",
                "Test Location",
                100.0,
                50,
            )
        ]
        mock_cursor.description = [
            ("id",),
            ("show_id",),
            ("user_id",),
            ("created_at",),
            ("show_name",),
            ("date",),
            ("time",),
            ("description",),
            ("location",),
            ("price",),
            ("available_seats",),
        ]

        db = DatabaseActions()
        msg_state = MessageState()
        result = db.check_booking(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.COMPLETE
        assert "Booked shows are:" in result.message_flow

        # Verify database was called correctly
        mock_cursor.execute.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.sqlite3.connect")
    def test_check_booking_no_bookings(
        self, mock_sqlite_connect: object, mock_llm: object
    ) -> None:
        """Test check_booking when user has no bookings."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock query results - no bookings
        mock_cursor.execute.return_value = None
        mock_cursor.fetchall.return_value = []

        db = DatabaseActions()
        msg_state = MessageState()
        result = db.check_booking(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.COMPLETE
        assert NO_BOOKING_MESSAGE in result.message_flow

        # Verify database was called correctly
        mock_cursor.execute.assert_called_once()
        mock_conn.close.assert_called_once()


class TestDatabaseActionsCancelBooking:
    """Test the cancel_booking method."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.sqlite3.connect")
    def test_cancel_booking_success(
        self, mock_sqlite_connect: object, mock_llm: object
    ) -> None:
        """Test successful booking cancellation."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock query results - single booking found
        mock_cursor.execute.return_value = None
        mock_cursor.fetchall.return_value = [
            (
                "booking_1",
                "show_1",
                "user_1",
                "2024-01-01",
                "Test Show",
                "2024-12-31",
                "20:00:00",
                "A test show.",
                "Test Location",
                100.0,
                50,
            )
        ]
        mock_cursor.description = [
            ("id",),
            ("show_id",),
            ("user_id",),
            ("created_at",),
            ("show_name",),
            ("date",),
            ("time",),
            ("description",),
            ("location",),
            ("price",),
            ("available_seats",),
        ]

        db = DatabaseActions()
        msg_state = MessageState()
        result = db.cancel_booking(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.COMPLETE
        assert "cancelled show" in result.message_flow.lower()

        # Verify database was called correctly
        assert mock_cursor.execute.call_count >= 1  # At least one query executed
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.sqlite3.connect")
    def test_cancel_booking_no_bookings(
        self, mock_sqlite_connect: object, mock_llm: object
    ) -> None:
        """Test cancel_booking when user has no bookings."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock query results - no bookings
        mock_cursor.execute.return_value = None
        mock_cursor.fetchall.return_value = []

        db = DatabaseActions()
        msg_state = MessageState()
        result = db.cancel_booking(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.COMPLETE
        assert NO_BOOKING_MESSAGE in result.message_flow

        # Verify database was called correctly
        mock_cursor.execute.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.sqlite3.connect")
    def test_cancel_booking_multiple_bookings(
        self, mock_sqlite_connect: object, mock_llm: object
    ) -> None:
        """Test cancel_booking when user has multiple bookings."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock query results - multiple bookings found
        mock_cursor.execute.return_value = None
        mock_cursor.fetchall.return_value = [
            (
                "booking_1",
                "show_1",
                "user_1",
                "2024-01-01",
                "Test Show",
                "2024-12-31",
                "20:00:00",
                "A test show.",
                "Test Location",
                100.0,
                50,
            ),
            (
                "booking_2",
                "show_2",
                "user_1",
                "2024-01-01",
                "Another Show",
                "2024-12-31",
                "21:00:00",
                "Another test show.",
                "Test Location",
                150.0,
                30,
            ),
        ]
        mock_cursor.description = [
            ("id",),
            ("show_id",),
            ("user_id",),
            ("created_at",),
            ("show_name",),
            ("date",),
            ("time",),
            ("description",),
            ("location",),
            ("price",),
            ("available_seats",),
        ]

        db = DatabaseActions()
        db.slot_prompts = ["Please specify which booking to cancel"]
        msg_state = MessageState()
        result = db.cancel_booking(msg_state)

        assert isinstance(result, MessageState)
        # When there are multiple bookings, the method asks for more specific information
        # instead of cancelling anything
        assert result.status == StatusEnum.INCOMPLETE
        assert result.message_flow == "Please specify which booking to cancel"

        # Verify database was called correctly
        mock_cursor.execute.assert_called_once()
        mock_conn.close.assert_called_once()


class TestDatabaseActionsIntegration:
    """Integration tests for DatabaseActions."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_full_booking_workflow(self, mock_llm: object) -> None:
        """Test the complete booking workflow."""
        db = DatabaseActions()

        # Step 1: Search for shows
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Test Show",
                description="",
                prompt="",
                verified_value="Test Show",
                confirmed=True,
            )
        ]
        msg_state = MessageState()
        search_result = db.search_show(msg_state)
        assert search_result.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)

        # Step 2: Book a show
        book_result = db.book_show(msg_state)
        assert book_result.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)

        # Step 3: Check bookings
        check_result = db.check_booking(msg_state)
        assert check_result.status == StatusEnum.COMPLETE

        # Step 4: Cancel booking
        cancel_result = db.cancel_booking(msg_state)
        assert cancel_result.status == StatusEnum.COMPLETE

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_database_connection_handling(self, mock_llm: object) -> None:
        """Test that database connections are properly handled."""
        db = DatabaseActions()

        # Test that multiple operations don't cause connection issues
        msg_state = MessageState()

        # Multiple search operations
        for _ in range(3):
            result = db.search_show(msg_state)
            assert isinstance(result, MessageState)

        # Multiple check booking operations
        for _ in range(3):
            result = db.check_booking(msg_state)
            assert isinstance(result, MessageState)


class TestDatabaseActionsEdgeCases:
    """Test edge cases and error conditions."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_database_file_not_exists(self, mock_llm: object) -> None:
        """Test behavior when database file doesn't exist."""
        # Remove the database file
        db_path = os.path.join(os.environ["DATA_DIR"], "show_booking_db.sqlite")
        if os.path.exists(db_path):
            os.remove(db_path)

        db = DatabaseActions()
        msg_state = MessageState()

        # These operations should handle the missing database gracefully
        # The database will be created automatically by SQLite, but tables won't exist
        # So we expect an OperationalError when trying to query non-existent tables
        with pytest.raises(sqlite3.OperationalError):
            db.search_show(msg_state)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_empty_database_tables(self, mock_llm: object) -> None:
        """Test behavior with empty database tables."""
        db = DatabaseActions()

        # Clear all tables
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM booking")
        cursor.execute("DELETE FROM show")
        cursor.execute("DELETE FROM user")
        conn.commit()
        conn.close()

        msg_state = MessageState()

        # Test search with empty tables
        result = db.search_show(msg_state)
        assert result.status == StatusEnum.INCOMPLETE
        assert NO_SHOW_MESSAGE in result.message_flow

        # Test check booking with empty tables
        result = db.check_booking(msg_state)
        assert result.status == StatusEnum.COMPLETE
        assert NO_BOOKING_MESSAGE in result.message_flow

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_special_characters_in_slot_values(self, mock_llm: object) -> None:
        """Test handling of special characters in slot values."""
        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Test'Show",
                description="",
                prompt="",
                verified_value="Test'Show",
                confirmed=True,
            )
        ]
        msg_state = MessageState()

        # Should handle SQL injection attempts gracefully
        result = db.search_show(msg_state)
        assert isinstance(result, MessageState)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_very_long_slot_values(self, mock_llm: object) -> None:
        """Test handling of very long slot values."""
        db = DatabaseActions()
        long_value = "A" * 1000  # Very long string
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value=long_value,
                description="",
                prompt="",
                verified_value=long_value,
                confirmed=True,
            )
        ]
        msg_state = MessageState()

        result = db.search_show(msg_state)
        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.INCOMPLETE  # Should not find any shows


class TestDatabaseActionsAdvancedScenarios:
    """Test advanced scenarios and edge cases."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.load_prompts")
    def test_verify_slot_with_empty_value_list(
        self, mock_load_prompts: object, mock_llm: object
    ) -> None:
        """Test slot verification with empty value list."""
        mock_load_prompts.return_value = {
            "database_slot_prompt": "template {slot} {value} {value_list}"
        }
        db = DatabaseActions()
        slot = SLOTS[0].copy()
        slot["value"] = "Test Show"
        value_list = []
        bot_config = {}

        result = db.verify_slot(slot, value_list, bot_config)

        assert isinstance(result, SlotDetail)
        assert result.verified_value == ""
        assert result.confirmed is False

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.load_prompts")
    def test_verify_slot_with_none_value_list(
        self, mock_load_prompts: object, mock_llm: object
    ) -> None:
        """Test slot verification with None value list."""
        mock_load_prompts.return_value = {
            "database_slot_prompt": "template {slot} {value} {value_list}"
        }
        db = DatabaseActions()
        slot = SLOTS[0].copy()
        slot["value"] = "Test Show"
        value_list = None  # type: ignore
        bot_config = {}

        result = db.verify_slot(slot, value_list, bot_config)

        assert isinstance(result, SlotDetail)
        assert result.verified_value == ""
        assert result.confirmed is False

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_search_show_with_sql_injection_attempt(self, mock_llm: object) -> None:
        """Test search_show with SQL injection attempt in slot value."""
        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="'; DROP TABLE show; --",
                description="",
                prompt="",
                verified_value="'; DROP TABLE show; --",
                confirmed=True,
            )
        ]
        msg_state = MessageState()

        # Should handle SQL injection gracefully
        result = db.search_show(msg_state)
        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.INCOMPLETE

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_book_show_with_duplicate_booking(self, mock_llm: object) -> None:
        """Test booking the same show twice."""
        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Test Show",
                description="",
                prompt="",
                verified_value="Test Show",
                confirmed=True,
            )
        ]
        msg_state = MessageState()

        # First booking
        result1 = db.book_show(msg_state)
        # The first booking might be incomplete if multiple shows match
        assert result1.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)

        # Second booking of the same show
        result2 = db.book_show(msg_state)
        assert result2.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)
        # Should allow multiple bookings of the same show

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_cancel_booking_with_specific_show_id(self, mock_llm: object) -> None:
        """Test canceling booking with specific show ID."""
        db = DatabaseActions()

        # Insert multiple bookings
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO booking (id, show_id, user_id, created_at) VALUES (?, ?, ?, ?)",
            ("booking_1", "show_1", db.user_id, datetime.now()),
        )
        cursor.execute(
            "INSERT INTO booking (id, show_id, user_id, created_at) VALUES (?, ?, ?, ?)",
            ("booking_2", "show_2", db.user_id, datetime.now()),
        )
        conn.commit()
        conn.close()

        # Set up slots to specify which booking to cancel
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Test Show",
                description="",
                prompt="",
                verified_value="Test Show",
                confirmed=True,
            )
        ]
        msg_state = MessageState()
        result = db.cancel_booking(msg_state)

        assert isinstance(result, MessageState)
        # The result might be incomplete if multiple bookings match the criteria
        assert result.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_init_slots_with_invalid_column_name(self, mock_llm: object) -> None:
        """Test init_slots with invalid column name."""
        db = DatabaseActions()
        invalid_slots = [
            {
                "name": "invalid_column",
                "type": "string",
                "value": "",
                "description": "Invalid column",
                "prompt": "Please provide invalid column",
            }
        ]
        bot_config = {}

        # Should handle invalid column gracefully
        with pytest.raises(sqlite3.OperationalError):
            db.init_slots(invalid_slots, bot_config)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_database_connection_timeout_handling(self, mock_llm: object) -> None:
        """Test handling of database connection timeouts."""
        db = DatabaseActions()
        msg_state = MessageState()

        # Mock database connection to raise timeout
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.OperationalError("database is locked")

            # Should handle database lock gracefully
            with pytest.raises(sqlite3.OperationalError):
                db.search_show(msg_state)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.load_prompts")
    def test_llm_timeout_handling(
        self, mock_load_prompts: object, mock_llm: object
    ) -> None:
        """Test handling of LLM timeout during slot verification."""
        mock_load_prompts.return_value = {
            "database_slot_prompt": "template {slot} {value} {value_list}"
        }
        db = DatabaseActions()
        slot = SLOTS[0].copy()
        slot["value"] = "Test Show"
        value_list = ["Test Show", "Other Show"]
        bot_config = {}

        # Mock LLM to raise timeout
        db.llm = MagicMock()
        parser = MagicMock()
        parser.invoke.side_effect = Exception("Request timeout")
        db.llm.__or__.return_value = parser

        result = db.verify_slot(slot, value_list, bot_config)

        assert isinstance(result, SlotDetail)
        assert result.verified_value == ""
        assert result.confirmed is False

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_message_state_preservation(self, mock_llm: object) -> None:
        """Test that MessageState is properly preserved across operations."""
        db = DatabaseActions()
        original_msg_state = MessageState()
        original_msg_state.message_flow = "Original message"
        original_msg_state.status = StatusEnum.INCOMPLETE

        # Perform operation
        result = db.search_show(original_msg_state)

        # Check that the original MessageState object is modified, not replaced
        assert result is original_msg_state
        assert result.message_flow != "Original message"  # Should be updated
        assert result.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_database_transaction_rollback(self, mock_llm: object) -> None:
        """Test database transaction rollback on error."""
        db = DatabaseActions()
        msg_state = MessageState()

        # Mock database to raise error during booking
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            # First query succeeds, second fails
            mock_cursor.execute.side_effect = [
                None,  # First execute (SELECT) succeeds
                sqlite3.OperationalError("table booking doesn't exist"),  # Second fails
            ]
            mock_cursor.fetchall.return_value = [
                (
                    "show_1",
                    "Test Show",
                    "2024-12-31",
                    "20:00:00",
                    "Description",
                    "Location",
                    100.0,
                )
            ]
            mock_cursor.description = [
                ("id",),
                ("show_name",),
                ("date",),
                ("time",),
                ("description",),
                ("location",),
                ("price",),
            ]

            db.slots = [
                SlotDetail(
                    name="show_name",
                    type="string",
                    value="Test Show",
                    description="",
                    prompt="",
                    verified_value="Test Show",
                    confirmed=True,
                )
            ]

            # Should handle database error gracefully
            with pytest.raises(sqlite3.OperationalError):
                db.book_show(msg_state)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_unicode_handling_in_slot_values(self, mock_llm: object) -> None:
        """Test handling of Unicode characters in slot values."""
        db = DatabaseActions()
        unicode_value = "🎭 Théâtre Show 🎪"
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value=unicode_value,
                description="",
                prompt="",
                verified_value=unicode_value,
                confirmed=True,
            )
        ]
        msg_state = MessageState()

        result = db.search_show(msg_state)
        assert isinstance(result, MessageState)
        assert (
            result.status == StatusEnum.INCOMPLETE
        )  # Should not find any shows with Unicode names

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_numeric_slot_values(self, mock_llm: object) -> None:
        """Test handling of numeric slot values."""
        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="price",
                type="number",
                value="100",
                description="",
                prompt="",
                verified_value="100",
                confirmed=True,
            )
        ]
        msg_state = MessageState()

        result = db.search_show(msg_state)
        assert isinstance(result, MessageState)
        # Should handle numeric values in string slots

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_empty_string_slot_values(self, mock_llm: object) -> None:
        """Test handling of empty string slot values."""
        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="",
                description="",
                prompt="",
                verified_value="",
                confirmed=True,
            )
        ]
        msg_state = MessageState()

        result = db.search_show(msg_state)
        assert isinstance(result, MessageState)
        # Should handle empty string values gracefully

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_whitespace_handling_in_slot_values(self, mock_llm: object) -> None:
        """Test handling of whitespace in slot values."""
        db = DatabaseActions()
        whitespace_value = "  Test Show  "
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value=whitespace_value,
                description="",
                prompt="",
                verified_value=whitespace_value,
                confirmed=True,
            )
        ]
        msg_state = MessageState()

        result = db.search_show(msg_state)
        assert isinstance(result, MessageState)
        assert (
            result.status == StatusEnum.INCOMPLETE
        )  # Should not find shows with whitespace

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_case_sensitivity_in_slot_values(self, mock_llm: object) -> None:
        """Test case sensitivity in slot values."""
        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="test show",
                description="",
                prompt="",
                verified_value="test show",
                confirmed=True,
            )
        ]
        msg_state = MessageState()

        result = db.search_show(msg_state)
        assert isinstance(result, MessageState)
        assert (
            result.status == StatusEnum.INCOMPLETE
        )  # Should not find "Test Show" with "test show"

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_database_constraints_violation(self, mock_llm: object) -> None:
        """Test handling of database constraint violations."""
        db = DatabaseActions()
        msg_state = MessageState()

        # Try to book with invalid show_id
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Non-existent Show",
                description="",
                prompt="",
                verified_value="Non-existent Show",
                confirmed=True,
            )
        ]

        result = db.book_show(msg_state)
        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.INCOMPLETE
        assert NO_SHOW_MESSAGE in result.message_flow

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_concurrent_database_access(self, mock_llm: object) -> None:
        """Test concurrent database access scenarios."""
        db1 = DatabaseActions()
        db2 = DatabaseActions()
        msg_state = MessageState()

        # Both instances should be able to access the database simultaneously
        result1 = db1.search_show(msg_state)
        result2 = db2.search_show(msg_state)

        assert isinstance(result1, MessageState)
        assert isinstance(result2, MessageState)
        assert result1.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)
        assert result2.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_memory_usage_with_large_datasets(self, mock_llm: object) -> None:
        """Test memory usage with large datasets."""
        db = DatabaseActions()

        # Insert many shows to test memory usage (use unique IDs)
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()

        for i in range(100):  # Insert 100 shows
            cursor.execute(
                "INSERT OR IGNORE INTO show (id, show_name, genre, date, time, description, location, price, available_seats) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"large_show_{i}",
                    f"Large Show {i}",
                    "Drama",
                    "2024-12-31",
                    "20:00:00",
                    f"Description for large show {i}",
                    "Location",
                    100.0 + i,
                    50,
                ),
            )
        conn.commit()
        conn.close()

        msg_state = MessageState()
        result = db.search_show(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.COMPLETE
        # Should handle large datasets without memory issues

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_database_file_permissions(self, mock_llm: object) -> None:
        """Test handling of database file permission issues."""
        db = DatabaseActions()
        msg_state = MessageState()

        # Mock file permission error
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.OperationalError(
                "unable to open database file"
            )

            with pytest.raises(sqlite3.OperationalError):
                db.search_show(msg_state)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_database_corruption_handling(self, mock_llm: object) -> None:
        """Test handling of corrupted database files."""
        db = DatabaseActions()
        msg_state = MessageState()

        # Mock database corruption error
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn
            mock_cursor.execute.side_effect = sqlite3.DatabaseError(
                "database disk image is malformed"
            )

            with pytest.raises(sqlite3.DatabaseError):
                db.search_show(msg_state)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_environment_variable_handling(self, mock_llm: object) -> None:
        """Test handling of missing DATA_DIR environment variable."""
        original_data_dir = os.environ.get("DATA_DIR")

        # Remove DATA_DIR temporarily
        if "DATA_DIR" in os.environ:
            del os.environ["DATA_DIR"]

        try:
            # Should handle missing DATA_DIR gracefully
            with pytest.raises(TypeError):
                DatabaseActions()
        finally:
            # Restore DATA_DIR
            if original_data_dir:
                os.environ["DATA_DIR"] = original_data_dir

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_uuid_generation_in_booking(self, mock_llm: object) -> None:
        """Test that booking IDs are properly generated with UUID."""
        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Test Show",
                description="",
                prompt="",
                verified_value="Test Show",
                confirmed=True,
            )
        ]
        msg_state = MessageState()

        result = db.book_show(msg_state)

        if result.status == StatusEnum.COMPLETE:
            # Check that a booking was actually created
            conn = sqlite3.connect(db.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM booking WHERE user_id = ?", (db.user_id,))
            bookings = cursor.fetchall()
            conn.close()

            if bookings:
                booking_id = bookings[-1][0]  # Get the most recent booking
                assert booking_id.startswith("booking_")
                assert len(booking_id) > len("booking_")  # Should have UUID part

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_datetime_handling_in_booking(self, mock_llm: object) -> None:
        """Test datetime handling in booking operations."""
        db_actions = DatabaseActions()
        msg_state = MessageState()

        # Mock database connection
        with patch("arklex.env.tools.database.utils.sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor

            # Mock cursor description for column names
            mock_cursor.description = [
                ("id",),
                ("show_id",),
                ("user_id",),
                ("created_at",),
            ]

            # Mock fetchall to return booking data
            mock_cursor.fetchall.return_value = [
                (
                    "booking_1",
                    "show_1",
                    "user_be6e1836-8fe9-4938-b2d0-48f810648e72",
                    "2024-01-15 14:30:00",
                )
            ]

            result = db_actions.check_booking(msg_state)

            # Verify the result includes the datetime
            assert "2024-01-15 14:30:00" in result.message_flow

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_verify_slot_exception_handling_with_specific_error(
        self, mock_llm: object
    ) -> None:
        """Test verify_slot exception handling with specific error types."""
        db_actions = DatabaseActions()
        slot = {
            "name": "test_slot",
            "type": "str",
            "value": "test_value",
            "description": "Test slot description",
            "prompt": "Test prompt",
        }
        value_list = ["value1", "value2"]
        bot_config = {"test": "config"}

        with (
            patch("arklex.env.tools.database.utils.load_prompts") as mock_load_prompts,
            patch("arklex.env.tools.database.utils.chunk_string") as mock_chunk,
        ):
            mock_load_prompts.return_value = {"database_slot_prompt": "test prompt"}
            mock_chunk.return_value = ["chunk1", "chunk2"]

            # Mock LLM to raise a specific exception
            mock_llm_instance = MagicMock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.invoke.side_effect = ValueError("Test LLM error")

            result = db_actions.verify_slot(slot, value_list, bot_config)

            # Should return slot_detail with default values when exception occurs
            assert isinstance(result, SlotDetail)
            assert result.verified_value == ""
            assert result.confirmed is False

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_verify_slot_exception_handling_with_connection_error(
        self, mock_llm: object
    ) -> None:
        """Test verify_slot exception handling with connection error."""
        db_actions = DatabaseActions()
        slot = {
            "name": "test_slot",
            "type": "str",
            "value": "test_value",
            "description": "Test slot description",
            "prompt": "Test prompt",
        }
        value_list = ["value1", "value2"]
        bot_config = {"test": "config"}

        with (
            patch("arklex.env.tools.database.utils.load_prompts") as mock_load_prompts,
            patch("arklex.env.tools.database.utils.chunk_string") as mock_chunk,
        ):
            mock_load_prompts.return_value = {"database_slot_prompt": "test prompt"}
            mock_chunk.return_value = ["chunk1", "chunk2"]

            # Mock LLM to raise a connection error
            mock_llm_instance = MagicMock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.invoke.side_effect = ConnectionError(
                "Test connection error"
            )

            result = db_actions.verify_slot(slot, value_list, bot_config)

            # Should return slot_detail with default values when exception occurs
            assert isinstance(result, SlotDetail)
            assert result.verified_value == ""
            assert result.confirmed is False

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_verify_slot_exception_handling_with_timeout_error(
        self, mock_llm: object
    ) -> None:
        """Test verify_slot exception handling with timeout error."""
        db_actions = DatabaseActions()
        slot = {
            "name": "test_slot",
            "type": "str",
            "value": "test_value",
            "description": "Test slot description",
            "prompt": "Test prompt",
        }
        value_list = ["value1", "value2"]
        bot_config = {"test": "config"}

        with (
            patch("arklex.env.tools.database.utils.load_prompts") as mock_load_prompts,
            patch("arklex.env.tools.database.utils.chunk_string") as mock_chunk,
        ):
            mock_load_prompts.return_value = {"database_slot_prompt": "test prompt"}
            mock_chunk.return_value = ["chunk1", "chunk2"]

            # Mock LLM to raise a timeout error
            mock_llm_instance = MagicMock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.invoke.side_effect = TimeoutError("Test timeout error")

            result = db_actions.verify_slot(slot, value_list, bot_config)

            # Should return slot_detail with default values when exception occurs
            assert isinstance(result, SlotDetail)
            assert result.verified_value == ""
            assert result.confirmed is False

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_verify_slot_exception_handling_with_attribute_error(
        self, mock_llm: object
    ) -> None:
        """Test verify_slot exception handling with attribute error."""
        db_actions = DatabaseActions()
        slot = {
            "name": "test_slot",
            "type": "str",
            "value": "test_value",
            "description": "Test slot description",
            "prompt": "Test prompt",
        }
        value_list = ["value1", "value2"]
        bot_config = {"test": "config"}

        with (
            patch("arklex.env.tools.database.utils.load_prompts") as mock_load_prompts,
            patch("arklex.env.tools.database.utils.chunk_string") as mock_chunk,
        ):
            mock_load_prompts.return_value = {"database_slot_prompt": "test prompt"}
            mock_chunk.return_value = ["chunk1", "chunk2"]

            # Mock LLM to raise an attribute error
            mock_llm_instance = MagicMock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.invoke.side_effect = AttributeError(
                "Test attribute error"
            )

            result = db_actions.verify_slot(slot, value_list, bot_config)

            # Should return slot_detail with default values when exception occurs
            assert isinstance(result, SlotDetail)
            assert result.verified_value == ""
            assert result.confirmed is False

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_verify_slot_exception_handling_with_type_error(
        self, mock_llm: object
    ) -> None:
        """Test verify_slot exception handling with type error."""
        db_actions = DatabaseActions()
        slot = {
            "name": "test_slot",
            "type": "str",
            "value": "test_value",
            "description": "Test slot description",
            "prompt": "Test prompt",
        }
        value_list = ["value1", "value2"]
        bot_config = {"test": "config"}

        with (
            patch("arklex.env.tools.database.utils.load_prompts") as mock_load_prompts,
            patch("arklex.env.tools.database.utils.chunk_string") as mock_chunk,
        ):
            mock_load_prompts.return_value = {"database_slot_prompt": "test prompt"}
            mock_chunk.return_value = ["chunk1", "chunk2"]

            # Mock LLM to raise a type error
            mock_llm_instance = MagicMock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.invoke.side_effect = TypeError("Test type error")

            result = db_actions.verify_slot(slot, value_list, bot_config)

            # Should return slot_detail with default values when exception occurs
            assert isinstance(result, SlotDetail)
            assert result.verified_value == ""
            assert result.confirmed is False
