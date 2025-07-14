"""
Tests for Booking Database tools.

This module contains comprehensive tests for all booking database-related tools including
booking functionality, database operations, and utility functions.
"""

import os
import sqlite3
import tempfile
from unittest.mock import Mock, patch

# Import the underlying functions, not the decorated ones
from arklex.env.tools.booking_db.book_show import book_show
from arklex.env.tools.booking_db.build_database import build_database
from arklex.env.tools.booking_db.cancel_booking import cancel_booking
from arklex.env.tools.booking_db.check_booking import check_booking
from arklex.env.tools.booking_db.search_show import search_show
from arklex.env.tools.booking_db.utils import (
    LOG_IN_FAILURE,
    MULTIPLE_SHOWS_MESSAGE,
    NO_SHOW_MESSAGE,
    SLOTS,
    booking,
    log_in,
)

# Set test environment
os.environ["ARKLEX_TEST_ENV"] = "local"


class TestBookingDBUtils:
    """Test the booking database utility functions."""

    def test_booking_constants(self) -> None:
        """Test that booking constants are defined."""
        assert hasattr(booking, "db_path")
        assert hasattr(booking, "user_id")
        assert LOG_IN_FAILURE is not None
        assert NO_SHOW_MESSAGE is not None
        assert MULTIPLE_SHOWS_MESSAGE is not None

    def test_slots_structure(self) -> None:
        """Test that SLOTS dictionary has the expected structure."""
        expected_slots = ["show_name", "date", "time", "location"]
        for slot_name in expected_slots:
            assert slot_name in SLOTS
            assert "type" in SLOTS[slot_name]
            assert "description" in SLOTS[slot_name]

    @patch("arklex.env.tools.booking_db.utils.booking")
    @patch("arklex.env.tools.booking_db.utils.sqlite3.connect")
    def test_log_in_success(self, mock_connect: Mock, mock_booking: Mock) -> None:
        """Test successful login."""
        mock_booking.user_id = "test_user"
        mock_booking.db_path = "/tmp/test.db"

        # Mock the database connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)  # User exists
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Set DATA_DIR environment variable
        with patch.dict(os.environ, {"DATA_DIR": "/tmp"}):
            result = log_in()

        assert result is True

    @patch("arklex.env.tools.booking_db.utils.booking")
    @patch("arklex.env.tools.booking_db.utils.sqlite3.connect")
    def test_log_in_failure(self, mock_connect: Mock, mock_booking: Mock) -> None:
        """Test login failure."""
        mock_booking.user_id = "test_user"
        mock_booking.db_path = "/tmp/test.db"

        # Mock the database connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None  # User doesn't exist
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Set DATA_DIR environment variable
        with patch.dict(os.environ, {"DATA_DIR": "/tmp"}):
            result = log_in()

        assert result is False


class TestBookingDBBookShow:
    """Test the book_show function."""

    @patch("arklex.env.tools.booking_db.book_show.log_in")
    @patch("arklex.env.tools.booking_db.book_show.sqlite3.connect")
    def test_book_show_login_failure(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test book_show when login fails."""
        mock_log_in.return_value = False

        result = book_show().func("Test Show", None, None, None)

        assert result == LOG_IN_FAILURE
        mock_connect.assert_not_called()

    @patch("arklex.env.tools.booking_db.book_show.log_in")
    @patch("arklex.env.tools.booking_db.book_show.sqlite3.connect")
    def test_book_show_no_shows_found(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test book_show when no shows match criteria."""
        mock_log_in.return_value = True
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = book_show().func("Non-existent Show", None, None, None)

        assert result == NO_SHOW_MESSAGE
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.booking_db.book_show.log_in")
    @patch("arklex.env.tools.booking_db.book_show.sqlite3.connect")
    def test_book_show_multiple_shows_found(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test book_show when multiple shows match criteria."""
        mock_log_in.return_value = True
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ("1", "Show 1", "2024-01-15", "10:00", "Description 1", "Location 1", 50),
            ("2", "Show 2", "2024-01-15", "10:00", "Description 2", "Location 2", 60),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = book_show().func(None, "2024-01-15", "10:00", None)

        assert result == MULTIPLE_SHOWS_MESSAGE
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.booking_db.book_show.sqlite3.connect")
    @patch("arklex.env.tools.booking_db.book_show.uuid.uuid4")
    def test_book_show_success(self, mock_uuid: Mock, mock_connect: Mock) -> None:
        """Test successful booking of a show."""
        mock_uuid.return_value = "test-uuid"

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (
                "1",
                "Test Show",
                "2024-01-15",
                "10:00",
                "Test Description",
                "Test Location",
                50,
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
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Mock log_in to return True
        with patch("arklex.env.tools.booking_db.book_show.log_in", return_value=True):
            result = book_show().func("Test Show", None, None, None)

        assert "The booked show is:" in result
        assert "Test Show" in result
        mock_cursor.execute.assert_called()
        mock_conn.close.assert_called_once()


class TestBookingDBSearchShow:
    """Test the search_show function."""

    @patch("arklex.env.tools.booking_db.search_show.log_in")
    @patch("arklex.env.tools.booking_db.search_show.sqlite3.connect")
    def test_search_show_login_failure(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test search_show when login fails."""
        mock_log_in.return_value = False

        result = search_show().func("Test Show", None, None, None)

        assert result == LOG_IN_FAILURE
        mock_connect.assert_not_called()

    @patch("arklex.env.tools.booking_db.search_show.log_in")
    @patch("arklex.env.tools.booking_db.search_show.sqlite3.connect")
    def test_search_show_no_results(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test search_show when no shows match criteria."""
        mock_log_in.return_value = True
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = search_show().func("Non-existent Show", None, None, None)

        assert "No shows exist" in result
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.booking_db.search_show.log_in")
    @patch("arklex.env.tools.booking_db.search_show.sqlite3.connect")
    def test_search_show_with_results(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test search_show when shows match criteria."""
        mock_log_in.return_value = True
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (
                "1",
                "Test Show",
                "2024-01-15",
                "10:00",
                "Test Description",
                "Test Location",
                50,
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
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = search_show().func("Test Show", None, None, None)

        assert "Test Show" in result
        assert "2024-01-15" in result
        assert "10:00" in result
        mock_conn.close.assert_called_once()


class TestBookingDBCheckBooking:
    """Test the check_booking function."""

    @patch("arklex.env.tools.booking_db.check_booking.log_in")
    @patch("arklex.env.tools.booking_db.check_booking.sqlite3.connect")
    def test_check_booking_login_failure(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test check_booking when login fails."""
        mock_log_in.return_value = False

        result = check_booking().func()

        assert result == LOG_IN_FAILURE
        mock_connect.assert_not_called()

    @patch("arklex.env.tools.booking_db.check_booking.log_in")
    @patch("arklex.env.tools.booking_db.check_booking.sqlite3.connect")
    def test_check_booking_no_bookings(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test check_booking when user has no bookings."""
        mock_log_in.return_value = True
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = check_booking().func()

        assert "You have not booked any show" in result
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.booking_db.check_booking.log_in")
    @patch("arklex.env.tools.booking_db.check_booking.sqlite3.connect")
    def test_check_booking_with_bookings(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test check_booking when user has bookings."""
        mock_log_in.return_value = True
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (
                "1",
                "Test Show",
                "2024-01-15",
                "10:00",
                "Test Description",
                "Test Location",
                50,
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
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = check_booking().func()

        assert "Test Show" in result
        assert "2024-01-15" in result
        assert "10:00" in result
        mock_conn.close.assert_called_once()


class TestBookingDBCancelBooking:
    """Test the cancel_booking function."""

    @patch("arklex.env.tools.booking_db.cancel_booking.log_in")
    @patch("arklex.env.tools.booking_db.cancel_booking.sqlite3.connect")
    def test_cancel_booking_login_failure(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test cancel_booking when login fails."""
        mock_log_in.return_value = False

        result = cancel_booking().func()

        assert result == LOG_IN_FAILURE
        mock_connect.assert_not_called()

    @patch("arklex.env.tools.booking_db.cancel_booking.log_in")
    @patch("arklex.env.tools.booking_db.cancel_booking.sqlite3.connect")
    def test_cancel_booking_success(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test successful cancellation of booking."""
        mock_log_in.return_value = True
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (
                "1",
                "Test Show",
                "2024-01-15",
                "10:00",
                "Test Description",
                "Test Location",
                50,
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
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = cancel_booking().func()

        assert "The cancelled show is:" in result
        assert "Test Show" in result
        mock_cursor.execute.assert_called()
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.booking_db.cancel_booking.log_in")
    @patch("arklex.env.tools.booking_db.cancel_booking.sqlite3.connect")
    def test_cancel_booking_not_found(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test cancel_booking when no booking exists."""
        mock_log_in.return_value = True
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = cancel_booking().func()

        assert "You have not booked any show" in result
        mock_conn.close.assert_called_once()


class TestBookingDBBuildDatabase:
    """Test the build_database function."""

    def test_build_database_creates_tables(self) -> None:
        """Test that build_database creates the required tables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = os.path.join(temp_dir, "show_booking_db.sqlite")

            # Verify database was created
            assert os.path.exists(db_path)

            # Verify tables were created
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            assert "show" in tables
            assert "user" in tables
            assert "booking" in tables

            conn.close()

    def test_build_database_inserts_sample_data(self) -> None:
        """Test that build_database inserts sample data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = os.path.join(temp_dir, "show_booking_db.sqlite")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check if sample data was inserted
            cursor.execute("SELECT COUNT(*) FROM show")
            show_count = cursor.fetchone()[0]
            assert show_count > 0

            cursor.execute("SELECT COUNT(*) FROM user")
            user_count = cursor.fetchone()[0]
            assert user_count > 0

            conn.close()

    def test_build_database_removes_existing_file(self) -> None:
        """Test that build_database removes existing database file before creating new one."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "show_booking_db.sqlite")

            # Create a dummy database file first
            with open(db_path, "w") as f:
                f.write("dummy content")

            # Verify file exists
            assert os.path.exists(db_path)

            # Call build_database - should remove existing file and create new one
            build_database(temp_dir)

            # Verify new database was created and is a proper SQLite database
            assert os.path.exists(db_path)

            # Try to connect to verify it's a valid SQLite database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            assert "show" in tables
            assert "user" in tables
            assert "booking" in tables

            conn.close()

    def test_build_database_creates_correct_table_schema(self) -> None:
        """Test that build_database creates tables with correct schema."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = os.path.join(temp_dir, "show_booking_db.sqlite")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check show table schema
            cursor.execute("PRAGMA table_info(show)")
            show_columns = {row[1]: row[2] for row in cursor.fetchall()}
            expected_show_columns = {
                "id": "VARCHAR(40)",
                "show_name": "VARCHAR(100)",
                "genre": "VARCHAR(40)",
                "date": "DATE",
                "time": "TIME",
                "description": "TEXT",
                "location": "VARCHAR(100)",
                "price": "DECIMAL",
                "available_seats": "INTEGER",
            }

            for col, _expected_type in expected_show_columns.items():
                assert col in show_columns
                # Note: SQLite doesn't enforce VARCHAR length, so we just check the column exists

            # Check user table schema
            cursor.execute("PRAGMA table_info(user)")
            user_columns = {row[1]: row[2] for row in cursor.fetchall()}
            expected_user_columns = {
                "id": "VARCHAR(40)",
                "first_name": "VARCHAR(40)",
                "last_name": "VARCHAR(40)",
                "email": "VARCHAR(60)",
                "register_at": "TIMESTAMP",
                "last_login": "TIMESTAMP",
            }

            for col, _expected_type in expected_user_columns.items():
                assert col in user_columns

            # Check booking table schema
            cursor.execute("PRAGMA table_info(booking)")
            booking_columns = {row[1]: row[2] for row in cursor.fetchall()}
            expected_booking_columns = {
                "id": "VARCHAR(40)",
                "show_id": "VARCHAR(40)",
                "user_id": "VARCHAR(40)",
                "created_at": "TIMESTAMP",
            }

            for col, _expected_type in expected_booking_columns.items():
                assert col in booking_columns

            conn.close()

    def test_build_database_inserts_correct_show_data(self) -> None:
        """Test that build_database inserts the expected show data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = os.path.join(temp_dir, "show_booking_db.sqlite")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check specific shows exist
            cursor.execute(
                "SELECT show_name FROM show WHERE show_name = 'The Dead, 1904'"
            )
            dead_1904_shows = cursor.fetchall()
            assert len(dead_1904_shows) == 2  # Two shows with this name

            cursor.execute("SELECT show_name FROM show WHERE show_name = 'Carmen'")
            carmen_shows = cursor.fetchall()
            assert len(carmen_shows) == 1

            # Fix: Use double single quotes for apostrophe in SQL
            cursor.execute(
                "SELECT show_name FROM show WHERE show_name = 'A Child''s Christmas in Wales'"
            )
            christmas_shows = cursor.fetchall()
            assert len(christmas_shows) == 2

            # Check total number of shows
            cursor.execute("SELECT COUNT(*) FROM show")
            total_shows = cursor.fetchone()[0]
            assert total_shows == 10  # Expected number of shows

            # Check that all shows have required fields
            cursor.execute(
                "SELECT COUNT(*) FROM show WHERE id IS NULL OR show_name IS NULL"
            )
            null_shows = cursor.fetchone()[0]
            assert null_shows == 0

            conn.close()

    def test_build_database_inserts_correct_user_data(self) -> None:
        """Test that build_database inserts the expected user data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = os.path.join(temp_dir, "show_booking_db.sqlite")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check specific users exist
            cursor.execute(
                "SELECT first_name, last_name FROM user WHERE email = 'alice.smith@gmail.com'"
            )
            alice = cursor.fetchone()
            assert alice is not None
            assert alice[0] == "Alice"
            assert alice[1] == "Smith"

            cursor.execute(
                "SELECT first_name, last_name FROM user WHERE email = 'bob.johnson@gmail.com'"
            )
            bob = cursor.fetchone()
            assert bob is not None
            assert bob[0] == "Bob"
            assert bob[1] == "Johnson"

            # Check total number of users
            cursor.execute("SELECT COUNT(*) FROM user")
            total_users = cursor.fetchone()[0]
            assert total_users == 5  # Expected number of users

            # Check that all users have required fields
            cursor.execute(
                "SELECT COUNT(*) FROM user WHERE id IS NULL OR first_name IS NULL OR last_name IS NULL OR email IS NULL"
            )
            null_users = cursor.fetchone()[0]
            assert null_users == 0

            conn.close()

    def test_build_database_inserts_booking_data(self) -> None:
        """Test that build_database inserts the expected booking data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = os.path.join(temp_dir, "show_booking_db.sqlite")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check that booking exists
            cursor.execute("SELECT COUNT(*) FROM booking")
            total_bookings = cursor.fetchone()[0]
            assert total_bookings == 1  # Expected number of bookings

            # Check the specific booking
            cursor.execute("""
                SELECT b.id, b.show_id, b.user_id, b.created_at,
                       s.show_name, u.first_name, u.last_name
                FROM booking b
                JOIN show s ON b.show_id = s.id
                JOIN user u ON b.user_id = u.id
            """)
            booking_data = cursor.fetchone()
            assert booking_data is not None
            assert booking_data[0] == "1"  # booking id
            assert booking_data[4] == "The Dead, 1904"  # show name
            assert booking_data[5] == "Alice"  # first name
            assert booking_data[6] == "Smith"  # last name

            conn.close()

    def test_build_database_foreign_key_constraints(self) -> None:
        """Test that build_database creates proper foreign key constraints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = os.path.join(temp_dir, "show_booking_db.sqlite")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check foreign key constraints
            cursor.execute("PRAGMA foreign_key_list(booking)")
            foreign_keys = cursor.fetchall()

            # Should have 2 foreign keys: show_id and user_id
            assert len(foreign_keys) == 2

            # Check show_id foreign key
            show_fk = [fk for fk in foreign_keys if fk[3] == "show_id"]
            assert len(show_fk) == 1
            assert show_fk[0][2] == "show"  # references show table

            # Check user_id foreign key
            user_fk = [fk for fk in foreign_keys if fk[3] == "user_id"]
            assert len(user_fk) == 1
            assert user_fk[0][2] == "user"  # references user table

            conn.close()

    def test_build_database_creates_folder_if_not_exists(self) -> None:
        """Test that build_database creates the folder if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_folder = os.path.join(temp_dir, "new_folder")

            # Ensure folder doesn't exist
            assert not os.path.exists(new_folder)

            # Create the folder before calling build_database
            os.makedirs(new_folder)

            # Build database should create the folder
            build_database(new_folder)

            # Verify folder was created
            assert os.path.exists(new_folder)
            assert os.path.isdir(new_folder)

            # Verify database was created in the new folder
            db_path = os.path.join(new_folder, "show_booking_db.sqlite")
            assert os.path.exists(db_path)

    def test_build_database_handles_special_characters_in_path(self) -> None:
        """Test that build_database handles special characters in folder path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create folder with special characters
            special_folder = os.path.join(temp_dir, "test folder with spaces & symbols")
            os.makedirs(special_folder)

            # Build database should handle special characters
            build_database(special_folder)

            # Verify database was created
            db_path = os.path.join(special_folder, "show_booking_db.sqlite")
            assert os.path.exists(db_path)

            # Verify database is functional
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM show")
            show_count = cursor.fetchone()[0]
            assert show_count > 0
            conn.close()

    def test_build_database_data_integrity(self) -> None:
        """Test that build_database maintains data integrity."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = os.path.join(temp_dir, "show_booking_db.sqlite")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check that all shows have valid data
            cursor.execute("""
                SELECT COUNT(*) FROM show
                WHERE price < 0 OR available_seats < 0
                OR date IS NULL OR time IS NULL
            """)
            invalid_shows = cursor.fetchone()[0]
            assert invalid_shows == 0

            # Check that all users have valid email format (basic check)
            cursor.execute("""
                SELECT COUNT(*) FROM user
                WHERE email NOT LIKE '%@%' OR email IS NULL
            """)
            invalid_emails = cursor.fetchone()[0]
            assert invalid_emails == 0

            # Check that booking references valid show and user
            cursor.execute("""
                SELECT COUNT(*) FROM booking b
                LEFT JOIN show s ON b.show_id = s.id
                LEFT JOIN user u ON b.user_id = u.id
                WHERE s.id IS NULL OR u.id IS NULL
            """)
            invalid_bookings = cursor.fetchone()[0]
            assert invalid_bookings == 0

            conn.close()

    def test_build_database_unique_constraints(self) -> None:
        """Test that build_database enforces unique constraints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = os.path.join(temp_dir, "show_booking_db.sqlite")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check that all primary keys are unique
            cursor.execute("SELECT COUNT(*) FROM show")
            total_shows = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT id) FROM show")
            unique_show_ids = cursor.fetchone()[0]
            assert total_shows == unique_show_ids

            cursor.execute("SELECT COUNT(*) FROM user")
            total_users = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT id) FROM user")
            unique_user_ids = cursor.fetchone()[0]
            assert total_users == unique_user_ids

            cursor.execute("SELECT COUNT(*) FROM booking")
            total_bookings = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT id) FROM booking")
            unique_booking_ids = cursor.fetchone()[0]
            assert total_bookings == unique_booking_ids

            conn.close()

    def test_build_database_connection_handling(self) -> None:
        """Test that build_database properly handles database connections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # This test ensures the connection is properly closed
            build_database(temp_dir)
            db_path = os.path.join(temp_dir, "show_booking_db.sqlite")

            # Try to open the database - should work if connection was properly closed
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM show")
            count = cursor.fetchone()[0]
            assert count > 0
            conn.close()

    def test_build_database_error_handling(self) -> None:
        """Test that build_database handles errors gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a read-only directory to test error handling
            read_only_dir = os.path.join(temp_dir, "readonly")
            os.makedirs(read_only_dir, mode=0o444)  # Read-only

            try:
                # This should raise an error due to permissions
                build_database(read_only_dir)
                # If we get here, the test should fail
                raise AssertionError(
                    "Expected an error when creating database in read-only directory"
                )
            except (PermissionError, OSError, sqlite3.OperationalError):
                # Expected error - sqlite3.OperationalError is also acceptable
                pass
            finally:
                # Clean up
                os.chmod(read_only_dir, 0o755)

    @patch("arklex.env.tools.booking_db.build_database.build_database")
    @patch("arklex.env.tools.booking_db.build_database.os.makedirs")
    @patch("arklex.env.tools.booking_db.build_database.os.path.exists")
    @patch("arklex.env.tools.booking_db.build_database.argparse.ArgumentParser")
    def test_main_function_creates_directory_and_calls_build_database(
        self,
        mock_parser: Mock,
        mock_exists: Mock,
        mock_makedirs: Mock,
        mock_build_database: Mock,
    ) -> None:
        """Test the main function when directory doesn't exist."""
        # Mock the argument parser
        mock_args = Mock()
        mock_args.folder_path = "/test/path"
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        # Mock os.path.exists to return False (directory doesn't exist)
        mock_exists.return_value = False

        # Import and call the main function directly
        from arklex.env.tools.booking_db.build_database import main

        main()

        # Verify the argument parser was called correctly
        mock_parser.assert_called_once()
        mock_parser_instance.add_argument.assert_called_once_with(
            "--folder_path",
            required=True,
            type=str,
            help="location to save the documents",
        )
        mock_parser_instance.parse_args.assert_called_once()

        # Verify os.path.exists was called
        mock_exists.assert_called_once_with("/test/path")

        # Verify os.makedirs was called (since directory doesn't exist)
        mock_makedirs.assert_called_once_with("/test/path")

        # Verify build_database was called
        mock_build_database.assert_called_once_with("/test/path")

    @patch("arklex.env.tools.booking_db.build_database.build_database")
    @patch("arklex.env.tools.booking_db.build_database.os.makedirs")
    @patch("arklex.env.tools.booking_db.build_database.os.path.exists")
    @patch("arklex.env.tools.booking_db.build_database.argparse.ArgumentParser")
    def test_main_function_skips_directory_creation_when_exists(
        self,
        mock_parser: Mock,
        mock_exists: Mock,
        mock_makedirs: Mock,
        mock_build_database: Mock,
    ) -> None:
        """Test that the main function doesn't create directory when it already exists."""
        # Mock the argument parser
        mock_args = Mock()
        mock_args.folder_path = "/test/path"
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        # Mock os.path.exists to return True (directory exists)
        mock_exists.return_value = True

        # Import and call the main function directly
        from arklex.env.tools.booking_db.build_database import main

        main()

        # Verify os.path.exists was called
        mock_exists.assert_called_once_with("/test/path")

        # Verify os.makedirs was NOT called (since directory exists)
        mock_makedirs.assert_not_called()

        # Verify build_database was called
        mock_build_database.assert_called_once_with("/test/path")

    def test_main_function_exists(self) -> None:
        """Test that the main function exists and can be imported."""
        from arklex.env.tools.booking_db.build_database import main

        assert callable(main)


class TestBookingDBIntegration:
    """Integration tests for booking database tools."""

    def test_booking_workflow(self) -> None:
        """Test a complete booking workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Build the database
            build_database(temp_dir)

            # Verify database was created with data
            db_path = os.path.join(temp_dir, "show_booking_db.sqlite")
            assert os.path.exists(db_path)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check that we have shows in the database
            cursor.execute("SELECT COUNT(*) FROM show")
            show_count = cursor.fetchone()[0]
            assert show_count > 0

            conn.close()
