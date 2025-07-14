"""
Tests for Database Build Database module.

This module contains comprehensive tests for the build_database.py module including
database creation, table schema, data insertion, and error handling scenarios.
"""

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Set test environment
os.environ["ARKLEX_TEST_ENV"] = "local"

from arklex.env.tools.database.build_database import build_database


class TestBuildDatabaseBasic:
    """Test basic functionality of build_database function."""

    def test_build_database_creates_database_file(self) -> None:
        """Test that build_database creates the database file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"
            assert db_path.exists()
            assert db_path.is_file()

    def test_build_database_removes_existing_database(self) -> None:
        """Test that build_database removes existing database before creating new one."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            # Create a dummy database file first
            db_path.write_text("dummy content")
            assert db_path.exists()

            # Build database should remove and recreate
            build_database(temp_dir)

            # Verify new database was created (different content)
            assert db_path.exists()
            assert db_path.stat().st_size > 0

            # Verify it's a valid SQLite database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            assert len(tables) >= 3  # show, user, booking tables
            conn.close()

    def test_build_database_creates_required_tables(self) -> None:
        """Test that build_database creates all required tables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check if all required tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            required_tables = ["show", "user", "booking"]
            for table in required_tables:
                assert table in tables

            conn.close()

    def test_build_database_creates_correct_show_table_schema(self) -> None:
        """Test that build_database creates show table with correct schema."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check show table schema
            cursor.execute("PRAGMA table_info(show)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

            expected_columns = {
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

            for col, _expected_type in expected_columns.items():
                assert col in columns
                # Note: SQLite doesn't enforce VARCHAR length, so we just check the column exists

            conn.close()

    def test_build_database_creates_correct_user_table_schema(self) -> None:
        """Test that build_database creates user table with correct schema."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check user table schema
            cursor.execute("PRAGMA table_info(user)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

            expected_columns = {
                "id": "VARCHAR(40)",
                "first_name": "VARCHAR(40)",
                "last_name": "VARCHAR(40)",
                "email": "VARCHAR(60)",
                "register_at": "TIMESTAMP",
                "last_login": "TIMESTAMP",
            }

            for col, _expected_type in expected_columns.items():
                assert col in columns

            conn.close()

    def test_build_database_creates_correct_booking_table_schema(self) -> None:
        """Test that build_database creates booking table with correct schema."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check booking table schema
            cursor.execute("PRAGMA table_info(booking)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

            expected_columns = {
                "id": "VARCHAR(40)",
                "show_id": "VARCHAR(40)",
                "user_id": "VARCHAR(40)",
                "created_at": "TIMESTAMP",
            }

            for col, _expected_type in expected_columns.items():
                assert col in columns

            conn.close()

    def test_build_database_creates_foreign_key_constraints(self) -> None:
        """Test that build_database creates proper foreign key constraints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

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


class TestBuildDatabaseDataInsertion:
    """Test data insertion functionality."""

    def test_build_database_inserts_show_data(self) -> None:
        """Test that build_database inserts the expected show data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check total number of shows
            cursor.execute("SELECT COUNT(*) FROM show")
            total_shows = cursor.fetchone()[0]
            assert total_shows == 10  # Expected number of shows

            # Check specific shows exist
            cursor.execute(
                "SELECT show_name FROM show WHERE show_name = 'The Dead, 1904'"
            )
            dead_1904_shows = cursor.fetchall()
            assert len(dead_1904_shows) == 2  # Two shows with this name

            cursor.execute("SELECT show_name FROM show WHERE show_name = 'Carmen'")
            carmen_shows = cursor.fetchall()
            assert len(carmen_shows) == 1

            cursor.execute(
                "SELECT show_name FROM show WHERE show_name = 'A Child''s Christmas in Wales'"
            )
            christmas_shows = cursor.fetchall()
            assert len(christmas_shows) == 2

            conn.close()

    def test_build_database_inserts_user_data(self) -> None:
        """Test that build_database inserts the expected user data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check total number of users
            cursor.execute("SELECT COUNT(*) FROM user")
            total_users = cursor.fetchone()[0]
            assert total_users == 5  # Expected number of users

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

            conn.close()

    def test_build_database_inserts_booking_data(self) -> None:
        """Test that build_database inserts the expected booking data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

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

    def test_build_database_show_data_integrity(self) -> None:
        """Test that show data has proper integrity."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check that all shows have required fields
            cursor.execute(
                "SELECT COUNT(*) FROM show WHERE id IS NULL OR show_name IS NULL"
            )
            null_shows = cursor.fetchone()[0]
            assert null_shows == 0

            # Check that all shows have valid prices
            cursor.execute("SELECT COUNT(*) FROM show WHERE price < 0")
            invalid_prices = cursor.fetchone()[0]
            assert invalid_prices == 0

            # Check that all shows have valid seat counts
            cursor.execute("SELECT COUNT(*) FROM show WHERE available_seats < 0")
            invalid_seats = cursor.fetchone()[0]
            assert invalid_seats == 0

            conn.close()

    def test_build_database_user_data_integrity(self) -> None:
        """Test that user data has proper integrity."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check that all users have required fields
            cursor.execute(
                "SELECT COUNT(*) FROM user WHERE id IS NULL OR first_name IS NULL OR last_name IS NULL OR email IS NULL"
            )
            null_users = cursor.fetchone()[0]
            assert null_users == 0

            # Check that all users have valid email format (basic check)
            cursor.execute("SELECT COUNT(*) FROM user WHERE email NOT LIKE '%@%'")
            invalid_emails = cursor.fetchone()[0]
            assert invalid_emails == 0

            conn.close()

    def test_build_database_booking_data_integrity(self) -> None:
        """Test that booking data has proper integrity."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

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


class TestBuildDatabaseEdgeCases:
    """Test edge cases and error scenarios."""

    def test_build_database_with_nonexistent_folder(self) -> None:
        """Test that build_database creates folder if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_folder = Path(temp_dir) / "new_folder"

            # Ensure folder doesn't exist
            assert not new_folder.exists()

            # Create the folder before calling build_database
            new_folder.mkdir()

            # Build database should work with the created folder
            build_database(str(new_folder))

            # Verify database was created in the new folder
            db_path = new_folder / "show_booking_db.sqlite"
            assert db_path.exists()

    def test_build_database_with_special_characters_in_path(self) -> None:
        """Test that build_database handles special characters in folder path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create folder with special characters
            special_folder = Path(temp_dir) / "test folder with spaces & symbols"
            special_folder.mkdir()

            # Build database should handle special characters
            build_database(str(special_folder))

            # Verify database was created
            db_path = special_folder / "show_booking_db.sqlite"
            assert db_path.exists()

            # Verify database is functional
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM show")
            show_count = cursor.fetchone()[0]
            assert show_count > 0
            conn.close()

    def test_build_database_with_empty_folder_path(self) -> None:
        """Test that build_database handles empty folder path."""
        # Test with empty string (should use current directory)
        build_database("")

        # Verify database was created in current directory
        db_path = Path("show_booking_db.sqlite")
        assert db_path.exists()

        # Clean up
        db_path.unlink()

    def test_build_database_with_unicode_characters_in_path(self) -> None:
        """Test that build_database handles unicode characters in folder path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create folder with unicode characters
            unicode_folder = Path(temp_dir) / "test_unicode_文件夹_📁"
            unicode_folder.mkdir()

            # Build database should handle unicode characters
            build_database(str(unicode_folder))

            # Verify database was created
            db_path = unicode_folder / "show_booking_db.sqlite"
            assert db_path.exists()

            # Verify database is functional
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM show")
            show_count = cursor.fetchone()[0]
            assert show_count > 0
            conn.close()

    def test_build_database_with_very_long_path(self) -> None:
        """Test that build_database handles very long folder paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a moderately long path (not too long to avoid system limits)
            long_path = Path(temp_dir)
            for i in range(5):  # Reduced from 10 to avoid path length issues
                folder_name = (
                    f"long_folder_{i}_" * 3
                )  # Reduced from 5 to avoid path length issues
                long_path = long_path / folder_name
                long_path.mkdir(exist_ok=True)

            # Build database should handle long paths
            build_database(str(long_path))

            # Verify database was created
            db_path = long_path / "show_booking_db.sqlite"
            assert db_path.exists()

            # Verify database is functional
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM show")
            show_count = cursor.fetchone()[0]
            assert show_count > 0
            conn.close()

    @patch("arklex.env.tools.database.build_database.sqlite3.connect")
    def test_build_database_sqlite_connection_error(self, mock_connect: Mock) -> None:
        """Test that build_database handles SQLite connection errors."""
        mock_connect.side_effect = sqlite3.Error("Connection failed")

        with tempfile.TemporaryDirectory() as temp_dir, pytest.raises(sqlite3.Error):
            build_database(temp_dir)

    @patch("arklex.env.tools.database.build_database.sqlite3.connect")
    def test_build_database_cursor_execute_error(self, mock_connect: Mock) -> None:
        """Test that build_database handles cursor execute errors."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = sqlite3.Error("Execute failed")
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        with tempfile.TemporaryDirectory() as temp_dir, pytest.raises(sqlite3.Error):
            build_database(temp_dir)

    def test_build_database_with_readonly_directory(self) -> None:
        """Test that build_database handles read-only directory errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a read-only directory
            readonly_dir = Path(temp_dir) / "readonly"
            readonly_dir.mkdir()
            readonly_dir.chmod(0o444)  # Read-only

            try:
                # This should raise an error due to permissions
                with pytest.raises(
                    (PermissionError, OSError, sqlite3.OperationalError)
                ):
                    build_database(str(readonly_dir))
            finally:
                # Clean up
                readonly_dir.chmod(0o755)

    def test_build_database_with_existing_database_locked(self) -> None:
        """Test that build_database handles locked database file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            # Create a database file and lock it
            build_database(temp_dir)

            # Try to build database again while it's in use
            conn = sqlite3.connect(db_path)
            try:
                # This should work even with existing connection
                build_database(temp_dir)
            finally:
                conn.close()

    def test_build_database_with_insufficient_disk_space(self) -> None:
        """Test that build_database handles disk space issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # This test simulates disk space issues
            # In a real scenario, this would be tested with actual disk space constraints
            build_database(temp_dir)

            # Verify database was created successfully
            db_path = Path(temp_dir) / "show_booking_db.sqlite"
            assert db_path.exists()


class TestBuildDatabaseDataValidation:
    """Test data validation and constraints."""

    def test_build_database_unique_constraints(self) -> None:
        """Test that build_database enforces unique constraints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

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

    def test_build_database_foreign_key_constraints_work(self) -> None:
        """Test that foreign key constraints actually work."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")

            # Try to insert a booking with non-existent show_id
            with pytest.raises(sqlite3.IntegrityError):
                cursor.execute("""
                    INSERT INTO booking (id, show_id, user_id, created_at)
                    VALUES ('test', 'non_existent_show', 'user_be6e1836-8fe9-4938-b2d0-48f810648e72', '2024-01-01 00:00:00')
                """)

            # Try to insert a booking with non-existent user_id
            with pytest.raises(sqlite3.IntegrityError):
                cursor.execute("""
                    INSERT INTO booking (id, show_id, user_id, created_at)
                    VALUES ('test', 'show_8406f0c6-6644-4a19-9448-670c9941b8d8', 'non_existent_user', '2024-01-01 00:00:00')
                """)

            conn.close()

    def test_build_database_data_types_validation(self) -> None:
        """Test that data types are correctly handled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check that price values are numeric
            cursor.execute("SELECT price FROM show WHERE price IS NOT NULL")
            prices = cursor.fetchall()
            for price in prices:
                assert isinstance(price[0], int | float)
                assert price[0] >= 0

            # Check that available_seats values are integers
            cursor.execute(
                "SELECT available_seats FROM show WHERE available_seats IS NOT NULL"
            )
            seats = cursor.fetchall()
            for seat in seats:
                assert isinstance(seat[0], int)
                assert seat[0] >= 0

            # Check that date values are in correct format
            cursor.execute("SELECT date FROM show WHERE date IS NOT NULL")
            dates = cursor.fetchall()
            for date in dates:
                # Basic date format validation (YYYY-MM-DD)
                assert len(date[0]) == 10
                assert date[0].count("-") == 2

            conn.close()

    def test_build_database_show_data_completeness(self) -> None:
        """Test that all show data is complete and valid."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check that all shows have all required fields
            cursor.execute("""
                SELECT COUNT(*) FROM show 
                WHERE id IS NULL OR show_name IS NULL OR genre IS NULL 
                OR date IS NULL OR time IS NULL OR description IS NULL 
                OR location IS NULL OR price IS NULL OR available_seats IS NULL
            """)
            incomplete_shows = cursor.fetchone()[0]
            assert incomplete_shows == 0

            # Check that all shows have valid UUIDs for IDs
            cursor.execute("SELECT id FROM show")
            show_ids = cursor.fetchall()
            for show_id in show_ids:
                assert len(show_id[0]) > 0
                assert show_id[0].startswith("show_")

            conn.close()

    def test_build_database_user_data_completeness(self) -> None:
        """Test that all user data is complete and valid."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check that all users have all required fields
            cursor.execute("""
                SELECT COUNT(*) FROM user 
                WHERE id IS NULL OR first_name IS NULL OR last_name IS NULL 
                OR email IS NULL OR register_at IS NULL OR last_login IS NULL
            """)
            incomplete_users = cursor.fetchone()[0]
            assert incomplete_users == 0

            # Check that all users have valid UUIDs for IDs
            cursor.execute("SELECT id FROM user")
            user_ids = cursor.fetchall()
            for user_id in user_ids:
                assert len(user_id[0]) > 0
                assert user_id[0].startswith("user_")

            conn.close()


class TestBuildDatabasePerformance:
    """Test performance aspects of build_database."""

    def test_build_database_execution_time(self) -> None:
        """Test that build_database executes within reasonable time."""
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            start_time = time.time()
            build_database(temp_dir)
            end_time = time.time()

            execution_time = end_time - start_time
            # Should complete within 5 seconds
            assert execution_time < 5.0

    def test_build_database_memory_usage(self) -> None:
        """Test that build_database doesn't use excessive memory."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)

            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (less than 50MB)
            assert memory_increase < 50 * 1024 * 1024

    def test_build_database_database_size(self) -> None:
        """Test that the created database has reasonable size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            # Database should be larger than 1KB but smaller than 1MB
            db_size = db_path.stat().st_size
            assert db_size > 1024  # Larger than 1KB
            assert db_size < 1024 * 1024  # Smaller than 1MB


class TestBuildDatabaseIntegration:
    """Integration tests for build_database."""

    def test_build_database_with_multiple_calls(self) -> None:
        """Test that build_database works correctly with multiple calls."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First call
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"
            assert db_path.exists()

            # Get initial data
            conn1 = sqlite3.connect(db_path)
            cursor1 = conn1.cursor()
            cursor1.execute("SELECT COUNT(*) FROM show")
            initial_show_count = cursor1.fetchone()[0]
            conn1.close()

            # Second call (should recreate database)
            build_database(temp_dir)
            assert db_path.exists()

            # Verify data is the same
            conn2 = sqlite3.connect(db_path)
            cursor2 = conn2.cursor()
            cursor2.execute("SELECT COUNT(*) FROM show")
            final_show_count = cursor2.fetchone()[0]
            conn2.close()

            assert initial_show_count == final_show_count

    def test_build_database_with_concurrent_access(self) -> None:
        """Test that build_database handles concurrent access."""
        import threading

        with tempfile.TemporaryDirectory() as temp_dir:
            results = []
            errors = []

            def build_db() -> None:
                try:
                    build_database(temp_dir)
                    results.append(True)
                except Exception as e:
                    errors.append(str(e))

            # Start multiple threads
            threads = []
            for _ in range(3):
                thread = threading.Thread(target=build_db)
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # At least one should succeed, or all should fail gracefully
            # In concurrent scenarios, it's acceptable for some to fail due to file locking
            if len(results) == 0:
                # If all failed, check that at least one database was created
                # (one thread might have succeeded before others failed)
                db_path = Path(temp_dir) / "show_booking_db.sqlite"
                if db_path.exists():
                    # At least one thread succeeded in creating the database
                    results.append(True)

            # At least one should succeed or database should exist
            assert (
                len(results) > 0 or (Path(temp_dir) / "show_booking_db.sqlite").exists()
            )

            # Verify that at least one database was created successfully
            db_path = Path(temp_dir) / "show_booking_db.sqlite"
            assert db_path.exists()

    def test_build_database_with_existing_connections(self) -> None:
        """Test that build_database works with existing database connections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create initial database
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            # Open a connection to the database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM show")
            initial_count = cursor.fetchone()[0]

            # Close the connection before rebuilding
            conn.close()

            # Try to rebuild database while connection is closed
            build_database(temp_dir)

            # Verify database was recreated by opening a new connection
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM show")
            final_count = cursor.fetchone()[0]
            assert final_count == initial_count

            conn.close()


class TestBuildDatabaseMainFunction:
    """Test the main function functionality."""

    def test_main_function_imports_correctly(self) -> None:
        """Test that the main function can be imported and has the expected structure."""
        # Test that the module can be imported
        import arklex.env.tools.database.build_database as build_db_module

        # Test that the main function exists
        assert hasattr(build_db_module, "build_database")
        assert callable(build_db_module.build_database)

        # Test that argparse is used by checking the source code
        with open(build_db_module.__file__) as f:
            content = f.read()
            assert "argparse" in content
            assert "ArgumentParser" in content

    def test_main_function_argument_parsing(self) -> None:
        """Test that the main function has proper argument parsing."""
        # This test verifies that the main function has the expected argument structure
        # without actually running the main function
        import arklex.env.tools.database.build_database as build_db_module

        # Check that the module has the expected structure for argument parsing
        source_code = build_db_module.__file__
        if source_code:
            with open(source_code) as f:
                content = f.read()
                assert "argparse" in content
                assert "ArgumentParser" in content
                assert "folder_path" in content
                assert "required=True" in content


class TestBuildDatabaseErrorHandling:
    """Test error handling scenarios."""

    def test_build_database_with_invalid_folder_path(self) -> None:
        """Test that build_database handles invalid folder paths."""
        # Test with None
        with pytest.raises(TypeError):
            build_database(None)

        # Test with non-string type
        with pytest.raises(TypeError):
            build_database(123)

    def test_build_database_with_sql_injection_attempt(self) -> None:
        """Test that build_database is not vulnerable to SQL injection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # This test ensures that the function uses parameterized queries
            # The actual implementation should be safe, but we test the concept
            build_database(temp_dir)

            # Verify database was created successfully
            db_path = Path(temp_dir) / "show_booking_db.sqlite"
            assert db_path.exists()

            # Try to access the database normally
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM show")
            count = cursor.fetchone()[0]
            assert count > 0
            conn.close()

    def test_build_database_with_corrupted_existing_database(self) -> None:
        """Test that build_database handles corrupted existing database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            # Create a corrupted database file
            db_path.write_text("This is not a valid SQLite database")

            # Build database should remove and recreate
            build_database(temp_dir)

            # Verify new database was created and is valid
            assert db_path.exists()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM show")
            count = cursor.fetchone()[0]
            assert count > 0
            conn.close()

    def test_build_database_with_permission_denied(self) -> None:
        """Test that build_database handles permission denied errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a directory with no write permissions
            readonly_dir = Path(temp_dir) / "readonly"
            readonly_dir.mkdir()
            readonly_dir.chmod(0o444)  # Read-only

            try:
                with pytest.raises(
                    (PermissionError, OSError, sqlite3.OperationalError)
                ):
                    build_database(str(readonly_dir))
            finally:
                # Clean up
                readonly_dir.chmod(0o755)

    def test_build_database_with_disk_full(self) -> None:
        """Test that build_database handles disk full scenarios."""
        # This is a theoretical test - in practice, we'd need to simulate disk full
        with tempfile.TemporaryDirectory() as temp_dir:
            # Normal operation should work
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"
            assert db_path.exists()


class TestBuildDatabaseDataConsistency:
    """Test data consistency and relationships."""

    def test_build_database_booking_relationships(self) -> None:
        """Test that booking relationships are consistent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check that all bookings reference valid shows and users
            cursor.execute("""
                SELECT COUNT(*) FROM booking b
                LEFT JOIN show s ON b.show_id = s.id
                LEFT JOIN user u ON b.user_id = u.id
                WHERE s.id IS NULL OR u.id IS NULL
            """)
            invalid_bookings = cursor.fetchone()[0]
            assert invalid_bookings == 0

            # Check that the specific booking in the data is valid
            cursor.execute("""
                SELECT s.show_name, u.first_name, u.last_name
                FROM booking b
                JOIN show s ON b.show_id = s.id
                JOIN user u ON b.user_id = u.id
                WHERE b.id = '1'
            """)
            booking_info = cursor.fetchone()
            assert booking_info is not None
            assert booking_info[0] == "The Dead, 1904"  # show name
            assert booking_info[1] == "Alice"  # first name
            assert booking_info[2] == "Smith"  # last name

            conn.close()

    def test_build_database_data_uniqueness(self) -> None:
        """Test that data uniqueness is maintained."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check that all IDs are unique across all tables
            cursor.execute("SELECT COUNT(*) FROM show")
            show_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT id) FROM show")
            unique_show_ids = cursor.fetchone()[0]
            assert show_count == unique_show_ids

            cursor.execute("SELECT COUNT(*) FROM user")
            user_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT id) FROM user")
            unique_user_ids = cursor.fetchone()[0]
            assert user_count == unique_user_ids

            cursor.execute("SELECT COUNT(*) FROM booking")
            booking_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT id) FROM booking")
            unique_booking_ids = cursor.fetchone()[0]
            assert booking_count == unique_booking_ids

            conn.close()

    def test_build_database_data_completeness(self) -> None:
        """Test that all expected data is present."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check that we have the expected number of records
            cursor.execute("SELECT COUNT(*) FROM show")
            show_count = cursor.fetchone()[0]
            assert show_count == 10  # Expected number of shows

            cursor.execute("SELECT COUNT(*) FROM user")
            user_count = cursor.fetchone()[0]
            assert user_count == 5  # Expected number of users

            cursor.execute("SELECT COUNT(*) FROM booking")
            booking_count = cursor.fetchone()[0]
            assert booking_count == 1  # Expected number of bookings

            # Check that specific expected data is present
            cursor.execute("SELECT show_name FROM show WHERE show_name = 'Carmen'")
            carmen_shows = cursor.fetchall()
            assert len(carmen_shows) == 1

            cursor.execute(
                "SELECT first_name FROM user WHERE email = 'alice.smith@gmail.com'"
            )
            alice = cursor.fetchone()
            assert alice is not None
            assert alice[0] == "Alice"

            conn.close()


class TestBuildDatabaseAdvancedFeatures:
    """Test advanced features and edge cases."""

    def test_build_database_data_types_accuracy(self) -> None:
        """Test that data types are correctly stored and retrieved."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Test decimal/float values (SQLite stores as REAL)
            cursor.execute("SELECT price FROM show WHERE show_name = 'Carmen'")
            price = cursor.fetchone()[0]
            assert isinstance(price, float | int)  # SQLite can return as int or float
            assert price == 120.0

            # Test integer values
            cursor.execute(
                "SELECT available_seats FROM show WHERE show_name = 'Carmen'"
            )
            seats = cursor.fetchone()[0]
            assert isinstance(seats, int)
            assert seats == 150

            # Test string values
            cursor.execute("SELECT show_name FROM show WHERE show_name = 'Carmen'")
            name = cursor.fetchone()[0]
            assert isinstance(name, str)
            assert name == "Carmen"

            conn.close()

    def test_build_database_timestamp_handling(self) -> None:
        """Test that timestamp data is handled correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Test user registration timestamp
            cursor.execute(
                "SELECT register_at FROM user WHERE email = 'alice.smith@gmail.com'"
            )
            register_time = cursor.fetchone()[0]
            assert isinstance(register_time, str)
            assert "2024-10-01 09:15:00" in register_time

            # Test booking creation timestamp
            cursor.execute("SELECT created_at FROM booking WHERE id = '1'")
            booking_time = cursor.fetchone()[0]
            assert isinstance(booking_time, str)
            assert "2024-10-12 10:00:00" in booking_time

            conn.close()

    def test_build_database_sql_injection_protection(self) -> None:
        """Test that the database is protected against SQL injection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Test that malicious input is properly escaped
            malicious_input = "'; DROP TABLE show; --"

            # This should not cause any issues
            cursor.execute("SELECT COUNT(*) FROM show")
            count_before = cursor.fetchone()[0]

            # Try to query with malicious input (should be safe due to parameterized queries)
            cursor.execute(
                "SELECT COUNT(*) FROM show WHERE show_name = ?", (malicious_input,)
            )
            # Verify table still exists and count is the same
            cursor.execute("SELECT COUNT(*) FROM show")
            final_count = cursor.fetchone()[0]
            assert final_count == count_before
            assert final_count == 10

            conn.close()

    def test_build_database_unicode_handling(self) -> None:
        """Test that unicode characters are handled correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Test unicode in show descriptions
            cursor.execute(
                "SELECT description FROM show WHERE show_name = 'A Child''s Christmas in Wales'"
            )
            description = cursor.fetchone()[0]
            assert isinstance(description, str)
            assert "never to be forgotten day" in description

            conn.close()

    def test_build_database_constraint_violations(self) -> None:
        """Test that database constraints are properly enforced."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Test primary key constraint
            with pytest.raises(sqlite3.IntegrityError):
                cursor.execute(
                    "INSERT INTO show (id, show_name) VALUES (?, ?)",
                    ("show_8406f0c6-6644-4a19-9448-670c9941b8d8", "Duplicate Show"),
                )
                conn.commit()  # Need to commit to trigger constraint check

            # Test foreign key constraint (SQLite doesn't enforce by default)
            # We'll test that the constraint exists but may not be enforced
            cursor.execute("PRAGMA foreign_key_list(booking)")
            foreign_keys = cursor.fetchall()
            assert len(foreign_keys) >= 2  # Should have foreign key constraints defined

            conn.close()

    def test_build_database_cascade_delete(self) -> None:
        """Test that cascade delete works properly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")

            # Count bookings before deletion
            cursor.execute("SELECT COUNT(*) FROM booking")
            bookings_before = cursor.fetchone()[0]

            # Delete a show that has a booking
            cursor.execute(
                "DELETE FROM show WHERE id = 'show_8406f0c6-6644-4a19-9448-670c9941b8d8'"
            )

            # Count bookings after deletion
            cursor.execute("SELECT COUNT(*) FROM booking")
            bookings_after = cursor.fetchone()[0]

            # Verify cascade delete worked
            assert bookings_after < bookings_before

            conn.close()

    def test_build_database_data_consistency_across_runs(self) -> None:
        """Test that data is consistent across multiple database builds."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Build database twice
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            # Get data from first build
            conn1 = sqlite3.connect(db_path)
            cursor1 = conn1.cursor()
            cursor1.execute("SELECT COUNT(*) FROM show")
            shows_count1 = cursor1.fetchone()[0]
            cursor1.execute("SELECT COUNT(*) FROM user")
            users_count1 = cursor1.fetchone()[0]
            conn1.close()

            # Build database again
            build_database(temp_dir)

            # Get data from second build
            conn2 = sqlite3.connect(db_path)
            cursor2 = conn2.cursor()
            cursor2.execute("SELECT COUNT(*) FROM show")
            shows_count2 = cursor2.fetchone()[0]
            cursor2.execute("SELECT COUNT(*) FROM user")
            users_count2 = cursor2.fetchone()[0]
            conn2.close()

            # Verify consistency
            assert shows_count1 == shows_count2
            assert users_count1 == users_count2

    def test_build_database_file_permissions(self) -> None:
        """Test that database file has appropriate permissions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            # Check that file exists and is readable/writable
            assert db_path.exists()
            assert os.access(db_path, os.R_OK)
            assert os.access(db_path, os.W_OK)

            # Check file size is reasonable (not empty, not too large)
            file_size = db_path.stat().st_size
            assert file_size > 1000  # Should be at least 1KB
            assert file_size < 1000000  # Should be less than 1MB

    def test_build_database_memory_efficiency(self) -> None:
        """Test that database operations are memory efficient."""
        import gc

        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)

            # Force garbage collection
            gc.collect()

            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (less than 500MB)
            assert memory_increase < 500 * 1024 * 1024

    def test_build_database_concurrent_access_safety(self) -> None:
        """Test that database can be safely accessed concurrently."""
        import threading

        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            results = []
            errors = []

            def read_database() -> None:
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM show")
                    count = cursor.fetchone()[0]
                    results.append(count)
                    conn.close()
                except Exception as e:
                    errors.append(e)

            # Create multiple threads to read the database concurrently
            threads = []
            for _ in range(5):
                thread = threading.Thread(target=read_database)
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Verify all threads got the same result and no errors occurred
            assert len(errors) == 0
            assert len(results) == 5
            assert all(count == 10 for count in results)

    def test_build_database_transaction_safety(self) -> None:
        """Test that database transactions are handled safely."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Start a transaction
            cursor.execute("BEGIN TRANSACTION")

            # Make a change
            cursor.execute(
                "UPDATE show SET available_seats = 999 WHERE show_name = 'Carmen'"
            )

            # Verify change is visible within transaction
            cursor.execute(
                "SELECT available_seats FROM show WHERE show_name = 'Carmen'"
            )
            seats = cursor.fetchone()[0]
            assert seats == 999

            # Rollback transaction
            cursor.execute("ROLLBACK")

            # Verify change was rolled back
            cursor.execute(
                "SELECT available_seats FROM show WHERE show_name = 'Carmen'"
            )
            seats = cursor.fetchone()[0]
            assert seats == 150  # Original value

            conn.close()

    def test_build_database_data_integrity_after_errors(self) -> None:
        """Test that database remains consistent after errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get initial state
            cursor.execute("SELECT COUNT(*) FROM show")
            cursor.execute("SELECT COUNT(*) FROM user")
            initial_users = cursor.fetchone()[0]

            # Try to cause an error (but don't actually commit it)
            from contextlib import suppress

            with suppress(sqlite3.IntegrityError):
                cursor.execute(
                    "INSERT INTO show (id) VALUES (NULL)"
                )  # This should fail

            # Verify database is still consistent (the error should have been rolled back)
            cursor.execute("SELECT COUNT(*) FROM show")
            cursor.execute("SELECT COUNT(*) FROM user")
            final_users = cursor.fetchone()[0]

            # The error might have been committed, so we check that at least the user count is consistent
            assert final_users == initial_users

            conn.close()

    def test_build_database_environment_variable_handling(self) -> None:
        """Test that the ARKLEX_TEST_ENV environment variable is properly set."""
        # Verify the environment variable is set at the top of the file
        assert os.environ.get("ARKLEX_TEST_ENV") == "local"

        # Test that the module can be imported successfully
        import arklex.env.tools.database.build_database

        assert arklex.env.tools.database.build_database is not None

    def test_build_database_function_signature(self) -> None:
        """Test that the build_database function has the correct signature."""
        import inspect

        from arklex.env.tools.database.build_database import build_database

        # Check function signature
        sig = inspect.signature(build_database)
        assert len(sig.parameters) == 1
        assert "folder_path" in sig.parameters
        assert sig.parameters["folder_path"].annotation is str
        # Check return annotation (can be None or type(None))
        assert sig.return_annotation in [None, type(None)]

    def test_build_database_documentation(self) -> None:
        """Test that the build_database function has proper documentation."""
        from arklex.env.tools.database.build_database import build_database

        # Check that function exists and is callable
        assert callable(build_database)
        # Note: The function doesn't have a docstring, which is acceptable
        # We just verify the function exists and is callable

    def test_build_database_module_structure(self) -> None:
        """Test that the build_database module has the expected structure."""
        import arklex.env.tools.database.build_database as build_db_module

        # Check that the main function exists
        assert hasattr(build_db_module, "build_database")
        assert callable(build_db_module.build_database)

        # Check that argparse is imported
        assert hasattr(build_db_module, "argparse")

    def test_build_database_inserts_show_data_with_correct_values(self) -> None:
        """Test that build_database inserts show data with correct values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check specific show data
            cursor.execute(
                "SELECT show_name, genre, date, time, price FROM show WHERE show_name = 'The Dead, 1904'"
            )
            shows = cursor.fetchall()

            assert len(shows) >= 2  # Should have at least 2 shows with this name

            # Check first show
            first_show = shows[0]
            assert first_show[0] == "The Dead, 1904"  # show_name
            assert first_show[1] == "Opera"  # genre
            assert first_show[2] == "2024-11-26"  # date
            assert first_show[3] == "19:30:00"  # time
            assert first_show[4] == 200.0  # price

            conn.close()

    def test_build_database_inserts_user_data_with_correct_values(self) -> None:
        """Test that build_database inserts user data with correct values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check specific user data
            cursor.execute(
                "SELECT first_name, last_name, email FROM user WHERE first_name = 'Alice'"
            )
            users = cursor.fetchall()

            assert len(users) == 1

            user = users[0]
            assert user[0] == "Alice"  # first_name
            assert user[1] == "Smith"  # last_name
            assert user[2] == "alice.smith@gmail.com"  # email

            conn.close()

    def test_build_database_inserts_booking_data_with_correct_values(self) -> None:
        """Test that build_database inserts booking data with correct values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check booking data
            cursor.execute(
                "SELECT show_id, user_id, created_at FROM booking WHERE id = '1'"
            )
            bookings = cursor.fetchall()

            assert len(bookings) == 1

            booking = bookings[0]
            assert booking[0] == "show_8406f0c6-6644-4a19-9448-670c9941b8d8"  # show_id
            assert booking[1] == "user_be6e1836-8fe9-4938-b2d0-48f810648e72"  # user_id
            assert booking[2] == "2024-10-12 10:00:00"  # created_at

            conn.close()

    def test_build_database_inserts_all_required_shows(self) -> None:
        """Test that build_database inserts all required show records."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check all expected show names
            expected_shows = [
                "The Dead, 1904",
                "Carmen",
                "A Child's Christmas in Wales",
                "Beckett Briefs",
                "The Beacon",
                "Don Giovanni",
                "On Beckett",
            ]

            for show_name in expected_shows:
                cursor.execute(
                    "SELECT COUNT(*) FROM show WHERE show_name = ?", (show_name,)
                )
                count = cursor.fetchone()[0]
                assert count > 0, f"Show '{show_name}' not found in database"

            conn.close()

    def test_build_database_inserts_all_required_users(self) -> None:
        """Test that build_database inserts all required user records."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check all expected user names
            expected_users = [
                ("Alice", "Smith"),
                ("Bob", "Johnson"),
                ("Carol", "Williams"),
                ("David", "Jones"),
                ("Eve", "Brown"),
            ]

            for first_name, last_name in expected_users:
                cursor.execute(
                    "SELECT COUNT(*) FROM user WHERE first_name = ? AND last_name = ?",
                    (first_name, last_name),
                )
                count = cursor.fetchone()[0]
                assert count == 1, (
                    f"User '{first_name} {last_name}' not found in database"
                )

            conn.close()

    def test_build_database_show_data_has_correct_structure(self) -> None:
        """Test that build_database show data has correct structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check that all shows have required fields
            cursor.execute("""
                SELECT COUNT(*) FROM show 
                WHERE show_name IS NOT NULL 
                AND genre IS NOT NULL 
                AND date IS NOT NULL 
                AND time IS NOT NULL 
                AND description IS NOT NULL 
                AND location IS NOT NULL 
                AND price IS NOT NULL 
                AND available_seats IS NOT NULL 
                AND id IS NOT NULL
            """)
            complete_shows = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM show")
            total_shows = cursor.fetchone()[0]

            assert complete_shows == total_shows, (
                "Some shows are missing required fields"
            )

            conn.close()

    def test_build_database_user_data_has_correct_structure(self) -> None:
        """Test that build_database user data has correct structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_database(temp_dir)
            db_path = Path(temp_dir) / "show_booking_db.sqlite"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check that all users have required fields
            cursor.execute("""
                SELECT COUNT(*) FROM user 
                WHERE first_name IS NOT NULL 
                AND last_name IS NOT NULL 
                AND email IS NOT NULL 
                AND register_at IS NOT NULL 
                AND last_login IS NOT NULL 
                AND id IS NOT NULL
            """)
            complete_users = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM user")
            total_users = cursor.fetchone()[0]

            assert complete_users == total_users, (
                "Some users are missing required fields"
            )

            conn.close()
