import os
import sqlite3
import logging
from typing import Dict, Optional, Tuple
from langchain_openai import ChatOpenAI

from arklex.utils.model_config import MODEL

DBNAME: str = "show_booking_db.sqlite"
USER_ID: str = "user_be6e1836-8fe9-4938-b2d0-48f810648e72"

logger = logging.getLogger(__name__)

SLOTS: Dict[str, Dict[str, str]] = {
    "show_name": {
        "name": "show_name",
        "type": "string",
        "value": "",
        "description": "Name of the show",
        "prompt": "Please provide the name of the show",
    },
    "location": {
        "name": "location",
        "type": "string",
        "value": "",
        "description": "Location of the show",
        "prompt": "Please provide the location of the show",
    },
    "date": {
        "name": "date",
        "type": "date",
        "value": "",
        "description": "Date of the show",
        "prompt": "Please provide the date of the show",
    },
    "time": {
        "name": "time",
        "type": "time",
        "value": "",
        "description": "Time of the show",
        "prompt": "Please provide the time of the show",
    },
}

LOG_IN_FAILURE: str = (
    "Failed to login. Please ensure user ID and database information is correct"
)
NO_SHOW_MESSAGE: str = (
    "Show is not found. Please check whether the information is correct."
)
MULTIPLE_SHOWS_MESSAGE: str = (
    "There are multiple shows found. Please provide more details."
)
NO_BOOKING_MESSAGE: str = "You have not booked any show."


class Booking:
    db_path: Optional[str] = None
    llm: ChatOpenAI = ChatOpenAI(model=MODEL["model_type_or_path"], timeout=30000)
    user_id: str = USER_ID
    # actions = {
    #     "SearchShow": "Search for shows",
    #     "BookShow": "Book a show",
    #     "CheckBooking": "Check details of booked show(s)",
    #     "CancelBooking": "Cancel a booking",
    #     "Others": "Other actions not mentioned above"
    # }


booking: Booking = Booking()


def log_in() -> bool:
    booking.db_path = os.path.join(os.environ.get("DATA_DIR"), DBNAME)
    conn: sqlite3.Connection = sqlite3.connect(booking.db_path)
    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM user WHERE id = ?", (booking.user_id,))
    result: Optional[Tuple[int, ...]] = cursor.fetchone()
    if result is None:
        logger.info(f"User {booking.user_id} not found in the database.")
    else:
        logger.info(f"User {booking.user_id} successfully logged in.")
    return result is not None
