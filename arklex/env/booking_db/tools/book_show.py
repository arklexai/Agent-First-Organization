from typing import List, Dict, Any, Optional, Union
from ..tools import register_tool
from .utils import *

import datetime
import uuid
import pandas as pd


@register_tool(
    "Books an existing show",
    [
        {**SLOTS["show_name"], "required": False},
        {**SLOTS["date"], "required": False},
        {**SLOTS["time"], "required": False},
        {**SLOTS["location"], "required": False},
    ],
    [
        {
            "name": "query_result",
            "type": "str",
            "description": "A list of available shows that satisfies the given criteria (displays the first 10 results). If no show satisfies the criteria, returns 'No shows exist'",
        }
    ],
    lambda x: x and x not in (LOG_IN_FAILURE, NO_SHOW_MESSAGE, MULTIPLE_SHOWS_MESSAGE),
)
def book_show(
    show_name: Optional[str] = None,
    date: Optional[str] = None,
    time: Optional[str] = None,
    location: Optional[str] = None,
) -> Union[str, None]:
    if not log_in():
        return LOG_IN_FAILURE

    logger.info("Enter book show function")
    conn: sqlite3.Connection = sqlite3.connect(booking.db_path)
    cursor: sqlite3.Cursor = conn.cursor()
    query: str = "SELECT id, show_name, date, time, description, location, price FROM show WHERE 1 = 1"
    params: List[str] = []
    slots: Dict[str, Optional[str]] = {
        "show_name": show_name,
        "date": date,
        "time": time,
        "location": location,
    }
    logger.info(f"{slots=}")
    for slot_name, slot_value in slots.items():
        if slot_value:
            query += f" AND {slot_name} = ?"
            params.append(slot_value)

    # Execute the query
    cursor.execute(query, params)
    rows: List[tuple] = cursor.fetchall()
    logger.info(f"Rows found: {len(rows)}")

    response: Optional[str] = None
    # Check whether info is enough to book a show
    if len(rows) == 0:
        response = NO_SHOW_MESSAGE
    elif len(rows) > 1:
        response = MULTIPLE_SHOWS_MESSAGE
    else:
        column_names: List[str] = [column[0] for column in cursor.description]
        results: Dict[str, Any] = dict(zip(column_names, rows[0]))
        show_id: str = results["id"]

        # Insert a row into the booking table
        cursor.execute(
            """
            INSERT INTO booking (id, show_id, user_id, created_at)
            VALUES (?, ?, ?, ?)
        """,
            (
                "booking_" + str(uuid.uuid4()),
                show_id,
                booking.user_id,
                datetime.datetime.now(),
            ),
        )

        results_df: pd.DataFrame = pd.DataFrame([results])
        response = "The booked show is:\n" + results_df.to_string(index=False)

    cursor.close()
    conn.close()
    return response
