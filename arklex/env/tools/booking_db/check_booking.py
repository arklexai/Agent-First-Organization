from typing import List, Dict, Any, Union
from ..tools import register_tool
from .utils import *

import pandas as pd


@register_tool(
    "Checks details of booked show(s)",
    [],
    [
        {
            "name": "query_result",
            "type": "str",
            "description": "A list of booked shows. If no booking exists, returns 'No bookings found.'",
        }
    ],
    lambda x: x and x not in (LOG_IN_FAILURE, "No bookings found."),
)
def check_booking() -> Union[str, None]:
    if not log_in():
        return LOG_IN_FAILURE

    logger.info("Enter check booking function")
    conn: sqlite3.Connection = sqlite3.connect(booking.db_path)
    cursor: sqlite3.Cursor = conn.cursor()

    query: str = """
    SELECT * FROM
        booking b
        JOIN show s ON b.show_id = s.id
    WHERE
        b.user_id = ?
    """
    cursor.execute(query, (booking.user_id,))
    rows: List[tuple] = cursor.fetchall()
    cursor.close()
    conn.close()

    response: str = "No bookings found."
    if len(rows) == 0:
        response = NO_BOOKING_MESSAGE
    else:
        column_names: List[str] = [column[0] for column in cursor.description]
        results: List[Dict[str, Any]] = [dict(zip(column_names, row)) for row in rows]
        results_df: pd.DataFrame = pd.DataFrame(results)
        response = "Booked shows are:\n" + results_df.to_string(index=False)
    return response
