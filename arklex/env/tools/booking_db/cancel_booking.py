from typing import List, Dict, Any, Union
from ..tools import register_tool
from .utils import *

import pandas as pd


@register_tool(
    "Cancels existing booking",
    [],
    [
        {
            "name": "query_result",
            "type": "str",
            "description": "A string listing the show that was cancelled",
        }
    ],
    lambda x: x and x not in (LOG_IN_FAILURE, NO_BOOKING_MESSAGE),
)
def cancel_booking() -> Union[str, None]:
    if not log_in():
        return LOG_IN_FAILURE

    logger.info("Enter cancel booking function")
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

    response: str = ""
    if len(rows) == 0:
        response = NO_BOOKING_MESSAGE
    elif len(rows) > 1:
        response = MULTIPLE_SHOWS_MESSAGE
    else:
        column_names: List[str] = [column[0] for column in cursor.description]
        results: List[Dict[str, Any]] = [dict(zip(column_names, row)) for row in rows]
        show: Dict[str, Any] = results[0]
        # Delete a row from the booking table based on show_id
        cursor.execute(
            """DELETE FROM booking WHERE show_id = ?
        """,
            (show["id"],),
        )
        # Respond to user the cancellation
        results_df: pd.DataFrame = pd.DataFrame(results)
        response = "The cancelled show is:\n" + results_df.to_string(index=False)

    conn.close()
    cursor.commit()

    return response
