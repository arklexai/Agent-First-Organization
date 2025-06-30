import sqlite3
from typing import Any

import pandas as pd

from arklex.env.tools.booking_db.utils import (
    LOG_IN_FAILURE,
    MULTIPLE_SHOWS_MESSAGE,
    NO_BOOKING_MESSAGE,
    booking,
    log_in,
)
from arklex.env.tools.tools import register_tool
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


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
def cancel_booking() -> str | None:
    if not log_in():
        return LOG_IN_FAILURE

    log_context.info("Enter cancel booking function")
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
    rows: list[tuple] = cursor.fetchall()

    response: str = ""
    if len(rows) == 0:
        response = NO_BOOKING_MESSAGE
    elif len(rows) > 1:
        response = MULTIPLE_SHOWS_MESSAGE
    else:
        column_names: list[str] = [column[0] for column in cursor.description]
        results: list[dict[str, Any]] = [
            dict(zip(column_names, row, strict=False)) for row in rows
        ]
        show: dict[str, Any] = results[0]
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
