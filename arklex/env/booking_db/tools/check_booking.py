from ..tools import register_tool
from .utils import *

import pandas as pd

@register_tool(
    "Checks details of booked show(s)",
    [],
    [{
        "name": "query_result",
        "type": "str",
        "description": "A list of booked shows. If no booking exists, returns 'No bookings found.'",
    }],
    lambda x: x and x not in (LOG_IN_FAILURE, 'No bookings found.')
)
def check_booking() -> str | None:
    if not log_in(): return LOG_IN_FAILURE
    
    logger.info("Enter check booking function")
    conn = sqlite3.connect(booking.db_path)
    cursor = conn.cursor()

    query = """
    SELECT * FROM
        booking b
        JOIN show s ON b.show_id = s.id
    WHERE
        b.user_id = ?
    """
    cursor.execute(query, (booking.user_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    response = "No bookings found."
    if len(rows) == 0:
        response = NO_BOOKING_MESSAGE
    else:
        column_names = [column[0] for column in cursor.description]
        results = [dict(zip(column_names, row)) for row in rows]
        results_df = pd.DataFrame(results)
        response = "Booked shows are:\n" + results_df.to_string(index=False)
    return response