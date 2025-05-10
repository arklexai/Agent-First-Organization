from ..tools import register_tool
from .utils import *

import pandas as pd

@register_tool(
    "Cancels existing booking",
    [],
    [{
        "name": "query_result",
        "type": "str",
        "description": "A string listing the show that was cancelled",
    }],
    lambda x: x and x not in (LOG_IN_FAILURE, NO_BOOKING_MESSAGE)
)
def cancel_booking() -> str | None:
    if not log_in(): return LOG_IN_FAILURE  
    
    logger.info("Enter cancel booking function")
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
    
    response = ''
    if len(rows) == 0:
        response = NO_BOOKING_MESSAGE
    elif len(rows) > 1:
        response = MULTIPLE_SHOWS_MESSAGE
    else:
        column_names = [column[0] for column in cursor.description]
        results = [dict(zip(column_names, row)) for row in rows]
        show = results[0]
        # Delete a row from the booking table based on show_id
        cursor.execute('''DELETE FROM booking WHERE show_id = ?
        ''', (show["id"],))
        # Respond to user the cancellation
        results_df = pd.DataFrame(results)
        response = "The cancelled show is:\n" + results_df.to_string(index=False)
        
    conn.close()
    cursor.commit()
    
    return response
