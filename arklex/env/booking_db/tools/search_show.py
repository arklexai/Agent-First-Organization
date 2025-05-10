from ..tools import register_tool
from .utils import *

import pandas as pd

@register_tool(
    "Searches the database for shows given descriptions",
    [{**SLOTS['show_name'], 'required': False}, 
     {**SLOTS['date'], 'required': False},
     {**SLOTS['time'], 'required': False},
     {**SLOTS['location'], 'required': False},
    ],
    [{
        "name": "query_result",
        "type": "str",
        "description": "A list of available shows that satisfies the given criteria (displays the first 10 results). If no show satisfies the criteria, returns 'No shows exist'",
    }],
    lambda x: x not in (LOG_IN_FAILURE or "No shows exist.")
)
def search_show(show_name=None, date=None, time=None, location=None) -> str | None:
    if not log_in(): return LOG_IN_FAILURE
    
    # Populate the slots with verified values
    conn = sqlite3.connect(booking.db_path)
    cursor = conn.cursor()
    query = "SELECT show_name, date, time, description, location, price, available_seats FROM show WHERE 1 = 1"
    params = []
    slots = {"show_name": show_name, "date": date, "time": time, "location": location}
    logger.info(f"{slots=}")
    for slot_name, slot_value in slots.items():
        if slot_value:
            query += f" AND {slot_name} = ?"
            params.append(slot_value)
    query += " LIMIT 10"
    
    # Execute the query
    cursor.execute(query, params)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    result = "No shows exist."
    if len(rows):
        column_names = [column[0] for column in cursor.description]
        results = [dict(zip(column_names, row)) for row in rows]
        results_df = pd.DataFrame(results)
        result = "Available shows are:\n" + results_df.to_string(index=False)
    return result