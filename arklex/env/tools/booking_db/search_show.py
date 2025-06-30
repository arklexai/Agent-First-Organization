import sqlite3
from typing import Any

import pandas as pd

from arklex.env.tools.booking_db.utils import (
    LOG_IN_FAILURE,
    SLOTS,
    booking,
    log_in,
)
from arklex.env.tools.tools import register_tool
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


@register_tool(
    "Searches the database for shows given descriptions",
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
    lambda x: x not in (LOG_IN_FAILURE or "No shows exist."),
)
def search_show(
    show_name: str | None = None,
    date: str | None = None,
    time: str | None = None,
    location: str | None = None,
) -> str | None:
    if not log_in():
        return LOG_IN_FAILURE

    # Populate the slots with verified values
    conn: sqlite3.Connection = sqlite3.connect(booking.db_path)
    cursor: sqlite3.Cursor = conn.cursor()
    query: str = "SELECT show_name, date, time, description, location, price, available_seats FROM show WHERE 1 = 1"
    params: list[str] = []
    slots: dict[str, str | None] = {
        "show_name": show_name,
        "date": date,
        "time": time,
        "location": location,
    }
    log_context.info(f"{slots=}")
    for slot_name, slot_value in slots.items():
        if slot_value:
            query += f" AND {slot_name} = ?"
            params.append(slot_value)
    query += " LIMIT 10"

    # Execute the query
    cursor.execute(query, params)
    rows: list[tuple] = cursor.fetchall()
    cursor.close()
    conn.close()
    result: str = "No shows exist."
    if len(rows):
        column_names: list[str] = [column[0] for column in cursor.description]
        results: list[dict[str, Any]] = [
            dict(zip(column_names, row, strict=False)) for row in rows
        ]
        results_df: pd.DataFrame = pd.DataFrame(results)
        result = "Available shows are:\n" + results_df.to_string(index=False)
    return result
