"""Database utility tools for the Arklex framework.

This module provides database utility tools and helper functions for managing show bookings
and related operations. It includes the DatabaseActions class for handling database
operations such as searching shows, booking shows, checking bookings, and canceling
bookings. The module also manages slot verification and user authentication for database
operations.
"""

import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from arklex.env.prompts import load_prompts
from arklex.orchestrator.entities.msg_state_entities import (
    MessageState,
    Slot,
    SlotDetail,
    StatusEnum,
)
from arklex.utils.logging_utils import LogContext
from arklex.utils.model_config import MODEL
from arklex.utils.utils import chunk_string

log_context = LogContext(__name__)
DBNAME: str = "show_booking_db.sqlite"
USER_ID: str = "user_be6e1836-8fe9-4938-b2d0-48f810648e72"
SLOTS: list[dict[str, str]] = [
    {
        "name": "show_name",
        "type": "string",
        "value": "",
        "description": "Name of the show",
        "prompt": "Please provide the name of the show",
    },
    {
        "name": "location",
        "type": "string",
        "value": "",
        "description": "Location of the show",
        "prompt": "Please provide the location of the show",
    },
    {
        "name": "date",
        "type": "date",
        "value": "",
        "description": "Date of the show",
        "prompt": "Please provide the date of the show",
    },
    {
        "name": "time",
        "type": "time",
        "value": "",
        "description": "Time of the show",
        "prompt": "Please provide the time of the show",
    },
]

NO_SHOW_MESSAGE: str = (
    "Show is not found. Please check whether the information is correct."
)
MULTIPLE_SHOWS_MESSAGE: str = (
    "There are multiple shows found. Please provide more details."
)
NO_BOOKING_MESSAGE: str = "You have not booked any show."


class DatabaseActions:
    def __init__(self, user_id: str = USER_ID) -> None:
        self.db_path: str = os.path.join(os.environ.get("DATA_DIR"), DBNAME)
        self.llm: ChatOpenAI = ChatOpenAI(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        self.user_id: str = user_id
        self.slots: list[SlotDetail] = []
        self.slot_prompts: list[str] = []

    def log_in(self) -> bool:
        conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM user WHERE id = ?", (self.user_id,))
        result: tuple[int, ...] | None = cursor.fetchone()
        if result is None:
            log_context.info(f"User {self.user_id} not found in the database.")
        else:
            log_context.info(f"User {self.user_id} successfully logged in.")
        return result is not None

    def init_slots(
        self, slots: list[Slot], bot_config: dict[str, Any]
    ) -> list[dict[str, str]]:
        if not slots:
            slots = SLOTS
        self.slots = []
        self.slot_prompts = []
        conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        cursor: sqlite3.Cursor = conn.cursor()
        for slot in slots:
            query: str = f"SELECT DISTINCT {slot['name']} FROM show"
            cursor.execute(query)
            results: list[tuple[Any, ...]] = cursor.fetchall()
            value_list: list[Any] = [result[0] for result in results]
            self.slots.append(self.verify_slot(slot, value_list, bot_config))
            if not self.slots[-1].confirmed:
                self.slot_prompts.append(slot["prompt"])
        cursor.close()
        conn.close()
        return SLOTS

    def verify_slot(
        self, slot: Slot, value_list: list[Any], bot_config: dict[str, Any]
    ) -> SlotDetail:
        slot_detail: SlotDetail = SlotDetail(**slot, verified_value="", confirmed=False)
        prompts: dict[str, str] = load_prompts(bot_config)
        prompt: PromptTemplate = PromptTemplate.from_template(
            prompts["database_slot_prompt"]
        )
        input_prompt: Any = prompt.invoke(
            {
                "slot": {
                    "name": slot["name"],
                    "description": slot["description"],
                    "slot": slot["type"],
                },
                "value": slot["value"],
                "value_list": value_list,
            }
        )
        chunked_prompt: list[str] = chunk_string(
            input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"]
        )
        log_context.info(f"Chunked prompt for verifying slot: {chunked_prompt}")
        final_chain: Any = self.llm | StrOutputParser()
        try:
            answer: str = final_chain.invoke(chunked_prompt)
            log_context.info(f"Result for verifying slot value: {answer}")
            for value in value_list:
                if value in answer:
                    log_context.info(
                        f"Chosen slot value in the database worker: {value}"
                    )
                    slot_detail.verified_value = value
                    slot_detail.confirmed = True
                    return slot_detail
        except Exception as e:
            log_context.error(
                f"Error occurred while verifying slot in the database worker: {e}"
            )
        return slot_detail

    def search_show(self, msg_state: MessageState) -> MessageState:
        # Populate the slots with verified values
        conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        cursor: sqlite3.Cursor = conn.cursor()
        query: str = "SELECT show_name, date, time, description, location, price FROM show WHERE 1 = 1"
        params: list[Any] = []
        for slot in self.slots:
            if slot.confirmed:
                query += f" AND {slot.name} = ?"
                params.append(slot.verified_value)
        query += " LIMIT 10"
        # Execute the query
        cursor.execute(query, params)
        rows: list[tuple[Any, ...]] = cursor.fetchall()
        cursor.close()
        conn.close()
        if len(rows) == 0:
            msg_state.status = StatusEnum.INCOMPLETE
            msg_state.message_flow = NO_SHOW_MESSAGE
        else:
            column_names: list[str] = [column[0] for column in cursor.description]
            results: list[dict[str, Any]] = [
                dict(zip(column_names, row, strict=False)) for row in rows
            ]
            results_df: pd.DataFrame = pd.DataFrame(results)
            msg_state.status = StatusEnum.COMPLETE
            msg_state.message_flow = "Available shows are:\n" + results_df.to_string(
                index=False
            )
        return msg_state

    def book_show(self, msg_state: MessageState) -> MessageState:
        log_context.info("Enter book show function")
        conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        cursor: sqlite3.Cursor = conn.cursor()
        query: str = "SELECT id, show_name, date, time, description, location, price FROM show WHERE 1 = 1"
        params: list[Any] = []
        for slot in self.slots:
            if slot.confirmed:
                query += f" AND {slot.name} = ?"
                params.append(slot.verified_value)
        # Execute the query
        cursor.execute(query, params)
        rows: list[tuple[Any, ...]] = cursor.fetchall()
        log_context.info(f"Rows found: {len(rows)}")
        # Check whether info is enough to book a show
        if len(rows) == 0:
            msg_state.status = StatusEnum.INCOMPLETE
            msg_state.message_flow = NO_SHOW_MESSAGE
        elif len(rows) > 1:
            msg_state.status = StatusEnum.INCOMPLETE
            if self.slot_prompts:
                msg_state.message_flow = self.slot_prompts[0]
            else:
                msg_state.message_flow = MULTIPLE_SHOWS_MESSAGE
        elif not self.slots or not any(slot.confirmed for slot in self.slots):
            # No specific show information provided
            msg_state.status = StatusEnum.INCOMPLETE
            msg_state.message_flow = "Please provide specific show information to book."
        else:
            column_names: list[str] = [column[0] for column in cursor.description]
            results: dict[str, Any] = dict(zip(column_names, rows[0], strict=False))
            show_id: str = results["id"]

            # Insert a row into the booking table
            cursor.execute(
                """
                INSERT INTO booking (id, show_id, user_id, created_at)
                VALUES (?, ?, ?, ?)
            """,
                ("booking_" + str(uuid.uuid4()), show_id, self.user_id, datetime.now()),
            )

            results_df: pd.DataFrame = pd.DataFrame([results])
            msg_state.status = StatusEnum.COMPLETE
            msg_state.message_flow = "The booked show is:\n" + results_df.to_string(
                index=False
            )
        cursor.close()
        conn.close()
        return msg_state

    def check_booking(self, msg_state: MessageState) -> MessageState:
        log_context.info("Enter check booking function")
        conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        cursor: sqlite3.Cursor = conn.cursor()

        query: str = """
        SELECT * FROM
            booking b
            JOIN show s ON b.show_id = s.id
        WHERE
            b.user_id = ?
        """
        cursor.execute(query, (self.user_id,))
        rows: list[tuple[Any, ...]] = cursor.fetchall()
        cursor.close()
        conn.close()
        if len(rows) == 0:
            msg_state.message_flow = NO_BOOKING_MESSAGE
        else:
            column_names: list[str] = [column[0] for column in cursor.description]
            results: list[dict[str, Any]] = [
                dict(zip(column_names, row, strict=False)) for row in rows
            ]
            results_df: pd.DataFrame = pd.DataFrame(results)
            msg_state.message_flow = "Booked shows are:\n" + results_df.to_string(
                index=False
            )
        msg_state.status = StatusEnum.COMPLETE
        return msg_state

    def cancel_booking(self, msg_state: MessageState) -> MessageState:
        log_context.info("Enter cancel booking function")
        conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        cursor: sqlite3.Cursor = conn.cursor()

        query: str = """
        SELECT * FROM
            booking b
            JOIN show s ON b.show_id = s.id
        WHERE
            b.user_id = ?
        """
        cursor.execute(query, (self.user_id,))
        rows: list[tuple[Any, ...]] = cursor.fetchall()
        if len(rows) == 0:
            msg_state.status = StatusEnum.COMPLETE
            msg_state.message_flow = NO_BOOKING_MESSAGE
        elif len(rows) > 1:
            msg_state.status = StatusEnum.INCOMPLETE
            if self.slot_prompts:
                msg_state.message_flow = self.slot_prompts[0]
            else:
                msg_state.message_flow = MULTIPLE_SHOWS_MESSAGE
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
                (show["show_id"],),
            )
            conn.commit()
            # Respond to user the cancellation
            results_df: pd.DataFrame = pd.DataFrame(results)
            msg_state.status = StatusEnum.COMPLETE
            msg_state.message_flow = "The cancelled show is:\n" + results_df.to_string(
                index=False
            )
        cursor.close()
        conn.close()
        return msg_state
