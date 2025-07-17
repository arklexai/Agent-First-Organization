"""
Acuity Scheduling tools package for the Arklex framework.

This package contains tool implementations for managing appointments, scheduling, and related operations using Acuity Scheduling in the Arklex framework.
"""

from ._exception_prompt import AcuityExceptionPrompt
from .book_info_session import book_info_session
from .cancel import cancel
from .get_apt_by_email import get_apt_by_email
from .get_available_dates import get_available_dates
from .get_available_times import get_available_times
from .get_session_types import get_session_types
from .get_type_id_by_apt_name import get_type_id_by_apt_name
from .reschedule import reschedule
from .utils import get_acuity_client

# Export the exception prompt constants for backward compatibility
_exception_prompt = AcuityExceptionPrompt()

__all__ = [
    "AcuityExceptionPrompt",
    "_exception_prompt",
    "book_info_session",
    "cancel",
    "get_acuity_client",
    "get_apt_by_email",
    "get_available_dates",
    "get_available_times",
    "get_session_types",
    "get_type_id_by_apt_name",
    "reschedule",
]
