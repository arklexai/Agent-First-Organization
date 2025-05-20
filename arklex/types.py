from enum import Enum
from typing import Final


# Enum for the different types of streams that can be used.
class StreamType(str, Enum):
    """
    Enum for the different types of streams that can be used.

    Attributes:
        AUDIO (str): Denotes audio streams.
        TEXT (str): Denotes text streams.
    """

    # AUDIO is used to denote audio streams
    AUDIO: Final[str] = "audio"
    # TEXT is used to denote text streams
    TEXT: Final[str] = "text"


# Enum for event types used when streaming data.
class EventType(str, Enum):
    """
    Enum for event types used when streaming data.

    Attributes:
        LAST (str): Denotes the last event in the stream.
        CHUNK (str): Denotes a chunk of data in the stream.
        TEXT (str): Denotes a chunk of text-only data in the stream.
        AUDIO_CHUNK (str): Denotes a chunk of audio.
        ERROR (str): Denotes an error.
    """

    # LAST is used to denote the last event in the stream
    LAST: Final[str] = "last"
    # CHUNK is used to denote a chunk of data in the stream
    CHUNK: Final[str] = "chunk"
    # TEXT is used to denote a chunk of text-only data in the stream
    TEXT: Final[str] = "text"
    # AUDIO is used to denote a chunk of audio
    AUDIO_CHUNK: Final[str] = "audio"
    # ERROR is used to denote an error
    ERROR: Final[str] = "error"
