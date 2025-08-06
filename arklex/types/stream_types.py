"""Type definitions for the Arklex framework.

This module contains type definitions and enums used throughout the Arklex framework.
"""

from enum import Enum


class StreamType(str, Enum):
    """Enumeration of supported stream types.

    This enum defines the different types of data streams that can be processed
    by the framework. Each type represents a specific kind of data flow that
    requires specialized handling.

    Values:
        AUDIO: Audio data streams (e.g., voice input/output)
        TEXT: Text data streams (e.g., chat messages)
        SPEECH: Speech data streams (e.g., speech-to-text)
        OPENAI_REALTIME_AUDIO: OpenAI real-time audio streams
        NON_STREAM: Non-stream types
        STREAM: Stream types
    """

    # NON_STREAM is used to denote non-stream types
    NON_STREAM = "non_stream"
    # TEXT is used to denote text streams
    TEXT = "text"
    # SPEECH is used to denote speech streams
    SPEECH = "speech"
    # OPENAI_REALTIME_AUDIO is used to denote OpenAI real-time audio streams
    OPENAI_REALTIME_AUDIO = "openai_realtime_audio"


class EventType(str, Enum):
    """Enumeration of event types in stream processing.

    This enum defines the different types of events that can occur during
    stream processing. Each type represents a specific event that requires
    different handling in the processing pipeline.

    Values:
        LAST: Final event in a stream
        CHUNK: Regular data chunk in the stream
        TEXT: Text-only data chunk
        AUDIO_CHUNK: Audio data chunk
        ERROR: Error event in the stream
    """

    # LAST is used to denote the last event in the stream
    LAST = "last"
    # CHUNK is used to denote a chunk of data in the stream
    CHUNK = "chunk"
    # TEXT is used to denote a chunk of text-only data in the stream
    TEXT = "text"
    # AUDIO is used to denote a chunk of audio
    AUDIO_CHUNK = "audio"
    # ERROR is used to denote an error
    ERROR = "error"
