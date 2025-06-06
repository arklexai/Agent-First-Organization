"""Type definitions for the Arklex framework.

This module contains enumerations and type definitions used throughout the Arklex framework,
particularly for handling different types of data streams and events. It provides standardized
type definitions for stream processing and event handling across the system.

Key Components:
1. StreamType: Defines different types of data streams (audio, text, speech)
2. EventType: Defines different types of events in stream processing

Usage:
    # Check stream type
    if stream_type == StreamType.AUDIO:
        process_audio_stream()

    # Handle events
    if event_type == EventType.CHUNK:
        process_data_chunk()
    elif event_type == EventType.LAST:
        finalize_processing()
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

    Example:
        stream_type = StreamType.AUDIO
        if stream_type == StreamType.AUDIO:
            configure_audio_processing()
    """

    # AUDIO is used to denote audio streams
    AUDIO = "audio"
    # TEXT is used to denote text streams
    TEXT = "text"
    # SPEECH is used to denote speech streams
    SPEECH = "speech"


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

    Example:
        event_type = EventType.CHUNK
        if event_type == EventType.CHUNK:
            process_data_chunk()
        elif event_type == EventType.ERROR:
            handle_error()
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
