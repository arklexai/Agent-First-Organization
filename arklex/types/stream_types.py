"""Type definitions for the Arklex framework.

This module contains type definitions and enums used throughout the Arklex framework.
"""

from enum import Enum

from pydantic import BaseModel


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

    # AUDIO is used to denote audio streams
    AUDIO = "audio"
    # TEXT is used to denote text streams
    TEXT = "text"
    # SPEECH is used to denote speech streams
    SPEECH = "speech"
    # OPENAI_REALTIME_AUDIO is used to denote OpenAI real-time audio streams
    OPENAI_REALTIME_AUDIO = "openai_realtime_audio"
    # NON_STREAM is used to denote non-stream types
    NON_STREAM = "non_stream"
    # STREAM is used to denote stream types
    STREAM = "stream"


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


class LLMConfig(BaseModel):
    """Configuration for language model settings.

    This class defines the configuration parameters for language models used in the system.
    It specifies which model to use and from which provider.

    The class provides:
    1. Model selection and configuration
    2. Provider specification
    3. Type-safe configuration management

    Attributes:
        model_type_or_path (str): The model identifier or path to use.
        llm_provider (str): The provider of the language model (e.g., 'openai', 'anthropic').
    """

    model_type_or_path: str
    llm_provider: str
