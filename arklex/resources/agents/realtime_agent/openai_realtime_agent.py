import asyncio
import base64
import datetime
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from typing import Any, Literal

import numpy as np
from agents.realtime import (
    RealtimeAgent,
    RealtimeInputAudioTranscriptionConfig,
    RealtimeModelConfig,
    RealtimeModelRawClientMessage,
    RealtimeModelSendRawMessage,
    RealtimeModelSendSessionUpdate,
    RealtimePlaybackTracker,
    RealtimeRunConfig,
    RealtimeRunner,
    RealtimeSession,
    RealtimeSessionModelSettings,
)
from pydantic import BaseModel

from arklex.resources.agents.base.agent import BaseAgent, register_agent
from arklex.resources.agents.base.entities import PromptVariable
from arklex.resources.tools.tools import Tool
from arklex.resources.tools.types import ChatRole, Transcript

logger = logging.getLogger(__name__)


class TurnDetection(BaseModel):
    """
    Turn detection is a configuration for the turn detection of the agent.

    Valid types are:
    - "server_vad": Server-side voice activity detection
    - "semantic_vad": Semantic voice activity detection
    """

    type: Literal["server_vad", "semantic_vad"] | None = "server_vad"
    create_response: bool | None = True
    interrupt_response: bool | None = True
    prefix_padding_ms: int | None = 300
    silence_duration_ms: int | None = 750
    threshold: float | None = 0.5
    eagerness: str | None = None

    def model_dump(self) -> dict:
        """
        Convert the TurnDetection object to a dictionary.

        Returns:
            dict: Dictionary representation of turn detection configuration

        Note:
            For "server_vad" type, eagerness is removed.
            For "semantic_vad" type, prefix_padding_ms, silence_duration_ms, and threshold are removed.
        """
        # super call to the base class
        data = super().model_dump()
        if self.type == "server_vad":
            del data["eagerness"]
        elif self.type == "semantic_vad":
            del data["prefix_padding_ms"]
            del data["silence_duration_ms"]
            del data["threshold"]
        return data


class OpenAIRealtimeAgentData(BaseModel):
    """Data for the OpenAIAgent."""

    name: str
    prompt: str
    prompt_variables: list[PromptVariable] = []
    handoff_description: str | None = None
    start_agent: bool = False
    voice: str = "alloy"
    transcription_language: str | None = None
    speed: float = 1.0
    turn_detection: TurnDetection | None = None


@register_agent
class OpenAIRealtimeAgent(BaseAgent):
    """
    OpenAI Realtime Agent is an agent that uses the OpenAI Realtime API to interact with the user.

    This agent supports real-time audio and text interactions with OpenAI's GPT-4o Realtime model.
    It can handle audio streaming, transcription, tool calls, and turn detection for natural
    conversation flow.
    """

    def __init__(
        self,
        realtime_agent: RealtimeAgent,
        telephony_mode: bool = False,
        voice: str = "alloy",
        transcription_language: str | None = None,
        speed: float = 1.0,
        turn_detection: TurnDetection | None = None,
        voicemail_tool: Tool | None = None,
    ) -> None:
        """
        Initialize the OpenAI Realtime Agent.

        Args:
            realtime_agent: The RealtimeAgent instance to use for the session
            telephony_mode: Whether the agent is in telephone mode (uses g711_ulaw audio format)
            voice: The voice for the agent (default: "alloy")
            transcription_language: The language for the transcription (optional)
            speed: The speech speed for the agent (default: 1.0)
            turn_detection: The turn detection configuration for the agent
            voicemail_tool: Optional tool to execute when a call goes to voicemail. This is triggered by
            twilio AMD.
        """
        self.ws = None
        self.modalities: list[str] = ["text"]
        self.voice = voice
        self.speed = speed
        self.turn_detection = turn_detection
        self.internal_queue: asyncio.Queue = asyncio.Queue()
        self.external_queue: asyncio.Queue = asyncio.Queue()
        self.input_audio_buffer_event_queue: asyncio.Queue = asyncio.Queue()
        self.text_buffer = defaultdict(str)
        self.telephony_mode = telephony_mode
        self.input_audio_format = "g711_ulaw" if telephony_mode else "pcm16"
        self.output_audio_format = "g711_ulaw" if telephony_mode else "pcm16"
        self.transcript = []
        self.transcript_available: asyncio.Event = asyncio.Event()
        self.call_sid = None
        # this event is used to signal that the audio response has finished playing through twilio
        self.response_played: threading.Event = threading.Event()
        self.transcription_language = transcription_language
        self.session: RealtimeSession | None = None
        self.session_context: RealtimeSession | None = None
        self.realtime_agent = realtime_agent
        # Rate limiting for audio_delta logging (log once every 2 seconds)
        self.last_audio_delta_log_time = 0.0
        self.voicemail_tool = voicemail_tool
        self.playback_tracker = RealtimePlaybackTracker()

        # Mark event tracking for playback
        self._mark_counter = 0
        self._mark_data: dict[
            str, tuple[str, int, int]
        ] = {}  # mark_id -> (item_id, content_index, byte_count)

    def set_telephone_mode(self) -> None:
        """
        Enable telephone mode for the agent.

        This sets the audio format to g711_ulaw which is commonly used in telephony systems.
        """
        self.telephony_mode = True
        self.input_audio_format = "g711_ulaw"
        self.output_audio_format = "g711_ulaw"

    def set_audio_modality(self) -> None:
        """
        Enable audio modality for the agent.

        This allows the agent to both receive and send audio in addition to text.
        """
        self.modalities = ["text", "audio"]

    def set_text_modality(self) -> None:
        """
        Set the agent to text-only modality.

        This disables audio processing and limits the agent to text interactions only.
        """
        self.modalities = ["text"]

    async def connect(self) -> None:
        """
        Initialize and start a Realtime session using the RealtimeRunner.

        This method creates a RealtimeRunner with the configured realtime_agent and
        establishes a session with the OpenAI Realtime API using the specified
        model configuration and settings.

        Raises:
            Exception: If OPENAI_API_KEY environment variable is not set
            Exception: If session initialization fails
        """

        runner = RealtimeRunner(
            starting_agent=self.realtime_agent,
            config=RealtimeRunConfig(
                model_settings=RealtimeSessionModelSettings(
                    model_name="gpt-4o-realtime-preview-2025-06-03",
                    input_audio_format=self.input_audio_format,
                    output_audio_format=self.output_audio_format,
                    input_audio_transcription=RealtimeInputAudioTranscriptionConfig(
                        model="gpt-4o-transcribe",
                        language=self.transcription_language,
                    ),
                    voice=self.voice,
                    speed=self.speed,
                    turn_detection=self.turn_detection.model_dump()
                    if self.turn_detection
                    else None,
                ),
            ),
        )
        self.session_context = await runner.run(
            model_config=RealtimeModelConfig(
                playback_tracker=self.playback_tracker,
            )
        )
        self.session = await self.session_context.__aenter__()

    async def close(self) -> None:
        """
        Close the WebSocket connection to OpenAI Realtime API.
        """
        await self.session_context.__aexit__(None, None, None)

    async def send_audio(self, audio: bytes) -> None:
        """
        Send audio data to the OpenAI Realtime API.

        Args:
            audio: Audio data to send. Should be in the format of pcm16 or g711_ulaw.
        """
        await self.session.send_audio(audio)

    async def truncate_audio(self, item_id: str, audio_end_ms: int) -> None:
        """
        Truncate audio at a specific time point.

        Args:
            item_id: The ID of the conversation item to truncate
            audio_end_ms: The end time in milliseconds where audio should be truncated
        """
        logger.info(f"Truncating audio for item_id: {item_id} at {audio_end_ms} ms")
        try:
            await self.session.model.send_event(
                RealtimeModelSendRawMessage(
                    message=RealtimeModelRawClientMessage(
                        type="conversation.item.truncate",
                        item_id=item_id,
                        content_index=0,
                        audio_end_ms=audio_end_ms,
                    ),
                )
            )
        except Exception as e:
            logger.exception(e)

    async def commit_audio(self) -> None:
        """
        Commit the current audio buffer to be processed by the API.

        This signals that the current audio input is complete and ready for processing.
        """
        await self.session.model.send_event(
            RealtimeModelSendRawMessage(
                message=RealtimeModelRawClientMessage(
                    type="input_audio_buffer.commit",
                ),
            )
        )

    async def create_response(self) -> None:
        """
        Request the creation of a response from the OpenAI model.

        This triggers the model to generate a response based on the current conversation context.
        """
        logger.info("Creating response")
        await self.session.model.send_event(
            RealtimeModelSendRawMessage(
                message=RealtimeModelRawClientMessage(
                    type="response.create",
                ),
            )
        )

    async def run_voicemail_tool(self) -> None:
        """
        Execute a voicemail tool when a call goes to voicemail. This is triggered by
        twilio AMD.

        This method configures the realtime session to handle voicemail scenarios by:
        1. Updating the session instructions to leave a specific voicemail message
        2. Clearing all tools to prevent further interactions
        3. Executing the voicemail tool with the configured message and call context

        Raises:
            ValueError: If no voicemail tool is configured

        Note:
            The method uses the voicemail tool's fixed_args to get the message content
            and passes the call_sid and response_played_event for proper call handling.
        """

        if self.voicemail_tool is None:
            raise ValueError("Voicemail tool not found")
        message = self.voicemail_tool.fixed_args["message"]
        logger.info(f"Running voicemail tool with message: {message}")
        self.response_played.clear()
        await self.session.model.send_event(
            RealtimeModelSendSessionUpdate(
                session_settings=RealtimeSessionModelSettings(
                    instructions=f"The call has gone to voicemail. Leave the exact following message in the language of the text: {message}",
                    tools=[],
                    voice=self.voice,
                    speed=self.speed,
                    turn_detection=self.turn_detection.model_dump()
                    if self.turn_detection
                    else None,
                    input_audio_format=self.input_audio_format,
                    output_audio_format=self.output_audio_format,
                    input_audio_transcription=RealtimeInputAudioTranscriptionConfig(
                        model="gpt-4o-transcribe",
                        language=self.transcription_language,
                    ),
                ),
            )
        )
        combined_kwargs = {**self.voicemail_tool.fixed_args}
        combined_kwargs["call_sid"] = self.call_sid
        combined_kwargs["response_played_event"] = self.response_played
        logger.info(f"Running voicemail tool with kwargs: {combined_kwargs}")
        await asyncio.to_thread(
            self.voicemail_tool.func, self.voicemail_tool.auth, **combined_kwargs
        )

    async def _handle_mark_event(self, message: dict[str, Any]) -> None:
        """Handle mark events from Twilio to update playback tracker."""
        try:
            mark_data = message.get("mark", {})
            mark_id = mark_data.get("name", "")

            # Look up stored data for this mark ID
            if mark_id in self._mark_data:
                item_id, item_content_index, byte_count = self._mark_data[mark_id]

                # Convert byte count back to bytes for playback tracker
                audio_bytes = b"\x00" * byte_count  # Placeholder bytes

                # Update playback tracker
                self.playback_tracker.on_play_bytes(
                    item_id, item_content_index, audio_bytes
                )

                # Clean up the stored data
                del self._mark_data[mark_id]
                if len(self._mark_data) == 0:
                    logger.info("All mark data deleted. Setting response_played event.")
                    self.response_played.set()

        except Exception as e:
            logger.error(f"Error handling mark event: {e}")
            logger.exception(e)

    async def receive_events(self) -> None:
        """
        Main event loop for receiving and processing events from the OpenAI Realtime session.

        This method iterates over events from the Realtime session and routes them to appropriate
        handlers based on event type. It processes various events including:
        - Response completion and tool calls
        - Audio streaming and transcription
        - Input audio buffer events
        - Conversation item creation
        - Function call arguments

        Note:
            This method runs continuously until the session is closed.
            It automatically handles error events and logs them appropriately.
        """
        try:
            async for event in self.session:
                if event.type == "error":
                    logger.error(f"Error from OpenAI: {event}")
                    continue

                if event.type == "agent_start":
                    logger.info(f"agent_start: {event.agent.name}")
                elif event.type == "agent_end":
                    logger.info(f"agent_end: {event.agent.name}")
                elif event.type == "handoff":
                    logger.info(
                        f"handoff: {event.from_agent.name} -> {event.to_agent.name}"
                    )
                    await self.external_queue.put(
                        {
                            "type": "message",
                            "origin": ChatRole.TOOL,
                            "id": str(uuid.uuid4()),
                            "text": f"Handing off from {event.from_agent.name} to {event.to_agent.name}",
                            "audio_url": "",
                            "debug": True,
                        }
                    )
                elif event.type == "tool_end":
                    text = json.dumps(
                        {
                            "function_name": event.tool.name,
                            "response": str(event.output),
                        }
                    )
                    self.transcript.append(
                        Transcript(
                            id=str(uuid.uuid4()),
                            text=text,
                            origin=ChatRole.TOOL,
                            created_at=datetime.datetime.now(datetime.timezone.utc),
                        )
                    )
                    await self.external_queue.put(
                        {
                            "type": "message",
                            "origin": ChatRole.TOOL,
                            "id": str(uuid.uuid4()),
                            "text": text,
                            "audio_url": "",
                            "debug": True,
                        }
                    )
                elif event.type == "audio":
                    # Send mark event for playback tracking
                    self._mark_counter += 1
                    mark_id = str(self._mark_counter)
                    self._mark_data[mark_id] = (
                        event.audio.item_id,
                        event.audio.content_index,
                        len(event.audio.data),
                    )
                    await self.external_queue.put(
                        {
                            "type": "audio_stream",
                            "origin": "bot",
                            "id": event.item_id,
                            "audio_bytes": base64.b64encode(event.audio.data).decode(
                                "utf-8"
                            )
                            if self.telephony_mode
                            else np.frombuffer(event.audio.data, np.int16).tolist(),
                            "mark_id": mark_id,
                        }
                    )

                elif event.type == "audio_interrupted":
                    logger.info("audio_interrupted")
                    await self.external_queue.put({"type": "audio_interrupted"})

                elif event.type == "raw_model_event":
                    if event.data.type == "audio":
                        pass
                    elif event.data.type == "raw_server_event":
                        event_type = event.data.data.get("type", "")

                        if event_type == "response.audio.delta":
                            # Rate limit logging to once every 2 seconds
                            current_time = time.time()
                            if current_time - self.last_audio_delta_log_time >= 2.0:
                                logger.info("raw_model_event audio_delta")
                                self.last_audio_delta_log_time = current_time
                        elif event_type == "response.audio_transcript.delta":
                            openai_event = event.data.data
                            if "delta" in openai_event and not self.telephony_mode:
                                self.text_buffer[openai_event["item_id"]] += (
                                    openai_event["delta"]
                                )
                                event = {
                                    "type": "text_stream",
                                    "origin": ChatRole.BOT,
                                    "id": openai_event["item_id"],
                                    "text": self.text_buffer[openai_event["item_id"]],
                                }
                                await self.external_queue.put(event)
                        elif event_type == "response.audio_transcript.done":
                            transcript_text = event.data.data.get("transcript", "")
                            self.transcript.append(
                                Transcript(
                                    id=str(uuid.uuid4()),
                                    text=transcript_text,
                                    origin="bot",
                                    created_at=datetime.datetime.now(
                                        datetime.timezone.utc
                                    ),
                                )
                            )
                            await self.external_queue.put(
                                {
                                    "type": "message",
                                    "origin": ChatRole.BOT,
                                    "id": event.data.data.get("item_id", ""),
                                    "text": transcript_text,
                                    "audio_url": "",
                                }
                            )
                        elif event_type in [
                            "input_audio_buffer.speech_started",
                            "input_audio_buffer.speech_stopped",
                        ]:
                            await self.input_audio_buffer_event_queue.put(
                                event.data.data
                            )
                            await self.external_queue.put({"type": event_type})
                        elif event_type == "input_audio_buffer.committed":
                            await self.input_audio_buffer_event_queue.put(
                                event.data.data
                            )
                        elif event_type == "conversation.item.created":
                            item = event.data.data.get("item", {})
                            if item.get("role") in ["user", "assistant"]:
                                await self.external_queue.put(
                                    {
                                        "type": "message",
                                        "origin": ChatRole.USER
                                        if item["role"] == "user"
                                        else ChatRole.BOT,
                                        "id": item["id"],
                                        "text": " ",
                                        "audio_url": "",
                                    }
                                )
                        elif (
                            event_type
                            == "conversation.item.input_audio_transcription.completed"
                        ):
                            transcript_text = event.data.data["transcript"]
                            self.transcript.append(
                                Transcript(
                                    id=str(uuid.uuid4()),
                                    text=transcript_text,
                                    origin="user",
                                    created_at=datetime.datetime.now(
                                        datetime.timezone.utc
                                    ),
                                )
                            )
                            await self.external_queue.put(
                                {
                                    "type": "message",
                                    "origin": "user",
                                    "id": event.data.data["item_id"],
                                    "text": transcript_text,
                                    "audio_url": "",
                                }
                            )
                        elif event_type == "response.function_call_arguments.done":
                            await self.internal_queue.put(event.data.data)
                            text = json.dumps(
                                {
                                    "function_name": event.data.data.get("name", ""),
                                    "arguments": event.data.data.get("arguments", ""),
                                }
                            )
                            self.transcript.append(
                                Transcript(
                                    id=str(uuid.uuid4()),
                                    text=text,
                                    origin=ChatRole.TOOL,
                                    created_at=datetime.datetime.now(
                                        datetime.timezone.utc
                                    ),
                                )
                            )
                            await self.external_queue.put(
                                {
                                    "type": "message",
                                    "origin": ChatRole.TOOL,
                                    "id": event.data.data.get("item_id", ""),
                                    "text": text,
                                    "audio_url": "",
                                    "debug": True,
                                }
                            )
                    elif event.data.type != "item_updated":
                        logger.debug(f"raw_model_event: {event}")

                elif event.type != "history_updated":
                    logger.debug(f"event: {event.type}")

        except Exception as e:
            logger.error(f"Error in receive_events: {e}")
            logger.exception(e)

        logger.info("receive_events ended")
        await self.end_queues()
        await self.close()

    async def end_queues(self) -> None:
        """
        Signal the end of all queues by sending None to each queue.

        This method is called when the WebSocket connection is closed to properly
        terminate any waiting consumers of the queues.
        """
        await self.internal_queue.put(None)
        await self.input_audio_buffer_event_queue.put(None)
        await self.external_queue.put(None)
