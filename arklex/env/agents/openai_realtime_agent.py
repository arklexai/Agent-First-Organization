import asyncio
import base64
import datetime
import json
import logging
import os
import threading
import uuid
from collections import defaultdict
from typing import Literal

import numpy as np
import websockets
from jinja2 import Template
from pydantic import BaseModel

from arklex.env.agents.agent import BaseAgent, register_agent
from arklex.env.tools.tools import Tool
from arklex.env.tools.types import Transcript

logger = logging.getLogger(__name__)


class PromptVariable(BaseModel):
    """
    Prompt variable is a variable that is used in the prompt. e.g. Say hello {{name}}
    """

    name: str
    value: str = ""


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

    @classmethod
    def from_dict(cls, data: dict | None) -> "TurnDetection":
        """
        Convert a dictionary to a TurnDetection object.

        Args:
            data: Dictionary containing turn detection configuration

        Returns:
            TurnDetection: Configured turn detection object

        Note:
            If data is None, returns default configuration.
            If type is "server_vad", eagerness is set to None.
            Valid types are "server_vad" or "semantic_vad".
        """
        if data is None:
            return cls()
        elif data.get("type") == "server_vad":
            return cls(
                type="server_vad",
                eagerness=None,
                **{k: v for k, v in data.items() if k != "type"},
            )
        elif data.get("type") == "semantic_vad":
            return cls(
                type="semantic_vad", **{k: v for k, v in data.items() if k != "type"}
            )
        else:
            # If no valid type is specified, return default configuration
            return cls()

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
        telephony_mode: bool = False,
        prompt: str = "",
        voice: str = "alloy",
        transcription_language: str | None = None,
        speed: float = 1.0,
        turn_detection: TurnDetection | None = None,
        tool_map: dict[str, Tool] = None,
        prompt_variables: list[PromptVariable] | None = None,
    ) -> None:
        """
        Initialize the OpenAI Realtime Agent.

        Args:
            telephony_mode: Whether the agent is in telephone mode (uses g711_ulaw audio format)
            prompt: The prompt for the agent (supports Jinja2 templating)
            voice: The voice for the agent (default: "alloy")
            transcription_language: The language for the transcription (optional)
            speed: The speech speed for the agent (default: 1.0)
            turn_detection: The turn detection configuration for the agent
            tool_map: Dictionary mapping tool names to Tool objects
            prompt_variables: List of PromptVariable objects for template rendering
        """
        self.ws = None
        self.modalities: list[str] = ["text"]
        self.prompt_variables = prompt_variables or []
        if self.prompt_variables:
            template = Template(prompt)
            # convert prompt_variables to a dict
            prompt_variables_dict = {pv.name: pv.value for pv in self.prompt_variables}
            prompt = template.render(prompt_variables_dict)
        self.prompt = prompt
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
        self.tool_map: dict[str, Tool] = tool_map or {}
        self.tool_defs = [
            tool.to_openai_tool_def() for tool in (tool_map or {}).values()
        ]
        self.transcript = []
        self.transcript_available: asyncio.Event = asyncio.Event()
        self.call_sid = None
        # this event is used to signal that the audio response has finished playing through twilio
        self.response_played: threading.Event = threading.Event()
        self.transcription_language = transcription_language

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
        Establish WebSocket connection to OpenAI Realtime API.

        Raises:
            Exception: If OPENAI_API_KEY environment variable is not set
            websockets.exceptions: If WebSocket connection fails
        """
        api_key = os.getenv("OPENAI_API_KEY")
        self.ws = await websockets.connect(
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2025-06-03",
            extra_headers={
                "Authorization": f"Bearer {api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
        )

    async def close(self) -> None:
        """
        Close the WebSocket connection to OpenAI Realtime API.
        """
        await self.ws.close()

    def set_automatic_turn_detection(self) -> None:
        """
        Configure automatic turn detection using server-side voice activity detection.

        This enables the agent to automatically detect when the user stops speaking
        and create responses accordingly.
        """
        self.turn_detection = {"type": "server_vad", "create_response": False}

    async def update_session(self) -> None:
        """
        Update the session configuration with current agent settings.

        This sends a session.update event to the OpenAI API with the current
        configuration including turn detection, audio formats, voice settings,
        instructions, modalities, and tools.
        """
        event = {
            "type": "session.update",
            "session": {
                "turn_detection": self.turn_detection.model_dump()
                if self.turn_detection
                else None,
                "input_audio_format": self.input_audio_format,
                "input_audio_transcription": {
                    "model": "gpt-4o-transcribe",
                },
                "output_audio_format": self.output_audio_format,
                "voice": self.voice,
                "instructions": self.prompt,
                "modalities": self.modalities,
                "speed": self.speed,
                "temperature": 0.8,
                "tools": self.tool_defs,
                "tool_choice": "auto",
            },
        }
        if self.transcription_language:
            event["session"]["input_audio_transcription"]["language"] = (
                self.transcription_language
            )
        await self.ws.send(json.dumps(event))

    async def send_audio(self, b64_encoded_audio: str) -> None:
        """
        Send audio data to the OpenAI Realtime API.

        Args:
            b64_encoded_audio: Base64 encoded audio data to send
        """
        event = {"type": "input_audio_buffer.append", "audio": b64_encoded_audio}
        await self.ws.send(json.dumps(event))

    async def truncate_audio(self, item_id: str, audio_end_ms: int) -> None:
        """
        Truncate audio at a specific time point.

        Args:
            item_id: The ID of the conversation item to truncate
            audio_end_ms: The end time in milliseconds where audio should be truncated
        """
        logger.info(f"Truncating audio for item_id: {item_id} at {audio_end_ms} ms")
        event = {
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": 0,
            "audio_end_ms": audio_end_ms,
        }
        await self.ws.send(json.dumps(event))

    async def commit_audio(self) -> None:
        """
        Commit the current audio buffer to be processed by the API.

        This signals that the current audio input is complete and ready for processing.
        """
        event = {"type": "input_audio_buffer.commit"}
        await self.ws.send(json.dumps(event))

    async def create_response(self) -> None:
        """
        Request the creation of a response from the OpenAI model.

        This triggers the model to generate a response based on the current conversation context.
        """
        logger.info("Creating response")
        await self.ws.send(json.dumps({"type": "response.create"}))

    async def wait_till_input_audio(self) -> bool:
        """
        Wait until input audio buffer is committed.

        Returns:
            bool: True if audio was committed successfully, False if interrupted

        Note:
            This method waits for the input_audio_buffer.committed event,
            which indicates that the audio input has been processed.
        """
        logger.info("Waiting for input audio buffer speech stopped event")
        while True:
            openai_message = await self.input_audio_buffer_event_queue.get()
            if openai_message is None:
                return False
            # if openai_message.get("type") == "input_audio_buffer.speech_stopped":
            #     return True
            elif openai_message.get("type") == "input_audio_buffer.committed":
                return True
            else:
                logger.info(
                    f"Skipping message(wait_till_input_audio): {openai_message}"
                )

    async def add_function_call_output(self, call_id: str, output: str) -> None:
        """
        Add the output of a function call to the conversation.

        Args:
            call_id: The ID of the function call
            output: The output/result of the function call
        """
        await self.ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output,
                    },
                }
            )
        )

    async def run_voicemail_tool(self, tool: Tool) -> None:
        """
        Execute a voicemail tool with specific configuration.

        This method updates the agent's prompt and tools to handle voicemail scenarios,
        then executes the voicemail tool with the provided message.

        Args:
            tool: The voicemail tool to execute

        Note:
            The agent's prompt is temporarily modified to focus on voicemail functionality,
            and tools are cleared except for the voicemail tool.
        """
        # update the instructions and tools with just the voicemail tool
        logger.info(
            f"Running voicemail tool with message: {tool.fixed_args['message']}"
        )
        self.prompt = f"The call has gone to voicemail. Leave the following message: {tool.fixed_args['message']}"
        self.tool_defs = []
        self.response_played.clear()
        await self.update_session()
        combined_kwargs = {**tool.fixed_args, **tool.auth}
        combined_kwargs["call_sid"] = self.call_sid
        combined_kwargs["response_played_event"] = self.response_played
        logger.info(f"Running voicemail tool with kwargs: {combined_kwargs}")
        await asyncio.to_thread(tool.func, **combined_kwargs)

    async def run_tool(self, call_id: str, tool_name: str, tool_args: dict) -> None:
        """
        Execute a tool with the given arguments.

        Args:
            call_id: The ID of the function call
            tool_name: The name of the tool to execute
            tool_args: The arguments to pass to the tool

        Raises:
            Exception: If the specified tool is not found in the tool map

        Note:
            This method handles slot filling for complex tool arguments,
            especially for group-type slots with fixed values.
            Tool execution is performed in a separate thread to avoid blocking.
        """
        tool = self.tool_map.get(tool_name)
        if not tool:
            raise Exception(f"Tool not found: {tool_name}")

        logger.info(f"Realtime execution for tool {tool.name} with args: {tool_args}")
        for slot in tool.slots:
            if slot.name in tool_args:
                slot.value = tool_args[slot.name]
            if slot.type == "group":
                for schema_obj in slot.schema:
                    if schema_obj.get("valueSource", "") == "fixed":
                        for filled_ob in tool_args[slot.name]:
                            if schema_obj.get("type") == "bool":
                                filled_ob[schema_obj.get("name")] = (
                                    schema_obj.get("value", "").lower() == "true"
                                )
                            else:
                                filled_ob[schema_obj.get("name")] = schema_obj.get(
                                    "value"
                                )

        if tool.func.__name__ == "http_tool":
            kwargs = {"slots": tool.slots}
        else:
            kwargs = {slot.name: slot.value for slot in tool.slots}
        combined_kwargs = {**kwargs, **tool.fixed_args, **tool.auth}
        combined_kwargs["call_sid"] = self.call_sid
        combined_kwargs["response_played_event"] = self.response_played
        try:
            response = await asyncio.to_thread(tool.func, **combined_kwargs)
        except Exception as e:
            logger.error(f"Error running tool {tool.name}: {e}")
            logger.exception(e)
            response = "unexpected error calling tool"
        logger.info(f"Tool {tool.name} response: {response}")

        await self.add_function_call_output(call_id, response)
        await self.create_response()

        return

    async def create_audio_response(self, prompt: str) -> None:
        """
        Create an audio response with a specific prompt.

        Args:
            prompt: The prompt to use for generating the audio response

        Note:
            This method temporarily switches the agent to audio modality
            and updates the session with the new prompt before creating a response.
        """
        logger.info(f"Creating audio response with: {prompt}")
        self.prompt = prompt
        self.set_audio_modality()
        await self.update_session()
        await self.create_response()

    async def receive_events(self) -> None:
        """
        Main event loop for receiving and processing events from the OpenAI Realtime API.

        This method handles all incoming WebSocket messages and routes them to appropriate
        handlers based on event type. It processes various events including:
        - Response completion and tool calls
        - Audio streaming and transcription
        - Input audio buffer events
        - Conversation item creation
        - Function call arguments

        Note:
            This method runs continuously until the WebSocket connection is closed.
            It automatically handles error events and logs them appropriately.
        """
        async for openai_message in self.ws:
            try:
                openai_event = json.loads(openai_message)
                event_type = openai_event.get("type")
                logger.info(f"Received event type: {event_type}")

                if event_type == "error":
                    logger.error(f"Error from OpenAI: {openai_event}")
                    continue

                if event_type == "response.done":
                    logger.info(f"response.done received: {openai_event}")
                    await self.internal_queue.put(openai_event)
                    # check if the response is a tool call
                    if openai_event.get("response") and openai_event["response"].get(
                        "output"
                    ):
                        for output in openai_event["response"]["output"]:
                            if output.get("type") == "function_call":
                                logger.info(f"function call received: {output['name']}")
                                try:
                                    await self.run_tool(
                                        output["call_id"],
                                        output["name"],
                                        json.loads(output["arguments"]),
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Error running tool {output['name']}: {e}"
                                    )
                                    logger.exception(e)
                                    raise e

                if event_type == "response.text.done" and "text" in openai_event:
                    await self.internal_queue.put(openai_event)

                if event_type == "response.audio.delta" and "delta" in openai_event:
                    event = {
                        "type": "audio_stream",
                        "origin": "bot",
                        "id": openai_event["item_id"],
                        "audio_bytes": base64.b64encode(
                            base64.b64decode(openai_event["delta"])
                        ).decode("utf-8")
                        if self.telephony_mode
                        else np.frombuffer(
                            base64.b64decode(openai_event["delta"]), np.int16
                        ).tolist(),
                    }
                    await self.external_queue.put(event)

                if (
                    event_type == "response.audio_transcript.delta"
                    and "delta" in openai_event
                    and not self.telephony_mode
                ):
                    self.text_buffer[openai_event["item_id"]] += openai_event["delta"]
                    event = {
                        "type": "text_stream",
                        "origin": "bot",
                        "id": openai_event["item_id"],
                        "text": self.text_buffer[openai_event["item_id"]],
                    }
                    await self.external_queue.put(event)

                if event_type == "response.audio_transcript.done":
                    event = {
                        "type": "message",
                        "origin": "bot",
                        "id": openai_event["item_id"],
                        "text": openai_event["transcript"],
                        "audio_url": "",
                    }
                    self.transcript.append(
                        Transcript(
                            id=str(uuid.uuid4()),
                            text=openai_event["transcript"],
                            origin="bot",
                            created_at=datetime.datetime.now(datetime.timezone.utc),
                        )
                    )
                    await self.external_queue.put(event)

                if event_type == "input_audio_buffer.speech_started":
                    await self.input_audio_buffer_event_queue.put(openai_event)
                    await self.external_queue.put(
                        {"type": "input_audio_buffer.speech_started"}
                    )

                if event_type == "input_audio_buffer.speech_stopped":
                    await self.input_audio_buffer_event_queue.put(openai_event)
                    await self.external_queue.put(
                        {"type": "input_audio_buffer.speech_stopped"}
                    )

                if event_type == "input_audio_buffer.committed":
                    await self.input_audio_buffer_event_queue.put(openai_event)

                if (
                    event_type == "conversation.item.created"
                    and openai_event.get("item")
                    and openai_event["item"].get("role")
                    and (
                        openai_event["item"]["role"] == "user"
                        or openai_event["item"]["role"] == "assistant"
                    )
                ):
                    event = {
                        "type": "message",
                        "origin": "user"
                        if openai_event["item"]["role"] == "user"
                        else "bot",
                        "id": openai_event["item"]["id"],
                        "text": " ",
                        "audio_url": "",
                    }
                    await self.external_queue.put(event)

                if (
                    event_type
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    event = {
                        "type": "message",
                        "origin": "user",
                        "id": openai_event["item_id"],
                        "text": openai_event["transcript"],
                        "audio_url": "",
                    }
                    self.transcript.append(
                        Transcript(
                            id=str(uuid.uuid4()),
                            text=openai_event["transcript"],
                            origin="user",
                            created_at=datetime.datetime.now(datetime.timezone.utc),
                        )
                    )
                    await self.external_queue.put(event)

                if event_type == "response.function_call_arguments.done":
                    await self.internal_queue.put(openai_event)
            except Exception as e:
                logger.error(f"Error processing openai event: {e.with_traceback()}")
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
