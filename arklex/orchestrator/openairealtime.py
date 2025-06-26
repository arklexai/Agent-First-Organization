import asyncio
import base64
from collections import defaultdict
import json
import os
import threading
from typing import Any, List
import numpy as np
import websockets
import logging
from arklex.env.tools.tools import Tool
from arklex.env.tools.types import Transcript


logger = logging.getLogger(__name__)


class RealtimeClient:
    def __init__(
        self, telephony_mode: bool = False, prompt: str = "", voice: str = "alloy"
    ) -> None:
        self.ws = None
        self.modalities: List[str] = ["text"]
        self.prompt = prompt
        self.voice = voice
        self.turn_detection = {
            "type": "server_vad",
            "create_response": True,
            "silence_duration_ms": 750,
        }
        self.internal_queue: asyncio.Queue = asyncio.Queue()
        self.external_queue: asyncio.Queue = asyncio.Queue()
        self.input_audio_buffer_event_queue: asyncio.Queue = asyncio.Queue()
        self.text_buffer = defaultdict(str)
        self.telephony_mode = telephony_mode
        self.input_audio_format = "g711_ulaw" if telephony_mode else "pcm16"
        self.output_audio_format = "g711_ulaw" if telephony_mode else "pcm16"
        self.tool_map: dict[str, Tool] = {}
        self.tool_defs = []
        self.transcript = []
        self.transcript_available: asyncio.Event = asyncio.Event()
        self.call_sid = None
        # this event is used to signal that the audio response has finished playing through twilio
        self.response_played: threading.Event = threading.Event()

    def set_audio_modality(self) -> None:
        self.modalities = ["text", "audio"]

    def set_text_modality(self) -> None:
        self.modalities = ["text"]

    async def connect(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        self.ws = await websockets.connect(
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17",
            extra_headers={
                "Authorization": f"Bearer {api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
        )

    async def close(self) -> None:
        await self.ws.close()

    def set_automatic_turn_detection(self) -> None:
        self.turn_detection = {"type": "server_vad", "create_response": False}

    async def update_session(self) -> None:
        event = {
            "type": "session.update",
            "session": {
                "turn_detection": self.turn_detection,
                "input_audio_format": self.input_audio_format,
                "input_audio_transcription": {"model": "whisper-1"},
                "output_audio_format": self.output_audio_format,
                "voice": self.voice,
                "instructions": self.prompt,
                "modalities": self.modalities,
                "temperature": 0.8,
                "tools": self.tool_defs,
                "tool_choice": "auto",
            },
        }
        logger.info(f"Updating session to {json.dumps(event, indent=4)}")
        await self.ws.send(json.dumps(event))

    async def send_audio(self, b64_encoded_audio: str) -> None:
        event = {"type": "input_audio_buffer.append", "audio": b64_encoded_audio}
        await self.ws.send(json.dumps(event))

    async def truncate_audio(self, item_id: str, audio_end_ms: int) -> None:
        logger.info(f"Truncating audio for item_id: {item_id} at {audio_end_ms} ms")
        event = {
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": 0,
            "audio_end_ms": audio_end_ms,
        }
        await self.ws.send(json.dumps(event))

    async def commit_audio(self) -> None:
        event = {"type": "input_audio_buffer.commit"}
        await self.ws.send(json.dumps(event))

    async def create_response(self) -> None:
        logger.info("Creating response")
        await self.ws.send(json.dumps({"type": "response.create"}))

    async def wait_till_input_audio(self) -> bool:
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
        tool = self.tool_map.get(tool_name)
        if not tool:
            raise Exception(f"Tool not found: {tool_name}")

        logger.info(f"Realtime execution for tool {tool.name} with args: {tool_args}")
        for slot in tool.slots:
            if slot.name in tool_args:
                slot.value = tool_args[slot.name]
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
        logger.info(f"Creating audio response with: {prompt}")
        self.prompt = prompt
        self.set_audio_modality()
        await self.update_session()
        await self.create_response()

    async def receive_events(self) -> None:
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
        await self.internal_queue.put(None)
        await self.input_audio_buffer_event_queue.put(None)
        await self.external_queue.put(None)


def postprocess_json(input: str) -> dict[str, Any]:
    input = input.replace("'", '"')
    valid_phrases = ['"', "{", "}", "[", "]"]

    valid_lines = []
    for line in input.split("\n"):
        if len(line) == 0:
            continue
        # If the line not starts with any of the valid phrases, skip it
        should_skip = not any(
            [line.strip().startswith(phrase) for phrase in valid_phrases]
        )
        if should_skip:
            continue
        valid_lines.append(line)

    generated_result = "\n".join(valid_lines)
    result = json.loads(generated_result)
    if len(result.keys()) == 0:
        raise Exception(f"Failed to parse response: {input}")
    return result
