"""Test cases for OpenAIRealtimeAgent class."""

# TODO: Add tests for OpenAIRealtimeAgent class

# import asyncio
# import base64
# import contextlib
# import json
# import os
# from unittest.mock import AsyncMock, Mock, patch

# import pytest
# import websockets

# from arklex.env.agents.openai_realtime_agent import (
#     OpenAIRealtimeAgent,
#     PromptVariable,
#     TurnDetection,
# )
# from arklex.env.tools.tools import Tool


# @pytest.fixture
# def mock_tool() -> Tool:
#     """Create a mock Tool for testing."""
#     tool_func = Mock(return_value="tool result")
#     tool_func.__name__ = "mock_tool"

#     tool = Mock(spec=Tool)
#     tool.name = "mock_tool"
#     tool.func = tool_func
#     tool.slots = []
#     tool.fixed_args = {}
#     tool.auth = {}
#     tool.to_openai_tool_def.return_value = {
#         "type": "function",
#         "function": {
#             "name": "mock_tool",
#             "description": "Mock tool for testing",
#             "parameters": {},
#         },
#     }
#     return tool


# @pytest.fixture
# def mock_tool_map(mock_tool: Tool) -> dict[str, Tool]:
#     """Create a mock tool map for testing."""
#     return {"mock_tool": mock_tool}


# @pytest.fixture
# def mock_websocket() -> AsyncMock:
#     """Create a mock WebSocket connection."""
#     ws = AsyncMock()
#     ws.send = AsyncMock()
#     ws.close = AsyncMock()
#     return ws


# @pytest.fixture
# def prompt_variables() -> list[PromptVariable]:
#     """Create sample prompt variables for testing."""
#     return [
#         PromptVariable(name="user_name", value="John"),
#         PromptVariable(name="company", value="Acme Corp"),
#     ]


# @pytest.fixture
# def turn_detection() -> TurnDetection:
#     """Create a sample turn detection configuration."""
#     return TurnDetection(
#         type="server_vad",
#         create_response=True,
#         interrupt_response=True,
#         prefix_padding_ms=300,
#         silence_duration_ms=750,
#         threshold=0.5,
#     )


# class TestPromptVariable:
#     """Test cases for PromptVariable class."""

#     def test_prompt_variable_creation(self) -> None:
#         """Test creating a PromptVariable with default value."""
#         pv = PromptVariable(name="test")
#         assert pv.name == "test"
#         assert pv.value == ""

#     def test_prompt_variable_with_value(self) -> None:
#         """Test creating a PromptVariable with a specific value."""
#         pv = PromptVariable(name="test", value="value")
#         assert pv.name == "test"
#         assert pv.value == "value"


# class TestTurnDetection:
#     """Test cases for TurnDetection class."""

#     def test_turn_detection_default_creation(self) -> None:
#         """Test creating TurnDetection with default values."""
#         td = TurnDetection()
#         assert td.type == "server_vad"
#         assert td.create_response is True
#         assert td.interrupt_response is True
#         assert td.prefix_padding_ms == 300
#         assert td.silence_duration_ms == 750
#         assert td.threshold == 0.5
#         assert td.eagerness is None

#     def test_from_dict_none(self) -> None:
#         """Test TurnDetection.from_dict with None input."""
#         td = TurnDetection.from_dict(None)
#         assert td.type == "server_vad"
#         assert td.eagerness is None

#     def test_from_dict_server_vad(self) -> None:
#         """Test TurnDetection.from_dict with server_vad type."""
#         data = {"type": "server_vad", "create_response": False}
#         td = TurnDetection.from_dict(data)
#         assert td.type == "server_vad"
#         assert td.create_response is False
#         assert td.eagerness is None

#     def test_from_dict_semantic_vad_type(self) -> None:
#         """Test TurnDetection.from_dict with semantic_vad type."""
#         data = {"type": "semantic_vad", "create_response": False, "eagerness": "high"}
#         td = TurnDetection.from_dict(data)
#         assert td.type == "semantic_vad"
#         assert td.create_response is False
#         assert td.eagerness == "high"

#     def test_model_dump_server_vad(self) -> None:
#         """Test model_dump for server_vad type."""
#         td = TurnDetection(type="server_vad")
#         result = td.model_dump()
#         assert "eagerness" not in result
#         assert "prefix_padding_ms" in result
#         assert "silence_duration_ms" in result
#         assert "threshold" in result

#     def test_model_dump_semantic_vad_type(self) -> None:
#         """Test model_dump for semantic_vad type."""
#         td = TurnDetection(type="semantic_vad", eagerness="high")
#         result = td.model_dump()
#         assert "eagerness" in result
#         assert "prefix_padding_ms" not in result
#         assert "silence_duration_ms" not in result
#         assert "threshold" not in result

#     def test_from_dict_invalid_type(self) -> None:
#         """Test TurnDetection.from_dict with invalid type returns default."""
#         data = {"type": "invalid_type", "create_response": False, "eagerness": "high"}
#         td = TurnDetection.from_dict(data)
#         # Should return default configuration when invalid type is provided
#         assert td.type == "server_vad"
#         assert td.create_response is True
#         assert td.eagerness is None

#     def test_from_dict_no_type(self) -> None:
#         """Test TurnDetection.from_dict with no type specified."""
#         data = {"create_response": False, "eagerness": "high"}
#         td = TurnDetection.from_dict(data)
#         # Should return default configuration when no type is provided
#         assert td.type == "server_vad"
#         assert td.create_response is True
#         assert td.eagerness is None

#     def test_valid_type_values(self) -> None:
#         """Test that only valid type values are accepted."""
#         # These should work
#         td1 = TurnDetection(type="server_vad")
#         assert td1.type == "server_vad"

#         td2 = TurnDetection(type="semantic_vad")
#         assert td2.type == "semantic_vad"

#         # Default should be server_vad
#         td3 = TurnDetection()
#         assert td3.type == "server_vad"


# class TestOpenAIRealtimeAgent:
#     """Test cases for OpenAIRealtimeAgent class."""

#     def test_init_default_values(self) -> None:
#         """Test OpenAIRealtimeAgent initialization with default values."""
#         agent = OpenAIRealtimeAgent()

#         assert agent.ws is None
#         assert agent.modalities == ["text"]
#         assert agent.prompt == ""
#         assert agent.voice == "alloy"
#         assert agent.speed == 1.0
#         assert agent.turn_detection is None
#         assert agent.telephony_mode is False
#         assert agent.input_audio_format == "pcm16"
#         assert agent.output_audio_format == "pcm16"
#         assert agent.tool_map == {}
#         assert agent.tool_defs == []
#         assert agent.transcript == []
#         assert agent.call_sid is None
#         assert agent.transcription_language is None

#     def test_init_with_parameters(
#         self,
#         mock_tool_map: dict[str, Tool],
#         prompt_variables: list[PromptVariable],
#         turn_detection: TurnDetection,
#     ) -> None:
#         """Test OpenAIRealtimeAgent initialization with all parameters."""
#         agent = OpenAIRealtimeAgent(
#             telephony_mode=True,
#             prompt="Hello {{user_name}} from {{company}}",
#             voice="nova",
#             transcription_language="en-US",
#             speed=1.5,
#             turn_detection=turn_detection,
#             tool_map=mock_tool_map,
#             prompt_variables=prompt_variables,
#         )

#         assert agent.telephony_mode is True
#         assert agent.prompt == "Hello John from Acme Corp"
#         assert agent.voice == "nova"
#         assert agent.transcription_language == "en-US"
#         assert agent.speed == 1.5
#         assert agent.turn_detection == turn_detection
#         assert agent.tool_map == mock_tool_map
#         assert len(agent.tool_defs) == 1
#         assert agent.input_audio_format == "g711_ulaw"
#         assert agent.output_audio_format == "g711_ulaw"

#     def test_set_telephone_mode(self) -> None:
#         """Test setting telephone mode."""
#         agent = OpenAIRealtimeAgent()
#         agent.set_telephone_mode()

#         assert agent.telephony_mode is True
#         assert agent.input_audio_format == "g711_ulaw"
#         assert agent.output_audio_format == "g711_ulaw"

#     def test_set_audio_modality(self) -> None:
#         """Test setting audio modality."""
#         agent = OpenAIRealtimeAgent()
#         agent.set_audio_modality()

#         assert agent.modalities == ["text", "audio"]

#     def test_set_text_modality(self) -> None:
#         """Test setting text modality."""
#         agent = OpenAIRealtimeAgent()
#         agent.modalities = ["text", "audio"]
#         agent.set_text_modality()

#         assert agent.modalities == ["text"]

#     async def test_connect_success(self) -> None:
#         """Test successful WebSocket connection."""
#         mock_websocket = AsyncMock()

#         async def mock_connect(*args: object, **kwargs: object) -> AsyncMock:
#             return mock_websocket

#         with patch("websockets.connect", side_effect=mock_connect):
#             agent = OpenAIRealtimeAgent()
#             await agent.connect()

#             assert agent.ws == mock_websocket

#     @patch("websockets.connect")
#     async def test_connect_missing_api_key(self, mock_connect: Mock) -> None:
#         """Test connection failure when API key is missing."""
#         # Remove API key from environment
#         with patch.dict(os.environ, {}, clear=True):
#             agent = OpenAIRealtimeAgent()

#             # The actual implementation doesn't check for missing API key,
#             # so it will try to connect with None as the API key
#             # This should result in a WebSocket connection error
#             with pytest.raises(TypeError):
#                 await agent.connect()

#     @patch("websockets.connect")
#     async def test_connect_websocket_error(self, mock_connect: Mock) -> None:
#         """Test connection failure due to WebSocket error."""
#         mock_connect.side_effect = websockets.exceptions.ConnectionClosed(None, None)

#         agent = OpenAIRealtimeAgent(tool_map={})

#         with pytest.raises(websockets.exceptions.ConnectionClosed):
#             await agent.connect()

#     async def test_close(self, mock_websocket: AsyncMock) -> None:
#         """Test closing WebSocket connection."""
#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         await agent.close()

#         mock_websocket.close.assert_called_once()

#     def test_set_automatic_turn_detection(self) -> None:
#         """Test setting automatic turn detection."""
#         agent = OpenAIRealtimeAgent()
#         agent.set_automatic_turn_detection()

#         assert agent.turn_detection == {"type": "server_vad", "create_response": False}

#     @patch("json.dumps")
#     async def test_update_session(
#         self, mock_json_dumps: Mock, mock_websocket: AsyncMock
#     ) -> None:
#         """Test updating session configuration."""
#         mock_json_dumps.return_value = '{"test": "data"}'

#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket
#         agent.prompt = "Test prompt"
#         agent.voice = "nova"
#         agent.speed = 1.2

#         await agent.update_session()

#         mock_websocket.send.assert_called_once_with('{"test": "data"}')
#         mock_json_dumps.assert_called_once()

#     @patch("json.dumps")
#     async def test_update_session_with_turn_detection(
#         self,
#         mock_json_dumps: Mock,
#         mock_websocket: AsyncMock,
#         turn_detection: TurnDetection,
#     ) -> None:
#         """Test updating session with turn detection configuration."""
#         mock_json_dumps.return_value = '{"test": "data"}'

#         agent = OpenAIRealtimeAgent(turn_detection=turn_detection)
#         agent.ws = mock_websocket

#         await agent.update_session()

#         mock_websocket.send.assert_called_once_with('{"test": "data"}')
#         mock_json_dumps.assert_called_once()

#     @patch("json.dumps")
#     async def test_update_session_with_transcription_language(
#         self, mock_json_dumps: Mock, mock_websocket: AsyncMock
#     ) -> None:
#         """Test updating session with transcription language."""
#         mock_json_dumps.return_value = '{"test": "data"}'

#         agent = OpenAIRealtimeAgent(transcription_language="es-ES")
#         agent.ws = mock_websocket

#         await agent.update_session()

#         mock_websocket.send.assert_called_once_with('{"test": "data"}')
#         mock_json_dumps.assert_called_once()

#     @patch("json.dumps")
#     async def test_send_audio(
#         self, mock_json_dumps: Mock, mock_websocket: AsyncMock
#     ) -> None:
#         """Test sending audio data."""
#         mock_json_dumps.return_value = '{"test": "data"}'

#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         audio_data = base64.b64encode(b"test audio").decode("utf-8")
#         await agent.send_audio(audio_data)

#         mock_websocket.send.assert_called_once_with('{"test": "data"}')
#         mock_json_dumps.assert_called_once()

#     @patch("json.dumps")
#     async def test_truncate_audio(
#         self, mock_json_dumps: Mock, mock_websocket: AsyncMock
#     ) -> None:
#         """Test truncating audio."""
#         mock_json_dumps.return_value = '{"test": "data"}'

#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         await agent.truncate_audio("item_123", 5000)

#         mock_websocket.send.assert_called_once_with('{"test": "data"}')
#         mock_json_dumps.assert_called_once()

#     @patch("json.dumps")
#     async def test_commit_audio(
#         self, mock_json_dumps: Mock, mock_websocket: AsyncMock
#     ) -> None:
#         """Test committing audio buffer."""
#         mock_json_dumps.return_value = '{"test": "data"}'

#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         await agent.commit_audio()

#         mock_websocket.send.assert_called_once_with('{"test": "data"}')
#         mock_json_dumps.assert_called_once()

#     @patch("json.dumps")
#     async def test_create_response(
#         self, mock_json_dumps: Mock, mock_websocket: AsyncMock
#     ) -> None:
#         """Test creating a response."""
#         mock_json_dumps.return_value = '{"test": "data"}'

#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         await agent.create_response()

#         mock_websocket.send.assert_called_once_with('{"test": "data"}')
#         mock_json_dumps.assert_called_once()

#     async def test_wait_till_input_audio_committed(self) -> None:
#         """Test waiting for input audio to be committed."""
#         agent = OpenAIRealtimeAgent()

#         # Simulate committed event
#         asyncio.create_task(
#             agent.input_audio_buffer_event_queue.put(
#                 {"type": "input_audio_buffer.committed"}
#             )
#         )

#         result = await agent.wait_till_input_audio()
#         assert result is True

#     async def test_wait_till_input_audio_interrupted(self) -> None:
#         """Test waiting for input audio when interrupted."""
#         agent = OpenAIRealtimeAgent()

#         # Simulate interruption
#         asyncio.create_task(agent.input_audio_buffer_event_queue.put(None))

#         result = await agent.wait_till_input_audio()
#         assert result is False

#     @patch("json.dumps")
#     async def test_add_function_call_output(
#         self, mock_json_dumps: Mock, mock_websocket: AsyncMock
#     ) -> None:
#         """Test adding function call output."""
#         mock_json_dumps.return_value = '{"test": "data"}'

#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         await agent.add_function_call_output("call_123", "function result")

#         mock_websocket.send.assert_called_once_with('{"test": "data"}')
#         mock_json_dumps.assert_called_once()

#     @patch("asyncio.to_thread")
#     async def test_run_voicemail_tool(
#         self, mock_to_thread: Mock, mock_tool: Tool
#     ) -> None:
#         """Test running voicemail tool."""
#         mock_to_thread.return_value = None

#         agent = OpenAIRealtimeAgent()
#         agent.ws = AsyncMock()
#         agent.call_sid = "call_123"
#         mock_tool.fixed_args = {"message": "Leave a message"}

#         await agent.run_voicemail_tool(mock_tool)

#         assert (
#             agent.prompt
#             == "The call has gone to voicemail. Leave the following message: Leave a message"
#         )
#         assert agent.tool_defs == []
#         mock_to_thread.assert_called_once()

#     @patch("asyncio.to_thread")
#     @patch("json.dumps")
#     async def test_run_tool_success(
#         self,
#         mock_json_dumps: Mock,
#         mock_to_thread: Mock,
#         mock_tool: Tool,
#         mock_websocket: AsyncMock,
#     ) -> None:
#         """Test successful tool execution."""
#         mock_json_dumps.return_value = '{"test": "data"}'
#         mock_to_thread.return_value = "tool result"

#         agent = OpenAIRealtimeAgent(tool_map={"mock_tool": mock_tool})
#         agent.ws = mock_websocket
#         agent.call_sid = "call_123"

#         await agent.run_tool("call_123", "mock_tool", {"param": "value"})

#         mock_to_thread.assert_called_once()
#         mock_websocket.send.assert_called()

#     async def test_run_tool_not_found(self, mock_websocket: AsyncMock) -> None:
#         """Test tool execution when tool is not found."""
#         agent = OpenAIRealtimeAgent(tool_map={})
#         agent.ws = mock_websocket

#         with pytest.raises(Exception, match="Tool not found: nonexistent_tool"):
#             await agent.run_tool("call_123", "nonexistent_tool", {})

#     @patch("asyncio.to_thread")
#     async def test_run_tool_with_group_slots(
#         self, mock_to_thread: Mock, mock_tool: Tool, mock_websocket: AsyncMock
#     ) -> None:
#         """Test tool execution with group slots."""
#         mock_to_thread.return_value = "tool result"

#         # Create a mock slot with group type
#         group_slot = Mock()
#         group_slot.name = "group_param"
#         group_slot.type = "group"
#         group_slot.schema = [
#             {
#                 "name": "fixed_field",
#                 "valueSource": "fixed",
#                 "value": "true",
#                 "type": "bool",
#             }
#         ]

#         mock_tool.slots = [group_slot]

#         agent = OpenAIRealtimeAgent(tool_map={"mock_tool": mock_tool})
#         agent.ws = mock_websocket
#         agent.call_sid = "call_123"

#         tool_args = {"group_param": [{"other_field": "value"}]}
#         await agent.run_tool("call_123", "mock_tool", tool_args)

#         mock_to_thread.assert_called_once()

#     @patch("asyncio.to_thread")
#     async def test_run_tool_http_tool(
#         self, mock_to_thread: Mock, mock_tool: Tool, mock_websocket: AsyncMock
#     ) -> None:
#         """Test tool execution for HTTP tool."""
#         mock_to_thread.return_value = "tool result"
#         mock_tool.func.__name__ = "http_tool"
#         mock_tool.slots = []

#         agent = OpenAIRealtimeAgent(tool_map={"mock_tool": mock_tool})
#         agent.ws = mock_websocket
#         agent.call_sid = "call_123"

#         await agent.run_tool("call_123", "mock_tool", {})

#         mock_to_thread.assert_called_once()

#     @patch("asyncio.to_thread")
#     async def test_run_tool_exception(
#         self, mock_to_thread: Mock, mock_tool: Tool, mock_websocket: AsyncMock
#     ) -> None:
#         """Test tool execution with exception."""
#         mock_to_thread.side_effect = Exception("Tool error")

#         agent = OpenAIRealtimeAgent(tool_map={"mock_tool": mock_tool})
#         agent.ws = mock_websocket
#         agent.call_sid = "call_123"

#         await agent.run_tool("call_123", "mock_tool", {})

#         mock_to_thread.assert_called_once()

#     @patch("json.dumps")
#     async def test_create_audio_response(
#         self, mock_json_dumps: Mock, mock_websocket: AsyncMock
#     ) -> None:
#         """Test creating audio response."""
#         mock_json_dumps.return_value = '{"test": "data"}'

#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         await agent.create_audio_response("Audio prompt")

#         assert agent.prompt == "Audio prompt"
#         assert agent.modalities == ["text", "audio"]
#         mock_websocket.send.assert_called()

#     async def test_receive_events_response_done(
#         self, mock_websocket: AsyncMock
#     ) -> None:
#         """Test receiving response.done event."""
#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         # Mock the WebSocket to return a response.done event
#         mock_websocket.__aiter__.return_value = [
#             json.dumps(
#                 {
#                     "type": "response.done",
#                     "response": {
#                         "output": [
#                             {
#                                 "type": "function_call",
#                                 "name": "mock_tool",
#                                 "call_id": "call_123",
#                                 "arguments": '{"param": "value"}',
#                             }
#                         ]
#                     },
#                 }
#             )
#         ]

#         # Mock tool execution
#         with patch.object(agent, "run_tool") as mock_run_tool:
#             await agent.receive_events()

#             mock_run_tool.assert_called_once_with(
#                 "call_123", "mock_tool", {"param": "value"}
#             )

#     async def test_receive_events_response_text_done(
#         self, mock_websocket: AsyncMock
#     ) -> None:
#         """Test receiving response.text.done event."""
#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         mock_websocket.__aiter__.return_value = [
#             json.dumps({"type": "response.text.done", "text": "Hello world"})
#         ]

#         await agent.receive_events()

#         # Check that the event was put in the internal queue
#         event = await agent.internal_queue.get()
#         assert event["type"] == "response.text.done"
#         assert event["text"] == "Hello world"

#     async def test_receive_events_audio_delta(self, mock_websocket: AsyncMock) -> None:
#         """Test receiving response.audio.delta event."""
#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         audio_data = base64.b64encode(b"test audio").decode("utf-8")
#         mock_websocket.__aiter__.return_value = [
#             json.dumps(
#                 {
#                     "type": "response.audio.delta",
#                     "item_id": "item_123",
#                     "delta": audio_data,
#                 }
#             )
#         ]

#         await agent.receive_events()

#         # Check that the audio event was put in the external queue
#         event = await agent.external_queue.get()
#         assert event["type"] == "audio_stream"
#         assert event["origin"] == "bot"
#         assert event["id"] == "item_123"

#     async def test_receive_events_audio_delta_telephony(
#         self, mock_websocket: AsyncMock
#     ) -> None:
#         """Test receiving response.audio.delta event in telephony mode."""
#         agent = OpenAIRealtimeAgent(telephony_mode=True)
#         agent.ws = mock_websocket

#         audio_data = base64.b64encode(b"test audio").decode("utf-8")
#         mock_websocket.__aiter__.return_value = [
#             json.dumps(
#                 {
#                     "type": "response.audio.delta",
#                     "item_id": "item_123",
#                     "delta": audio_data,
#                 }
#             )
#         ]

#         await agent.receive_events()

#         # Check that the audio event was put in the external queue
#         event = await agent.external_queue.get()
#         assert event["type"] == "audio_stream"
#         assert event["origin"] == "bot"
#         assert event["id"] == "item_123"

#     async def test_receive_events_audio_transcript_delta(
#         self, mock_websocket: AsyncMock
#     ) -> None:
#         """Test receiving response.audio_transcript.delta event."""
#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         mock_websocket.__aiter__.return_value = [
#             json.dumps(
#                 {
#                     "type": "response.audio_transcript.delta",
#                     "item_id": "item_123",
#                     "delta": "Hello",
#                 }
#             ),
#             json.dumps(
#                 {
#                     "type": "response.audio_transcript.delta",
#                     "item_id": "item_123",
#                     "delta": " world",
#                 }
#             ),
#         ]

#         await agent.receive_events()

#         # Check that the text events were put in the external queue
#         event1 = await agent.external_queue.get()
#         event2 = await agent.external_queue.get()
#         assert event1["type"] == "text_stream"
#         assert event1["text"] == "Hello"
#         assert event2["type"] == "text_stream"
#         assert event2["text"] == "Hello world"

#     async def test_receive_events_audio_transcript_done(
#         self, mock_websocket: AsyncMock
#     ) -> None:
#         """Test receiving response.audio_transcript.done event."""
#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         mock_websocket.__aiter__.return_value = [
#             json.dumps(
#                 {
#                     "type": "response.audio_transcript.done",
#                     "item_id": "item_123",
#                     "transcript": "Hello world",
#                 }
#             )
#         ]

#         await agent.receive_events()

#         # Check that the message event was put in the external queue
#         event = await agent.external_queue.get()
#         assert event["type"] == "message"
#         assert event["origin"] == "bot"
#         assert event["text"] == "Hello world"

#         # Check that transcript was added
#         assert len(agent.transcript) == 1
#         assert agent.transcript[0].text == "Hello world"
#         assert agent.transcript[0].origin == "bot"

#     async def test_receive_events_input_audio_buffer_events(
#         self, mock_websocket: AsyncMock
#     ) -> None:
#         """Test receiving input audio buffer events."""
#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         mock_websocket.__aiter__.return_value = [
#             json.dumps({"type": "input_audio_buffer.speech_started"}),
#             json.dumps({"type": "input_audio_buffer.speech_stopped"}),
#             json.dumps({"type": "input_audio_buffer.committed"}),
#         ]

#         await agent.receive_events()

#         # Check that events were put in the external queue
#         event1 = await agent.external_queue.get()
#         event2 = await agent.external_queue.get()
#         assert event1["type"] == "input_audio_buffer.speech_started"
#         assert event2["type"] == "input_audio_buffer.speech_stopped"

#     async def test_receive_events_conversation_item_created(
#         self, mock_websocket: AsyncMock
#     ) -> None:
#         """Test receiving conversation.item.created event."""
#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         mock_websocket.__aiter__.return_value = [
#             json.dumps(
#                 {
#                     "type": "conversation.item.created",
#                     "item": {"id": "item_123", "role": "user"},
#                 }
#             ),
#             json.dumps(
#                 {
#                     "type": "conversation.item.created",
#                     "item": {"id": "item_456", "role": "assistant"},
#                 }
#             ),
#         ]

#         await agent.receive_events()

#         # Check that message events were put in the external queue
#         event1 = await agent.external_queue.get()
#         event2 = await agent.external_queue.get()
#         assert event1["type"] == "message"
#         assert event1["origin"] == "user"
#         assert event2["type"] == "message"
#         assert event2["origin"] == "bot"

#     async def test_receive_events_input_audio_transcription_completed(
#         self, mock_websocket: AsyncMock
#     ) -> None:
#         """Test receiving conversation.item.input_audio_transcription.completed event."""
#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         mock_websocket.__aiter__.return_value = [
#             json.dumps(
#                 {
#                     "type": "conversation.item.input_audio_transcription.completed",
#                     "item_id": "item_123",
#                     "transcript": "User said hello",
#                 }
#             )
#         ]

#         await agent.receive_events()

#         # Check that the message event was put in the external queue
#         event = await agent.external_queue.get()
#         assert event["type"] == "message"
#         assert event["origin"] == "user"
#         assert event["text"] == "User said hello"

#         # Check that transcript was added
#         assert len(agent.transcript) == 1
#         assert agent.transcript[0].text == "User said hello"
#         assert agent.transcript[0].origin == "user"

#     async def test_receive_events_error(self, mock_websocket: AsyncMock) -> None:
#         """Test receiving error event."""
#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         mock_websocket.__aiter__.return_value = [
#             json.dumps({"type": "error", "error": "Test error"})
#         ]

#         # Start the event loop and cancel it after a short delay
#         task = asyncio.create_task(agent.receive_events())
#         await asyncio.sleep(0.1)
#         task.cancel()

#         with contextlib.suppress(asyncio.CancelledError):
#             await task

#         # Error events should be logged but not put in queues
#         # The queue will have None from end_queues() being called
#         assert await agent.internal_queue.get() is None
#         assert await agent.external_queue.get() is None

#     async def test_receive_events_websocket_closed(
#         self, mock_websocket: AsyncMock
#     ) -> None:
#         """Test handling WebSocket connection closure."""
#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         # Simulate WebSocket closure by making the iterator return immediately
#         mock_websocket.__aiter__.return_value = []

#         # The receive_events method should handle the empty iterator and call end_queues
#         await agent.receive_events()

#         # Check that queues were ended
#         assert await agent.internal_queue.get() is None
#         assert await agent.external_queue.get() is None
#         assert await agent.input_audio_buffer_event_queue.get() is None

#     async def test_end_queues(self) -> None:
#         """Test ending all queues."""
#         agent = OpenAIRealtimeAgent()

#         await agent.end_queues()

#         # Check that None was put in all queues
#         assert await agent.internal_queue.get() is None
#         assert await agent.external_queue.get() is None
#         assert await agent.input_audio_buffer_event_queue.get() is None

#     def test_inheritance_from_base_agent(self) -> None:
#         """Test that OpenAIRealtimeAgent inherits from BaseAgent."""
#         from arklex.env.agents.agent import BaseAgent

#         agent = OpenAIRealtimeAgent()
#         assert isinstance(agent, BaseAgent)

#     def test_agent_registration(self) -> None:
#         """Test that OpenAIRealtimeAgent is properly registered."""
#         agent = OpenAIRealtimeAgent()
#         assert hasattr(agent.__class__, "name")
#         assert agent.__class__.name == "OpenAIRealtimeAgent"


# class TestOpenAIRealtimeAgentIntegration:
#     """Integration tests for OpenAIRealtimeAgent."""

#     @pytest.mark.asyncio
#     async def test_full_conversation_flow(self, mock_websocket: AsyncMock) -> None:
#         """Test a full conversation flow with audio and text."""
#         agent = OpenAIRealtimeAgent()
#         agent.ws = mock_websocket

#         # Mock WebSocket to simulate a conversation
#         conversation_events = [
#             # User starts speaking
#             {"type": "input_audio_buffer.speech_started"},
#             # User audio transcription
#             {
#                 "type": "conversation.item.input_audio_transcription.completed",
#                 "item_id": "user_1",
#                 "transcript": "Hello",
#             },
#             # Bot response
#             {
#                 "type": "response.audio_transcript.done",
#                 "item_id": "bot_1",
#                 "transcript": "Hi there!",
#             },
#             # User stops speaking
#             {"type": "input_audio_buffer.speech_stopped"},
#         ]

#         mock_websocket.__aiter__.return_value = [
#             json.dumps(event) for event in conversation_events
#         ]

#         # Start receiving events
#         task = asyncio.create_task(agent.receive_events())

#         # Wait for events to be processed
#         await asyncio.sleep(0.1)
#         task.cancel()

#         with contextlib.suppress(asyncio.CancelledError):
#             await task

#         # Check that events were processed
#         assert len(agent.transcript) == 2
#         assert agent.transcript[0].text == "Hello"
#         assert agent.transcript[0].origin == "user"
#         assert agent.transcript[1].text == "Hi there!"
#         assert agent.transcript[1].origin == "bot"

#     @pytest.mark.asyncio
#     async def test_tool_execution_flow(
#         self, mock_tool: Tool, mock_websocket: AsyncMock
#     ) -> None:
#         """Test tool execution flow."""
#         agent = OpenAIRealtimeAgent(tool_map={"mock_tool": mock_tool})
#         agent.ws = mock_websocket

#         # Mock tool execution
#         with patch.object(agent, "run_tool") as mock_run_tool:
#             # Simulate function call event
#             function_call_event = {
#                 "type": "response.done",
#                 "response": {
#                     "output": [
#                         {
#                             "type": "function_call",
#                             "name": "mock_tool",
#                             "call_id": "call_123",
#                             "arguments": '{"param": "value"}',
#                         }
#                     ]
#                 },
#             }

#             mock_websocket.__aiter__.return_value = [json.dumps(function_call_event)]

#             task = asyncio.create_task(agent.receive_events())
#             await asyncio.sleep(0.1)
#             task.cancel()

#             with contextlib.suppress(asyncio.CancelledError):
#                 await task

#             mock_run_tool.assert_called_once_with(
#                 "call_123", "mock_tool", {"param": "value"}
#             )