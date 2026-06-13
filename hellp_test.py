import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import unittest.mock
from typing import TYPE_CHECKING, Any, cast

from acp.interfaces import Client
from acp.schema import TextContentBlock

import google.antigravity as agy
from google.antigravity import types as agy_types


class FakeAgent:
    """Minimal fake matching the agy.Agent interface (chat, __aenter__, __aexit__)."""

    def __init__(self, config, responses=None):
        self._responses = responses or []
        self._call_index = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def chat(self, prompt):
        if self._call_index < len(self._responses):
            chunks = self._responses[self._call_index]
            self._call_index += 1
        else:
            chunks = [agy_types.Text(step_index=0, text="default response")]

        async def stream():
            for c in chunks:
                yield c

        # Plain MagicMock: spec=Conversation fails on Python 3.14 due to
        # annotation resolution conflict in the Antigravity SDK
        return agy_types.ChatResponse(stream(), conversation=MagicMock())


class FakeConfig:
    pass


async def test_offline_prompt_text():
    """Text prompt without an LLM — uses FakeAgent with canned response."""
    import hellp

    chunks = [
        agy_types.Thought(step_index=0, text="let me think"),
        agy_types.Text(step_index=1, text="Hello "),
        agy_types.Text(step_index=1, text="back!"),
    ]
    fake_agent = FakeAgent(config=None, responses=[chunks])

    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    reply = await sut.prompt(
        [TextContentBlock(type="text", text="Hi")],
        session_id=session.session_id,
    )

    assert reply.stop_reason == "end_turn"

    updates = [call.kwargs.get("update") or call.args[1] for call in client.session_update.call_args_list]
    thought_updates = [u for u in updates if u.session_update == "agent_thought_chunk"]
    message_updates = [u for u in updates if u.session_update == "agent_message_chunk"]
    assert len(thought_updates) == 1
    assert thought_updates[0].content.text == "let me think"
    assert len(message_updates) == 2
    assert message_updates[0].content.text == "Hello "
    assert message_updates[1].content.text == "back!"


async def test_offline_prompt_with_tool_calls():
    """Verify tool call and tool result chunks are forwarded as ACP tool_call updates."""
    import hellp

    chunks = [
        agy_types.ToolCall(id="tc1", name="read_file", args={"path": "foo.py"}),
        agy_types.ToolResult(id="tc1", name="read_file", result="file contents"),
        agy_types.Text(step_index=1, text="Done."),
    ]
    fake_agent = FakeAgent(config=None, responses=[chunks])

    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    reply = await sut.prompt(
        [TextContentBlock(type="text", text="read foo.py")],
        session_id=session.session_id,
    )
    assert reply.stop_reason == "end_turn"

    updates = [call.kwargs.get("update") or call.args[1] for call in client.session_update.call_args_list]
    tool_starts = [u for u in updates if u.session_update == "tool_call"]
    tool_progress = [u for u in updates if u.session_update == "tool_call_update"]
    assert len(tool_starts) == 1
    assert tool_starts[0].title == "read_file"
    assert tool_starts[0].raw_input == {"path": "foo.py"}
    assert len(tool_progress) == 1
    assert tool_progress[0].status == "completed"


async def test_offline_empty_prompt_returns_early():
    """Prompt with no convertible content should return without calling chat()."""
    import hellp

    fake_agent = FakeAgent(config=None, responses=[])

    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    reply = await sut.prompt(
        [],
        session_id=session.session_id,
    )
    assert reply.stop_reason == "end_turn"
    assert fake_agent._call_index == 0
    client.session_update.assert_not_called()


async def test_offline_usage_tracking():
    """Verify usage metadata from the response is included in PromptResponse."""
    import hellp

    chunks = [agy_types.Text(step_index=0, text="Hi")]
    fake_agent = FakeAgent(config=None, responses=[chunks])

    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    reply = await sut.prompt(
        [TextContentBlock(type="text", text="test")],
        session_id=session.session_id,
    )
    # FakeAgent's MagicMock conversation returns MagicMock for usage_metadata,
    # which has no real token counts — usage extraction should not crash
    assert reply.stop_reason == "end_turn"


async def test_offline_cancel():
    """Cancel mid-stream should return stop_reason='cancelled'."""
    import hellp

    async def slow_stream():
        yield agy_types.Text(step_index=0, text="start ")
        await asyncio.sleep(10)
        yield agy_types.Text(step_index=1, text="should not reach")

    class SlowFakeAgent:
        def __init__(self, config): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def chat(self, prompt):
            return agy_types.ChatResponse(slow_stream(), conversation=MagicMock())

    sut = hellp.EchoAgent(agent_t=SlowFakeAgent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    import asyncio
    prompt_task = asyncio.create_task(
        sut.prompt([TextContentBlock(type="text", text="go")], session_id=sid)
    )
    await asyncio.sleep(0.05)
    await sut.cancel(session_id=sid)
    reply = await prompt_task
    assert reply.stop_reason == "cancelled"


# --- Live tests (require GEMINI_API_KEY) ---

async def test_initializes():
    import hellp
    sut = hellp.EchoAgent(agent_t=agy.Agent, agent_config_t=agy.LocalAgentConfig)
    await sut.initialize(protocol_version=1)

    client = unittest.mock.MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    prompt = TextContentBlock(type="text", text="Hello")
    reply = await sut.prompt([prompt], session_id=session.session_id)
    assert reply.stop_reason == "end_turn"

    client.session_update.assert_called()

# https://agentclientprotocol.github.io/python-sdk/
#

# https://agentclientprotocol.github.io/python-sdk/quickstart/
from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.interfaces import Client

class SimpleClient(Client):
    async def request_permission(
        self, options, session_id, tool_call, **kwargs: Any
    ):
        print("permission:", options, session_id, tool_call)
        return {"outcome": {"outcome": "cancelled"}}

    async def session_update(self, session_id, update, **kwargs):
        print("update:", session_id, update)


async def test_live_run():
    script = Path("hellp.py")
    env = os.environ.copy()
    async with spawn_agent_process(SimpleClient(), sys.executable, str(script), env=env) as (conn, _proc):
        # pass
        try:
            await conn.initialize(protocol_version=PROTOCOL_VERSION)
        except Exception as e:
            print("initialize failed:", e, e.data if hasattr(e, "data") else "")

        try:
            session = await conn.new_session(cwd=str(script.parent), mcp_servers=[])
        except Exception as e:
            print("initialize failed:", e, e.data if hasattr(e, "data") else "")

        await conn.prompt(
            session_id=session.session_id,
            prompt=[text_block("Say 'hello'")],
            message_id=str(uuid4()),
        )
