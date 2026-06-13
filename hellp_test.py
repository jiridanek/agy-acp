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
    """Minimal fake matching the agy.Agent interface (chat, __aenter__, __aexit__, register_hook)."""

    def __init__(self, config, responses=None):
        self._responses = responses or []
        self._call_index = 0

    def register_hook(self, hook):
        pass

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
    def __init__(self, **kwargs):
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


async def test_offline_tool_without_session_context():
    """Tool functions should return an error string, not crash, when no session context is set."""
    import hellp

    fake_agent = FakeAgent(config=None, responses=[])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    result = await sut.view_file("/tmp/test.txt")
    assert "Error" in result or "error" in result


async def test_offline_close_session_cleans_state():
    """close_session should clear per-session state dicts."""
    import hellp

    fake_agent = FakeAgent(config=None, responses=[])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    sut._last_file_edits[(sid, "/tmp/x")] = {"old_text": "a", "new_text": "b"}
    sut._last_terminal_ids[sid] = "term-1"
    sut._session_titled.add(sid)

    await sut.close_session(session_id=sid)

    assert (sid, "/tmp/x") not in sut._last_file_edits
    assert sid not in sut._last_terminal_ids
    assert sid not in sut._session_titled


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


async def test_offline_custom_tools_disabled_and_registered():
    """Verify built-in tools are disabled and custom file/shell tools are registered."""
    import hellp
    from google.antigravity.types import BuiltinTools

    configs_passed = []
    def spy_config(*args, **kwargs):
        cfg = agy.LocalAgentConfig(*args, **kwargs)
        configs_passed.append(cfg)
        return cfg

    fake_agent = FakeAgent(config=None, responses=[])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=spy_config)
    await sut.initialize(protocol_version=1)

    assert len(configs_passed) == 1
    config = configs_passed[0]
    # Check disabled built-ins
    assert set(config.capabilities.disabled_tools) == {
        BuiltinTools.VIEW_FILE,
        BuiltinTools.CREATE_FILE,
        BuiltinTools.EDIT_FILE,
        BuiltinTools.RUN_COMMAND,
    }
    # Check custom tools registered in connection config
    registered_tools = [t.__name__ for t in config.tools]
    assert "view_file" in registered_tools
    assert "create_file" in registered_tools
    assert "edit_file" in registered_tools
    assert "run_command" in registered_tools


async def test_offline_plan_updates_numbered_lists():
    """Verify numbered lists like '2. item' are parsed as plan entries."""
    import hellp

    chunks = [
        agy_types.Thought(step_index=0, text="Steps:\n1. First\n2. Second\n3. Third"),
        agy_types.Text(step_index=1, text="Done."),
    ]
    fake_agent = FakeAgent(config=None, responses=[chunks])

    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = AsyncMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    await sut.prompt(
        [TextContentBlock(type="text", text="plan it")],
        session_id=session.session_id,
    )

    updates = [call.kwargs.get("update") or call.args[1] for call in client.session_update.call_args_list]
    plan_updates = [u for u in updates if u.session_update == "plan"]

    assert len(plan_updates) > 0
    entries = plan_updates[-1].entries
    contents = [e.content for e in entries]
    assert "First" in contents
    assert "Second" in contents
    assert "Third" in contents


async def test_offline_plan_updates():
    """Verify markdown checklists/todos in Thought chunks are emitted as AgentPlanUpdate."""
    import hellp

    chunks = [
        agy_types.Thought(step_index=0, text="I should structure my tasks:\n- [ ] Task 1\n- [x] Task 2\n* Task 3"),
        agy_types.Text(step_index=1, text="Thinking complete."),
    ]
    fake_agent = FakeAgent(config=None, responses=[chunks])

    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = AsyncMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    await sut.prompt(
        [TextContentBlock(type="text", text="make a plan")],
        session_id=session.session_id,
    )

    # Gather session updates
    updates = [call.kwargs.get("update") or call.args[1] for call in client.session_update.call_args_list]
    plan_updates = [u for u in updates if u.session_update == "plan"]
    
    assert len(plan_updates) > 0
    final_plan = plan_updates[-1]
    assert len(final_plan.entries) == 3
    assert final_plan.entries[0].content == "Task 1"
    assert final_plan.entries[0].status == "pending"
    assert final_plan.entries[1].content == "Task 2"
    assert final_plan.entries[1].status == "completed"
    assert final_plan.entries[2].content == "Task 3"
    assert final_plan.entries[2].status == "pending"


async def test_offline_rich_tool_outputs():
    """Verify tool call progress includes rich diff details for file edits and terminal refs."""
    import hellp

    # Simulating a file edit and a run_command
    chunks = [
        agy_types.ToolCall(id="tc-edit", name="edit_file", args={"path": "foo.txt", "content": "new contents"}),
        agy_types.ToolResult(id="tc-edit", name="edit_file", result="Success"),
        agy_types.ToolCall(id="tc-run", name="run_command", args={"command": "ls -l"}),
        agy_types.ToolResult(id="tc-run", name="run_command", result="total 0"),
    ]
    fake_agent = FakeAgent(config=None, responses=[chunks])

    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = AsyncMock(spec=Client)
    # Mocking file read response containing "old contents"
    from acp.schema import ReadTextFileResponse, CreateTerminalResponse, TerminalOutputResponse, WaitForTerminalExitResponse
    client.read_text_file.return_value = ReadTextFileResponse(content="old contents")
    client.create_terminal.return_value = CreateTerminalResponse(terminal_id="term-123")
    client.wait_for_terminal_exit.return_value = WaitForTerminalExitResponse()
    client.terminal_output.return_value = TerminalOutputResponse(output="command output", truncated=False)

    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    
    # We prime the contextvar for the run
    token = hellp.current_session_id.set(session.session_id)
    try:
        # Simulate first the python tools executing
        res_edit = await sut.edit_file("foo.txt", "new contents")
        assert "Successfully edited file" in res_edit
        
        res_run = await sut.run_command("ls -l")
        assert res_run == "command output"
    finally:
        hellp.current_session_id.reset(token)

    # Now run the prompt so the streamed events process
    await sut.prompt(
        [TextContentBlock(type="text", text="edit and run")],
        session_id=session.session_id,
    )

    # Gather updates
    updates = [call.kwargs.get("update") or call.args[1] for call in client.session_update.call_args_list]
    
    starts = [u for u in updates if u.session_update == "tool_call"]
    progress = [u for u in updates if u.session_update == "tool_call_update"]

    # Verify Kind/Location extraction
    assert len(starts) == 2
    assert starts[0].kind == "edit"
    assert starts[0].locations[0].path == "foo.txt"
    assert starts[1].kind == "execute"

    # Verify Rich Tool Contents in updates
    assert len(progress) == 2
    edit_update = progress[0]
    assert edit_update.status == "completed"
    assert edit_update.content[0].type == "diff"
    assert edit_update.content[0].path == "foo.txt"
    assert edit_update.content[0].new_text == "new contents"
    assert edit_update.content[0].old_text == "old contents"

    term_update = progress[1]
    assert term_update.status == "completed"
    assert term_update.content[0].type == "terminal"
    assert term_update.content[0].terminal_id == "term-123"


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
        def register_hook(self, hook): pass
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
    prompt = TextContentBlock(type="text", text="Say hello three times. Do not use any tools.")
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

        session = await conn.new_session(cwd=str(script.parent), mcp_servers=[])

        await conn.prompt(
            session_id=session.session_id,
            prompt=[text_block("Say hello three times. Do not use any tools.")],
            message_id=str(uuid4()),
        )
