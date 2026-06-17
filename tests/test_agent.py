import asyncio
import os
import sys
import unittest.mock
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import google.antigravity as agy
import pytest
from acp.interfaces import Client
from acp.schema import ClientCapabilities, TextContentBlock

from agy_acp.agent import EchoAgent
from agy_acp.session import SessionState, SessionStore, current_session_id
from agy_acp.skills import _skills_paths
from conftest import FakeAgent, FakeConfig, _TEST_CLIENT_CAPS


async def test_offline_prompt_text():
    """Text prompt without an LLM — uses FakeAgent with canned response."""
    chunks = [
        agy.types.Thought(step_index=0, text="let me think"),
        agy.types.Text(step_index=1, text="Hello "),
        agy.types.Text(step_index=1, text="back!"),
    ]
    fake_agent = FakeAgent(config=None, responses=[chunks])

    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    reply = await sut.prompt(
        [TextContentBlock(type="text", text="Hi")],
        session_id=session.session_id,
    )

    assert reply.stop_reason == "end_turn"

    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]
    thought_updates = [u for u in updates if u.session_update == "agent_thought_chunk"]
    message_updates = [u for u in updates if u.session_update == "agent_message_chunk"]
    assert len(thought_updates) == 1
    assert thought_updates[0].content.text == "let me think"
    assert len(message_updates) == 2
    assert message_updates[0].content.text == "Hello "
    assert message_updates[1].content.text == "back!"


async def test_offline_prompt_with_tool_calls():
    """ToolCalls in the stream are passed through (hooks handle start/complete in real agent)."""
    chunks = [
        agy.types.ToolCall(id="tc1", name="read_file", args={"path": "foo.py"}),
        agy.types.Text(step_index=1, text="Done."),
    ]
    fake_agent = FakeAgent(config=None, responses=[chunks])

    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    reply = await sut.prompt(
        [TextContentBlock(type="text", text="read foo.py")],
        session_id=session.session_id,
    )
    assert reply.stop_reason == "end_turn"

    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]
    message_updates = [u for u in updates if u.session_update == "agent_message_chunk"]
    assert len(message_updates) == 1
    assert message_updates[0].content.text == "Done."


async def test_offline_tool_execution_populates_edit_state():
    """Tool functions populate _last_file_edits so PostToolCallHook can send rich diffs."""
    from acp.schema import ReadTextFileResponse

    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = AsyncMock(spec=Client)
    client.read_text_file.return_value = ReadTextFileResponse(content="old content")
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id
    sut._active_session_id = sid

    result = await sut.edit_file("test.py", "old", "new")
    assert "Successfully edited" in result
    assert "test.py" in sut._sessions[sid].last_file_edits
    assert sut._sessions[sid].last_file_edits["test.py"]["old_text"] == "old content"
    assert sut._sessions[sid].last_file_edits["test.py"]["new_text"] == "new content"


async def test_offline_edit_file_not_found():
    """edit_file returns error when old_string is not in the file."""
    from acp.schema import ReadTextFileResponse

    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = AsyncMock(spec=Client)
    client.read_text_file.return_value = ReadTextFileResponse(content="hello world")
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sut._active_session_id = session.session_id

    result = await sut.edit_file("test.py", "nonexistent", "replacement")
    assert "old_string not found" in result


async def test_offline_tool_works_without_contextvar():
    """Tool functions must work even when ContextVar is not set (SDK dispatches on background tasks)."""
    from acp.schema import ReadTextFileResponse

    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = AsyncMock(spec=Client)
    client.read_text_file.return_value = ReadTextFileResponse(content="hello world")
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sut._active_session_id = session.session_id

    # Call view_file WITHOUT setting current_session_id ContextVar — simulates SDK background task
    result = await sut.view_file("/tmp/test.txt")
    assert result == "hello world"
    assert "Error" not in result


async def test_offline_tool_without_session_context():
    """Tool functions should return an error string, not crash, when no session context is set."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    result = await sut.view_file("/tmp/test.txt")
    assert "Error" in result or "error" in result


async def test_offline_close_session_cleans_state():
    """close_session should clear per-session state dicts."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    sut._sessions[sid].last_file_edits["/tmp/x"] = {"old_text": "a", "new_text": "b"}
    sut._sessions[sid].last_terminal_id = "term-1"
    sut._sessions[sid].state.title = "test"

    await sut.close_session(session_id=sid)

    assert sid not in sut._sessions


async def test_offline_empty_prompt_returns_early():
    """Prompt with no convertible content should return without calling chat()."""
    fake_agent = FakeAgent(config=None, responses=[])

    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

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
    """Verify built-in tools are disabled, custom tools are registered, and policies=[allow_all()]."""
    from google.antigravity.types import BuiltinTools

    configs_passed = []

    def spy_config(*args, **kwargs):
        cfg = agy.LocalAgentConfig(*args, **kwargs)
        configs_passed.append(cfg)
        return cfg

    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=spy_config)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)
    sut.on_connect(conn=MagicMock(spec=Client))
    await sut.new_session(cwd=".")

    assert len(configs_passed) == 1
    config = configs_passed[0]
    enabled = set(config.capabilities.enabled_tools)
    assert BuiltinTools.VIEW_FILE not in enabled
    assert BuiltinTools.CREATE_FILE not in enabled
    assert BuiltinTools.EDIT_FILE not in enabled
    assert BuiltinTools.RUN_COMMAND not in enabled
    assert BuiltinTools.LIST_DIR in enabled
    assert BuiltinTools.FIND_FILE in enabled
    assert BuiltinTools.SEARCH_DIR in enabled
    registered_tools = [t.__name__ for t in config.tools]
    assert "view_file" in registered_tools
    assert "create_file" in registered_tools
    assert "edit_file" in registered_tools
    assert "run_command" in registered_tools
    policy_names = [p.name for p in config.policies]
    assert "allow_all" in policy_names


async def test_offline_no_terminal_leaves_builtin_run_command():
    """When client has terminal=False, SDK's built-in run_command stays enabled."""
    from acp.schema import FileSystemCapabilities
    from google.antigravity.types import BuiltinTools

    no_terminal_caps = ClientCapabilities(
        fs=FileSystemCapabilities(read_text_file=True, write_text_file=True),
        terminal=False,
    )

    configs_passed = []

    def spy_config(*args, **kwargs):
        cfg = agy.LocalAgentConfig(*args, **kwargs)
        configs_passed.append(cfg)
        return cfg

    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=spy_config)
    await sut.initialize(protocol_version=1, client_capabilities=no_terminal_caps)
    sut.on_connect(conn=MagicMock(spec=Client))
    await sut.new_session(cwd=".")

    config = configs_passed[0]
    enabled = set(config.capabilities.enabled_tools)
    assert BuiltinTools.VIEW_FILE not in enabled
    assert BuiltinTools.CREATE_FILE not in enabled
    assert BuiltinTools.EDIT_FILE not in enabled
    assert BuiltinTools.RUN_COMMAND in enabled
    registered = [t.__name__ for t in config.tools]
    assert "view_file" in registered
    assert "run_command" not in registered


async def test_offline_plan_updates_numbered_lists():
    """Verify numbered lists like '2. item' are parsed as plan entries."""
    chunks = [
        agy.types.Thought(step_index=0, text="Steps:\n1. First\n2. Second\n3. Third"),
        agy.types.Text(step_index=1, text="Done."),
    ]
    fake_agent = FakeAgent(config=None, responses=[chunks])

    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = AsyncMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    await sut.prompt(
        [TextContentBlock(type="text", text="plan it")],
        session_id=session.session_id,
    )

    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]
    plan_updates = [u for u in updates if u.session_update == "plan"]

    assert len(plan_updates) > 0
    entries = plan_updates[-1].entries
    contents = [e.content for e in entries]
    assert "First" in contents
    assert "Second" in contents
    assert "Third" in contents


async def test_offline_plan_updates():
    """Verify markdown checklists/todos in Thought chunks are emitted as AgentPlanUpdate."""
    chunks = [
        agy.types.Thought(
            step_index=0,
            text="I should structure my tasks:\n- [ ] Task 1\n- [x] Task 2\n* Task 3",
        ),
        agy.types.Text(step_index=1, text="Thinking complete."),
    ]
    fake_agent = FakeAgent(config=None, responses=[chunks])

    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = AsyncMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    await sut.prompt(
        [TextContentBlock(type="text", text="make a plan")],
        session_id=session.session_id,
    )

    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]
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
    chunks = [
        agy.types.ToolCall(
            id="tc-edit",
            name="edit_file",
            args={"path": "foo.txt", "content": "new contents"},
        ),
        agy.types.ToolResult(id="tc-edit", name="edit_file", result="Success"),
        agy.types.ToolCall(id="tc-run", name="run_command", args={"command": "ls -l"}),
        agy.types.ToolResult(id="tc-run", name="run_command", result="total 0"),
    ]
    fake_agent = FakeAgent(config=None, responses=[chunks])

    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = AsyncMock(spec=Client)
    client.request_permission.return_value = MagicMock(
        outcome=MagicMock(option_id="approve")
    )
    from acp.schema import (
        CreateTerminalResponse,
        ReadTextFileResponse,
        TerminalOutputResponse,
        WaitForTerminalExitResponse,
    )

    client.read_text_file.return_value = ReadTextFileResponse(content="old contents")
    client.create_terminal.return_value = CreateTerminalResponse(terminal_id="term-123")
    client.wait_for_terminal_exit.return_value = WaitForTerminalExitResponse()
    client.terminal_output.return_value = TerminalOutputResponse(
        output="command output", truncated=False
    )

    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    sut._sessions[sid].state.mode = "accept_edits"

    sut._sessions[sid].last_file_edits["foo.txt"] = {
        "old_text": "old contents",
        "new_text": "new contents",
    }
    sut._sessions[sid].last_terminal_id = "term-123"

    await sut.prompt(
        [TextContentBlock(type="text", text="edit and run")],
        session_id=sid,
    )

    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]

    starts = [u for u in updates if u.session_update == "tool_call"]
    progress = [u for u in updates if u.session_update == "tool_call_update"]

    assert len(starts) == 1
    assert starts[0].kind == "edit"
    assert starts[0].locations[0].path == "foo.txt"

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


async def test_offline_session_modes():
    """Verify session modes are declared and set_session_mode works."""
    fake_agent = FakeAgent(
        config=None,
        responses=[
            [agy.types.Text(step_index=0, text="plan step 1")],
        ],
    )
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    assert session.modes is not None
    assert session.modes.current_mode_id == "agent"
    assert len(session.modes.available_modes) == 5
    mode_ids = {m.id for m in session.modes.available_modes}
    assert mode_ids == {"agent", "accept_edits", "plan", "dont_ask", "bypass"}

    await sut.set_session_mode(mode_id="plan", session_id=sid)

    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]
    mode_updates = [u for u in updates if u.session_update == "current_mode_update"]
    assert len(mode_updates) == 1
    assert mode_updates[0].current_mode_id == "plan"

    reply = await sut.prompt(
        [TextContentBlock(type="text", text="do something")],
        session_id=sid,
    )
    assert reply.stop_reason == "end_turn"


async def test_offline_config_option_model():
    """set_config_option with config_id='model' updates the session model."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    assert any(opt.id == "model" for opt in session.config_options)
    assert any(opt.id == "thinking_level" for opt in session.config_options)

    resp = await sut.set_config_option(
        config_id="model", session_id=sid, value="gemini-2.5-flash"
    )
    assert sut._sessions[sid].state.model == "gemini-2.5-flash"

    model_opt = next(o for o in resp.config_options if o.id == "model")
    assert model_opt.current_value == "gemini-2.5-flash"

    resp2 = await sut.set_config_option(
        config_id="thinking_level", session_id=sid, value="high"
    )
    assert sut._sessions[sid].state.thinking_level == "high"

    thinking_opt = next(o for o in resp2.config_options if o.id == "thinking_level")
    assert thinking_opt.current_value == "high"


async def test_offline_session_persistence(tmp_path):
    """new_session → prompt → list_sessions finds it → close_session → list_sessions doesn't."""
    store = SessionStore(path=tmp_path / "sessions.json")
    chunks = [agy.types.Text(step_index=0, text="hi")]
    fake_agent = FakeAgent(config=None, responses=[chunks])

    sut = EchoAgent(
        agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig, store=store
    )
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd="/tmp/myproject")
    sid = session.session_id

    await sut.prompt(
        [TextContentBlock(type="text", text="hello world")],
        session_id=sid,
    )

    listing = await sut.list_sessions(cwd="/tmp/myproject")
    assert len(listing.sessions) == 1
    assert listing.sessions[0].session_id == sid
    assert listing.sessions[0].cwd == "/tmp/myproject"
    assert listing.sessions[0].title == "hello world"

    listing_other = await sut.list_sessions(cwd="/other")
    assert len(listing_other.sessions) == 0

    await sut.close_session(session_id=sid)

    listing_after = await sut.list_sessions(cwd="/tmp/myproject")
    assert len(listing_after.sessions) == 0


async def test_offline_load_session(tmp_path):
    """load_session restores mode and config from a previously saved session."""
    store = SessionStore(path=tmp_path / "sessions.json")
    chunks = [agy.types.Text(step_index=0, text="response")]
    fake_agent = FakeAgent(
        config=None, responses=[chunks, [agy.types.Text(step_index=0, text="resumed")]]
    )

    sut = EchoAgent(
        agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig, store=store
    )
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd="/tmp/proj")
    sid = session.session_id

    await sut.set_session_mode(mode_id="plan", session_id=sid)
    await sut.prompt(
        [TextContentBlock(type="text", text="plan something")],
        session_id=sid,
    )

    loaded = await sut.load_session(session_id=sid, cwd="/tmp/proj")
    assert loaded is not None
    assert loaded.modes.current_mode_id == "plan"
    assert sut._sessions[sid].state.mode == "plan"


async def test_offline_usage_tracking():
    """Verify usage metadata from the response is included in PromptResponse."""
    chunks = [agy.types.Text(step_index=0, text="Hi")]
    fake_agent = FakeAgent(config=None, responses=[chunks])

    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    reply = await sut.prompt(
        [TextContentBlock(type="text", text="test")],
        session_id=session.session_id,
    )
    assert reply.stop_reason == "end_turn"


async def test_offline_cost_estimation():
    """Cost is computed from model pricing and included in UsageUpdate."""
    chunks = [agy.types.Text(step_index=0, text="Hi")]

    conv_mock = MagicMock()
    conv_mock.last_turn_usage = MagicMock(
        prompt_token_count=1000,
        candidates_token_count=500,
        total_token_count=1500,
        thoughts_token_count=0,
        cached_content_token_count=0,
    )

    class CostFakeAgent:
        def __init__(self, config):
            pass

        def register_hook(self, hook):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        @property
        def conversation_id(self):
            return None

        async def chat(self, prompt):
            async def stream():
                for c in chunks:
                    yield c

            return agy.types.ChatResponse(stream(), conversation=conv_mock)

    sut = EchoAgent(agent_t=CostFakeAgent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id
    # default model is gemini-3.1-flash-lite: (0.25, 1.50) per 1M tokens
    reply = await sut.prompt(
        [TextContentBlock(type="text", text="test")],
        session_id=sid,
    )
    assert reply.stop_reason == "end_turn"

    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]
    usage_updates = [u for u in updates if u.session_update == "usage_update"]
    assert len(usage_updates) == 1
    assert usage_updates[0].cost is not None
    assert usage_updates[0].cost.currency == "USD"
    # 1000 * 0.25/1M + 500 * 1.50/1M = 0.00025 + 0.00075 = 0.001
    assert abs(usage_updates[0].cost.amount - 0.001) < 1e-8


async def test_offline_cost_pro_long_context_surcharge():
    """Pro models over 200k tokens get 2x input, 1.5x output surcharge."""
    conv_mock = MagicMock()
    conv_mock.last_turn_usage = MagicMock(
        prompt_token_count=300_000,
        candidates_token_count=1000,
        total_token_count=301_000,
        thoughts_token_count=0,
        cached_content_token_count=0,
    )

    class CostProAgent:
        def __init__(self, config):
            pass

        def register_hook(self, hook):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        @property
        def conversation_id(self):
            return None

        async def chat(self, prompt):
            async def stream():
                yield agy.types.Text(step_index=0, text="Hi")

            return agy.types.ChatResponse(stream(), conversation=conv_mock)

    sut = EchoAgent(agent_t=CostProAgent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id
    sut._sessions[sid].state.model = "gemini-2.5-pro"

    await sut.prompt([TextContentBlock(type="text", text="test")], session_id=sid)

    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]
    usage_updates = [u for u in updates if u.session_update == "usage_update"]
    assert len(usage_updates) == 1
    # gemini-2.5-pro base: (1.25, 10.00), surcharge: (2.50, 15.00)
    # 300000 * 2.50/1M + 1000 * 15.00/1M = 0.75 + 0.015 = 0.765
    assert abs(usage_updates[0].cost.amount - 0.765) < 1e-6


async def test_offline_cost_unknown_model():
    """Unknown model produces no cost (cost=None)."""
    chunks = [agy.types.Text(step_index=0, text="Hi")]

    conv_mock = MagicMock()
    conv_mock.last_turn_usage = MagicMock(
        prompt_token_count=100,
        candidates_token_count=50,
        total_token_count=150,
        thoughts_token_count=0,
        cached_content_token_count=0,
    )

    class CostFakeAgent2:
        def __init__(self, config):
            pass

        def register_hook(self, hook):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        @property
        def conversation_id(self):
            return None

        async def chat(self, prompt):
            async def stream():
                for c in chunks:
                    yield c

            return agy.types.ChatResponse(stream(), conversation=conv_mock)

    sut = EchoAgent(agent_t=CostFakeAgent2, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id
    sut._sessions[sid].state.model = "unknown-model-xyz"

    await sut.prompt(
        [TextContentBlock(type="text", text="test")],
        session_id=sid,
    )

    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]
    usage_updates = [u for u in updates if u.session_update == "usage_update"]
    assert len(usage_updates) == 1
    assert usage_updates[0].cost is None


async def test_offline_cancel():
    """Cancel mid-stream should return stop_reason='cancelled'."""

    async def slow_stream():
        yield agy.types.Text(step_index=0, text="start ")
        await asyncio.sleep(10)
        yield agy.types.Text(step_index=1, text="should not reach")

    class SlowFakeAgent:
        def __init__(self, config):
            pass

        def register_hook(self, hook):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def chat(self, prompt):
            return agy.types.ChatResponse(slow_stream(), conversation=MagicMock())

    sut = EchoAgent(agent_t=SlowFakeAgent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    prompt_task = asyncio.create_task(
        sut.prompt([TextContentBlock(type="text", text="go")], session_id=sid)
    )
    await asyncio.sleep(0.05)
    await sut.cancel(session_id=sid)
    reply = await prompt_task
    assert reply.stop_reason == "cancelled"


async def test_offline_auth_methods_declared():
    """InitializeResponse advertises GEMINI_API_KEY as env var auth method."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    resp = await sut.initialize(
        protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS
    )

    assert resp.auth_methods is not None
    assert len(resp.auth_methods) == 1
    method = resp.auth_methods[0]
    assert method.type == "env_var"
    assert method.id == "gemini_api_key"
    assert len(method.vars) == 1
    assert method.vars[0].name == "GEMINI_API_KEY"


async def test_offline_authenticate():
    """authenticate() returns AuthenticateResponse without error."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)
    resp = await sut.authenticate(method_id="gemini_api_key")
    assert resp is not None


async def test_offline_model_switching():
    """set_session_model changes the model for a session."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    assert session.models is not None
    assert session.models.current_model_id == "gemini-3.1-flash-lite"
    assert len(session.models.available_models) == 7

    configs_seen = []
    original_config_t = FakeConfig

    def tracking_config_t(**kwargs):
        configs_seen.append(kwargs)
        return original_config_t(**kwargs)

    sut._agent_config_t = tracking_config_t

    resp = await sut.set_session_model(model_id="gemini-2.5-flash", session_id=sid)
    assert resp is not None
    assert sut._sessions[sid].state.model == "gemini-2.5-flash"
    assert len(configs_seen) == 1
    gemini_cfg = configs_seen[0]["gemini_config"]
    assert gemini_cfg.models.default.name == "gemini-2.5-flash"


async def test_offline_rebuild_passes_thinking_level():
    """Changing thinking level via config option rebuilds the agent with correct config."""
    configs_seen = []
    original_config_t = FakeConfig

    def tracking_config_t(**kwargs):
        configs_seen.append(kwargs)
        return original_config_t(**kwargs)

    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    sut._agent_config_t = tracking_config_t

    await sut.set_config_option(
        config_id="thinking_level", session_id=sid, value="high"
    )
    assert len(configs_seen) == 1
    gemini_cfg = configs_seen[0]["gemini_config"]
    assert gemini_cfg.models.default.generation.thinking_level.value == "high"


async def test_offline_model_switch_preserves_conversation():
    """Switching model preserves the current conversation_id."""
    configs_seen = []

    def tracking_config_t(**kwargs):
        configs_seen.append(kwargs)
        return FakeConfig(**kwargs)

    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)
    sut._agent_config_t = tracking_config_t

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    type(sut._sessions[sid].agent).conversation_id = property(lambda self: "conv-keep-me")

    await sut.set_session_model(model_id="gemini-2.5-flash", session_id=sid)
    assert configs_seen[-1].get("conversation_id") == "conv-keep-me"

    configs_seen.clear()
    type(sut._sessions[sid].agent).conversation_id = property(lambda self: "conv-keep-me-2")
    await sut.set_config_option(config_id="thinking_level", session_id=sid, value="low")
    assert configs_seen[-1].get("conversation_id") == "conv-keep-me-2"


async def test_offline_set_config_option_unchanged_skips_rebuild():
    """set_config_option with the current value should not trigger _rebuild_agent."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    rebuild_calls = 0
    original_rebuild = sut._rebuild_agent

    async def counting_rebuild(*a, **kw):
        nonlocal rebuild_calls
        rebuild_calls += 1
        await original_rebuild(*a, **kw)

    sut._rebuild_agent = counting_rebuild

    default_model = sut._sessions[sid].state.model
    default_thinking = sut._sessions[sid].state.thinking_level
    default_context = sut._sessions[sid].state.context_level
    default_mode = sut._sessions[sid].state.mode

    await sut.set_config_option(config_id="model", session_id=sid, value=default_model)
    await sut.set_config_option(config_id="thinking_level", session_id=sid, value=default_thinking)
    await sut.set_config_option(config_id="context", session_id=sid, value=default_context)
    await sut.set_config_option(config_id="mode", session_id=sid, value=default_mode)
    assert rebuild_calls == 0

    await sut.set_config_option(config_id="model", session_id=sid, value="gemini-2.5-flash")
    assert rebuild_calls == 1


async def test_offline_rebuild_agent_rollback():
    """If _rebuild_agent fails, the old agent is restored."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id
    original_agent = sut._sessions[sid].agent

    call_count = 0

    def failing_config_t(**kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("config creation failed")

    sut._agent_config_t = failing_config_t

    with pytest.raises(ValueError, match="config creation failed"):
        await sut.set_session_model(model_id="gemini-2.5-flash", session_id=sid)

    assert sut._sessions[sid].agent is original_agent


async def test_offline_rebuild_uses_valid_local_agent_config():
    """_rebuild_agent must produce a valid LocalAgentConfig (no conflicting model fields)."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(
        agent_t=lambda cfg: fake_agent, agent_config_t=agy.LocalAgentConfig
    )
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    await sut.set_session_model(model_id="gemini-2.5-flash", session_id=sid)


async def test_offline_model_persisted_in_session(tmp_path):
    """Model choice is persisted and restored on load_session."""
    store = SessionStore(path=tmp_path / "sessions.json")
    chunks = [agy.types.Text(step_index=0, text="ok")]
    fake_agent = FakeAgent(config=None, responses=[chunks])
    sut = EchoAgent(
        agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig, store=store
    )
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd="/tmp")
    sid = session.session_id
    await sut.set_session_model(model_id="gemini-2.5-flash-lite", session_id=sid)

    await sut.prompt([TextContentBlock(type="text", text="hi")], session_id=sid)

    stored = store.load(sid)
    assert stored.model == "gemini-2.5-flash-lite"

    sut2 = EchoAgent(
        agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig, store=store
    )
    await sut2.initialize(protocol_version=1)
    sut2.on_connect(conn=MagicMock(spec=Client))

    loaded = await sut2.load_session(cwd="/tmp", session_id=sid)
    assert loaded.models is not None
    assert loaded.models.current_model_id == "gemini-2.5-flash-lite"


async def test_offline_close_session_cleans_model():
    """close_session removes model state."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id
    assert sid in sut._sessions
    sut._active_tasks[sid] = asyncio.current_task()

    await sut.close_session(session_id=sid)
    assert sid not in sut._sessions
    assert sid not in sut._active_tasks


async def test_offline_reset_command():
    """/reset command rebuilds agent and clears session title."""
    chunks1 = [agy.types.Text(step_index=0, text="first response")]
    chunks2 = [agy.types.Text(step_index=0, text="after reset")]
    fake_agent = FakeAgent(config=None, responses=[chunks1, chunks2])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    await sut.prompt([TextContentBlock(type="text", text="hello")], session_id=sid)
    assert sut._sessions[sid].state.title is not None

    client.reset_mock()
    reply = await sut.prompt(
        [TextContentBlock(type="text", text="/reset")], session_id=sid
    )
    assert reply.stop_reason == "end_turn"
    assert sut._sessions[sid].state.title is None

    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]
    message_updates = [u for u in updates if u.session_update == "agent_message_chunk"]
    assert any("reset" in u.content.text.lower() for u in message_updates)


async def test_offline_fork_session(tmp_path):
    """fork_session creates a new session copying settings from the original."""
    store = SessionStore(path=tmp_path / "sessions.json")
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(
        agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig, store=store
    )
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd="/project")
    sid = session.session_id
    sut._sessions[sid].state.title = "My Session"
    await sut.set_config_option(
        config_id="model", session_id=sid, value="gemini-2.5-pro"
    )
    await sut.set_config_option(
        config_id="thinking_level", session_id=sid, value="high"
    )

    forked = await sut.fork_session(cwd="/project", session_id=sid)
    fid = forked.session_id

    assert fid != sid
    assert sut._sessions[fid].state.mode == "agent"
    assert sut._sessions[fid].state.model == "gemini-2.5-pro"
    assert sut._sessions[fid].state.thinking_level == "high"
    assert sut._sessions[fid].state.title == "My Session (fork)"
    assert forked.models is not None
    assert forked.models.current_model_id == "gemini-2.5-pro"

    stored = store.load(fid)
    assert stored is not None
    assert stored.model == "gemini-2.5-pro"


async def test_offline_fork_capability_declared():
    """InitializeResponse declares fork capability."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    resp = await sut.initialize(
        protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS
    )

    assert resp.agent_capabilities.session_capabilities.fork is not None
    assert resp.agent_capabilities.session_capabilities.resume is not None


async def test_offline_help_command():
    """/help command lists available commands without calling the LLM."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    client.reset_mock()
    reply = await sut.prompt(
        [TextContentBlock(type="text", text="/help")], session_id=sid
    )
    assert reply.stop_reason == "end_turn"

    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]
    message_updates = [u for u in updates if u.session_update == "agent_message_chunk"]
    assert len(message_updates) == 1
    assert "/reset" in message_updates[0].content.text
    assert "/help" in message_updates[0].content.text

    assert fake_agent._call_index == 0


async def test_offline_cost_command():
    """/cost shows model and cumulative cost."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id
    sut._sessions[sid].cumulative_cost = 0.042

    reply = await sut.prompt(
        [TextContentBlock(type="text", text="/cost")], session_id=sid
    )
    assert reply.stop_reason == "end_turn"

    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]
    msg = [u for u in updates if u.session_update == "agent_message_chunk"][
        0
    ].content.text
    assert "gemini-3.1-flash-lite" in msg
    assert "0.042" in msg


async def test_offline_model_command_show():
    """/model with no arg shows current model and available models."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    reply = await sut.prompt(
        [TextContentBlock(type="text", text="/model")], session_id=sid
    )
    assert reply.stop_reason == "end_turn"

    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]
    msg = [u for u in updates if u.session_update == "agent_message_chunk"][
        0
    ].content.text
    assert "gemini-3.1-flash-lite" in msg
    assert "gemini-2.5-pro" in msg


async def test_offline_model_command_switch():
    """/model <id> switches the model."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    reply = await sut.prompt(
        [TextContentBlock(type="text", text="/model gemini-2.5-flash")], session_id=sid
    )
    assert reply.stop_reason == "end_turn"
    assert sut._sessions[sid].state.model == "gemini-2.5-flash"


async def test_offline_thinking_command():
    """/thinking shows and sets thinking level."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    reply = await sut.prompt(
        [TextContentBlock(type="text", text="/thinking")], session_id=sid
    )
    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]
    msg = [u for u in updates if u.session_update == "agent_message_chunk"][
        0
    ].content.text
    assert "medium" in msg

    reply = await sut.prompt(
        [TextContentBlock(type="text", text="/thinking high")], session_id=sid
    )
    assert sut._sessions[sid].state.thinking_level == "high"


async def test_offline_context_command():
    """/context shows and sets context retention level."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    reply = await sut.prompt(
        [TextContentBlock(type="text", text="/context")], session_id=sid
    )
    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]
    msg = [u for u in updates if u.session_update == "agent_message_chunk"][0].content.text
    assert "normal" in msg
    assert "50,000" in msg

    reply = await sut.prompt(
        [TextContentBlock(type="text", text="/context max")], session_id=sid
    )
    assert sut._sessions[sid].state.context_level == "max"


async def test_offline_clear_is_reset_alias():
    """/clear works the same as /reset."""
    chunks = [agy.types.Text(step_index=0, text="response")]
    fake_agent = FakeAgent(config=None, responses=[chunks])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id
    sut._sessions[sid].state.title = "test"

    reply = await sut.prompt(
        [TextContentBlock(type="text", text="/clear")], session_id=sid
    )
    assert reply.stop_reason == "end_turn"
    assert sut._sessions[sid].state.title is None


async def test_offline_load_session_rebuilds_with_conversation_id(tmp_path, monkeypatch):
    """load_session rebuilds the agent with saved conversation_id."""
    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    (traj_dir / "traj-conv-xyz").write_text("{}")
    import agy_acp.session
    monkeypatch.setattr(agy_acp.session, "_DEFAULT_SAVE_DIR", str(traj_dir))

    store = SessionStore(path=tmp_path / "sessions.json")
    store.save(
        "sess-1",
        SessionState(
            session_id="sess-1",
            conversation_id="conv-xyz",
            cwd="/project",
            model="gemini-3.1-flash-lite",
            title="Test",
            updated_at="2026-01-01T00:00:00Z",
        ),
    )

    configs_seen = []

    def tracking_config_t(**kwargs):
        configs_seen.append(kwargs)
        return FakeConfig(**kwargs)

    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(
        agent_t=lambda cfg: fake_agent, agent_config_t=tracking_config_t, store=store
    )
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)
    sut.on_connect(conn=MagicMock(spec=Client))

    await sut.load_session(cwd="/project", session_id="sess-1")

    assert len(configs_seen) >= 1
    assert configs_seen[-1].get("conversation_id") == "conv-xyz"


async def test_offline_additional_directories():
    """additional_directories are passed as workspaces to the agent config."""
    configs_seen = []

    def tracking_config_t(**kwargs):
        configs_seen.append(kwargs)
        return FakeConfig(**kwargs)

    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)
    sut._agent_config_t = tracking_config_t

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(
        cwd="/project",
        additional_directories=["/lib", "/shared"],
    )
    sid = session.session_id

    assert sut._sessions[sid].additional_dirs == ["/lib", "/shared"]

    await sut.set_session_model(model_id="gemini-2.5-flash", session_id=sid)

    assert len(configs_seen) >= 1
    workspaces = configs_seen[-1].get("workspaces")
    assert workspaces == ["/project", "/lib", "/shared"]

    await sut.close_session(session_id=sid)
    assert sid not in sut._sessions


async def test_offline_agent_info_declared():
    """InitializeResponse includes agent name, version, and title."""
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    resp = await sut.initialize(
        protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS
    )

    assert resp.agent_info is not None
    assert resp.agent_info.name == "agy-acp"
    assert resp.agent_info.version == "0.1.0"
    assert resp.agent_info.title == "Antigravity ACP Adapter"


async def test_offline_resume_session(tmp_path, monkeypatch):
    """resume_session restores mode, model, thinking level, and rebuilds agent with conversation_id."""
    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    (traj_dir / "traj-conv-abc-123").write_text("{}")
    import agy_acp.session
    monkeypatch.setattr(agy_acp.session, "_DEFAULT_SAVE_DIR", str(traj_dir))

    store = SessionStore(path=tmp_path / "sessions.json")
    chunks = [agy.types.Text(step_index=0, text="hello")]
    fake_agent = FakeAgent(config=None, responses=[chunks])

    configs_seen = []
    original_config_t = FakeConfig

    def tracking_config_t(**kwargs):
        configs_seen.append(kwargs)
        return original_config_t(**kwargs)

    sut = EchoAgent(
        agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig, store=store
    )
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd="/project")
    sid = session.session_id
    await sut.set_config_option(
        config_id="model", session_id=sid, value="gemini-2.5-pro"
    )
    await sut.set_config_option(
        config_id="thinking_level", session_id=sid, value="high"
    )

    sut._sessions[sid].agent.conversation_id_value = "conv-abc-123"
    type(sut._sessions[sid].agent).conversation_id = property(
        lambda self: getattr(self, "conversation_id_value", None)
    )

    await sut.prompt([TextContentBlock(type="text", text="hi")], session_id=sid)

    stored = store.load(sid)
    assert stored.conversation_id == "conv-abc-123"

    sut2 = EchoAgent(
        agent_t=lambda cfg: fake_agent, agent_config_t=tracking_config_t, store=store
    )
    await sut2.initialize(protocol_version=1)
    sut2.on_connect(conn=MagicMock(spec=Client))

    resp = await sut2.resume_session(cwd="/project", session_id=sid)

    assert resp.models.current_model_id == "gemini-2.5-pro"
    assert resp.modes.current_mode_id == "agent"
    assert sut2._sessions[sid].state.thinking_level == "high"

    assert len(configs_seen) >= 1
    rebuild_config = configs_seen[-1]
    assert rebuild_config.get("conversation_id") == "conv-abc-123"


async def test_offline_resume_session_not_found(tmp_path):
    """resume_session raises ValueError for unknown session_id."""
    store = SessionStore(path=tmp_path / "sessions.json")
    fake_agent = FakeAgent(config=None, responses=[])
    sut = EchoAgent(
        agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig, store=store
    )
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)
    sut.on_connect(conn=MagicMock(spec=Client))

    with pytest.raises(ValueError, match="Session not found"):
        await sut.resume_session(cwd="/project", session_id="nonexistent")


# --- Live tests (require GEMINI_API_KEY) ---


async def test_live_skill_magic_word():
    """E2E: /magic-word skill triggers agent to say 'vlak'."""
    cwd = str(Path(".").resolve())
    config = agy.LocalAgentConfig(
        capabilities=agy.types.CapabilitiesConfig(enabled_tools=[]),
        skills_paths=_skills_paths(cwd),
        workspaces=[cwd],
        gemini_config=agy.types.GeminiConfig(
            models=agy.types.ModelConfig(
                default=agy.types.ModelEntry(
                    name="gemini-3.1-flash-lite",
                    generation=agy.types.GenerationConfig(
                        thinking_level=agy.types.ThinkingLevel("minimal"),
                    ),
                ),
            ),
        ),
    )
    agent = agy.Agent(config)
    async with agent:
        response = await agent.chat(["/magic-word"])
        chunks = []
        async for chunk in response.chunks:
            if isinstance(chunk, agy.types.Text):
                chunks.append(chunk.text)
        combined = "".join(chunks).lower()
        assert "vlak" in combined, f"Expected 'vlak' in response, got: {combined[:200]}"


async def test_initializes():
    sut = EchoAgent(agent_t=agy.Agent, agent_config_t=agy.LocalAgentConfig)
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = unittest.mock.MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    prompt = TextContentBlock(
        type="text", text="Say hello three times. Do not use any tools."
    )
    reply = await sut.prompt([prompt], session_id=session.session_id)
    assert reply.stop_reason == "end_turn"

    client.session_update.assert_called()


# https://agentclientprotocol.github.io/python-sdk/
#
# https://agentclientprotocol.github.io/python-sdk/quickstart/
from acp import PROTOCOL_VERSION, spawn_agent_process, text_block


class SimpleClient(Client):
    async def request_permission(self, options, session_id, tool_call, **kwargs: Any):
        print("permission:", options, session_id, tool_call)
        return {"outcome": {"outcome": "cancelled"}}

    async def session_update(self, session_id, update, **kwargs):
        print("update:", session_id, update)


async def test_subprocess_fake_tool_call():
    """Full subprocess test: fake LLM issues view_file, RPC round-trips to client."""
    from acp.schema import ReadTextFileResponse

    script = Path("fake_server.py")
    env = os.environ.copy()
    received_updates = []

    class ToolTestClient(Client):
        async def read_text_file(self, path, session_id, **kwargs):
            return ReadTextFileResponse(content=f"content of {path}")

        async def session_update(self, session_id, update, **kwargs):
            received_updates.append(update)

        async def request_permission(self, options, session_id, tool_call, **kwargs):
            return {"outcome": {"optionId": "approve"}}

    async with spawn_agent_process(
        ToolTestClient(), sys.executable, str(script), env=env
    ) as (conn, _proc):
        await conn.initialize(protocol_version=PROTOCOL_VERSION)
        session = await conn.new_session(cwd=".", mcp_servers=[])
        await conn.prompt(
            session_id=session.session_id,
            prompt=[text_block("read a file")],
            message_id=str(uuid4()),
        )

    message_updates = [
        u
        for u in received_updates
        if getattr(u, "session_update", None) == "agent_message_chunk"
    ]
    assert len(message_updates) > 0
    combined = "".join(u.content.text for u in message_updates)
    assert "content of /tmp/fake_test_file.txt" in combined


async def test_live_run():
    script = Path("hellp.py")
    env = os.environ.copy()
    async with spawn_agent_process(
        SimpleClient(), sys.executable, str(script), env=env
    ) as (conn, _proc):
        try:
            await conn.initialize(protocol_version=PROTOCOL_VERSION)
        except Exception as e:
            print("initialize failed:", e, e.data if hasattr(e, "data") else "")
            print("subprocess returncode:", _proc.returncode)
            raise

        session = await conn.new_session(cwd=str(script.parent), mcp_servers=[])

        await conn.set_session_model(
            model_id="gemini-2.5-flash", session_id=session.session_id
        )

        await conn.prompt(
            session_id=session.session_id,
            prompt=[text_block("Say hello three times. Do not use any tools.")],
            message_id=str(uuid4()),
        )
