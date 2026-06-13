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
    """Minimal fake matching the agy.Agent interface, with hook dispatch for ToolCall/ToolResult."""

    def __init__(self, config, responses=None):
        self._responses = responses or []
        self._call_index = 0
        self._pre_hooks = []
        self._post_hooks = []

    def register_hook(self, hook):
        from google.antigravity.hooks.hooks import PreToolCallDecideHook, PostToolCallHook
        if isinstance(hook, PreToolCallDecideHook):
            self._pre_hooks.append(hook)
        elif isinstance(hook, PostToolCallHook):
            self._post_hooks.append(hook)

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

        pre_hooks = self._pre_hooks
        post_hooks = self._post_hooks

        async def stream():
            from google.antigravity.hooks.hooks import OperationContext, TurnContext, SessionContext
            pending_contexts: dict[str, OperationContext] = {}
            for c in chunks:
                if isinstance(c, agy_types.ToolCall):
                    op_ctx = OperationContext(TurnContext(SessionContext()))
                    if c.id:
                        pending_contexts[c.id] = op_ctx
                    for h in pre_hooks:
                        await h.run(op_ctx, c)
                    yield c
                elif isinstance(c, agy_types.ToolResult):
                    op_ctx = pending_contexts.pop(c.id, None) if c.id else None
                    if op_ctx is None:
                        op_ctx = OperationContext(TurnContext(SessionContext()))
                    for h in post_hooks:
                        await h.run(op_ctx, c)
                else:
                    yield c

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
    """ToolCalls in the stream are passed through (hooks handle start/complete in real agent)."""
    import hellp

    chunks = [
        agy_types.ToolCall(id="tc1", name="read_file", args={"path": "foo.py"}),
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
    message_updates = [u for u in updates if u.session_update == "agent_message_chunk"]
    assert len(message_updates) == 1
    assert message_updates[0].content.text == "Done."


async def test_offline_tool_execution_populates_edit_state():
    """Tool functions populate _last_file_edits so PostToolCallHook can send rich diffs."""
    import hellp
    from acp.schema import ReadTextFileResponse

    fake_agent = FakeAgent(config=None, responses=[])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = AsyncMock(spec=Client)
    client.read_text_file.return_value = ReadTextFileResponse(content="old content")
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id
    sut._active_session_id = sid

    result = await sut.edit_file("test.py", "old", "new")
    assert "Successfully edited" in result
    assert (sid, "test.py") in sut._last_file_edits
    assert sut._last_file_edits[(sid, "test.py")]["old_text"] == "old content"
    assert sut._last_file_edits[(sid, "test.py")]["new_text"] == "new content"


async def test_offline_edit_file_not_found():
    """edit_file returns error when old_string is not in the file."""
    import hellp
    from acp.schema import ReadTextFileResponse

    fake_agent = FakeAgent(config=None, responses=[])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = AsyncMock(spec=Client)
    client.read_text_file.return_value = ReadTextFileResponse(content="hello world")
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sut._active_session_id = session.session_id

    result = await sut.edit_file("test.py", "nonexistent", "replacement")
    assert "old_string not found" in result


async def test_offline_tool_works_without_contextvar():
    """Tool functions must work even when ContextVar is not set (SDK dispatches on background tasks)."""
    import hellp
    from acp.schema import ReadTextFileResponse

    fake_agent = FakeAgent(config=None, responses=[])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = AsyncMock(spec=Client)
    client.read_text_file.return_value = ReadTextFileResponse(content="hello world")
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sut._active_session_id = session.session_id

    # Call view_file WITHOUT setting current_session_id ContextVar — simulates SDK background task
    result = await sut.view_file("/tmp/test.txt")
    assert result == "hello world"
    assert "Error" not in result


async def test_offline_hook_tool_tracking():
    """Verify PreToolCallDecideHook + PostToolCallHook send start and completed updates."""
    import hellp

    sut = hellp.EchoAgent(
        agent_t=lambda cfg: FakeAgent(config=None, responses=[]),
        agent_config_t=FakeConfig,
    )
    await sut.initialize(protocol_version=1)

    client = AsyncMock(spec=Client)
    # Auto-approve permissions
    client.request_permission.return_value = MagicMock(
        outcome=MagicMock(option_id="approve")
    )
    sut.on_connect(conn=client)
    session = await sut.new_session(cwd=".")
    sid = session.session_id

    token = hellp.current_session_id.set(sid)
    try:
        from google.antigravity.hooks.hooks import OperationContext, TurnContext, SessionContext
        op_ctx = OperationContext(TurnContext(SessionContext()))

        tc = agy_types.ToolCall(id="tc1", name="view_file", args={"path": "hello.py"})
        pre_hook = hellp.MyPreToolCallDecideHook(sut)
        result = await pre_hook.run(op_ctx, tc)
        assert result.allow is True

        post_hook = hellp.MyPostToolCallHook(sut)
        tr = agy_types.ToolResult(id="tc1", name="view_file", result="file contents")
        await post_hook.run(op_ctx, tr)
    finally:
        hellp.current_session_id.reset(token)

    updates = [call.kwargs.get("update") or call.args[1] for call in client.session_update.call_args_list]
    tool_starts = [u for u in updates if u.session_update == "tool_call"]
    tool_progress = [u for u in updates if u.session_update == "tool_call_update"]
    assert len(tool_starts) == 1
    assert tool_starts[0].title == "view_file: hello.py"
    assert tool_starts[0].kind == "read"
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
    sut._session_titles[sid] = "test"

    await sut.close_session(session_id=sid)

    assert (sid, "/tmp/x") not in sut._last_file_edits
    assert sid not in sut._last_terminal_ids
    assert sid not in sut._session_titles


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
    client.request_permission.return_value = MagicMock(
        outcome=MagicMock(option_id="approve")
    )
    from acp.schema import ReadTextFileResponse, CreateTerminalResponse, TerminalOutputResponse, WaitForTerminalExitResponse
    client.read_text_file.return_value = ReadTextFileResponse(content="old contents")
    client.create_terminal.return_value = CreateTerminalResponse(terminal_id="term-123")
    client.wait_for_terminal_exit.return_value = WaitForTerminalExitResponse()
    client.terminal_output.return_value = TerminalOutputResponse(output="command output", truncated=False)

    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    # Pre-populate state that tool functions would set during execution
    sut._last_file_edits[(sid, "foo.txt")] = {"old_text": "old contents", "new_text": "new contents"}
    sut._last_terminal_ids[sid] = "term-123"

    await sut.prompt(
        [TextContentBlock(type="text", text="edit and run")],
        session_id=sid,
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


async def test_offline_session_modes():
    """Verify session modes are declared and set_session_mode works."""
    import hellp

    fake_agent = FakeAgent(config=None, responses=[
        [agy_types.Text(step_index=0, text="plan step 1")],
    ])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    assert session.modes is not None
    assert session.modes.current_mode_id == "agent"
    assert len(session.modes.available_modes) == 2

    await sut.set_session_mode(mode_id="plan", session_id=sid)

    updates = [call.kwargs.get("update") or call.args[1] for call in client.session_update.call_args_list]
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
    import hellp

    fake_agent = FakeAgent(config=None, responses=[])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    # Config options include model and thinking_level
    assert any(opt.id == "model" for opt in session.config_options)
    assert any(opt.id == "thinking_level" for opt in session.config_options)

    resp = await sut.set_config_option(config_id="model", session_id=sid, value="gemini-2.5-flash")
    assert sut._session_models[sid] == "gemini-2.5-flash"

    model_opt = next(o for o in resp.config_options if o.id == "model")
    assert model_opt.current_value == "gemini-2.5-flash"

    # Set thinking level
    resp2 = await sut.set_config_option(config_id="thinking_level", session_id=sid, value="high")
    assert sut._session_thinking_levels[sid] == "high"

    thinking_opt = next(o for o in resp2.config_options if o.id == "thinking_level")
    assert thinking_opt.current_value == "high"


async def test_offline_session_persistence(tmp_path):
    """new_session → prompt → list_sessions finds it → close_session → list_sessions doesn't."""
    import hellp

    store = hellp.SessionStore(path=tmp_path / "sessions.json")
    chunks = [agy_types.Text(step_index=0, text="hi")]
    fake_agent = FakeAgent(config=None, responses=[chunks])

    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig, store=store)
    await sut.initialize(protocol_version=1)

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
    import hellp

    store = hellp.SessionStore(path=tmp_path / "sessions.json")
    chunks = [agy_types.Text(step_index=0, text="response")]
    fake_agent = FakeAgent(config=None, responses=[chunks, [agy_types.Text(step_index=0, text="resumed")]])

    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig, store=store)
    await sut.initialize(protocol_version=1)

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
    assert sut._session_modes[sid] == "plan"


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


async def test_offline_auth_methods_declared():
    """InitializeResponse advertises GEMINI_API_KEY as env var auth method."""
    import hellp

    fake_agent = FakeAgent(config=None, responses=[])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    resp = await sut.initialize(protocol_version=1)

    assert resp.auth_methods is not None
    assert len(resp.auth_methods) == 1
    method = resp.auth_methods[0]
    assert method.type == "env_var"
    assert method.id == "gemini_api_key"
    assert len(method.vars) == 1
    assert method.vars[0].name == "GEMINI_API_KEY"


async def test_offline_authenticate():
    """authenticate() returns AuthenticateResponse without error."""
    import hellp

    fake_agent = FakeAgent(config=None, responses=[])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)
    resp = await sut.authenticate(method_id="gemini_api_key")
    assert resp is not None


async def test_offline_model_switching():
    """set_session_model changes the model for a session."""
    import hellp

    fake_agent = FakeAgent(config=None, responses=[])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    # new_session returns model state
    assert session.models is not None
    assert session.models.current_model_id == "gemini-3.5-flash"
    assert len(session.models.available_models) == 7

    # Switch model — agent should be rebuilt
    configs_seen = []
    original_config_t = FakeConfig
    def tracking_config_t(**kwargs):
        configs_seen.append(kwargs)
        return original_config_t(**kwargs)
    sut._agent_config_t = tracking_config_t

    resp = await sut.set_session_model(model_id="gemini-2.5-flash", session_id=sid)
    assert resp is not None
    assert sut._session_models[sid] == "gemini-2.5-flash"
    assert len(configs_seen) == 1
    gemini_cfg = configs_seen[0]["gemini_config"]
    assert gemini_cfg.models.default.name == "gemini-2.5-flash"


async def test_offline_rebuild_passes_thinking_level():
    """Changing thinking level via config option rebuilds the agent with correct config."""
    import hellp

    configs_seen = []
    original_config_t = FakeConfig
    def tracking_config_t(**kwargs):
        configs_seen.append(kwargs)
        return original_config_t(**kwargs)

    fake_agent = FakeAgent(config=None, responses=[])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)
    sut._agent_config_t = tracking_config_t

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    await sut.set_config_option(config_id="thinking_level", session_id=sid, value="high")
    assert len(configs_seen) == 1
    gemini_cfg = configs_seen[0]["gemini_config"]
    assert gemini_cfg.models.default.generation.thinking_level.value == "high"


async def test_offline_rebuild_uses_valid_local_agent_config():
    """_rebuild_agent must produce a valid LocalAgentConfig (no conflicting model fields)."""
    import hellp

    fake_agent = FakeAgent(config=None, responses=[])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=agy.LocalAgentConfig)
    await sut.initialize(protocol_version=1)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    # This would raise "Cannot set both 'model' shorthand and
    # 'gemini_config.models.default'" if _rebuild_agent passes both.
    await sut.set_session_model(model_id="gemini-2.5-flash", session_id=sid)


async def test_offline_model_persisted_in_session(tmp_path):
    """Model choice is persisted and restored on load_session."""
    import hellp

    store = hellp.SessionStore(path=tmp_path / "sessions.json")
    chunks = [agy_types.Text(step_index=0, text="ok")]
    fake_agent = FakeAgent(config=None, responses=[chunks])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig, store=store)
    await sut.initialize(protocol_version=1)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd="/tmp")
    sid = session.session_id
    await sut.set_session_model(model_id="gemini-2.5-flash-lite", session_id=sid)

    # Trigger save via prompt
    await sut.prompt([TextContentBlock(type="text", text="hi")], session_id=sid)

    stored = store.load(sid)
    assert stored["model"] == "gemini-2.5-flash-lite"

    # Load into fresh agent
    sut2 = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig, store=store)
    await sut2.initialize(protocol_version=1)
    sut2.on_connect(conn=MagicMock(spec=Client))

    loaded = await sut2.load_session(cwd="/tmp", session_id=sid)
    assert loaded.models is not None
    assert loaded.models.current_model_id == "gemini-2.5-flash-lite"


async def test_offline_close_session_cleans_model():
    """close_session removes model state."""
    import hellp

    fake_agent = FakeAgent(config=None, responses=[])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id
    assert sid in sut._session_models

    await sut.close_session(session_id=sid)
    assert sid not in sut._session_models


async def test_offline_reset_command():
    """/reset command rebuilds agent and clears session title."""
    import hellp

    chunks1 = [agy_types.Text(step_index=0, text="first response")]
    chunks2 = [agy_types.Text(step_index=0, text="after reset")]
    fake_agent = FakeAgent(config=None, responses=[chunks1, chunks2])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    await sut.initialize(protocol_version=1)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd=".")
    sid = session.session_id

    # First prompt sets title
    await sut.prompt([TextContentBlock(type="text", text="hello")], session_id=sid)
    assert sid in sut._session_titles

    # /reset clears title and rebuilds agent
    client.reset_mock()
    reply = await sut.prompt([TextContentBlock(type="text", text="/reset")], session_id=sid)
    assert reply.stop_reason == "end_turn"
    assert sid not in sut._session_titles

    updates = [call.kwargs.get("update") or call.args[1] for call in client.session_update.call_args_list]
    message_updates = [u for u in updates if u.session_update == "agent_message_chunk"]
    assert any("reset" in u.content.text.lower() for u in message_updates)


async def test_offline_fork_session(tmp_path):
    """fork_session creates a new session copying settings from the original."""
    import hellp

    store = hellp.SessionStore(path=tmp_path / "sessions.json")
    fake_agent = FakeAgent(config=None, responses=[])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig, store=store)
    await sut.initialize(protocol_version=1)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    session = await sut.new_session(cwd="/project")
    sid = session.session_id
    sut._session_titles[sid] = "My Session"
    await sut.set_config_option(config_id="model", session_id=sid, value="gemini-2.5-pro")
    await sut.set_config_option(config_id="thinking_level", session_id=sid, value="high")

    forked = await sut.fork_session(cwd="/project", session_id=sid)
    fid = forked.session_id

    assert fid != sid
    assert sut._session_modes[fid] == "agent"
    assert sut._session_models[fid] == "gemini-2.5-pro"
    assert sut._session_thinking_levels[fid] == "high"
    assert sut._session_titles[fid] == "My Session (fork)"
    assert forked.models is not None
    assert forked.models.current_model_id == "gemini-2.5-pro"

    # Forked session is persisted
    stored = store.load(fid)
    assert stored is not None
    assert stored["model"] == "gemini-2.5-pro"


async def test_offline_fork_capability_declared():
    """InitializeResponse declares fork capability."""
    import hellp

    fake_agent = FakeAgent(config=None, responses=[])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    resp = await sut.initialize(protocol_version=1)

    assert resp.agent_capabilities.session_capabilities.fork is not None
    assert resp.agent_capabilities.session_capabilities.resume is not None


async def test_offline_agent_info_declared():
    """InitializeResponse includes agent name, version, and title."""
    import hellp

    fake_agent = FakeAgent(config=None, responses=[])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig)
    resp = await sut.initialize(protocol_version=1)

    assert resp.agent_info is not None
    assert resp.agent_info.name == "agy-acp"
    assert resp.agent_info.version == "0.1.0"
    assert resp.agent_info.title == "Antigravity ACP Adapter"


async def test_offline_resume_session(tmp_path):
    """resume_session restores mode, model, thinking level, and rebuilds agent with conversation_id."""
    import hellp

    store = hellp.SessionStore(path=tmp_path / "sessions.json")
    chunks = [agy_types.Text(step_index=0, text="hello")]
    fake_agent = FakeAgent(config=None, responses=[chunks])

    configs_seen = []
    original_config_t = FakeConfig
    def tracking_config_t(**kwargs):
        configs_seen.append(kwargs)
        return original_config_t(**kwargs)

    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig, store=store)
    await sut.initialize(protocol_version=1)

    client = MagicMock(spec=Client)
    sut.on_connect(conn=client)

    # Create a session, prompt to generate conversation_id, then save
    session = await sut.new_session(cwd="/project")
    sid = session.session_id
    await sut.set_config_option(config_id="model", session_id=sid, value="gemini-2.5-pro")
    await sut.set_config_option(config_id="thinking_level", session_id=sid, value="high")

    # Simulate a conversation_id being set (normally set by Go harness)
    sut._agent.conversation_id_value = "conv-abc-123"
    # Patch the agent to return a conversation_id
    type(sut._agent).conversation_id = property(lambda self: getattr(self, 'conversation_id_value', None))

    await sut.prompt([TextContentBlock(type="text", text="hi")], session_id=sid)

    stored = store.load(sid)
    assert stored["conversation_id"] == "conv-abc-123"

    # Resume in a fresh agent instance
    sut2 = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=tracking_config_t, store=store)
    await sut2.initialize(protocol_version=1)
    sut2.on_connect(conn=MagicMock(spec=Client))

    resp = await sut2.resume_session(cwd="/project", session_id=sid)

    assert resp.models.current_model_id == "gemini-2.5-pro"
    assert resp.modes.current_mode_id == "agent"
    assert sut2._session_thinking_levels[sid] == "high"

    # _rebuild_agent was called with conversation_id
    assert len(configs_seen) >= 1
    rebuild_config = configs_seen[-1]
    assert rebuild_config.get("conversation_id") == "conv-abc-123"


async def test_offline_resume_session_not_found(tmp_path):
    """resume_session raises ValueError for unknown session_id."""
    import hellp

    store = hellp.SessionStore(path=tmp_path / "sessions.json")
    fake_agent = FakeAgent(config=None, responses=[])
    sut = hellp.EchoAgent(agent_t=lambda cfg: fake_agent, agent_config_t=FakeConfig, store=store)
    await sut.initialize(protocol_version=1)
    sut.on_connect(conn=MagicMock(spec=Client))

    with pytest.raises(ValueError, match="Session not found"):
        await sut.resume_session(cwd="/project", session_id="nonexistent")


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

    async with spawn_agent_process(ToolTestClient(), sys.executable, str(script), env=env) as (conn, _proc):
        await conn.initialize(protocol_version=PROTOCOL_VERSION)
        session = await conn.new_session(cwd=".", mcp_servers=[])
        await conn.prompt(
            session_id=session.session_id,
            prompt=[text_block("read a file")],
            message_id=str(uuid4()),
        )

    message_updates = [u for u in received_updates if getattr(u, "session_update", None) == "agent_message_chunk"]
    assert len(message_updates) > 0
    combined = "".join(u.content.text for u in message_updates)
    assert "content of /tmp/fake_test_file.txt" in combined


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

        await conn.set_session_model(
            model_id="gemini-2.5-flash", session_id=session.session_id
        )

        await conn.prompt(
            session_id=session.session_id,
            prompt=[text_block("Say hello three times. Do not use any tools.")],
            message_id=str(uuid4()),
        )
