import asyncio
from unittest.mock import AsyncMock, MagicMock

import google.antigravity as agy
from acp.interfaces import Client

from agy_acp.agent import EchoAgent
from agy_acp.hooks import MyPreToolCallDecideHook, MyPostToolCallHook
from agy_acp.session import current_session_id
from conftest import FakeAgent, FakeConfig, _TEST_CLIENT_CAPS


async def _make_hook_sut():
    """Create a ready-to-use (sut, sid, pre_hook, client) tuple for mode tests."""
    sut = EchoAgent(
        agent_t=lambda cfg: FakeAgent(config=None, responses=[]),
        agent_config_t=FakeConfig,
    )
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)
    client = AsyncMock(spec=Client)
    client.request_permission.return_value = MagicMock(
        outcome=MagicMock(option_id="approve")
    )
    sut.on_connect(conn=client)
    session = await sut.new_session(cwd=".")
    sid = session.session_id
    pre_hook = MyPreToolCallDecideHook(sut)
    return sut, sid, pre_hook, client


async def _run_hook(pre_hook, tool_name, args=None):
    """Run the pre-tool-call hook and return the HookResult."""
    from google.antigravity.hooks.hooks import (
        OperationContext,
        SessionContext,
        TurnContext,
    )

    op_ctx = OperationContext(TurnContext(SessionContext()))
    tc = agy.types.ToolCall(id=f"tc-{tool_name}", name=tool_name, args=args or {})
    return await pre_hook.run(op_ctx, tc)


async def test_offline_hook_tool_tracking():
    """Verify PreToolCallDecideHook + PostToolCallHook send start and completed updates.
    File tools (view_file) auto-allow without requesting permission."""
    sut = EchoAgent(
        agent_t=lambda cfg: FakeAgent(config=None, responses=[]),
        agent_config_t=FakeConfig,
    )
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = AsyncMock(spec=Client)
    sut.on_connect(conn=client)
    session = await sut.new_session(cwd=".")
    sid = session.session_id

    token = current_session_id.set(sid)
    try:
        from google.antigravity.hooks.hooks import (
            OperationContext,
            SessionContext,
            TurnContext,
        )

        op_ctx = OperationContext(TurnContext(SessionContext()))

        tc = agy.types.ToolCall(id="tc1", name="view_file", args={"path": "hello.py"})
        pre_hook = MyPreToolCallDecideHook(sut)
        result = await pre_hook.run(op_ctx, tc)
        assert result.allow is True

        post_hook = MyPostToolCallHook(sut)
        tr = agy.types.ToolResult(id="tc1", name="view_file", result="file contents")
        await post_hook.run(op_ctx, tr)
    finally:
        current_session_id.reset(token)

    client.request_permission.assert_not_called()

    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]
    tool_starts = [u for u in updates if u.session_update == "tool_call"]
    tool_progress = [u for u in updates if u.session_update == "tool_call_update"]
    assert len(tool_starts) == 1
    assert tool_starts[0].title == "view_file: hello.py"
    assert tool_starts[0].kind == "read"
    assert len(tool_progress) == 1
    assert tool_progress[0].status == "completed"


async def test_offline_hook_run_command_requires_permission():
    """run_command tool calls require IDE permission approval."""
    sut = EchoAgent(
        agent_t=lambda cfg: FakeAgent(config=None, responses=[]),
        agent_config_t=FakeConfig,
    )
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = AsyncMock(spec=Client)
    client.request_permission.return_value = MagicMock(
        outcome=MagicMock(option_id="approve")
    )
    sut.on_connect(conn=client)
    session = await sut.new_session(cwd=".")
    sid = session.session_id

    token = current_session_id.set(sid)
    try:
        from google.antigravity.hooks.hooks import (
            OperationContext,
            SessionContext,
            TurnContext,
        )

        op_ctx = OperationContext(TurnContext(SessionContext()))

        tc = agy.types.ToolCall(id="tc2", name="run_command", args={"command": "ls"})
        pre_hook = MyPreToolCallDecideHook(sut)
        result = await pre_hook.run(op_ctx, tc)
        assert result.allow is True
    finally:
        current_session_id.reset(token)

    client.request_permission.assert_called_once()


async def test_offline_hook_run_command_denied():
    """Denied run_command returns clear message without sending a start notification."""
    sut = EchoAgent(
        agent_t=lambda cfg: FakeAgent(config=None, responses=[]),
        agent_config_t=FakeConfig,
    )
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = AsyncMock(spec=Client)
    client.request_permission.return_value = MagicMock(
        outcome=MagicMock(option_id="cancelled")
    )
    sut.on_connect(conn=client)
    session = await sut.new_session(cwd=".")
    sid = session.session_id

    token = current_session_id.set(sid)
    try:
        from google.antigravity.hooks.hooks import (
            OperationContext,
            SessionContext,
            TurnContext,
        )

        op_ctx = OperationContext(TurnContext(SessionContext()))

        tc = agy.types.ToolCall(
            id="tc3", name="run_command", args={"command": "rm -rf /"}
        )
        pre_hook = MyPreToolCallDecideHook(sut)
        result = await pre_hook.run(op_ctx, tc)
        assert result.allow is False
        assert "declined" in result.message
    finally:
        current_session_id.reset(token)

    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]
    tool_starts = [u for u in updates if u.session_update == "tool_call"]
    assert len(tool_starts) == 0


async def test_offline_hook_mcp_tool_requires_permission():
    """MCP server tools (e.g. mcp_idea_execute_tool) require permission — whitelist only allows known file tools."""
    sut = EchoAgent(
        agent_t=lambda cfg: FakeAgent(config=None, responses=[]),
        agent_config_t=FakeConfig,
    )
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = AsyncMock(spec=Client)
    client.request_permission.return_value = MagicMock(
        outcome=MagicMock(option_id="approve")
    )
    sut.on_connect(conn=client)
    session = await sut.new_session(cwd=".")
    sid = session.session_id

    token = current_session_id.set(sid)
    try:
        from google.antigravity.hooks.hooks import (
            OperationContext,
            SessionContext,
            TurnContext,
        )

        op_ctx = OperationContext(TurnContext(SessionContext()))

        tc = agy.types.ToolCall(
            id="tc-mcp", name="mcp_idea_execute_tool", args={"command": "echo hi"}
        )
        pre_hook = MyPreToolCallDecideHook(sut)
        result = await pre_hook.run(op_ctx, tc)
        assert result.allow is True
    finally:
        current_session_id.reset(token)

    client.request_permission.assert_called_once()


async def test_offline_hook_sdk_builtin_tools_auto_allow():
    """SDK built-in tools like ask_question and finish auto-allow without permission."""
    sut = EchoAgent(
        agent_t=lambda cfg: FakeAgent(config=None, responses=[]),
        agent_config_t=FakeConfig,
    )
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = AsyncMock(spec=Client)
    sut.on_connect(conn=client)
    session = await sut.new_session(cwd=".")
    sid = session.session_id

    token = current_session_id.set(sid)
    try:
        from google.antigravity.hooks.hooks import (
            OperationContext,
            SessionContext,
            TurnContext,
        )

        pre_hook = MyPreToolCallDecideHook(sut)

        for tool_name in ("ask_question", "finish", "start_subagent", "generate_image"):
            op_ctx = OperationContext(TurnContext(SessionContext()))
            tc = agy.types.ToolCall(id=f"tc-{tool_name}", name=tool_name, args={})
            result = await pre_hook.run(op_ctx, tc)
            assert result.allow is True, f"{tool_name} should auto-allow"
    finally:
        current_session_id.reset(token)

    client.request_permission.assert_not_called()


async def test_offline_nonzero_exit_code_marks_failed():
    """Non-zero exit code from run_command sets tool call status to 'failed'."""
    sut = EchoAgent(
        agent_t=lambda cfg: FakeAgent(config=None, responses=[]),
        agent_config_t=FakeConfig,
    )
    await sut.initialize(protocol_version=1, client_capabilities=_TEST_CLIENT_CAPS)

    client = AsyncMock(spec=Client)
    sut.on_connect(conn=client)
    session = await sut.new_session(cwd=".")
    sid = session.session_id

    sut._sessions[sid].last_exit_code = 1
    sut._sessions[sid].last_terminal_id = "term-fail"

    sut._tracker.start("tc-fail", title="run_command: false", kind="execute")

    token = current_session_id.set(sid)
    try:
        from google.antigravity.hooks.hooks import (
            OperationContext,
            SessionContext,
            TurnContext,
        )

        op_ctx = OperationContext(TurnContext(SessionContext()))
        op_ctx.set("acp_tc_id", "tc-fail")

        post_hook = MyPostToolCallHook(sut)
        tr = agy.types.ToolResult(id="tc-fail", name="run_command", result="")
        await post_hook.run(op_ctx, tr)
    finally:
        current_session_id.reset(token)

    updates = [
        call.kwargs.get("update") or call.args[1]
        for call in client.session_update.call_args_list
    ]
    progress = [u for u in updates if u.session_update == "tool_call_update"]
    assert len(progress) == 1
    assert progress[0].status == "failed"
    assert progress[0].raw_output == {"exit_code": 1, "output": ""}


async def test_mode_agent_prompts_for_file_writes():
    """In agent mode, create_file and edit_file trigger permission prompt."""
    sut, sid, hook, client = await _make_hook_sut()
    token = current_session_id.set(sid)
    try:
        for tool in ("create_file", "edit_file"):
            result = await _run_hook(hook, tool, {"path": "/tmp/x"})
            assert result.allow is True, f"{tool} should be allowed after approval"
        assert client.request_permission.call_count == 2
    finally:
        current_session_id.reset(token)


async def test_mode_accept_edits_allows_file_writes():
    """In accept_edits mode, file writes auto-allow without prompting."""
    sut, sid, hook, client = await _make_hook_sut()
    sut._sessions[sid].state.mode = "accept_edits"
    token = current_session_id.set(sid)
    try:
        for tool in ("create_file", "edit_file"):
            result = await _run_hook(hook, tool, {"path": "/tmp/x"})
            assert result.allow is True, f"{tool} should auto-allow in accept_edits"
        client.request_permission.assert_not_called()
    finally:
        current_session_id.reset(token)


async def test_mode_plan_denies_file_writes():
    """In plan mode, file write tools are denied."""
    sut, sid, hook, _ = await _make_hook_sut()
    sut._sessions[sid].state.mode = "plan"
    token = current_session_id.set(sid)
    try:
        for tool in ("create_file", "edit_file"):
            result = await _run_hook(hook, tool, {"path": "/tmp/x"})
            assert result.allow is False, f"{tool} should be denied in plan mode"
    finally:
        current_session_id.reset(token)


async def test_mode_plan_allows_reads_and_prompts_commands():
    """In plan mode, read tools auto-allow and run_command prompts via broker."""
    sut, sid, hook, client = await _make_hook_sut()
    sut._sessions[sid].state.mode = "plan"
    token = current_session_id.set(sid)
    try:
        result = await _run_hook(hook, "view_file", {"path": "/tmp/x"})
        assert result.allow is True, "view_file should auto-allow in plan mode"
        result = await _run_hook(hook, "list_directory", {"directory": "/tmp"})
        assert result.allow is True, "list_directory should auto-allow in plan mode"
        result = await _run_hook(hook, "run_command", {"command": "ls"})
        assert result.allow is True, "run_command should prompt and allow in plan mode"
        assert client.request_permission.call_count == 1
    finally:
        current_session_id.reset(token)


async def test_mode_dont_ask_denies_non_safe():
    """In dont_ask mode, non-safe tools are denied without prompting."""
    sut, sid, hook, client = await _make_hook_sut()
    sut._sessions[sid].state.mode = "dont_ask"
    token = current_session_id.set(sid)
    try:
        result = await _run_hook(hook, "view_file", {"path": "/tmp/x"})
        assert result.allow is True, "view_file should auto-allow"
        for tool in ("create_file", "edit_file", "run_command"):
            result = await _run_hook(hook, tool)
            assert result.allow is False, f"{tool} should be denied in dont_ask"
        client.request_permission.assert_not_called()
    finally:
        current_session_id.reset(token)


async def test_mode_bypass_allows_everything():
    """In bypass mode, all tools auto-allow without prompting."""
    sut, sid, hook, client = await _make_hook_sut()
    sut._sessions[sid].state.mode = "bypass"
    token = current_session_id.set(sid)
    try:
        for tool in ("create_file", "edit_file", "run_command", "view_file"):
            result = await _run_hook(hook, tool)
            assert result.allow is True, f"{tool} should auto-allow in bypass"
        client.request_permission.assert_not_called()
    finally:
        current_session_id.reset(token)
