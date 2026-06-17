from uuid import uuid4

import google.antigravity as agy
from acp import text_block
from acp.contrib.permissions import PermissionBroker
from acp.helpers import tool_content, tool_diff_content, tool_terminal_ref
from acp.schema import (
    RequestPermissionRequest,
    RequestPermissionResponse,
    ToolCallLocation,
    ToolCallUpdate,
)
from google.antigravity.hooks.hooks import (
    HookContext,
    PostToolCallHook,
    PreToolCallDecideHook,
)
from google.antigravity.types import HookResult

from agy_acp.config import _ALWAYS_SAFE_TOOLS, _FILE_WRITE_TOOLS
from agy_acp.log import log
from agy_acp.session import current_session_id
from agy_acp.tool_ui import _permission_content, _permission_description, _tool_kind, _tool_title


class MyPreToolCallDecideHook(PreToolCallDecideHook):
    """Intercepts tool calls: sends ACP start notification and requests permission."""

    def __init__(self, echo_agent):
        self.echo_agent = echo_agent

    async def run(self, context: HookContext, data: agy.types.ToolCall) -> HookResult:
        session_id = current_session_id.get(None) or self.echo_agent._active_session_id
        if not session_id:
            log.warning("No session ID found in context for tool call %s — denying", data.name)
            return HookResult(allow=False, message="No active session context")

        tool_call_id = data.id or uuid4().hex
        kind = _tool_kind(str(data.name))
        locations = None
        if isinstance(data.args, dict) and "path" in data.args:
            locations = [ToolCallLocation(path=data.args["path"])]
        title = _tool_title(str(data.name), data.args)

        context.set("acp_tc_id", tool_call_id)
        session = self.echo_agent._sessions.get(session_id)
        if not session:
            return HookResult(allow=False, message="Session no longer active")
        mode = session.state.mode
        tool_name = str(data.name)
        log.debug("Intercepted tool call %s in session %s (mode=%s)", data.name, session_id, mode)

        if tool_name in _ALWAYS_SAFE_TOOLS:
            await self._send_start(session_id, tool_call_id, title, kind, locations, data.args)
            return HookResult(allow=True)

        is_file_write = tool_name in _FILE_WRITE_TOOLS

        if mode == "bypass":
            await self._send_start(session_id, tool_call_id, title, kind, locations, data.args)
            return HookResult(allow=True)

        if mode == "plan" and is_file_write:
            return HookResult(
                allow=False,
                message="Plan mode is active — file writes are disabled. Describe what you would do instead.",
            )

        if mode == "dont_ask":
            return HookResult(
                allow=False,
                message=(
                    "This tool requires permission but the current mode denies unapproved tools."
                    " Suggest the user switch to Agent or Bypass mode."
                ),
            )

        if mode == "accept_edits" and is_file_write:
            await self._send_start(session_id, tool_call_id, title, kind, locations, data.args)
            return HookResult(allow=True)

        async def requester(
            request: RequestPermissionRequest,
        ) -> RequestPermissionResponse:
            return await self.echo_agent._conn.request_permission(
                options=request.options,
                session_id=request.session_id,
                tool_call=request.tool_call,
            )

        tool_call = ToolCallUpdate(
            tool_call_id=tool_call_id,
            title=title,
            kind=kind,
            raw_input=data.args,
        )

        broker = PermissionBroker(
            session_id=session_id,
            requester=requester,
        )

        content = _permission_content(tool_name, data.args)

        try:
            resp = await broker.request_for(
                external_id=tool_call_id,
                tool_call=tool_call,
                content=content,
                description=_permission_description(str(data.name), data.args),
            )
            outcome = resp.outcome
            if outcome is None:
                return HookResult(
                    allow=False,
                    message="The user declined this command. Ask what they'd like instead.",
                )

            if isinstance(outcome, dict):
                option_id = outcome.get("optionId") or outcome.get("option_id")
            else:
                option_id = getattr(outcome, "option_id", None)

            if option_id in ("approve", "approve_for_session"):
                log.debug("Tool call %s permitted", data.name)
                self.echo_agent._tracker.start(
                    tool_call_id,
                    title=title,
                    kind=kind,
                    locations=locations,
                    raw_input=data.args,
                )
                return HookResult(allow=True)
            else:
                log.debug("Tool call %s rejected/cancelled", data.name)
                return HookResult(
                    allow=False,
                    message="The user declined this command. Ask what they'd like instead.",
                )
        except Exception as e:
            log.exception("Error requesting permission via broker")
            return HookResult(allow=False, message=f"Internal permission broker error: {e}")

    async def _send_start(self, session_id, tool_call_id, title, kind, locations, raw_input):
        start = self.echo_agent._tracker.start(
            tool_call_id,
            title=title,
            kind=kind,
            locations=locations,
            raw_input=raw_input,
        )
        await self.echo_agent._conn.session_update(session_id=session_id, update=start)


class MyPostToolCallHook(PostToolCallHook):
    """Sends ACP tool_call_update with status=completed after tool execution."""

    def __init__(self, echo_agent):
        self.echo_agent = echo_agent

    async def run(self, context: HookContext, data: agy.types.ToolResult) -> None:
        session_id = current_session_id.get(None) or self.echo_agent._active_session_id
        tc_id = context.get("acp_tc_id")
        if not session_id or not tc_id:
            return

        status = "failed" if data.error else "completed"
        summary = str(data.error or data.result)[:2000]
        content = [tool_content(text_block(summary))]
        raw_output = data.error or data.result

        try:
            session = self.echo_agent._sessions.get(session_id)
            view = self.echo_agent._tracker.view(tc_id)
            if view.kind == "edit" and isinstance(view.raw_input, dict):
                path = view.raw_input.get("path")
                edit_info = session.last_file_edits.pop(path, None) if session else None
                if edit_info:
                    content = [
                        tool_diff_content(
                            path=path,
                            new_text=edit_info["new_text"],
                            old_text=edit_info["old_text"],
                        )
                    ]
            elif view.kind == "execute" and session:
                exit_code = session.last_exit_code
                session.last_exit_code = None
                if exit_code is not None and exit_code != 0:
                    status = "failed"
                    raw_output = {"exit_code": exit_code, "output": data.result}
                terminal_id = session.last_terminal_id
                session.last_terminal_id = None
                if terminal_id:
                    content = [tool_terminal_ref(terminal_id=terminal_id)]
        except KeyError:
            pass

        try:
            progress = self.echo_agent._tracker.progress(tc_id, status=status, content=content, raw_output=raw_output)
            await self.echo_agent._conn.session_update(session_id=session_id, update=progress)
        except KeyError:
            log.debug("post hook: unknown tracker id %s", tc_id)
