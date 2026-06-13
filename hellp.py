import asyncio
import base64
import logging
import re
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import google.antigravity as agy
from google.antigravity.hooks.hooks import PreToolCallDecideHook, HookContext
from google.antigravity.types import HookResult

log = logging.getLogger(__name__)
log.addHandler(logging.FileHandler("file.log"))
log.setLevel(logging.DEBUG)

from acp import (
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    text_block,
    update_agent_message, update_agent_thought_text,
)
from acp.helpers import (
    update_available_commands,
    tool_diff_content,
    tool_terminal_ref,
    update_plan,
    plan_entry,
)
from acp.contrib.permissions import PermissionBroker
from acp.contrib.tool_calls import ToolCallTracker
from acp.interfaces import Client
from acp.schema import (
    AudioContentBlock,
    BlobResourceContents,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    McpServerStdio,
    ResourceContentBlock,
    SseMcpServer,
    TextContentBlock,
    TextResourceContents,
    AgentCapabilities, AvailableCommand, CloseSessionResponse,
    PromptCapabilities, SessionInfoUpdate, Usage,
    SessionConfigOptionSelect,
    SessionConfigSelectGroup, SessionConfigSelectOption,
    SessionCapabilities, SessionCloseCapabilities,
    RequestPermissionRequest, RequestPermissionResponse,
    UsageUpdate, ToolCallUpdate, ToolCallLocation,
)

current_session_id = ContextVar("current_session_id")

_PLAN_LINE_RE = re.compile(
    r"^\s*(?:"
    r"[-*]\s+\[([xX /\-])\]\s+(.*)"   # - [x] item  or  * [ ] item
    r"|[-*]\s+(.*)"                      # - item  or  * item
    r"|(\d+)\.\s+(.*)"                   # 1. item  or  23. item
    r")$"
)


def _parse_plan_entries(lines: list[str]) -> list:
    entries = []
    for line in lines:
        m = _PLAN_LINE_RE.match(line)
        if not m:
            continue
        if m.group(1) is not None:
            marker, content = m.group(1), m.group(2)
            if marker in ("x", "X"):
                status = "completed"
            elif marker in ("-", "/"):
                status = "in_progress"
            else:
                status = "pending"
        elif m.group(3) is not None:
            content, status = m.group(3), "pending"
        else:
            content, status = m.group(5), "pending"
        content = content.strip()
        if content:
            entries.append(plan_entry(content=content, status=status))
    return entries


class MyPreToolCallDecideHook(PreToolCallDecideHook):
    """Safety decision hook requesting approval from the JetBrains IDE / client via ACP."""

    def __init__(self, echo_agent: "EchoAgent"):
        self.echo_agent = echo_agent

    async def run(self, context: HookContext, data: agy.types.ToolCall) -> HookResult:
        tool_name = str(data.name).lower()
        kind = "other"
        if "read" in tool_name or "view" in tool_name:
            kind = "read"
        elif "write" in tool_name or "edit" in tool_name or "replace" in tool_name:
            kind = "edit"
        elif "delete" in tool_name or "remove" in tool_name:
            kind = "delete"
        elif "move" in tool_name or "rename" in tool_name:
            kind = "move"
        elif "find" in tool_name or "search" in tool_name or "grep" in tool_name:
            kind = "search"
        elif "execute" in tool_name or "run" in tool_name:
            kind = "execute"
        elif "think" in tool_name:
            kind = "think"
        elif "fetch" in tool_name or "download" in tool_name:
            kind = "fetch"

        session_id = current_session_id.get(None)
        if not session_id:
            log.warning("No session ID found in context for tool call %s", data.name)
            return HookResult(allow=True)

        log.debug("Intercepted tool call %s in session %s", data.name, session_id)

        # Build custom permission requester tied to the client connection
        async def requester(request: RequestPermissionRequest) -> RequestPermissionResponse:
            return await self.echo_agent._conn.request_permission(
                options=request.options,
                session_id=request.session_id,
                tool_call=request.tool_call
            )

        tool_call_id = data.id or uuid4().hex
        tool_call = ToolCallUpdate(
            tool_call_id=tool_call_id,
            title=f"Call tool {data.name}",
            kind=kind,
            raw_input=data.args,
        )

        broker = PermissionBroker(
            session_id=session_id,
            requester=requester,
        )

        try:
            resp = await broker.request_for(
                external_id=tool_call_id,
                tool_call=tool_call,
                description=f"Approve running {data.name} with arguments {data.args}?",
            )
            outcome = resp.outcome
            if outcome is None:
                return HookResult(allow=False, message="Permission rejected (no outcome)")

            if isinstance(outcome, dict):
                option_id = outcome.get("optionId") or outcome.get("option_id")
            else:
                option_id = getattr(outcome, "option_id", None)

            if option_id in ("approve", "approve_for_session"):
                log.debug("Tool call %s permitted", data.name)
                return HookResult(allow=True)
            else:
                log.debug("Tool call %s rejected/cancelled", data.name)
                return HookResult(allow=False, message="Permission rejected by user")
        except Exception as e:
            log.exception("Error requesting permission via broker")
            return HookResult(allow=False, message=f"Internal permission broker error: {e}")


class EchoAgent(Agent):
    _conn: Client
    _agent: agy.Agent

    def __init__(self, agent_t, agent_config_t):
        self._agent_t = agent_t
        self._agent_config_t = agent_config_t
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._session_titled: set[str] = set()
        self._last_file_edits: dict[tuple[str, str], dict[str, str | None]] = {}
        self._last_terminal_ids: dict[str, str] = {}

    def on_connect(self, conn: Client) -> None:
        log.debug("on_connect")
        self._conn = conn

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        log.debug("cancel received for session %s", session_id)
        task = self._active_tasks.get(session_id)
        if task and not task.done():
            task.cancel()

    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse:
        await self._agent.__aexit__(None, None, None)
        self._session_titled.discard(session_id)
        self._last_terminal_ids.pop(session_id, None)
        for key in [k for k in self._last_file_edits if k[0] == session_id]:
            del self._last_file_edits[key]
        return CloseSessionResponse()

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        log.debug("initialize")

        from google.antigravity import types as agy_types

        # Build standalone tool functions that capture `self` via closure.
        # Bound methods can't be passed directly because LocalAgentConfig
        # deep-copies the config, which fails to pickle file descriptors on self._conn.
        agent_ref = self

        async def view_file(path: str) -> str:
            """Reads and returns the contents of a file via the IDE."""
            try:
                sid = current_session_id.get()
                resp = await agent_ref._conn.read_text_file(path=path, session_id=sid)
                return resp.content
            except Exception as e:
                return f"Error: Failed to read file '{path}': {e}"

        async def create_file(path: str, content: str) -> str:
            """Creates a new file with the specified content via the IDE."""
            try:
                sid = current_session_id.get()
                agent_ref._last_file_edits[(sid, path)] = {"old_text": None, "new_text": content}
                await agent_ref._conn.write_text_file(content=content, path=path, session_id=sid)
                return f"Successfully created file: {path}"
            except Exception as e:
                return f"Error: Failed to create file '{path}': {e}"

        async def edit_file(path: str, content: str) -> str:
            """Overwrites an existing file with new content via the IDE."""
            try:
                sid = current_session_id.get()
                old_text = None
                try:
                    old_resp = await agent_ref._conn.read_text_file(path=path, session_id=sid)
                    old_text = old_resp.content
                except Exception:
                    pass
                agent_ref._last_file_edits[(sid, path)] = {"old_text": old_text, "new_text": content}
                await agent_ref._conn.write_text_file(content=content, path=path, session_id=sid)
                return f"Successfully edited file: {path}"
            except Exception as e:
                return f"Error: Failed to edit file '{path}': {e}"

        async def run_command(command: str) -> str:
            """Runs a shell command in an IDE-managed terminal."""
            try:
                sid = current_session_id.get()
                term_resp = await agent_ref._conn.create_terminal(command=command, session_id=sid)
                terminal_id = term_resp.terminal_id
                agent_ref._last_terminal_ids[sid] = terminal_id
                await agent_ref._conn.wait_for_terminal_exit(session_id=sid, terminal_id=terminal_id)
                out_resp = await agent_ref._conn.terminal_output(session_id=sid, terminal_id=terminal_id)
                await agent_ref._conn.release_terminal(session_id=sid, terminal_id=terminal_id)
                return out_resp.output
            except Exception as e:
                return f"Error: Failed to run command '{command}': {e}"

        self.view_file = view_file
        self.create_file = create_file
        self.edit_file = edit_file
        self.run_command = run_command

        config = self._agent_config_t(
            capabilities=agy_types.CapabilitiesConfig(
                disabled_tools=[
                    agy_types.BuiltinTools.VIEW_FILE,
                    agy_types.BuiltinTools.CREATE_FILE,
                    agy_types.BuiltinTools.EDIT_FILE,
                    agy_types.BuiltinTools.RUN_COMMAND,
                ]
            ),
            tools=[view_file, create_file, edit_file, run_command],
        )
        self._agent = self._agent_t(config)

        # Register safety hook before starting the agent session
        hook = MyPreToolCallDecideHook(self)
        self._agent.register_hook(hook)

        await self._agent.__aenter__()

        log.debug("initialized")

        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(
                prompt_capabilities=PromptCapabilities(
                    image=True, audio=True, embedded_context=True,
                ),
                session_capabilities=SessionCapabilities(
                    close=SessionCloseCapabilities(),
                ),
            ),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        self._cwd = cwd
        session_id = uuid4().hex

        # Wire cwd to workspaces on LocalAgentConfig if supported by the agent config
        if hasattr(self._agent, "_config") and hasattr(self._agent._config, "workspaces"):
            self._agent._config.workspaces = [cwd]

        asyncio.ensure_future(self._send_available_commands(session_id))

        return NewSessionResponse(
            session_id=session_id,
            config_options=[
                SessionConfigOptionSelect(id="agent", name="Agent", description="Agenting",
                                          current_value="Agent",
                                          options=[
                                              SessionConfigSelectGroup(
                                                  group="agent", name="Agent", options=[
                                                      SessionConfigSelectOption(
                                                          description="Agent", name="Agent", value="agent",
                                                      ),
                                                      SessionConfigSelectOption(
                                                          description="Plan", name="Plan", value="plan",
                                                      ),
                                                  ]
                                              )],
            type="select")])

    async def _send_available_commands(self, session_id: str) -> None:
        await self._conn.session_update(
            session_id=session_id,
            update=update_available_commands([
                AvailableCommand(name="/reset", description="Clear conversation history"),
            ]),
        )

    async def prompt(
        self,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
            ],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        log.debug("prompt called, blocks=%d, session_id=%s", len(prompt), session_id)

        parts: list[agy.types.ContentPrimitive] = []
        for block in prompt:
            match block:
                case {"type": "text", "text": text}:
                    parts.append(text)
                case TextContentBlock(text=text):
                    parts.append(text)
                case ImageContentBlock(data=data, mime_type=mime):
                    parts.append(agy.types.Image(
                        data=base64.b64decode(data),
                        mime_type=mime,
                    ))
                case AudioContentBlock(data=data, mime_type=mime):
                    parts.append(agy.types.Audio(
                        data=base64.b64decode(data),
                        mime_type=mime,
                    ))
                case EmbeddedResourceContentBlock(resource=TextResourceContents(text=text)):
                    parts.append(text)
                case EmbeddedResourceContentBlock(resource=BlobResourceContents(blob=blob, mime_type=mime)):
                    parts.append(agy.types.Document(
                        data=base64.b64decode(blob),
                        mime_type=mime or "application/octet-stream",
                    ))
                case ResourceContentBlock(uri=uri, name=name):
                    parts.append(f"[Attached resource: {name}]({uri})")
                case _:
                    log.debug("skipping unknown block: %s", type(block))

        if not parts:
            log.debug("no content to send")
            return PromptResponse(user_message_id=message_id, stop_reason="end_turn")

        if session_id not in self._session_titled:
            self._session_titled.add(session_id)
            first_text = next((p for p in parts if isinstance(p, str)), None)
            if first_text:
                title = first_text[:80].split("\n")[0]
                await self._conn.session_update(
                    session_id=session_id,
                    update=SessionInfoUpdate(
                        session_update="session_info_update",
                        title=title,
                    ),
                )

        log.debug("calling agent.chat with %d parts", len(parts))
        self._active_tasks[session_id] = asyncio.current_task()
        tracker = ToolCallTracker()
        stop_reason = "end_turn"
        response = None
        
        thought_lines: list[str] = []
        last_plan_len = 0

        # Set session context var so safety hook can request permissions associated with session
        token = current_session_id.set(session_id)
        try:
            response = await self._agent.chat(parts)
            async for chunk in response.chunks:
                match chunk:
                    case agy.types.Thought(text=t):
                        await self._conn.session_update(
                            session_id=session_id,
                            update=update_agent_thought_text(t))

                        thought_lines.extend(t.split("\n"))
                        entries = _parse_plan_entries(thought_lines)
                        if entries and len(entries) != last_plan_len:
                            last_plan_len = len(entries)
                            await self._conn.session_update(
                                session_id=session_id,
                                update=update_plan(entries),
                            )
                    case agy.types.Text(text=t):
                        await self._conn.session_update(
                            session_id=session_id,
                            update=update_agent_message(text_block(t)))
                    case agy.types.ToolCall(name=name, id=tc_id, args=args):
                        tc_id = tc_id or uuid4().hex
                        tool_name = str(name).lower()
                        
                        kind = "execute"
                        locations = None
                        
                        if "read" in tool_name or "view" in tool_name:
                            kind = "read"
                        elif "create" in tool_name or "write" in tool_name or "edit" in tool_name:
                            kind = "edit"
                            
                        if isinstance(args, dict) and "path" in args:
                            locations = [ToolCallLocation(path=args["path"])]
                            
                        start = tracker.start(
                            tc_id,
                            title=str(name),
                            kind=kind,
                            locations=locations,
                            raw_input=args
                        )
                        await self._conn.session_update(
                            session_id=session_id, update=start)
                    case agy.types.ToolResult(name=name, id=tc_id, result=result, error=err):
                        tc_id = tc_id or ""
                        
                        content = None
                        status = "failed" if err else "completed"
                        
                        try:
                            view = tracker.view(tc_id)
                            if view.kind == "edit" and isinstance(view.raw_input, dict):
                                path = view.raw_input.get("path")
                                if path:
                                    edit_info = self._last_file_edits.get((session_id, path))
                                    if edit_info:
                                        content = [tool_diff_content(
                                            path=path,
                                            new_text=edit_info["new_text"],
                                            old_text=edit_info["old_text"]
                                        )]
                                        self._last_file_edits.pop((session_id, path), None)
                            elif view.kind == "execute":
                                terminal_id = self._last_terminal_ids.get(session_id)
                                if terminal_id:
                                    content = [tool_terminal_ref(terminal_id=terminal_id)]
                                    self._last_terminal_ids.pop(session_id, None)
                        except KeyError:
                            log.debug("tool result view query failed for %s", tc_id)

                        try:
                            progress = tracker.progress(
                                tc_id,
                                status=status,
                                content=content,
                                raw_output=err or result)
                            await self._conn.session_update(
                                session_id=session_id, update=progress)
                        except KeyError:
                            log.debug("tool result for unknown call %s", tc_id)
                    case _:
                        log.debug("unhandled chunk type: %s", type(chunk))
        except asyncio.CancelledError:
            log.debug("prompt cancelled for session %s", session_id)
            if response is not None and hasattr(response, "cancel"):
                await response.cancel()
            stop_reason = "cancelled"
        except Exception as e:
            log.exception("error during agent chat")
            await self._conn.session_update(
                session_id=session_id,
                update=update_agent_message(text_block(f"Error: {e}")))
        finally:
            current_session_id.reset(token)
            self._active_tasks.pop(session_id, None)

        usage = None
        try:
            if response is not None:
                meta = response.usage_metadata
                if meta:
                    usage = Usage(
                        input_tokens=meta.prompt_token_count or 0,
                        output_tokens=meta.candidates_token_count or 0,
                        total_tokens=meta.total_token_count or 0,
                        thought_tokens=meta.thoughts_token_count,
                        cached_read_tokens=meta.cached_content_token_count,
                    )
                    used = meta.total_token_count or 0
                    await self._conn.session_update(
                        session_id=session_id,
                        update=UsageUpdate(
                            session_update="usage_update",
                            size=max(used, 1),
                            used=used,
                        ),
                    )
        except Exception:
            pass

        log.debug("returning PromptResponse stop_reason=%s", stop_reason)
        return PromptResponse(
            user_message_id=message_id,
            stop_reason=stop_reason,
            usage=usage,
        )

async def main() -> None:
    await run_agent(EchoAgent(agent_config_t=agy.LocalAgentConfig, agent_t=agy.Agent))

if __name__ == "__main__":
    asyncio.run(main())
