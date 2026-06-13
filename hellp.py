import asyncio
import base64
import json
import logging
import os
import re
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import google.antigravity as agy
from google.antigravity.hooks.hooks import PostToolCallHook, PreToolCallDecideHook, HookContext
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
    tool_content,
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
    AgentCapabilities, AuthEnvVar, AuthenticateResponse,
    AvailableCommand, CloseSessionResponse,
    CurrentModeUpdate, EnvVarAuthMethod,
    ListSessionsResponse, LoadSessionResponse,
    ModelInfo, PromptCapabilities, SessionInfo, SessionInfoUpdate,
    SessionConfigOptionSelect, SessionConfigSelectOption,
    SessionListCapabilities, SessionMode, SessionModeState,
    SessionModelState,
    SetSessionConfigOptionResponse, SetSessionModeResponse,
    SetSessionModelResponse, Usage,
    SessionCapabilities, SessionCloseCapabilities,
    RequestPermissionRequest, RequestPermissionResponse,
    UsageUpdate, ToolCallUpdate, ToolCallLocation,
)

current_session_id = ContextVar("current_session_id")

_DEFAULT_STORE_PATH = Path.home() / ".agy-acp" / "sessions.json"

_AVAILABLE_MODELS = [
    ModelInfo(model_id="gemini-2.5-pro", name="Gemini 2.5 Pro", description="Most capable model"),
    ModelInfo(model_id="gemini-2.5-flash", name="Gemini 2.5 Flash", description="Fast and efficient"),
    ModelInfo(model_id="gemini-2.0-flash", name="Gemini 2.0 Flash", description="Previous generation flash"),
]
_DEFAULT_MODEL_ID = "gemini-2.5-pro"


class SessionStore:
    def __init__(self, path: Path = _DEFAULT_STORE_PATH):
        self._path = path

    def _read(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text())

    def _write(self, data: dict[str, dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2))

    def save(self, session_id: str, info: dict) -> None:
        data = self._read()
        data[session_id] = info
        self._write(data)

    def list(self, cwd: str | None = None) -> list[dict]:
        data = self._read()
        sessions = list(data.values())
        if cwd:
            sessions = [s for s in sessions if s.get("cwd") == cwd]
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    def load(self, session_id: str) -> dict | None:
        return self._read().get(session_id)

    def delete(self, session_id: str) -> None:
        data = self._read()
        data.pop(session_id, None)
        self._write(data)


def _tool_title(name: str, args: Any) -> str:
    n = str(name)
    if isinstance(args, dict):
        for key in ("path", "command", "query", "pattern", "directory"):
            if key in args:
                return f"{n}: {args[key]}"
    return n


def _tool_kind(name: str) -> str:
    n = str(name).lower()
    if "read" in n or "view" in n or "list" in n:
        return "read"
    if "find" in n or "search" in n or "grep" in n:
        return "search"
    if "create" in n or "write" in n or "edit" in n:
        return "edit"
    if "delete" in n or "remove" in n:
        return "delete"
    if "move" in n or "rename" in n:
        return "move"
    if "run" in n or "execute" in n or "command" in n:
        return "execute"
    return "other"


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
    """Intercepts tool calls: sends ACP start notification and requests permission."""

    def __init__(self, echo_agent: "EchoAgent"):
        self.echo_agent = echo_agent

    async def run(self, context: HookContext, data: agy.types.ToolCall) -> HookResult:
        session_id = current_session_id.get(None)
        if not session_id:
            log.warning("No session ID found in context for tool call %s", data.name)
            return HookResult(allow=True)

        tool_call_id = data.id or uuid4().hex
        kind = _tool_kind(str(data.name))
        locations = None
        if isinstance(data.args, dict) and "path" in data.args:
            locations = [ToolCallLocation(path=data.args["path"])]

        # Start ACP tool call tracking and notify client
        # https://github.com/google-antigravity/antigravity-sdk-python/tree/main/examples/deep_dives/host_tool_hooks.py
        start = self.echo_agent._tracker.start(
            tool_call_id,
            title=_tool_title(str(data.name), data.args),
            kind=kind,
            locations=locations,
            raw_input=data.args,
        )
        await self.echo_agent._conn.session_update(
            session_id=session_id, update=start)

        context.set("acp_tc_id", tool_call_id)

        log.debug("Intercepted tool call %s in session %s", data.name, session_id)

        async def requester(request: RequestPermissionRequest) -> RequestPermissionResponse:
            return await self.echo_agent._conn.request_permission(
                options=request.options,
                session_id=request.session_id,
                tool_call=request.tool_call
            )

        tool_call = ToolCallUpdate(
            tool_call_id=tool_call_id,
            title=_tool_title(str(data.name), data.args),
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


class MyPostToolCallHook(PostToolCallHook):
    """Sends ACP tool_call_update with status=completed after tool execution."""

    def __init__(self, echo_agent: "EchoAgent"):
        self.echo_agent = echo_agent

    async def run(self, context: HookContext, data: agy.types.ToolResult) -> None:
        session_id = current_session_id.get(None)
        tc_id = context.get("acp_tc_id")
        if not session_id or not tc_id:
            return

        status = "failed" if data.error else "completed"
        summary = str(data.error or data.result)[:2000]
        content = [tool_content(text_block(summary))]

        try:
            view = self.echo_agent._tracker.view(tc_id)
            if view.kind == "edit" and isinstance(view.raw_input, dict):
                path = view.raw_input.get("path")
                edit_info = self.echo_agent._last_file_edits.pop((session_id, path), None)
                if edit_info:
                    content = [tool_diff_content(
                        path=path,
                        new_text=edit_info["new_text"],
                        old_text=edit_info["old_text"],
                    )]
            elif view.kind == "execute":
                terminal_id = self.echo_agent._last_terminal_ids.pop(session_id, None)
                if terminal_id:
                    content = [tool_terminal_ref(terminal_id=terminal_id)]
        except KeyError:
            pass

        try:
            progress = self.echo_agent._tracker.progress(
                tc_id, status=status, content=content,
                raw_output=data.error or data.result)
            await self.echo_agent._conn.session_update(
                session_id=session_id, update=progress)
        except KeyError:
            log.debug("post hook: unknown tracker id %s", tc_id)


class EchoAgent(Agent):
    _conn: Client
    _agent: agy.Agent

    def __init__(self, agent_t, agent_config_t, store: SessionStore | None = None):
        self._agent_t = agent_t
        self._agent_config_t = agent_config_t
        self._store = store or SessionStore()
        self._tracker = ToolCallTracker()
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._session_titles: dict[str, str] = {}
        self._session_modes: dict[str, str] = {}
        self._active_session_id: str | None = None
        self._session_models: dict[str, str] = {}
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
        self._session_titles.pop(session_id, None)
        self._session_modes.pop(session_id, None)
        self._session_models.pop(session_id, None)
        self._last_terminal_ids.pop(session_id, None)
        for key in [k for k in self._last_file_edits if k[0] == session_id]:
            del self._last_file_edits[key]
        self._store.delete(session_id)
        return CloseSessionResponse()

    async def set_session_mode(self, mode_id: str, session_id: str, **kwargs: Any) -> SetSessionModeResponse:
        log.debug("set_session_mode mode=%s session=%s", mode_id, session_id)
        self._session_modes[session_id] = mode_id
        await self._conn.session_update(
            session_id=session_id,
            update=CurrentModeUpdate(
                session_update="current_mode_update",
                current_mode_id=mode_id,
            ),
        )
        return SetSessionModeResponse()

    def _build_config_options(self, session_id: str) -> list[SessionConfigOptionSelect]:
        current_mode = self._session_modes.get(session_id, "agent")
        return [
            SessionConfigOptionSelect(
                id="mode", name="Mode", type="select",
                description="Controls agent behavior",
                category="mode",
                current_value=current_mode,
                options=[
                    SessionConfigSelectOption(value="agent", name="Agent", description="Execute tools autonomously"),
                    SessionConfigSelectOption(value="plan", name="Plan", description="Produce a plan without executing tools"),
                ],
            ),
        ]

    # https://agentclientprotocol.com/protocol/v1/session-config-options
    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **kwargs: Any,
    ) -> SetSessionConfigOptionResponse:
        log.debug("set_config_option config_id=%s value=%s session=%s", config_id, value, session_id)
        if config_id == "mode" and isinstance(value, str):
            self._session_modes[session_id] = value
            await self._conn.session_update(
                session_id=session_id,
                update=CurrentModeUpdate(
                    session_update="current_mode_update",
                    current_mode_id=value,
                ),
            )
        return SetSessionConfigOptionResponse(config_options=self._build_config_options(session_id))

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None:
        log.debug("authenticate method_id=%s", method_id)
        if method_id == "gemini_api_key":
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                log.warning("GEMINI_API_KEY not set in environment")
        return AuthenticateResponse()

    def _build_model_state(self, session_id: str) -> SessionModelState:
        current = self._session_models.get(session_id, _DEFAULT_MODEL_ID)
        return SessionModelState(
            current_model_id=current,
            available_models=_AVAILABLE_MODELS,
        )

    async def set_session_model(
        self, model_id: str, session_id: str, **kwargs: Any,
    ) -> SetSessionModelResponse | None:
        log.debug("set_session_model model=%s session=%s", model_id, session_id)
        self._session_models[session_id] = model_id
        return SetSessionModelResponse()

    async def list_sessions(
        self,
        additional_directories: list[str] | None = None,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        sessions = self._store.list(cwd=cwd)
        return ListSessionsResponse(
            sessions=[
                SessionInfo(
                    session_id=s["session_id"],
                    cwd=s["cwd"],
                    title=s.get("title"),
                    updated_at=s.get("updated_at"),
                )
                for s in sessions
            ],
        )

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        stored = self._store.load(session_id)
        if not stored:
            return None

        self._cwd = cwd
        mode = stored.get("mode", "agent")
        model = stored.get("model", _DEFAULT_MODEL_ID)
        self._session_modes[session_id] = mode
        self._session_models[session_id] = model
        if stored.get("title"):
            self._session_titles[session_id] = stored.get("title", "")

        conv_id = stored.get("conversation_id")
        if conv_id:
            log.debug("resuming conversation %s for session %s", conv_id, session_id)

        asyncio.ensure_future(self._send_available_commands(session_id))

        return LoadSessionResponse(
            modes=SessionModeState(
                current_mode_id=mode,
                available_modes=[
                    SessionMode(id="agent", name="Agent", description="Execute tools autonomously"),
                    SessionMode(id="plan", name="Plan", description="Produce a plan without executing tools"),
                ],
            ),
            models=self._build_model_state(session_id),
            config_options=self._build_config_options(session_id),
        )

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
                sid = agent_ref._active_session_id
                if not sid:
                    return f"Error: no active session for reading '{path}'"
                resp = await agent_ref._conn.read_text_file(path=path, session_id=sid)
                return resp.content
            except Exception as e:
                return f"Error: Failed to read file '{path}': {e}"

        async def create_file(path: str, content: str) -> str:
            """Creates a new file with the specified content via the IDE."""
            try:
                sid = agent_ref._active_session_id
                agent_ref._last_file_edits[(sid, path)] = {"old_text": None, "new_text": content}
                await agent_ref._conn.write_text_file(content=content, path=path, session_id=sid)
                return f"Successfully created file: {path}"
            except Exception as e:
                return f"Error: Failed to create file '{path}': {e}"

        async def edit_file(path: str, content: str) -> str:
            """Overwrites an existing file with new content via the IDE."""
            try:
                sid = agent_ref._active_session_id
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
                sid = agent_ref._active_session_id
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

        self._agent.register_hook(MyPreToolCallDecideHook(self))
        self._agent.register_hook(MyPostToolCallHook(self))

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
                    list=SessionListCapabilities(),
                ),
                load_session=True,
            ),
            auth_methods=[
                EnvVarAuthMethod(
                    type="env_var",
                    id="gemini_api_key",
                    name="Gemini API Key",
                    description="Google Gemini API key for the Antigravity SDK",
                    vars=[
                        AuthEnvVar(name="GEMINI_API_KEY", label="API Key"),
                    ],
                ),
            ],
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

        self._session_modes[session_id] = "agent"
        self._session_models[session_id] = _DEFAULT_MODEL_ID
        asyncio.ensure_future(self._send_available_commands(session_id))

        return NewSessionResponse(
            session_id=session_id,
            modes=SessionModeState(
                current_mode_id="agent",
                available_modes=[
                    SessionMode(id="agent", name="Agent", description="Execute tools autonomously"),
                    SessionMode(id="plan", name="Plan", description="Produce a plan without executing tools"),
                ],
            ),
            models=self._build_model_state(session_id),
            config_options=self._build_config_options(session_id),
        )

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

        if self._session_modes.get(session_id) == "plan":
            parts.append("\n[PLAN MODE: Produce a step-by-step plan. Do not execute any tools.]")

        if session_id not in self._session_titles:
            first_text = next((p for p in parts if isinstance(p, str)), None)
            if first_text:
                title = first_text[:80].split("\n")[0]
                self._session_titles[session_id] = title
                await self._conn.session_update(
                    session_id=session_id,
                    update=SessionInfoUpdate(
                        session_update="session_info_update",
                        title=title,
                    ),
                )

        log.debug("calling agent.chat with %d parts", len(parts))
        self._active_tasks[session_id] = asyncio.current_task()
        stop_reason = "end_turn"
        response = None

        thought_lines: list[str] = []
        last_plan_len = 0

        # Tool call start/completion is handled by MyPreToolCallDecideHook / MyPostToolCallHook
        self._active_session_id = session_id
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
                    case agy.types.ToolCall():
                        pass  # handled by PreToolCallDecideHook
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

        self._store.save(session_id, {
            "session_id": session_id,
            "conversation_id": getattr(self._agent, "conversation_id", None),
            "cwd": getattr(self, "_cwd", "."),
            "mode": self._session_modes.get(session_id, "agent"),
            "model": self._session_models.get(session_id, _DEFAULT_MODEL_ID),
            "title": self._session_titles.get(session_id),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

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
