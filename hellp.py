import asyncio
import base64
import json
import logging
import os
import re
import sys
import tempfile
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import google.antigravity as agy
from google.antigravity.hooks import policy as agy_policy
from google.antigravity.hooks.hooks import (
    HookContext,
    PostToolCallHook,
    PreToolCallDecideHook,
)
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
    update_agent_message,
    update_agent_thought_text,
)
from acp.contrib.permissions import PermissionBroker
from acp.contrib.tool_calls import ToolCallTracker
from acp.helpers import (
    plan_entry,
    tool_content,
    tool_diff_content,
    tool_terminal_ref,
    update_available_commands,
    update_plan,
)
from acp.interfaces import Client
from acp.schema import (
    AgentAuthCapabilities,
    AgentCapabilities,
    AudioContentBlock,
    AuthenticateResponse,
    AuthEnvVar,
    AvailableCommand,
    BlobResourceContents,
    ClientCapabilities,
    CloseSessionResponse,
    ConfigOptionUpdate,
    Cost,
    CurrentModeUpdate,
    EmbeddedResourceContentBlock,
    EnvVarAuthMethod,
    ForkSessionResponse,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    ListSessionsResponse,
    LoadSessionResponse,
    McpCapabilities,
    McpServerStdio,
    ModelInfo,
    PromptCapabilities,
    RequestPermissionRequest,
    RequestPermissionResponse,
    ResourceContentBlock,
    ResumeSessionResponse,
    SessionAdditionalDirectoriesCapabilities,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionForkCapabilities,
    SessionInfo,
    SessionInfoUpdate,
    SessionListCapabilities,
    SessionMode,
    SessionModelState,
    SessionModeState,
    SessionResumeCapabilities,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
    SetSessionModeResponse,
    SseMcpServer,
    TextContentBlock,
    TextResourceContents,
    ToolCallLocation,
    ToolCallUpdate,
    Usage,
    UnstructuredCommandInput,
    UsageUpdate,
)

current_session_id = ContextVar("current_session_id")

_DEFAULT_STORE_PATH = Path.home() / ".agy-acp" / "sessions.json"
_DEFAULT_SAVE_DIR = str(Path.home() / ".agy-acp" / "trajectories")

_AVAILABLE_MODELS = [
    ModelInfo(model_id="gemini-3.5-flash", name="Gemini 3.5 Flash"),
    ModelInfo(model_id="gemini-3.1-pro-preview", name="Gemini 3.1 Pro"),
    ModelInfo(
        model_id="gemini-3.1-pro-preview-customtools",
        name="Gemini 3.1 Pro (Custom Tools)",
    ),
    ModelInfo(model_id="gemini-3.1-flash-lite", name="Gemini 3.1 Flash Lite"),
    ModelInfo(model_id="gemini-2.5-pro", name="Gemini 2.5 Pro"),
    ModelInfo(model_id="gemini-2.5-flash", name="Gemini 2.5 Flash"),
    ModelInfo(model_id="gemini-2.5-flash-lite", name="Gemini 2.5 Flash Lite"),
]
_DEFAULT_MODEL_ID = "gemini-3.1-flash-lite"

_THINKING_LEVELS = ["minimal", "low", "medium", "high"]
_DEFAULT_THINKING_LEVEL = "medium"

# USD per 1M tokens (input, output). Source: ai.google.dev/pricing
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3.1-pro-preview-customtools": (2.00, 12.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}

# Pro models: input 2x, output 1.5x when context exceeds 200k tokens
_LONG_CONTEXT_THRESHOLD = 200_000

from google.antigravity.types import BuiltinTools

# All SDK built-in tools except RUN_COMMAND auto-allow.
# RUN_COMMAND and MCP server tools (mcp_* prefix) require IDE permission.
_AUTO_ALLOW_TOOLS = {
    BuiltinTools.VIEW_FILE.value,
    BuiltinTools.CREATE_FILE.value,
    BuiltinTools.EDIT_FILE.value,
    BuiltinTools.LIST_DIR.value,
    BuiltinTools.FIND_FILE.value,
    BuiltinTools.SEARCH_DIR.value,
    BuiltinTools.ASK_QUESTION.value,
    BuiltinTools.FINISH.value,
    BuiltinTools.START_SUBAGENT.value,
    BuiltinTools.GENERATE_IMAGE.value,
}


def _get_token_rates(
    model_id: str, total_context_tokens: int
) -> tuple[float, float] | None:
    pricing = _MODEL_PRICING.get(model_id)
    if not pricing:
        return None
    base_in, base_out = pricing
    if "pro" in model_id and total_context_tokens > _LONG_CONTEXT_THRESHOLD:
        return (base_in * 2.0, base_out * 1.5)
    return (base_in, base_out)


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
        for key in ("path", "command", "command_line", "query", "pattern", "directory"):
            if key in args:
                return f"{n}: {args[key]}"
        # MCP tools: extract ServerName/ToolName from request_text
        if n.startswith("mcp_"):
            server, tool = _parse_mcp_request_text(args)
            if server and tool:
                return f"{tool} ({server})"
    return n


def _parse_mcp_request_text(args: dict) -> tuple[str | None, str | None]:
    """Extract ServerName and ToolName from MCP tool args' request_text JSON."""
    request_text = args.get("request_text", "")
    if request_text and "{" in request_text:
        try:
            start = request_text.index("{")
            embedded = json.loads(request_text[start:])
            return embedded.get("ServerName"), embedded.get("ToolName")
        except json.JSONDecodeError, ValueError:
            pass
    return None, None


def _permission_description(name: str, args: Any) -> str:
    """Build a human-readable description of tool arguments for the permission dialog.

    The title already identifies the tool, so this focuses on what's being passed to it.
    """
    if not isinstance(args, dict):
        return ""

    # Filter out SDK metadata keys that clutter the display
    display_args = {k: v for k, v in args.items() if k != "request_text"}

    # MCP tools: show the actual MCP arguments from request_text
    if name.startswith("mcp_"):
        request_text = args.get("request_text", "")
        if request_text and "{" in request_text:
            try:
                start = request_text.index("{")
                embedded = json.loads(request_text[start:])
                mcp_args = embedded.get("Arguments", {})
                if mcp_args:
                    return "\n".join(f"- **{k}**: `{v}`" for k, v in mcp_args.items())
                return "*(no arguments)*"
            except json.JSONDecodeError, ValueError:
                pass

    # run_command: show working dir (command is already in the title)
    if display_args.get("working_dir"):
        return f"in `{display_args['working_dir']}`"

    # Generic: list non-empty args as markdown
    if display_args:
        return "\n".join(f"- **{k}**: `{v}`" for k, v in display_args.items())

    return ""


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
    r"[-*]\s+\[([xX /\-])\]\s+(.*)"  # - [x] item  or  * [ ] item
    r"|[-*]\s+(.*)"  # - item  or  * item
    r"|(\d+)\.\s+(.*)"  # 1. item  or  23. item
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


def _convert_mcp_server(server: HttpMcpServer | SseMcpServer | McpServerStdio):
    """Convert ACP MCP server config to Antigravity SDK type."""
    from google.antigravity import types as agy_types

    if isinstance(server, (HttpMcpServer, SseMcpServer)):
        headers = {h.name: h.value for h in server.headers} if server.headers else {}
        return agy_types.McpStreamableHttpServer(
            name=server.name,
            url=server.url,
            type="http",
            headers=headers,
        )

    # agy SDK gap: McpStdioServer has no env field,
    # see github.com/google-antigravity/antigravity-sdk-python/issues/61
    if isinstance(server, McpServerStdio) and server.env:
        env_dict = {e.name: e.value for e in server.env}
        fd, env_file = tempfile.mkstemp(prefix="agy_mcp_env_", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(env_dict, f)
        os.chmod(env_file, 0o600)
        loader = (
            f"import json,os,sys;"
            f"e=json.load(open({env_file!r}));"
            f"os.unlink({env_file!r});"
            f"os.environ.update(e);"
            f"os.execvp({server.command!r},[{server.command!r}]+{server.args!r})"
        )
        return agy_types.McpStdioServer(
            name=server.name,
            command=sys.executable,
            args=["-ISs", "-c", loader],
            type="stdio",
        )

    return agy_types.McpStdioServer(
        name=server.name,
        command=server.command,
        args=server.args,
        type="stdio",
    )


def _convert_mcp_servers(
    servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None,
) -> list | None:
    if not servers:
        return None
    return [_convert_mcp_server(s) for s in servers]


class MyPreToolCallDecideHook(PreToolCallDecideHook):
    """Intercepts tool calls: sends ACP start notification and requests permission."""

    def __init__(self, echo_agent: "EchoAgent"):
        self.echo_agent = echo_agent

    async def run(self, context: HookContext, data: agy.types.ToolCall) -> HookResult:
        session_id = current_session_id.get(None) or self.echo_agent._active_session_id
        if not session_id:
            log.warning(
                "No session ID found in context for tool call %s — denying", data.name
            )
            return HookResult(allow=False, message="No active session context")

        tool_call_id = data.id or uuid4().hex
        kind = _tool_kind(str(data.name))
        locations = None
        if isinstance(data.args, dict) and "path" in data.args:
            locations = [ToolCallLocation(path=data.args["path"])]
        title = _tool_title(str(data.name), data.args)

        context.set("acp_tc_id", tool_call_id)
        log.debug("Intercepted tool call %s in session %s", data.name, session_id)

        if str(data.name) in _AUTO_ALLOW_TOOLS:
            await self._send_start(
                session_id, tool_call_id, title, kind, locations, data.args
            )
            return HookResult(allow=True)

        # Permission required — show the approval dialog first, don't send
        # the progress card yet (avoids duplicate cards in the IDE)
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

        try:
            resp = await broker.request_for(
                external_id=tool_call_id,
                tool_call=tool_call,
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
                # Register tracker entry (for PostToolCallHook) but don't send
                # a notification — the broker's request_for already showed the card
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
            return HookResult(
                allow=False, message=f"Internal permission broker error: {e}"
            )

    async def _send_start(
        self, session_id, tool_call_id, title, kind, locations, raw_input
    ):
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

    def __init__(self, echo_agent: "EchoAgent"):
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
            view = self.echo_agent._tracker.view(tc_id)
            if view.kind == "edit" and isinstance(view.raw_input, dict):
                path = view.raw_input.get("path")
                edit_info = self.echo_agent._last_file_edits.pop(
                    (session_id, path), None
                )
                if edit_info:
                    content = [
                        tool_diff_content(
                            path=path,
                            new_text=edit_info["new_text"],
                            old_text=edit_info["old_text"],
                        )
                    ]
            elif view.kind == "execute":
                exit_code = self.echo_agent._last_exit_codes.pop(session_id, None)
                if exit_code is not None and exit_code != 0:
                    status = "failed"
                    raw_output = {"exit_code": exit_code, "output": data.result}
                terminal_id = self.echo_agent._last_terminal_ids.pop(session_id, None)
                if terminal_id:
                    content = [tool_terminal_ref(terminal_id=terminal_id)]
        except KeyError:
            pass

        try:
            progress = self.echo_agent._tracker.progress(
                tc_id, status=status, content=content, raw_output=raw_output
            )
            await self.echo_agent._conn.session_update(
                session_id=session_id, update=progress
            )
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
        self._session_thinking_levels: dict[str, str] = {}
        self._session_additional_dirs: dict[str, list[str]] = {}
        self._session_mcp_servers: dict[str, list] = {}
        self._session_cumulative_cost: dict[str, float] = {}
        self._last_file_edits: dict[tuple[str, str], dict[str, str | None]] = {}
        self._last_terminal_ids: dict[str, str] = {}
        self._last_exit_codes: dict[str, int] = {}
        self._last_usage: dict[str, dict] = {}
        self._client_capabilities: ClientCapabilities | None = None
        self._client_info: Implementation | None = None

    @property
    def _is_intellij(self) -> bool:
        return bool(
            self._client_info
            and getattr(self._client_info, "name", "")
            and "JetBrains" in self._client_info.name
        )

    def on_connect(self, conn: Client) -> None:
        log.debug("on_connect")
        self._conn = conn

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        log.debug("cancel received for session %s", session_id)
        task = self._active_tasks.get(session_id)
        if task and not task.done():
            task.cancel()

    async def close_session(
        self, session_id: str, **kwargs: Any
    ) -> CloseSessionResponse:
        await self._agent.__aexit__(None, None, None)
        self._session_titles.pop(session_id, None)
        self._session_modes.pop(session_id, None)
        self._session_models.pop(session_id, None)
        self._session_thinking_levels.pop(session_id, None)
        self._session_additional_dirs.pop(session_id, None)
        self._session_mcp_servers.pop(session_id, None)
        self._session_cumulative_cost.pop(session_id, None)
        self._active_tasks.pop(session_id, None)
        self._last_terminal_ids.pop(session_id, None)
        self._last_exit_codes.pop(session_id, None)
        self._last_usage.pop(session_id, None)
        for key in [k for k in self._last_file_edits if k[0] == session_id]:
            del self._last_file_edits[key]
        self._store.delete(session_id)
        return CloseSessionResponse()

    async def set_session_mode(
        self, mode_id: str, session_id: str, **kwargs: Any
    ) -> SetSessionModeResponse:
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
        current_model = self._session_models.get(session_id, _DEFAULT_MODEL_ID)
        current_thinking = self._session_thinking_levels.get(
            session_id, _DEFAULT_THINKING_LEVEL
        )
        return [
            SessionConfigOptionSelect(
                id="mode",
                name="Mode",
                type="select",
                description="Controls agent behavior",
                category="mode",
                current_value=current_mode,
                options=[
                    SessionConfigSelectOption(
                        value="agent",
                        name="Agent",
                        description="Execute tools autonomously",
                    ),
                    SessionConfigSelectOption(
                        value="plan",
                        name="Plan",
                        description="Produce a plan without executing tools",
                    ),
                ],
            ),
            SessionConfigOptionSelect(
                id="model",
                name="Model",
                type="select",
                description="Gemini model to use",
                category="model",
                current_value=current_model,
                options=[
                    SessionConfigSelectOption(value=m.model_id, name=m.name)
                    for m in _AVAILABLE_MODELS
                ],
            ),
            SessionConfigOptionSelect(
                id="thinking_level",
                name="Thinking",
                type="select",
                description="Controls depth of reasoning (3.x models only, ignored for 2.x)",
                category="model",
                current_value=current_thinking,
                options=[
                    SessionConfigSelectOption(value=lvl, name=lvl.capitalize())
                    for lvl in _THINKING_LEVELS
                ],
            ),
        ]

    # https://agentclientprotocol.com/protocol/v1/session-config-options
    async def set_config_option(
        self,
        config_id: str,
        session_id: str,
        value: str | bool,
        **kwargs: Any,
    ) -> SetSessionConfigOptionResponse:
        log.debug(
            "set_config_option config_id=%s value=%s session=%s",
            config_id,
            value,
            session_id,
        )
        if config_id == "mode" and isinstance(value, str):
            self._session_modes[session_id] = value
            await self._conn.session_update(
                session_id=session_id,
                update=CurrentModeUpdate(
                    session_update="current_mode_update",
                    current_mode_id=value,
                ),
            )
        elif config_id == "model" and isinstance(value, str):
            self._session_models[session_id] = value
            await self._rebuild_agent(
                session_id,
                conversation_id=getattr(self._agent, "conversation_id", None),
            )
        elif config_id == "thinking_level" and isinstance(value, str):
            self._session_thinking_levels[session_id] = value
            await self._rebuild_agent(
                session_id,
                conversation_id=getattr(self._agent, "conversation_id", None),
            )
        updated = self._build_config_options(session_id)
        await self._conn.session_update(
            session_id=session_id,
            update=ConfigOptionUpdate(
                session_update="config_option_update",
                config_options=updated,
            ),
        )
        return SetSessionConfigOptionResponse(config_options=updated)

    async def authenticate(
        self, method_id: str, **kwargs: Any
    ) -> AuthenticateResponse | None:
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

    async def _rebuild_agent(
        self,
        session_id: str,
        conversation_id: str | None = None,
        save_dir: str | None = None,
    ) -> None:
        """Tear down and recreate the Antigravity agent with current model/thinking settings."""
        model_id = self._session_models.get(session_id, _DEFAULT_MODEL_ID)
        thinking = self._session_thinking_levels.get(
            session_id, _DEFAULT_THINKING_LEVEL
        )

        old_agent = self._agent
        await old_agent.__aexit__(None, None, None)

        from google.antigravity import types as agy_types

        try:
            disabled_tools, custom_tools = self._build_tools_config()
            config = self._agent_config_t(
                capabilities=agy_types.CapabilitiesConfig(
                    disabled_tools=disabled_tools
                ),
                policies=[agy_policy.allow_all()],
                tools=custom_tools,
                gemini_config=agy_types.GeminiConfig(
                    models=agy_types.ModelConfig(
                        default=agy_types.ModelEntry(
                            name=model_id,
                            generation=agy_types.GenerationConfig(
                                # 2.5 models don't support thinking level
                                thinking_level=agy_types.ThinkingLevel(thinking)
                                if not model_id.startswith("gemini-2.")
                                else None,
                            ),
                        ),
                    ),
                ),
                conversation_id=conversation_id,
                save_dir=save_dir or _DEFAULT_SAVE_DIR,
                workspaces=[getattr(self, "_cwd", ".")]
                + self._session_additional_dirs.get(session_id, []),
                mcp_servers=self._session_mcp_servers.get(session_id) or None,
            )
            new_agent = self._agent_t(config)
            new_agent.register_hook(MyPreToolCallDecideHook(self))
            new_agent.register_hook(MyPostToolCallHook(self))
            await new_agent.__aenter__()
            self._agent = new_agent
        except Exception:
            log.exception("_rebuild_agent failed, restoring previous agent")
            await old_agent.__aenter__()
            self._agent = old_agent
            raise
        log.debug(
            "rebuilt agent model=%s thinking=%s conv=%s",
            model_id,
            thinking,
            conversation_id,
        )

    async def set_session_model(
        self,
        model_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> SetSessionModelResponse | None:
        log.debug("set_session_model model=%s session=%s", model_id, session_id)
        self._session_models[session_id] = model_id
        await self._rebuild_agent(
            session_id, conversation_id=getattr(self._agent, "conversation_id", None)
        )
        return SetSessionModelResponse()

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> ForkSessionResponse:
        log.debug("fork_session from %s", session_id)
        new_id = uuid4().hex
        mode = self._session_modes.get(session_id, "agent")
        model = self._session_models.get(session_id, _DEFAULT_MODEL_ID)
        thinking = self._session_thinking_levels.get(
            session_id, _DEFAULT_THINKING_LEVEL
        )
        title = self._session_titles.get(session_id)

        self._cwd = cwd
        self._session_modes[new_id] = mode
        self._session_models[new_id] = model
        self._session_thinking_levels[new_id] = thinking
        self._session_additional_dirs[new_id] = (
            additional_directories or self._session_additional_dirs.get(session_id, [])
        )
        if mcp_servers:
            self._session_mcp_servers[new_id] = _convert_mcp_servers(mcp_servers) or []
        elif session_id in self._session_mcp_servers:
            self._session_mcp_servers[new_id] = self._session_mcp_servers[session_id]
        if title:
            self._session_titles[new_id] = f"{title} (fork)"

        self._store.save(
            new_id,
            {
                "session_id": new_id,
                "conversation_id": None,
                "cwd": cwd,
                "mode": mode,
                "model": model,
                "thinking_level": thinking,
                "title": self._session_titles.get(new_id),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        asyncio.ensure_future(self._send_available_commands(new_id))

        return ForkSessionResponse(
            session_id=new_id,
            modes=SessionModeState(
                current_mode_id=mode,
                available_modes=[
                    SessionMode(
                        id="agent",
                        name="Agent",
                        description="Execute tools autonomously",
                    ),
                    SessionMode(
                        id="plan",
                        name="Plan",
                        description="Produce a plan without executing tools",
                    ),
                ],
            ),
            models=self._build_model_state(new_id),
            config_options=self._build_config_options(new_id),
        )

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        log.debug("resume_session %s", session_id)
        stored = self._store.load(session_id)
        if not stored:
            raise ValueError(f"Session not found: {session_id}")

        self._cwd = cwd
        mode = stored.get("mode", "agent")
        model = stored.get("model", _DEFAULT_MODEL_ID)
        thinking = stored.get("thinking_level", _DEFAULT_THINKING_LEVEL)
        conv_id = stored.get("conversation_id")

        self._session_modes[session_id] = mode
        self._session_models[session_id] = model
        self._session_thinking_levels[session_id] = thinking
        self._session_additional_dirs[session_id] = additional_directories or []
        converted = _convert_mcp_servers(mcp_servers)
        if converted:
            self._session_mcp_servers[session_id] = converted
        if stored.get("title"):
            self._session_titles[session_id] = stored["title"]

        if conv_id:
            log.debug("resuming conversation %s", conv_id)
            await self._rebuild_agent(session_id, conversation_id=conv_id)

        asyncio.ensure_future(self._send_available_commands(session_id))

        return ResumeSessionResponse(
            modes=SessionModeState(
                current_mode_id=mode,
                available_modes=[
                    SessionMode(
                        id="agent",
                        name="Agent",
                        description="Execute tools autonomously",
                    ),
                    SessionMode(
                        id="plan",
                        name="Plan",
                        description="Produce a plan without executing tools",
                    ),
                ],
            ),
            models=self._build_model_state(session_id),
            config_options=self._build_config_options(session_id),
        )

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
        thinking = stored.get("thinking_level", _DEFAULT_THINKING_LEVEL)
        self._session_modes[session_id] = mode
        self._session_models[session_id] = model
        self._session_thinking_levels[session_id] = thinking
        self._session_additional_dirs[session_id] = additional_directories or []
        converted = _convert_mcp_servers(mcp_servers)
        if converted:
            self._session_mcp_servers[session_id] = converted
        if stored.get("title"):
            self._session_titles[session_id] = stored.get("title", "")

        conv_id = stored.get("conversation_id")
        if conv_id:
            log.debug("restoring conversation %s for session %s", conv_id, session_id)
            await self._rebuild_agent(session_id, conversation_id=conv_id)

        asyncio.ensure_future(self._send_available_commands(session_id))

        return LoadSessionResponse(
            modes=SessionModeState(
                current_mode_id=mode,
                available_modes=[
                    SessionMode(
                        id="agent",
                        name="Agent",
                        description="Execute tools autonomously",
                    ),
                    SessionMode(
                        id="plan",
                        name="Plan",
                        description="Produce a plan without executing tools",
                    ),
                ],
            ),
            models=self._build_model_state(session_id),
            config_options=self._build_config_options(session_id),
        )

    def _check_client_caps(self) -> tuple[bool, bool, bool]:
        """Returns (can_read_files, can_write_files, can_terminal) from stored client capabilities."""
        caps = self._client_capabilities
        fs = getattr(caps, "fs", None) if caps else None
        can_read = bool(getattr(fs, "read_text_file", False)) if fs else False
        can_write = bool(getattr(fs, "write_text_file", False)) if fs else False
        can_terminal = bool(getattr(caps, "terminal", False)) if caps else False
        return can_read, can_write, can_terminal

    def _build_tools_config(self) -> tuple[list, list]:
        """Build disabled_tools and custom tools lists based on client capabilities."""
        from google.antigravity import types as agy_types

        can_read, can_write, can_terminal = self._check_client_caps()
        disabled = []
        tools = []
        if can_read:
            disabled.append(agy_types.BuiltinTools.VIEW_FILE)
            tools.append(self.view_file)
        if can_write:
            disabled.append(agy_types.BuiltinTools.CREATE_FILE)
            disabled.append(agy_types.BuiltinTools.EDIT_FILE)
            tools.extend([self.create_file, self.edit_file])
        if can_terminal:
            disabled.append(agy_types.BuiltinTools.RUN_COMMAND)
            tools.append(self.run_command)
        return disabled, tools

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        log.debug(
            "initialize client_info=%s client_capabilities=%s",
            client_info,
            client_capabilities,
        )
        self._client_capabilities = client_capabilities
        self._client_info = client_info

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
                agent_ref._last_file_edits[(sid, path)] = {
                    "old_text": None,
                    "new_text": content,
                }
                await agent_ref._conn.write_text_file(
                    content=content, path=path, session_id=sid
                )
                return f"Successfully created file: {path}"
            except Exception as e:
                return f"Error: Failed to create file '{path}': {e}"

        async def edit_file(path: str, old_string: str, new_string: str) -> str:
            """Replaces the first occurrence of old_string with new_string in a file."""
            try:
                sid = agent_ref._active_session_id
                resp = await agent_ref._conn.read_text_file(path=path, session_id=sid)
                old_text = resp.content
                if old_string not in old_text:
                    return f"Error: old_string not found in '{path}'"
                new_text = old_text.replace(old_string, new_string, 1)
                agent_ref._last_file_edits[(sid, path)] = {
                    "old_text": old_text,
                    "new_text": new_text,
                }
                await agent_ref._conn.write_text_file(
                    content=new_text, path=path, session_id=sid
                )
                return f"Successfully edited file: {path}"
            except Exception as e:
                return f"Error: Failed to edit file '{path}': {e}"

        async def run_command(command: str) -> str:
            """Runs a shell command in an IDE-managed terminal."""
            try:
                sid = agent_ref._active_session_id
                term_resp = await agent_ref._conn.create_terminal(
                    command=command, session_id=sid
                )
                terminal_id = term_resp.terminal_id
                agent_ref._last_terminal_ids[sid] = terminal_id
                exit_resp = await agent_ref._conn.wait_for_terminal_exit(
                    session_id=sid, terminal_id=terminal_id
                )
                out_resp = await agent_ref._conn.terminal_output(
                    session_id=sid, terminal_id=terminal_id
                )
                await agent_ref._conn.release_terminal(
                    session_id=sid, terminal_id=terminal_id
                )
                exit_code = getattr(exit_resp, "exit_code", None)
                if exit_code is not None:
                    agent_ref._last_exit_codes[sid] = exit_code
                return out_resp.output
            except Exception as e:
                return f"Error: Failed to run command '{command}': {e}"

        self.view_file = view_file
        self.create_file = create_file
        self.edit_file = edit_file
        self.run_command = run_command

        disabled_tools, custom_tools = self._build_tools_config()
        config = self._agent_config_t(
            capabilities=agy_types.CapabilitiesConfig(disabled_tools=disabled_tools),
            policies=[agy_policy.allow_all()],
            tools=custom_tools,
            save_dir=_DEFAULT_SAVE_DIR,
        )
        self._agent = self._agent_t(config)

        self._agent.register_hook(MyPreToolCallDecideHook(self))
        self._agent.register_hook(MyPostToolCallHook(self))

        await self._agent.__aenter__()

        log.debug("initialized")

        return InitializeResponse(
            protocol_version=protocol_version,
            agent_info=Implementation(
                name="agy-acp", version="0.1.0", title="Antigravity ACP Adapter"
            ),
            agent_capabilities=AgentCapabilities(
                auth=AgentAuthCapabilities(),
                prompt_capabilities=PromptCapabilities(
                    image=True,
                    audio=True,
                    embedded_context=True,
                ),
                session_capabilities=SessionCapabilities(
                    additional_directories=SessionAdditionalDirectoriesCapabilities(),
                    close=SessionCloseCapabilities(),
                    fork=SessionForkCapabilities(),
                    list=SessionListCapabilities(),
                    resume=SessionResumeCapabilities(),
                ),
                load_session=True,
                mcp_capabilities=McpCapabilities(http=True, sse=True),
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

        # Wire cwd + additional_directories to workspaces
        workspaces = [cwd] + (additional_directories or [])
        if hasattr(self._agent, "_config") and hasattr(
            self._agent._config, "workspaces"
        ):
            self._agent._config.workspaces = workspaces

        self._session_modes[session_id] = "agent"
        self._session_models[session_id] = _DEFAULT_MODEL_ID
        self._session_thinking_levels[session_id] = _DEFAULT_THINKING_LEVEL
        self._session_additional_dirs[session_id] = additional_directories or []
        converted = _convert_mcp_servers(mcp_servers)
        if converted:
            self._session_mcp_servers[session_id] = converted
        asyncio.ensure_future(self._send_available_commands(session_id))

        return NewSessionResponse(
            session_id=session_id,
            modes=SessionModeState(
                current_mode_id="agent",
                available_modes=[
                    SessionMode(
                        id="agent",
                        name="Agent",
                        description="Execute tools autonomously",
                    ),
                    SessionMode(
                        id="plan",
                        name="Plan",
                        description="Produce a plan without executing tools",
                    ),
                ],
            ),
            models=self._build_model_state(session_id),
            config_options=self._build_config_options(session_id),
        )

    async def _send_available_commands(self, session_id: str) -> None:
        await self._conn.session_update(
            session_id=session_id,
            update=update_available_commands(
                [
                    AvailableCommand(
                        name="reset", description="Clear conversation history"
                    ),
                    AvailableCommand(
                        name="clear", description="Clear conversation history"
                    ),
                    AvailableCommand(name="cost", description="Show session cost"),
                    AvailableCommand(name="usage", description="Show token usage"),
                    AvailableCommand(name="model", description="Show or switch model"),
                    AvailableCommand(
                        name="thinking", description="Show or switch thinking level"
                    ),
                    AvailableCommand(
                        name="compact",
                        description="Summarize conversation and start fresh context",
                    ),
                    AvailableCommand(
                        name="plan",
                        description="Generate an implementation plan",
                        input=UnstructuredCommandInput(hint="task description"),
                    ),
                    AvailableCommand(
                        name="help", description="Show available commands"
                    ),
                ]
            ),
        )

    async def _handle_command(self, text: str, session_id: str) -> str | None:
        """Handle slash commands. Returns response text, or None if not a command."""
        cmd = text.lstrip("/").split(None, 1)
        if not cmd:
            return None
        name, arg = cmd[0], cmd[1] if len(cmd) > 1 else ""

        if name in ("reset", "clear"):
            await self._rebuild_agent(session_id)
            self._session_titles.pop(session_id, None)
            return "Conversation reset."

        if name == "help":
            return (
                "Available commands:\n"
                "- `/reset` `/clear` — Clear conversation history\n"
                "- `/cost` — Show session cost\n"
                "- `/usage` — Show token usage from last turn\n"
                "- `/model [id]` — Show or switch model\n"
                "- `/thinking [level]` — Show or switch thinking level\n"
                "- `/compact` — Summarize conversation and start fresh context\n"
                "- `/help` — Show this message"
            )

        if name == "cost":
            model = self._session_models.get(session_id, _DEFAULT_MODEL_ID)
            cost = self._session_cumulative_cost.get(session_id, 0.0)
            return f"**Model:** {model}\n**Cumulative cost:** ${cost:.6f} USD"

        if name == "usage":
            usage = self._last_usage.get(session_id)
            if not usage:
                return "No usage data yet — send a prompt first."
            lines = [f"**Last turn:**"]
            for k, v in usage.items():
                if v is not None:
                    lines.append(f"- {k}: {v:,}")
            cost = self._session_cumulative_cost.get(session_id, 0.0)
            lines.append(f"\n**Cumulative cost:** ${cost:.6f} USD")
            return "\n".join(lines)

        if name == "model":
            if arg:
                if self._is_intellij:
                    return "Model switching via `/model` is not supported in IntelliJ — use the Model dropdown in the bottom bar instead."
                valid = {m.model_id for m in _AVAILABLE_MODELS}
                if arg not in valid:
                    return f"Unknown model `{arg}`. Available: {', '.join(sorted(valid))}"
                await self.set_config_option(
                    config_id="model", session_id=session_id, value=arg
                )
                return f"Switched to **{arg}**."
            current = self._session_models.get(session_id, _DEFAULT_MODEL_ID)
            models = "\n".join(
                f"- {'**' if m.model_id == current else ''}`{m.model_id}`{'**' if m.model_id == current else ''} — {m.name}"
                for m in _AVAILABLE_MODELS
            )
            return f"Current: **{current}**\n\n{models}"

        if name == "thinking":
            if arg:
                if self._is_intellij:
                    return "Thinking level switching via `/thinking` is not supported in IntelliJ — use the Thinking dropdown in the bottom bar instead."
                if arg not in _THINKING_LEVELS:
                    return f"Unknown level `{arg}`. Available: {', '.join(_THINKING_LEVELS)}"
                await self.set_config_option(
                    config_id="thinking_level", session_id=session_id, value=arg
                )
                return f"Thinking set to **{arg}**."
            current = self._session_thinking_levels.get(
                session_id, _DEFAULT_THINKING_LEVEL
            )
            return f"Current: **{current}**\nAvailable: {', '.join(_THINKING_LEVELS)}"

        if name == "compact":
            return "Compaction is handled automatically by the SDK when context exceeds the threshold. Use `/reset` to start fresh."

        return None

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
                    parts.append(
                        agy.types.Image(
                            data=base64.b64decode(data),
                            mime_type=mime,
                        )
                    )
                case AudioContentBlock(data=data, mime_type=mime):
                    parts.append(
                        agy.types.Audio(
                            data=base64.b64decode(data),
                            mime_type=mime,
                        )
                    )
                case EmbeddedResourceContentBlock(
                    resource=TextResourceContents(text=text)
                ):
                    parts.append(text)
                case EmbeddedResourceContentBlock(
                    resource=BlobResourceContents(blob=blob, mime_type=mime)
                ):
                    parts.append(
                        agy.types.Document(
                            data=base64.b64decode(blob),
                            mime_type=mime or "application/octet-stream",
                        )
                    )
                case ResourceContentBlock(uri=uri, name=name):
                    parts.append(f"[Attached resource: {name}]({uri})")
                case _:
                    log.debug("skipping unknown block: %s", type(block))

        if not parts:
            log.debug("no content to send")
            return PromptResponse(user_message_id=message_id, stop_reason="end_turn")

        first_text = next((p for p in parts if isinstance(p, str)), "")

        # /plan is special — it modifies content for agent.chat(), doesn't return early
        plan_cmd = first_text.strip().lstrip("/")
        if plan_cmd == "plan" or plan_cmd.startswith("plan "):
            task = plan_cmd.removeprefix("plan").strip()
            parts = [agy.types.SlashCommand(name=agy.types.BuiltinSlashCommandName.PLAN)]
            if task:
                parts.append(task)

        cmd_result = await self._handle_command(first_text.strip(), session_id)
        if cmd_result is not None:
            await self._conn.session_update(
                session_id=session_id,
                update=update_agent_message(text_block(cmd_result)),
            )
            return PromptResponse(user_message_id=message_id, stop_reason="end_turn")

        if self._session_modes.get(session_id) == "plan":
            parts.append(
                "\n[PLAN MODE: Produce a step-by-step plan. Do not execute any tools.]"
            )

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
                            session_id=session_id, update=update_agent_thought_text(t)
                        )

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
                            update=update_agent_message(text_block(t)),
                        )
                    case agy.types.ToolCall():
                        pass  # handled by PreToolCallDecideHook
                    case _:
                        log.debug("unhandled chunk type: %s", type(chunk))
        except asyncio.CancelledError:
            log.debug("prompt cancelled for session %s", session_id)
            if response is not None and hasattr(response, "cancel"):
                await response.cancel()
            terminal_id = self._last_terminal_ids.pop(session_id, None)
            if terminal_id:
                try:
                    await self._conn.release_terminal(
                        session_id=session_id, terminal_id=terminal_id
                    )
                except Exception:
                    log.debug("failed to release terminal %s on cancel", terminal_id)
            stop_reason = "cancelled"
        except Exception as e:
            log.exception("error during agent chat")
            await self._conn.session_update(
                session_id=session_id,
                update=update_agent_message(text_block(f"Error: {e}")),
            )
        finally:
            current_session_id.reset(token)
            self._active_session_id = None
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
                    self._last_usage[session_id] = {
                        "input_tokens": meta.prompt_token_count or 0,
                        "output_tokens": meta.candidates_token_count or 0,
                        "thought_tokens": meta.thoughts_token_count,
                        "cached_tokens": meta.cached_content_token_count,
                        "total_tokens": meta.total_token_count or 0,
                    }
                    used = meta.total_token_count or 0
                    model_id = self._session_models.get(session_id, _DEFAULT_MODEL_ID)
                    rates = _get_token_rates(model_id, meta.prompt_token_count or 0)
                    cost = None
                    if rates:
                        in_rate, out_rate = rates
                        turn_cost = (
                            (meta.prompt_token_count or 0) * in_rate
                            + (meta.candidates_token_count or 0) * out_rate
                        ) / 1_000_000
                        cumulative = (
                            self._session_cumulative_cost.get(session_id, 0.0)
                            + turn_cost
                        )
                        self._session_cumulative_cost[session_id] = cumulative
                        cost = Cost(amount=round(cumulative, 6), currency="USD")
                    await self._conn.session_update(
                        session_id=session_id,
                        update=UsageUpdate(
                            session_update="usage_update",
                            size=max(used, 1),
                            used=used,
                            cost=cost,
                        ),
                    )
        except Exception:
            log.debug("usage extraction failed", exc_info=True)

        self._store.save(
            session_id,
            {
                "session_id": session_id,
                "conversation_id": getattr(self._agent, "conversation_id", None),
                "cwd": getattr(self, "_cwd", "."),
                "mode": self._session_modes.get(session_id, "agent"),
                "model": self._session_models.get(session_id, _DEFAULT_MODEL_ID),
                "thinking_level": self._session_thinking_levels.get(
                    session_id, _DEFAULT_THINKING_LEVEL
                ),
                "title": self._session_titles.get(session_id),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        log.debug("returning PromptResponse stop_reason=%s", stop_reason)
        return PromptResponse(
            user_message_id=message_id,
            stop_reason=stop_reason,
            usage=usage,
        )


async def main() -> None:
    # use_unstable_protocol enables session/set_model and session/close RPCs
    # which are registered as unstable in the ACP SDK router.
    # The stable path for model switching is session/set_config_option with id="model".
    await run_agent(
        EchoAgent(agent_config_t=agy.LocalAgentConfig, agent_t=agy.Agent),
        use_unstable_protocol=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
