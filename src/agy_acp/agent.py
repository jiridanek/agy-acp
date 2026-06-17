import asyncio
import base64
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import google.antigravity as agy
from google.antigravity.hooks import policy as agy_policy

from acp import (
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    text_block,
    update_agent_message,
    update_agent_thought_text,
)
from acp.contrib.tool_calls import ToolCallTracker
from acp.helpers import (
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
    PromptCapabilities,
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
    SessionModelState,
    SessionResumeCapabilities,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
    SetSessionModeResponse,
    SseMcpServer,
    TextContentBlock,
    TextResourceContents,
    UnstructuredCommandInput,
    Usage,
    UsageUpdate,
)

from agy_acp.config import (
    _AVAILABLE_MODES,
    _AVAILABLE_MODELS,
    _CONTEXT_PRESETS,
    _DEFAULT_CONTEXT,
    _INTELLIJ_EXTERNAL_SKILLS,
    _THINKING_LEVELS,
)
from agy_acp.hooks import MyPostToolCallHook, MyPreToolCallDecideHook
from agy_acp.log import _log_mcp_servers, _log_prompt_blocks, log
from agy_acp.mcp import _convert_mcp_servers
from agy_acp.session import (
    Session,
    SessionState,
    SessionStore,
    _DEFAULT_SAVE_DIR,
    _check_trajectory,
    _ensure_dir,
    current_session_id,
)
from agy_acp.skills import _discover_skills, _setup_external_skills, _skills_paths
from agy_acp.tools import _build_mode_state, _get_token_rates, _parse_plan_entries


class EchoAgent(Agent):
    _conn: Client

    def __init__(self, agent_t, agent_config_t, store: SessionStore | None = None):
        self._agent_t = agent_t
        self._agent_config_t = agent_config_t
        self._store = store or SessionStore()
        self._tracker = ToolCallTracker()
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._sessions: dict[str, Session] = {}
        self._active_session_id: str | None = None
        self._client_capabilities: ClientCapabilities | None = None
        self._client_info: Implementation | None = None
        self._external_skills_dir: str | None = None

    def _get_session(self, session_id: str) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise ValueError(f"Unknown session: {session_id}")

    def _hooks(self) -> list:
        return [MyPreToolCallDecideHook(self), MyPostToolCallHook(self)]

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
        session = self._sessions.pop(session_id, None)
        if session:
            await session.close()
        self._active_tasks.pop(session_id, None)
        self._store.delete(session_id)
        return CloseSessionResponse()

    async def set_session_mode(
        self, mode_id: str, session_id: str, **kwargs: Any
    ) -> SetSessionModeResponse:
        log.debug("set_session_mode mode=%s session=%s", mode_id, session_id)
        self._get_session(session_id).state.mode = mode_id
        await self._conn.session_update(
            session_id=session_id,
            update=CurrentModeUpdate(
                session_update="current_mode_update",
                current_mode_id=mode_id,
            ),
        )
        return SetSessionModeResponse()

    def _build_config_options(self, session_id: str) -> list[SessionConfigOptionSelect]:
        s = self._get_session(session_id).state
        current_mode = s.mode
        current_model = s.model
        current_thinking = s.thinking_level
        current_context = s.context_level
        return [
            SessionConfigOptionSelect(
                id="mode",
                name="Mode",
                type="select",
                description="Controls agent behavior",
                category="mode",
                current_value=current_mode,
                options=[
                    SessionConfigSelectOption(value=m.id, name=m.name, description=m.description)
                    for m in _AVAILABLE_MODES
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
            SessionConfigOptionSelect(
                id="context",
                name="Context",
                type="select",
                description="Conversation history retained before compacting",
                category="model",
                current_value=current_context,
                options=[
                    SessionConfigSelectOption(
                        value=k,
                        name=k.capitalize(),
                        description=f"{v:,} tokens",
                    )
                    for k, v in _CONTEXT_PRESETS.items()
                ],
            ),
        ]

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
        session = self._get_session(session_id)
        s = session.state
        current = {
            "mode": s.mode,
            "model": s.model,
            "thinking_level": s.thinking_level,
            "context": s.context_level,
        }
        if isinstance(value, str) and current.get(config_id) == value:
            log.debug("set_config_option: value unchanged, skipping")
            return SetSessionConfigOptionResponse(
                config_options=self._build_config_options(session_id)
            )
        if config_id == "mode" and isinstance(value, str):
            s.mode = value
            await self._conn.session_update(
                session_id=session_id,
                update=CurrentModeUpdate(
                    session_update="current_mode_update",
                    current_mode_id=value,
                ),
            )
        elif config_id == "model" and isinstance(value, str):
            s.model = value
            await self._rebuild_agent(
                session_id,
                conversation_id=getattr(session.agent, "conversation_id", None),
            )
        elif config_id == "thinking_level" and isinstance(value, str):
            s.thinking_level = value
            await self._rebuild_agent(
                session_id,
                conversation_id=getattr(session.agent, "conversation_id", None),
            )
        elif config_id == "context" and isinstance(value, str):
            s.context_level = value
            await self._rebuild_agent(
                session_id,
                conversation_id=getattr(session.agent, "conversation_id", None),
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
        current = self._get_session(session_id).state.model
        return SessionModelState(
            current_model_id=current,
            available_models=_AVAILABLE_MODELS,
        )

    def _build_agent_config(
        self,
        session: Session,
        conversation_id: str | None = None,
        save_dir: str | None = None,
    ):
        """Build an Antigravity SDK config from session state."""
        s = session.state
        compaction_threshold = _CONTEXT_PRESETS.get(s.context_level, 50_000)
        enabled_tools, custom_tools = self._build_tools_config()
        cwd = s.cwd
        return self._agent_config_t(
            capabilities=agy.types.CapabilitiesConfig(
                enabled_tools=enabled_tools,
                compaction_threshold=compaction_threshold,
            ),
            policies=[agy_policy.allow_all()],
            tools=custom_tools,
            gemini_config=agy.types.GeminiConfig(
                models=agy.types.ModelConfig(
                    default=agy.types.ModelEntry(
                        name=s.model,
                        generation=agy.types.GenerationConfig(
                            thinking_level=agy.types.ThinkingLevel(s.thinking_level)
                            if not s.model.startswith("gemini-2.")
                            else None,
                        ),
                    ),
                ),
            ),
            conversation_id=conversation_id,
            save_dir=_ensure_dir(save_dir or _DEFAULT_SAVE_DIR),
            workspaces=[cwd] + session.additional_dirs,
            mcp_servers=_convert_mcp_servers(session.mcp_servers_raw or None),
            skills_paths=_skills_paths(cwd)
            + ([self._external_skills_dir] if self._external_skills_dir else []),
        )

    async def _rebuild_agent(
        self,
        session_id: str,
        conversation_id: str | None = None,
        save_dir: str | None = None,
    ) -> None:
        """Tear down and recreate the session's agent with current model/thinking settings."""
        session = self._get_session(session_id)
        old_agent = session.agent
        if old_agent:
            await old_agent.__aexit__(None, None, None)

        try:
            config = self._build_agent_config(session, conversation_id, save_dir)
            new_agent = self._agent_t(config)
            for hook in self._hooks():
                new_agent.register_hook(hook)
            await new_agent.__aenter__()
            session.agent = new_agent
        except Exception:
            log.exception("_rebuild_agent failed, restoring previous agent")
            if old_agent:
                await old_agent.__aenter__()
            session.agent = old_agent
            raise
        log.debug(
            "rebuilt agent model=%s thinking=%s conv=%s",
            session.state.model,
            session.state.thinking_level,
            conversation_id,
        )

    async def set_session_model(
        self,
        model_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> SetSessionModelResponse | None:
        log.debug("set_session_model model=%s session=%s", model_id, session_id)
        session = self._get_session(session_id)
        session.state.model = model_id
        await self._rebuild_agent(
            session_id, conversation_id=getattr(session.agent, "conversation_id", None)
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
        parent = self._get_session(session_id)
        ps = parent.state
        title = f"{ps.title} (fork)" if ps.title else None

        new_state = SessionState(
            session_id=new_id,
            cwd=cwd,
            mode=ps.mode,
            model=ps.model,
            thinking_level=ps.thinking_level,
            context_level=ps.context_level,
            title=title,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        new_dirs = additional_directories or list(parent.additional_dirs)
        new_mcp = list(mcp_servers) if mcp_servers else list(parent.mcp_servers_raw)

        new_session = Session(
            state=new_state,
            additional_dirs=new_dirs,
            mcp_servers_raw=new_mcp,
        )
        config = self._build_agent_config(new_session)
        await new_session.start_agent(self._agent_t, config, self._hooks())
        self._sessions[new_id] = new_session

        self._store.save(new_id, new_state)

        asyncio.ensure_future(self._send_available_commands(new_id))

        return ForkSessionResponse(
            session_id=new_id,
            modes=_build_mode_state(new_state.mode),
            models=self._build_model_state(new_id),
            config_options=self._build_config_options(new_id),
        )

    async def _restore_session(
        self,
        stored: SessionState,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None,
    ) -> Session:
        stored.cwd = cwd
        conv_id = _check_trajectory(stored.conversation_id)
        if not conv_id:
            stored.conversation_id = None
        session = Session(
            state=stored,
            additional_dirs=additional_directories or [],
            mcp_servers_raw=list(mcp_servers) if mcp_servers else [],
        )
        config = self._build_agent_config(session, conversation_id=conv_id)
        await session.start_agent(self._agent_t, config, self._hooks())
        self._sessions[session_id] = session
        if conv_id:
            log.debug("restoring conversation %s for session %s", conv_id, session_id)
        asyncio.ensure_future(self._send_available_commands(session_id))
        return session

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
        session = await self._restore_session(stored, cwd, session_id, additional_directories, mcp_servers)
        return ResumeSessionResponse(
            modes=_build_mode_state(session.state.mode),
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
                    session_id=s.session_id,
                    cwd=s.cwd,
                    title=s.title,
                    updated_at=s.updated_at,
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
        log.debug("load_session %s", session_id)
        stored = self._store.load(session_id)
        if not stored:
            return None
        session = await self._restore_session(stored, cwd, session_id, additional_directories, mcp_servers)
        return LoadSessionResponse(
            modes=_build_mode_state(session.state.mode),
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
        """Build enabled_tools (builtin allowlist) and custom tools based on client capabilities."""
        can_read, can_write, can_terminal = self._check_client_caps()
        enabled = [
            agy.types.BuiltinTools.LIST_DIR,
            agy.types.BuiltinTools.FIND_FILE,
            agy.types.BuiltinTools.SEARCH_DIR,
            agy.types.BuiltinTools.ASK_QUESTION,
            agy.types.BuiltinTools.FINISH,
            agy.types.BuiltinTools.START_SUBAGENT,
            agy.types.BuiltinTools.GENERATE_IMAGE,
        ]
        custom_tools = []

        if can_read:
            custom_tools.append(self.view_file)
        else:
            enabled.append(agy.types.BuiltinTools.VIEW_FILE)

        if can_write:
            custom_tools.extend([self.create_file, self.edit_file])
        else:
            enabled.append(agy.types.BuiltinTools.CREATE_FILE)
            enabled.append(agy.types.BuiltinTools.EDIT_FILE)

        if can_terminal:
            custom_tools.append(self.run_command)
        else:
            enabled.append(agy.types.BuiltinTools.RUN_COMMAND)

        return enabled, custom_tools

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

        if self._is_intellij and self._external_skills_dir is None:
            self._external_skills_dir = _setup_external_skills(_INTELLIJ_EXTERNAL_SKILLS)

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
                session = agent_ref._sessions.get(sid)
                if session:
                    session.last_file_edits[path] = {
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
                session = agent_ref._sessions.get(sid)
                if session:
                    session.last_file_edits[path] = {
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
                session = agent_ref._sessions.get(sid)
                if session:
                    session.last_terminal_id = terminal_id
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
                if exit_code is not None and session:
                    session.last_exit_code = exit_code
                return out_resp.output
            except Exception as e:
                return f"Error: Failed to run command '{command}': {e}"

        self.view_file = view_file
        self.create_file = create_file
        self.edit_file = edit_file
        self.run_command = run_command

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

        session_id = uuid4().hex

        _log_mcp_servers("new_session", mcp_servers)
        session = Session(
            state=SessionState(session_id=session_id, cwd=cwd),
            additional_dirs=additional_directories or [],
            mcp_servers_raw=list(mcp_servers) if mcp_servers else [],
        )
        config = self._build_agent_config(session)
        await session.start_agent(self._agent_t, config, self._hooks())
        self._sessions[session_id] = session
        asyncio.ensure_future(self._send_available_commands(session_id))

        return NewSessionResponse(
            session_id=session_id,
            modes=_build_mode_state("agent"),
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
                        name="context", description="Show or set context retention level"
                    ),
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
                + _discover_skills(
                    self._sessions[session_id].state.cwd if session_id in self._sessions else ".",
                    extra_skills=_INTELLIJ_EXTERNAL_SKILLS if self._is_intellij else None,
                )

            ),
        )

    async def _handle_command(self, text: str, session_id: str) -> str | None:
        """Handle slash commands. Returns response text, or None if not a command."""
        cmd = text.lstrip("/").split(None, 1)
        if not cmd:
            return None
        name, arg = cmd[0], cmd[1] if len(cmd) > 1 else ""

        session = self._get_session(session_id)

        if name in ("reset", "clear"):
            await self._rebuild_agent(session_id)
            session.state.title = None
            return "Conversation reset."

        if name == "help":
            return (
                "Available commands:\n"
                "- `/reset` `/clear` — Clear conversation history\n"
                "- `/cost` — Show session cost\n"
                "- `/usage` — Show token usage from last turn\n"
                "- `/model [id]` — Show or switch model\n"
                "- `/thinking [level]` — Show or switch thinking level\n"
                "- `/context [level]` — Show or set context retention (compact/normal/extended/max)\n"
                "- `/compact` — Info about automatic compaction\n"
                "- `/help` — Show this message"
            )

        if name == "cost":
            model = session.state.model
            return f"**Model:** {model}\n**Cumulative cost:** ${session.cumulative_cost:.6f} USD"

        if name == "usage":
            usage = session.last_usage
            if not usage:
                return "No usage data yet — send a prompt first."
            lines = [f"**Last turn:**"]
            for k, v in usage.items():
                if v is not None:
                    lines.append(f"- {k}: {v:,}")
            lines.append(f"\n**Cumulative cost:** ${session.cumulative_cost:.6f} USD")
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
            current = session.state.model
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
            current = session.state.thinking_level
            return f"Current: **{current}**\nAvailable: {', '.join(_THINKING_LEVELS)}"

        if name == "context":
            if arg:
                if arg not in _CONTEXT_PRESETS:
                    return f"Unknown level `{arg}`. Available: {', '.join(_CONTEXT_PRESETS)}"
                if self._is_intellij:
                    session.state.context_level = arg
                    await self._rebuild_agent(
                        session_id,
                        conversation_id=getattr(session.agent, "conversation_id", None),
                    )
                else:
                    await self.set_config_option(
                        config_id="context", session_id=session_id, value=arg
                    )
                return f"Context set to **{arg}** ({_CONTEXT_PRESETS[arg]:,} tokens)."
            current = session.state.context_level
            levels = ", ".join(
                f"**{k}** ({v:,})" if k == current else f"{k} ({v:,})"
                for k, v in _CONTEXT_PRESETS.items()
            )
            return f"Current: **{current}** ({_CONTEXT_PRESETS[current]:,} tokens)\n{levels}"

        if name == "compact":
            return "Compaction is automatic. Use `/context` to change the threshold, or `/reset` to start fresh."

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
        log.debug("prompt session=%s", session_id)
        _log_prompt_blocks(prompt)

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

        session = self._get_session(session_id)

        if session.state.mode == "plan":
            parts.append(
                "\n[PLAN MODE: Produce a step-by-step plan. Do not execute any tools.]"
            )

        if not session.state.title:
            first_text = next((p for p in parts if isinstance(p, str)), None)
            if first_text:
                title = first_text[:80].split("\n")[0]
                session.state.title = title
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

        self._active_session_id = session_id
        token = current_session_id.set(session_id)
        try:
            response = await session.agent.chat(parts)
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
                        pass
                    case _:
                        log.debug("unhandled chunk type: %s", type(chunk))
        except asyncio.CancelledError:
            log.debug("prompt cancelled for session %s", session_id)
            if response is not None and hasattr(response, "cancel"):
                await response.cancel()
            terminal_id = session.last_terminal_id
            session.last_terminal_id = None
            if terminal_id:
                try:
                    await self._conn.release_terminal(
                        session_id=session_id, terminal_id=terminal_id
                    )
                except Exception:
                    log.debug("failed to release terminal %s on cancel", terminal_id)
            stop_reason = "cancelled"
        except Exception as e:
            log.exception("error during agent chat: %s", e)
            try:
                await self._conn.session_update(
                    session_id=session_id,
                    update=update_agent_message(text_block(f"Error: {e}")),
                )
            except Exception:
                log.debug("failed to send error update to client")
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
                    session.last_usage = {
                        "input_tokens": meta.prompt_token_count or 0,
                        "output_tokens": meta.candidates_token_count or 0,
                        "thought_tokens": meta.thoughts_token_count,
                        "cached_tokens": meta.cached_content_token_count,
                        "total_tokens": meta.total_token_count or 0,
                    }
                    used = meta.total_token_count or 0
                    model_id = session.state.model
                    rates = _get_token_rates(model_id, meta.prompt_token_count or 0)
                    cost = None
                    if rates:
                        in_rate, out_rate = rates
                        turn_cost = (
                            (meta.prompt_token_count or 0) * in_rate
                            + (meta.candidates_token_count or 0) * out_rate
                        ) / 1_000_000
                        session.cumulative_cost += turn_cost
                        cost = Cost(amount=round(session.cumulative_cost, 6), currency="USD")
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

        s = session.state
        s.conversation_id = getattr(session.agent, "conversation_id", None)
        s.updated_at = datetime.now(timezone.utc).isoformat()
        self._store.save(session_id, s)

        log.debug("returning PromptResponse stop_reason=%s", stop_reason)
        return PromptResponse(
            user_message_id=message_id,
            stop_reason=stop_reason,
            usage=usage,
        )
