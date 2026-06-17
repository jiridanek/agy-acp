import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

import google.antigravity as agy
from acp.schema import HttpMcpServer, McpServerStdio, SseMcpServer
from pydantic import BaseModel

from agy_acp.config import _DEFAULT_CONTEXT, _DEFAULT_MODEL_ID, _DEFAULT_THINKING_LEVEL
from agy_acp.log import log

current_session_id = ContextVar("current_session_id")

_DEFAULT_STORE_PATH = Path.home() / ".agy-acp" / "sessions.json"
_DEFAULT_SAVE_DIR = str(Path.home() / ".agy-acp" / "trajectories")


def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _check_trajectory(conversation_id: str | None) -> str | None:
    """Return conversation_id only if its trajectory file exists on disk."""
    if not conversation_id:
        return None
    traj_path = Path(_DEFAULT_SAVE_DIR) / f"traj-{conversation_id}"
    if traj_path.exists():
        return conversation_id
    log.warning("trajectory %s not found, starting fresh", conversation_id)
    return None


class SessionState(BaseModel):
    session_id: str
    conversation_id: str | None = None
    cwd: str = "."
    mode: str = "agent"
    model: str = _DEFAULT_MODEL_ID
    thinking_level: str = _DEFAULT_THINKING_LEVEL
    context_level: str = _DEFAULT_CONTEXT
    title: str | None = None
    updated_at: str | None = None


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

    def save(self, session_id: str, state: SessionState) -> None:
        data = self._read()
        data[session_id] = state.model_dump()
        self._write(data)

    def list(self, cwd: str | None = None) -> list[SessionState]:
        data = self._read()
        sessions = [SessionState.model_validate(v) for v in data.values()]
        if cwd:
            sessions = [s for s in sessions if s.cwd == cwd]
        sessions.sort(key=lambda s: s.updated_at or "", reverse=True)
        return sessions

    def load(self, session_id: str) -> SessionState | None:
        raw = self._read().get(session_id)
        if raw is None:
            return None
        return SessionState.model_validate(raw)

    def delete(self, session_id: str) -> None:
        data = self._read()
        data.pop(session_id, None)
        self._write(data)


@dataclass
class Session:
    state: SessionState
    agent: agy.Agent | None = None
    mcp_servers_raw: list[HttpMcpServer | SseMcpServer | McpServerStdio] = field(default_factory=list)
    additional_dirs: list[str] = field(default_factory=list)
    cumulative_cost: float = 0.0
    last_file_edits: dict[str, dict[str, str | None]] = field(default_factory=dict)
    last_terminal_id: str | None = None
    last_exit_code: int | None = None
    last_usage: dict[str, int | None] = field(default_factory=dict)

    async def start_agent(self, agent_t, config, hooks: list | None = None):
        self.agent = agent_t(config)
        for hook in hooks or []:
            self.agent.register_hook(hook)
        await self.agent.__aenter__()

    async def close(self):
        if self.agent:
            await self.agent.__aexit__(None, None, None)
            self.agent = None
