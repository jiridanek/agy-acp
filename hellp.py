"""Backward-compatible entry point for existing ACP agent configs."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from agy_acp.agent import EchoAgent  # noqa: E402
from agy_acp.hooks import MyPostToolCallHook, MyPreToolCallDecideHook  # noqa: E402
from agy_acp.log import SecretMaskingFilter, log  # noqa: E402
from agy_acp.session import Session, SessionState, SessionStore, current_session_id  # noqa: E402
from agy_acp.tool_ui import _permission_content, _permission_description, _tool_kind, _tool_title  # noqa: E402
from agy_acp.tools import _build_mode_state, _get_token_rates, _parse_plan_entries  # noqa: E402
from agy_acp.mcp import _convert_mcp_server, _convert_mcp_servers, _parse_mcp_request_text  # noqa: E402
from agy_acp.skills import _discover_skills, _setup_external_skills, _skills_paths  # noqa: E402
from agy_acp.config import (  # noqa: E402
    _ALWAYS_SAFE_TOOLS,
    _AVAILABLE_MODELS,
    _AVAILABLE_MODES,
    _CONTEXT_PRESETS,
    _FILE_WRITE_TOOLS,
    _INTELLIJ_EXTERNAL_SKILLS,
    _MODEL_PRICING,
    _THINKING_LEVELS,
)

__all__ = [
    "EchoAgent",
    "MyPreToolCallDecideHook",
    "MyPostToolCallHook",
    "SecretMaskingFilter",
    "Session",
    "SessionState",
    "SessionStore",
    "current_session_id",
    "log",
]

if __name__ == "__main__":
    from agy_acp.__main__ import main
    main()
