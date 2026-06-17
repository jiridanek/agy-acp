# agy_acp Package Architecture

**Keep this file up to date.** When you add, remove, rename, or change the responsibility of a module, update this document in the same commit.

## Overview

ACP agent adapter wrapping Google's Antigravity SDK. Receives ACP JSON-RPC over stdio from JetBrains IDEs or Zed, delegates to a per-session Antigravity `Agent` for LLM interaction.

## Module Map

```
__main__.py  ─── entry point (asyncio.run, signal handling, cleanup)
    │
agent.py  ─── EchoAgent(acp.Agent): the ACP protocol implementation
    │         session lifecycle, config options, slash commands, prompt streaming
    │         ~685 lines, largest module — resist splitting further unless a
    │         clear seam emerges (e.g. slash commands becoming a plugin system)
    │
    ├── hooks.py  ─── MyPreToolCallDecideHook, MyPostToolCallHook
    │                  intercepts SDK tool calls, enforces permission modes,
    │                  sends ACP tool_call start/progress notifications
    │
    ├── session.py  ─── SessionState (Pydantic), SessionStore (JSON), Session (dataclass)
    │                    current_session_id ContextVar, _check_trajectory, _ensure_dir
    │
    ├── config.py  ─── pure constants: models, pricing, thinking levels, context presets,
    │                   tool sets (_ALWAYS_SAFE_TOOLS, _FILE_WRITE_TOOLS), modes
    │
    ├── tool_ui.py  ─── formatting for tool call display and permission prompts
    │                    _tool_title, _tool_kind, _permission_description, _permission_content
    │
    ├── tools.py  ─── small helpers: _parse_plan_entries, _build_mode_state, _get_token_rates
    │
    ├── mcp.py  ─── ACP→SDK MCP server conversion (_convert_mcp_server)
    │               _parse_mcp_request_text (extracts ServerName/ToolName from SDK args)
    │
    ├── skills.py  ─── skill/command discovery from .gemini/ and .agents/ directories
    │                   _setup_external_skills (temp symlink dir for IntelliJ skills)
    │
    └── log.py  ─── SecretMaskingFilter, logger setup, _log_prompt_blocks, _log_mcp_servers
```

## Import Dependency Order (no cycles)

```
log.py, config.py          ← leaf modules (stdlib only)
session.py                 ← config, log
tools.py                   ← config, acp helpers
mcp.py                     ← agy.types
tool_ui.py                 ← mcp
skills.py                  ← stdlib, acp schema
hooks.py                   ← config, session, tool_ui, log
agent.py                   ← everything above
__main__.py                ← agent
```

Adding a new module: place it in this graph without creating cycles. If module A imports from module B, A must be below B in this list.

## Key Design Decisions

- **One EchoAgent, many Sessions.** Each ACP session owns a `Session` dataclass with its own Antigravity `Agent` instance and Go harness subprocess. Sessions don't share state.
- **Hooks, not subclassing.** Permission enforcement and tool call tracking use the Antigravity SDK's hook system (`PreToolCallDecideHook`, `PostToolCallHook`), not ACP-level middleware.
- **IDE routing via closures.** File I/O and terminal tools are closures created in `initialize()` that capture `self` and delegate to the IDE via ACP client RPCs. This avoids deep-copy issues with bound methods.
- **Shim at root.** `hellp.py` re-exports package symbols so existing `~/.jetbrains/acp.json` configs keep working without path changes.

## Test Layout

Tests live in `tests/` (pytest auto-discovers via `testpaths` in pyproject.toml):
- `test_agent.py` — all EchoAgent tests (to be split further as the suite grows)
- `test_logging.py` — SecretMaskingFilter tests
- `conftest.py` — adds project root to sys.path for `import hellp` backward compat
- `evals/` — evaluation data fixtures
