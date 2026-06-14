# agy-acp

ACP (Agent Client Protocol) adapter that wraps Google's Antigravity SDK for use in JetBrains IDEs and Zed.

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- `GEMINI_API_KEY` environment variable (get one from [AI Studio](https://aistudio.google.com/apikey))

## Setup

```bash
uv sync
```

## Running

The agent communicates over stdio using the ACP JSON-RPC protocol. To run standalone:

```bash
python hellp.py
```

### IntelliJ / JetBrains IDEs

Add the agent to `~/.jetbrains/acp.json`
([docs](https://www.jetbrains.com/help/ai-assistant/acp.html#add-custom-agent)):

```json
{
  "agent_servers": {
    "Antigravity": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/hellp.py"],
      "env": {
        "GEMINI_API_KEY": "your-key-here"
      }
    }
  }
}
```

The agent appears in the AI Chat tool window (look for the agent icon).

#### Enabling terminal support

Terminal/command execution is gated behind a registry flag in IntelliJ's ACP plugin (disabled by default in 2026.2 EAP builds):

1. **Help > Find Action** (Cmd+Shift+A) > type `Registry`
2. Search for `llm.chat.agent.acp.terminal.enabled`
3. Check the box
4. Restart the ACP session (or the IDE)

Without this, the IDE sends `terminal=False` in its client capabilities and the agent falls back to the SDK's native command execution (commands run outside the IDE terminal UI).

### Zed

Add the agent to your Zed settings ([docs](https://zed.dev/docs/ai/external-agents#custom-agents)):

**Settings > Extensions > Agent Servers**, or edit `~/.config/zed/settings.json`:

```json
{
  "agent": {
    "custom_agents": [
      {
        "id": "antigravity",
        "name": "Antigravity",
        "command": "/path/to/.venv/bin/python",
        "args": ["/path/to/hellp.py"],
        "env": {
          "GEMINI_API_KEY": "your-key-here"
        }
      }
    ]
  }
}
```

## IDE tools (IntelliJ MCP server)

IntelliJ exposes IDE tools (build, inspections, symbol info, refactoring, debugger, etc.) via a built-in [MCP server](https://www.jetbrains.com/help/idea/mcp-server.html). The recommended way to connect these to the agent is via streamable-http as a custom MCP server, **not** via `use_idea_mcp` in `acp.json`.

**Why not `use_idea_mcp`?** It passes the IDE's MCP server in router-only mode, exposing a single `execute_tool` wrapper with no tool schemas. The LLM struggles with this indirection. It also spawns a redundant `idea stdioMcpServer` process.

**Recommended setup:**

1. In IntelliJ, go to **Settings > Tools > MCP Server**
2. Under Manual Client Configuration, click **Copy HTTP Stream Config**
3. Paste into `.ai/mcp/mcp.json` in your project root (or create the file):

```json
{
  "mcpServers": {
    "idea": {
      "type": "streamable-http",
      "url": "http://127.0.0.1:<port>/stream",
      "headers": {
        "IJ_MCP_SERVER_PROJECT_PATH": "/path/to/your/project"
      }
    }
  }
}
```

4. In **Settings > AI Assistant > Agents**, ensure **Pass custom MCP servers** is checked

![IntelliJ MCP settings](docs/img/intellij-mcp-settings.svg)

This gives the agent individual tools with full schemas. Key tools ([full list](https://www.jetbrains.com/help/ai-assistant/mcp.html#supported-tools)):

| Category | Tools |
|----------|-------|
| Analysis | `build_project`, `get_file_problems`, `get_project_dependencies` |
| Code Insight | `get_symbol_info` |
| Refactoring | `rename_refactoring`, `reformat_file` |
| Search | `search_symbol`, `search_text`, `search_regex` |
| Execution | `execute_run_configuration`, `execute_terminal_command` |
| VCS | `get_repositories`, `git_status` |
| Debugger | via [`ij-debugger` skill](https://www.jetbrains.com/help/idea/mcp-server.html) (Find Action > "Copy Debugger Skill to Agents") |

Toggle individual tools on/off in **Settings > Tools > MCP Server > Exposed Tools**.

![IntelliJ MCP tools](docs/img/intellij-mcp-tools.svg)

## Diagnostics

### IntelliJ

With the MCP server configured (see above), the agent can call `build_project` for build errors and `get_file_problems` for inspections/warnings.

### Zed

External ACP agents [cannot access Zed's LSP diagnostics](https://github.com/zed-industries/zed/discussions/58546). Workaround: the agent can run linters via `run_command` (e.g. `tsc --noEmit`, `cargo check`).

For Go: [gopls v0.20+](https://go.dev/gopls/features/mcp) has built-in MCP mode with a `go_diagnostics` tool that can be configured as an MCP server.

## IDE context

Both IDEs can enrich prompts with editor state (open file, selection).

### IntelliJ

The "IDE context enabled" toggle in the chat bottom bar controls this. When enabled, IntelliJ adds two extra prompt blocks alongside your message:

- A **resource link** with the open file's URI (no file content)
- A **resource** with selection byte offsets (JSON, ~250 bytes)

The agent reads the file via `view_file` only if needed. No full file content is sent automatically.

![IDE context bar](docs/img/ide-context-bar.svg)

### Zed

Zed automatically includes active buffer context. Use `@` mentions to explicitly attach files, diagnostics, or symbols.

## Modes

The agent supports 5 permission modes that control how tool calls are handled:

![Mode dropdown](docs/img/mode-dropdown.svg)

| Mode | Read tools | File writes | Commands / MCP | Notes |
|------|-----------|-------------|----------------|-------|
| **Agent** (default) | auto-allow | prompt | prompt | Standard behavior |
| **Accept Edits** | auto-allow | auto-allow | prompt | Auto-accepts file changes |
| **Plan** | auto-allow | deny | prompt | File writes disabled, exploration OK |
| **Don't Ask** | auto-allow | deny | deny | Silently denies non-safe tools |
| **Bypass** | auto-allow | auto-allow | auto-allow | No permission checks |

Switch modes via the Mode dropdown in the IDE, or via `set_session_mode` / `set_config_option` RPCs.

## IntelliJ-specific behavior

The agent detects IntelliJ via `client_info.name` containing "JetBrains" and adjusts:

- `/model` and `/thinking` slash commands tell IntelliJ users to use the IDE dropdown instead (IntelliJ has a config feedback loop that overwrites agent-initiated changes)
- The config feedback loop causes unnecessary agent rebuilds on each prompt — this is a known IntelliJ ACP plugin issue

## Testing

```bash
# Offline tests (no API key needed)
python -m pytest hellp_test.py -k 'not test_initializes and not test_live_run'

# All tests (requires GEMINI_API_KEY)
python -m pytest hellp_test.py
```

## Architecture

`EchoAgent` extends `acp.Agent` and wraps `google.antigravity.Agent`:

```
IDE (IntelliJ) <--ACP JSON-RPC--> EchoAgent <---> Antigravity SDK <---> Gemini API
                                      |
                                      +-- view_file/create_file/edit_file --> IDE RPCs
                                      +-- run_command --> IDE terminal (if supported)
                                      +-- PreToolCallDecideHook --> permission broker
```

- **File I/O** is routed through IDE RPCs (`read_text_file`, `write_text_file`) when the client supports it, otherwise falls back to the SDK's built-in tools.
- **Command execution** goes through the IDE terminal when `client_capabilities.terminal=True`, otherwise the SDK's native `run_command` handles it.
- **Permission gating** is mode-dependent: read-only tools always auto-allow; file writes and command execution behavior depends on the active mode (see below).

## Features

- **Models**: Gemini 3.5 Flash (default), 3.1 Pro, 2.5 Pro/Flash/Flash-Lite, and more
- **Thinking** (`thinking_level`): Minimal/Low/Medium/High (3.x models only)
- **Modes**: Agent (default, prompts for writes/commands), Accept Edits (auto-allows file edits), Plan (read-only, no file writes), Don't Ask (deny non-safe silently), Bypass (allow everything)
- **Sessions**: Create, list, load, fork, resume with conversation persistence
- **MCP servers**: HTTP, SSE, and stdio transports (with env variable workaround)
- **Cost tracking**: Per-turn and cumulative USD estimates with long-context surcharge
- **Context retention**: Compact (25k), Normal (50k), Extended (200k), Max (1M) token thresholds
- **Slash commands**: `/reset`, `/clear`, `/cost`, `/usage`, `/model [id]`, `/thinking [level]`, `/context [level]`, `/compact`, `/help`
- **Authentication**: `GEMINI_API_KEY` env var via ACP auth flow

## Files

| File | Description |
|------|-------------|
| `hellp.py` | Main ACP adapter — `EchoAgent` and hook implementations |
| `hellp_test.py` | Test suite (offline + live tests) |
| `fake_server.py` | Fake agent server for subprocess integration tests |
| `hello.py` | Standalone Antigravity SDK example (no ACP) |
