# agy-acp

ACP (Agent Client Protocol) adapter that wraps Google's Antigravity SDK for use in JetBrains IDEs.

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
