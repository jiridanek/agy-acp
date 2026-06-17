from agy_acp.tool_ui import _permission_description, _tool_title


def test_tool_title_command_line():
    """_tool_title extracts command_line (SDK built-in run_command key)."""
    assert (
        _tool_title(
            "run_command", {"command_line": "git status", "working_dir": "/tmp"}
        )
        == "run_command: git status"
    )


def test_tool_title_command():
    """_tool_title extracts command (our custom closure key)."""
    assert _tool_title("run_command", {"command": "ls -la"}) == "run_command: ls -la"


def test_tool_title_mcp():
    """_tool_title extracts ServerName/ToolName from MCP request_text."""
    args = {
        "request_text": "Requesting permission with args "
        '{"Arguments": {}, "ServerName": "github", "ToolName": "get_me", '
        '"toolAction": "Call get_me on github", "toolSummary": "Calling get_me MCP tool"}'
    }
    assert _tool_title("mcp_github_get_me", args) == "get_me (github)"


def test_permission_description_run_command():
    """_permission_description shows working dir for run_command (command is in title)."""
    desc = _permission_description(
        "run_command", {"command_line": "git status", "working_dir": "/project"}
    )
    assert desc == "in `/project`"


def test_permission_description_mcp_tool_with_args():
    """_permission_description shows MCP tool arguments."""
    args = {
        "request_text": "Requesting permission with args "
        '{"Arguments": {"owner": "google", "repo": "antigravity"}, '
        '"ServerName": "github", "ToolName": "get_repo"}'
    }
    desc = _permission_description("mcp_github_get_repo", args)
    assert "**owner**" in desc
    assert "`google`" in desc


def test_permission_description_mcp_tool_no_args():
    """_permission_description shows 'no arguments' for argless MCP tools."""
    args = {
        "request_text": "Requesting permission with args "
        '{"Arguments": {}, "ServerName": "github", "ToolName": "get_me"}'
    }
    desc = _permission_description("mcp_github_get_me", args)
    assert "no arguments" in desc


def test_permission_description_generic():
    """_permission_description lists args as markdown for unknown tools."""
    desc = _permission_description("some_tool", {"foo": "bar"})
    assert "**foo**" in desc
    assert "`bar`" in desc
