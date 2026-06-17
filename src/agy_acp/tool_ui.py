import json
from typing import Any

from acp import text_block
from acp.helpers import tool_content, tool_diff_content

from agy_acp.mcp import _parse_mcp_request_text


def _tool_title(name: str, args: Any) -> str:
    n = str(name)
    if isinstance(args, dict):
        for key in ("path", "command", "command_line", "query", "pattern", "directory"):
            if key in args:
                return f"{n}: {args[key]}"
        if n.startswith("mcp_"):
            server, tool = _parse_mcp_request_text(args)
            if server and tool:
                return f"{tool} ({server})"
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


def _permission_description(name: str, args: Any) -> str:
    """Build a human-readable description of tool arguments for the permission dialog."""
    if not isinstance(args, dict):
        return ""

    display_args = {k: v for k, v in args.items() if k != "request_text"}

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

    if display_args.get("working_dir"):
        return f"in `{display_args['working_dir']}`"

    if display_args:
        return "\n".join(f"- **{k}**: `{v}`" for k, v in display_args.items())

    return ""


def _permission_content(tool_name: str, args: Any) -> list | None:
    if not isinstance(args, dict):
        return None
    if tool_name == "edit_file":
        path = args.get("path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        return [tool_diff_content(path=path, new_text=new_string, old_text=old_string)]
    if tool_name == "create_file":
        path = args.get("path", "")
        content = args.get("content", "")
        return [tool_diff_content(path=path, new_text=content)]
    if tool_name == "run_command":
        command = args.get("command", "")
        return [tool_content(text_block(f"```\n{command}\n```"))]
    return None
