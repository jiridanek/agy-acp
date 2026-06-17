import json
import os
import sys
import tempfile

import google.antigravity as agy
from acp.schema import HttpMcpServer, McpServerStdio, SseMcpServer


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


def _convert_mcp_server(server: HttpMcpServer | SseMcpServer | McpServerStdio):
    """Convert ACP MCP server config to Antigravity SDK type."""
    if isinstance(server, (HttpMcpServer, SseMcpServer)):
        headers = {h.name: h.value for h in server.headers} if server.headers else {}
        return agy.types.McpStreamableHttpServer(
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
        return agy.types.McpStdioServer(
            name=server.name,
            command=sys.executable,
            args=["-ISs", "-c", loader],
            type="stdio",
        )

    return agy.types.McpStdioServer(
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
