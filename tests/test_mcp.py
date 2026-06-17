import sys

from agy_acp.mcp import _convert_mcp_server, _convert_mcp_servers


def test_convert_mcp_http_server():
    """HttpMcpServer converts to McpStreamableHttpServer."""
    from acp.schema import HttpHeader, HttpMcpServer
    from google.antigravity.types import McpStreamableHttpServer

    server = HttpMcpServer(
        type="http",
        name="test-http",
        url="http://localhost:8080",
        headers=[HttpHeader(name="Authorization", value="Bearer tok")],
    )
    result = _convert_mcp_server(server)
    assert isinstance(result, McpStreamableHttpServer)
    assert result.name == "test-http"
    assert result.url == "http://localhost:8080"
    assert result.headers == {"Authorization": "Bearer tok"}


def test_convert_mcp_stdio_server():
    """McpServerStdio without env converts directly."""
    from acp.schema import McpServerStdio
    from google.antigravity.types import McpStdioServer

    server = McpServerStdio(
        name="test-stdio", command="node", args=["server.js"], env=[]
    )
    result = _convert_mcp_server(server)
    assert isinstance(result, McpStdioServer)
    assert result.command == "node"
    assert result.args == ["server.js"]


def test_convert_mcp_stdio_server_with_env():
    """McpServerStdio with env uses temp file loader workaround."""
    from acp.schema import EnvVariable, McpServerStdio
    from google.antigravity.types import McpStdioServer

    server = McpServerStdio(
        name="test-env",
        command="node",
        args=["server.js"],
        env=[EnvVariable(name="API_KEY", value="secret123")],
    )
    result = _convert_mcp_server(server)
    assert isinstance(result, McpStdioServer)
    assert result.command == sys.executable
    assert "-ISs" in result.args
    assert "-c" in result.args
    loader_script = result.args[-1]
    assert "node" in loader_script
    assert "server.js" in loader_script
    assert "os.unlink" in loader_script


def test_convert_mcp_servers_empty():
    """None and empty list return None."""
    assert _convert_mcp_servers(None) is None
    assert _convert_mcp_servers([]) is None
