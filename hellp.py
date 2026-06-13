import asyncio
from typing import Any
from uuid import uuid4

import google.antigravity as agy


from acp import (
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    text_block,
    update_agent_message,
)
from acp.interfaces import Client
from acp.schema import (
    AudioContentBlock,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    McpServerStdio,
    ResourceContentBlock,
    SseMcpServer,
    TextContentBlock, AgentCapabilities, CloseSessionResponse,
)


class EchoAgent(Agent):
    _conn: Client
    _agent : agy.Agent

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse | None:
        await self._agent.__aexit__(None, None, None)
        #return self._agent.close_session(session_id, **kwargs)

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        config = agy.LocalAgentConfig()
        self._agent = agy.Agent(config)

        await self._agent.__aenter__()

        return InitializeResponse(
            protocol_version=protocol_version,
            #agent_capabilities=AgentCapabilities(nes=False),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        return NewSessionResponse(session_id=uuid4().hex)

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
        for block in prompt:
            prompt = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
            # if prompt:
            response = await self._agent.chat(prompt)
            text = await response.text()
            # else:
            #     text = ""

            chunk = update_agent_message(text_block(text))
            chunk.field_meta = {"echo": True}
            chunk.content.field_meta = {"echo": True}

            await self._conn.session_update(session_id=session_id, update=chunk, source="echo_agent")
        return PromptResponse(stop_reason="end_turn")


async def main() -> None:
    await run_agent(EchoAgent())


if __name__ == "__main__":
    asyncio.run(main())