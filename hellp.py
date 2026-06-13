import asyncio
from typing import Any
from uuid import uuid4

import google.antigravity as agy

f = open("file.log", 'at')

from acp import (
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    text_block,
    update_agent_message, update_agent_thought_text,
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
    TextContentBlock, AgentCapabilities, CloseSessionResponse, SessionModeState, SessionMode, SessionConfigOptionSelect,
    SessionConfigSelectGroup, SessionConfigSelectOption,
)


class EchoAgent(Agent):
    _conn: Client
    _agent: agy.Agent

    def on_connect(self, conn: Client) -> None:
        print("on_connect", file=f)
        self._conn = conn

    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse | None:
        await self._agent.__aexit__(None, None, None)
        # return self._agent.close_session(session_id, **kwargs)

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        print("initialize", file=f)

        config = agy.LocalAgentConfig()
        self._agent = agy.Agent(config)

        await self._agent.__aenter__()

        print("initialized", file=f)

        return InitializeResponse(
            protocol_version=protocol_version,
            # agent_capabilities=AgentCapabilities(nes=False),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        return NewSessionResponse(
            session_id=uuid4().hex,
            config_options=[
                SessionConfigOptionSelect(id="agent", name="Agent", description="Agenting",
                                          current_value="Agent",
                                          options=[
                                              SessionConfigSelectGroup(
                                                  group="agent", name="Agent", options=[
                                                      SessionConfigSelectOption(
                                                          description="Agent", name="Agent", value="agent",
                                                      ),
                                                      SessionConfigSelectOption(
                                                          description="Plan", name="Plan", value="plan",
                                                      ),
                                                  ]
                                              )],
            type="select")])
        # SessionConfigOptionSelect(id="plan", name="Plan", description="Planning"),
        # current_mode_id="agent",

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
        print(f"prompt called, blocks={len(prompt)}, session_id={session_id}", file=f, flush=True)
        content_blocks = []

        for block in prompt:
            block_text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
            if not block_text:
                print(f"skipping empty block: {type(block)}", file=f, flush=True)
                continue

            print(f"calling agent.chat with {len(block_text)} chars", file=f, flush=True)
            try:
                response = await self._agent.chat(block_text)

                print("streaming thoughts...", file=f, flush=True)
                async for t in response.thoughts:
                    await self._conn.session_update(
                        session_id=session_id, update=update_agent_thought_text(t),
                                                    source="echo_agent")

                print("extracting text...", file=f, flush=True)
                text = await response.text()
            except Exception as e:
                print(f"error during agent chat: {e}", file=f, flush=True)
                text = f"Error: {e}"

            print(f"sending message chunk ({len(text)} chars)", file=f, flush=True)
            chunk = update_agent_message(text_block(text))
            chunk.message_id = message_id

            await self._conn.session_update(
                session_id=session_id,
                update=chunk,
                source="echo_agent")
            content_blocks.append(text_block(text))

        print("returning PromptResponse", file=f, flush=True)
        return PromptResponse(
            user_message_id=message_id,
            stop_reason="end_turn"
        )

async def main() -> None:
    await run_agent(EchoAgent())

if __name__ == "__main__":
    asyncio.run(main())
