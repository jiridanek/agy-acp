import asyncio
import base64
import logging
from typing import Any
from uuid import uuid4

import google.antigravity as agy

log = logging.getLogger(__name__)
log.addHandler(logging.FileHandler("file.log"))
log.setLevel(logging.DEBUG)

from acp import (
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    text_block,
    update_agent_message, update_agent_thought_text,
)
from acp.contrib.tool_calls import ToolCallTracker
from acp.interfaces import Client
from acp.schema import (
    AudioContentBlock,
    BlobResourceContents,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    McpServerStdio,
    ResourceContentBlock,
    SseMcpServer,
    TextContentBlock,
    TextResourceContents,
    AgentCapabilities, CloseSessionResponse, PromptCapabilities,
    SessionModeState, SessionMode, SessionConfigOptionSelect,
    SessionConfigSelectGroup, SessionConfigSelectOption,
)


class EchoAgent(Agent):
    _conn: Client
    _agent: agy.Agent

    def __init__(self, agent_t, agent_config_t):
        self._agent_t = agent_t
        self._agent_config_t = agent_config_t
        self._active_tasks: dict[str, asyncio.Task] = {}

    def on_connect(self, conn: Client) -> None:
        log.debug("on_connect")
        self._conn = conn

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        log.debug("cancel received for session %s", session_id)
        task = self._active_tasks.get(session_id)
        if task and not task.done():
            task.cancel()

    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse:
        await self._agent.__aexit__(None, None, None)
        return CloseSessionResponse()

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        log.debug("initialize")

        config = self._agent_config_t()
        self._agent = self._agent_t(config)

        await self._agent.__aenter__()

        log.debug("initialized")

        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(
                prompt_capabilities=PromptCapabilities(
                    image=True, audio=True, embedded_context=True,
                ),
            ),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        self._cwd = cwd
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
        log.debug("prompt called, blocks=%d, session_id=%s", len(prompt), session_id)

        parts: list[agy.types.ContentPrimitive] = []
        for block in prompt:
            match block:
                case {"type": "text", "text": text}:
                    parts.append(text)
                case TextContentBlock(text=text):
                    parts.append(text)
                case ImageContentBlock(data=data, mime_type=mime):
                    parts.append(agy.types.Image(
                        data=base64.b64decode(data),
                        mime_type=mime,
                    ))
                case AudioContentBlock(data=data, mime_type=mime):
                    parts.append(agy.types.Audio(
                        data=base64.b64decode(data),
                        mime_type=mime,
                    ))
                case EmbeddedResourceContentBlock(resource=TextResourceContents(text=text)):
                    parts.append(text)
                case EmbeddedResourceContentBlock(resource=BlobResourceContents(blob=blob, mime_type=mime)):
                    parts.append(agy.types.Document(
                        data=base64.b64decode(blob),
                        mime_type=mime or "application/octet-stream",
                    ))
                case ResourceContentBlock(uri=uri, name=name):
                    parts.append(f"[Attached resource: {name}]({uri})")
                case _:
                    log.debug("skipping unknown block: %s", type(block))

        if not parts:
            log.debug("no content to send")
            return PromptResponse(user_message_id=message_id, stop_reason="end_turn")

        log.debug("calling agent.chat with %d parts", len(parts))
        self._active_tasks[session_id] = asyncio.current_task()
        tracker = ToolCallTracker()
        stop_reason = "end_turn"
        try:
            response = await self._agent.chat(parts)
            async for chunk in response.chunks:
                match chunk:
                    case agy.types.Thought(text=t):
                        await self._conn.session_update(
                            session_id=session_id,
                            update=update_agent_thought_text(t))
                    case agy.types.Text(text=t):
                        await self._conn.session_update(
                            session_id=session_id,
                            update=update_agent_message(text_block(t)))
                    case agy.types.ToolCall(name=name, id=tc_id, args=args):
                        tc_id = tc_id or uuid4().hex
                        start = tracker.start(tc_id, title=str(name), kind="execute",
                                              raw_input=args)
                        await self._conn.session_update(
                            session_id=session_id, update=start)
                    case agy.types.ToolResult(name=name, id=tc_id, result=result, error=err):
                        tc_id = tc_id or ""
                        try:
                            progress = tracker.progress(
                                tc_id,
                                status="failed" if err else "completed",
                                raw_output=err or result)
                            await self._conn.session_update(
                                session_id=session_id, update=progress)
                        except KeyError:
                            log.debug("tool result for unknown call %s", tc_id)
                    case _:
                        log.debug("unhandled chunk type: %s", type(chunk))
        except asyncio.CancelledError:
            log.debug("prompt cancelled for session %s", session_id)
            if hasattr(response, "cancel"):
                await response.cancel()
            stop_reason = "cancelled"
        except Exception as e:
            log.exception("error during agent chat")
            await self._conn.session_update(
                session_id=session_id,
                update=update_agent_message(text_block(f"Error: {e}")))
        finally:
            self._active_tasks.pop(session_id, None)

        log.debug("returning PromptResponse stop_reason=%s", stop_reason)
        return PromptResponse(
            user_message_id=message_id,
            stop_reason=stop_reason,
        )

async def main() -> None:
    await run_agent(EchoAgent(agent_config_t=agy.LocalAgentConfig, agent_t=agy.Agent))

if __name__ == "__main__":
    asyncio.run(main())
