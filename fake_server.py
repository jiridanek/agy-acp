"""Fake ACP agent server for integration testing. No LLM needed.

Spawned as a subprocess by test_subprocess_fake_tool_call. The fake
agent calls view_file() during chat(), which sends fs/read_text_file
RPC back to the client over JSON-RPC stdio transport.
"""
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import google.antigravity as agy
from acp import run_agent
from hellp import EchoAgent, SessionStore


class FakeToolAgent:
    _echo_agent = None

    def __init__(self, config):
        pass

    def register_hook(self, hook):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass

    @property
    def conversation_id(self):
        return None

    async def chat(self, prompt):
        result = await FakeToolAgent._echo_agent.view_file("/tmp/fake_test_file.txt")

        async def stream():
            yield agy.types.Text(step_index=0, text=f"File content: {result}")

        return agy.types.ChatResponse(stream(), conversation=MagicMock())


class FakeConfig:
    def __init__(self, **kw):
        pass


async def main():
    agent = EchoAgent(
        agent_t=FakeToolAgent,
        agent_config_t=FakeConfig,
        store=SessionStore(path=Path("/tmp/agy-acp-fake-sessions.json")),
    )
    FakeToolAgent._echo_agent = agent
    await run_agent(agent, use_unstable_protocol=True)


if __name__ == "__main__":
    asyncio.run(main())