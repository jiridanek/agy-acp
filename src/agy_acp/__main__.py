import asyncio
import os

import google.antigravity as agy
from acp import run_agent

from agy_acp.agent import EchoAgent
from agy_acp.log import log


async def _main() -> None:
    agent = EchoAgent(agent_config_t=agy.LocalAgentConfig, agent_t=agy.Agent)
    log.info("run_agent starting (pid=%d)", os.getpid())
    try:
        await run_agent(agent, use_unstable_protocol=True)
        log.info("run_agent returned normally")
    except Exception:
        log.exception("run_agent raised")
    finally:
        for sid in list(agent._sessions):
            try:
                await agent.close_session(session_id=sid)
            except Exception:
                log.debug("cleanup: failed to close session %s", sid)


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
