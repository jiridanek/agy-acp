from unittest.mock import MagicMock

import google.antigravity as agy
from acp.schema import (
    AuthCapabilities,
    ClientCapabilities,
    FileSystemCapabilities,
)

_TEST_CLIENT_CAPS = ClientCapabilities(
    fs=FileSystemCapabilities(read_text_file=True, write_text_file=True),
    terminal=True,
    auth=AuthCapabilities(terminal=False),
)


class FakeAgent:
    """Minimal fake matching the agy.Agent interface, with hook dispatch for ToolCall/ToolResult."""

    def __init__(self, config, responses=None):
        self._responses = responses or []
        self._call_index = 0
        self._pre_hooks = []
        self._post_hooks = []

    def register_hook(self, hook):
        from google.antigravity.hooks.hooks import (
            PostToolCallHook,
            PreToolCallDecideHook,
        )

        if isinstance(hook, PreToolCallDecideHook):
            self._pre_hooks.append(hook)
        elif isinstance(hook, PostToolCallHook):
            self._post_hooks.append(hook)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def chat(self, prompt):
        if self._call_index < len(self._responses):
            chunks = self._responses[self._call_index]
            self._call_index += 1
        else:
            chunks = [agy.types.Text(step_index=0, text="default response")]

        pre_hooks = self._pre_hooks
        post_hooks = self._post_hooks

        async def stream():
            from google.antigravity.hooks.hooks import (
                OperationContext,
                SessionContext,
                TurnContext,
            )

            pending_contexts: dict[str, OperationContext] = {}
            for c in chunks:
                if isinstance(c, agy.types.ToolCall):
                    op_ctx = OperationContext(TurnContext(SessionContext()))
                    if c.id:
                        pending_contexts[c.id] = op_ctx
                    for h in pre_hooks:
                        await h.run(op_ctx, c)
                    yield c
                elif isinstance(c, agy.types.ToolResult):
                    op_ctx = pending_contexts.pop(c.id, None) if c.id else None
                    if op_ctx is None:
                        op_ctx = OperationContext(TurnContext(SessionContext()))
                    for h in post_hooks:
                        await h.run(op_ctx, c)
                else:
                    yield c

        return agy.types.ChatResponse(stream(), conversation=MagicMock())


class FakeConfig:
    def __init__(self, **kwargs):
        pass
