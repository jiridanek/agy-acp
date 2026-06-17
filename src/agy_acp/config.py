from acp.schema import ModelInfo, SessionMode
from google.antigravity.types import BuiltinTools

_AVAILABLE_MODELS = [
    ModelInfo(model_id="gemini-3.5-flash", name="Gemini 3.5 Flash"),
    ModelInfo(model_id="gemini-3.1-pro-preview", name="Gemini 3.1 Pro"),
    ModelInfo(
        model_id="gemini-3.1-pro-preview-customtools",
        name="Gemini 3.1 Pro (Custom Tools)",
    ),
    ModelInfo(model_id="gemini-3.1-flash-lite", name="Gemini 3.1 Flash Lite"),
    ModelInfo(model_id="gemini-2.5-pro", name="Gemini 2.5 Pro"),
    ModelInfo(model_id="gemini-2.5-flash", name="Gemini 2.5 Flash"),
    ModelInfo(model_id="gemini-2.5-flash-lite", name="Gemini 2.5 Flash Lite"),
]
_DEFAULT_MODEL_ID = "gemini-3.1-flash-lite"

_THINKING_LEVELS = ["minimal", "low", "medium", "high"]
_DEFAULT_THINKING_LEVEL = "medium"

# USD per 1M tokens (input, output). Source: ai.google.dev/pricing
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3.1-pro-preview-customtools": (2.00, 12.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}

# Pro models: input 2x, output 1.5x when context exceeds 200k tokens
_LONG_CONTEXT_THRESHOLD = 200_000

_CONTEXT_PRESETS: dict[str, int] = {
    "compact": 25_000,
    "normal": 50_000,
    "extended": 200_000,
    "max": 1_000_000,
}
_DEFAULT_CONTEXT = "normal"

_INTELLIJ_EXTERNAL_SKILLS = [
    __import__("pathlib").Path.home() / ".claude" / "skills" / "ij-debugger",
]

# Read-only and always-safe tools — auto-allowed in every mode.
# google.antigravity.types.BuiltinTools enum values
_ALWAYS_SAFE_TOOLS = {
    BuiltinTools.VIEW_FILE.value,
    BuiltinTools.LIST_DIR.value,
    BuiltinTools.FIND_FILE.value,
    BuiltinTools.SEARCH_DIR.value,
    BuiltinTools.ASK_QUESTION.value,
    BuiltinTools.FINISH.value,
    BuiltinTools.START_SUBAGENT.value,
    BuiltinTools.GENERATE_IMAGE.value,
}

# File write tools — prompted in agent mode, auto-allowed in accept_edits/bypass,
# denied in plan/dont_ask.
_FILE_WRITE_TOOLS = {
    BuiltinTools.CREATE_FILE.value,
    BuiltinTools.EDIT_FILE.value,
}

_AVAILABLE_MODES = [
    SessionMode(id="agent", name="Agent", description="Standard behavior, prompts for dangerous operations"),
    SessionMode(id="accept_edits", name="Accept Edits", description="Auto-accept file edit operations"),
    SessionMode(id="plan", name="Plan", description="Planning mode, file writes disabled"),
    SessionMode(id="dont_ask", name="Don't Ask", description="Don't prompt for permissions, deny if not pre-approved"),
    SessionMode(id="bypass", name="Bypass Permissions", description="Bypass all permission checks"),
]
