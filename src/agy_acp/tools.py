import re

from acp.helpers import plan_entry
from acp.schema import SessionModeState

from agy_acp.config import _AVAILABLE_MODES, _LONG_CONTEXT_THRESHOLD, _MODEL_PRICING


def _build_mode_state(mode_id: str) -> SessionModeState:
    return SessionModeState(current_mode_id=mode_id, available_modes=_AVAILABLE_MODES)


def _get_token_rates(model_id: str, total_context_tokens: int) -> tuple[float, float] | None:
    pricing = _MODEL_PRICING.get(model_id)
    if not pricing:
        return None
    base_in, base_out = pricing
    if "pro" in model_id and total_context_tokens > _LONG_CONTEXT_THRESHOLD:
        return (base_in * 2.0, base_out * 1.5)
    return (base_in, base_out)


_PLAN_LINE_RE = re.compile(
    r"^\s*(?:"
    r"[-*]\s+\[([xX /\-])\]\s+(.*)"  # - [x] item  or  * [ ] item
    r"|[-*]\s+(.*)"  # - item  or  * item
    r"|(\d+)\.\s+(.*)"  # 1. item  or  23. item
    r")$"
)


def _parse_plan_entries(lines: list[str]) -> list:
    entries = []
    for line in lines:
        m = _PLAN_LINE_RE.match(line)
        if not m:
            continue
        if m.group(1) is not None:
            marker, content = m.group(1), m.group(2)
            if marker in ("x", "X"):
                status = "completed"
            elif marker in ("-", "/"):
                status = "in_progress"
            else:
                status = "pending"
        elif m.group(3) is not None:
            content, status = m.group(3), "pending"
        else:
            content, status = m.group(5), "pending"
        content = content.strip()
        if content:
            entries.append(plan_entry(content=content, status=status))
    return entries
