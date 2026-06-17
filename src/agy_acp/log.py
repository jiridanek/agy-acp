import logging
import os
import re


class SecretMaskingFilter(logging.Filter):
    def filter(self, record):
        if hasattr(record, "msg") and isinstance(record.msg, str):
            record.msg = self.scrub(record.msg)
        if hasattr(record, "args") and record.args:
            try:
                if isinstance(record.args, dict):
                    scrubbed_args = {}
                    for k, v in record.args.items():
                        if isinstance(v, str):
                            scrubbed_args[k] = self.scrub(v)
                        else:
                            scrubbed_args[k] = v
                    record.args = scrubbed_args
                else:
                    scrubbed_list = []
                    for arg in record.args:
                        if isinstance(arg, str):
                            scrubbed_list.append(self.scrub(arg))
                        else:
                            scrubbed_list.append(arg)
                    record.args = tuple(scrubbed_list)
            except TypeError:
                pass
        return True

    def scrub(self, text: str) -> str:
        masked = re.sub(
            r'([\'"]?(?:api_?key|secret|token|password|auth|credential)[a-zA-Z0-9_]*[\'"]?\s*[:=]\s*[\'"])(?:[^\'"]+)([\'"])',
            r'\g<1>***REDACTED***\g<2>',
            text,
            flags=re.IGNORECASE
        )
        masked = re.sub(
            r'(name=[\'"][^\'"]*(?:api_key|secret|token|password|auth|credential)[^\'"]*[\'"]\s*,\s*value=[\'"])(?:[^\'"]+)([\'"])',
            r'\g<1>***REDACTED***\g<2>',
            masked,
            flags=re.IGNORECASE
        )
        masked = re.sub(
            r'([\'"]name[\'"]\s*:\s*[\'"][^\'"]*(?:api_?key|secret|token|password|auth|credential)[^\'"]*[\'"]\s*,\s*[\'"]value[\'"]\s*:\s*[\'"])(?:[^\'"]+)([\'"])',
            r'\g<1>***REDACTED***\g<2>',
            masked,
            flags=re.IGNORECASE
        )
        return masked


log = logging.getLogger("agy_acp")
_handler = logging.FileHandler("file.log")
_handler.addFilter(SecretMaskingFilter())
log.addHandler(_handler)
log.setLevel(logging.DEBUG)

_TRACE = bool(os.environ.get("AGY_TRACE"))


def _log_prompt_blocks(prompt: list) -> None:
    log.debug("prompt blocks=%d", len(prompt))
    if not _TRACE:
        return
    for i, block in enumerate(prompt):
        btype = getattr(block, "type", type(block).__name__)
        preview = ""
        if hasattr(block, "text"):
            preview = block.text[:200].replace("\n", "\\n")
        elif hasattr(block, "resource"):
            res = block.resource
            content = getattr(res, "text", None) or getattr(res, "content", None)
            content_len = len(content) if content else 0
            preview = f"resource: {getattr(res, 'uri', '?')} ({content_len} chars)"
        log.debug("  block[%d] type=%s: %s", i, btype, preview)
        log.debug("    raw[%d]: %s", i, str(block)[:1000])


def _log_mcp_servers(label: str, servers: list | None) -> None:
    if not servers:
        log.debug("%s: no mcp_servers", label)
        return
    log.debug("%s mcp_servers=%d", label, len(servers))
    if not _TRACE:
        return
    for i, s in enumerate(servers):
        if hasattr(s, "model_dump"):
            safe_s = s.model_dump()
        elif hasattr(s, "dict"):
            safe_s = s.dict()
        else:
            safe_s = str(s)

        if isinstance(safe_s, dict):
            if "env" in safe_s and isinstance(safe_s["env"], list):
                for env_var in safe_s["env"]:
                    if isinstance(env_var, dict) and "value" in env_var:
                        env_var["value"] = "***REDACTED***"
            if "headers" in safe_s and isinstance(safe_s["headers"], list):
                for header_var in safe_s["headers"]:
                    if isinstance(header_var, dict) and "value" in header_var:
                        header_var["value"] = "***REDACTED***"

        log.debug("  mcp[%d]: %s", i, str(safe_s)[:500])
