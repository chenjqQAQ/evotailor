from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SECRET = re.compile(r"sk-[A-Za-z0-9_-]{12,}")


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            sensitive = normalized in {
                "api_key",
                "authorization",
                "access_token",
                "refresh_token",
                "secret",
            } or normalized.endswith(("_api_key", "_secret"))
            result[str(key)] = "***REDACTED***" if sensitive else _redact(item)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _SECRET.sub("***REDACTED***", value)
    return value


class ConsoleTrace:
    """Human-readable live trace plus a machine-readable JSONL audit trail."""

    def __init__(self, run_dir: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.console_path = run_dir / "console.log"
        self.jsonl_path = run_dir / "trace.jsonl"

    def emit(self, step: str, message: str, **details: Any) -> None:
        if not self.enabled:
            return
        now = datetime.now(UTC)
        safe_details = _redact(details)
        major = step in {"Task launch", "New round", "Task finished"}
        separator = "═" * 78 if major else "─" * 78
        prefix = f"[{now.astimezone().strftime('%H:%M:%S')}]  {step}"
        lines = ["", separator, prefix, message]
        for key, value in safe_details.items():
            lines.extend(self._format_detail(str(key), value))
        text = "\n".join(lines)
        print(text, file=sys.stderr, flush=True)
        with self.console_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        record = {
            "timestamp": now.isoformat(),
            "step": step,
            "message": message,
            "details": safe_details,
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _format_detail(key: str, value: Any) -> list[str]:
        if value is None:
            return [f"  • {key}：无"]
        if isinstance(value, list):
            if not value:
                return [f"  • {key}：无"]
            lines = [f"  • {key}："]
            for index, item in enumerate(value, 1):
                if isinstance(item, dict):
                    lines.append(f"      [{index}]")
                    for child_key, child_value in item.items():
                        lines.append(f"          {child_key}：{child_value}")
                else:
                    lines.append(f"      - {item}")
            return lines
        if isinstance(value, dict):
            lines = [f"  • {key}："]
            for child_key, child_value in value.items():
                rendered = json.dumps(child_value, ensure_ascii=False, default=str)
                lines.append(f"      {child_key}：{rendered}")
            return lines
        return [f"  • {key}：{value}"]
