from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    """UTC 밀리초, 'Z' (수집기 envelope.parse_ts가 받는 형식)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def log(event: str, **fields: Any) -> None:
    print(json.dumps({"ts": now_iso(), "event": event, **fields}, ensure_ascii=False, default=str), flush=True)
