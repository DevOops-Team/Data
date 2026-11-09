"""한 줄 JSON 로그 (stdout). Loki 등에서 그대로 파싱."""
from __future__ import annotations

import json
import sys

from .fencing import now_iso


def log(event: str, **fields) -> None:
    print(json.dumps({"ts": now_iso(), "event": event, **fields}, ensure_ascii=False), file=sys.stdout, flush=True)
