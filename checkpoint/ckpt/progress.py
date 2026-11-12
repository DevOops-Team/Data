"""학습 진행률 → Kafka job.progress (GPU-141). rank 0 사이드카만.

trainer(rank 0)가 PROGRESS_FILE(/progress/progress.json, 공유 emptyDir)에 현재 상태를 덮어쓴다 (Backend trainer/train.py).
  {"generation": 1, "status": "RUNNING", "step": 1200, "total_steps": 2080, "loss": 2.31,
   "last_checkpoint_step": 694, "checkpoint_every_steps": 694, "checkpoint_count": 3, "step_seconds": 1.21}
사이드카가 이 파일을 읽어 바뀌었을 때만 job.progress 사건을 보낸다 → 수집기가 data.job_progress에 작업당 1행으로 저장
→ Backend가 화면 진행률을 PostgreSQL에서 읽는다 (진행률 DynamoDB 테이블 · progress-reporter 사이드카 대체).

- 최소 간격 PROGRESS_INTERVAL_SECONDS(30). status가 바뀌면(COMPLETED · FAILED) 간격과 상관없이 바로
- 사이드카가 끝날 때(SIGTERM) 아직 안 보낸 값이 있으면 한 번 더
- event_id = progress-{작업}-g{세대 4자리}-s{step 8자리}, RUNNING이 아니면 끝에 -{status 소문자}
  → 같은 값을 다시 보내도 수집기가 한 번만 저장. 마지막 step의 COMPLETED는 RUNNING과 다른 사건
- 발행은 최선 노력: Kafka가 안 돼도 학습 · 체크포인트 승인에 영향 없음
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import events
from .log import log

FIELDS = ("status", "step", "total_steps", "loss", "last_checkpoint_step", "checkpoint_every_steps",
          "checkpoint_count", "step_seconds")


def progress_event_id(job_id: str, generation: int, step: int, status: str | None) -> str:
    base = f"progress-{job_id}-g{generation:04d}-s{int(step):08d}"
    return base if (status or "RUNNING").upper() == "RUNNING" else f"{base}-{status.lower()}"


class ProgressReporter:
    def __init__(self, path: str | Path | None, *, job_id: str, generation: int, cluster: str,
                 publisher: events.Publisher, topic: str = "job.lifecycle",
                 interval_seconds: float | None = None, clock: Callable[[], float] = time.monotonic):
        self.path = Path(path) if path else None
        self.job_id, self.generation, self.cluster = job_id, generation, cluster
        self.publisher, self.topic, self.clock = publisher, topic, clock
        self.interval = (interval_seconds if interval_seconds is not None
                         else float(os.environ.get("PROGRESS_INTERVAL_SECONDS", "30")))
        self._last_sent: dict[str, Any] | None = None
        self._last_sent_at: float | None = None
        self._pending: tuple[dict[str, Any], float] | None = None   # 간격 때문에 미룬 최신 값

    def _read(self) -> tuple[dict[str, Any], float] | None:
        if self.path is None:
            return None
        try:
            mtime = self.path.stat().st_mtime
            data = json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None                                             # 아직 없음 · 쓰는 중 → 다음 tick
        if not isinstance(data, dict) or data.get("step") is None:
            return None
        if data.get("generation") not in (None, self.generation):
            return None                                             # 다른 세대가 쓴 파일이면 무시
        return {k: data[k] for k in FIELDS if data.get(k) is not None}, mtime

    def tick(self, *, force: bool = False) -> bool:
        """보냈으면 True. force=True면 간격을 무시하고 미룬 값까지 보냄 (종료 때)."""
        read = self._read()
        if read is not None:
            cur, mtime = read
            if cur != (self._last_sent or {}):
                self._pending = (cur, mtime)
        if self._pending is None:
            return False
        cur, mtime = self._pending
        status_changed = cur.get("status") != (self._last_sent or {}).get("status")
        due = self._last_sent_at is None or self.clock() - self._last_sent_at >= self.interval
        if not (force or status_changed or due):
            return False
        self._send(cur, mtime)
        return True

    def _send(self, cur: dict[str, Any], mtime: float) -> None:
        occurred = (datetime.fromtimestamp(mtime, timezone.utc)
                    .isoformat(timespec="milliseconds").replace("+00:00", "Z"))
        ev = events.envelope(type_="job.progress",
                             event_id=progress_event_id(self.job_id, self.generation, cur["step"], cur.get("status")),
                             cluster=self.cluster, job_id=self.job_id, generation=self.generation,
                             payload=cur, occurred_at=occurred)
        try:
            self.publisher.publish(topic=self.topic, key=self.job_id, value=ev)
        except Exception as e:  # noqa: BLE001 — 진행률 발행 실패가 사이드카를 멈추면 안 됨
            log("progress.publish_failed", step=cur.get("step"), error=repr(e))
            return
        self._last_sent, self._last_sent_at, self._pending = cur, self.clock(), None
        log("progress.sent", step=cur.get("step"), status=cur.get("status"))
