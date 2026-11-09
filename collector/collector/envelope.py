"""사건 봉투 파싱 · 검증 (데이터 파트 설계 2절 「사건 공통 형식」).

필수: event_id · schema_version · type · occurred_at
선택: source · cluster · node · job_id · generation · payload(객체)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class RejectError(ValueError):
    """형식 오류. event_rejects에 원본을 보관한다."""


@dataclass
class Event:
    event_id: str
    schema_version: int
    type: str
    topic: str
    occurred_at: datetime
    source: str | None
    cluster: str | None
    node: str | None
    job_id: str | None
    generation: int | None
    payload: dict[str, Any] = field(default_factory=dict)
    kafka_partition: int = 0
    kafka_offset: int = 0


def parse_ts(v: Any, name: str) -> datetime:
    if not isinstance(v, str):
        raise RejectError(f"{name}: ISO 8601 문자열이 아님")
    try:
        ts = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError as e:
        raise RejectError(f"{name}: 시각 형식 오류 ({v})") from e
    if ts.tzinfo is None:
        raise RejectError(f"{name}: 시간대 없음 (UTC 'Z' 필요)")
    return ts


def parse(raw: bytes | None, topic: str, partition: int, offset: int) -> Event:
    if raw is None:
        raise RejectError("빈 메시지")
    try:
        obj = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise RejectError(f"invalid json: {e}") from e
    if not isinstance(obj, dict):
        raise RejectError("JSON 객체가 아님")

    for k in ("event_id", "schema_version", "type", "occurred_at"):
        if obj.get(k) in (None, ""):
            raise RejectError(f"missing {k}")
    if not isinstance(obj["event_id"], str):
        raise RejectError("event_id: 문자열이 아님")
    if not isinstance(obj["type"], str):
        raise RejectError("type: 문자열이 아님")
    try:
        sv = int(obj["schema_version"])
    except (TypeError, ValueError) as e:
        raise RejectError("schema_version: 정수가 아님") from e

    payload = obj.get("payload") or {}
    if not isinstance(payload, dict):
        raise RejectError("payload: 객체가 아님")
    gen = obj.get("generation")
    if gen is not None:
        try:
            gen = int(gen)
        except (TypeError, ValueError) as e:
            raise RejectError("generation: 정수가 아님") from e

    return Event(
        event_id=obj["event_id"], schema_version=sv, type=obj["type"], topic=topic,
        occurred_at=parse_ts(obj["occurred_at"], "occurred_at"),
        source=obj.get("source"), cluster=obj.get("cluster"), node=obj.get("node"),
        job_id=obj.get("job_id"), generation=gen, payload=payload,
        kafka_partition=partition, kafka_offset=offset,
    )
