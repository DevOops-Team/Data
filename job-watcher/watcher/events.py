"""사건 봉투 · Kafka 발행.

봉투는 수집기 형식(collector/envelope.py)과 체크포인트 사이드카(ckpt/events.py)와 같다.
  필수: event_id · schema_version · type · occurred_at(UTC, 시간대 포함)
  job_runs 파생에 필요: job_id · generation  (payload: jobset_name · queue · gpus · exp_id · config_sha256 · image_digest)

event_id = job-{작업ID}-g{세대 4자리}-{key}  예: job-3f9a1c2e-g0002-started
  → 감시기가 재시작해 같은 사건을 다시 보내도 수집기가 한 번만 저장한다.

감시기 · Kafka 모두 온프레미스라 VPN을 지나지 않으므로(ADR-29) S3 대기열(outbox)은 두지 않는다.
전달 실패는 librdkafka가 message.timeout.ms까지 재시도하고, 그래도 실패하면 로그를 남긴다.
감시기가 재시작하면 list로 현재 상태를 다시 읽어 빠진 사건을 다시 보낸다(중복은 event_id로 걸러짐).
"""
from __future__ import annotations

import json
import os
from typing import Any, Protocol

from .log import log
from .model import Snapshot
from .rules import Planned

SCHEMA_VERSION = 1
SOURCE = "job-watcher"


def event_id(job_id: str, generation: int, key: str) -> str:
    return f"job-{job_id}-g{generation:04d}-{key}"


def build(s: Snapshot, p: Planned) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "jobset_name": s.name, "queue": s.queue, "gpus": s.gpus, "exp_id": s.exp_id,
        "recovery_id": s.recovery_id, "config_sha256": s.config_sha256, "image_digest": s.image_digest,
        "cluster": s.cluster, "time_source": p.time_source, **p.extra,
    }
    return {
        "event_id": event_id(s.job_id, s.generation, p.key), "schema_version": SCHEMA_VERSION,
        "type": p.type, "occurred_at": _iso_z(p.occurred_at), "source": SOURCE, "cluster": s.cluster,
        "node": None, "job_id": s.job_id, "generation": s.generation,
        "payload": {k: v for k, v in payload.items() if v is not None},
    }


def _iso_z(ts: str) -> str:
    """쿠버네티스 시각('2026-11-12T01:02:03Z')은 그대로. '+00:00'은 'Z'로."""
    return ts.replace("+00:00", "Z")


class Publisher(Protocol):
    def publish(self, *, topic: str, key: str, value: dict) -> None: ...
    def flush(self, timeout: float = 5.0) -> int: ...


class StdoutPublisher:
    """KAFKA_BOOTSTRAP 없음 → 표준 출력으로만 (로컬 kind 확인 · Kafka 준비 전)."""
    def publish(self, *, topic: str, key: str, value: dict) -> None:
        log("event.dry_run", topic=topic, key=key, value=value)

    def flush(self, timeout: float = 5.0) -> int:
        return 0


class KafkaPublisher:
    def __init__(self, bootstrap: str, producer=None):
        if producer is None:
            from confluent_kafka import Producer
            producer = Producer({
                "bootstrap.servers": bootstrap,
                "client.id": os.environ.get("HOSTNAME", SOURCE),
                "acks": "all",                      # min.insync.replicas=2 → 브로커 1대 장애에도 유실 없음
                "enable.idempotence": True,         # 재시도로 생기는 중복 방지
                "message.timeout.ms": int(os.environ.get("KAFKA_MESSAGE_TIMEOUT_MS", "1800000")),
                "linger.ms": 50,
            })
        self.producer = producer

    @staticmethod
    def _on_delivery(err, msg) -> None:
        key = msg.key().decode() if msg.key() else None
        if err is not None:
            log("event.delivery_failed", topic=msg.topic(), key=key, error=str(err))
        else:
            log("event.delivered", topic=msg.topic(), key=key, partition=msg.partition(), offset=msg.offset())

    def publish(self, *, topic: str, key: str, value: dict) -> None:
        data = json.dumps(value, ensure_ascii=False).encode()
        while True:
            try:
                self.producer.produce(topic, key=key.encode(), value=data, on_delivery=self._on_delivery)
                break
            except BufferError:                      # 로컬 큐 가득 → 비울 때까지 기다렸다 다시
                self.producer.poll(1.0)
        self.producer.poll(0)

    def flush(self, timeout: float = 5.0) -> int:
        left = self.producer.flush(timeout)
        if left:
            log("event.flush_incomplete", pending=left)
        return left


def from_env() -> Publisher:
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP")
    if not bootstrap:
        log("event.kafka_disabled", reason="KAFKA_BOOTSTRAP 없음 → 표준 출력")
        return StdoutPublisher()
    return KafkaPublisher(bootstrap)
