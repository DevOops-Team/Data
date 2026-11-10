"""Kafka 사건 발행 (데이터 파트 설계 3절 · 수집기 README).

- 사이드카는 승인한 체크포인트를 job.lifecycle 토픽에 checkpoint.approved로 보낸다 → 수집기가 data.checkpoints에 저장
- 발행은 '최선 노력': Kafka가 안 되면 체크포인트 저장 · 승인은 그대로 진행한다
- Kafka로 못 보낸 사건은 S3 대기열(outbox.py)에 보관 → 온프레에서 다시 보낸다
    · 전달 최종 실패(message.timeout.ms 30분 초과) · 보내기 즉시 오류(버퍼 가득 등) · 종료 때 남은 사건
- event_id는 job · 세대 · step으로 고정 → 같은 step을 다시 보내도 DB에는 한 번만 들어간다
- KAFKA_BOOTSTRAP이 없으면 발행하지 않는다 (로컬 테스트 · Kafka 준비 전)

환경 변수: KAFKA_BOOTSTRAP · KAFKA_TOPIC_LIFECYCLE(job.lifecycle) · NODE_NAME(Downward API)
          KAFKA_MESSAGE_TIMEOUT_MS(1800000) · OUTBOX_PREFIX(events-outbox/, 빈 값이면 대기열 끔)
"""
from __future__ import annotations

import json
import os
from typing import Any, Protocol

from .fencing import now_iso
from .log import log

SCHEMA_VERSION = 1
SOURCE = "ckpt-sidecar"


class Publisher(Protocol):
    def publish(self, *, topic: str, key: str, value: dict) -> None: ...
    def flush(self, timeout: float = 5.0) -> None: ...


class NullPublisher:
    """KAFKA_BOOTSTRAP 없음 → 아무것도 보내지 않음."""
    def publish(self, *, topic: str, key: str, value: dict) -> None:
        pass

    def flush(self, timeout: float = 5.0) -> None:
        pass


class KafkaPublisher:
    """confluent-kafka Producer. 재시도는 librdkafka가 message.timeout.ms(기본 30분)까지 하고,
    그래도 못 보낸 사건은 outbox(S3 대기열)에 넘긴다. outbox가 없으면 로그만."""

    def __init__(self, bootstrap: str, producer=None, outbox=None):
        if producer is None:
            from confluent_kafka import Producer
            producer = Producer({
                "bootstrap.servers": bootstrap,
                "client.id": os.environ.get("HOSTNAME", "ckpt-sidecar"),
                "acks": "all",                      # min.insync.replicas=2와 함께 브로커 1대 장애에도 유실 없음
                "enable.idempotence": True,         # librdkafka 재시도로 생기는 중복 방지
                # 30분까지 재시도 후 포기 → outbox. 장애 복구 규약의 재개 대기 30분과 맞춤
                "message.timeout.ms": int(os.environ.get("KAFKA_MESSAGE_TIMEOUT_MS", "1800000")),
                "linger.ms": 50,
            })
        self.producer = producer
        self.outbox = outbox

    def _to_outbox(self, topic: str, key, value, reason: str) -> None:
        if self.outbox is not None:
            self.outbox.put(topic, _s(key), value, reason)

    def _on_delivery(self, err, msg) -> None:
        if err is not None:
            log("events.delivery_failed", topic=msg.topic(), key=_s(msg.key()), error=str(err))
            self._to_outbox(msg.topic(), msg.key(), msg.value(), f"delivery_failed: {err}")
        else:
            log("events.delivered", topic=msg.topic(), key=_s(msg.key()), partition=msg.partition(),
                offset=msg.offset())

    def publish(self, *, topic: str, key: str, value: dict) -> None:
        data = json.dumps(value, ensure_ascii=False).encode()
        try:
            self.producer.produce(topic, key=key.encode(), value=data, on_delivery=self._on_delivery)
        except Exception as e:  # noqa: BLE001 — 버퍼 가득 참(BufferError) 등. 체크포인트 흐름은 막지 않음
            log("events.produce_failed", topic=topic, key=key, error=repr(e))
            self._to_outbox(topic, key, data, f"produce_failed: {e!r}")
        self.producer.poll(0)                        # 전달 결과 콜백 처리 (막지 않음)

    def flush(self, timeout: float = 5.0) -> None:
        """종료할 때. timeout 안에 못 보낸 사건은 큐에서 비워(purge) 실패 콜백으로 outbox에 넘긴다."""
        left = self.producer.flush(timeout)
        if left:
            log("events.flush_incomplete", pending=left)
            if self.outbox is not None:
                try:
                    self.producer.purge(in_queue=True, in_flight=True, blocking=True)
                    self.producer.flush(0)           # purge된 메시지의 실패 콜백 실행 → outbox
                except Exception as e:  # noqa: BLE001
                    log("events.purge_failed", error=repr(e))


def _s(b) -> str | None:
    return b.decode() if isinstance(b, (bytes, bytearray)) else b


def from_env(bucket: str | None = None, region: str = "us-east-2") -> Publisher:
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP")
    if not bootstrap:
        log("events.disabled", reason="KAFKA_BOOTSTRAP 없음")
        return NullPublisher()
    outbox = None
    prefix = os.environ.get("OUTBOX_PREFIX", "events-outbox/")
    if bucket and prefix:
        from .outbox import S3Outbox
        outbox = S3Outbox(bucket, prefix=prefix, source=SOURCE, region=region)
    else:
        log("events.outbox_disabled", reason="버킷 없음" if not bucket else "OUTBOX_PREFIX 비어 있음")
    return KafkaPublisher(bootstrap, outbox=outbox)


def envelope(*, type_: str, event_id: str, cluster: str, job_id: str, generation: int,
             payload: dict[str, Any], occurred_at: str | None = None) -> dict:
    """수집기 봉투 형식 (collector/envelope.py: event_id · schema_version · type · occurred_at 필수, 시각은 시간대 포함)."""
    return {
        "event_id": event_id, "schema_version": SCHEMA_VERSION, "type": type_,
        "occurred_at": occurred_at or now_iso(), "source": SOURCE, "cluster": cluster,
        "node": os.environ.get("NODE_NAME"), "job_id": job_id, "generation": generation, "payload": payload,
    }


def checkpoint_approved_id(job_id: str, generation: int, step: int) -> str:
    return f"ckpt-{job_id}-g{generation:04d}-s{step:08d}"
