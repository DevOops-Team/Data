"""S3 대기열(outbox): Kafka로 못 보낸 사건을 S3에 보관했다가 다시 보낸다.

왜: EKS → 온프레 Kafka는 사이트 간 VPN을 지난다. VPN이 오래 끊기거나 그 사이 Pod가 교체되면
프로듀서 메모리에만 있던 사건이 사라진다. S3는 AWS 안이라 VPN이 끊겨도 쓸 수 있고 Pod · 노드가 죽어도 남는다.
(Confluence 「VPN 끊김 시 Kafka 사건 유실 문제」 제안 C)

저장 위치: s3://{CKPT_BUCKET}/{OUTBOX_PREFIX}{source}/{event_id}.json
  내용: {"topic", "key", "value"(사건 JSON), "stored_at", "reason"}
  같은 사건은 같은 키라 여러 번 저장돼도 1개. 수집기가 event_id로 중복을 버리므로 다시 보내도 안전.

다시 보내기(replay): 온프레에서 주기적으로 실행 → 대기열 객체를 Kafka로 보내고, 전달이 확인된 것만 지운다.
  python -m ckpt.outbox --once        # 한 번
  python -m ckpt.outbox               # REPLAY_INTERVAL_S(60)마다

필요 권한: 사이드카 s3:PutObject on events-outbox/*, 다시 보내기 s3:ListBucket(prefix) · GetObject · DeleteObject.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from typing import Any

import boto3

from .fencing import now_iso
from .log import log

DEFAULT_PREFIX = "events-outbox/"


class S3Outbox:
    def __init__(self, bucket: str, *, prefix: str = DEFAULT_PREFIX, source: str = "ckpt-sidecar",
                 client=None, region: str = "us-east-2"):
        self.bucket, self.prefix, self.source = bucket, prefix, source
        self.s3 = client or boto3.client("s3", region_name=region)

    def _key(self, value: Any) -> str:
        eid = value.get("event_id") if isinstance(value, dict) else None
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in (eid or f"noid-{uuid.uuid4().hex}"))
        return f"{self.prefix}{self.source}/{safe}.json"

    def put(self, topic: str, key: str | None, value: Any, reason: str) -> bool:
        """사건 1건 보관. 실패해도 예외를 던지지 않는다 (체크포인트 흐름을 막지 않음)."""
        if isinstance(value, (bytes, bytearray)):
            try:
                value = json.loads(value)
            except ValueError:
                value = value.decode(errors="replace")
        body = {"topic": topic, "key": key, "value": value, "stored_at": now_iso(), "reason": reason}
        k = self._key(value)
        try:
            self.s3.put_object(Bucket=self.bucket, Key=k, Body=json.dumps(body, ensure_ascii=False).encode(),
                               ContentType="application/json")
            log("outbox.stored", key=k, reason=reason)
            return True
        except Exception as e:  # noqa: BLE001
            log("outbox.store_failed", key=k, reason=reason, error=repr(e))
            return False

    def keys(self, limit: int = 500) -> list[str]:
        out: list[str] = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=self.prefix):
            for o in page.get("Contents", []):
                out.append(o["Key"])
                if len(out) >= limit:
                    return out
        return out

    def get(self, key: str) -> dict:
        return json.loads(self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read())

    def delete(self, key: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=key)


def replay(outbox: S3Outbox, producer, *, limit: int = 500, flush_timeout: float = 30.0) -> dict:
    """대기열 → Kafka. 전달 확인(acks=all)된 것만 S3에서 지운다. 실패한 것은 다음 번에 다시."""
    keys = outbox.keys(limit)
    delivered: list[str] = []
    failed: dict[str, str] = {}

    def cb_for(k):
        def cb(err, _msg):
            if err is None:
                delivered.append(k)
            else:
                failed[k] = str(err)
        return cb

    bad = 0
    for k in keys:
        try:
            item = outbox.get(k)
            value = item["value"]
            data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            key = item.get("key")
            producer.produce(item["topic"], key=key.encode() if key else None, value=data.encode(),
                             on_delivery=cb_for(k))
        except Exception as e:  # noqa: BLE001 — 깨진 객체 하나가 나머지를 막지 않게
            bad += 1
            log("outbox.replay_skip", key=k, error=repr(e))
        producer.poll(0)
    left = producer.flush(flush_timeout)
    for k in delivered:
        try:
            outbox.delete(k)
        except Exception as e:  # noqa: BLE001
            log("outbox.delete_failed", key=k, error=repr(e))
    stats = {"found": len(keys), "delivered": len(delivered), "failed": len(failed), "skipped": bad, "pending": left}
    log("outbox.replay", **stats)
    return stats


def main() -> None:
    from confluent_kafka import Producer
    bucket = os.environ["CKPT_BUCKET"]
    outbox = S3Outbox(bucket, prefix=os.environ.get("OUTBOX_PREFIX", DEFAULT_PREFIX), source="",
                      region=os.environ.get("AWS_REGION", "us-east-2"))
    producer = Producer({"bootstrap.servers": os.environ["KAFKA_BOOTSTRAP"], "acks": "all",
                         "enable.idempotence": True, "client.id": "outbox-replay"})
    interval = float(os.environ.get("REPLAY_INTERVAL_S", "60"))
    while True:
        replay(outbox, producer)
        if "--once" in sys.argv:
            break
        time.sleep(interval)


if __name__ == "__main__":
    main()
