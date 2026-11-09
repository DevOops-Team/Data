"""수집기 실행 루프.

한 묶음(batch) 처리 순서
1. Kafka에서 최대 BATCH_SIZE개 읽기
2. DB 트랜잭션 하나에서 메시지마다:
   - 형식 오류 → event_rejects
   - events에 저장(event_id 중복은 무시) → 새로 들어간 것만 파생 테이블 갱신
   - 메시지 하나가 DB 오류(값 범위 등)를 내면 그 메시지만 되돌리고 event_rejects로 → 나머지는 계속
3. DB 커밋 → 그다음 Kafka 오프셋 커밋
DB 연결 오류면 커밋하지 않고, 읽은 위치를 되감은 뒤 기다렸다가 같은 묶음을 다시 처리한다.

환경 변수
  KAFKA_BOOTSTRAP (필수, 예 kafka1:9092,kafka2:9092,kafka3:9092) · KAFKA_GROUP(data-collector)
  KAFKA_TOPICS(gpu.fault,recovery.action,job.lifecycle,fault.injection,monitoring.log)
  PGHOST · PGPORT · PGDATABASE(gpuf_data) · PGUSER · PGPASSWORD · PGSSLMODE  (libpq 표준)
  BATCH_SIZE(500) · POLL_TIMEOUT_S(1.0) · RETRY_SLEEP_S(5)
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import psycopg

from .envelope import RejectError, parse
from .store import DeriveSkip, derive, insert_event, insert_reject

DEFAULT_TOPICS = "gpu.fault,recovery.action,job.lifecycle,fault.injection,monitoring.log"


def log(event: str, **fields: Any) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    print(json.dumps({"ts": ts, "event": event, **fields}, ensure_ascii=False, default=str), flush=True)


class Collector:
    def __init__(self, consumer, connect: Callable[[], psycopg.Connection], *,
                 batch_size: int = 500, poll_timeout: float = 1.0, retry_sleep: float = 5.0,
                 sleep: Callable[[float], None] = time.sleep):
        self.consumer, self.connect = consumer, connect
        self.batch_size, self.poll_timeout, self.retry_sleep, self.sleep = batch_size, poll_timeout, retry_sleep, sleep
        self.conn: psycopg.Connection | None = None
        self.stats: Counter = Counter()
        self.running = True

    # ---------- DB ----------
    def _db(self) -> psycopg.Connection:
        if self.conn is None or self.conn.closed:
            self.conn = self.connect()
        return self.conn

    def process(self, msgs: Iterable) -> Counter:
        """메시지 묶음을 DB 트랜잭션 하나로 저장. 연결 오류는 호출자에게 던진다."""
        c: Counter = Counter()
        conn = self._db()
        with conn.transaction():
            with conn.cursor() as cur:
                for m in msgs:
                    if m.error():
                        c["kafka_error"] += 1
                        log("collector.kafka_msg_error", error=str(m.error()))
                        continue
                    topic, part, off, raw = m.topic(), m.partition(), m.offset(), m.value()
                    try:
                        with conn.transaction():          # 메시지마다 savepoint
                            ev = parse(raw, topic, part, off)
                            if insert_event(cur, ev):
                                c["inserted"] += 1
                                try:
                                    with conn.transaction():
                                        table = derive(cur, ev)
                                    if table:
                                        c[f"derived_{table}"] += 1
                                except DeriveSkip as e:
                                    c["derive_skipped"] += 1
                                    log("collector.derive_skipped", event_id=ev.event_id, type=ev.type, reason=str(e))
                            else:
                                c["duplicate"] += 1
                    except RejectError as e:
                        with conn.transaction():
                            insert_reject(cur, topic, part, off, str(e), raw)
                        c["rejected"] += 1
                    except psycopg.OperationalError:
                        raise                              # 연결 문제 → 묶음 전체 재시도
                    except psycopg.Error as e:             # 이 메시지만의 DB 오류 → 보관하고 계속
                        with conn.transaction():
                            insert_reject(cur, topic, part, off, f"db: {type(e).__name__}: {e}", raw)
                        c["rejected_db"] += 1
        return c

    # ---------- Kafka ----------
    def _rewind(self, msgs: list) -> None:
        """커밋하지 못한 묶음을 다음에 다시 읽도록 파티션별 가장 앞 오프셋으로 되감는다."""
        from confluent_kafka import TopicPartition
        first: dict[tuple[str, int], int] = {}
        for m in msgs:
            if m.error():
                continue
            key = (m.topic(), m.partition())
            first[key] = min(first.get(key, m.offset()), m.offset())
        for (topic, part), off in first.items():
            self.consumer.seek(TopicPartition(topic, part, off))

    def run_once(self) -> None:
        msgs = self.consumer.consume(num_messages=self.batch_size, timeout=self.poll_timeout)
        if not msgs:
            return
        try:
            c = self.process(msgs)
        except psycopg.OperationalError as e:
            log("collector.db_unavailable", error=str(e), batch=len(msgs), retry_in_s=self.retry_sleep)
            if self.conn is not None:
                try:
                    self.conn.close()
                except Exception:  # noqa: BLE001
                    pass
                self.conn = None
            self._rewind(msgs)
            self.sleep(self.retry_sleep)
            return
        self.consumer.commit(asynchronous=False)        # DB 커밋 뒤에만
        self.stats.update(c)
        log("collector.batch", batch=len(msgs), **c)

    def run(self) -> None:
        while self.running:
            self.run_once()
        self.consumer.close()
        if self.conn is not None:
            self.conn.close()


def main() -> None:
    from confluent_kafka import Consumer

    bootstrap = os.environ.get("KAFKA_BOOTSTRAP")
    if not bootstrap:
        log("collector.config_error", error="KAFKA_BOOTSTRAP 없음")
        sys.exit(5)
    consumer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": os.environ.get("KAFKA_GROUP", "data-collector"),
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "client.id": os.environ.get("HOSTNAME", "collector"),
    })
    topics = [t.strip() for t in os.environ.get("KAFKA_TOPICS", DEFAULT_TOPICS).split(",") if t.strip()]
    consumer.subscribe(topics)
    col = Collector(
        consumer, lambda: psycopg.connect(autocommit=False, application_name="collector"),
        batch_size=int(os.environ.get("BATCH_SIZE", "500")),
        poll_timeout=float(os.environ.get("POLL_TIMEOUT_S", "1.0")),
        retry_sleep=float(os.environ.get("RETRY_SLEEP_S", "5")),
    )

    def _stop(*_: Any) -> None:
        col.running = False
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    log("collector.start", topics=topics, group=os.environ.get("KAFKA_GROUP", "data-collector"))
    col.run()
    log("collector.stop", **col.stats)
    sys.exit(0)


if __name__ == "__main__":
    main()
