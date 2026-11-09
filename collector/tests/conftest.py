"""테스트용 임시 PostgreSQL 16 (initdb → 임시 폴더) + 실제 db/001_init.sql 적용.

PG_BIN 환경 변수로 PostgreSQL 바이너리 위치 지정 (기본: Homebrew postgresql@16, CI: /usr/lib/postgresql/16/bin)
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pytest

ROOT = Path(__file__).resolve().parents[2]          # 레포 루트
sys.path.insert(0, str(ROOT / "collector"))

PG_BIN = Path(os.environ.get("PG_BIN") or next(
    (p for p in ("/opt/homebrew/opt/postgresql@16/bin", "/usr/lib/postgresql/16/bin") if Path(p, "initdb").exists()),
    "/usr/lib/postgresql/16/bin"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def pg_dsn():
    if not (PG_BIN / "initdb").exists():
        pytest.skip(f"PostgreSQL 바이너리 없음: {PG_BIN}")
    tmp = Path(tempfile.mkdtemp(prefix="collector-pg-"))
    data, port = tmp / "data", _free_port()
    subprocess.run([PG_BIN / "initdb", "-D", data, "-U", "postgres", "-A", "trust", "--no-sync", "-E", "UTF8"],
                   check=True, capture_output=True)
    subprocess.run([PG_BIN / "pg_ctl", "-D", data, "-l", tmp / "log", "-w", "start",
                    "-o", f"-p {port} -k {tmp} -c listen_addresses=127.0.0.1 -c fsync=off"], check=True, capture_output=True)
    admin = f"host=127.0.0.1 port={port} user=postgres dbname=postgres"
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute("CREATE ROLE data_app LOGIN")
        c.execute("CREATE DATABASE gpuf_data OWNER data_app")
    dsn = f"host=127.0.0.1 port={port} user=data_app dbname=gpuf_data"
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("CREATE SCHEMA data")
        c.execute((ROOT / "db" / "001_init.sql").read_text())   # 실제 적용한 스키마 그대로
    yield dsn
    subprocess.run([PG_BIN / "pg_ctl", "-D", data, "-m", "immediate", "stop"], capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def conn(pg_dsn):
    with psycopg.connect(pg_dsn, autocommit=True) as c:
        c.execute("TRUNCATE data.events, data.event_rejects, data.results, data.checkpoints, data.job_runs, "
                  "data.experiments CASCADE")
    c = psycopg.connect(pg_dsn)
    yield c
    c.close()


@dataclass
class FakeMsg:
    _value: bytes | None
    _topic: str = "job.lifecycle"
    _partition: int = 0
    _offset: int = 0
    _error: object = None

    def value(self): return self._value
    def topic(self): return self._topic
    def partition(self): return self._partition
    def offset(self): return self._offset
    def error(self): return self._error


_counter = {"off": 0}


def msg(type_: str, *, topic="job.lifecycle", event_id=None, occurred_at="2026-11-16T10:00:00.000Z",
        payload=None, raw: bytes | None = None, **env) -> FakeMsg:
    _counter["off"] += 1
    if raw is not None:
        return FakeMsg(raw, topic, 0, _counter["off"])
    body = {"event_id": event_id or f"ev-{_counter['off']}", "schema_version": 1, "type": type_,
            "occurred_at": occurred_at, "payload": payload or {}, **env}
    return FakeMsg(json.dumps(body).encode(), topic, 0, _counter["off"])


class FakeConsumer:
    def __init__(self, batches):
        self.batches = list(batches)
        self.commits = 0
        self.seeks = []

    def consume(self, num_messages, timeout):
        return self.batches.pop(0) if self.batches else []

    def commit(self, asynchronous=False):
        self.commits += 1

    def seek(self, tp):
        self.seeks.append((tp.topic, tp.partition, tp.offset))

    def close(self):
        pass
