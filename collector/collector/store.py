"""데이터 DB 쓰기 (db/001_init.sql, 스키마 data).

파생 테이블에 필요한 필드가 없으면 파생만 건너뛴다(사건 원본은 events에 남아 나중에 SQL로 다시 뽑을 수 있음).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from psycopg import Cursor
from psycopg.types.json import Jsonb

from .envelope import Event, parse_ts

END_REASONS = {
    "job.completed": "completed", "job.failed": "failed", "job.fenced": "fenced",
    "job.stopped": "stopped", "job.cancelled": "cancelled",
}
JOB_TYPES = {"job.submitted", "job.started", "job.resumed", *END_REASONS}


class DeriveSkip(Exception):
    """파생에 필요한 필드가 없음. 사건은 저장하고 파생만 건너뜀."""


def insert_event(cur: Cursor, e: Event) -> bool:
    """새로 들어갔으면 True, 이미 있던 event_id면 False."""
    cur.execute(
        """INSERT INTO data.events (event_id, schema_version, type, topic, occurred_at, source, cluster, node,
                                    job_id, generation, payload, kafka_partition, kafka_offset)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (event_id) DO NOTHING RETURNING 1""",
        (e.event_id, e.schema_version, e.type, e.topic, e.occurred_at, e.source, e.cluster, e.node,
         e.job_id, e.generation, Jsonb(e.payload), e.kafka_partition, e.kafka_offset),
    )
    return cur.fetchone() is not None


def insert_reject(cur: Cursor, topic: str, partition: int, offset: int, error: str, raw: bytes | None) -> None:
    cur.execute(
        """INSERT INTO data.event_rejects (topic, kafka_partition, kafka_offset, error, raw)
           VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
        (topic, partition, offset, error[:1000], raw or b""),
    )


def derive(cur: Cursor, e: Event) -> str | None:
    """파생 테이블 갱신. 갱신한 테이블 이름(없으면 None)."""
    t = e.type
    if t == "experiment.started":
        _experiment_started(cur, e)
        return "experiments"
    if t == "experiment.ended":
        _experiment_ended(cur, e)
        return "experiments"
    if t == "checkpoint.approved":
        _checkpoint_approved(cur, e)
        return "checkpoints"
    if t in JOB_TYPES:
        _job_run(cur, e)
        return "job_runs"
    return None


def _req(p: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if p.get(k) in (None, "")]
    if missing:
        raise DeriveSkip(f"payload에 {', '.join(missing)} 없음")


# ---------- experiments ----------
def _experiment_started(cur: Cursor, e: Event) -> None:
    p = e.payload
    _req(p, "exp_id", "scenario", "policy", "spare_enabled")
    cur.execute(
        """INSERT INTO data.experiments (exp_id, name, scenario, policy, fault_type, spare_enabled,
                                         checkpoint_every_steps, conditions, started_at, note)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (exp_id) DO UPDATE SET
             name = EXCLUDED.name, scenario = EXCLUDED.scenario, policy = EXCLUDED.policy,
             fault_type = EXCLUDED.fault_type, spare_enabled = EXCLUDED.spare_enabled,
             checkpoint_every_steps = EXCLUDED.checkpoint_every_steps, conditions = EXCLUDED.conditions,
             started_at = COALESCE(data.experiments.started_at, EXCLUDED.started_at)""",
        (p["exp_id"], p.get("name") or p["exp_id"], p["scenario"], p["policy"], p.get("fault_type"),
         bool(p["spare_enabled"]), p.get("checkpoint_every_steps"), Jsonb(p.get("conditions") or {}),
         e.occurred_at, p.get("note")),
    )
    # 실험보다 먼저 도착한 job.submitted가 있으면 exp_id 연결 (토픽이 달라 순서 보장 없음)
    cur.execute(
        """UPDATE data.job_runs jr SET exp_id = %s
           FROM data.events ev
           WHERE ev.type = 'job.submitted' AND ev.payload->>'exp_id' = %s
             AND jr.job_id = ev.job_id AND jr.generation = ev.generation AND jr.exp_id IS NULL""",
        (p["exp_id"], p["exp_id"]),
    )


def _experiment_ended(cur: Cursor, e: Event) -> None:
    _req(e.payload, "exp_id")
    cur.execute("UPDATE data.experiments SET ended_at = %s WHERE exp_id = %s",
                (e.occurred_at, e.payload["exp_id"]))
    if cur.rowcount == 0:
        raise DeriveSkip(f"experiments에 {e.payload['exp_id']} 없음 (experiment.started 먼저 필요)")


# ---------- checkpoints ----------
def _ts(p: dict[str, Any], key: str, default: datetime) -> datetime:
    return parse_ts(p[key], key) if p.get(key) else default


def _checkpoint_approved(cur: Cursor, e: Event) -> None:
    p = e.payload
    _req(p, "step", "format", "world_size", "manifest_key", "file_count", "size_bytes", "config_sha256")
    gen = e.generation if e.generation is not None else p.get("generation")
    if not e.job_id or gen is None:
        raise DeriveSkip("job_id · generation 없음")
    cur.execute(
        """INSERT INTO data.checkpoints (job_id, generation, step, format, world_size, manifest_key, file_count,
                                         size_bytes, config_sha256, image_digest, save_ms, upload_ms, saved_at, approved_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (job_id, generation, step) DO NOTHING""",
        (e.job_id, int(gen), int(p["step"]), p["format"], int(p["world_size"]), p["manifest_key"],
         int(p["file_count"]), int(p["size_bytes"]), p["config_sha256"], p.get("image_digest"),
         p.get("save_ms"), p.get("upload_ms"), _ts(p, "saved_at", e.occurred_at), _ts(p, "approved_at", e.occurred_at)),
    )


# ---------- job_runs ----------
def _job_run(cur: Cursor, e: Event) -> None:
    if not e.job_id or e.generation is None:
        raise DeriveSkip("job_id · generation 없음")
    p, t = e.payload, e.type
    submitted = t == "job.submitted"
    started = t in ("job.started", "job.resumed")
    end_reason = END_REASONS.get(t)
    cur.execute(
        """INSERT INTO data.job_runs (job_id, generation, exp_id, jobset_name, cluster, queue, gpus, strategy,
                                      config_sha256, image_digest, submitted_at, started_at, ended_at, end_reason)
           VALUES (%(job_id)s, %(gen)s, (SELECT exp_id FROM data.experiments WHERE exp_id = %(exp_id)s),
                   COALESCE(%(jobset)s, %(jobset_default)s), %(cluster)s, %(queue)s,
                   COALESCE(%(gpus)s, 0), COALESCE(%(strategy)s, 'unknown'), %(cfg)s, %(img)s,
                   %(submitted_at)s, %(started_at)s, %(ended_at)s, %(end_reason)s)
           ON CONFLICT (job_id, generation) DO UPDATE SET
             exp_id        = COALESCE(data.job_runs.exp_id, EXCLUDED.exp_id),
             jobset_name   = COALESCE(%(jobset)s, data.job_runs.jobset_name),
             cluster       = COALESCE(%(cluster)s, data.job_runs.cluster),
             queue         = COALESCE(%(queue)s, data.job_runs.queue),
             gpus          = COALESCE(%(gpus)s, data.job_runs.gpus),
             strategy      = COALESCE(%(strategy)s, data.job_runs.strategy),
             config_sha256 = COALESCE(%(cfg)s, data.job_runs.config_sha256),
             image_digest  = COALESCE(%(img)s, data.job_runs.image_digest),
             submitted_at  = COALESCE(data.job_runs.submitted_at, %(submitted_at)s),
             started_at    = COALESCE(data.job_runs.started_at, %(started_at)s),
             ended_at      = COALESCE(%(ended_at)s, data.job_runs.ended_at),
             end_reason    = COALESCE(%(end_reason)s, data.job_runs.end_reason)""",
        {
            "job_id": e.job_id, "gen": e.generation, "exp_id": p.get("exp_id"),
            "jobset": p.get("jobset_name"), "jobset_default": f"job-{e.job_id}-g{e.generation:04d}",
            "cluster": e.cluster or p.get("cluster"), "queue": p.get("queue"),
            "gpus": p.get("gpus"), "strategy": p.get("strategy"),
            "cfg": p.get("config_sha256"), "img": p.get("image_digest"),
            "submitted_at": e.occurred_at if submitted else None,
            "started_at": e.occurred_at if started else None,
            "ended_at": e.occurred_at if end_reason else None,
            "end_reason": end_reason,
        },
    )
