"""수집기 동작 (데이터 DB 스키마 · 데이터 파트 설계 3절)."""
import psycopg

from collector.main import Collector
from conftest import FakeConsumer, msg


def run(conn, *batches):
    consumer = FakeConsumer(batches)
    col = Collector(consumer, lambda: conn, sleep=lambda _: None)
    for _ in batches:
        col.run_once()
    return col, consumer


def one(conn, sql, *args):
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchone()


def test_events_stored_and_duplicates_ignored(conn, collector_conn):
    m = msg("fault.detected", topic="gpu.fault", event_id="trace-1", node="ip-10-0-1-23", job_id="3f9a1c2e",
            generation=1, payload={"category": "hardware", "xid": 79, "ts": "2026-11-16T10:00:00Z"})
    col, consumer = run(collector_conn, [m, m])                     # 같은 사건 두 번 (최소 한 번 전달)
    assert one(conn, "SELECT count(*) FROM data.events")[0] == 1
    assert col.stats["inserted"] == 1 and col.stats["duplicate"] == 1
    assert consumer.commits == 1
    row = one(conn, "SELECT topic, node, job_id, generation, payload->>'xid', kafka_offset FROM data.events")
    assert row[:5] == ("gpu.fault", "ip-10-0-1-23", "3f9a1c2e", 1, "79")


def test_rejects_keep_raw(conn, collector_conn):
    bad = [msg("x", raw=b"{not json"), msg("x", raw=b'{"event_id":"a","type":"t"}'),
           msg("x", occurred_at="2026-11-16 10:00:00")]   # 시간대 없음
    col, _ = run(collector_conn, bad)
    assert col.stats["rejected"] == 3
    errs = [r[0] for r in conn.execute("SELECT error FROM data.event_rejects ORDER BY kafka_offset")]
    assert errs[0].startswith("invalid json") and "missing" in errs[1] and "시간대" in errs[2]
    assert one(conn, "SELECT raw FROM data.event_rejects ORDER BY kafka_offset LIMIT 1")[0] == b"{not json"
    assert one(conn, "SELECT count(*) FROM data.events")[0] == 0


def test_experiment_started_and_ended(conn, collector_conn):
    run(collector_conn, [
        msg("experiment.started", occurred_at="2026-11-16T10:00:00.000Z",
            payload={"exp_id": "exp-0001", "name": "xid79-spare-on", "scenario": "scenarios/xid79.yaml",
                     "policy": "ours", "fault_type": "xid79", "spare_enabled": True, "checkpoint_every_steps": 100}),
        msg("experiment.ended", occurred_at="2026-11-16T11:30:00.000Z", payload={"exp_id": "exp-0001"}),
    ])
    r = one(conn, "SELECT name, policy, spare_enabled, checkpoint_every_steps, started_at, ended_at "
                  "FROM data.experiments WHERE exp_id='exp-0001'")
    assert r[:4] == ("xid79-spare-on", "ours", True, 100)
    assert (r[5] - r[4]).total_seconds() == 5400


def test_experiment_missing_fields_keeps_event(conn, collector_conn):
    col, _ = run(collector_conn, [msg("experiment.started", payload={"exp_id": "exp-x"})])
    assert col.stats["derive_skipped"] == 1
    assert one(conn, "SELECT count(*) FROM data.events")[0] == 1          # 원본은 남음
    assert one(conn, "SELECT count(*) FROM data.experiments")[0] == 0


def test_job_runs_lifecycle_and_late_experiment_link(conn, collector_conn):
    env = {"job_id": "3f9a1c2e", "generation": 1, "cluster": "eks-a"}
    run(collector_conn, [
        # 실험 사건보다 job.submitted가 먼저 도착 (토픽 · 파티션이 달라 순서 보장 없음)
        msg("job.submitted", occurred_at="2026-11-16T10:00:00.000Z",
            payload={"jobset_name": "job-3f9a1c2e-g0001", "gpus": 4, "strategy": "fsdp",
                     "queue": "queue-heavy", "exp_id": "exp-0002"}, **env),
        msg("job.started", occurred_at="2026-11-16T10:02:00.000Z", **env),
        msg("job.fenced", occurred_at="2026-11-16T10:30:00.000Z", **env),
    ], [
        msg("experiment.started", payload={"exp_id": "exp-0002", "scenario": "s", "policy": "ours",
                                           "spare_enabled": False}),
    ])
    r = one(conn, "SELECT jobset_name, gpus, strategy, queue, cluster, exp_id, end_reason, "
                  "started_at - submitted_at, ended_at IS NOT NULL FROM data.job_runs")
    assert r[:7] == ("job-3f9a1c2e-g0001", 4, "fsdp", "queue-heavy", "eks-a", "exp-0002", "fenced")
    assert r[7].total_seconds() == 120 and r[8] is True


def test_job_resumed_without_submitted_uses_defaults(conn, collector_conn):
    run(collector_conn, [msg("job.resumed", job_id="3f9a1c2e", generation=2)])
    r = one(conn, "SELECT jobset_name, gpus, strategy, started_at IS NOT NULL FROM data.job_runs")
    assert r == ("job-3f9a1c2e-g0002", 0, "unknown", True)


def test_checkpoint_approved(conn, collector_conn):
    run(collector_conn, [msg("checkpoint.approved", job_id="3f9a1c2e", generation=2, occurred_at="2026-11-16T10:05:00.000Z",
                   payload={"step": 1200, "format": "fsdp-dcp", "world_size": 4,
                            "manifest_key": "jobs/3f9a1c2e/gen-0002/step-00001200/manifest.json",
                            "file_count": 5, "size_bytes": 1500000000, "config_sha256": "9f2c",
                            "upload_ms": 8123, "saved_at": "2026-11-16T10:04:50.000Z"})])
    r = one(conn, "SELECT step, world_size, size_bytes, upload_ms, approved_at - saved_at FROM data.checkpoints")
    assert r[:4] == (1200, 4, 1500000000, 8123) and r[4].total_seconds() == 10


def test_one_bad_record_does_not_block_batch(conn, collector_conn):
    # generation이 int 범위를 넘어 DB 오류 → 그 메시지만 event_rejects, 나머지는 저장
    big = msg("job.started", job_id="j", generation=10**12)
    good = msg("fault.injected", topic="fault.injection", node="n1")
    col, consumer = run(collector_conn, [big, good])
    assert col.stats["rejected_db"] == 1 and col.stats["inserted"] == 1
    assert one(conn, "SELECT error FROM data.event_rejects")[0].startswith("db:")
    assert consumer.commits == 1


def test_db_down_rewinds_and_does_not_commit(conn, collector_conn):
    m1, m2 = msg("fault.injected", topic="fault.injection"), msg("fault.injected", topic="fault.injection")

    def broken():
        raise psycopg.OperationalError("connection refused")
    consumer = FakeConsumer([[m1, m2]])
    col = Collector(consumer, broken, sleep=lambda _: None)
    col.run_once()
    assert consumer.commits == 0                                           # 커밋 안 함
    assert consumer.seeks == [("fault.injection", 0, m1.offset())]         # 다시 읽도록 되감음
    # DB가 돌아오면 같은 묶음을 다시 처리
    col.connect = lambda: collector_conn
    consumer.batches = [[m1, m2]]
    col.run_once()
    assert consumer.commits == 1 and one(conn, "SELECT count(*) FROM data.events")[0] == 2


def test_v_incidents_from_collected_events(conn, collector_conn):
    """수집한 사건으로 측정 뷰가 계산되는지 (ADR-15)."""
    run(collector_conn, [
        msg("fault.injected", topic="fault.injection", node="n1", job_id="j1", occurred_at="2026-11-16T10:00:00.000Z"),
        msg("fault.held", topic="gpu.fault", node="n1", job_id="j1", occurred_at="2026-11-16T10:00:05.000Z",
            payload={"ts": "2026-11-16T10:00:03Z"}),
        msg("fault.reclassified", topic="gpu.fault", node="n1", job_id="j1", occurred_at="2026-11-16T10:00:20.000Z"),
        msg("job.resumed", job_id="j1", generation=2, occurred_at="2026-11-16T10:03:20.000Z"),
    ])
    r = one(conn, "SELECT detect_s, classify_s, recover_s, signal_lag_s FROM data.v_incidents")
    assert [float(x) for x in r] == [5.0, 20.0, 180.0, 2.0]


def test_new_generation_inherits_from_previous(conn, collector_conn):
    """복구로 생긴 세대(job.submitted 없이 job.resumed만 옴)는 이전 세대 값을 이어받는다. queue · cluster는 제외."""
    env1 = {"job_id": "j-inh", "generation": 1, "cluster": "eks-a"}
    run(collector_conn, [
        msg("experiment.started", payload={"exp_id": "exp-inh", "scenario": "s", "policy": "ours",
                                           "spare_enabled": True}),
        msg("job.submitted", payload={"jobset_name": "job-j-inh-g0001", "gpus": 4, "strategy": "fsdp",
                                      "queue": "queue-heavy", "exp_id": "exp-inh", "config_sha256": "cfg1"}, **env1),
        msg("job.fenced", **env1),
        msg("job.resumed", job_id="j-inh", generation=2, cluster="eks-b"),
    ])
    r = one(conn, "SELECT exp_id, jobset_name, gpus, strategy, config_sha256, queue, cluster "
                  "FROM data.job_runs WHERE job_id='j-inh' AND generation=2")
    assert r == ("exp-inh", "job-j-inh-g0002", 4, "fsdp", "cfg1", None, "eks-b")


def test_late_experiment_links_all_generations(conn, collector_conn):
    """실험 사건이 늦게 와도 그 작업의 모든 세대에 exp_id가 붙는다."""
    run(collector_conn, [
        msg("job.submitted", payload={"gpus": 2, "strategy": "fsdp", "exp_id": "exp-late"},
            job_id="j-late", generation=1),
        msg("job.fenced", job_id="j-late", generation=1),
        msg("job.resumed", job_id="j-late", generation=2),
    ], [
        msg("experiment.started", payload={"exp_id": "exp-late", "scenario": "s", "policy": "ours",
                                           "spare_enabled": False}),
    ])
    rows = conn.execute("SELECT generation, exp_id, gpus FROM data.job_runs WHERE job_id='j-late' "
                        "ORDER BY generation").fetchall()
    assert rows == [(1, "exp-late", 2), (2, "exp-late", 2)]
