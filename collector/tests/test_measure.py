"""측정 (GPU-187): 사건 · 세대 · 체크포인트로 MTTD · MTTR · ETTR · 날아간 step 계산 → data.results."""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from collector import measure

_n = {"off": 0}


def event(conn, type_, at, *, job=None, gen=None, node=None, payload=None):
    _n["off"] += 1
    conn.execute(
        "INSERT INTO data.events (event_id, schema_version, type, topic, occurred_at, node, job_id, generation, payload, "
        "kafka_partition, kafka_offset) VALUES (%s,1,%s,'t',%s,%s,%s,%s,%s,0,%s)",
        (f"e{_n['off']}", type_, at, node, job, gen, json.dumps(payload or {}), _n["off"]))


def run_row(conn, job, gen, submitted, started, ended, reason):
    conn.execute(
        "INSERT INTO data.job_runs (job_id, generation, jobset_name, gpus, strategy, submitted_at, started_at, ended_at, "
        "end_reason) VALUES (%s,%s,%s,2,'fsdp',%s,%s,%s,%s)",
        (job, gen, f"job-{job}-g{gen:04d}", submitted, started, ended, reason))


def ckpt(conn, job, gen, step, approved):
    conn.execute(
        "INSERT INTO data.checkpoints (job_id, generation, step, format, world_size, manifest_key, file_count, size_bytes, "
        "config_sha256, saved_at, approved_at) VALUES (%s,%s,%s,'fsdp-dcp',2,'k',4,1,'c',%s,%s)",
        (job, gen, step, approved, approved))


T = "2026-11-17T10:{}+09:00".format


@pytest.fixture
def scenario(conn):
    """작업 J: 세대 1에서 step 400 승인 뒤 장애, step 500까지 진행했다가 펜싱 → 세대 2가 400에서 재개해 완료."""
    run_row(conn, "J", 1, T("00:00"), T("01:00"), T("20:00"), "fenced")
    run_row(conn, "J", 2, T("20:30"), T("23:00"), T("40:00"), "completed")
    ckpt(conn, "J", 1, 400, T("15:00"))
    for step, at in ((200, "08:00"), (400, "15:00"), (500, "18:30")):
        event(conn, "job.progress", T(at), job="J", gen=1, payload={"step": step, "step_seconds": 1.2})
    event(conn, "fault.injected", T("19:00"), job="J", node="n1")
    event(conn, "fault.detected", T("19:30"), job="J", node="n1", payload={"ts": T("19:10")})
    event(conn, "job.resumed", T("23:00"), job="J", gen=2)
    return conn


def test_one_failover_gives_adr15_numbers(scenario):
    r = measure.compute_job(scenario.cursor(), "J")
    assert (r.generations, r.incidents) == (2, 1)
    assert (r.detect_s, r.classify_s, r.recover_s) == (30.0, 30.0, 210.0)   # 주입 → 감지 · 분류, 분류 → 재개
    assert r.lost_steps == 100                                              # 500까지 갔다가 400에서 재개
    assert r.wait_s == 60.0
    assert r.downtime_s == 240.0                                            # 장애 시작(10:19) → 세대 2 시작(10:23)
    assert r.recompute_s == pytest.approx(120.0)                            # 100 step × 1.2초
    assert r.ettr == pytest.approx((2340 - 240 - 120) / 2340, abs=1e-4)     # 할당 10:01 → 10:40


def test_run_writes_results_and_is_repeatable(scenario):
    out1 = measure.run(scenario, "exp-1", jobs=["J"], create_exp="시험")
    out2 = measure.run(scenario, "exp-1", jobs=["J"])                        # 다시 돌려도 같은 값으로 덮어씀
    assert out1["summary"] == out2["summary"]
    assert out1["summary"]["mttd_s"] == 30.0 and out1["summary"]["mttr_s"] == 240.0
    rows = scenario.execute("SELECT job_id, lost_steps, incidents, ettr, calc_version FROM data.results").fetchall()
    assert len(rows) == 1 and rows[0][:3] == ("J", 100, 1) and float(rows[0][3]) == pytest.approx(0.8462, abs=1e-4)


def test_jobs_by_exp_id_or_time_range(scenario):
    run_row(scenario, "K", 1, T("50:00"), T("51:00"), T("55:00"), "completed")
    scenario.execute("INSERT INTO data.experiments (exp_id, name, scenario, policy, spare_enabled) VALUES ('e2','x','s','ours',true)")
    scenario.execute("UPDATE data.job_runs SET exp_id = 'e2' WHERE job_id = 'K'")
    cur = scenario.cursor()
    assert measure.select_jobs(cur, "e2", None, None, None) == ["K"]
    assert measure.select_jobs(cur, "x", None, T("00:00"), T("30:00")) == ["J"]


def test_no_failure_job_has_full_ettr_and_no_incidents(conn):
    run_row(conn, "Q", 1, T("00:00"), T("00:30"), T("10:30"), "completed")
    r = measure.compute_job(conn.cursor(), "Q")
    assert (r.incidents, r.lost_steps, r.ettr, r.wait_s) == (0, 0, 1.0, 30.0)


def test_unknown_experiment_needs_create_flag_and_dry_run_writes_nothing(scenario):
    with pytest.raises(SystemExit):
        measure.run(scenario, "nope", jobs=["J"])
    out = measure.run(scenario, "new", jobs=["J"], create_exp="x", dry_run=True)
    assert out["summary"]["incidents"] == 1
    assert scenario.execute("SELECT count(*) FROM data.results").fetchone()[0] == 0
    assert scenario.execute("SELECT count(*) FROM data.experiments WHERE exp_id = 'new'").fetchone()[0] == 0


def test_exp_id_from_submit_without_experiment_started(scenario):
    """투입기 없이 exp_id만 붙여 제출 (submit_job.py --exp-id): 실험 행이 없어 job_runs.exp_id는 비어 있다.
    measure --create-exp가 실험 행을 만들고 세대 1 · 2 줄을 모두 연결해 --jobs 없이 찾는다."""
    event(scenario, "job.submitted", T("00:00"), job="J", gen=1, payload={"exp_id": "e2e-x"})
    cur = scenario.cursor()
    assert measure.select_jobs(cur, "e2e-x", None, None, None) == ["J"]          # 연결 전에도 제출 사건으로 찾음
    out = measure.run(scenario, "e2e-x", create_exp="11/17 E2E")
    assert out["linked_job_runs"] == 2 and [j["job_id"] for j in out["jobs"]] == ["J"]
    assert scenario.execute("SELECT DISTINCT exp_id FROM data.job_runs WHERE job_id = 'J'").fetchall() == [("e2e-x",)]
    assert measure.run(scenario, "e2e-x")["linked_job_runs"] == 0                 # 다시 돌려도 그대로


# ---- 장애 시작: 수동 주입 기록 · 분류기 payload.ts (GPU-239, db/007) ----

def _scenario_without_injection_event(conn):
    """scenario와 같지만 fault.injected 사건이 없다 (ADR-33: 주입은 CLI 수동, 사건 없음)."""
    run_row(conn, "J", 1, T("00:00"), T("01:00"), T("20:00"), "fenced")
    run_row(conn, "J", 2, T("20:30"), T("23:00"), T("40:00"), "completed")
    ckpt(conn, "J", 1, 400, T("15:00"))
    for step, at in ((200, "08:00"), (400, "15:00"), (500, "18:30")):
        event(conn, "job.progress", T(at), job="J", gen=1, payload={"step": step, "step_seconds": 1.2})
    event(conn, "fault.detected", T("19:30"), job="J", node="n1", payload={"ts": T("19:10"), "dedup_key": "d1"})
    event(conn, "job.resumed", T("23:00"), job="J", gen=2)


def test_manual_injection_record_gives_same_numbers(conn):
    _scenario_without_injection_event(conn)
    from collector import injection
    injection.add(conn.cursor(), at=injection.parse_at("2026-11-17 10:19:00"), node="n1", kind="trainer_kill", job_id="J")
    rows = conn.execute("SELECT start_source, detect_s, classify_s, recover_s FROM data.v_incidents").fetchall()
    assert [(r[0], *map(float, r[1:])) for r in rows] == [("manual", 30.0, 30.0, 210.0)]   # 신호는 주입의 감지로만
    r = measure.compute_job(conn.cursor(), "J")
    assert (r.incidents, r.detect_s, r.downtime_s) == (1, 30.0, 240.0)


def test_natural_fault_starts_at_signal_payload_ts(conn):
    _scenario_without_injection_event(conn)
    rows = conn.execute("SELECT start_source, injected_at, detect_s, classify_s, signal_lag_s FROM data.v_incidents").fetchall()
    assert len(rows) == 1 and rows[0][0] == "signal"
    assert rows[0][1] == datetime.fromisoformat(T("19:10"))                        # payload.ts (10:19:10 KST)
    assert [float(x) for x in rows[0][2:]] == [20.0, 20.0, 20.0]                     # 신호 시각 − 원본 시각


def test_same_dedup_key_is_one_natural_fault(conn):
    event(conn, "fault.held", T("05:00"), job="Z", node="n9", payload={"ts": T("04:50"), "dedup_key": "dz"})
    event(conn, "fault.reclassified", T("05:40"), job="Z", node="n9", payload={"dedup_key": "dz"})
    event(conn, "fault.detected", T("05:41"), job="Z", node="n9", payload={"ts": T("04:50"), "dedup_key": "dz"})
    rows = conn.execute("SELECT start_source, detect_s, classify_s FROM data.v_incidents").fetchall()
    assert [(r[0], float(r[1]), float(r[2])) for r in rows] == [("signal", 10.0, 50.0)]  # held 감지, reclassified 분류


def test_injection_cli_add_list_delete(conn):
    from collector import injection
    cur = conn.cursor()
    assert injection.parse_at("2026-11-20 14:03:21").utcoffset().total_seconds() == 9 * 3600   # 시간대 없으면 KST
    iid = injection.add(cur, at=injection.parse_at("2026-11-20T05:03:21Z"), node=" n1 ", kind="xid", note="79")
    got = injection.recent(cur)
    assert [(g["injection_id"], g["node"], g["kind"]) for g in got] == [(iid, "n1", "xid")]
    with pytest.raises(ValueError):
        injection.add(cur, at=injection.parse_at("2026-11-20 14:00"), node="n1", kind="bogus")
    assert injection.delete(cur, iid) and injection.recent(cur) == []
