"""측정 (GPU-187): 사건 · 세대 · 체크포인트로 MTTD · MTTR · ETTR · 날아간 step 계산 → data.results."""
from __future__ import annotations

import json

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
