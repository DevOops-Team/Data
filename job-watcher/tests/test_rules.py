from conftest import jobset, relocation, workload

from watcher.model import snapshot
from watcher.rules import plan

NOW = "2026-11-12T09:00:00.000Z"
REC = {"devoops/recovery-id": "rec-7c21"}


def keys_types(planned):
    return [(p.key, p.type) for p in planned]


def test_first_submission_full_path_without_submitted():
    s = snapshot(jobset(ready=1, completed="2026-11-12T02:00:00Z"),
                 workload("job-3f9a1c2e-g0001", admitted="2026-11-12T01:00:05Z", cluster="eks-a",
                          pods_ready="2026-11-12T01:01:30Z"))
    out = plan(s, observed_at=NOW)
    # 첫 제출의 job.submitted는 Backend가 보냄 → 감시기는 안 보냄
    assert keys_types(out) == [("admitted", "job.admitted"), ("placed", "job.placed"),
                               ("started", "job.started"), ("end", "job.completed")]
    assert all(p.time_source == "k8s" for p in out)
    assert out[2].occurred_at == "2026-11-12T01:01:30Z"           # 시작 = Workload PodsReady 시각


def test_emit_first_submitted_option():
    out = plan(snapshot(jobset()), observed_at=NOW, emit_first_submitted=True)
    assert keys_types(out) == [("submitted", "job.submitted")]


def test_resubmission_sends_submitted_and_resumed():
    s = snapshot(jobset(gen=2, ready=1, annotations=REC),
                 workload("job-3f9a1c2e-g0002", admitted="2026-11-12T03:00:00Z", pods_ready="2026-11-12T03:01:00Z"))
    kt = keys_types(plan(s, observed_at=NOW))
    assert ("submitted", "job.submitted") in kt and ("started", "job.resumed") in kt


def test_resumed_is_decided_by_recovery_id_not_generation():
    s = snapshot(jobset(gen=2, ready=1), workload("job-3f9a1c2e-g0002", pods_ready="2026-11-12T03:01:00Z"))
    assert ("started", "job.started") in keys_types(plan(s, observed_at=NOW))


def test_start_falls_back_to_observed_time_without_pods_ready():
    out = plan(snapshot(jobset(ready=1)), observed_at=NOW)
    started = [p for p in out if p.key == "started"][0]
    assert (started.occurred_at, started.time_source) == (NOW, "observed")


def test_failed():
    end = plan(snapshot(jobset(failed="2026-11-12T02:00:00Z")), observed_at=NOW)[-1]
    assert (end.type, end.occurred_at) == ("job.failed", "2026-11-12T02:00:00Z")


def test_deleted_with_relocation_request_is_fenced():
    rr = relocation("job-3f9a1c2e-g0001", request_id="rec-7c21", action="moveToOtherCluster")
    s = snapshot(jobset(ready=1, deletion="2026-11-12T02:00:00Z"), deleted=True, relocation=rr)
    end = plan(s, observed_at=NOW)[-1]
    assert (end.type, end.occurred_at) == ("job.fenced", "2026-11-12T02:00:00Z")
    assert end.extra == {"request_id": "rec-7c21", "action": "moveToOtherCluster"}


def test_deleted_without_relocation_request_is_stopped():
    end = plan(snapshot(jobset(ready=1), deleted=True), observed_at=NOW)[-1]
    assert (end.type, end.time_source) == ("job.stopped", "observed")


def test_completed_or_failed_wins_over_delete():
    rr = relocation("job-3f9a1c2e-g0001")
    s = snapshot(jobset(completed="2026-11-12T02:00:00Z"), deleted=True, relocation=rr)
    assert plan(s, observed_at=NOW)[-1].type == "job.completed"
    s = snapshot(jobset(failed="2026-11-12T02:00:00Z"), deleted=True, relocation=rr)
    assert plan(s, observed_at=NOW)[-1].type == "job.failed"


def test_nothing_yet_for_first_submission():
    assert plan(snapshot(jobset()), observed_at=NOW) == []
