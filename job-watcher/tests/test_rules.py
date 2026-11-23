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


RESTART = {"devoops/restart-reason": "user", "devoops/restart-requested-at": "2026-11-20T03:00:00Z"}


def test_user_restart_sends_submitted_and_resumed_with_reason():
    s = snapshot(jobset(gen=2, ready=1, annotations=RESTART),
                 workload("job-3f9a1c2e-g0002", admitted="2026-11-20T03:00:05Z", pods_ready="2026-11-20T03:01:00Z"))
    out = plan(s, observed_at=NOW)
    assert ("submitted", "job.submitted") in keys_types(out) and ("started", "job.resumed") in keys_types(out)
    for p in out:
        if p.key in ("submitted", "started"):
            assert p.extra["restart_reason"] == "user"
            assert p.extra["restart_requested_at"] == "2026-11-20T03:00:00Z"


def test_recovery_resubmission_has_no_restart_reason():
    s = snapshot(jobset(gen=2, ready=1, annotations=REC), workload("job-3f9a1c2e-g0002", pods_ready="2026-11-12T03:01:00Z"))
    assert all("restart_reason" not in p.extra for p in plan(s, observed_at=NOW))


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


USER_STOP = {"devoops/stop-reason": "user", "devoops/stop-requested-at": "2026-11-18T01:59:58Z"}


def test_user_stop_annotation_is_cancelled():
    s = snapshot(jobset(ready=1, deletion="2026-11-18T02:00:00Z", annotations=USER_STOP), deleted=True)
    end = plan(s, observed_at=NOW)[-1]
    assert (end.key, end.type, end.occurred_at) == ("end", "job.cancelled", "2026-11-18T02:00:00Z")
    assert end.extra == {"reason": "user", "requested_at": "2026-11-18T01:59:58Z"}


def test_relocation_wins_over_user_stop_and_other_reasons_are_stopped():
    rr = relocation("job-3f9a1c2e-g0001", request_id="rec-7c21", action="moveToSpare")
    s = snapshot(jobset(ready=1, annotations=USER_STOP), deleted=True, relocation=rr)
    assert plan(s, observed_at=NOW)[-1].type == "job.fenced"                  # 복구가 먼저
    s = snapshot(jobset(ready=1, annotations={"devoops/stop-reason": "admin"}), deleted=True)
    assert plan(s, observed_at=NOW)[-1].type == "job.stopped"


def test_user_stop_annotation_without_delete_sends_nothing():
    assert plan(snapshot(jobset(annotations=USER_STOP)), observed_at=NOW) == []


def test_completed_or_failed_wins_over_delete():
    rr = relocation("job-3f9a1c2e-g0001")
    s = snapshot(jobset(completed="2026-11-12T02:00:00Z"), deleted=True, relocation=rr)
    assert plan(s, observed_at=NOW)[-1].type == "job.completed"
    s = snapshot(jobset(failed="2026-11-12T02:00:00Z"), deleted=True, relocation=rr)
    assert plan(s, observed_at=NOW)[-1].type == "job.failed"


def test_nothing_yet_for_first_submission():
    assert plan(snapshot(jobset()), observed_at=NOW) == []


# ---------- Kueue가 내림 · 비활성화 (GPU-259) ----------
WL = "job-3f9a1c2e-g0001"


def test_pods_ready_timeout_eviction_is_reported_each_time():
    first = plan(snapshot(jobset(), workload(WL, admitted="2026-11-23T05:00:00Z", cluster="eks-a",
                                             evicted="2026-11-23T05:10:00Z", requeue_count=1)), observed_at=NOW)
    ev = [p for p in first if p.type == "job.evicted"]
    assert [p.key for p in ev] == ["evicted-20261123T051000Z"]
    assert ev[0].occurred_at == "2026-11-23T05:10:00Z" and ev[0].time_source == "k8s"
    assert ev[0].extra["reason"] == "PodsReadyTimeout" and ev[0].extra["requeue_count"] == 1
    assert "PodsReady timeout" in ev[0].extra["message"]
    # 다시 넣은 뒤 또 시간 초과 → 다른 시각 = 다른 사건
    second = plan(snapshot(jobset(), workload(WL, admitted="2026-11-23T05:12:00Z", cluster="eks-a",
                                              evicted="2026-11-23T05:22:00Z", requeue_count=2)), observed_at=NOW)
    assert [p.key for p in second if p.type == "job.evicted"] == ["evicted-20261123T052200Z"]
    assert not any(p.key == "end" for p in second)                       # 내림은 종료가 아님


def test_deactivated_after_requeue_limit():
    out = plan(snapshot(jobset(), workload(WL, admitted="2026-11-23T05:00:00Z", evicted="2026-11-23T05:40:00Z",
                                           evicted_reason="InactiveWorkload", requeue_count=3, active=False)),
               observed_at=NOW)
    de = [p for p in out if p.type == "job.deactivated"]
    assert len(de) == 1 and de[0].key == "deactivated" and de[0].time_source == "observed"
    assert de[0].extra == {"reason": "InactiveWorkload", "message": f"Exceeded the PodsReady timeout default/{WL}",
                           "requeue_count": 3}


def test_no_eviction_events_on_normal_run_or_after_end():
    normal = plan(snapshot(jobset(ready=1), workload(WL, admitted="2026-11-23T05:00:00Z", cluster="eks-a",
                                                     pods_ready="2026-11-23T05:01:00Z")), observed_at=NOW)
    assert not any(p.type in ("job.evicted", "job.deactivated") for p in normal)
    done = plan(snapshot(jobset(completed="2026-11-23T06:00:00Z"), workload(WL, active=False)), observed_at=NOW)
    assert not any(p.type == "job.deactivated" for p in done)            # 이미 끝난 세대는 비활성화 안 보냄
