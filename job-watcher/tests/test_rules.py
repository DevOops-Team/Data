from conftest import jobset, workload

from watcher.model import snapshot
from watcher.rules import plan

NOW = "2026-11-12T09:00:00.000Z"


def keys_types(planned):
    return [(p.key, p.type) for p in planned]


def test_full_normal_path_gen1():
    s = snapshot(jobset(ready=1, completed="2026-11-12T02:00:00Z"),
                 workload("job-3f9a1c2e-g0001", admitted="2026-11-12T01:00:05Z", cluster="eks-a",
                          pods_ready="2026-11-12T01:01:30Z"))
    out = plan(s, newer_generation_exists=False, observed_at=NOW)
    assert keys_types(out) == [("submitted", "job.submitted"), ("admitted", "job.admitted"),
                               ("placed", "job.placed"), ("started", "job.started"), ("end", "job.completed")]
    assert all(p.time_source == "k8s" for p in out)
    assert out[3].occurred_at == "2026-11-12T01:01:30Z"           # 시작 = Workload PodsReady 시각


def test_generation_2_start_is_resumed():
    s = snapshot(jobset(gen=2, ready=1), workload("job-3f9a1c2e-g0002", pods_ready="2026-11-12T03:00:00Z"))
    assert ("started", "job.resumed") in keys_types(plan(s, newer_generation_exists=False, observed_at=NOW))


def test_start_falls_back_to_observed_time_without_pods_ready():
    out = plan(snapshot(jobset(ready=1)), newer_generation_exists=False, observed_at=NOW)
    started = [p for p in out if p.key == "started"][0]
    assert (started.occurred_at, started.time_source) == (NOW, "observed")


def test_failed_vs_fenced():
    s = snapshot(jobset(failed="2026-11-12T02:00:00Z"))
    assert plan(s, newer_generation_exists=False, observed_at=NOW)[-1].type == "job.failed"
    assert plan(s, newer_generation_exists=True, observed_at=NOW)[-1].type == "job.fenced"


def test_deleted_stopped_or_cancelled():
    s = snapshot(jobset(deletion="2026-11-12T02:00:00Z", annotations={"devoops/stop-reason": "relocate"}),
                 deleted=True)
    end = plan(s, newer_generation_exists=False, observed_at=NOW)[-1]
    assert (end.type, end.occurred_at, end.extra["stop_reason"]) == ("job.stopped", "2026-11-12T02:00:00Z", "relocate")

    s = snapshot(jobset(annotations={"devoops/stop-reason": "cancel"}), deleted=True)
    end = plan(s, newer_generation_exists=False, observed_at=NOW)[-1]
    assert (end.type, end.time_source) == ("job.cancelled", "observed")


def test_completed_wins_over_delete():
    s = snapshot(jobset(completed="2026-11-12T02:00:00Z"), deleted=True)
    assert plan(s, newer_generation_exists=False, observed_at=NOW)[-1].type == "job.completed"


def test_nothing_running_yet_only_submitted():
    assert keys_types(plan(snapshot(jobset()), newer_generation_exists=False, observed_at=NOW)) == [
        ("submitted", "job.submitted")]
