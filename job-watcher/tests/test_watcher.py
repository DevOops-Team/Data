import json

from conftest import jobset, relocation, workload

from collector.envelope import parse
from watcher.main import ADDED, DELETED, MODIFIED, SYNC, Watcher

NOW = "2026-11-12T09:00:00.000Z"
JS1 = "job-3f9a1c2e-g0001"
JS2 = "job-3f9a1c2e-g0002"
REC = {"devoops/recovery-id": "rec-7c21"}


def make(pub, **kw):
    w = Watcher(pub, clock=lambda: NOW, **kw)
    w.on_jobset(SYNC, [])
    w.on_workload(SYNC, [])
    w.on_relocation(SYNC, [])
    return w


def test_lifecycle_sends_each_event_once(pub):
    w = make(pub)
    w.on_jobset(ADDED, jobset())
    w.on_workload(ADDED, workload(JS1))
    w.on_workload(MODIFIED, workload(JS1, admitted="2026-11-12T01:00:05Z", cluster="eks-a"))
    w.on_workload(MODIFIED, workload(JS1, admitted="2026-11-12T01:00:05Z", cluster="eks-a",
                                     pods_ready="2026-11-12T01:01:30Z"))
    w.on_jobset(MODIFIED, jobset(ready=1))
    w.on_jobset(MODIFIED, jobset(ready=1))                                   # 같은 상태 반복 → 다시 안 보냄
    w.on_jobset(MODIFIED, jobset(ready=0, completed="2026-11-12T02:00:00Z"))
    w.on_jobset(DELETED, jobset(completed="2026-11-12T02:00:00Z", deletion="2026-11-12T02:05:00Z"))
    assert pub.types() == ["job.admitted", "job.placed", "job.started", "job.completed"]
    assert {k for _, k, _ in pub.sent} == {"3f9a1c2e"}                       # 파티션 키 = 작업 ID
    assert {t for t, _, _ in pub.sent} == {"job.lifecycle"}


def test_events_pass_collector_envelope(pub):
    w = make(pub)
    w.on_workload(ADDED, workload(JS1, admitted="2026-11-12T01:00:05Z", cluster="eks-a"))
    w.on_jobset(ADDED, jobset(ready=1))
    for i, (_, _, v) in enumerate(pub.sent):
        e = parse(json.dumps(v).encode(), "job.lifecycle", 0, i)              # 수집기가 받는지
        assert (e.job_id, e.generation, e.source) == ("3f9a1c2e", 1, "job-watcher")
    started = [v for _, _, v in pub.sent if v["type"] == "job.started"][0]
    assert started["event_id"] == "job-3f9a1c2e-g0001-started"
    assert started["cluster"] == "eks-a"
    assert started["payload"]["gpus"] == 4 and started["payload"]["exp_id"] == "exp-0001"


def test_waits_for_all_initial_lists(pub):
    w = Watcher(pub, clock=lambda: NOW)
    w.on_jobset(SYNC, [jobset(ready=1)])
    w.on_workload(SYNC, [workload(JS1, pods_ready="2026-11-12T01:01:30Z")])
    assert pub.sent == []                                                    # 재배치 요청 목록 전에는 판정 안 함
    w.on_relocation(SYNC, [])
    started = [v for _, _, v in pub.sent if v["type"] == "job.started"][0]
    assert started["occurred_at"] == "2026-11-12T01:01:30Z"                  # 본 시각이 아니라 PodsReady 시각
    assert started["payload"]["time_source"] == "k8s"


def test_relocation_delete_is_fenced_then_resubmission_is_resumed(pub):
    w = make(pub)
    w.on_jobset(ADDED, jobset(gen=1, ready=1))
    w.on_relocation(ADDED, relocation(JS1, request_id="rec-7c21", action="moveToSpare"))
    w.on_jobset(DELETED, jobset(gen=1, ready=1, deletion="2026-11-12T02:00:00Z"))
    fenced = pub.sent[-1][2]
    assert (fenced["type"], fenced["event_id"], fenced["occurred_at"]) == (
        "job.fenced", "job-3f9a1c2e-g0001-end", "2026-11-12T02:00:00Z")
    assert (fenced["payload"]["request_id"], fenced["payload"]["action"]) == ("rec-7c21", "moveToSpare")

    w.on_jobset(ADDED, jobset(gen=2, annotations=REC, created="2026-11-12T02:01:00Z"))
    w.on_workload(ADDED, workload(JS2, admitted="2026-11-12T02:01:05Z", cluster="eks-b",
                                  pods_ready="2026-11-12T02:02:30Z"))
    w.on_jobset(MODIFIED, jobset(gen=2, ready=1, annotations=REC, created="2026-11-12T02:01:00Z"))
    gen2 = [v for _, _, v in pub.sent if v["generation"] == 2]
    assert [v["type"] for v in gen2] == ["job.submitted", "job.admitted", "job.placed", "job.resumed"]
    assert gen2[-1]["payload"]["recovery_id"] == "rec-7c21"


def test_delete_without_relocation_is_stopped(pub):
    w = make(pub)
    w.on_jobset(ADDED, jobset(ready=1))
    w.on_jobset(DELETED, jobset(ready=1, deletion="2026-11-12T02:00:00Z"))
    assert pub.sent[-1][2]["type"] == "job.stopped"


def test_failed_then_relocated_delete_sends_nothing_more(pub):
    w = make(pub)
    w.on_jobset(ADDED, jobset(ready=1))
    w.on_jobset(MODIFIED, jobset(ready=0, failed="2026-11-12T02:00:00Z"))
    w.on_relocation(ADDED, relocation(JS1))
    w.on_jobset(DELETED, jobset(failed="2026-11-12T02:00:00Z", deletion="2026-11-12T02:01:00Z"))
    ends = [v["type"] for _, _, v in pub.sent if v["event_id"].endswith("-end")]
    assert ends == ["job.failed"]                                            # 이미 끝난 잡을 지우면 사건 없음


def test_relist_detects_delete_while_disconnected(pub):
    w = make(pub)
    w.on_jobset(ADDED, jobset(ready=1))
    w.on_jobset(SYNC, [])                                                    # 다시 list했더니 없어짐
    end = pub.sent[-1][2]
    assert (end["type"], end["payload"]["time_source"]) == ("job.stopped", "observed")


def test_delete_during_startup_is_not_lost(pub):
    w = Watcher(pub, clock=lambda: NOW)
    w.on_jobset(SYNC, [jobset(ready=1)])
    w.on_jobset(DELETED, jobset(ready=1, deletion="2026-11-12T02:00:00Z"))
    w.on_workload(SYNC, [])
    assert pub.sent == []
    w.on_relocation(SYNC, [relocation(JS1)])
    assert pub.types()[-1] == "job.fenced"                                   # 시작 중 삭제도 재배치 요청으로 판정


def test_restart_resends_same_ids(pub):
    a = make(pub)
    a.on_jobset(ADDED, jobset(ready=1, completed="2026-11-12T02:00:00Z"))
    first = [v["event_id"] for _, _, v in pub.sent]
    pub.sent.clear()
    b = Watcher(pub, clock=lambda: "2026-11-12T10:00:00.000Z")               # 재시작
    b.on_jobset(SYNC, [jobset(ready=1, completed="2026-11-12T02:00:00Z")])
    b.on_workload(SYNC, [])
    b.on_relocation(SYNC, [])
    assert [v["event_id"] for _, _, v in pub.sent] == first                  # 수집기가 중복으로 걸러냄


def test_ignores_non_training_jobsets(pub):
    w = make(pub)
    other = jobset()
    other["metadata"]["name"] = "kube-system-thing"
    other["metadata"]["labels"] = {}
    w.on_jobset(ADDED, other)
    assert pub.sent == []
