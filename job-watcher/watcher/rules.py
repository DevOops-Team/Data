"""세대 하나의 상태(Snapshot) → 보낼 사건 목록. 쿠버네티스 · Kafka 없이 시험할 수 있게 순수 함수로 둔다.

사건 (세대마다 종류별로 최대 1번, key가 event_id 끝부분)
  key        type                          판정                                         occurred_at
  submitted  job.submitted                 JobSet이 보임                                creationTimestamp
  admitted   job.admitted                  Workload Admitted=True (Kueue 쿼터 확보)      그 조건의 lastTransitionTime
  placed     job.placed                    Workload status.clusterName (MultiKueue)     admissionCheck Ready 시각, 없으면 Admitted 시각
  started    job.started (세대 1)          Workload PodsReady=True                      그 조건의 lastTransitionTime
             job.resumed (세대 2 이상)     없으면 JobSet replicatedJob 전부 ready         감시기가 본 시각 (time_source=observed)
  end        job.completed                 JobSet Completed=True                        lastTransitionTime
             job.fenced                    JobSet Failed=True 이고 같은 작업의 다음 세대가 이미 있음
             job.failed                    JobSet Failed=True                            lastTransitionTime
             job.stopped / job.cancelled   종료 조건 없이 삭제됨. 주석 devoops/stop-reason=cancel → cancelled, 그 외 stopped
                                                                                         deletionTimestamp, 없으면 본 시각

종료(end)는 세대마다 하나: completed · failed · fenced · stopped · cancelled가 같은 key("end")를 쓴다.
감시기가 재시작해 판정이 달라져도(예: failed → fenced) 수집기는 먼저 들어간 하나만 남긴다 (event_id 중복 무시).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import Snapshot


@dataclass(frozen=True)
class Planned:
    key: str
    type: str
    occurred_at: str
    time_source: str = "k8s"                 # k8s: 쿠버네티스가 찍은 시각 · observed: 감시기가 본 시각
    extra: dict[str, Any] = field(default_factory=dict)


def plan(s: Snapshot, *, newer_generation_exists: bool, observed_at: str) -> list[Planned]:
    out: list[Planned] = []

    def add(key: str, type_: str, at: str | None, **extra: Any) -> None:
        if at:
            out.append(Planned(key, type_, at, "k8s", extra))
        else:
            out.append(Planned(key, type_, observed_at, "observed", extra))

    add("submitted", "job.submitted", s.created_at)
    if s.admitted_at is not None:
        add("admitted", "job.admitted", s.admitted_at)
    if s.cluster:
        add("placed", "job.placed", s.placed_at)

    start_type = "job.started" if s.generation <= 1 else "job.resumed"
    if s.pods_ready_at is not None:
        add("started", start_type, s.pods_ready_at, basis="workload_pods_ready")
    elif s.all_ready:
        add("started", start_type, None, basis="jobset_all_ready")

    if s.completed_at is not None:
        add("end", "job.completed", s.completed_at)
    elif s.failed_at is not None:
        if newer_generation_exists:
            add("end", "job.fenced", s.failed_at, message=s.failed_message, basis="newer_generation_exists")
        else:
            add("end", "job.failed", s.failed_at, message=s.failed_message)
    elif s.deleted:
        if s.stop_reason == "cancel":
            add("end", "job.cancelled", s.deleted_at, stop_reason=s.stop_reason)
        else:
            add("end", "job.stopped", s.deleted_at, stop_reason=s.stop_reason or "unknown")
    return out
