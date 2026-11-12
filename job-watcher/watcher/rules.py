"""세대 하나의 상태(Snapshot) → 보낼 사건 목록. 쿠버네티스 · Kafka 없이 시험할 수 있게 순수 함수로 둔다.
판정 기준: 김성모 11/12 답변 (스케줄링) + ETTR 시작 시각은 PodsReady (데이터).

사건 (세대마다 종류별로 최대 1번, key가 event_id 끝부분)
  key        type                          판정                                              occurred_at
  submitted  job.submitted                 재제출본(주석 devoops/recovery-id 있음)이 보임.     creationTimestamp
                                           첫 제출(세대 1)은 Backend가 이미 보내므로 기본으로 안 보냄
  admitted   job.admitted                  Workload Admitted=True (Kueue 쿼터 확보 · 입장)      조건 lastTransitionTime
  placed     job.placed                    Workload status.clusterName (MultiKueue 배치)       admissionCheck Ready 시각, 없으면 Admitted 시각
  started    job.started   (recovery-id 없음)  Workload PodsReady=True                         조건 lastTransitionTime
             job.resumed   (recovery-id 있음)  없으면 JobSet replicatedJob 전부 ready            감시기가 본 시각 (time_source=observed)
  end        job.completed                 JobSet Completed=True                              조건 시각
             job.failed                    JobSet Failed=True                                 조건 시각
             job.fenced                    종료 조건 없이 삭제 + 이 JobSet을 가리키는 RelocationRequest 있음
                                           (payload: request_id · action)                      deletionTimestamp, 없으면 본 시각
             job.stopped                   종료 조건 없이 삭제 + RelocationRequest 없음 (사용자 · Backend 중지)

종료(end)는 세대마다 하나: completed · failed · fenced · stopped가 같은 key("end")를 쓴다.
이미 끝난(Completed · Failed) JobSet을 나중에 지우면 새 사건이 없다. 감시기가 재시작해 판정이 달라져도
수집기는 먼저 들어간 하나만 남긴다 (event_id 중복 무시).
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


def plan(s: Snapshot, *, observed_at: str, emit_first_submitted: bool = False) -> list[Planned]:
    out: list[Planned] = []

    def add(key: str, type_: str, at: str | None, **extra: Any) -> None:
        extra = {k: v for k, v in extra.items() if v is not None}
        if at:
            out.append(Planned(key, type_, at, "k8s", extra))
        else:
            out.append(Planned(key, type_, observed_at, "observed", extra))

    resubmitted = bool(s.recovery_id)
    if resubmitted or emit_first_submitted:
        add("submitted", "job.submitted", s.created_at)
    if s.admitted_at is not None:
        add("admitted", "job.admitted", s.admitted_at)
    if s.cluster:
        add("placed", "job.placed", s.placed_at)

    start_type = "job.resumed" if resubmitted else "job.started"
    if s.pods_ready_at is not None:
        add("started", start_type, s.pods_ready_at, basis="workload_pods_ready")
    elif s.all_ready:
        add("started", start_type, None, basis="jobset_all_ready")

    if s.completed_at is not None:
        add("end", "job.completed", s.completed_at)
    elif s.failed_at is not None:
        add("end", "job.failed", s.failed_at, message=s.failed_message)
    elif s.deleted:
        if s.relocated:
            add("end", "job.fenced", s.deleted_at, request_id=s.relocation_request_id, action=s.relocation_action)
        else:
            add("end", "job.stopped", s.deleted_at)
    return out
