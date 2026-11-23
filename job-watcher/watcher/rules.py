"""세대 하나의 상태(Snapshot) → 보낼 사건 목록. 쿠버네티스 · Kafka 없이 시험할 수 있게 순수 함수로 둔다.
판정 기준: 김성모 11/12 답변 (스케줄링) + ETTR 시작 시각은 PodsReady (데이터).

사건 (세대마다 종류별로 최대 1번, key가 event_id 끝부분)
  key        type                          판정                                              occurred_at
  submitted  job.submitted                 재제출본이 보임 — 복구 재배치(주석 devoops/recovery-id)  creationTimestamp
                                           또는 사용자 다시 시작(주석 devoops/restart-reason, GPU-234).
                                           첫 제출(세대 1)은 Backend가 이미 보내므로 기본으로 안 보냄
  admitted   job.admitted                  Workload Admitted=True (Kueue 쿼터 확보 · 입장)      조건 lastTransitionTime
  placed     job.placed                    Workload status.clusterName (MultiKueue 배치)       admissionCheck Ready 시각, 없으면 Admitted 시각
  started    job.started   (recovery-id 없음)  Workload PodsReady=True                         조건 lastTransitionTime
             job.resumed   (재제출본)          없으면 JobSet replicatedJob 전부 ready            감시기가 본 시각 (time_source=observed)
             사용자 다시 시작이면 submitted · resumed payload에 restart_reason · restart_requested_at
             → 측정(v_incidents)은 이 resumed를 장애 복구로 세지 않는다 (db/006)
  end        job.completed                 JobSet Completed=True                              조건 시각
             job.failed                    JobSet Failed=True                                 조건 시각
             job.fenced                    종료 조건 없이 삭제 + 이 JobSet을 가리키는 RelocationRequest 있음
                                           (payload: request_id · action)                      deletionTimestamp, 없으면 본 시각
             job.cancelled                 종료 조건 없이 삭제 + RelocationRequest 없음 + 주석 devoops/stop-reason=user
                                           (Backend 사용자 중단, payload: reason · requested_at, 한송이 11/18 동의)
             job.stopped                   종료 조건 없이 삭제 + RelocationRequest 없음 + 중단 주석 없음 (수동 삭제 등)

Kueue가 내림 · 비활성화 (GPU-259, GPU-256 후속) — 종료가 아님. 세대 안에서 Kueue가 같은 세대로 다시 넣는다
  evicted-<시각>  job.evicted                Workload Evicted=True. reason: PodsReadyTimeout(10분 안에 Pod 전부   조건 lastTransitionTime
                                             Ready 실패, ckpt-init 다운로드 · 이미지 pull이 길 때) · Preempted 등.
                                             발생마다 시각이 달라 여러 번 나감. payload reason · message · requeue_count
  deactivated     job.deactivated            Workload spec.active=false (다시 넣기 한도 backoffLimitCount 초과 등)   감시기가 본 시각
                                             → 그 뒤 Kueue는 다시 넣지 않음. 이미 끝난(end) 세대면 안 보냄
  이 사건이 없으면 Kueue가 시간 초과로 내린 작업이 수집기 · 복구 컨트롤러 · 화면에서 "입장 후 조용함"으로만 보인다.

종료(end)는 세대마다 하나: completed · failed · fenced · cancelled · stopped가 같은 key("end")를 쓴다.
재배치 요청이 있으면 중단 주석이 있어도 fenced (복구가 먼저). 삭제를 놓쳐 재목록에서 발견하면 마지막으로 본
객체의 주석으로 판정한다 — 주석을 단 수정을 못 봤으면 stopped로 나갈 수 있다.
이미 끝난(Completed · Failed) JobSet을 나중에 지우면 새 사건이 없다. 감시기가 재시작해 판정이 달라져도
수집기는 먼저 들어간 하나만 남긴다 (event_id 중복 무시).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import Snapshot


def _compact(ts: str) -> str:
    """'2026-11-23T05:48:12Z' → '20261123T054812Z' (event_id용, 같은 발생은 같은 값)."""
    return "".join(ch for ch in ts if ch.isalnum())


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

    restarted = bool(s.restart_reason)
    resubmitted = bool(s.recovery_id) or restarted
    restart = {"restart_reason": s.restart_reason, "restart_requested_at": s.restart_requested_at} if restarted else {}
    if resubmitted or emit_first_submitted:
        add("submitted", "job.submitted", s.created_at, **restart)
    if s.admitted_at is not None:
        add("admitted", "job.admitted", s.admitted_at)
    if s.cluster:
        add("placed", "job.placed", s.placed_at)

    start_type = "job.resumed" if resubmitted else "job.started"
    if s.pods_ready_at is not None:
        add("started", start_type, s.pods_ready_at, basis="workload_pods_ready", **restart)
    elif s.all_ready:
        add("started", start_type, None, basis="jobset_all_ready", **restart)

    # Kueue가 내림 · 비활성화 (GPU-259). 내림은 세대 안에서 여러 번(최대 backoffLimitCount+1) → 발생 시각으로 key
    if s.evicted_at:
        add(f"evicted-{_compact(s.evicted_at)}", "job.evicted", s.evicted_at, reason=s.evicted_reason,
            message=(s.evicted_message or "")[:300] or None, requeue_count=s.requeue_count)
    ended = s.completed_at is not None or s.failed_at is not None or s.deleted
    if s.deactivated and not ended:
        add("deactivated", "job.deactivated", None, reason=s.evicted_reason,
            message=(s.evicted_message or "")[:300] or None, requeue_count=s.requeue_count)

    if s.completed_at is not None:
        add("end", "job.completed", s.completed_at)
    elif s.failed_at is not None:
        add("end", "job.failed", s.failed_at, message=s.failed_message)
    elif s.deleted:
        if s.relocated:
            add("end", "job.fenced", s.deleted_at, request_id=s.relocation_request_id, action=s.relocation_action)
        elif s.stop_reason == "user":
            add("end", "job.cancelled", s.deleted_at, reason=s.stop_reason, requested_at=s.stop_requested_at)
        else:
            add("end", "job.stopped", s.deleted_at)
    return out
