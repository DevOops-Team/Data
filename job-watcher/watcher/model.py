"""JobSet · Workload 객체에서 사건에 필요한 값만 뽑는다.

입력은 쿠버네티스 API가 돌려주는 dict 그대로 (CustomObjectsApi). 관리 클러스터의 JobSet은 MultiKueue가
워커 EKS의 상태(status)를 복사해 오는 사본이고, Pod는 워커에만 있어 여기서는 보이지 않는다.

이름 · 라벨 규칙 (JobSet 템플릿 · ADR-12)
  이름   job-{작업ID}-g{세대 4자리}        예: job-3f9a1c2e-g0002
  라벨   devoops/generation · kueue.x-k8s.io/queue-name · (있으면) devoops/job-id
  주석   devoops/recovery-id · (있으면) devoops/exp-id · devoops/config-sha256 · devoops/stop-reason
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

NAME_RE = re.compile(r"^job-(?P<job>[0-9a-z]+)-g(?P<gen>\d+)$")

LABEL_JOB = "devoops/job-id"
LABEL_GEN = "devoops/generation"
LABEL_QUEUE = "kueue.x-k8s.io/queue-name"
ANN_EXP = "devoops/exp-id"
ANN_RECOVERY = "devoops/recovery-id"
ANN_CONFIG = "devoops/config-sha256"
ANN_STOP = "devoops/stop-reason"      # 삭제하는 쪽이 붙임: relocate(재배치 실행기) · cancel(사용자 취소)
GPU_RESOURCE = "nvidia.com/gpu"
TRAINER = "trainer"


@dataclass(frozen=True)
class JobRef:
    job_id: str
    generation: int


@dataclass
class Snapshot:
    """세대 하나(JobSet 하나)의 지금 상태. 시각은 쿠버네티스가 찍은 값(ISO 8601 · UTC)."""
    name: str
    job_id: str
    generation: int
    created_at: str | None = None
    queue: str | None = None
    gpus: int | None = None
    exp_id: str | None = None
    recovery_id: str | None = None
    config_sha256: str | None = None
    image_digest: str | None = None
    # JobSet status (MultiKueue가 워커에서 복사)
    all_ready: bool = False
    completed_at: str | None = None
    failed_at: str | None = None
    failed_message: str | None = None
    # Workload (Kueue)
    admitted_at: str | None = None
    cluster: str | None = None
    placed_at: str | None = None
    pods_ready_at: str | None = None
    # 삭제
    deleted: bool = False
    deleted_at: str | None = None
    stop_reason: str | None = None


def job_ref(meta: dict[str, Any]) -> JobRef | None:
    """작업 ID · 세대. 라벨이 있으면 라벨, 없으면 이름에서. 둘 다 안 되면 None (우리 학습 JobSet이 아님)."""
    labels = meta.get("labels") or {}
    m = NAME_RE.match(meta.get("name", ""))
    job_id = labels.get(LABEL_JOB) or (m.group("job") if m else None)
    gen = labels.get(LABEL_GEN) or (m.group("gen") if m else None)
    if not job_id or gen is None:
        return None
    try:
        return JobRef(job_id, int(gen))
    except ValueError:
        return None


def condition_time(obj: dict[str, Any], ctype: str) -> str | None:
    """status.conditions에서 type=ctype · status=True인 조건의 lastTransitionTime."""
    for c in (obj.get("status") or {}).get("conditions") or []:
        if c.get("type") == ctype and str(c.get("status")) == "True":
            return c.get("lastTransitionTime") or ""
    return None


def _condition_message(obj: dict[str, Any], ctype: str) -> str | None:
    for c in (obj.get("status") or {}).get("conditions") or []:
        if c.get("type") == ctype:
            return c.get("message")
    return None


def _pod_spec(rj: dict[str, Any]) -> dict[str, Any]:
    return ((((rj.get("template") or {}).get("spec") or {}).get("template") or {}).get("spec")) or {}


def _gpu_count(container: dict[str, Any]) -> int:
    res = container.get("resources") or {}
    v = (res.get("limits") or {}).get(GPU_RESOURCE) or (res.get("requests") or {}).get(GPU_RESOURCE)
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def gpus(jobset: dict[str, Any]) -> int | None:
    """요청 GPU 합 = Σ replicas × parallelism × (Pod 하나의 nvidia.com/gpu). GPU 요청이 없으면 None."""
    total = 0
    for rj in (jobset.get("spec") or {}).get("replicatedJobs") or []:
        job_spec = (rj.get("template") or {}).get("spec") or {}
        per_pod = sum(_gpu_count(c) for c in _pod_spec(rj).get("containers") or [])
        total += int(rj.get("replicas", 1)) * int(job_spec.get("parallelism", 1)) * per_pod
    return total or None


def image_digest(jobset: dict[str, Any]) -> str | None:
    """trainer 컨테이너 이미지의 sha256 digest (태그만 있으면 None)."""
    for rj in (jobset.get("spec") or {}).get("replicatedJobs") or []:
        containers = _pod_spec(rj).get("containers") or []
        for c in sorted(containers, key=lambda c: c.get("name") != TRAINER):
            image = c.get("image") or ""
            if "@sha256:" in image:
                return image.split("@", 1)[1]
    return None


def all_ready(jobset: dict[str, Any]) -> bool:
    """replicatedJob마다 ready ≥ replicas. JobSet 1 = Job 1 = Pod N이라 Job ready = Pod N개 전부 Ready."""
    specs = {rj.get("name"): int(rj.get("replicas", 1)) for rj in (jobset.get("spec") or {}).get("replicatedJobs") or []}
    if not specs:
        return False
    status = {s.get("name"): int(s.get("ready", 0)) for s in (jobset.get("status") or {}).get("replicatedJobsStatus") or []}
    return all(status.get(name, 0) >= replicas for name, replicas in specs.items())


def workload_cluster(workload: dict[str, Any]) -> tuple[str | None, str | None]:
    """MultiKueue가 고른 워커 클러스터와 그 시각. status.clusterName(Kueue v0.9+)이 원본."""
    status = workload.get("status") or {}
    cluster = status.get("clusterName")
    if not cluster:
        return None, None
    at = None
    for ac in status.get("admissionChecks") or []:
        if ac.get("state") == "Ready":
            at = ac.get("lastTransitionTime") or at
    return cluster, at


def owner_jobset(workload: dict[str, Any]) -> str | None:
    for o in (workload.get("metadata") or {}).get("ownerReferences") or []:
        if o.get("kind") == "JobSet":
            return o.get("name")
    return None


def snapshot(jobset: dict[str, Any], workload: dict[str, Any] | None = None, *,
             deleted: bool = False) -> Snapshot | None:
    meta = jobset.get("metadata") or {}
    ref = job_ref(meta)
    if ref is None:
        return None
    labels, ann = meta.get("labels") or {}, meta.get("annotations") or {}
    s = Snapshot(
        name=meta.get("name", ""), job_id=ref.job_id, generation=ref.generation,
        created_at=meta.get("creationTimestamp"), queue=labels.get(LABEL_QUEUE), gpus=gpus(jobset),
        exp_id=ann.get(ANN_EXP), recovery_id=ann.get(ANN_RECOVERY), config_sha256=ann.get(ANN_CONFIG),
        image_digest=image_digest(jobset), all_ready=all_ready(jobset),
        completed_at=condition_time(jobset, "Completed"), failed_at=condition_time(jobset, "Failed"),
        failed_message=_condition_message(jobset, "Failed"),
        deleted=deleted, deleted_at=meta.get("deletionTimestamp"), stop_reason=ann.get(ANN_STOP),
    )
    if workload is not None:
        s.admitted_at = condition_time(workload, "Admitted")
        s.pods_ready_at = condition_time(workload, "PodsReady")
        s.cluster, s.placed_at = workload_cluster(workload)
        s.placed_at = s.placed_at or s.admitted_at
    return s
