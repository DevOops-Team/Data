"""JobSet · Workload 예시 객체 (관리 클러스터에서 CustomObjectsApi가 돌려주는 dict 모양)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "collector"))   # 수집기 envelope.parse로 형식 호환 확인


def jobset(job_id: str = "3f9a1c2e", gen: int = 1, *, pods: int = 4, ready: int = 0,
           completed: str | None = None, failed: str | None = None, deletion: str | None = None,
           annotations: dict[str, str] | None = None, created: str = "2026-11-12T01:00:00Z",
           image: str = "ghcr.io/devoops-team/trainer@sha256:abc123") -> dict[str, Any]:
    name = f"job-{job_id}-g{gen:04d}"
    conditions = []
    if completed:
        conditions.append({"type": "Completed", "status": "True", "lastTransitionTime": completed})
    if failed:
        conditions.append({"type": "Failed", "status": "True", "lastTransitionTime": failed,
                           "message": "jobset failed due to job failure"})
    meta: dict[str, Any] = {
        "name": name, "namespace": "training", "creationTimestamp": created,
        "labels": {"devoops/generation": str(gen), "kueue.x-k8s.io/queue-name": "queue-heavy"},
        "annotations": {"devoops/exp-id": "exp-0001", **(annotations or {})},
    }
    if deletion:
        meta["deletionTimestamp"] = deletion
    return {
        "apiVersion": "jobset.x-k8s.io/v1alpha2", "kind": "JobSet", "metadata": meta,
        "spec": {"replicatedJobs": [{
            "name": "workers", "replicas": 1,
            "template": {"spec": {"parallelism": pods, "completions": pods, "template": {"spec": {"containers": [
                {"name": "ckpt-sidecar", "image": "ghcr.io/devoops-team/checkpoint:main"},
                {"name": "trainer", "image": image, "resources": {"limits": {"nvidia.com/gpu": "1"}}},
            ]}}}},
        }]},
        "status": {"conditions": conditions,
                   "replicatedJobsStatus": [{"name": "workers", "ready": ready, "active": 1 if ready else 0}]},
    }


def workload(jobset_name: str, *, admitted: str | None = None, cluster: str | None = None,
             cluster_at: str | None = None, pods_ready: str | None = None) -> dict[str, Any]:
    conditions = []
    if admitted:
        conditions.append({"type": "QuotaReserved", "status": "True", "lastTransitionTime": admitted})
        conditions.append({"type": "Admitted", "status": "True", "lastTransitionTime": admitted})
    if pods_ready:
        conditions.append({"type": "PodsReady", "status": "True", "lastTransitionTime": pods_ready})
    status: dict[str, Any] = {"conditions": conditions}
    if cluster:
        status["clusterName"] = cluster
        status["admissionChecks"] = [{"name": "multikueue", "state": "Ready",
                                      "lastTransitionTime": cluster_at or admitted}]
    return {
        "apiVersion": "kueue.x-k8s.io/v1beta1", "kind": "Workload",
        "metadata": {"name": f"jobset-{jobset_name}-abcde", "namespace": "training",
                     "ownerReferences": [{"apiVersion": "jobset.x-k8s.io/v1alpha2", "kind": "JobSet",
                                          "name": jobset_name}]},
        "status": status,
    }


class FakePublisher:
    def __init__(self):
        self.sent: list[tuple[str, str, dict]] = []

    def publish(self, *, topic: str, key: str, value: dict) -> None:
        self.sent.append((topic, key, value))

    def flush(self, timeout: float = 5.0) -> int:
        return 0

    def types(self) -> list[str]:
        return [v["type"] for _, _, v in self.sent]


@pytest.fixture
def pub() -> FakePublisher:
    return FakePublisher()
