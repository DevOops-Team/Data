from conftest import jobset, workload

from watcher.model import JobRef, job_ref, snapshot


def test_job_ref_from_name_and_label():
    assert job_ref({"name": "job-3f9a1c2e-g0002", "labels": {"devoops/generation": "2"}}) == JobRef("3f9a1c2e", 2)
    assert job_ref({"name": "job-3f9a1c2e-g0003"}) == JobRef("3f9a1c2e", 3)            # 라벨 없어도 이름으로
    assert job_ref({"name": "x", "labels": {"devoops/job-id": "abc", "devoops/generation": "1"}}) == JobRef("abc", 1)


def test_job_ref_ignores_other_jobsets():
    assert job_ref({"name": "some-other-jobset"}) is None
    assert job_ref({"name": "job-abc-gXYZ"}) is None


def test_snapshot_fields():
    s = snapshot(jobset(pods=4, ready=1), workload("job-3f9a1c2e-g0001", admitted="2026-11-12T01:00:05Z",
                                                    cluster="eks-a", cluster_at="2026-11-12T01:00:07Z",
                                                    pods_ready="2026-11-12T01:01:30Z"))
    assert (s.job_id, s.generation, s.queue, s.gpus, s.exp_id) == ("3f9a1c2e", 1, "queue-heavy", 4, "exp-0001")
    assert s.image_digest == "sha256:abc123"
    assert s.all_ready is True
    assert (s.admitted_at, s.cluster, s.placed_at, s.pods_ready_at) == (
        "2026-11-12T01:00:05Z", "eks-a", "2026-11-12T01:00:07Z", "2026-11-12T01:01:30Z")


def test_not_ready_until_every_replicated_job_ready():
    assert snapshot(jobset(ready=0)).all_ready is False


def test_tag_only_image_has_no_digest():
    assert snapshot(jobset(image="ghcr.io/devoops-team/trainer:main")).image_digest is None
