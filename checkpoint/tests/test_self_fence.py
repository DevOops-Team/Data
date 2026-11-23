"""자기 펜싱 기록 (GPU-257): 펜싱 STOP한 옛 세대 Pod가 trainer 종료 뒤 펜싱 항목에 Pod 번호를 남기는지 (relocator GPU-255)."""
from __future__ import annotations

from ckpt.config import Config
from ckpt.sidecar import Sidecar, pod_index

JOB = "job-sf"


def _cfg(tmp_path, *, gen=1, rank=0, cluster="eks-a", world=2):
    return Config(job_id=JOB, generation=gen, cluster=cluster, bucket="b", table="t", ckpt_dir=tmp_path / f"r{rank}",
                  code_dir=tmp_path, data_dir=tmp_path, world_size=world, rank=rank, checkpoint_format="fsdp-dcp",
                  config_uri=None, config_sha256=None, dataset_uri=None, dataset_manifest_sha256=None,
                  region="us-east-2")


def _advance(fencing, to_gen):
    """복구 컨트롤러가 세대를 to_gen까지 올림 (rid마다 한 번)."""
    item = fencing.get(JOB)
    for g in range(item.generation, to_gen):
        rid = f"recovery-{g}"
        assert fencing.start_recovery(JOB, g, rid) and fencing.advance_generation(JOB, g, rid)
        fencing.set_phase(JOB, rid, "COMPLETED")


def _job(fencing, gen=1, owner="eks-a"):
    fencing.create_job(JOB)
    fencing.claim_owner(JOB, 1, owner)
    if gen > 1:
        _advance(fencing, gen)


def _fields(fencing):
    raw = fencing.get(JOB).raw
    ranks = sorted(int(x) for x in raw["self_fence_ranks"]["NS"]) if "self_fence_ranks" in raw else None
    gen = int(raw["self_fence_gen"]["N"]) if "self_fence_gen" in raw else None
    return gen, ranks, raw


def test_every_pod_adds_its_index_after_generation_advanced(fencing):
    _job(fencing)
    _advance(fencing, 2)
    assert fencing.record_self_fence(JOB, 1, "eks-a", 2, 1) == "reset"     # 처음 쓰는 Pod
    assert fencing.record_self_fence(JOB, 1, "eks-a", 2, 0) == "added"     # 같은 세대 → ADD
    assert fencing.record_self_fence(JOB, 1, "eks-a", 2, 0) == "added"     # 다시 와도 집합이라 그대로
    gen, ranks, raw = _fields(fencing)
    assert (gen, ranks) == (1, [0, 1])
    assert raw["self_fence_cluster"]["S"] == "eks-a" and raw["self_fence_world_size"]["N"] == "2"
    assert "self_fence_at" in raw


def test_not_written_when_generation_not_advanced(fencing):
    _job(fencing)                                                          # 세대 그대로(1) = 펜싱된 게 아님
    assert fencing.record_self_fence(JOB, 1, "eks-a", 2, 0) == "skipped"
    assert _fields(fencing)[0] is None


def test_older_generation_record_is_replaced_not_mixed(fencing):
    _job(fencing)
    _advance(fencing, 2)
    fencing.record_self_fence(JOB, 1, "eks-a", 2, 0)
    fencing.record_self_fence(JOB, 1, "eks-a", 2, 1)
    _advance(fencing, 3)                                                   # 세대 2도 펜싱됨
    assert fencing.record_self_fence(JOB, 2, "eks-b", 2, 1) == "reset"     # 옛 세대(1) 기록 → 내 번호 하나로 SET
    gen, ranks, raw = _fields(fencing)
    assert (gen, ranks, raw["self_fence_cluster"]["S"]) == (2, [1], "eks-b")


def test_newer_generation_record_is_not_touched(fencing):
    _job(fencing)
    _advance(fencing, 3)
    fencing.record_self_fence(JOB, 2, "eks-b", 2, 0)                       # 세대 2 기록
    assert fencing.record_self_fence(JOB, 1, "eks-a", 2, 1) == "skipped"   # 늦게 온 세대 1은 못 덮음
    assert _fields(fencing)[:2] == (2, [0])


def test_rank_that_never_saw_fencing_still_records(fencing, store, tmp_path, monkeypatch):
    """11/23 EKS: rank 1 사이드카는 하트비트를 안 해서 펜싱을 모른 채 SIGTERM을 받는다 → 그래도 기록해야 함."""
    monkeypatch.delenv("JOB_COMPLETION_INDEX", raising=False)
    _job(fencing)
    rank0 = Sidecar(_cfg(tmp_path, rank=0), fencing, store, heartbeat_seconds=0)
    rank1 = Sidecar(_cfg(tmp_path, rank=1), fencing, store, heartbeat_seconds=0)
    _advance(fencing, 2)
    rank0.tick()
    rank1.tick()                                                           # 저장이 없으면 rank 1은 세대를 안 봄
    assert rank0.fence_stopped and not rank1.stopped
    assert rank0.record_self_fence() == "reset"
    assert rank1.record_self_fence() == "added"
    assert _fields(fencing)[:2] == (1, [0, 1])


def test_normal_end_or_user_stop_is_skipped_by_condition(fencing, store, tmp_path):
    _job(fencing)                                                          # 세대 그대로 = 정상 완료 · 사용자 중단
    sc = Sidecar(_cfg(tmp_path), fencing, store)
    assert sc.record_self_fence() == "skipped"
    gone = Sidecar(_cfg(tmp_path), fencing, store)
    fencing.ddb.delete_item(TableName=fencing.table, Key={"job_id": {"S": JOB}})
    assert gone.record_self_fence() == "skipped"                           # 항목 없음(no_item)
    assert fencing.get(JOB) is None                                        # 조건부 쓰기가 항목을 새로 만들지 않음


def test_heartbeat_fence_then_record(fencing, store, tmp_path):
    _job(fencing)
    sc = Sidecar(_cfg(tmp_path), fencing, store, heartbeat_seconds=0)
    _advance(fencing, 2)
    sc.tick()                                                              # 하트비트가 조건부 갱신에서 거부됨
    assert sc.stopped and sc.stop_reason == "heartbeat_fenced"
    assert sc.record_self_fence() == "reset"
    assert _fields(fencing)[:2] == (1, [0])


def test_pod_index_prefers_job_completion_index(tmp_path, monkeypatch):
    monkeypatch.setenv("JOB_COMPLETION_INDEX", "3")
    assert pod_index(_cfg(tmp_path, rank=0)) == 3
    monkeypatch.delenv("JOB_COMPLETION_INDEX")
    assert pod_index(_cfg(tmp_path, rank=2)) == 2
