"""복구 재개 증거 (GPU-254): 복구 컨트롤러 evidence.py progress 계약대로, 조건이 맞을 때만 쓰는지."""
from __future__ import annotations

import json

from boto3.dynamodb.types import TypeDeserializer

from ckpt.config import Config
from ckpt.evidence import RecoveryEvidence, build

JOB, RID, UID = "job-ev", "recovery-7c21", "uid-mgr-new-g2"
NAME = f"job-{JOB}-g0002"


def _cfg(tmp_path, gen=2, cluster="eks-b", rank=0):
    return Config(job_id=JOB, generation=gen, cluster=cluster, bucket="b", table="t", ckpt_dir=tmp_path,
                  code_dir=tmp_path, data_dir=tmp_path, world_size=2, rank=rank, checkpoint_format="fsdp-dcp",
                  config_uri=None, config_sha256=None, dataset_uri=None, dataset_manifest_sha256=None,
                  region="us-east-2")


def _recovered(fencing, *, phase="NEW_RUN_SUBMITTED", new_uid=UID, new_name=NAME):
    """세대 1에서 step 1200 승인 → 복구로 세대 2 (eks-b) · relocator가 새 JobSet 제출까지."""
    fencing.create_job(JOB)
    fencing.claim_owner(JOB, 1, "eks-a")
    fencing.approve(JOB, 1, "eks-a", 1200)
    fencing.start_recovery(JOB, 1, RID)
    fencing.advance_generation(JOB, 1, RID)
    fencing.claim_owner(JOB, 2, "eks-b")
    fencing.ddb.update_item(
        TableName=fencing.table, Key={"job_id": {"S": JOB}},
        UpdateExpression="SET recovery_phase = :p, new_jobset_name = :n, new_jobset_uid = :u, recovery_resume_gen = :rg, recovery_resume_step = :rs",
        ExpressionAttributeValues={":p": {"S": phase}, ":n": {"S": new_name}, ":u": {"S": new_uid}, ":rg": {"N": "1"}, ":rs": {"N": "1200"}})


def _files(tmp_path, *, step, resume=True, status="RUNNING", gen=2, error=None):
    (tmp_path / "resume.json").write_text(json.dumps(
        {"resume": resume, "approved_gen": 1 if resume else None, "approved_step": 1200 if resume else None}))
    prog = tmp_path / "progress.json"
    prog.write_text(json.dumps({"generation": gen, "step": step, "status": status, "error": error}))
    return prog


def _evidence(fencing):
    raw = fencing.get(JOB).raw.get("recovery_training_evidence")
    return TypeDeserializer().deserialize(raw) if raw else None


def test_writes_progress_evidence_after_first_step(fencing, tmp_path):
    _recovered(fencing)
    prog = _files(tmp_path, step=1201)
    ev = RecoveryEvidence(_cfg(tmp_path), fencing, str(prog), check_seconds=0, refresh_seconds=0, clock=lambda: 1_800_000_000)
    assert ev.tick()
    got = _evidence(fencing)
    assert got == {
        "kind": "progress", "schema_version": 1, "job_id": JOB, "recovery_id": RID, "generation": 2,
        "cluster": "eks-b", "jobset_uid": UID, "event_id": f"{JOB}-g0002-{RID}-progress-s1201-t1800000000",
        "observed_at": 1_800_000_000,
        "outcome": "succeeded", "completed_step": 1201,
        "checkpoint_loaded": True, "restored_gen": 1, "restored_step": 1200,
    }
    assert "producer" not in got and "received_at" not in got        # 컨트롤러가 붙이는 칸


def test_waits_until_trainer_finishes_a_step_past_the_restore_point(fencing, tmp_path):
    _recovered(fencing)
    prog = _files(tmp_path, step=1200)                               # 불러오기만 하고 아직 step 안 끝냄
    ev = RecoveryEvidence(_cfg(tmp_path), fencing, str(prog), check_seconds=0)
    assert not ev.tick() and _evidence(fencing) is None


def test_refreshes_with_newer_step_and_time(fencing, tmp_path):
    _recovered(fencing, phase="RESUMING")
    now = [1000.0]
    prog = _files(tmp_path, step=1210)
    ev = RecoveryEvidence(_cfg(tmp_path), fencing, str(prog), check_seconds=0, refresh_seconds=60, clock=lambda: now[0])
    assert ev.tick()
    _files(tmp_path, step=1250)
    now[0] = 1030
    assert not ev.tick()                                             # 60초 전에는 다시 안 씀
    now[0] = 1061
    assert ev.tick()
    got = _evidence(fencing)
    assert (got["completed_step"], got["observed_at"]) == (1250, 1061)
    assert got["event_id"] == f"{JOB}-g0002-{RID}-progress-s1250-t1061"   # 관측마다 새 ID (같은 ID 내용 변경 금지)


def test_blocks_other_jobset_name_owner_or_generation(fencing, tmp_path):
    _recovered(fencing, new_name=f"job-{JOB}-g0003")                 # relocator가 적은 JobSet이 내 것이 아님
    prog = _files(tmp_path, step=1201)
    assert not RecoveryEvidence(_cfg(tmp_path), fencing, str(prog), check_seconds=0).tick()
    fencing.ddb.delete_item(TableName=fencing.table, Key={"job_id": {"S": JOB}})
    _recovered(fencing)
    assert not RecoveryEvidence(_cfg(tmp_path, cluster="eks-c"), fencing, str(prog), check_seconds=0).tick()
    assert not RecoveryEvidence(_cfg(tmp_path, gen=1, cluster="eks-a"), fencing, str(prog), check_seconds=0).tick()
    assert _evidence(fencing) is None                                # 옛 실행 · 다른 클러스터는 못 씀


def test_uid_change_between_read_and_write_is_rejected(fencing, tmp_path):
    _recovered(fencing)
    prog = _files(tmp_path, step=1201)
    ev = RecoveryEvidence(_cfg(tmp_path), fencing, str(prog), check_seconds=0)
    real_get = fencing.get

    def get_then_swap(job_id):                                       # 읽은 뒤 relocator가 다른 JobSet으로 다시 적음
        item = real_get(job_id)
        fencing.ddb.update_item(TableName=fencing.table, Key={"job_id": {"S": JOB}},
                                UpdateExpression="SET new_jobset_uid = :u", ExpressionAttributeValues={":u": {"S": "uid-other"}})
        return item
    fencing.get = get_then_swap
    assert not ev.tick() and _evidence(fencing) is None              # 조건부 쓰기가 막음


def test_no_write_outside_recovery_or_after_completion(fencing, tmp_path):
    fencing.create_job(JOB)                                          # 복구 없는 평범한 실행
    fencing.claim_owner(JOB, 1, "eks-a")
    prog = _files(tmp_path, step=5, resume=False, gen=1)
    assert not RecoveryEvidence(_cfg(tmp_path, gen=1, cluster="eks-a"), fencing, str(prog), check_seconds=0).tick()
    fencing.ddb.delete_item(TableName=fencing.table, Key={"job_id": {"S": JOB}})
    _recovered(fencing, phase="COMPLETED")
    prog = _files(tmp_path, step=1300)
    assert not RecoveryEvidence(_cfg(tmp_path), fencing, str(prog), check_seconds=0).tick()


def test_before_relocator_records_jobset_or_non_zero_rank_does_nothing(fencing, tmp_path):
    _recovered(fencing)
    fencing.ddb.update_item(TableName=fencing.table, Key={"job_id": {"S": JOB}},
                            UpdateExpression="REMOVE new_jobset_name, new_jobset_uid")
    prog = _files(tmp_path, step=1201)
    assert not RecoveryEvidence(_cfg(tmp_path), fencing, str(prog), check_seconds=0).tick()
    _recovered_fields = {":n": {"S": NAME}, ":u": {"S": UID}}
    fencing.ddb.update_item(TableName=fencing.table, Key={"job_id": {"S": JOB}},
                            UpdateExpression="SET new_jobset_name = :n, new_jobset_uid = :u", ExpressionAttributeValues=_recovered_fields)
    assert not RecoveryEvidence(_cfg(tmp_path, rank=1), fencing, str(prog), check_seconds=0).tick()
    assert _evidence(fencing) is None


def test_build_from_scratch_and_failure(tmp_path):
    cfg = _cfg(tmp_path)
    scratch = build(cfg, RID, UID, {"resume": False}, {"generation": 2, "step": 3, "status": "RUNNING"}, 5)
    assert scratch["started_from_scratch"] is True and "checkpoint_loaded" not in scratch and scratch["completed_step"] == 3
    failed = build(cfg, RID, UID, {"resume": True, "approved_gen": 1, "approved_step": 1200},
                   {"generation": 2, "step": 1200, "status": "FAILED", "error": "RuntimeError: CUDA error"}, 5)
    assert (failed["outcome"], failed["failure_reason"]) == ("failed", "RuntimeError: CUDA error")
    assert build(cfg, RID, UID, None, {"generation": 1, "step": 9999}, 5) is None   # 다른 세대의 진행률 파일
