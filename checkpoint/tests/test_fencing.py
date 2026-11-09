"""펜싱 테이블 조건부 갱신 (규약 5 · 10절)."""
from ckpt.fencing import ApproveResult

JOB = "3f9a1c2e"


def test_create_job_only_once(fencing):
    assert fencing.create_job(JOB) is True
    assert fencing.create_job(JOB) is False
    assert fencing.get(JOB).generation == 1


def test_claim_owner(fencing):
    fencing.create_job(JOB)
    assert fencing.claim_owner(JOB, 1, "eks-a")
    assert fencing.claim_owner(JOB, 1, "eks-a")          # 같은 클러스터 재시도는 허용
    assert not fencing.claim_owner(JOB, 1, "eks-b")      # 같은 세대를 다른 클러스터가 선점 못 함
    assert not fencing.claim_owner(JOB, 2, "eks-a")      # 세대 불일치


def test_approve_results_and_prev(fencing):
    fencing.create_job(JOB)
    fencing.claim_owner(JOB, 1, "eks-a")
    assert fencing.approve(JOB, 1, "eks-a", 950) is ApproveResult.OK
    assert fencing.approve(JOB, 1, "eks-a", 1200) is ApproveResult.OK
    item = fencing.get(JOB)
    assert (item.approved_gen, item.approved_step) == (1, 1200)
    assert int(item.raw["prev_approved_step"]["N"]) == 950

    assert fencing.approve(JOB, 1, "eks-a", 1200) is ApproveResult.ALREADY   # 같은 저장본
    assert fencing.approve(JOB, 1, "eks-a", 1100) is ApproveResult.STALE     # 더 최신이 이미 승인
    assert fencing.approve(JOB, 1, "eks-b", 1300) is ApproveResult.FENCED    # owner 불일치
    assert fencing.approve(JOB, 2, "eks-a", 1300) is ApproveResult.FENCED    # 세대 불일치
    assert fencing.approve("nope", 1, "eks-a", 1) is ApproveResult.NO_ITEM


def test_recovery_blocks_old_generation(fencing):
    fencing.create_job(JOB)
    fencing.claim_owner(JOB, 1, "eks-a")
    fencing.approve(JOB, 1, "eks-a", 1200)

    assert fencing.start_recovery(JOB, 1, "recovery-7c21")
    assert fencing.approve(JOB, 1, "eks-a", 1300) is ApproveResult.RECOVERING  # 복구 중 승인 금지

    assert fencing.advance_generation(JOB, 1, "recovery-7c21")
    assert not fencing.advance_generation(JOB, 1, "recovery-7c21")            # 같은 복구로 두 번 올리지 않음
    item = fencing.get(JOB)
    assert item.generation == 2 and item.owner is None
    assert (item.approved_gen, item.approved_step) == (1, 1200)                # 승인본 유지
    assert int(item.raw["retry_count"]["N"]) == 1

    assert fencing.approve(JOB, 1, "eks-a", 1300) is ApproveResult.FENCED      # 옛 세대 차단
    assert not fencing.heartbeat(JOB, 1, "eks-a")

    assert fencing.claim_owner(JOB, 2, "eks-b")
    fencing.set_phase(JOB, "recovery-7c21", "RESUMING")
    assert fencing.approve(JOB, 2, "eks-b", 1500) is ApproveResult.OK         # RESUMING부터 승인 가능
    item = fencing.get(JOB)
    assert (item.approved_gen, item.approved_step) == (2, 1500)
    assert int(item.raw["prev_approved_gen"]["N"]) == 1
