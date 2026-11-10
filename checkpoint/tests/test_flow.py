"""저장 → 승인 → 장애 → 재개 전체 흐름 (규약 6 · 7 · 8절, 「흐름으로 읽기」 4~8절).

Pod 2개(rank 0 · 1)를 각자 다른 로컬 폴더로 흉내 낸다. trainer는 조각 파일 + .done만 만드는 가짜.
"""
import json
import os
from pathlib import Path

from ckpt import layout
from ckpt.config import Config
from ckpt.init_main import EXIT_CORRUPT, EXIT_GEN, EXIT_OK, EXIT_OWNER
from ckpt.init_main import run as init_run
from ckpt.sidecar import Sidecar

JOB = "3f9a1c2e"
WORLD = 2


def cfg(names, tmp: Path, *, gen: int, rank: int, cluster: str, pod: str) -> Config:
    root = tmp / pod
    return Config(job_id=JOB, generation=gen, cluster=cluster, bucket=names["bucket"], table=names["table"],
                  ckpt_dir=root / "ckpt", code_dir=root / "code", data_dir=root / "data",
                  world_size=WORLD, rank=rank, checkpoint_format="fsdp-dcp",
                  config_uri=None, config_sha256="cfg-sha", dataset_uri=None, dataset_manifest_sha256=None,
                  region="us-east-2")


def fake_save(c: Config, step: int) -> None:
    """trainer 대역: DCP처럼 rank마다 자기 조각을 쓰고, 다 쓰면 .done."""
    d = layout.local_step_dir(c.ckpt_dir, step)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"__{c.rank}_0.distcp").write_bytes(os.urandom(4096) + f"rank{c.rank}-step{step}".encode())
    (d / f"rank-state-{c.rank}.pt").write_bytes(f"rng-{c.rank}-{step}".encode())
    if c.rank == 0:
        (d / ".metadata").write_bytes(b"dcp-metadata-" + str(step).encode())
    layout.done_marker(c.ckpt_dir, step).touch()


def sidecar(c, fencing, store, **kw):
    return Sidecar(c, fencing, store, rank_wait_seconds=kw.get("rank_wait", 60), keep_local=2, heartbeat_seconds=0)


def test_save_approve_fail_resume(names, fencing, store, tmp_path):
    fencing.create_job(JOB)

    # ① 세대 1 시작 (eks-a, Pod 2개)
    g1 = [cfg(names, tmp_path, gen=1, rank=r, cluster="eks-a", pod=f"g1-r{r}") for r in range(WORLD)]
    for c in g1:
        assert init_run(c, fencing, store) == EXIT_OK
        assert json.loads((c.ckpt_dir / "resume.json").read_text())["start_step"] == 1
    sc1 = [sidecar(c, fencing, store) for c in g1]

    # ② step 100 저장 → rank 1이 먼저 올리고, rank 0이 모아서 승인
    for c in g1:
        fake_save(c, 100)
    sc1[1].tick()
    sc1[0].tick()
    item = fencing.get(JOB)
    assert (item.approved_gen, item.approved_step) == (1, 100)
    manifest = store.get_json(layout.s3_manifest_key(JOB, 1, 100))
    assert {f["name"] for f in manifest["files"]} == {
        ".metadata", "__0_0.distcp", "__1_0.distcp", "rank-state-0.pt", "rank-state-1.pt"}
    assert manifest["config_sha256"] == "cfg-sha" and manifest["world_size"] == WORLD
    sc1[1].tick()                                                   # rank 1은 DynamoDB를 보고 .approved
    assert all(layout.approved_marker(c.ckpt_dir, 100).exists() for c in g1)

    # ③ 장애: 복구 컨트롤러가 세대 +1
    assert fencing.start_recovery(JOB, 1, "recovery-7c21")
    assert fencing.advance_generation(JOB, 1, "recovery-7c21")

    # ④ 옛 세대가 늦게 step 200을 저장 → 업로드 전 펜싱으로 STOP, S3에 아무것도 안 올라감
    for c in g1:
        fake_save(c, 200)
        layout.stop_file(c.ckpt_dir).unlink(missing_ok=True)
    sc1[0].tick()
    assert layout.stop_file(g1[0].ckpt_dir).exists()
    assert not store.exists(layout.s3_rank_list_key(JOB, 1, 200, 0))
    assert fencing.get(JOB).approved_step == 100

    # ⑤ 옛 세대 설정으로 init을 다시 띄우면 시작 안 함
    assert init_run(g1[0], fencing, store) == EXIT_GEN

    # ⑥ 세대 2 재개 (eks-b, 새 Pod) → 승인본(gen 1, step 100) 전체 다운로드, 101부터
    g2 = [cfg(names, tmp_path, gen=2, rank=r, cluster="eks-b", pod=f"g2-r{r}") for r in range(WORLD)]
    for c in g2:
        assert init_run(c, fencing, store) == EXIT_OK
        resume = json.loads((c.ckpt_dir / "resume.json").read_text())
        assert resume == {"resume": True, "approved_gen": 1, "approved_step": 100,
                          "step_dir": "step-00000100", "start_step": 101}
        for f in manifest["files"]:                                  # 모든 Pod가 manifest 전체를 받음
            src = layout.local_step_dir(g1[0 if f["name"] in (".metadata", "__0_0.distcp", "rank-state-0.pt") else 1].ckpt_dir, 100) / f["name"]
            assert (layout.local_step_dir(c.ckpt_dir, 100) / f["name"]).read_bytes() == src.read_bytes()

    # 같은 세대를 다른 클러스터가 띄우면 owner 선점 실패
    other = cfg(names, tmp_path, gen=2, rank=0, cluster="eks-c", pod="g2-other")
    assert init_run(other, fencing, store) == EXIT_OWNER

    # ⑦ 세대 2에서 step 150 저장 → 승인 (복구 단계 RESUMING 이후)
    fencing.set_phase(JOB, "recovery-7c21", "RESUMING")
    sc2 = [sidecar(c, fencing, store) for c in g2]
    for c in g2:
        fake_save(c, 150)
    sc2[1].tick()
    sc2[0].tick()
    item = fencing.get(JOB)
    assert (item.approved_gen, item.approved_step) == (2, 150)
    assert store.exists("jobs/3f9a1c2e/gen-0002/step-00000150/manifest.json")
    # 재개용으로 받은 승인본(step 100)은 새 세대로 다시 올리지 않음
    assert not any(store.exists(layout.s3_rank_list_key(JOB, 2, 100, r)) for r in range(WORLD))
    assert not store.exists(layout.s3_step_prefix(JOB, 2, 100) + "__0_0.distcp")


def test_corrupted_checkpoint_blocks_start(names, fencing, store, tmp_path):
    fencing.create_job(JOB)
    c = cfg(names, tmp_path, gen=1, rank=0, cluster="eks-a", pod="p")
    c = Config(**{**c.__dict__, "world_size": 1})
    assert init_run(c, fencing, store) == EXIT_OK
    fake_save(c, 100)
    Sidecar(c, fencing, store, heartbeat_seconds=0).tick()
    assert fencing.get(JOB).approved_step == 100

    # S3의 조각 하나를 망가뜨림 → 재개 init이 손상(7)으로 시작 안 함
    store.s3.put_object(Bucket=store.bucket, Key=layout.s3_step_prefix(JOB, 1, 100) + "__0_0.distcp", Body=b"broken")
    fencing.start_recovery(JOB, 1, "r1")
    fencing.advance_generation(JOB, 1, "r1")
    c2 = Config(**{**c.__dict__, "generation": 2, "ckpt_dir": tmp_path / "p2" / "ckpt"})
    assert init_run(c2, fencing, store) == EXIT_CORRUPT


def test_missing_rank_is_not_approved(names, fencing, store, tmp_path):
    fencing.create_job(JOB)
    c0 = cfg(names, tmp_path, gen=1, rank=0, cluster="eks-a", pod="r0")
    assert init_run(c0, fencing, store) == EXIT_OK
    fake_save(c0, 100)
    sc = Sidecar(c0, fencing, store, rank_wait_seconds=0, heartbeat_seconds=0)
    sc.tick()                     # rank 1 목록이 없음 → 대기 시작
    sc.tick()                     # 대기 시간 0초 초과 → 포기, 승인 안 함
    assert fencing.get(JOB).approved_step is None
    assert not store.exists(layout.s3_manifest_key(JOB, 1, 100))


def test_cleanup_keeps_recent_and_approved(names, fencing, store, tmp_path):
    fencing.create_job(JOB)
    c = cfg(names, tmp_path, gen=1, rank=0, cluster="eks-a", pod="p")
    c = Config(**{**c.__dict__, "world_size": 1})
    init_run(c, fencing, store)
    sc = Sidecar(c, fencing, store, keep_local=2, heartbeat_seconds=0)
    for step in (100, 200, 300):
        fake_save(c, step)
        sc.tick()
    left = sorted(p.name for p in c.ckpt_dir.iterdir() if p.is_dir())
    assert left == ["step-00000200", "step-00000300"]
