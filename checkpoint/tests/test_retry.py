"""재시도 정책 (규약 10절 · 한송이 「장애 복구 규약」 4절).

- 연결 오류는 재시도하고, 끝내 안 되면 init 종료 코드 4 / 사이드카는 그 step만 미승인
- 손상은 재시도로 해결하지 않고 init 종료 코드 7
- 사이드카는 연결 오류로 STOP하거나 죽지 않는다
"""
from pathlib import Path

from botocore.exceptions import EndpointConnectionError

from ckpt import layout
from ckpt.config import Config
from ckpt.init_main import EXIT_CORRUPT, EXIT_DOWNLOAD, EXIT_OK
from ckpt.init_main import run as init_run
from ckpt.retry import no_wait
from ckpt.sidecar import Sidecar

JOB = "retryjob"


def cfg(names, tmp: Path, *, gen=1, pod="p", cluster="eks-a") -> Config:
    root = tmp / pod
    return Config(job_id=JOB, generation=gen, cluster=cluster, bucket=names["bucket"], table=names["table"],
                  ckpt_dir=root / "ckpt", code_dir=root / "code", data_dir=root / "data", world_size=1, rank=0,
                  checkpoint_format="fsdp-dcp", config_uri=None, config_sha256=None, dataset_uri=None,
                  dataset_manifest_sha256=None, region="us-east-2")


def save(c: Config, step: int) -> None:
    d = layout.local_step_dir(c.ckpt_dir, step)
    d.mkdir(parents=True, exist_ok=True)
    (d / "__0_0.distcp").write_bytes(b"x" * 1000 + str(step).encode())
    layout.done_marker(c.ckpt_dir, step).touch()


def flaky(fn, fail_times: int):
    """처음 fail_times번은 연결 오류, 그 뒤엔 원래 함수."""
    state = {"n": 0}

    def wrapper(*a, **k):
        if state["n"] < fail_times:
            state["n"] += 1
            raise EndpointConnectionError(endpoint_url="https://s3.test")
        return fn(*a, **k)
    wrapper.state = state
    return wrapper


def approved_job(names, fencing, store, tmp_path):
    """세대 1에서 step 100 승인 → 세대 2로 올린 상태를 만든다."""
    fencing.create_job(JOB)
    c = cfg(names, tmp_path, pod="g1")
    assert init_run(c, fencing, store) == EXIT_OK
    save(c, 100)
    Sidecar(c, fencing, store, heartbeat_seconds=0).tick()
    assert fencing.get(JOB).approved_step == 100
    fencing.start_recovery(JOB, 1, "r1")
    fencing.advance_generation(JOB, 1, "r1")
    return cfg(names, tmp_path, gen=2, pod="g2", cluster="eks-b")


def test_init_download_recovers_from_transient(names, fencing, store, tmp_path):
    c2 = approved_job(names, fencing, store, tmp_path)
    store.s3.download_file = flaky(store.s3.download_file, 2)       # 2번 끊기고 3번째 성공
    assert init_run(c2, fencing, store, no_wait(3)) == EXIT_OK
    assert store.s3.download_file.state["n"] == 2


def test_init_download_gives_up_with_4(names, fencing, store, tmp_path):
    c2 = approved_job(names, fencing, store, tmp_path)
    store.s3.download_file = flaky(store.s3.download_file, 99)      # 계속 끊김
    assert init_run(c2, fencing, store, no_wait(3)) == EXIT_DOWNLOAD


def test_init_missing_file_is_corrupt_7(names, fencing, store, tmp_path):
    c2 = approved_job(names, fencing, store, tmp_path)
    store.s3.delete_object(Bucket=store.bucket, Key=layout.s3_step_prefix(JOB, 1, 100) + "__0_0.distcp")
    assert init_run(c2, fencing, store, no_wait(3)) == EXIT_CORRUPT


def test_init_ddb_unreachable_is_4(names, fencing, store, tmp_path):
    fencing.create_job(JOB)
    fencing.ddb.get_item = flaky(fencing.ddb.get_item, 99)
    assert init_run(cfg(names, tmp_path), fencing, store, no_wait(3)) == EXIT_DOWNLOAD


def test_sidecar_survives_ddb_outage_without_stop(names, fencing, store, tmp_path):
    fencing.create_job(JOB)
    c = cfg(names, tmp_path)
    assert init_run(c, fencing, store) == EXIT_OK
    save(c, 100)
    real = fencing.ddb.get_item
    fencing.ddb.get_item = flaky(real, 2)
    sc = Sidecar(c, fencing, store, heartbeat_seconds=0)
    sc.tick()                                                        # 연결 오류 → 이번 사이클만 건너뜀
    assert not sc.stopped and not layout.stop_file(c.ckpt_dir).exists()
    sc.tick()
    sc.tick()                                                        # 복구된 뒤 승인
    assert fencing.get(JOB).approved_step == 100
    assert not layout.stop_file(c.ckpt_dir).exists()


def test_sidecar_upload_retries_then_succeeds(names, fencing, store, tmp_path):
    fencing.create_job(JOB)
    c = cfg(names, tmp_path)
    init_run(c, fencing, store)
    save(c, 100)
    store.retry = no_wait(3)
    store.s3.upload_file = flaky(store.s3.upload_file, 2)
    Sidecar(c, fencing, store, heartbeat_seconds=0).tick()
    assert fencing.get(JOB).approved_step == 100


def test_sidecar_upload_exhausted_not_approved_no_stop(names, fencing, store, tmp_path):
    fencing.create_job(JOB)
    c = cfg(names, tmp_path)
    init_run(c, fencing, store)
    save(c, 100)
    store.retry = no_wait(3)
    real_upload = store.s3.upload_file
    store.s3.upload_file = flaky(real_upload, 99)
    sc = Sidecar(c, fencing, store, heartbeat_seconds=0)
    sc.tick()
    assert fencing.get(JOB).approved_step is None                    # 이 step은 미승인
    assert not layout.stop_file(c.ckpt_dir).exists() and not sc.stopped
    store.s3.upload_file = real_upload                               # 네트워크 복구 → 다음 저장은 승인
    save(c, 200)
    sc.tick()
    assert fencing.get(JOB).approved_step == 200
