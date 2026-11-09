"""1단계 테스트: 맥 → 실제 AWS S3 · DynamoDB.

Confluence 「체크포인트 모듈 테스트 — 1단계: 맥 → 실제 AWS (GPU-64)」 5절 흐름 ①~⑨를 그대로 실행한다.
작업 ID는 test-로 시작하고, 끝나면(실패해도) S3 · DynamoDB의 테스트 데이터를 지운다.

실행
  CKPT_BUCKET=devoops-prod-ckpt-354363695096-us-east-2 FENCING_TABLE=devoops-prod-fencing \\
  AWS_REGION=us-east-2 python scripts/aws_smoke.py [--mb 64] [--keep]

로컬 검증(moto): --moto 를 주면 moto 서버를 띄워 같은 흐름을 돌린다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import boto3  # noqa: E402

from ckpt import layout  # noqa: E402
from ckpt.config import Config  # noqa: E402
from ckpt.fencing import ApproveResult, FencingTable, create_table  # noqa: E402
from ckpt.init_main import EXIT_DOWNLOAD, EXIT_GEN, EXIT_OK, EXIT_OWNER  # noqa: E402
from ckpt.init_main import run as init_run  # noqa: E402
from ckpt.sidecar import Sidecar  # noqa: E402
from ckpt.storage import Store  # noqa: E402

WORLD = 2
results: list[tuple[str, bool, str]] = []
timings: dict[str, float] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def make_cfg(job, bucket, table, region, root: Path, *, gen, rank, cluster, world=WORLD) -> Config:
    return Config(job_id=job, generation=gen, cluster=cluster, bucket=bucket, table=table,
                  ckpt_dir=root / "ckpt", code_dir=root / "code", data_dir=root / "data",
                  world_size=world, rank=rank, checkpoint_format="fsdp-dcp", config_uri=None,
                  config_sha256="smoke-cfg", dataset_uri=None, dataset_manifest_sha256=None, region=region)


def fake_save(c: Config, step: int, mb: int) -> None:
    d = layout.local_step_dir(c.ckpt_dir, step)
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"__{c.rank}_0.distcp").open("wb") as f:
        for _ in range(mb):
            f.write(os.urandom(1024 * 1024))
    (d / f"rank-state-{c.rank}.pt").write_bytes(f"rng-{c.rank}-{step}".encode())
    if c.rank == 0:
        (d / ".metadata").write_bytes(f"dcp-metadata-{step}".encode())
    layout.done_marker(c.ckpt_dir, step).touch()


def save_and_approve(g, scs, step, mb):
    for c in g:
        fake_save(c, step, mb)
    t0 = time.monotonic()
    for sc in reversed(scs):  # rank 1 먼저, rank 0이 모아서 승인
        sc.tick()
    for sc in scs[1:]:
        sc.tick()
    timings[f"save_to_approve_step{step}_s"] = round(time.monotonic() - t0, 2)


def run(bucket: str, table: str, region: str, mb: int, keep: bool) -> int:
    s3 = boto3.client("s3", region_name=region)
    ddb = boto3.client("dynamodb", region_name=region)
    fencing, store = FencingTable(table, client=ddb), Store(bucket, client=s3)
    job = f"test-{time.strftime('%m%d')}-{uuid.uuid4().hex[:4]}"
    tmp = Path(tempfile.mkdtemp(prefix="ckpt-smoke-"))
    print(f"job_id={job} bucket={bucket} table={table} tmp={tmp}", flush=True)
    try:
        # ① 준비
        ident = boto3.client("sts", region_name=region).get_caller_identity()
        print(f"caller={ident['Arn']}", flush=True)
        check("① 첫 항목 생성", fencing.create_job(job))

        # ② 세대 1 시작
        g1 = [make_cfg(job, bucket, table, region, tmp / f"g1-r{r}", gen=1, rank=r, cluster="local-a") for r in range(WORLD)]
        codes = [init_run(c, fencing, store) for c in g1]
        check("② init 세대 1", codes == [EXIT_OK] * WORLD, f"codes={codes}")
        check("② owner = local-a", fencing.get(job).owner == "local-a")
        sc1 = [Sidecar(c, fencing, store, rank_wait_seconds=120, keep_local=2, heartbeat_seconds=0) for c in g1]

        # ③ 저장 · 승인
        save_and_approve(g1, sc1, 100, mb)
        it = fencing.get(job)
        check("③ step 100 승인", (it.approved_gen, it.approved_step) == (1, 100), f"{it.approved_gen}·{it.approved_step}")
        check("③ manifest 존재", store.exists(layout.s3_manifest_key(job, 1, 100)))
        save_and_approve(g1, sc1, 200, mb)
        it = fencing.get(job)
        check("③ step 200 승인 · prev 100", (it.approved_step, int(it.raw.get("prev_approved_step", {"N": "-1"})["N"])) == (200, 100))

        # ④ 장애 → 세대 +1
        rid = f"recovery-{uuid.uuid4().hex[:4]}"
        check("④ CLAIMED", fencing.start_recovery(job, 1, rid))
        check("④ 복구 중 승인 금지", fencing.approve(job, 1, "local-a", 999) is ApproveResult.RECOVERING)
        check("④ 세대 +1", fencing.advance_generation(job, 1, rid))
        it = fencing.get(job)
        check("④ gen 2 · owner 없음 · 승인본 유지",
              (it.generation, it.owner, it.approved_gen, it.approved_step) == (2, None, 1, 200))
        check("④ 같은 복구로 재증가 거부", not fencing.advance_generation(job, 1, rid))

        # ⑤ 옛 세대 차단
        for c in g1:
            fake_save(c, 300, 1)
        sc1[0].tick()
        check("⑤ 옛 세대 STOP", layout.stop_file(g1[0].ckpt_dir).exists())
        check("⑤ 옛 세대 미업로드", not store.exists(layout.s3_rank_list_key(job, 1, 300, 0)))
        check("⑤ 옛 세대 init 거부", init_run(g1[0], fencing, store) == EXIT_GEN)

        # ⑥ 세대 2 재개
        g2 = [make_cfg(job, bucket, table, region, tmp / f"g2-r{r}", gen=2, rank=r, cluster="local-b") for r in range(WORLD)]
        t0 = time.monotonic()
        codes = [init_run(c, fencing, store) for c in g2]
        timings["resume_download_s"] = round(time.monotonic() - t0, 2)
        check("⑥ init 세대 2", codes == [EXIT_OK] * WORLD, f"codes={codes}")
        resume = json.loads((g2[0].ckpt_dir / "resume.json").read_text())
        check("⑥ start_step 201", resume["start_step"] == 201, json.dumps(resume))
        other = make_cfg(job, bucket, table, region, tmp / "g2-other", gen=2, rank=0, cluster="local-c")
        check("⑥ 다른 클러스터 선점 실패", init_run(other, fencing, store) == EXIT_OWNER)

        # ⑦ 세대 2 저장 · 승인
        fencing.set_phase(job, rid, "RESUMING")
        sc2 = [Sidecar(c, fencing, store, rank_wait_seconds=120, keep_local=2, heartbeat_seconds=0) for c in g2]
        save_and_approve(g2, sc2, 250, mb)
        it = fencing.get(job)
        check("⑦ approved 2·250 · prev 1·200",
              (it.approved_gen, it.approved_step, int(it.raw["prev_approved_gen"]["N"]), int(it.raw["prev_approved_step"]["N"])) == (2, 250, 1, 200))

        # ⑧ 실패 경로
        s3.put_object(Bucket=bucket, Key=layout.s3_step_prefix(job, 2, 250) + "__1_0.distcp", Body=b"broken")
        rid2 = f"recovery-{uuid.uuid4().hex[:4]}"
        fencing.set_phase(job, rid, "COMPLETED")
        fencing.start_recovery(job, 2, rid2)
        fencing.advance_generation(job, 2, rid2)
        g3 = make_cfg(job, bucket, table, region, tmp / "g3", gen=3, rank=0, cluster="local-d")
        check("⑧ 손상 → init 시작 안 함", init_run(g3, fencing, store) == EXIT_DOWNLOAD)

        job2 = job + "-x"
        fencing.create_job(job2)
        c0 = make_cfg(job2, bucket, table, region, tmp / "x-r0", gen=1, rank=0, cluster="local-a")
        init_run(c0, fencing, store)
        fake_save(c0, 100, 1)
        sc = Sidecar(c0, fencing, store, rank_wait_seconds=0, heartbeat_seconds=0)
        sc.tick(); sc.tick()
        check("⑧ rank 누락 → 승인 안 함", fencing.get(job2).approved_step is None)
    except Exception as e:  # noqa: BLE001
        check("예외 없이 끝까지", False, repr(e))
    finally:
        if not keep:
            cleanup(s3, ddb, bucket, table, [job, job + "-x"])
    print("timings=" + json.dumps(timings), flush=True)
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} PASS", flush=True)
    return 1 if failed else 0


def cleanup(s3, ddb, bucket, table, jobs) -> None:
    # ⑨ 정리
    for job in jobs:
        keys = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"jobs/{job}/"):
            keys += [{"Key": o["Key"]} for o in page.get("Contents", [])]
        for i in range(0, len(keys), 1000):
            s3.delete_objects(Bucket=bucket, Delete={"Objects": keys[i:i + 1000]})
        ddb.delete_item(TableName=table, Key={"job_id": {"S": job}})
        left = s3.list_objects_v2(Bucket=bucket, Prefix=f"jobs/{job}/").get("KeyCount", 0)
        gone = "Item" not in ddb.get_item(TableName=table, Key={"job_id": {"S": job}})
        check(f"⑨ 정리 {job}", left == 0 and gone, f"s3_left={left}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=64, help="rank당 조각 크기(MB)")
    ap.add_argument("--keep", action="store_true", help="테스트 데이터를 지우지 않음")
    ap.add_argument("--moto", action="store_true", help="실제 AWS 대신 moto 서버로 실행")
    a = ap.parse_args()
    region = os.environ.get("AWS_REGION", "us-east-2")
    if a.moto:
        from moto.server import ThreadedMotoServer
        srv = ThreadedMotoServer(ip_address="127.0.0.1", port=5055, verbose=False)
        srv.start()
        os.environ.update(AWS_ENDPOINT_URL="http://127.0.0.1:5055", AWS_ACCESS_KEY_ID="t", AWS_SECRET_ACCESS_KEY="t")
        bucket, table = "smoke-bucket", "smoke-fencing"
        boto3.client("s3", region_name=region).create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
        create_table(boto3.client("dynamodb", region_name=region), table)
    else:
        bucket, table = os.environ["CKPT_BUCKET"], os.environ["FENCING_TABLE"]
        if "AWS_ENDPOINT_URL" in os.environ:
            sys.exit("실제 AWS 테스트인데 AWS_ENDPOINT_URL이 설정돼 있습니다. 지우고 다시 실행하세요.")
    sys.exit(run(bucket, table, region, a.mb, a.keep))


if __name__ == "__main__":
    main()
