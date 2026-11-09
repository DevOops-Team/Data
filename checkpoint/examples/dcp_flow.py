"""실제 DCP(FSDP2) + ckpt-init + 사이드카 흐름 확인 (moto, AWS 불필요).

A. 기준: 체크포인트 없이 step 1 → 30 끊김 없이 학습 → 체크섬 A
B. 세대 1: init → 학습(저장 10 · 20) → step 25에서 강제 종료 → 사이드카가 20까지 승인
   복구: 세대 +1 → 세대 2 init(새 폴더 = 새 Pod)이 승인본 20 다운로드 → 21부터 30까지 → 체크섬 B
   A == B 이면 재개가 정확하다.
C. 옛 세대 차단: 세대 1 설정으로 trainer를 다시 띄우면 사이드카가 STOP → trainer 종료 코드 75
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import boto3  # noqa: E402
from moto.server import ThreadedMotoServer  # noqa: E402

from ckpt import layout  # noqa: E402
from ckpt.config import Config  # noqa: E402
from ckpt.fencing import FencingTable, create_table  # noqa: E402
from ckpt.init_main import EXIT_OK  # noqa: E402
from ckpt.init_main import run as init_run  # noqa: E402
from ckpt.sidecar import Sidecar  # noqa: E402
from ckpt.storage import Store  # noqa: E402

REGION, WORLD, JOB = "us-east-2", 2, "dcpflow"
PY = sys.executable


def torchrun(ckpt_root: Path, extra: dict) -> tuple[int, str]:
    env = {**os.environ, "CKPT_ROOT": str(ckpt_root), "TOTAL_STEPS": "30", "SAVE_EVERY": "10",
           "WAIT_APPROVED_S": "30", **{k: str(v) for k, v in extra.items()}}
    p = subprocess.run([PY, "-m", "torch.distributed.run", "--nproc-per-node", str(WORLD),
                        "--master-port", str(29500 + int(time.time()) % 1000), str(HERE / "tiny_trainer.py")],
                       env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    print("\n".join(l for l in out.splitlines() if l.startswith(("[rank", "CHECKSUM"))), flush=True)
    return p.returncode, out


def checksum(out: str) -> str | None:
    m = re.search(r"CHECKSUM (\w+)", out)
    return m.group(1) if m else None


class SidecarThreads:
    """사이드카 컨테이너 흉내: rank마다 0.3초 간격으로 tick."""
    def __init__(self, scs):
        self.scs, self.stop_ev = scs, threading.Event()
        self.ts = [threading.Thread(target=self._loop, args=(s,), daemon=True) for s in scs]

    def _loop(self, sc):
        while not self.stop_ev.is_set():
            try:
                sc.tick()
            except Exception as e:  # noqa: BLE001
                print("sidecar error", e, flush=True)
            time.sleep(0.3)

    def __enter__(self):
        for t in self.ts:
            t.start()
        return self

    def __exit__(self, *a):
        time.sleep(1.5)  # 마지막 업로드 · 승인 마무리
        self.stop_ev.set()
        for t in self.ts:
            t.join()


def main() -> int:
    srv = ThreadedMotoServer(ip_address="127.0.0.1", port=5056, verbose=False)
    srv.start()
    os.environ.update(AWS_ENDPOINT_URL="http://127.0.0.1:5056", AWS_ACCESS_KEY_ID="t",
                      AWS_SECRET_ACCESS_KEY="t", AWS_DEFAULT_REGION=REGION)
    s3, ddb = boto3.client("s3", region_name=REGION), boto3.client("dynamodb", region_name=REGION)
    s3.create_bucket(Bucket="dcpflow-bucket", CreateBucketConfiguration={"LocationConstraint": REGION})
    create_table(ddb, "t")
    fencing, store = FencingTable("t", client=ddb), Store("dcpflow-bucket", client=s3)
    tmp = Path(tempfile.mkdtemp(prefix="dcpflow-"))
    ok = True

    def cfg(root, gen, rank, cluster):
        return Config(job_id=JOB, generation=gen, cluster=cluster, bucket="dcpflow-bucket", table="t",
                      ckpt_dir=root / f"r{rank}", code_dir=root / "code", data_dir=root / "data",
                      world_size=WORLD, rank=rank, checkpoint_format="fsdp-dcp", config_uri=None,
                      config_sha256=None, dataset_uri=None, dataset_manifest_sha256=None, region=REGION)

    def report(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)

    # A. 기준 (끊김 없이)
    print("== A. 기준 학습 (1 → 30, 체크포인트 없음)", flush=True)
    code, out = torchrun(tmp / "baseline", {"SAVE_EVERY": "1000"})
    sum_a = checksum(out)
    report("A 정상 종료", code == 0 and sum_a, f"code={code}")

    # B. 세대 1 → 장애 → 세대 2 재개
    print("== B-1. 세대 1 (저장 10 · 20, step 25에서 강제 종료)", flush=True)
    fencing.create_job(JOB)
    g1 = [cfg(tmp / "g1", 1, r, "eks-a") for r in range(WORLD)]
    report("B init 세대 1", [init_run(c, fencing, store) for c in g1] == [EXIT_OK] * WORLD)
    with SidecarThreads([Sidecar(c, fencing, store, rank_wait_seconds=60, heartbeat_seconds=1) for c in g1]):
        code, out = torchrun(tmp / "g1", {"CRASH_AT": "25"})
    report("B 세대 1 강제 종료", code != 0, f"code={code}")
    it = fencing.get(JOB)
    report("B 승인본 = 1 · 20", (it.approved_gen, it.approved_step) == (1, 20), f"{it.approved_gen}·{it.approved_step}")
    files = store.get_json(layout.s3_manifest_key(JOB, 1, 20))["files"]
    print("   manifest:", [f["name"] for f in files], flush=True)

    print("== B-2. 복구: 세대 +1 → 세대 2 (새 Pod, eks-b)", flush=True)
    fencing.start_recovery(JOB, 1, "r1")
    fencing.advance_generation(JOB, 1, "r1")
    g2 = [cfg(tmp / "g2", 2, r, "eks-b") for r in range(WORLD)]
    report("B init 세대 2", [init_run(c, fencing, store) for c in g2] == [EXIT_OK] * WORLD)
    fencing.set_phase(JOB, "r1", "RESUMING")
    with SidecarThreads([Sidecar(c, fencing, store, rank_wait_seconds=60, heartbeat_seconds=1) for c in g2]):
        code, out = torchrun(tmp / "g2", {})
    sum_b = checksum(out)
    report("B 세대 2 정상 종료 · 21부터 재개", code == 0 and "start 21" in out, f"code={code}")
    it = fencing.get(JOB)
    report("B 승인본 = 2 · 30", (it.approved_gen, it.approved_step) == (2, 30), f"{it.approved_gen}·{it.approved_step}")
    report("A == B (재개 결과가 끊김 없는 학습과 같음)", sum_a == sum_b, f"A={sum_a[:12] if sum_a else None} B={sum_b[:12] if sum_b else None}")

    # C. 학습 도중 세대가 바뀌면 옛 실행이 스스로 멈추는지
    print("== C. 세대 2 학습 도중(step 40 저장 후) 세대 +1 → 옛 실행 STOP", flush=True)
    fencing.set_phase(JOB, "r1", "COMPLETED")
    g2b = [cfg(tmp / "g2b", 2, r, "eks-b") for r in range(WORLD)]
    report("C init 세대 2 (같은 클러스터 재시작)", [init_run(c, fencing, store) for c in g2b] == [EXIT_OK] * WORLD)

    def advance_after_step40():
        mark = layout.done_marker(g2b[0].ckpt_dir, 40)
        while not mark.exists():
            time.sleep(0.1)
        time.sleep(0.5)
        fencing.start_recovery(JOB, 2, "r2")
        fencing.advance_generation(JOB, 2, "r2")
        print("   (복구 컨트롤러 흉내) 세대 2 → 3", flush=True)

    t = threading.Thread(target=advance_after_step40, daemon=True)
    t.start()
    with SidecarThreads([Sidecar(c, fencing, store, rank_wait_seconds=60, heartbeat_seconds=0.5) for c in g2b]):
        code, out = torchrun(tmp / "g2b", {"TOTAL_STEPS": "80", "STEP_SLEEP": "0.1"})
    t.join(timeout=5)
    report("C 옛 실행 종료 코드 75 (STOP)", code != 0 and "STOP 감지" in out, f"code={code}")
    it = fencing.get(JOB)
    report("C 세대 3 · 승인본은 세대 2의 40 이하", it.generation == 3 and it.approved_gen == 2 and it.approved_step <= 40,
           f"gen={it.generation} approved={it.approved_gen}·{it.approved_step}")
    report("C 세대 2의 step 50 이후는 승인 안 됨", not store.exists(layout.s3_manifest_key(JOB, 2, 50)))

    srv.stop()
    print("\nRESULT:", "ALL PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
