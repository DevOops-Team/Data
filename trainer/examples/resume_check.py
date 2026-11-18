"""재개 확인 (GPU-119 · GPU-209): 끊김 없이 6 step 학습한 결과 == 3 step 저장 → 새 모델로 불러와 3 step 더.

사용자 코드가 쓰는 공개 API 그대로: devoops_ckpt.fit() (+ 루프를 직접 짜는 사용자를 위한 Session의 STOP · 실패 기록).
CPU · gloo · 2 프로세스.
  python3 trainer/examples/resume_check.py          # fsdp · ddp 둘 다
torch만 있으면 된다 (GPU · transformers 불필요). 모델은 작은 Linear 묶음.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import devoops_ckpt as ckpt  # noqa: E402

WORLD, BATCH, TOTAL, CUT, SPE = 2, 4, 6, 3, 4  # 6 step 중 3에서 끊음, epoch당 4 step (재개 위치가 epoch 중간)


def config(mode, max_steps):
    raw = {
        "schema_version": 2, "model": "tiny", "training_mode": mode,
        "checkpoint_format": ckpt.CHECKPOINT_FORMATS[mode],
        "dataset": {"uri": "s3://b/datasets/x/v1/", "format": "text-jsonl", "manifest_sha256": "aa"},
        "epochs": 10, "max_steps": max_steps, "batch_size_per_gpu": BATCH, "learning_rate": 1e-2,
        "warmup_steps": 0, "seed": 0, "world_size": WORLD,
        # 간격이 길어 마지막 step에만 저장
        "checkpoint_policy": {"interval_seconds": 1e9, "estimated_save_seconds": 1, "mtbf_seconds": 1e9,
                              "warmup_measure_steps": 1},
    }
    cfg, _ = ckpt.parse_config(json.dumps(raw).encode(), None)
    # 실제로는 재개해도 config가 같다. 여기선 끊으려고 max_steps만 바꿨으니 체크포인트 config 확인용 해시는 고정
    return cfg, "cfg-sha"


def net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))


DATA = TensorDataset(torch.randn(WORLD * BATCH * SPE, 8, generator=torch.Generator().manual_seed(7)))


def loss_fn(model, x):
    return model(x).pow(2).mean()


def full_params(mode, model):
    if mode == "ddp":
        return [p.detach().clone() for p in model.module.parameters()]
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    with FSDP.summon_full_params(model):
        return [p.detach().clone() for p in model.parameters()]


def approve(ckpt_dir, step):
    # 사이드카 대신 승인 표시 (fit()이 마지막에 기다린다)
    os.makedirs(ckpt_dir, exist_ok=True)
    open(os.path.join(ckpt_dir, ckpt.step_name(step) + ".approved"), "w").close()


def progress(root):
    with open(os.path.join(root, "progress.json")) as f:
        return json.load(f)


def run(mode, root, ckpt_dir, max_steps, **kw):
    # 지표 서버 끔(METRICS_PORT 0), 진행률은 임시 파일에 (rank 0만 씀). lr은 고정(끊은 길이와 무관하게 같게)
    return ckpt.fit(model=net, dataset=DATA, loss_fn=loss_fn, config=config(mode, max_steps),
                    lr_schedule=lambda step, total: None, ckpt_dir=ckpt_dir,
                    progress_file=os.path.join(root, "progress.json"), generation=1, metrics_port=0, env={}, **kw)


def worker(rank, mode, root, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)

    a_dir = os.path.join(root, "a")               # A. 끊김 없이 6 step
    if rank == 0:
        approve(a_dir, TOTAL)
    dist.barrier()
    a = full_params(mode, run(mode, root, a_dir, TOTAL))

    ckpt_dir = os.path.join(root, "ckpt")         # B. 3 step에서 끝 → 저장 · 승인 → ckpt-init처럼 resume.json
    if rank == 0:
        approve(ckpt_dir, CUT)
        approve(ckpt_dir, TOTAL)
    dist.barrier()
    run(mode, root, ckpt_dir, CUT)
    if rank == 0:
        with open(os.path.join(ckpt_dir, "resume.json"), "w") as f:
            json.dump({"resume": True, "approved_gen": 1, "approved_step": CUT,
                       "step_dir": ckpt.step_name(CUT), "start_step": CUT + 1}, f)
    dist.barrier()

    b = full_params(mode, run(mode, root, ckpt_dir, TOTAL))   # C. 새로 띄운 것처럼: 새 모델 → 불러와 이어서
    if rank == 0:
        prog = progress(root)
        assert (prog["status"], prog["step"], prog["last_checkpoint_step"]) == ("COMPLETED", TOTAL, TOTAL), prog
    dist.barrier()

    # D. 사용자 loss 함수의 예외 → FAILED + 이유 한 줄, 예외는 그대로 올라감
    def bad(model, x):
        raise ValueError("bad batch")
    try:
        ckpt.fit(model=net, dataset=DATA, loss_fn=bad, config=config(mode, TOTAL), ckpt_dir=os.path.join(root, "d"),
                 progress_file=os.path.join(root, "progress.json"), generation=1, metrics_port=0, env={})
        raise AssertionError("error swallowed")
    except ValueError:
        pass
    if rank == 0:
        prog = progress(root)
        assert (prog["status"], prog["error"]) == ("FAILED", "ValueError: bad batch"), prog
    dist.barrier()

    # E. 펜싱 STOP(이행 기간): 종료 코드 75, 실패로 안 남김 (Session을 직접 쓰는 사용자도 같은 동작)
    if rank == 0:
        open(os.path.join(ckpt_dir, "STOP"), "w").close()
    dist.barrier()
    try:
        with ckpt.Session(mode=mode, total_steps=TOTAL, ckpt_dir=ckpt_dir, progress_file=os.path.join(root, "progress.json"),
                          generation=1, metrics_port=0, env={}) as s:
            s.report(TOTAL + 1, 0.0)                  # float loss → all_reduce 없음
        raise AssertionError("STOP ignored")
    except SystemExit as e:
        assert e.code == ckpt.STOP_EXIT, e.code

    diff = max(float((x - y).abs().max()) for x, y in zip(a, b))
    if rank == 0:
        print(f"{mode}: 끊김 없이 {TOTAL} step vs {CUT}에서 저장 · 재개 → 최대 차이 {diff:.3e}", flush=True)
    dist.destroy_process_group()
    if diff > 1e-6:
        raise SystemExit(1)


def main():
    ok = True
    for i, mode in enumerate(sys.argv[1:] or ["fsdp", "ddp"]):
        with tempfile.TemporaryDirectory() as root:
            try:
                mp.spawn(worker, args=(mode, root, 29610 + i), nprocs=WORLD, join=True)
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"{mode}: 실패 {e}", flush=True)
    print("결과:", "통과" if ok else "실패")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
