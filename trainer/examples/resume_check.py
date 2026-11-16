"""재개 확인 (GPU-119): 끊김 없이 6 step 학습한 결과 == 3 step 저장 → 새 모델 · 옵티마이저로 불러와 3 step 더.

train.py의 save_checkpoint · read_resume · load_checkpoint · resume_position을 그대로 쓴다. CPU · gloo · 2 프로세스.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import train  # noqa: E402

WORLD, TOTAL, CUT, SPE = 2, 6, 3, 4  # 6 step 중 3에서 끊음, epoch당 4 step (재개 위치가 epoch 중간)
CFG = "cfg-sha"


def make(mode):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
    if mode == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(net)
    else:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        # train.wrap_model과 같은 설정 (device만 CPU로 명시: 맥에서는 MPS를 잡으려 해서)
        model = FSDP(net, auto_wrap_policy=ModuleWrapPolicy({nn.Linear}), use_orig_params=True,
                     device_id=torch.device("cpu"))
    return model, torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)


def batches(rank):
    # train.py처럼 seed · epoch로 순서가 정해지는 데이터: epoch e의 i번째 배치
    def get(epoch, i):
        g = torch.Generator().manual_seed(1000 * epoch + 10 * i + rank)
        return torch.randn(4, 8, generator=g)
    return get


def run(model, opt, get, start, total):
    epoch, skip = train.resume_position(start, SPE)
    step = start
    while step < total:
        for i in range(SPE):
            if skip:
                skip -= 1
                continue
            x = get(epoch, i)
            opt.zero_grad(set_to_none=True)
            model(x).pow(2).mean().backward()
            opt.step()
            step += 1
            if step >= total:
                break
        epoch += 1


def full_params(mode, model):
    if mode == "ddp":
        return [p.detach().clone() for p in model.module.parameters()]
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    with FSDP.summon_full_params(model):
        return [p.detach().clone() for p in model.parameters()]


def worker(rank, mode, root, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    get = batches(rank)
    cpu = torch.device("cpu")

    model, opt = make(mode)                       # A. 끊김 없이 6 step
    run(model, opt, get, 0, TOTAL)
    a = full_params(mode, model)

    ckpt = os.path.join(root, "ckpt")             # B. 3 step → 저장 → ckpt-init처럼 resume.json
    os.makedirs(ckpt, exist_ok=True)
    model, opt = make(mode)
    run(model, opt, get, 0, CUT)
    train.save_checkpoint(mode, CUT, model, opt, CFG, rank, ckpt)
    if rank == 0:
        with open(os.path.join(ckpt, "resume.json"), "w") as f:
            json.dump({"resume": True, "approved_gen": 1, "approved_step": CUT,
                       "step_dir": train.step_name(CUT), "start_step": CUT + 1}, f)
    dist.barrier()

    model, opt = make(mode)                       # C. 새로 띄운 것처럼: 새 모델 · 옵티마이저 → 불러와 이어서
    start, step_dir = train.read_resume(ckpt)
    train.load_checkpoint(mode, model, opt, step_dir, cpu, start, CFG)
    run(model, opt, get, start, TOTAL)
    b = full_params(mode, model)

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
