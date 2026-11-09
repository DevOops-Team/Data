"""작은 FSDP2 trainer (CPU · gloo). 체크포인트 규약의 trainer 쪽 약속을 지키는 예시.

약속 (규약 3 · 7절)
- 저장: dcp.save → $CKPT_DIR/step-XXXXXXXX/ 다 쓴 뒤 마지막에 step-XXXXXXXX.done
- 중단: step마다 $CKPT_DIR/STOP 확인 → 있으면 저장 없이 종료 코드 75
- 재개: $CKPT_DIR/resume.json의 step_dir를 dcp.load → start_step부터

실행: torchrun --nproc-per-node 2 tiny_trainer.py
환경 변수: CKPT_ROOT(rank별 폴더 = CKPT_ROOT/r{rank}, Pod마다 로컬 디스크가 다른 것을 흉내),
          TOTAL_STEPS, SAVE_EVERY, CRASH_AT(이 step 직후 강제 종료), WAIT_APPROVED_S
데이터는 step 번호로 고정해서, 끊김 없이 돈 결과와 재개한 결과의 파라미터가 같아야 한다.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

STOP_EXIT = 75


def step_name(s: int) -> str:
    return f"step-{s:08d}"


def main() -> None:
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    ckpt = Path(os.environ.get("CKPT_DIR") or Path(os.environ["CKPT_ROOT"]) / f"r{rank}")
    ckpt.mkdir(parents=True, exist_ok=True)
    total = int(os.environ.get("TOTAL_STEPS", "30"))
    save_every = int(os.environ.get("SAVE_EVERY", "10"))
    crash_at = int(os.environ.get("CRASH_AT", "0"))
    wait_approved = float(os.environ.get("WAIT_APPROVED_S", "120"))
    step_sleep = float(os.environ.get("STEP_SLEEP", "0"))  # 학습 시간 흉내 (테스트용)

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4))
    mesh = init_device_mesh("cpu", (world,))
    for layer in model:
        if isinstance(layer, nn.Linear):
            fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # 재개
    start = 1
    rj = ckpt / "resume.json"
    if rj.exists():
        r = json.loads(rj.read_text())
        if r["resume"]:
            msd, osd = get_state_dict(model, opt)
            state = {"model": msd, "optim": osd}
            dcp.load(state, checkpoint_id=str(ckpt / r["step_dir"]))
            set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
            start = r["start_step"]
            print(f"[rank{rank}] resumed from {r['step_dir']} → start {start}", flush=True)

    last_saved = None
    for step in range(start, total + 1):
        # 중단 확인 (모든 rank가 같이 판단: 하나라도 STOP이면 전부 종료)
        flag = torch.tensor([1 if (ckpt / "STOP").exists() else 0])
        dist.all_reduce(flag)
        if flag.item() > 0:
            print(f"[rank{rank}] STOP 감지 → 저장 없이 종료 {STOP_EXIT}", flush=True)
            dist.destroy_process_group()
            sys.exit(STOP_EXIT)

        g = torch.Generator().manual_seed(step * 1000 + rank)   # step으로 고정한 데이터
        x = torch.randn(16, 32, generator=g)
        y = torch.randn(16, 4, generator=g)
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step_sleep:
            time.sleep(step_sleep)

        if step % save_every == 0:
            sdir = ckpt / step_name(step)
            msd, osd = get_state_dict(model, opt)
            dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(sdir))
            dist.barrier()
            (ckpt / f"{step_name(step)}.done").touch()   # 다 쓴 뒤 마지막에
            last_saved = step
            print(f"[rank{rank}] saved {step_name(step)}", flush=True)

        if crash_at and step == crash_at:
            print(f"[rank{rank}] 강제 종료 (장애 흉내) at step {step}", flush=True)
            os._exit(1)

    # 마지막 저장본 승인 대기 (규약 3절)
    if last_saved is not None:
        mark = ckpt / f"{step_name(last_saved)}.approved"
        t0 = time.monotonic()
        while not mark.exists() and time.monotonic() - t0 < wait_approved:
            time.sleep(0.2)
        print(f"[rank{rank}] approved wait {'ok' if mark.exists() else 'timeout'}", flush=True)

    # 최종 파라미터 체크섬 (전체 텐서 기준, 모든 rank 같은 값)
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.full_tensor().detach().numpy().tobytes() if hasattr(p, "full_tensor") else p.detach().numpy().tobytes())
    if rank == 0:
        print(f"CHECKSUM {h.hexdigest()}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
