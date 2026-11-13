"""Hugging Face causal LM FSDP · DDP 학습. 설정은 config.json 하나로 받는다 (Confluence `학습 파라미터 주입 방식`).

기본 학습 코드다. 사용자가 제출 때 올리면 ckpt-init이 DATA_DIR/code/train.py로 받아 두고,
백엔드가 만든 JobSet의 torchrun이 trainer 이미지(학습 환경) 위에서 Pod마다 실행한다.
  - TRAIN_CONFIG(/code/config.json)을 읽고 CONFIG_SHA256 · 키 · world_size를 확인한다.
  - ckpt-init이 받아 둔 업로드: 모델 폴더 DATA_DIR/model, 학습 데이터 DATA_DIR/data/*.jsonl ({"text": ...}).
  - 어떤 HF 모델이든 AutoModelForCausalLM으로 읽는다. FSDP는 모델의 _no_split_modules(트랜스포머 블록) 단위로 감싼다.
  - rank 0이 매 step PROGRESS_FILE에 상태를 쓰면 progress-reporter가 DynamoDB로 옮긴다.

체크포인트 (설계 5절 · JobSet 템플릿 7절):
  - 간격: config의 checkpoint_policy.interval_seconds(τ, Young/Daly)를 처음 warmup_measure_steps step의
    시간으로 step 간격으로 바꾼다. 예상 학습 시간 T = 전체 step × step 시간, 저장 횟수 n = max(1, ⌊T/τ⌋),
    step 간격 k = ⌈전체 step / n⌉. 마지막 step은 항상 저장한다.
  - 저장: 모든 rank가 CKPT_DIR/step-XXXXXXXX/에 쓴 뒤 마지막에 step-XXXXXXXX.done (DDP는 rank 0만 파일,
    나머지 rank는 빈 디렉토리 + .done). 각 Pod의 ckpt-sidecar가 올리고 승인한다.
  - 중단: step마다 CKPT_DIR/STOP을 보고, 있으면 저장 없이 종료 코드 75.
  - 끝: 마지막 저장 뒤 .approved를 최대 120초 기다린다. 넘으면 실패로 끝낸다.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import socket
import sys
import time
from dataclasses import dataclass, fields

SCHEMA_VERSION = 2
CHECKPOINT_FORMATS = {"fsdp": "fsdp-dcp", "ddp": "ddp-pt"}
MAX_BLOCK_SIZE = 1024  # 긴 문맥 모델도 g4dn(16GB)에 맞게 자른다
STOP_EXIT = 75  # ckpt-sidecar가 STOP을 썼을 때. 장애로 분류하지 않는다
APPROVAL_WAIT_SECONDS = 120


@dataclass(frozen=True)
class DatasetRef:
    uri: str
    format: str
    manifest_sha256: str


@dataclass(frozen=True)
class CheckpointPolicy:
    interval_seconds: float
    estimated_save_seconds: float
    mtbf_seconds: float
    warmup_measure_steps: int


@dataclass(frozen=True)
class TrainConfig:  # 기본값 없음: 백엔드가 모든 키를 채운다
    model: str
    training_mode: str
    checkpoint_format: str
    dataset: DatasetRef
    epochs: int
    max_steps: int | None
    batch_size_per_gpu: int
    learning_rate: float
    warmup_steps: int
    seed: int
    world_size: int
    checkpoint_policy: CheckpointPolicy


def parse_config(raw: bytes, expected_sha: str | None) -> tuple[TrainConfig, str]:
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha and expected_sha != digest:
        raise SystemExit(f"config hash mismatch: {digest} != {expected_sha}")

    data = json.loads(raw)
    if data.pop("schema_version", None) != SCHEMA_VERSION:
        raise SystemExit(f"config schema_version must be {SCHEMA_VERSION}")
    names = {f.name for f in fields(TrainConfig)}
    if set(data) != names:  # 오타 · 누락을 조용히 넘기지 않는다
        raise SystemExit(f"config keys: missing {names - set(data)}, unknown {set(data) - names}")
    data["dataset"] = DatasetRef(**data["dataset"])
    data["checkpoint_policy"] = CheckpointPolicy(**data["checkpoint_policy"])
    cfg = TrainConfig(**data)
    want = CHECKPOINT_FORMATS.get(cfg.training_mode)
    if want is None:  # 다른 방식으로 돌면서 요청한 방식으로 보이는 일을 막는다
        raise SystemExit(f"this train.py does not implement {cfg.training_mode}")
    if cfg.checkpoint_format != want:
        raise SystemExit(f"{cfg.training_mode} needs checkpoint_format {want}, got {cfg.checkpoint_format}")
    if cfg.dataset.format != "text-jsonl":
        raise SystemExit(f"unsupported dataset format {cfg.dataset.format}")
    return cfg, digest


def load_config() -> tuple[TrainConfig, str]:
    with open(os.environ["TRAIN_CONFIG"], "rb") as f:
        return parse_config(f.read(), os.environ.get("CONFIG_SHA256"))


def total_steps(steps_per_epoch: int, epochs: int, max_steps: int | None) -> int:
    total = steps_per_epoch * epochs
    return min(total, max_steps) if max_steps is not None else total


def checkpoint_every(total: int, step_seconds: float, interval_seconds: float) -> tuple[int, int]:
    """Young/Daly τ(초)를 step 간격으로 바꾼다. (step 간격 k, 예상 저장 횟수 n)"""
    expected = total * step_seconds
    n = max(1, math.floor(expected / interval_seconds)) if interval_seconds > 0 else 1
    k = max(1, math.ceil(total / n))
    return k, math.ceil(total / k)


def lr_at(step: int, base: float, warmup: int, total: int) -> float:
    # warmup 후 cosine으로 lr의 10%까지 줄인다.
    if warmup > 0 and step <= warmup:
        return base * step / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def split_blocks(ids: list[int], size: int) -> list[list[int]]:
    n = len(ids) // size
    if n == 0:
        raise SystemExit(f"dataset too small: {len(ids)} tokens, need at least {size}")
    return [ids[i * size : (i + 1) * size] for i in range(n)]


def data_files(data_dir: str) -> list[str]:
    # 정렬: rank마다 순서가 다르면 DistributedSampler가 나눌 때 샘플이 겹치거나 빠진다
    paths = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not paths:
        raise SystemExit(f"no .jsonl files in {data_dir}")
    return paths


def block_size(model_config, tokenizer) -> int:
    limits = [MAX_BLOCK_SIZE]
    for key in ("max_position_embeddings", "n_positions", "max_seq_len"):
        v = getattr(model_config, key, None)
        if isinstance(v, int) and v > 0:
            limits.append(v)
    if isinstance(tokenizer.model_max_length, int) and 0 < tokenizer.model_max_length < 1_000_000:
        limits.append(tokenizer.model_max_length)
    return min(limits)


def step_name(step: int) -> str:
    return f"step-{step:08d}"


class ProgressFile:
    """현재 상태를 PROGRESS_FILE에 덮어쓴다. 경로가 없으면(rank 0 외) 아무것도 하지 않는다.

    임시 파일 후 rename으로 바꿔 사이드카가 반쯤 쓴 파일을 읽지 않게 한다.
    """

    def __init__(self, path: str, generation: int):
        self.path = path
        self.state = {"generation": generation}

    def write(self, **values):
        if not self.path:
            return
        self.state.update(values)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.path)


def read_tokens(paths, tokenizer):
    ids = []
    eos = tokenizer.eos_token_id
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    text = json.loads(line)["text"]
                except (ValueError, KeyError) as e:
                    raise SystemExit(f"{os.path.basename(path)}:{n}: each line must be {{\"text\": ...}} ({e})")
                ids.extend(tokenizer(text)["input_ids"])
                if eos is not None:
                    ids.append(eos)
    return ids


def save_fsdp(step, model, optimizer, cfg_hash, step_dir):
    # 모든 rank가 자기 조각을 쓴다. 모델 상태가 GPU마다 다르기 때문이다.
    import torch.distributed.checkpoint as dcp
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state = {
            "model": model.state_dict(),
            "optim": FSDP.optim_state_dict(model, optimizer),
            "step": step,
            "config_sha256": cfg_hash,
        }
    dcp.save(state, checkpoint_id=step_dir)


def save_ddp(step, model, optimizer, cfg_hash, step_dir, rank):
    # 모든 rank의 상태가 같으므로 rank 0만 파일 하나를 쓴다.
    import torch

    if rank != 0:
        return
    state = {
        "model": model.module.state_dict(),
        "optim": optimizer.state_dict(),
        "step": step,
        "config_sha256": cfg_hash,
    }
    torch.save(state, os.path.join(step_dir, "checkpoint.pt"))


def save_checkpoint(mode, step, model, optimizer, cfg_hash, rank, ckpt_dir) -> float:
    """CKPT_DIR/step-N/에 쓰고 .done을 마지막에 남긴다. 걸린 초를 돌려준다."""
    import torch.distributed as dist

    start = time.perf_counter()
    step_dir = os.path.join(ckpt_dir, step_name(step))
    os.makedirs(step_dir, exist_ok=True)
    if mode == "fsdp":
        save_fsdp(step, model, optimizer, cfg_hash, step_dir)
    else:
        save_ddp(step, model, optimizer, cfg_hash, step_dir, rank)
    dist.barrier()  # 모든 조각이 다 쓰인 뒤에만 .done
    with open(step_dir + ".done", "wb"):
        pass
    return time.perf_counter() - start


def stop_requested(ckpt_dir: str) -> bool:
    return os.path.exists(os.path.join(ckpt_dir, "STOP"))


def wait_approved(ckpt_dir: str, step: int, timeout: float) -> bool:
    marker = os.path.join(ckpt_dir, step_name(step) + ".approved")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(marker):
            return True
        if stop_requested(ckpt_dir):
            return False
        time.sleep(2)
    return os.path.exists(marker)


def wrap_model(mode, net, device):
    from torch.nn.parallel import DistributedDataParallel as DDP

    if mode == "ddp":
        return DDP(net.to(device), device_ids=[device.index])

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    # 어떤 HF 모델이든 트랜스포머 블록(_no_split_modules) 단위로 나눈다. 없으면 모델 전체 하나.
    names = set(getattr(net, "_no_split_modules", None) or [])
    blocks = {type(m) for m in net.modules() if type(m).__name__ in names}
    policy = ModuleWrapPolicy(blocks) if blocks else None
    # use_orig_params: optimizer와 체크포인트가 원래 파라미터 이름을 그대로 쓴다.
    return FSDP(net, auto_wrap_policy=policy, device_id=device, use_orig_params=True)


def clip_gradients(mode, model, max_norm):
    import torch

    if mode == "ddp":
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    else:
        model.clip_grad_norm_(max_norm)  # FSDP는 조각을 모아 전체 노름을 계산한다


def main():
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg, cfg_hash = load_config()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    ckpt_dir = os.environ["CKPT_DIR"]
    data_dir = os.environ["DATA_DIR"]
    progress = ProgressFile(os.environ.get("PROGRESS_FILE", "") if rank == 0 else "", int(os.environ["GENERATION"]))
    try:
        if world != cfg.world_size:
            raise SystemExit(f"WORLD_SIZE {world} != config.world_size {cfg.world_size}")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.manual_seed(cfg.seed)
        print(f"rank {rank}/{world} on {socket.gethostname()}: {cfg.training_mode} {cfg.model}", flush=True)
        progress.write(status="RUNNING", step=0, total_steps=0, last_checkpoint_step=0)

        model_dir = os.path.join(data_dir, "model")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        net = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
        net.config.use_cache = False
        size = block_size(net.config, tokenizer)
        blocks = split_blocks(read_tokens(data_files(os.path.join(data_dir, "data")), tokenizer), size)
        dataset = TensorDataset(torch.tensor(blocks, dtype=torch.long))
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=cfg.seed, drop_last=True)
        loader = DataLoader(dataset, batch_size=cfg.batch_size_per_gpu, sampler=sampler, drop_last=True)
        steps_per_epoch = len(loader)
        if steps_per_epoch == 0:
            raise SystemExit(f"{len(blocks)} blocks of {size} tokens is less than one batch per GPU")
        total = total_steps(steps_per_epoch, cfg.epochs, cfg.max_steps)
        progress.write(total_steps=total)
        if rank == 0:
            print(f"{len(blocks)} blocks x {size} tokens, {steps_per_epoch} steps/epoch, {total} steps", flush=True)

        model = wrap_model(cfg.training_mode, net, device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.1)

        policy = cfg.checkpoint_policy
        measure = max(1, min(policy.warmup_measure_steps, total))
        every = None  # 측정이 끝나기 전에는 저장하지 않는다 (마지막 step 제외)
        measure_start = None
        step = 0
        epoch = 0
        last_saved = 0
        while step < total:
            sampler.set_epoch(epoch)
            for (batch,) in loader:
                if stop_requested(ckpt_dir):
                    print(f"rank {rank}: STOP at step {step}", flush=True)
                    sys.exit(STOP_EXIT)
                if step == 1:  # 첫 step(CUDA 초기화 · 컴파일)은 재지 않는다
                    torch.cuda.synchronize()
                    measure_start = time.perf_counter()
                step += 1
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step, cfg.learning_rate, cfg.warmup_steps, total)
                x = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = model(input_ids=x, labels=x).loss  # labels를 한 칸 밀어 다음 토큰을 맞히는 건 모델이 한다
                loss.backward()
                clip_gradients(cfg.training_mode, model, 1.0)
                optimizer.step()

                reduced = loss.detach().clone()  # 화면에는 전체 rank 평균을 보여 준다
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                values = {"step": step, "loss": float(reduced.item()) / world}

                if every is None and step >= measure and measure_start is not None:
                    torch.cuda.synchronize()
                    elapsed = torch.tensor([(time.perf_counter() - measure_start) / max(1, step - 1)], device=device)
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)  # 모든 rank가 같은 간격을 쓴다
                    step_seconds = float(elapsed.item())
                    every, count = checkpoint_every(total, step_seconds, policy.interval_seconds)
                    values.update(checkpoint_every_steps=every, checkpoint_count=count, step_seconds=round(step_seconds, 4))
                    if rank == 0:
                        print(f"step time {step_seconds:.3f}s, expected {total * step_seconds / 60:.1f} min, "
                              f"interval {policy.interval_seconds:.0f}s -> save every {every} steps ({count} saves)",
                              flush=True)

                if step == total or (every is not None and step % every == 0):  # 마지막 step은 항상 저장
                    took = save_checkpoint(cfg.training_mode, step, model, optimizer, cfg_hash, rank, ckpt_dir)
                    last_saved = step
                    values["last_checkpoint_step"] = step
                    if rank == 0:
                        print(f"step {step}: saved {step_name(step)} in {took:.1f}s (loss {values['loss']:.4f})", flush=True)
                progress.write(**values)
                if step >= total:
                    break
            epoch += 1

        # 마지막 저장이 승인돼야 재개 가능한 상태로 끝난다 (JobSet 템플릿 7절: 최대 120초)
        if not wait_approved(ckpt_dir, last_saved, APPROVAL_WAIT_SECONDS):
            if stop_requested(ckpt_dir):
                sys.exit(STOP_EXIT)
            raise SystemExit(f"{step_name(last_saved)} not approved within {APPROVAL_WAIT_SECONDS}s")
        dist.barrier()
        progress.write(status="COMPLETED")
    except SystemExit as e:
        if e.code != STOP_EXIT:
            progress.write(status="FAILED")
        raise
    except BaseException:
        progress.write(status="FAILED")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
