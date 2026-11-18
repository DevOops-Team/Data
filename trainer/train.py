"""Hugging Face causal LM FSDP · DDP 학습 — 기본 학습 코드이자 사용자 train.py 예시.

사용자가 제출 때 올리면 ckpt-init이 DATA_DIR/code/train.py로 받아 두고, 백엔드가 만든 JobSet의 torchrun이
trainer 이미지(학습 환경) 위에서 Pod마다 실행한다.
  - 이 파일은 학습만 한다: 데이터 읽기 · 모델 · 옵티마이저 · 학습 루프.
  - 시스템과 맞물리는 일(재개 · 체크포인트 저장 규약 · 진행률 · 지표 · 펜싱 STOP · 마지막 승인 대기 · 실패 기록)은
    trainer 이미지의 라이브러리 devoops_ckpt가 한다 (GPU-209). 사용자 코드가 부르는 것은
    Session.load · data_position · report · should_save · save · finish 뿐이다.
  - 설정은 config.json 하나 (ckpt.load_config: CONFIG_SHA256 · 키 · 학습 방식 확인).
  - 업로드: 모델 폴더 DATA_DIR/model, 학습 데이터 DATA_DIR/data/*.jsonl ({"text": ...}).
  - 어떤 HF 모델이든 AutoModelForCausalLM으로 읽는다. FSDP는 모델의 _no_split_modules(트랜스포머 블록) 단위로 감싼다.
"""

from __future__ import annotations

import glob
import json
import math
import os
import socket
import time

import devoops_ckpt as ckpt

MAX_BLOCK_SIZE = 1024  # 긴 문맥 모델도 g4dn(16GB)에 맞게 자른다


def total_steps(steps_per_epoch: int, epochs: int, max_steps: int | None) -> int:
    total = steps_per_epoch * epochs
    return min(total, max_steps) if max_steps is not None else total


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

    cfg, cfg_hash = ckpt.load_config()                  # /code/config.json (Backend가 만든 학습 설정)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    data_dir = os.environ["DATA_DIR"]
    try:
        # total을 알기 전에 실패해도 FAILED가 남도록 먼저 0으로 연다 (데이터를 읽은 뒤 바로 고친다)
        with ckpt.Session(mode=cfg.training_mode, total_steps=0, cfg_hash=cfg_hash, policy=cfg.checkpoint_policy,
                          warmup_steps=cfg.warmup_steps) as s:
            if world != cfg.world_size:
                raise SystemExit(f"WORLD_SIZE {world} != config.world_size {cfg.world_size}")
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            torch.manual_seed(cfg.seed)
            print(f"rank {rank}/{world} on {socket.gethostname()}: {cfg.training_mode} {cfg.model}", flush=True)

            # ---- 데이터 · 모델 (학습 코드) ----
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
            s.total = total
            s.progress.write(total_steps=total)
            if rank == 0:
                print(f"{len(blocks)} blocks x {size} tokens, {steps_per_epoch} steps/epoch, {total} steps", flush=True)

            model = wrap_model(cfg.training_mode, net, device)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.1)

            # ---- 재개: 승인본이 있으면 그 step까지 불러오고, 데이터도 그 위치부터 ----
            start = s.load(model, optimizer, device)
            epoch, skip = s.data_position(start, steps_per_epoch)

            # ---- 학습 ----
            step = start
            mark = time.perf_counter()
            while step < total:
                sampler.set_epoch(epoch)                        # 같은 seed · epoch → 같은 순서 (재개해도 이어짐)
                for (batch,) in loader:
                    if skip:                                    # 재개한 epoch에서 이미 학습한 배치
                        skip -= 1
                        mark = time.perf_counter()
                        continue
                    step += 1
                    for group in optimizer.param_groups:
                        group["lr"] = lr_at(step, cfg.learning_rate, cfg.warmup_steps, total)
                    x = batch.to(device)
                    data_wait = time.perf_counter() - mark
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(input_ids=x, labels=x).loss    # labels를 한 칸 밀어 다음 토큰을 맞히는 건 모델이 한다
                    loss.backward()
                    clip_gradients(cfg.training_mode, model, 1.0)
                    optimizer.step()

                    s.report(step, loss, data_wait)             # 진행률 · 지표 · 저장 간격 · 펜싱 STOP
                    if s.should_save(step):                     # 마지막 step은 항상
                        s.save(model, optimizer, step)
                    mark = time.perf_counter()
                    if step >= total:
                        break
                epoch += 1

            s.finish()                                          # 마지막 저장이 승인될 때까지 (최대 120초)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
