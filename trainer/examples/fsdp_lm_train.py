"""FSDP causal LM 학습 — 장애 복구 실험 묶음 공용 train.py (GPU-160, 개인 문서 25591809 F1 · F2 · Q2).

이 파일 하나로 Full fine-tuning(F1 0.5B · F2 1.5B)과 QLoRA(Q2 14B)를 모두 돌린다. 학습 방식(full · qlora)과
LoRA 설정은 제출 화면에서 고른 config.json `finetune`을 따른다 (GPU-262). 묶음마다 다른 것은 아래 「묶음 설정」
상수(FSDP 방식 · 기울기 누적)뿐이고, 묶음(`datasets/<id>/v<n>/code/train.py`)에 올릴 때 그 값만 바꾼다.

trainer 이미지의 하네스 devoops_ckpt(GPU-209)의 Session 위에서 돈다. 이 파일은 모델 · 데이터 · FSDP 설정 ·
학습 한 step만 정하고, 재개 · 저장 규약 · 진행률 · 지표 · 펜싱 STOP · 마지막 승인 대기 · 실패 기록은 Session이 한다.
fit()을 쓰지 않는 이유: fit()은 FSDP 기본값(FULL_SHARD · fp32)으로 감싸고 기울기 누적 · QLoRA가 없다.

config.json(Backend 제출 화면)에서 쓰는 것: training_mode(fsdp만) · finetune(method · lora) · world_size · seed ·
batch_size_per_gpu · epochs · max_steps · learning_rate · warmup_steps · checkpoint_policy.
config.json에 칸이 없는 설정은 아래 상수다 (Backend · 하네스는 모르는 config 키를 거절한다).

1 step = optimizer step 1번 = 마이크로 배치 GRAD_ACCUM개. 진행률 · 저장 간격 · max_steps도 이 단위다.

QLoRA(config finetune.method = "qlora")
  - 기본 가중치: bitsandbytes NF4 + double quant로 읽는다. compute · quant_storage 모두 fp16
    (T4는 bf16이 없고, FSDP는 한 조각 안의 dtype이 같아야 나눌 수 있다)
  - 학습: peft LoRA 어댑터만 (r · alpha · dropout · 대상 층은 config finetune.lora). 어댑터는 fp32로 두고 FSDP MixedPrecision이 fp16으로 계산 · 통신, ShardedGradScaler
  - 감싸기: 어댑터(lora_A · lora_B)를 따로 감싸 얼린 조각과 학습 조각을 나눈다 (peft fsdp_auto_wrap_policy와 같은 방식)
  - 저장 · 재개: Session(model_keys=어댑터 이름)이라 체크포인트에는 어댑터 + 옵티마이저만 들어간다.
    기본 가중치는 재개 때도 /data/model에서 다시 읽는다
  - 필요한 trainer 이미지: bitsandbytes · peft · accelerate가 든 버전 (Data trainer/Dockerfile, GPU-160 이후)
"""

from __future__ import annotations

import functools
import glob
import json
import os
import time
from contextlib import nullcontext

import devoops_ckpt as ckpt

# ---------- 묶음 설정 (개인 문서 5절 "후보별 config") ----------
#                F1 qwen25-0p5b        F2 qwen25-1p5b        Q2 qwen25-14b
# 제출 학습 방식   Full                  Full                  QLoRA
# GRAD_ACCUM     4                     1                     4
# ACCUM_NO_SYNC  True                  False                 False
# 권장 제출       batch 6 · lr 2e-5     batch 8 · lr 1e-5     batch 2 · lr 2e-4
GRAD_ACCUM = 4
ACCUM_NO_SYNC = False
# FSDP 방식은 학습 방식에 따라: Full은 all-gather 1번 적은 SHARD_GRAD_OP, QLoRA는 4bit라 통신이 작아 FULL_SHARD
SHARDING = {"full": "SHARD_GRAD_OP", "qlora": "FULL_SHARD"}
# ----------------------------------------------------------------------
LOSS_CHUNK_TOKENS = 512      # logits(어휘 15만)를 512토큰씩만 만든다
MAX_BLOCK_SIZE = 1024
MAX_GRAD_NORM = 1.0
DATA_DIR = os.environ.get("DATA_DIR", "/data")
MODEL_DIR = os.path.join(DATA_DIR, "model")


def data_files(data_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))  # rank마다 같은 순서
    if not paths:
        raise SystemExit(f"no .jsonl files in {data_dir}")
    return paths


def read_tokens(paths, tokenizer) -> list[int]:
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


def load_dataset(model_dir: str, data_dir: str):
    import torch
    from torch.utils.data import TensorDataset
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    size = min(MAX_BLOCK_SIZE, getattr(config, "max_position_embeddings", MAX_BLOCK_SIZE))
    ids = read_tokens(data_files(data_dir), tokenizer)
    n = len(ids) // size
    if n == 0:
        raise SystemExit(f"dataset too small: {len(ids)} tokens, need at least {size}")
    return TensorDataset(torch.tensor(ids[: n * size], dtype=torch.long).view(n, size))


def chunked_lm(lm):
    """causal LM + 청크 loss를 forward 하나로 묶는 nn.Module (FSDP가 이것을 감싼다).

    lm_head + cross-entropy를 LOSS_CHUNK_TOKENS씩 재계산(checkpoint)해 logits 전체를 메모리에 두지 않는다.
    FSDP 루트 forward 안에서 lm_head를 부르므로 루트 파라미터(임베딩 · lm_head)가 모인 상태다.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    class ChunkedLM(torch.nn.Module):
        def __init__(self, lm):
            super().__init__()
            self.lm = lm

        def forward(self, input_ids):
            hidden = self.lm.model(input_ids=input_ids).last_hidden_state[:, :-1]
            labels = input_ids[:, 1:]
            total = hidden.new_zeros((), dtype=torch.float32)
            for i in range(0, hidden.size(1), LOSS_CHUNK_TOKENS):
                total = total + checkpoint(self._chunk_loss, hidden[:, i : i + LOSS_CHUNK_TOKENS],
                                           labels[:, i : i + LOSS_CHUNK_TOKENS], use_reentrant=False)
            return total / labels.numel()

        def _chunk_loss(self, h, y):
            logits = self.lm.lm_head(h).float()
            return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                                     reduction="sum")

    return ChunkedLM(lm)


def load_full(model_dir: str):
    import torch
    from transformers import AutoModelForCausalLM

    lm = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float32)
    lm.config.use_cache = False
    # activation checkpointing: 통신 병목에선 늘어난 계산(+33%)이 통신 뒤에 숨는다 (개인 문서 1절)
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return chunked_lm(lm)


def load_qlora(model_dir: str, device, lora_cfg):
    """NF4 4bit 기본 가중치 + LoRA 어댑터. 각 rank가 자기 GPU에 4bit 전체를 올린 뒤 FSDP가 나눈다.

    노드 RAM이 16 GiB라 CPU에 전체를 올리지 않는다 (safetensors 조각 단위로 읽어 바로 GPU에서 양자화).
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if device.type != "cuda":
        raise SystemExit("QLoRA needs a GPU (bitsandbytes 4bit)")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_storage=torch.float16)
    lm = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, quantization_config=quant,
                                              torch_dtype=torch.float16, device_map={"": device.index},
                                              low_cpu_mem_usage=True)
    lm.config.use_cache = False
    # prepare_model_for_kbit_training은 쓰지 않는다: 임베딩 · norm을 fp32로 바꿔 FSDP 조각 dtype이 섞이고 메모리가 2배
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=lora_cfg.r, lora_alpha=lora_cfg.alpha, lora_dropout=lora_cfg.dropout,
                      target_modules=list(lora_cfg.target_modules), bias="none", task_type="CAUSAL_LM")
    lm = get_peft_model(lm, lora).get_base_model()   # LoRA가 주입된 원래 모델 (PeftModel 껍데기는 안 씀)
    for p in lm.parameters():
        if p.requires_grad:
            p.data = p.data.float()                  # 어댑터는 fp32 원본 (계산 · 통신은 MixedPrecision fp16)
    return chunked_lm(lm)


def trainable_keys(net) -> set[str]:
    """감싸기 전 학습 파라미터 이름 = FSDP state_dict 키 (FSDP는 감싼 이름 접두사를 지운다)."""
    return {name for name, p in net.named_parameters() if p.requires_grad}


def wrap(net, device, method: str):
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import (ModuleWrapPolicy, _or_policy, lambda_auto_wrap_policy,
                                             transformer_auto_wrap_policy)

    names = set(getattr(net.lm, "_no_split_modules", None) or [])
    blocks = {type(m) for m in net.modules() if type(m).__name__ in names}
    # T4는 bf16이 없다 → fp16 계산 · 통신. 버퍼(RoPE inv_freq)는 fp32 그대로 둔다
    mp = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float32)
    common = dict(device_id=device, mixed_precision=mp, sharding_strategy=ShardingStrategy[SHARDING[method]],
                  limit_all_gathers=True, forward_prefetch=True)
    if method == "full":
        return FSDP(net, auto_wrap_policy=ModuleWrapPolicy(blocks) if blocks else None, use_orig_params=True,
                    **common)

    def lora_leaf(m):  # 자식 없는 모듈 중 학습하는 weight가 있는 것 = lora_A · lora_B
        w = getattr(m, "weight", None)
        return not list(m.children()) and isinstance(w, torch.nn.Parameter) and w.requires_grad

    policy = functools.partial(_or_policy, policies=[
        functools.partial(lambda_auto_wrap_policy, lambda_fn=lora_leaf),
        functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=blocks),
    ])
    # use_orig_params=False: 4bit Params4bit를 FSDP가 평범한 view로 바꾸지 않게 (FSDP-QLoRA 표준 설정)
    return FSDP(net, auto_wrap_policy=policy, use_orig_params=False, **common)


def main():
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    from torch.utils.data import DataLoader, DistributedSampler

    cfg, cfg_hash = ckpt.load_config()
    method = cfg.finetune.method
    if cfg.training_mode != "fsdp":
        raise SystemExit(f"this train.py is FSDP only ({method}), got {cfg.training_mode}")
    if method not in SHARDING:
        raise SystemExit(f"this train.py supports {sorted(SHARDING)}, got {method}")
    gpu = torch.cuda.is_available()
    dist.init_process_group("nccl" if gpu else "gloo")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if gpu:
            device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        with ckpt.Session(mode=cfg.training_mode, total_steps=0, cfg_hash=cfg_hash, policy=cfg.checkpoint_policy,
                          warmup_steps=cfg.warmup_steps) as s:
            if world != cfg.world_size:
                raise SystemExit(f"WORLD_SIZE {world} != config.world_size {cfg.world_size}")
            torch.manual_seed(cfg.seed)                  # 어댑터 초기값(무작위)이 모든 rank에서 같게
            t0 = time.perf_counter()
            net = load_full(MODEL_DIR) if method == "full" else load_qlora(MODEL_DIR, device, cfg.finetune.lora)
            if method == "qlora":                        # 체크포인트 = 어댑터 + 옵티마이저 (4bit 기본 가중치는 안 넣음)
                s.model_keys = frozenset(trainable_keys(net))
            if rank == 0:
                n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
                print(f"{method}: model loaded in {time.perf_counter() - t0:.0f}s, trainable {n_train / 1e6:.1f}M"
                      + (f", max GPU {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB" if gpu else ""),
                      flush=True)

            dataset = load_dataset(MODEL_DIR, os.path.join(DATA_DIR, "data"))
            sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=cfg.seed, drop_last=True)
            loader = DataLoader(dataset, batch_size=cfg.batch_size_per_gpu, sampler=sampler, drop_last=True)
            steps_per_epoch = len(loader) // GRAD_ACCUM
            if steps_per_epoch == 0:
                raise SystemExit(f"{len(dataset)} blocks is less than one step "
                                 f"({cfg.batch_size_per_gpu} x {GRAD_ACCUM} x {world})")
            total = ckpt.total_steps(steps_per_epoch, cfg.epochs, cfg.max_steps)
            s.total = total
            s.progress.write(total_steps=total)
            if rank == 0:
                print(f"fsdp {method} {SHARDING[method]} x {world}: {len(dataset)} blocks, micro batch "
                      f"{cfg.batch_size_per_gpu} x accum {GRAD_ACCUM}, {steps_per_epoch} steps/epoch, {total} steps",
                      flush=True)

            model = wrap(net, device, method)
            del net
            model.train()
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0 if method == "qlora" else 0.1)
            # loss scale은 체크포인트에 없다(Session은 모델 · 옵티마이저만 저장) → 재개 직후 몇 step은 scale을 다시 찾느라 건너뛸 수 있다
            scaler = ShardedGradScaler(enabled=gpu)
            if gpu and rank == 0:
                print(f"wrapped: GPU {torch.cuda.memory_allocated(device) / 2**30:.1f} GiB", flush=True)

            start = s.load(model, opt, device)
            epoch, skip = s.data_position(start, steps_per_epoch)
            skip *= GRAD_ACCUM                                   # 건너뛸 마이크로 배치
            step = start
            mark = time.perf_counter()
            while step < total:
                sampler.set_epoch(epoch)
                micro = []
                for batch in loader:
                    if skip:
                        skip -= 1
                        mark = time.perf_counter()
                        continue
                    micro.append(batch[0])
                    if len(micro) < GRAD_ACCUM:
                        continue
                    step += 1
                    for group in opt.param_groups:
                        group["lr"] = ckpt.lr_at(step, cfg.learning_rate, cfg.warmup_steps, total)
                    data_wait = time.perf_counter() - mark
                    opt.zero_grad(set_to_none=True)
                    loss_sum = torch.zeros((), device=device)
                    for i, x in enumerate(micro):
                        last = i == len(micro) - 1
                        with model.no_sync() if (ACCUM_NO_SYNC and not last) else nullcontext():
                            loss = model(x.to(device, non_blocking=True)) / GRAD_ACCUM
                            scaler.scale(loss).backward()
                        loss_sum += loss.detach()
                    micro = []
                    scaler.unscale_(opt)
                    model.clip_grad_norm_(MAX_GRAD_NORM)
                    scaler.step(opt)
                    scaler.update()

                    s.report(step, loss_sum, data_wait)
                    if gpu and rank == 0 and step == start + 1:
                        print(f"step {step}: max GPU {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB",
                              flush=True)
                    if s.should_save(step):
                        s.save(model, opt, step)
                    mark = time.perf_counter()
                    if step >= total:
                        break
                epoch += 1
            s.finish()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
