"""Hugging Face causal LM FSDP · DDP 학습 — 기본 학습 코드이자 사용자 train.py 예시.

사용자가 제출 때 올리면 ckpt-init이 DATA_DIR/code/train.py로 받아 두고, 백엔드가 만든 JobSet의 torchrun이
trainer 이미지(학습 환경) 위에서 Pod마다 실행한다.
  - 이 파일은 모델 · 데이터 · loss만 정한다.
  - 학습 루프와 시스템 일(분산 감싸기 · 재개 · 체크포인트 저장 · 진행률 · 지표 · 펜싱 STOP · 마지막 승인 대기 ·
    실패 기록)은 trainer 이미지의 라이브러리 devoops_ckpt.fit()이 한다 (GPU-209).
  - 배치 크기 · epoch · 학습률 · seed · 학습 방식은 config.json (Backend가 제출 화면 값으로 만든다).
  - 업로드: 모델 폴더 DATA_DIR/model, 학습 데이터 DATA_DIR/data/*.jsonl ({"text": ...}).
"""

from __future__ import annotations

import glob
import json
import os

import devoops_ckpt as ckpt

MAX_BLOCK_SIZE = 1024  # 긴 문맥 모델도 g4dn(16GB)에 맞게 자른다
DATA_DIR = os.environ.get("DATA_DIR", "/data")
MODEL_DIR = os.path.join(DATA_DIR, "model")


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


def load_model():
    from transformers import AutoModelForCausalLM

    net = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True)
    net.config.use_cache = False
    return net


def load_dataset():
    # jsonl 텍스트를 이어 붙여 block_size 토큰씩 자른다
    import torch
    from torch.utils.data import TensorDataset
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    size = block_size(AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True), tokenizer)
    blocks = split_blocks(read_tokens(data_files(os.path.join(DATA_DIR, "data")), tokenizer), size)
    return TensorDataset(torch.tensor(blocks, dtype=torch.long))


def loss_fn(model, batch):
    # labels를 한 칸 밀어 다음 토큰을 맞히는 건 모델이 한다
    return model(input_ids=batch, labels=batch).loss


if __name__ == "__main__":
    ckpt.fit(model=load_model, dataset=load_dataset, loss_fn=loss_fn)
