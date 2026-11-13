# 학습 trainer 이미지 = 학습 환경만. train.py는 사용자가 제출 때 올리고, ckpt-init이 /data/code/train.py로 넣는다.
# 빌드 context는 trainer/
#   docker build -t ghcr.io/devoops-team/trainer trainer/
# torch 2.5.1 · CUDA 12.4 (g4dn T4). transformers는 torch 2.5.1과 맞춘 고정 버전.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN pip install "transformers==4.46.3" "safetensors==0.4.5" "sentencepiece==0.2.0" "tokenizers==0.20.3"

WORKDIR /opt/train

# ckpt-agent 이미지(Data checkpoint/)와 같은 UID: /ckpt를 함께 쓴다. JobSet Pod securityContext도 1000
RUN useradd --uid 1000 --create-home trainer
USER 1000
ENV HOME=/home/trainer
