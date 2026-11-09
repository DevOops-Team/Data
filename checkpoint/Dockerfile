# ckpt-init · ckpt-sidecar 공용 이미지 (체크포인트 규약 v14)
#   init 컨테이너:   command: ["python", "-m", "ckpt.init_main"]
#   사이드카 컨테이너: command: ["python", "-m", "ckpt.sidecar"]
# 학습 Pod 안에서 trainer와 /ckpt · /code · /data 볼륨을 공유한다.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ckpt/ ckpt/

# trainer와 같은 UID로 맞춰야 /ckpt 파일을 서로 읽고 쓸 수 있다 (JobSet 템플릿 securityContext와 맞춤)
RUN useradd --uid 1000 --create-home ckpt
USER 1000

# 기본값은 사이드카. init 컨테이너는 command로 덮어쓴다
CMD ["python", "-m", "ckpt.sidecar"]
