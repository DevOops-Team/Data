# (잡) 감시기 — 관리 클러스터 data-system에 Pod 1개 (ADR-29). 권한은 training의 JobSet · Workload 읽기 전용.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY watcher/ watcher/

RUN useradd --uid 1000 --create-home watcher
USER 1000

CMD ["python", "-m", "watcher.main"]
