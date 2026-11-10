# collector — Kafka 사건 → 데이터 DB

Kafka 토픽의 사건을 읽어 PostgreSQL `gpuf_data`의 `data` 스키마(`db/001_init.sql`)에 저장한다.

## 동작

| 단계 | 내용 |
|---|---|
| 읽기 | 소비자 그룹 `data-collector`, 자동 커밋 끔, 최대 `BATCH_SIZE`개씩 |
| 저장 | 모든 사건 → `events` (같은 `event_id`는 무시) |
| 거부 | 형식 오류(JSON 아님 · 필수 필드 없음 · 시간대 없는 시각) → `event_rejects`에 원본 바이트 보관 |
| 파생 | `experiment.*` → `experiments`, `checkpoint.approved` → `checkpoints`, `job.*` → `job_runs` |
| 커밋 | DB 커밋 **뒤에** Kafka 오프셋 커밋 (최소 한 번 전달 + `event_id` 중복 제거 = 정확히 한 번 저장) |

- 메시지마다 savepoint: 한 메시지가 DB 오류(값 범위 등)를 내면 그것만 `event_rejects`로 보내고 나머지는 계속 저장한다.
- DB 연결 오류: 커밋하지 않고 읽은 위치를 되감은 뒤 `RETRY_SLEEP_S` 기다렸다가 같은 묶음을 다시 처리한다.
- 파생에 필요한 필드가 없으면 파생만 건너뛴다 (원본은 `events`에 남음, 로그 `collector.derive_skipped`).
- 복구로 생긴 세대(`job.submitted` 없이 `job.resumed`만 옴)는 같은 작업의 이전 세대에서 `exp_id` · `gpus` · `strategy` · `config_sha256` · `image_digest`를 이어받는다. `queue` · `cluster`는 복구 때 바뀌므로 이어받지 않는다.
- 실험 사건이 작업보다 늦게 와도 그 작업의 모든 세대에 `exp_id`를 붙인다.

### 파생에 필요한 payload 필드

| 사건 | 필수 | 선택 |
|---|---|---|
| `experiment.started` | `exp_id`, `scenario`, `policy`, `spare_enabled` | `name`, `fault_type`, `checkpoint_every_steps`, `conditions`, `note` |
| `experiment.ended` | `exp_id` | |
| `checkpoint.approved` | 봉투 `job_id`·`generation` + `step`, `format`, `world_size`, `manifest_key`, `file_count`, `size_bytes`, `config_sha256` | `image_digest`, `save_ms`, `upload_ms`, `saved_at`, `approved_at` |
| `job.submitted` | 봉투 `job_id`·`generation` | `jobset_name`, `gpus`, `strategy`, `queue`, `exp_id`, `config_sha256`, `image_digest` |
| `job.started` · `job.resumed` · `job.completed` · `job.failed` · `job.fenced` · `job.stopped` · `job.cancelled` | 봉투 `job_id`·`generation` | |

## 환경 변수

| 변수 | 기본값 |
|---|---|
| `KAFKA_BOOTSTRAP` | (필수) 브로커 목록 `host:port,...` |
| `KAFKA_GROUP` | `data-collector` |
| `KAFKA_TOPICS` | `gpu.fault,recovery.action,job.lifecycle,fault.injection,monitoring.log` |
| `PGHOST` `PGPORT` `PGDATABASE` `PGUSER` `PGPASSWORD` `PGSSLMODE` | libpq 표준. `PGUSER=data_collector` (비밀번호는 Secret으로 주입) |

### DB 계정 (`db/002_roles.sql`)

| 계정 | 권한 | 용도 |
|---|---|---|
| `data_collector` | `events` 읽기·추가, `event_rejects` 추가, `experiments`·`job_runs` 읽기·추가·수정, `checkpoints` 읽기·추가. 삭제·DDL·`results` 없음. `statement_timeout` 30s | 수집기 |
| `postgres_exporter` | `pg_monitor` (통계 읽기만), 데이터 테이블 접근 없음, 연결 3개까지 | 모니터링 (ADR-18) |

슈퍼유저로 실행한 뒤 `\password data_collector`, `\password postgres_exporter`로 비밀번호를 지정하고, `pg_hba.conf`에 필요한 주소만 허용한다.
| `BATCH_SIZE` · `POLL_TIMEOUT_S` · `RETRY_SLEEP_S` | `500` · `1.0` · `5` |

## 실행

```bash
docker build -t devoops/collector collector
docker run --rm -e KAFKA_BOOTSTRAP=... -e PGHOST=... -e PGDATABASE=gpuf_data -e PGUSER=... -e PGPASSWORD=... devoops/collector
```

## 테스트

임시 PostgreSQL 16을 띄우고 실제 `db/001_init.sql`을 적용한 뒤, 가짜 Kafka 메시지로 검증한다 (Kafka 불필요).

```bash
cd collector
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
PG_BIN=/opt/homebrew/opt/postgresql@16/bin .venv/bin/python -m pytest -q
```

확인 항목: 중복 무시 · 형식 오류 보관 · experiments/job_runs/checkpoints 파생 · 실험보다 먼저 온 job 연결 · 세대 이어받기 · 불량 메시지 격리 · DB 끊김 시 되감기·미커밋 · `v_incidents` 계산.

### 실제 경로 시험 (Kafka → 수집기 → DB)

`scripts/e2e_events.py`가 작업 1개의 흐름(실험 시작 → 제출 → 시작 → 체크포인트 승인 → 장애 주입 → 펜싱 → 세대 2 재개 → 실험 종료)과 중복 1건 · 형식 오류 1건을 `키|JSON` 줄로 만든다. 표준 라이브러리만 써서 Kafka VM에서 바로 실행된다. `--prefix`를 시험마다 바꾸면 사건 ID가 겹치지 않는다. `gpu.fault` · `recovery.action`은 복구 컨트롤러가 읽을 수 있어 쓰지 않는다.

```bash
B=172.30.0.40:9092,172.30.0.41:9092,172.30.0.42:9092
python3 e2e_events.py --prefix t1 --topic job.lifecycle | /opt/kafka/bin/kafka-console-producer.sh \
  --bootstrap-server $B --topic job.lifecycle --producer-property acks=all --property parse.key=true --property key.separator='|'
```
