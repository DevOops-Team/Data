# checkpoint — ckpt-init · ckpt-sidecar

기준: Confluence 「체크포인트 규약」 v11 · 「체크포인트 규약 — 흐름으로 읽기」. Jira GPU-64.

| 파일 | 역할 |
|---|---|
| `ckpt/config.py` | 환경 변수 → 설정 (규약 12절) |
| `ckpt/layout.py` | 로컬 · S3 경로 규칙 (`step-00001200`, `gen-0002`) |
| `ckpt/fencing.py` | DynamoDB 펜싱 테이블 조건부 갱신: 첫 항목 · owner 선점 · 승인 · 하트비트 · (복구 쪽) 세대 +1 |
| `ckpt/storage.py` | S3 업로드 · 다운로드 · sha256 검증 · manifest |
| `ckpt/init_main.py` | ckpt-init (init 컨테이너) |
| `ckpt/sidecar.py` | ckpt-sidecar (사이드카 컨테이너) |

## 실행

```bash
python -m ckpt.init_main          # 종료 코드는 아래 표
python -m ckpt.sidecar            # 계속 실행 (--once: 한 사이클만)
```

| ckpt-init 종료 코드 | 뜻 | 복구 쪽 판단 |
|---|---|---|
| 0 | 정상 | — |
| 2 | 세대 불일치 | 옛 세대. 띄우면 안 됨 |
| 3 | owner 선점 실패 | 같은 세대가 다른 클러스터에서 실행 중 |
| 4 | 연결 오류로 실패 (재시도 · 제한 시간 초과) | 다시 띄우면 될 수 있음 |
| 5 | 승인본이 지금 작업과 다름 (config · world_size) | 설정 문제 |
| 6 | DynamoDB 항목 없음 | 제출 과정 문제 |
| 7 | 승인본 손상 (체크섬 · 크기 불일치, 파일 없음) | 같은 승인본으로는 안 됨 → prev_approved 등 판단 |

S3 · DynamoDB 주소는 boto3 표준 `AWS_ENDPOINT_URL`로 바꾼다(로컬만). EKS에선 비워 둔다.

재시도 (규약 10절 · 한송이 「장애 복구 규약」 4절과 맞춤): `RETRY_ATTEMPTS`(3) · `RETRY_BACKOFF`(5,10,20초) · `DOWNLOAD_DEADLINE_S`(600, ckpt-init 전체). 연결 오류만 재시도, 손상 · 조건 거부는 재시도 안 함. 사이드카는 연결 오류로 STOP하거나 죽지 않고 다음 사이클에 다시 함. 업로드가 재시도까지 실패하면 그 step만 미승인.

사이드카 추가 환경 변수: `RANK_WAIT_SECONDS`(300) · `KEEP_LOCAL`(2) · `HEARTBEAT_SECONDS`(10) · `POLL_SECONDS`(2) · `IMAGE_DIGEST`

### 승인 사건 발행 (Kafka)

rank 0 사이드카가 승인(OK · ALREADY)하면 `job.lifecycle` 토픽에 `checkpoint.approved`를 보낸다 → 수집기가 `data.checkpoints`에 저장.

- 환경 변수: `KAFKA_BOOTSTRAP`(없으면 발행 안 함) · `KAFKA_TOPIC_LIFECYCLE`(job.lifecycle) · `NODE_NAME`(Downward API `spec.nodeName`)
- `event_id` = `ckpt-{job}-g{세대 4자리}-s{step 8자리}` → 같은 step을 다시 보내도 DB에는 한 번만
- 메시지 키 = `job_id` (같은 job 사건은 같은 파티션 → 순서 유지)
- payload: `step` `format` `world_size` `manifest_key` `file_count` `size_bytes` `config_sha256`(필수, 수집기 기준) + `image_digest` `upload_ms`(rank 0 업로드) `saved_at`(.done 시각) `approved_at`
- 최선 노력: Kafka 장애 · 전달 실패가 있어도 체크포인트 저장 · 승인은 그대로. 프로듀서는 `acks=all` · 멱등 · **30분**까지 재시도(`KAFKA_MESSAGE_TIMEOUT_MS`, 장애 복구 규약의 재개 대기 30분과 맞춤)

### 진행률 사건 발행 (Kafka `job.progress`, GPU-141)

rank 0 사이드카가 trainer가 쓰는 `PROGRESS_FILE`(`/progress/progress.json`)을 읽어 바뀌었을 때 `job.lifecycle`에 `job.progress`를 보낸다 → 수집기가 `data.job_progress`(작업당 1행)에 저장 → Backend가 `data.v_job_status`로 화면에 표시. 진행률 DynamoDB 테이블 · progress-reporter 사이드카를 대신한다 (`ckpt/progress.py`).

- 환경 변수: `PROGRESS_FILE`(없으면 끔) · `PROGRESS_INTERVAL_SECONDS`(30)
- 최소 30초 간격, 상태(`RUNNING` → `COMPLETED` · `FAILED`)가 바뀌면 바로, 사이드카 종료(SIGTERM) 때 남은 값 한 번 더
- `event_id` = `progress-{job}-g{세대 4자리}-s{step 8자리}`, `RUNNING`이 아니면 끝에 `-{status 소문자}`
- payload(파일에 있는 것만): `status` `step` `total_steps` `loss` `last_checkpoint_step` `checkpoint_every_steps` `checkpoint_count` `step_seconds`. `occurred_at` = 파일 갱신 시각
- 다른 세대가 쓴 파일 · 쓰는 중인 파일은 무시. 발행 실패는 로그만 (학습 · 승인 영향 없음)

### S3 대기열 (outbox) — VPN이 끊겨도 사건을 잃지 않게

EKS → 온프레 Kafka는 사이트 간 VPN을 지난다. 오래 끊기거나 그 사이 Pod가 교체되면 프로듀서 메모리에만 있던 사건이 사라진다. 그래서 Kafka로 못 보낸 사건은 S3에 보관했다가 다시 보낸다 (Confluence 「VPN 끊김 시 Kafka 사건 유실 문제」 제안 C).

| 언제 대기열로 가나 | 어떻게 |
|---|---|
| 전달 최종 실패 (30분 재시도 초과) | 실패 콜백에서 저장 |
| 보내기 즉시 오류 (버퍼 가득 등) | 예외 처리에서 저장 |
| 종료(SIGTERM) 때 못 보낸 사건 | `flush` 5초 안에 못 보내면 `purge` → 실패 콜백 → 저장 |

- 위치: `s3://{CKPT_BUCKET}/events-outbox/ckpt-sidecar/{event_id}.json` (`OUTBOX_PREFIX`, 빈 값이면 대기열 끔). 내용 = 토픽 · 키 · 사건 JSON · 저장 시각 · 이유
- 같은 사건은 같은 키라 여러 번 저장돼도 1개. 수집기가 `event_id`로 중복을 버려서 다시 보내도 안전
- 대기열 저장이 실패해도 예외 없이 로그(`outbox.store_failed`)만 남김
- **다시 보내기** (온프레에서 실행): `python -m ckpt.outbox` — `REPLAY_INTERVAL_S`(60)마다 대기열을 읽어 Kafka로 보내고 전달 확인(acks=all)된 것만 지움. `--once`는 한 번만. 깨진 객체는 건너뛰고 남겨 둠(사람이 확인)
- 필요 권한 (박기현): 학습 Pod 역할 `s3:PutObject` on `events-outbox/*` · 다시 보내기(온프레 인증 경로) `s3:ListBucket`(prefix `events-outbox/`) · `GetObject` · `DeleteObject`
- 남는 한계: 강제 종료(SIGKILL) · 노드 전원 차단 순간 메모리에만 있던 사건은 여전히 잃을 수 있음

## trainer와의 약속

- 저장: `CKPT_DIR/step-XXXXXXXX/`에 다 쓴 뒤 마지막에 `step-XXXXXXXX.done`
- step마다 `CKPT_DIR/STOP` 확인 → 있으면 저장 없이 종료 코드 75
- 재개: ckpt-init이 `CKPT_DIR/resume.json`(`approved_step` · `start_step` · `step_dir`)을 남긴다 **[노찬경과 형식 합의 필요]**

## 테스트 (AWS 없이, moto 서버)

```bash
python -m venv .venv && .venv/bin/pip install -r checkpoint/requirements-dev.txt
cd checkpoint && ../.venv/bin/python -m pytest -q
```

## 실제 DCP 흐름 확인 (`examples/`)

```bash
.venv/bin/pip install -r checkpoint/requirements-dev.txt
cd checkpoint && ../.venv/bin/python examples/dcp_flow.py     # moto, AWS 불필요
```

- `tiny_trainer.py`: FSDP2(CPU · gloo) 작은 trainer. 위 약속 3가지를 지키는 예시
- `dcp_flow.py`: A 끊김 없는 학습 체크섬 == B step 25 강제 종료 → 승인본 20에서 재개한 학습 체크섬, C 학습 도중 세대 +1 → STOP → 종료
- **torchrun은 worker의 종료 코드 75를 1로 바꿔 돌려준다.** 컨테이너가 75로 끝나야 하면, torchrun이 끝난 뒤 `$CKPT_DIR/STOP`이 있으면 `exit 75` 하도록 진입 스크립트에서 처리 **[노찬경과 합의 필요]**

## 남은 것

- 노찬경 train.py에 약속 3가지 적용 확인 (예시: `examples/tiny_trainer.py`)
- 실제 Kafka(3대)로 발행 → 수집기 저장까지 확인
- 저장 · 업로드 시간 측정 → 저장 주기 결정
- 학습 데이터 manifest 형식 (백엔드와 합의)
- 컨테이너 이미지 · JobSet 템플릿 연결
