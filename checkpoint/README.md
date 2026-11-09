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

## trainer와의 약속

- 저장: `CKPT_DIR/step-XXXXXXXX/`에 다 쓴 뒤 마지막에 `step-XXXXXXXX.done`
- step마다 `CKPT_DIR/STOP` 확인 → 있으면 저장 없이 종료 코드 75
- 재개: ckpt-init이 `CKPT_DIR/resume.json`(`approved_step` · `start_step` · `step_dir`)을 남긴다 **[노찬경과 형식 합의 필요]**

## 테스트 (AWS 없이, moto 서버)

```bash
python -m venv .venv && .venv/bin/pip install -r checkpoint/requirements.txt
cd checkpoint && ../.venv/bin/python -m pytest -q
```

## 실제 DCP 흐름 확인 (`examples/`)

```bash
.venv/bin/pip install "torch>=2.6" numpy
cd checkpoint && ../.venv/bin/python examples/dcp_flow.py     # moto, AWS 불필요
```

- `tiny_trainer.py`: FSDP2(CPU · gloo) 작은 trainer. 위 약속 3가지를 지키는 예시
- `dcp_flow.py`: A 끊김 없는 학습 체크섬 == B step 25 강제 종료 → 승인본 20에서 재개한 학습 체크섬, C 학습 도중 세대 +1 → STOP → 종료
- **torchrun은 worker의 종료 코드 75를 1로 바꿔 돌려준다.** 컨테이너가 75로 끝나야 하면, torchrun이 끝난 뒤 `$CKPT_DIR/STOP`이 있으면 `exit 75` 하도록 진입 스크립트에서 처리 **[노찬경과 합의 필요]**

## 남은 것

- 노찬경 train.py에 약속 3가지 적용 확인 (예시: `examples/tiny_trainer.py`)
- 승인 사건 Kafka `job.lifecycle` 발행
- 저장 · 업로드 시간 측정 → 저장 주기 결정
- 학습 데이터 manifest 형식 (백엔드와 합의)
- 컨테이너 이미지 · JobSet 템플릿 연결
