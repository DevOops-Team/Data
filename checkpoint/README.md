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
python -m ckpt.init_main          # 종료 코드 0 정상 · 2 세대 불일치 · 3 owner 선점 실패 · 4 다운로드/체크섬 실패 · 5 승인본 불일치 · 6 항목 없음
python -m ckpt.sidecar            # 계속 실행 (--once: 한 사이클만)
```

S3 · DynamoDB 주소는 boto3 표준 `AWS_ENDPOINT_URL`로 바꾼다(로컬만). EKS에선 비워 둔다.

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

## 남은 것

- 실제 DCP(`dcp.save` / `dcp.load`) trainer와 연결 확인 (노찬경 train.py)
- 승인 사건 Kafka `job.lifecycle` 발행
- 저장 · 업로드 시간 측정 → 저장 주기 결정
- 학습 데이터 manifest 형식 (백엔드와 합의)
- 컨테이너 이미지 · JobSet 템플릿 연결
