# job-watcher — (잡) 감시기

관리 클러스터의 JobSet · Workload 상태 변화를 Kafka `job.lifecycle` 사건으로 올린다. 수집기가 이 사건으로 `data.job_runs`(세대별 제출 · 시작 · 종료 시각)를 채운다.

기준: ADR-24(이름), ADR-29(위치: 관리 클러스터 Pod 1개), ADR-15(측정 지표), Jira GPU-118. 설명: Confluence 「(잡) 감시기 설계」 · 「(잡) 감시기 쉽게 보기」.

| 파일 | 내용 |
|---|---|
| `watcher/model.py` | JobSet · Workload dict → 필요한 값(작업 ID · 세대 · 큐 · GPU · 클러스터 · 조건 시각) |
| `watcher/rules.py` | 상태 → 보낼 사건 목록 (순수 함수, 판정 표는 파일 머리말) |
| `watcher/events.py` | 사건 봉투(수집기 형식) · Kafka 발행 |
| `watcher/main.py` | list · watch 루프, 캐시, 중복 방지 |

## 사건

| type | 판정 | occurred_at |
|---|---|---|
| `job.submitted` | JobSet이 보임 | `creationTimestamp` |
| `job.admitted` | Workload `Admitted=True` | 조건 시각 |
| `job.placed` | Workload `status.clusterName` | admissionCheck Ready 시각 |
| `job.started` (세대 1) · `job.resumed` (2 이상) | Workload `PodsReady=True`, 없으면 JobSet 전부 ready | 조건 시각, 없으면 본 시각 |
| `job.completed` | JobSet `Completed=True` | 조건 시각 |
| `job.failed` · `job.fenced` | JobSet `Failed=True` (다음 세대 JobSet이 이미 있으면 fenced) | 조건 시각 |
| `job.stopped` · `job.cancelled` | 종료 조건 없이 삭제 (주석 `devoops/stop-reason=cancel`이면 cancelled) | `deletionTimestamp` |

- `event_id = job-{작업ID}-g{세대 4자리}-{key}` (key: submitted · admitted · placed · started · end). 세대마다 종료는 하나(`-end`)
- `payload.time_source`: `k8s`(쿠버네티스가 찍은 시각) · `observed`(감시기가 본 시각)

## 실행

```bash
python -m watcher.main     # 클러스터 안이면 ServiceAccount, 밖이면 ~/.kube/config
```

| 환경 변수 | 기본값 |
|---|---|
| `WATCH_NAMESPACE` | `training` |
| `KAFKA_BOOTSTRAP` | 없음 → 표준 출력으로만 (dry run) |
| `KAFKA_TOPIC_LIFECYCLE` | `job.lifecycle` |
| `JOBSET_API` | `jobset.x-k8s.io/v1alpha2` |
| `WORKLOAD_API` | `kueue.x-k8s.io/v1beta2` |
| `WATCH_TIMEOUT_S` | `300` |
| `KAFKA_MESSAGE_TIMEOUT_MS` | `1800000` |

권한(ServiceAccount `job-watcher`, 네임스페이스 `data-system`): `training`의 `jobsets.jobset.x-k8s.io` · `workloads.kueue.x-k8s.io`에 `get · list · watch`. 매니페스트는 Gitops(ADR-21, 노찬경).

## 테스트

```bash
pip install -r requirements-dev.txt
python -m pytest -q        # 수집기 envelope.parse로 형식 호환까지 확인
```

## 확인 필요

- ETTR 시작: DB `job_runs.started_at`은 "첫 step 완료"로 정의됐지만 감시기는 Pod 준비 시각까지만 안다 → 첫 step은 학습 코드 · 사이드카가 따로 올릴지 (김현수 · 노찬경)
- `stopped` · `cancelled` 구분: 지우는 쪽이 `devoops/stop-reason` 주석을 붙여야 함 (김성모 · 노찬경)
- `exp_id`: JobSet에 `devoops/exp-id` 주석이 있어야 실험과 연결됨 (잡 투입기 → Backend, 김성모 · 노찬경)
- `fenced`: 실패 시점에 다음 세대 JobSet이 아직 없으면 `failed`로 나감 (복구는 세대를 먼저 올리고, 새 JobSet은 옛 실행 종료 확인 뒤 생김)
