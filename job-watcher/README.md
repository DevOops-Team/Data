# job-watcher — (잡) 감시기

관리 클러스터의 JobSet · Workload · RelocationRequest 상태 변화를 Kafka `job.lifecycle` 사건으로 올린다. 수집기가 이 사건으로 `data.job_runs`(세대별 제출 · 시작 · 종료 시각)를 채운다.

기준: ADR-24(이름), ADR-29(위치: 관리 클러스터 Pod 1개), ADR-19(재배치 요청 CRD), ADR-15(측정 지표), 사건 판정은 김성모 11/12 답변, Jira GPU-118. 설명: Confluence 「(잡) 감시기 (GPU-118)」 (1부 쉽게 보기 · 2부 상세, 흐름도).

| 파일 | 내용 |
|---|---|
| `watcher/model.py` | JobSet · Workload · RelocationRequest dict → 필요한 값(작업 ID · 세대 · 큐 · GPU · 클러스터 · 조건 시각 · 재배치 요청) |
| `watcher/rules.py` | 상태 → 보낼 사건 목록 (순수 함수, 판정 표는 파일 머리말) |
| `watcher/events.py` | 사건 봉투(수집기 형식) · Kafka 발행 |
| `watcher/main.py` | list · watch 루프(3종), 캐시, 시작 동기화, 중복 방지 |

## 사건

| type | 판정 | occurred_at |
|---|---|---|
| `job.submitted` | **재제출본**이 보임 — 복구 재배치(주석 `devoops/recovery-id`) 또는 사용자 다시 시작(주석 `devoops/restart-reason`, payload `restart_reason` · `restart_requested_at`, GPU-234). 첫 제출은 Backend가 보내므로 안 보냄 (`EMIT_FIRST_SUBMITTED=true`면 보냄) | `creationTimestamp` |
| `job.admitted` | Workload `Admitted=True` (Kueue 입장) | 조건 시각 |
| `job.placed` | Workload `status.clusterName` (MultiKueue 배치) | admissionCheck Ready 시각 |
| `job.started` / `job.resumed` | Workload `PodsReady=True`, 없으면 JobSet 전부 ready. 재제출본(`devoops/recovery-id` 또는 `devoops/restart-reason`)이면 resumed. 사용자 다시 시작의 resumed는 측정 `v_incidents`에서 장애 복구로 세지 않음 (`db/006`) | 조건 시각, 없으면 본 시각 |
| `job.completed` | JobSet `Completed=True` | 조건 시각 |
| `job.failed` | JobSet `Failed=True` | 조건 시각 |
| `job.fenced` | 종료 조건 없이 삭제 + `spec.jobRef`가 이 JobSet인 RelocationRequest 있음. payload `request_id` · `action` | `deletionTimestamp` |
| `job.cancelled` | 종료 조건 없이 삭제 + RelocationRequest 없음 + 주석 `devoops/stop-reason: user` (Backend 사용자 중단이 지우기 전에 닮, payload `reason` · `requested_at`) → 화면 `CANCELLED` | `deletionTimestamp` |
| `job.stopped` | 종료 조건 없이 삭제 + RelocationRequest 없음 + 중단 주석 없음 (수동 삭제 등) | `deletionTimestamp` |
| `job.evicted` | Workload `Evicted=True` — Kueue가 내림. `reason` `PodsReadyTimeout`(입장 뒤 `waitForPodsReady.timeout` 10분 안에 Pod 전부 Ready 실패: ckpt-init 다운로드 · 이미지 pull이 길 때, GPU-256) · `Preempted` 등. 종료가 아니라 Kueue가 같은 세대로 다시 넣음(최대 `backoffLimitCount` 3번). 발생마다 한 번, payload `reason` · `message` · `requeue_count` (GPU-259) | 조건 시각 |
| `job.deactivated` | Workload `spec.active=false` — 다시 넣기 한도를 넘어 Kueue가 멈춤, 더 안 넣음. 이미 끝난 세대면 안 보냄. payload 같음 (GPU-259) | 본 시각 |

- `event_id = job-{작업ID}-g{세대 4자리}-{key}` (key: submitted · admitted · placed · started · end · `evicted-{조건 시각 숫자}` · deactivated). 세대마다 종료는 하나(`-end`), 내림은 발생마다. 이미 끝난 잡을 나중에 지우면 사건 없음
- Kueue 내림 · 비활성화 사건이 없으면 시간 초과로 내린 작업이 수집기 · 복구 컨트롤러 · 화면에서 "입장 후 조용함"으로만 보였다. 수집기는 이 두 종류를 `data.events`에만 저장(`job_runs`는 안 바꿈). MultiKueue에서 워커 쪽 내림이 관리 클러스터 Workload 조건으로 어떻게 보이는지는 EKS에서 확인 필요
- `payload.time_source`: `k8s`(쿠버네티스가 찍은 시각) · `observed`(감시기가 본 시각)
- 시작은 ETTR 기준에 맞춰 **PodsReady**(Pod 준비) 시각. Kueue 입장 시각은 `job.admitted`로 따로 남음
- 시작 직후에는 세 목록(JobSet · Workload · RelocationRequest)을 다 받은 뒤부터 판정 (시작 시각 · fenced 판정이 틀린 값으로 굳는 것 방지)

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
| `RELOCATION_API` | `scheduling.devoops.site/v1alpha1` |
| `EMIT_FIRST_SUBMITTED` | `false` |
| `WATCH_TIMEOUT_S` | `300` |
| `KAFKA_MESSAGE_TIMEOUT_MS` | `1800000` |

권한(ServiceAccount `job-watcher`, 네임스페이스 `data-system`) — `training`에서 읽기만:

| 리소스 | 동사 |
|---|---|
| `jobsets.jobset.x-k8s.io` | `get · list · watch` |
| `workloads.kueue.x-k8s.io` | `get · list · watch` |
| `relocationrequests.scheduling.devoops.site` | `get · list · watch` |

네트워크: 관리 클러스터 Pod → Kafka `172.30.0.40~42:9092`. 매니페스트는 Gitops(ADR-21, 노찬경), 경로 `onprem/data/job-watcher/overlays/onprem`.

## 테스트

```bash
pip install -r requirements-dev.txt
python -m pytest -q        # 수집기 envelope.parse로 형식 호환까지 확인
```

## 확인 필요

- ETTR 시작: DB `job_runs.started_at`은 "첫 step 완료"로 정의됐지만 감시기는 Pod 준비 시각까지만 안다 → 첫 step은 학습 코드 · 사이드카가 따로 올릴지 (김현수 · 노찬경)
- `exp_id`: JobSet에 `devoops/exp-id` 주석이 있어야 실험과 연결됨 (잡 투입기 → Backend, 김성모 · 노찬경)
