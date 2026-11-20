# experiments — 시연 · 실험 시나리오 (GPU-77)

데이터 파트 몫: 잡 투입기(K8s `scheduling/submitter`, 김성모 GPU-100)가 읽는 시나리오 파일, 실험 표, 시연에서 보여줄 측정값, 체크포인트 저장 주기 의견.
다른 파트 몫: 모델 · 데이터셋 · 학습 설정(노찬경), 장애 종류 · 복구 단계(한송이 GPU-104), 장애 주입 CLI(노찬경), 투입기(김성모).

## 시나리오 파일

형식은 투입기와 같다: `제출 시각(초), GPU 수, step 수, 방식[, 실행 시간(초)]` (`#` 줄은 주석).

| 파일 | 용도 | 내용 |
|---|---|---|
| `scenarios/demo-single.csv` | 시연 · 장애 실험 1회 | 작업 1개, FSDP 2장, 1000 step (≈ 27분) |
| `scenarios/mixed-now.csv` | 스케줄링 부하 (지금 용량) | 15분 동안 8개, 1 · 2장 |
| `scenarios/mixed-target.csv` | 스케줄링 부하 (목표 용량 EKS 3 × (4+1)) | 30분 동안 14개, 4장 작업 포함 |

길이 근거 (실행 시간 칸 = step × 1.6초, kind(KWOK) 흉내용): 11/20 EKS SmolLM2-135M + wikitext-2, 시작 → 완료 벽시계(체크포인트 저장 · 승인 포함) — `eb497561` FSDP 2장 3100 step 5217초(1.68초), `2c512e82` 2장 1.54초, `f8f6afef` FSDP 4장 1550 step 2493초(1.61초). 순수 step 시간은 p50 1.25초(이태림 지표). 1장은 아직 실측 없음(같은 값으로 둠). 체크포인트 승인 간격 약 10.8분(약 390 step), 첫 승인까지 약 11분. 옛 근거(gpt2, step당 1.22초)는 11/12 `4734d7ac`.

```bash
python submit.py experiments/scenarios/demo-single.csv --exp-id exp-a-ours-xid79-r1 \
  --policy ours --fault-type xid79 --spare --checkpoint-every-steps 100
```

장애 주입 시점은 파일에 넣지 않는다. 실험 절차: 제출 → 첫 `checkpoint.approved` 확인 → 정해진 시간 뒤 주입 → 복구 · 재개 → 완료 → `experiment.ended`.

## 작업 1개 손으로 제출 — `submit_job.py` (GPU-119)

투입기 없이 같은 조건의 작업 하나를 반복해서 낼 때 쓴다 (E2E 장애 시험, 측정 확인). Backend `POST /api/jobs`로 낸다.

```bash
python experiments/submit_job.py --model-dir ~/models/gpt2 --name e2e-fault-2 --steps 1500 --exp-id e2e-1117
```

| 인자 | 기본값 | 뜻 |
|---|---|---|
| `--model-dir` | (필수) | Hugging Face 모델 폴더. 받는 방법은 스크립트 맨 위 주석 |
| `--steps` | 1500 | `max_steps`. SmolLM2-135M · FSDP 2장은 저장 포함 step당 약 1.6초 → 1500이면 학습 약 40분 (11/20 실측, gpt2는 약 1.32초) |
| `--gpus` · `--strategy` · `--batch` | 2 · fsdp · 4 | GPU 수 · 방식 · GPU당 배치 |
| `--exp-id` | 없음 | 실험 ID. 측정 때 `--exp`로 묶는다 |
| `--dry-run` | | 데이터만 만들고 spec 출력 |

- **학습 데이터는 스크립트가 만든다.** train.py는 토큰을 1024개씩 묶어 샘플 하나로 쓰므로, `steps × gpus × batch × 1024` 토큰(+30%)이 되게 줄 수를 정한다. 같은 인자면 같은 파일 (1500 · 2 · 4 → 약 16만 줄, 100MB).
- **train.py는 이 레포 `trainer/train.py`를 올린다.** 학습 이미지에는 train.py가 없고 사용자 업로드를 쓴다 (재개 · 진행률 · 지표가 들어 있는 최신본이어야 장애 뒤 이어서 학습함).
- **저장 간격은 Backend가 정한다** (지금 845초). 첫 승인이 학습 시작 후 약 14분이라 장애는 그 뒤에 넣는다.

### 장애 시험 순서 (11/17 1차 기록: Jira GPU-119)

1. 제출 → DB에서 `job.started` · 첫 `checkpoint.approved` 확인
2. 학습 노드 찾기: EKS API는 비공개라 맥에서 kubectl이 안 된다 → EC2 CloudWatch `CPUUtilization` · `NetworkOut`이 높은 학습 노드 2대가 그 작업 (FSDP는 두 노드가 계속 통신)
3. 장애 주입: 그 노드 `aws ec2 terminate-instances` (노드 그룹이 새 노드로 교체)
4. 확인: `job.failed` → 복구 컨트롤러 펜싱(`recovery_id`, 세대 +1) → 세대 2 `job.started` · 첫 진행률이 승인 step + 1부터
5. 측정: `python -m collector.measure --exp e2e-1117 --create-exp "11/17 E2E"`

11/17 1차: 감지(`job.failed`)까지 2분 11초. 복구는 분류기(`gpu.fault` 생산)와 복구 컨트롤러 실행 모드가 아직 없어서 시작 안 됨 → 완성 후 다시.

### 실험 ↔ 작업 연결 (`exp_id`)

| 경우 | 연결되는 방법 |
|---|---|
| 투입기로 제출 | 투입기가 `experiment.started`(실험 행)를 먼저 내고 제출에 `exp_id`를 실음 → 수집기가 `job_runs.exp_id`를 채움. 제출이 먼저 와도 `experiment.started`가 오면 나중에 채움 |
| `submit_job.py --exp-id`로 제출 | `experiment.started`가 없어 실험 행이 없음 → 수집기는 `job_runs.exp_id`를 비워 둠. 측정 때 `--create-exp`가 실험 행을 만들고, 제출 사건의 `exp_id`로 세대 1 · 2 줄을 모두 채움 (`linked_job_runs`에 개수) |
| `exp_id` 없이 제출 | 측정 때 `--jobs ID,...` 또는 `--since` · `--until`로 고름 |

## 실험 표

변수: 정책(`ours` · `naive` 자동 재시작만 · `hyperpod` 흉내 · `manual`), 장애 종류(필수: 노드 하드웨어 `xid79` · 소프트웨어 오류 `sw_error` · 스팟 회수 `spot`, 목표: AZ 장애 `az` · 스트래글러 `straggler`), 예비 GPU on/off, 저장 주기.

| 실험 | 질문 | 조건 | 반복 | 실행 수 |
|---|---|---|---|---|
| A 정책 비교 (필수) | 우리 정책이 단순 재시작보다 빨리 · 덜 잃고 복구하나 | `ours` · `naive` × `xid79` · `sw_error` · `spot`, 예비 on | 5 | 30 |
| B 예비 GPU 효과 | 예비가 있으면 복구가 얼마나 빨라지나 | `ours` × `xid79` × 예비 off (on은 A 재사용) | 5 | 5 |
| C AZ 장애 대피 (목표) | 클러스터 통째 장애에서 다른 EKS로 이어지나 | `ours` × `az` (EKS 2개 이상 필요) | 3 | 3 |
| D 스트래글러 (목표) | 느린 GPU를 먼저 빼서 학습 시간이 줄어드나 | `ours` × `straggler` vs 미격리 | 3 | 6 |
| E 저장 주기 | 저장을 자주 하면 잃는 양 vs 저장 시간 손해 | `ours` × `xid79` × 120초 · 공식값(≈845초) | 3 | 6 |
| F 스케줄링 부하 (김성모) | 빈 패킹 · 복구 큐 우선이 대기 시간을 줄이나 | `mixed-*.csv` × 디스패처 on/off | 3 | 6 |

비용 추정 (확인 필요): g4dn.xlarge 온디맨드 약 $0.526/시간(us-east-2 공시가) 기준, 실험 1회 = 학습 2장 + 예비 1장 × 약 30분 ≈ $0.8. A~E 50회 ≈ $40 + EKS 컨트롤 플레인 · NAT 등 켜둔 시간 비용. 스팟이면 GPU 비용은 더 낮다. 1순위는 A(필수) → E → B → C · D.

## 시연에서 보여줄 측정값 (ADR-15)

| 값 | 뜻 | 출처 |
|---|---|---|
| 감지 시간 (MTTD) | 장애 시작 → 분류기 판정 | `v_incidents` (장애 시작 = 수동 주입 기록 `python -m collector.injection add`, 없으면 분류기 신호 `payload.ts` → `fault.detected` · `fault.reclassified`, GPU-239) |
| 복구 시간 (MTTR) | 장애 주입 → 새 세대 시작 | `v_incidents` (→ `job.resumed`) |
| ETTR | 학습이 실제로 돌았던 시간 비율 · 재개까지 걸린 시간 | `job_runs` · `job_progress` · `results` |
| 날아간 step | 장애 시점 step − 마지막 승인 step | `job_progress.step` · `checkpoints` |
| 대기 시간 | 제출 → Kueue 입장 → 시작 | `job.submitted` · `job.admitted` · `job.started` |

시연 화면: Grafana 한 장에 위 다섯 값 + 정책별 비교 막대(실험 A).

## 체크포인트 저장 주기 의견 (노찬경에게)

Backend는 Young–Daly 공식으로 간격을 정한다: √(2 × 저장 시간 16.5초 × MTBF 21600초) ≈ 845초. 11/12 E2E에서 실제로 845초마다 저장됐다.

1. **운영 기본값은 공식 그대로 좋다.** 저장 비용과 잃는 양을 같이 따진 표준 방식이다.
2. **실험 · 시연에서는 간격을 실험 조건으로 고정할 수 있어야 한다.** MTBF 6시간 가정과 달리 실험은 몇 분 안에 장애를 넣으므로, 845초면 첫 저장 전에 장애가 나거나 최대 14분 학습을 잃어 정책 차이가 저장 시점 운에 묻힌다. 제출 때 `checkpoint_policy.interval_seconds`(또는 step 간격)를 덮어쓸 수 있게 해 주면, 투입기가 실험 E 조건(120초 · 공식값)을 넣고 `experiment.started`의 `checkpoint_every_steps`에 기록한다.
3. **첫 측정 20 step(`warmup_measure_steps`) 동안은 저장하지 않으므로** 장애 주입은 첫 승인 뒤에 한다 (위 실험 절차).
4. **저장량**: 1회 저장 약 1.65GB(2 rank), job `4734d7ac`는 3회로 약 4.9GB. 120초 간격으로 20분 돌리면 10회 ≈ 16GB → ADR-30(학습 끝나면 최종 체크포인트만 남김) 정리가 꼭 필요하다.

## 확인 필요

- ~~실험 ↔ 작업 연결~~: 해결 (11/17). Backend `POST /api/jobs`가 `exp_id`를 받아 `job.submitted` payload · JobSet 주석 `devoops/exp-id`에 넣는다 (GPU-172). 연결 방법은 위 "실험 ↔ 작업 연결"
- `exp_id` 매기기: 투입기 `--exp-id`, 행은 수집기가 `experiment.started`로 만든다 (투입기 README의 질문에 대한 답)
- step 수를 Backend에 넘길 칸 (투입기 README "아직 정할 것")
- 스팟 회수 실험(`spot`)의 마지막 저장은 ADR-28 · GPU-120 확정 뒤
