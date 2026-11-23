# trainer

학습 Pod의 trainer 컨테이너 이미지 `ghcr.io/devoops-team/trainer`는 **학습 환경 + 시스템 라이브러리 `devoops_ckpt`**를 담는다 (torch 2.5.1 · CUDA 12.4 · transformers, QLoRA용 bitsandbytes · peft · accelerate).
이미지는 하나다: Full fine-tuning · `fit()` 학습과 QLoRA 학습이 같은 이미지에서 돈다 (QLoRA 패키지는 쓰는 train.py만 불러온다, GPU-160).
`train.py`는 이미지에 없다. 사용자가 제출 화면에서 올리면 Backend가 데이터셋(`datasets/<id>/v1/code/train.py`, manifest에 sha256)에 넣고,
ckpt-init이 모델 · 데이터와 함께 `/data/code/train.py`로 받아 trainer가 `torchrun /data/code/train.py`로 실행한다.

## 사용자 코드와 시스템의 분리 (GPU-209)

| 파일 | 누구 것 | 하는 일 |
|---|---|---|
| `train.py` | 사용자 (이건 기본 예시) | 모델 · 데이터셋 · loss 함수만 정한다 |
| `devoops_ckpt.py` | 시스템 (이미지 `/opt/train`, `PYTHONPATH`) | 학습 루프 · 분산 감싸기(FSDP/DDP) · DataLoader · 학습률 · 재개 · 체크포인트 저장 규약 · 진행률 · 학습 지표 · 펜싱 STOP · 마지막 승인 대기 · 실패 기록 · config.json 확인 |

### 기본: `ckpt.fit()` (루프 소유형)

```python
import devoops_ckpt as ckpt

def loss_fn(model, batch):                 # batch는 이미 GPU에 (원소 하나인 tuple은 풀어서)
    return model(input_ids=batch, labels=batch).loss

ckpt.fit(model=load_model, dataset=load_dataset, loss_fn=loss_fn)
```

| 인자 | 필수 | 내용 |
|---|---|---|
| `model` | O | `nn.Module` 또는 만드는 함수 (감싸기 전). 함수면 fit 안에서 만들어 실패가 FAILED로 남는다 |
| `dataset` | O | torch `Dataset` 또는 만드는 함수 |
| `loss_fn` | O | `loss_fn(model, batch) -> loss` |
| `optimizer` | | `optimizer(params) -> Optimizer`. 기본 `AdamW(lr=config.learning_rate, weight_decay=0.1)` |
| `lr_schedule` | | `lr_schedule(step, total) -> lr` (None이면 안 바꿈). 기본 warmup → cosine(10%까지) |
| `max_grad_norm` | | 기본 1.0, None이면 clip 안 함 |
| `collate_fn` | | DataLoader에 그대로 |

배치 크기 · epoch · max_steps · seed · 학습 방식(fsdp/ddp) · world_size는 config.json(제출 화면 값)에서 온다.

### 루프를 직접 짤 때: `ckpt.Session`

gradient accumulation · 옵티마이저 여러 개처럼 `fit()`이 못 하는 루프는 `with ckpt.Session(...) as s:` 안에서 직접 부른다 (모든 rank가 같은 순서로).
`fit()`도 내부에서 이걸 쓴다.

| 호출 | 필수 | 하는 일 |
|---|---|---|
| `ckpt.load_config()` | 권장 | `/code/config.json` 읽기 · 체크섬 · 키 확인 → `(cfg, cfg_hash)` |
| `s.load(model, optimizer, device)` | 재개하려면 | ckpt-init이 받아 둔 승인본을 불러옴 → 불러온 step (없으면 0) |
| `s.data_position(start, steps_per_epoch)` | 재개하려면 | 데이터 위치 `(epoch, 건너뛸 배치 수)` |
| `s.report(step, loss, data_wait)` | 선택 | 진행률(화면) · 지표(:8000) · 저장 간격 측정 · 펜싱 STOP 확인. **안 부르면 측정 전이라 마지막 step에만 저장** |
| `s.should_save(step)` · `s.save(model, optimizer, step)` | 저장하려면 | 간격마다 `/ckpt/step-N/` + `.done` → 사이드카가 업로드 · 승인 |
| `s.finish()` | 권장 | 마지막 저장 `.approved`를 최대 120초 대기 → COMPLETED |
| `Session(model_keys=...)` · `s.model_keys` | QLoRA | 저장 · 재개할 모델 state_dict 키 (기본 None = 전부). 어댑터 이름만 주면 체크포인트에 4bit 기본 가중치가 안 들어간다. 재개 때 그 키만 읽고, 없는 키가 있으면 실패 |

### 예시: `examples/fsdp_lm_train.py` (장애 복구 묶음 공용, GPU-160)

`Session` 직접 루프 하나로 Full fine-tuning(F1 Qwen2.5-0.5B · F2 1.5B)과 QLoRA(Q2 Qwen2.5-14B)를 돌린다. 묶음마다 다른 건 파일 맨 위 「묶음 설정」 상수(`METHOD` · `SHARDING` · `GRAD_ACCUM` · `ACCUM_NO_SYNC`)뿐이다.

- 공통: FSDP fp16 MixedPrecision(버퍼 fp32) + `ShardedGradScaler`, 기울기 누적, 청크 loss 512, activation checkpointing
- QLoRA: NF4 + double quant(compute · quant_storage fp16), LoRA r 16 · alpha 32 · 전 선형층, 어댑터를 따로 감싸고 `use_orig_params=False`, `s.model_keys` = 어댑터 이름
- 묶음: 체크포인트 버킷 `datasets/qwen25-14b-wikitext103/v1/` (권장 제출: gpus 4 · batch 2 · lr 2e-4)

- 라이브러리 없이 직접 짜도 된다. 그때는 `devoops_ckpt.py` 맨 위 「규약」(체크포인트 규약 3 · 6 · 7절)을 지켜야 저장 · 재개가 된다
- 안 지켜도 **안전은 깨지지 않는다**: 옛 세대는 사이드카가 업로드 · 승인 전에 DynamoDB 펜싱으로 막고, relocator가 옛 JobSet을 지워 멈춘다. 재개 · 진행률이 안 될 뿐
- 한계: Pod의 AWS 권한(ServiceAccount)이 하나라 사용자 코드도 S3 · DynamoDB에 쓸 수 있다 — 실수 방지까지이고 보안 분리는 아님

## 동작

- 체크포인트 간격: config.json `checkpoint_policy.interval_seconds`(Young/Daly)를 처음 `warmup_measure_steps` step의 시간으로 step 간격으로 바꾼다 (설계 문서 5절, `report()` 안)
- 재개: ckpt-init이 승인본을 `/ckpt/step-N/`에 받고 `/ckpt/resume.json`을 남기면 그 step의 모델 · 옵티마이저(Adam 모멘트)를 불러와
  step N 다음부터 같은 데이터 순서로 이어 간다 (GPU-119). 옵티마이저는 `state`만 저장하고 `param_groups`(lr · betas 등)는 코드 · 스케줄이 정한다
- 학습 지표 (GPU-96): 모든 rank가 `:8000/metrics`(env `METRICS_PORT`, 0이면 끔)로 8개를 연다 — 이름 · 라벨 · 버킷은 Monitoring 레포 `docs/training-metrics.md`.
  `train_step_seconds`(저장 전까지) · `train_data_wait_seconds` · `train_current_step` · `train_loss`(전체 rank 평균) · `train_stage` · `train_last_checkpoint_step`,
  `train_rank_compute_seconds` · `train_rank_comm_seconds`는 `PROFILE_EVERY`(기본 25) step마다 1 step만 torch.profiler로 (GPU에서만). JobSet: trainer `containerPort: 8000` + Pod `prometheus.io/*` 어노테이션
- 실패 기록 (GPU-188): `with Session` 안에서 예외로 끝나면 진행률에 `FAILED` + 이유 한 줄 → `v_job_status.progress_error`. 펜싱 STOP은 실패로 안 남김 — 어느 rank든 STOP을 받으면 모든 rank가 같은 step에서 멈추고 진행률 `STOPPED` (GPU-244). 종료 코드 75는 없앰 (ADR-43, 종료 코드 1은 뜻 없음)

## 확인

- 테스트: `cd trainer && python3 -m unittest test_train.py` (torch 없이 순수 함수만, prometheus_client가 있으면 /metrics 노출까지)
- 재개 확인: `python3 trainer/examples/resume_check.py` (torch만, CPU · gloo · 2 프로세스) — 공개 API `fit()` 그대로:
  끊김 없이 6 step = 3 step 저장 → 새로 불러와 3 step, FSDP · DDP 최종 가중치 차이 0 + 마지막 승인 · COMPLETED + loss 예외 → FAILED + 이유 + (Session) rank 0에만 STOP → 모든 rank Stopped · STOPPED, 마지막 승인 대기 중 STOP도 같음. torch 2.5.1(이미지와 같음) · 2.15에서 통과
- CI: 이 레포 `.github/workflows/ci.yml` matrix `trainer`. `trainer/`가 바뀐 main push에서 이미지 push → `DEPLOY_IMAGES`에 trainer가 있으면 봇이 Gitops `onprem/platform/backend/overlays/onprem`의 `images:` digest 갱신 → Backend env `TRAINER_IMAGE`
- **배포 순서**: 새 `train.py`(라이브러리 import)는 라이브러리가 든 trainer 이미지에서만 돈다. 이미지가 먼저 배포된 뒤 새 train.py를 올린다. 옛 train.py(규약 직접 구현)는 새 이미지에서도 그대로 돈다
- 담당: 김현수 (GPU-154로 Backend 레포에서 이전)
