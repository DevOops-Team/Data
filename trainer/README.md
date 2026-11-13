# trainer

학습 Pod의 trainer 컨테이너 이미지 `ghcr.io/devoops-team/trainer`는 **학습 환경만** 담는다 (torch 2.5.1 · CUDA 12.4 · transformers).
`train.py`는 이미지에 없다. 사용자가 제출 화면에서 올리면 Backend가 데이터셋(`datasets/<id>/v1/code/train.py`, manifest에 sha256)에 넣고,
ckpt-init이 모델 · 데이터와 함께 `/data/code/train.py`로 받아 trainer가 `torchrun /data/code/train.py`로 실행한다.

- `train.py`: 기본 학습 코드(업로드용 예시). 어떤 Hugging Face causal LM이든 FSDP · DDP로 학습. 업로드한 모델은 `/data/model`, 데이터는 `/data/data/*.jsonl`(ckpt-init이 받아 둠)
- 체크포인트 간격: config.json `checkpoint_policy.interval_seconds`(Young/Daly)를 처음 몇 step의 시간으로 step 간격으로 바꾼다 (설계 문서 5절)
- ckpt-sidecar 약속: `step-N/` 뒤 `step-N.done`, `STOP`이면 종료 코드 75, 마지막 저장 뒤 `.approved`를 120초 기다림
- 테스트: `cd trainer && python3 -m unittest test_train.py` (torch 없이 순수 함수만)
- 직접 쓴 train.py도 위 환경 변수 · 체크포인트 규약(`TRAIN_CONFIG` · `DATA_DIR` · `CKPT_DIR` · `PROGRESS_FILE`, `.done`, `STOP`)을 따라야 저장 · 재개가 된다
- CI: 이 레포 `.github/workflows/ci.yml` matrix `trainer`. `trainer/`가 바뀐 main push에서 이미지 push → `DEPLOY_IMAGES`에 trainer가 있으면 봇이 Gitops `onprem/platform/backend/overlays/onprem`의 `images:` digest 갱신 → Backend env `TRAINER_IMAGE`
- 담당: 김현수 (GPU-154로 Backend 레포에서 이전)
