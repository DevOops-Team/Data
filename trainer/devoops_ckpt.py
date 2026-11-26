"""devoops_ckpt — 학습 코드가 시스템과 맞물리는 부분을 대신 해 주는 라이브러리 (GPU-209).

사용자 train.py는 학습만 하고, 재개 · 저장 규약 · 진행률 · 지표 · 펜싱 STOP · 마지막 승인 대기 · 실패 기록은 여기서 한다.
trainer 이미지에 들어 있다 (/opt/train, PYTHONPATH). 안 써도 되지만, 그러면 아래 규약을 직접 지켜야 저장 · 재개가 된다.

1) 기본: fit() — 학습 루프까지 시스템이 돈다. 사용자는 모델 · 데이터셋 · loss만

    import devoops_ckpt as ckpt

    def loss_fn(model, batch):                              # batch는 이미 GPU에
        return model(input_ids=batch, labels=batch).loss

    ckpt.fit(model=lambda: AutoModelForCausalLM.from_pretrained(...),   # 모듈 또는 만드는 함수
             dataset=TensorDataset(...),                                # Dataset 또는 만드는 함수
             loss_fn=loss_fn,
             optimizer=lambda params: torch.optim.AdamW(params, lr=1e-5))  # 선택, 기본 AdamW(config lr)

2) 루프를 직접 짜야 할 때 (gradient accumulation · 여러 옵티마이저 등): Session

    cfg, cfg_hash = ckpt.load_config()                    # /code/config.json (Backend가 만든 학습 설정)
    dist.init_process_group("nccl")
    ... 모델 · 옵티마이저 · DataLoader (사용자 코드) ...
    with ckpt.Session(mode=cfg.training_mode, total_steps=total, cfg_hash=cfg_hash,
                      policy=cfg.checkpoint_policy, warmup_steps=cfg.warmup_steps) as s:
        start = s.load(model, optimizer, device)            # 승인본이 있으면 불러오고 그 step, 없으면 0
        epoch, skip = s.data_position(start, len(loader))   # 재개할 데이터 위치 (같은 seed · set_epoch 전제)
        step = start
        while step < total:
            sampler.set_epoch(epoch)
            for batch in loader:
                if skip: skip -= 1; continue
                step += 1
                loss = ... 학습 한 step ...
                s.report(step, loss)                        # 진행률 · 지표 · 저장 간격 측정 · 펜싱 STOP 확인
                if s.should_save(step):
                    s.save(model, optimizer, step)          # CKPT_DIR/step-N/ (DCP) + .done → 사이드카가 업로드 · 승인
                if step >= total: break
            epoch += 1
        s.finish()                                          # 마지막 저장이 승인될 때까지 대기

규약 (라이브러리 없이 직접 할 때 — 체크포인트 규약 3 · 6 · 7절, JobSet 템플릿 7절)
  - 재개: CKPT_DIR/resume.json {"resume", "approved_step", "step_dir"}의 step 폴더를 불러와 다음 step부터
  - 저장: CKPT_DIR/step-XXXXXXXX/에 모든 rank가 쓴 뒤 step-XXXXXXXX.done (빈 파일)을 마지막에
  - 끝: 마지막 저장의 step-XXXXXXXX.approved가 생길 때까지 기다린 뒤 종료
  - 진행률(선택): rank 0이 PROGRESS_FILE에 JSON (step · total_steps · loss · status · error)
  - 중단: 어느 rank든 CKPT_DIR/STOP이 생기면 모든 rank가 같은 step에서 저장 없이 끝낸다 (진행률 STOPPED, 종료 코드 1).
    종료 코드에는 뜻이 없다 (ADR-43 — 펜싱은 감시기 job.fenced로 알림). 0이 아닌 건 JobSet이 Completed로 보지 않게.
    펜싱은 relocator가 옛 JobSet을 지워서도 멈춘다

환경 변수 (Backend jobspec이 넣음): TRAIN_CONFIG · CONFIG_SHA256 · CKPT_DIR · PROGRESS_FILE · GENERATION · JOB_ID ·
  METRICS_PORT(8000) · PROFILE_EVERY(25)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, fields

SCHEMA_VERSION = 3          # Backend가 쓰는 최신. 3: finetune(학습 방식 full · qlora) 추가 (GPU-262)
SCHEMA_VERSIONS = (2, 3, 4)  # 2는 finetune 칸이 없다 = full · 4는 checkpoint_policy.every_steps (GPU-298). 배포 순서: 하네스 먼저, Backend 나중
FINETUNE_METHODS = ("full", "qlora")


CHECKPOINT_FORMATS = {"fsdp": "fsdp-dcp", "ddp": "ddp-pt"}


class Stopped(SystemExit):
    """ckpt-sidecar가 STOP을 써서(펜싱) 끝남. 실패로 남기지 않고 진행률 STOPPED. 종료 코드 1 (뜻 없음, ADR-43)."""

    def __init__(self, step: int):
        super().__init__(1)
        self.step = step


APPROVAL_WAIT_SECONDS = 120


@dataclass(frozen=True)
class DatasetRef:
    uri: str
    format: str
    manifest_sha256: str


@dataclass(frozen=True)
class CheckpointPolicy:
    interval_seconds: float
    estimated_save_seconds: float
    mtbf_seconds: float
    warmup_measure_steps: int
    # 고정 저장 간격(step, GPU-298). 0보다 크면 이 간격으로 저장하고 Young/Daly 측정 · 변환은 하지 않는다.
    # 0(schema 2 · 3은 칸 없음 = 0)이면 원래대로 interval_seconds를 측정한 step 시간으로 바꾼다.
    every_steps: int = 0


@dataclass(frozen=True)
class LoraConfig:
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class Finetune:
    """학습 방식 (GPU-262). full = 모든 파라미터 학습, qlora = 4bit 기본 가중치 + LoRA 어댑터만 학습."""
    method: str
    lora: LoraConfig | None


FULL = Finetune(method="full", lora=None)


@dataclass(frozen=True)
class TrainConfig:  # 기본값 없음: 백엔드가 모든 키를 채운다
    model: str
    training_mode: str
    checkpoint_format: str
    dataset: DatasetRef
    epochs: int
    max_steps: int | None
    batch_size_per_gpu: int
    learning_rate: float
    warmup_steps: int
    seed: int
    world_size: int
    checkpoint_policy: CheckpointPolicy
    finetune: Finetune


def parse_finetune(raw) -> Finetune:
    if not isinstance(raw, dict) or set(raw) != {"method", "lora"}:
        raise SystemExit("config finetune must be {method, lora}")
    method = raw["method"]
    if method not in FINETUNE_METHODS:
        raise SystemExit(f"config finetune.method must be one of {FINETUNE_METHODS}, got {method}")
    if method == "full":
        if raw["lora"] is not None:
            raise SystemExit("config finetune.lora must be null for full")
        return FULL
    lora = raw["lora"]
    keys = {f.name for f in fields(LoraConfig)}
    if not isinstance(lora, dict) or set(lora) != keys:
        raise SystemExit(f"config finetune.lora must have {sorted(keys)}")
    if not lora["target_modules"] or int(lora["r"]) < 1:
        raise SystemExit("config finetune.lora needs r >= 1 and target_modules")
    return Finetune(method=method, lora=LoraConfig(r=int(lora["r"]), alpha=int(lora["alpha"]),
                                                   dropout=float(lora["dropout"]),
                                                   target_modules=tuple(lora["target_modules"])))


def parse_config(raw: bytes, expected_sha: str | None) -> tuple[TrainConfig, str]:
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha and expected_sha != digest:
        raise SystemExit(f"config hash mismatch: {digest} != {expected_sha}")

    data = json.loads(raw)
    version = data.pop("schema_version", None)
    if version not in SCHEMA_VERSIONS:
        raise SystemExit(f"config schema_version must be one of {SCHEMA_VERSIONS}, got {version}")
    if version == 2:
        if "finetune" in data:
            raise SystemExit("config schema_version 2 has no finetune")
        data["finetune"] = {"method": "full", "lora": None}
    names = {f.name for f in fields(TrainConfig)}
    if set(data) != names:  # 오타 · 누락을 조용히 넘기지 않는다
        raise SystemExit(f"config keys: missing {names - set(data)}, unknown {set(data) - names}")
    data["dataset"] = DatasetRef(**data["dataset"])
    policy = data["checkpoint_policy"]
    if not isinstance(policy, dict) or (version == 4) != ("every_steps" in policy):  # schema 4에만, 반드시 있다
        raise SystemExit(f"config schema_version {version}: checkpoint_policy.every_steps must be "
                         f"{'present' if version == 4 else 'absent'}")
    data["checkpoint_policy"] = CheckpointPolicy(**policy)
    if data["checkpoint_policy"].every_steps < 0:
        raise SystemExit(f"checkpoint_policy.every_steps must be >= 0, got {data['checkpoint_policy'].every_steps}")
    data["finetune"] = parse_finetune(data["finetune"])
    cfg = TrainConfig(**data)
    want = CHECKPOINT_FORMATS.get(cfg.training_mode)
    if want is None:  # 다른 방식으로 돌면서 요청한 방식으로 보이는 일을 막는다
        raise SystemExit(f"this train.py does not implement {cfg.training_mode}")
    if cfg.checkpoint_format != want:
        raise SystemExit(f"{cfg.training_mode} needs checkpoint_format {want}, got {cfg.checkpoint_format}")
    if cfg.dataset.format != "text-jsonl":
        raise SystemExit(f"unsupported dataset format {cfg.dataset.format}")
    return cfg, digest


def load_config() -> tuple[TrainConfig, str]:
    with open(os.environ["TRAIN_CONFIG"], "rb") as f:
        return parse_config(f.read(), os.environ.get("CONFIG_SHA256"))


def checkpoint_every(total: int, step_seconds: float, interval_seconds: float) -> tuple[int, int]:
    """Young/Daly τ(초)를 step 간격으로 바꾼다. (step 간격 k, 예상 저장 횟수 n)"""
    expected = total * step_seconds
    n = max(1, math.floor(expected / interval_seconds)) if interval_seconds > 0 else 1
    k = max(1, math.ceil(total / n))
    return k, math.ceil(total / k)


def step_name(step: int) -> str:
    return f"step-{step:08d}"


class ProgressFile:
    """현재 상태를 PROGRESS_FILE에 덮어쓴다. 경로가 없으면(rank 0 외) 아무것도 하지 않는다.

    임시 파일 후 rename으로 바꿔 사이드카가 반쯤 쓴 파일을 읽지 않게 한다.
    """

    def __init__(self, path: str, generation: int):
        self.path = path
        self.state = {"generation": generation}

    def write(self, **values):
        if not self.path:
            return
        self.state.update(values)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.path)


STAGES = ("init", "train", "checkpoint", "finalize")


# 초. Monitoring docs/training-metrics.md 3절. SmolLM2(step 약 1.3초) · Qwen 큰 모델(7~23초 추정) 근처에 칸이 2~3개씩
# 있게 촘촘히 (11/20 이태림 제안 — 옛 0.1 · 0.3 · 1 · 3 · 10 · 30은 1~3초 한 칸에 몰려 p95 · p99가 안 보였음).
STEP_BUCKETS = (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 60)
# train_data_wait_seconds 전용 (11/20 이태림 실측: 평균 0.25ms → STEP_BUCKETS로는 전부 첫 칸이라 0.2초로 늘어도
# 안 보여 랭크별 대기 비교(노드 쪽 스트래글러)를 못 함). ms ~ 수 초를 촘촘히
WAIT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


PROFILE_EVERY = 25  # N step마다 1 step만 프로파일 (규약 3-1절, 실측 뒤 조정)


def metric_labels(env) -> dict:
    """규약 2절: job_id · generation · rank(= Pod 번호 JOB_COMPLETION_INDEX)."""
    rank = env.get("JOB_COMPLETION_INDEX", env.get("RANK", "0"))
    return {"job_id": env.get("JOB_ID", "unknown"), "generation": str(env.get("GENERATION", "0")), "rank": str(rank)}


def phase_of(step: int, warmup_steps: int) -> str:
    return "warmup" if step <= warmup_steps else "steady"


def split_kernel_seconds(events) -> tuple[float, float]:
    """프로파일된 커널 이벤트 → (계산 초, NCCL 초). 이름에 nccl이 들어가면 통신 + 대기 (규약 3-1절)."""
    comp = comm = 0.0
    for e in events:
        us = getattr(e, "self_device_time_total", None)
        if us is None:  # torch 2.3 이하 이름
            us = getattr(e, "self_cuda_time_total", 0.0)
        if "nccl" in e.name.lower():
            comm += us
        else:
            comp += us
    return comp / 1e6, comm / 1e6


class Metrics:
    """학습 지표 8개를 /metrics로 연다 (GPU-96, Monitoring docs/training-metrics.md).

    prometheus_client가 없거나 METRICS_PORT=0이면 아무것도 하지 않는다 (학습은 그대로).
    """

    def __init__(self, labels: dict, port: int):
        self.on = False
        self.labels = labels
        try:
            from prometheus_client import CollectorRegistry, Gauge, Histogram, start_http_server
        except ImportError:
            return
        if port <= 0:
            return
        r = self.registry = CollectorRegistry()
        base = ["job_id", "generation", "rank"]
        self.step_seconds = Histogram("train_step_seconds", "한 step 벽시계 시간", base + ["phase"], buckets=STEP_BUCKETS, registry=r)
        self.data_wait = Histogram("train_data_wait_seconds", "step 시작부터 배치가 GPU에 올라올 때까지", base + ["phase"],
                                   buckets=WAIT_BUCKETS, registry=r)
        self.compute = Gauge("train_rank_compute_seconds", "프로파일된 step의 계산 커널 시간 합", base, registry=r)
        self.comm = Gauge("train_rank_comm_seconds", "프로파일된 step의 NCCL 커널 시간 합", base, registry=r)
        self.current_step = Gauge("train_current_step", "완료한 global step", base, registry=r)
        self.loss = Gauge("train_loss", "마지막 step loss (전체 rank 평균)", base, registry=r)
        self.stage_g = Gauge("train_stage", "현재 단계 (해당 단계만 1)", base + ["stage"], registry=r)
        self.last_ckpt = Gauge("train_last_checkpoint_step", "마지막으로 .done을 만든 step", base, registry=r)
        start_http_server(port, registry=r)
        self.on = True

    def stage(self, name: str) -> None:
        if self.on:
            for s in STAGES:
                self.stage_g.labels(**self.labels, stage=s).set(1 if s == name else 0)

    def step(self, step: int, loss: float, step_seconds: float, data_wait: float, phase: str) -> None:
        if self.on:
            self.step_seconds.labels(**self.labels, phase=phase).observe(step_seconds)
            self.data_wait.labels(**self.labels, phase=phase).observe(data_wait)
            self.current_step.labels(**self.labels).set(step)
            self.loss.labels(**self.labels).set(loss)

    def checkpoint(self, step: int) -> None:
        if self.on:
            self.last_ckpt.labels(**self.labels).set(step)

    def kernels(self, events) -> None:
        if self.on:
            comp, comm = split_kernel_seconds(events)
            self.compute.labels(**self.labels).set(comp)
            self.comm.labels(**self.labels).set(comm)


def start_profiler(metrics: Metrics, every: int):
    """N step마다 1 step만 CUDA 커널 시간을 잰다. 지표가 꺼졌거나 GPU가 없거나 N <= 2면 None."""
    import torch

    if not metrics.on or every <= 2 or not torch.cuda.is_available():
        return None
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile, schedule

    def ready(prof):
        metrics.kernels(e for e in prof.events() if e.device_type == DeviceType.CUDA)

    prof = profile(activities=[ProfilerActivity.CUDA], schedule=schedule(wait=every - 2, warmup=1, active=1),
                   on_trace_ready=ready, record_shapes=False, with_stack=False)
    prof.start()
    return prof


def error_line(e: BaseException, limit: int = 500) -> str:
    """실패 이유 한 줄 (GPU-188): progress.json error → sidecar job.progress → DB v_job_status.progress_error."""
    if isinstance(e, SystemExit):
        text = e.code if isinstance(e.code, str) else f"exit code {e.code}"
    else:
        text = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
    text = " ".join(str(text).split())  # 줄바꿈 · 연속 공백을 한 칸으로
    return text if len(text) <= limit else text[: limit - 1] + "…"


def read_resume(ckpt_dir: str) -> tuple[int, str | None]:
    """ckpt-init이 남긴 resume.json → (이어 갈 승인 step, 그 step 폴더). 승인본이 없으면 (0, None).

    resume.json이 아예 없으면 ckpt-init 없이 띄운 경우라 처음부터 한다.
    """
    path = os.path.join(ckpt_dir, "resume.json")
    if not os.path.exists(path):
        return 0, None
    with open(path) as f:
        info = json.load(f)
    if not info.get("resume"):
        return 0, None
    step = int(info["approved_step"])
    step_dir = os.path.join(ckpt_dir, info.get("step_dir") or step_name(step))
    if not os.path.isdir(step_dir):
        raise SystemExit(f"resume.json says {step_name(step)} but {step_dir} is missing")
    return step, step_dir


def resume_position(step: int, steps_per_epoch: int) -> tuple[int, int]:
    """step개를 마친 뒤의 데이터 위치 → (epoch, 그 epoch에서 건너뛸 배치 수)."""
    return step // steps_per_epoch, step % steps_per_epoch


def pick_keys(state: dict, keys) -> dict:
    """저장 · 재개할 모델 키만 고른다 (QLoRA처럼 학습하는 파라미터만 저장할 때). keys가 None이면 state 그대로.

    없는 키가 있으면 실패한다: 이름이 바뀌어 어댑터가 저장 · 재개에서 조용히 빠지는 일을 막는다.
    """
    if keys is None:
        return state
    missing = sorted(set(keys) - set(state))
    if missing:
        more = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
        raise SystemExit(f"model_keys not in state_dict: {', '.join(missing[:3])}{more}")
    return {k: state[k] for k in sorted(keys)}


def save_fsdp(step, model, optimizer, cfg_hash, step_dir, keys=None):
    # 모든 rank가 자기 조각을 쓴다. 모델 상태가 GPU마다 다르기 때문이다.
    # (torch 2.5.1의 get_state_dict는 조각이 0개인 rank가 있으면 실패해서 FSDP 자체 API를 쓴다)
    import torch.distributed.checkpoint as dcp
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state = {
            "model": pick_keys(model.state_dict(), keys),  # keys: 학습하는 파라미터만 (QLoRA 어댑터)
            # 옵티마이저는 Adam 모멘트(state)만. param_groups(lr · betas · weight_decay)는 코드 · 스케줄이 정하고,
            # 텐서 없는 목록이라 DCP가 저장 · 불러오기 때 키를 다르게 펼쳐 불러오기가 실패한다
            "optim": {"state": FSDP.optim_state_dict(model, optimizer)["state"]},
            "step": step,
            "config_sha256": cfg_hash,
        }
    dcp.save(state, checkpoint_id=step_dir)


def init_optimizer_state(optimizer) -> None:
    """학습 전 옵티마이저는 상태(Adam exp_avg · exp_avg_sq · step)가 비어 있다. 비어 있으면 불러올 칸이 없어서
    저장된 상태가 안 들어간다 → 학습률 0 · 기울기 0으로 한 번 step해 칸만 만든다 (가중치는 안 바뀜)."""
    import torch

    if optimizer.state:
        return
    lrs = [g["lr"] for g in optimizer.param_groups]
    for g in optimizer.param_groups:
        g["lr"] = 0.0
        for p in g["params"]:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
    optimizer.step()
    for g, lr in zip(optimizer.param_groups, lrs):
        g["lr"] = lr
    optimizer.zero_grad(set_to_none=True)


def load_fsdp(model, optimizer, step_dir, keys=None) -> tuple[int, str]:
    # 저장과 같은 모양의 빈 state를 만들고 DCP가 내 조각을 채운다.
    # keys가 있으면 그 키만 체크포인트에서 읽고, 나머지(얼린 기본 가중치)는 지금 값을 그대로 다시 넣는다.
    import torch.distributed.checkpoint as dcp
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType

    init_optimizer_state(optimizer)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        current = FSDP.optim_state_dict(model, optimizer)
        full = model.state_dict()
        state = {"model": pick_keys(full, keys), "optim": {"state": current["state"]},
                 "step": 0, "config_sha256": ""}
        dcp.load(state, checkpoint_id=step_dir)
        if keys is None:
            model.load_state_dict(state["model"])
        else:
            full.update(state["model"])
            # strict=False: 4bit 가중치의 양자화 상태(absmax 등)는 파라미터가 아니라 모듈이 따로 들고 있다
            missing = set(keys) & set(model.load_state_dict(full, strict=False).missing_keys)
            if missing:
                raise SystemExit(f"checkpoint did not fill {sorted(missing)[:3]}")
        osd = {"state": state["optim"]["state"], "param_groups": current["param_groups"]}
        optimizer.load_state_dict(FSDP.optim_state_dict_to_load(model, optimizer, osd))
    return int(state["step"]), str(state["config_sha256"])


def load_ddp(model, optimizer, step_dir, device, keys=None) -> tuple[int, str]:
    # rank 0이 쓴 파일 하나를 모든 rank가 읽는다 (ckpt-init이 manifest의 파일을 Pod마다 다 받아 둔다).
    import torch

    state = torch.load(os.path.join(step_dir, "checkpoint.pt"), map_location=device, weights_only=False)
    missing = model.module.load_state_dict(state["model"], strict=keys is None).missing_keys
    if keys is not None and set(keys) & set(missing):
        raise SystemExit(f"checkpoint did not fill {sorted(set(keys) & set(missing))[:3]}")
    optimizer.load_state_dict(state["optim"])
    return int(state["step"]), str(state["config_sha256"])


def load_checkpoint(mode, model, optimizer, step_dir, device, want_step, cfg_hash, keys=None) -> None:
    """승인본을 불러오고, 안에 적힌 step · 설정 해시가 맞는지 확인한다."""
    if mode == "fsdp":
        step, sha = load_fsdp(model, optimizer, step_dir, keys)
    else:
        step, sha = load_ddp(model, optimizer, step_dir, device, keys)
    if step != want_step:
        raise SystemExit(f"{step_dir} holds step {step}, resume.json says {want_step}")
    if sha and sha != cfg_hash:
        raise SystemExit(f"checkpoint config {sha} != current config {cfg_hash}")


def save_ddp(step, model, optimizer, cfg_hash, step_dir, rank, keys=None):
    # 모든 rank의 상태가 같으므로 rank 0만 파일 하나를 쓴다.
    import torch

    if rank != 0:
        return
    state = {
        "model": pick_keys(model.module.state_dict(), keys),
        "optim": optimizer.state_dict(),
        "step": step,
        "config_sha256": cfg_hash,
    }
    torch.save(state, os.path.join(step_dir, "checkpoint.pt"))


def save_checkpoint(mode, step, model, optimizer, cfg_hash, rank, ckpt_dir, keys=None) -> float:
    """CKPT_DIR/step-N/에 쓰고 .done을 마지막에 남긴다. 걸린 초를 돌려준다."""
    import torch.distributed as dist

    start = time.perf_counter()
    step_dir = os.path.join(ckpt_dir, step_name(step))
    os.makedirs(step_dir, exist_ok=True)
    if mode == "fsdp":
        save_fsdp(step, model, optimizer, cfg_hash, step_dir, keys)
    else:
        save_ddp(step, model, optimizer, cfg_hash, step_dir, rank, keys)
    dist.barrier()  # 모든 조각이 다 쓰인 뒤에만 .done
    with open(step_dir + ".done", "wb"):
        pass
    return time.perf_counter() - start


def stop_requested(ckpt_dir: str) -> bool:
    return os.path.exists(os.path.join(ckpt_dir, "STOP"))


def wait_approved(ckpt_dir: str, step: int, timeout: float) -> bool:
    marker = os.path.join(ckpt_dir, step_name(step) + ".approved")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(marker):
            return True
        if stop_requested(ckpt_dir):
            return False
        time.sleep(2)
    return os.path.exists(marker)



class Session:
    """학습 1회 동안 시스템 쪽 일을 맡는다. torch.distributed가 초기화된 뒤 모든 rank에서 같은 순서로 부른다.

    with 블록으로 쓰면 예외로 끝날 때 진행률에 FAILED와 실패 이유 한 줄을 남긴다 (STOP 75는 실패로 안 남김).
    report() 안에 all_reduce가 있어 모든 rank가 같은 step에 불러야 한다.
    """

    def __init__(self, *, mode: str, total_steps: int, cfg_hash: str = "", policy: CheckpointPolicy | None = None,
                 warmup_steps: int = 0, ckpt_dir: str | None = None, progress_file: str | None = None,
                 generation: int | None = None, metrics_port: int | None = None, profile_every: int | None = None,
                 model_keys=None, env=None):
        import torch.distributed as dist

        env = os.environ if env is None else env
        if mode not in CHECKPOINT_FORMATS:
            raise SystemExit(f"devoops_ckpt supports {sorted(CHECKPOINT_FORMATS)}, got {mode}")
        self.mode, self.total, self.cfg_hash, self.policy = mode, total_steps, cfg_hash, policy
        # 저장 · 재개할 모델 키 (None = 전부). QLoRA는 학습하는 어댑터 이름만 → 체크포인트에 4bit 기본 가중치가 안 들어간다
        self.model_keys = None if model_keys is None else frozenset(model_keys)
        self.warmup_steps = warmup_steps
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.ckpt_dir = ckpt_dir or env["CKPT_DIR"]
        path = progress_file if progress_file is not None else env.get("PROGRESS_FILE", "")
        gen = generation if generation is not None else int(env.get("GENERATION", "0"))
        self.progress = ProgressFile(path if self.rank == 0 else "", gen)
        port = metrics_port if metrics_port is not None else int(env.get("METRICS_PORT", "8000"))
        self.metrics = Metrics(metric_labels(env), port)        # 모든 rank (GPU-96)
        self.profile_every = profile_every if profile_every is not None else int(env.get("PROFILE_EVERY", PROFILE_EVERY))
        self.start = 0
        self.last_saved = 0
        fixed = policy.every_steps if policy else 0
        # 저장 간격(step). 고정 간격(GPU-298)이면 처음부터, 아니면 측정이 끝나기 전에는 None
        self.every: int | None = fixed if fixed > 0 else None
        self._measure_start: float | None = None
        self._measure_until = 0
        self._mark: float | None = None
        self._prof = None
        self.metrics.stage("init")
        self.progress.write(status="RUNNING", step=0, total_steps=total_steps, last_checkpoint_step=0)
        if self.every is not None:
            self.progress.write(checkpoint_every_steps=self.every)
            if self.rank == 0:
                print(f"fixed checkpoint interval: save every {self.every} steps", flush=True)

    # ---------- with ----------
    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop_profiler()
        if isinstance(exc, Stopped):
            self.progress.write(status="STOPPED")
        elif exc is not None:
            self.progress.write(status="FAILED", error=error_line(exc))
        return False                                            # 예외는 그대로 올려 보낸다 (종료 코드 유지)

    # ---------- 재개 ----------
    def load(self, model, optimizer, device=None) -> int:
        """ckpt-init이 받아 둔 승인본을 불러온다. 불러온 step(다음 학습은 step+1부터), 승인본이 없으면 0."""
        import torch

        start, step_dir = read_resume(self.ckpt_dir)
        if start:
            load_checkpoint(self.mode, model, optimizer, step_dir, device or torch.device("cpu"), start, self.cfg_hash,
                            self.model_keys)
            self.progress.write(step=start, last_checkpoint_step=start)
            self.metrics.checkpoint(start)
            if self.rank == 0:
                print(f"resumed from {step_name(start)}", flush=True)
        self.start = self.last_saved = start
        warm = self.policy.warmup_measure_steps if self.policy else 1
        self._measure_until = min(start + max(1, warm), self.total)
        return start

    @staticmethod
    def data_position(start: int, steps_per_epoch: int) -> tuple[int, int]:
        """start step까지 마친 뒤의 데이터 위치 → (epoch, 그 epoch에서 건너뛸 배치 수).

        DistributedSampler(seed 고정) + 매 epoch sampler.set_epoch(epoch)이면 순서가 같아 끊김 없이 이어진다.
        """
        return resume_position(start, steps_per_epoch)

    # ---------- 학습 중 ----------
    def report(self, step: int, loss, data_wait: float | None = None) -> None:
        """step 하나가 끝날 때마다 모든 rank가 부른다: loss 평균 · 진행률 · 지표 · 저장 간격 측정 · 펜싱 STOP."""
        import torch
        import torch.distributed as dist

        now = time.perf_counter()
        if self._mark is None:                                  # 첫 report: 학습 단계 시작
            self.metrics.stage("train")
            self._prof = start_profiler(self.metrics, self.profile_every)
        step_seconds = now - self._mark if self._mark is not None else 0.0
        if isinstance(loss, torch.Tensor):                      # 화면에는 전체 rank 평균
            reduced = loss.detach().float().reshape(1).clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            loss_value = float(reduced.item()) / self.world
        else:
            loss_value = float(loss)
        values = {"step": step, "loss": loss_value}
        if self._mark is not None:
            self.metrics.step(step, loss_value, step_seconds, data_wait or 0.0, phase_of(step, self.warmup_steps))

        # 저장 간격: 첫 step(CUDA 초기화 · 컴파일)을 뺀 warmup_measure_steps의 평균 step 시간 → Young/Daly τ를 step으로
        if step == self.start + 1:
            _sync()
            self._measure_start = time.perf_counter()
        elif self.every is None and step >= self._measure_until and self._measure_start is not None:
            _sync()
            per = (time.perf_counter() - self._measure_start) / max(1, step - self.start - 1)
            t = torch.tensor([per], device=_device_of(dist))
            dist.all_reduce(t, op=dist.ReduceOp.MAX)            # 모든 rank가 같은 간격을 쓴다
            per = float(t.item())
            interval = self.policy.interval_seconds if self.policy else 0
            self.every, count = checkpoint_every(self.total, per, interval)
            values.update(checkpoint_every_steps=self.every, checkpoint_count=count, step_seconds=round(per, 4))
            if self.rank == 0:
                print(f"step time {per:.3f}s, expected {self.total * per / 60:.1f} min, interval {interval:.0f}s "
                      f"-> save every {self.every} steps ({count} saves)", flush=True)

        self.progress.write(**values)
        if self._prof is not None:
            self._prof.step()
        if stop_anywhere(dist, stop_requested(self.ckpt_dir)):   # 펜싱 STOP: 한 rank라도 받으면 모두 같은 step에서 멈춤
            print(f"rank {self.rank}: STOP at step {step}", flush=True)
            raise Stopped(step)
        self._mark = time.perf_counter()

    def should_save(self, step: int) -> bool:
        """이번 step에 저장할지. 마지막 step은 항상, 그 밖에는 고정 간격 또는 측정한 간격마다 (측정 전에는 안 함)."""
        return step == self.total or (self.every is not None and step % self.every == 0)

    def save(self, model, optimizer, step: int) -> float:
        """CKPT_DIR/step-N/에 저장하고 .done을 남긴다 (모든 rank). 걸린 초."""
        self.metrics.stage("checkpoint")                        # 저장 중 step이 안 오르는 걸 정체로 보지 않게
        took = save_checkpoint(self.mode, step, model, optimizer, self.cfg_hash, self.rank, self.ckpt_dir,
                               self.model_keys)
        self.metrics.checkpoint(step)
        self.metrics.stage("train")
        self.last_saved = step
        self.progress.write(last_checkpoint_step=step)
        if self.rank == 0:
            print(f"step {step}: saved {step_name(step)} in {took:.1f}s", flush=True)
        self._mark = time.perf_counter()                        # 저장 시간이 다음 step 시간에 섞이지 않게
        return took

    # ---------- 끝 ----------
    def finish(self, timeout: float = APPROVAL_WAIT_SECONDS) -> None:
        """마지막 저장이 승인(.approved)될 때까지 기다린 뒤 COMPLETED. 못 받으면 실패로 끝낸다."""
        import torch.distributed as dist

        self._stop_profiler()
        self.metrics.stage("finalize")
        state = wait_approved_all(dist, self.ckpt_dir, self.last_saved, timeout)
        if state == "stopped":
            print(f"rank {self.rank}: STOP while waiting for {step_name(self.last_saved)}", flush=True)
            raise Stopped(self.last_saved)
        if state != "approved":
            raise SystemExit(f"{step_name(self.last_saved)} not approved within {timeout:.0f}s")
        self.progress.write(status="COMPLETED")

    def _stop_profiler(self) -> None:
        if self._prof is not None:
            self._prof.stop()
            self._prof = None


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def stop_anywhere(dist, local: bool) -> bool:
    """이 rank의 STOP 여부를 모든 rank에서 모은다 (MAX). STOP 파일은 Pod마다 따로라
    rank 0만 멈추면 나머지는 NCCL 오류로 죽는다 (11/20 펜싱 시험, GPU-244)."""
    import torch

    t = torch.tensor([1.0 if local else 0.0], device=_device_of(dist))
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item() > 0


def wait_approved_all(dist, ckpt_dir: str, step: int, timeout: float, poll: float = 2.0) -> str:
    """마지막 저장을 모든 rank가 같이 기다린다: "approved"(모든 rank에 .approved) · "stopped"(어느 rank든 STOP) ·
    "timeout". 매 회 all_reduce로 판단을 맞춰서 모든 rank가 같은 결과로 끝난다."""
    import torch

    marker = os.path.join(ckpt_dir, step_name(step) + ".approved")
    deadline = time.monotonic() + timeout
    device = _device_of(dist)
    while True:
        t = torch.tensor([0.0 if os.path.exists(marker) else 1.0,          # 아직 안 된 rank가 있나
                          1.0 if stop_requested(ckpt_dir) else 0.0,         # STOP 받은 rank가 있나
                          1.0 if time.monotonic() >= deadline else 0.0],   # 시간 다 된 rank가 있나
                         device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        pending, stop, late = (v > 0 for v in t.tolist())
        if stop:
            return "stopped"
        if not pending:
            return "approved"
        if late:
            return "timeout"
        time.sleep(poll)


def _device_of(dist):
    """all_reduce에 쓸 장치: NCCL이면 현재 GPU, 그 밖(gloo)은 CPU."""
    import torch

    if dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


# ---------- 루프 소유형: fit() (GPU-209) ----------
# 사용자는 모델 · 데이터셋 · loss 함수만 준다. 분산 감싸기 · DataLoader · 학습률 · 재개 · 저장 · 진행률 · 지표 ·
# STOP · 마지막 승인 대기 · 실패 기록은 전부 fit()이 한다. 루프를 직접 짜야 하면(gradient accumulation 등) Session을 쓴다.

def total_steps(steps_per_epoch: int, epochs: int, max_steps: int | None) -> int:
    total = steps_per_epoch * epochs
    return min(total, max_steps) if max_steps is not None else total


def lr_at(step: int, base: float, warmup: int, total: int) -> float:
    """기본 학습률: warmup 동안 선형으로 올리고, 그 뒤 cosine으로 base의 10%까지 내린다."""
    if warmup > 0 and step <= warmup:
        return base * step / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def wrap_model(mode: str, net, device):
    """config.training_mode대로 감싼다. FSDP는 모델의 _no_split_modules(HF 트랜스포머 블록) 단위, 없으면 통째로."""
    gpu = device.type == "cuda"
    if mode == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        return DDP(net.to(device), device_ids=[device.index] if gpu else None)

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    names = set(getattr(net, "_no_split_modules", None) or [])
    blocks = {type(m) for m in net.modules() if type(m).__name__ in names}
    policy = ModuleWrapPolicy(blocks) if blocks else None
    # use_orig_params: 옵티마이저와 체크포인트가 원래 파라미터 이름을 그대로 쓴다
    return FSDP(net, auto_wrap_policy=policy, device_id=device, use_orig_params=True)


def clip_gradients(mode: str, model, max_norm: float) -> None:
    import torch

    if mode == "ddp":
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    else:
        model.clip_grad_norm_(max_norm)  # FSDP는 조각을 모아 전체 노름을 계산한다


def to_device(batch, device):
    """배치(텐서 · tuple · list · dict)를 device로. 원소가 하나인 tuple/list(TensorDataset)는 풀어서 준다."""
    import torch

    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [to_device(v, device) for v in batch]
        return moved[0] if len(moved) == 1 else (tuple(moved) if isinstance(batch, tuple) else moved)
    return batch


def _local_device():
    import torch

    if torch.cuda.is_available():
        index = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(index)
        return torch.device("cuda", index)
    return torch.device("cpu")


def fit(*, model, dataset, loss_fn, optimizer=None, lr_schedule=None, max_grad_norm: float | None = 1.0,
        collate_fn=None, config: tuple[TrainConfig, str] | None = None, **session_kwargs):
    """학습 전체를 돈다. 모든 rank에서 같은 인자로 부른다. 감싼 모델을 돌려준다.

    model         nn.Module, 또는 그걸 돌려주는 인자 없는 함수 (함수면 fit 안에서 만들어 실패가 FAILED로 남는다)
    dataset       torch Dataset, 또는 그걸 돌려주는 인자 없는 함수
    loss_fn       loss_fn(model, batch) -> loss 텐서. batch는 이미 GPU에 있다 (원소 하나인 tuple은 풀어서)
    optimizer     optimizer(params) -> Optimizer. 없으면 AdamW(lr=config.learning_rate, weight_decay=0.1)
    lr_schedule   lr_schedule(step, total) -> lr. 없으면 lr_at(warmup → cosine). None을 돌려주면 그 step은 안 바꿈
    max_grad_norm gradient clip. None이면 안 함
    collate_fn    DataLoader에 그대로
    config        (TrainConfig, cfg_hash). 없으면 load_config() (/code/config.json)
    session_kwargs Session에 그대로 (ckpt_dir · progress_file · generation · metrics_port · profile_every · env, 테스트용)

    config에서 쓰는 것: training_mode · world_size · seed · batch_size_per_gpu · epochs · max_steps ·
    learning_rate · warmup_steps · checkpoint_policy
    """
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler

    cfg, cfg_hash = config if config is not None else load_config()
    own_group = not dist.is_initialized()
    if own_group:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        # total을 알기 전에 실패해도 FAILED가 남도록 먼저 0으로 연다 (데이터를 읽은 뒤 바로 고친다)
        with Session(mode=cfg.training_mode, total_steps=0, cfg_hash=cfg_hash, policy=cfg.checkpoint_policy,
                     warmup_steps=cfg.warmup_steps, **session_kwargs) as s:
            if world != cfg.world_size:
                raise SystemExit(f"WORLD_SIZE {world} != config.world_size {cfg.world_size}")
            if cfg.finetune.method != "full":   # 화면에서 고른 방식과 다르게 돌지 않게
                raise SystemExit(f"fit() trains every parameter (full); this train.py does not support "
                                 f"{cfg.finetune.method}. Use a train.py that reads cfg.finetune "
                                 f"(trainer/examples/fsdp_lm_train.py)")
            device = _local_device()
            torch.manual_seed(cfg.seed)

            if callable(dataset) and not hasattr(dataset, "__getitem__"):
                dataset = dataset()
            if not isinstance(model, torch.nn.Module):
                model = model()

            sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=cfg.seed, drop_last=True)
            loader = DataLoader(dataset, batch_size=cfg.batch_size_per_gpu, sampler=sampler, drop_last=True,
                                collate_fn=collate_fn)
            steps_per_epoch = len(loader)
            if steps_per_epoch == 0:
                raise SystemExit(f"{len(dataset)} samples is less than one batch per GPU "
                                 f"({cfg.batch_size_per_gpu} x {world})")
            total = total_steps(steps_per_epoch, cfg.epochs, cfg.max_steps)
            s.total = total
            s.progress.write(total_steps=total)
            if rank == 0:
                print(f"{cfg.training_mode} x {world}: {len(dataset)} samples, {steps_per_epoch} steps/epoch, "
                      f"{total} steps", flush=True)

            wrapped = wrap_model(cfg.training_mode, model, device)
            wrapped.train()
            opt = (optimizer(wrapped.parameters()) if optimizer is not None
                   else torch.optim.AdamW(wrapped.parameters(), lr=cfg.learning_rate, weight_decay=0.1))
            if lr_schedule is None:
                def lr_schedule(step, n):
                    return lr_at(step, cfg.learning_rate, cfg.warmup_steps, n)

            start = s.load(wrapped, opt, device)               # 승인본이 있으면 그 step까지
            epoch, skip = s.data_position(start, steps_per_epoch)
            step = start
            mark = time.perf_counter()
            while step < total:
                sampler.set_epoch(epoch)                        # 같은 seed · epoch → 같은 순서 (재개해도 이어짐)
                for batch in loader:
                    if skip:                                    # 재개한 epoch에서 이미 학습한 배치
                        skip -= 1
                        mark = time.perf_counter()
                        continue
                    step += 1
                    lr = lr_schedule(step, total)
                    if lr is not None:
                        for group in opt.param_groups:
                            group["lr"] = lr
                    data_wait = time.perf_counter() - mark
                    batch = to_device(batch, device)
                    opt.zero_grad(set_to_none=True)
                    loss = loss_fn(wrapped, batch)
                    loss.backward()
                    if max_grad_norm is not None:
                        clip_gradients(cfg.training_mode, wrapped, max_grad_norm)
                    opt.step()

                    s.report(step, loss, data_wait)             # 진행률 · 지표 · 저장 간격 · 펜싱 STOP
                    if s.should_save(step):                     # 마지막 step은 항상
                        s.save(wrapped, opt, step)
                    mark = time.perf_counter()
                    if step >= total:
                        break
                epoch += 1
            s.finish()                                          # 마지막 저장이 승인될 때까지 (최대 120초)
            return wrapped
    finally:
        if own_group and dist.is_initialized():
            dist.destroy_process_group()
