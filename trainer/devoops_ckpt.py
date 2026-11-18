"""devoops_ckpt — 학습 코드가 시스템과 맞물리는 부분을 대신 해 주는 라이브러리 (GPU-209).

사용자 train.py는 학습만 하고, 재개 · 저장 규약 · 진행률 · 지표 · 펜싱 STOP · 마지막 승인 대기 · 실패 기록은 여기서 한다.
trainer 이미지에 들어 있다 (/opt/train, PYTHONPATH). 안 써도 되지만, 그러면 아래 규약을 직접 지켜야 저장 · 재개가 된다.

    import devoops_ckpt as ckpt

    cfg, cfg_hash = ckpt.load_config()                      # /code/config.json (Backend가 만든 학습 설정)
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
  - 중단(이행 기간): CKPT_DIR/STOP이 생기면 저장 없이 종료 코드 75. 펜싱은 relocator가 옛 JobSet을 지워서도 멈춘다

환경 변수 (Backend jobspec이 넣음): TRAIN_CONFIG · CONFIG_SHA256 · CKPT_DIR · PROGRESS_FILE · GENERATION · JOB_ID ·
  METRICS_PORT(8000) · PROFILE_EVERY(25)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, fields

SCHEMA_VERSION = 2


CHECKPOINT_FORMATS = {"fsdp": "fsdp-dcp", "ddp": "ddp-pt"}


STOP_EXIT = 75  # ckpt-sidecar가 STOP을 썼을 때. 장애로 분류하지 않는다


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


def parse_config(raw: bytes, expected_sha: str | None) -> tuple[TrainConfig, str]:
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha and expected_sha != digest:
        raise SystemExit(f"config hash mismatch: {digest} != {expected_sha}")

    data = json.loads(raw)
    if data.pop("schema_version", None) != SCHEMA_VERSION:
        raise SystemExit(f"config schema_version must be {SCHEMA_VERSION}")
    names = {f.name for f in fields(TrainConfig)}
    if set(data) != names:  # 오타 · 누락을 조용히 넘기지 않는다
        raise SystemExit(f"config keys: missing {names - set(data)}, unknown {set(data) - names}")
    data["dataset"] = DatasetRef(**data["dataset"])
    data["checkpoint_policy"] = CheckpointPolicy(**data["checkpoint_policy"])
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


STEP_BUCKETS = (0.1, 0.3, 1, 3, 10, 30)  # 초. Monitoring docs/training-metrics.md 3절


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
                                   buckets=STEP_BUCKETS, registry=r)
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


def save_fsdp(step, model, optimizer, cfg_hash, step_dir):
    # 모든 rank가 자기 조각을 쓴다. 모델 상태가 GPU마다 다르기 때문이다.
    # (torch 2.5.1의 get_state_dict는 조각이 0개인 rank가 있으면 실패해서 FSDP 자체 API를 쓴다)
    import torch.distributed.checkpoint as dcp
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state = {
            "model": model.state_dict(),
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


def load_fsdp(model, optimizer, step_dir) -> tuple[int, str]:
    # 저장과 같은 모양의 빈 state를 만들고 DCP가 내 조각을 채운다.
    import torch.distributed.checkpoint as dcp
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType

    init_optimizer_state(optimizer)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        current = FSDP.optim_state_dict(model, optimizer)
        state = {"model": model.state_dict(), "optim": {"state": current["state"]},
                 "step": 0, "config_sha256": ""}
        dcp.load(state, checkpoint_id=step_dir)
        model.load_state_dict(state["model"])
        osd = {"state": state["optim"]["state"], "param_groups": current["param_groups"]}
        optimizer.load_state_dict(FSDP.optim_state_dict_to_load(model, optimizer, osd))
    return int(state["step"]), str(state["config_sha256"])


def load_ddp(model, optimizer, step_dir, device) -> tuple[int, str]:
    # rank 0이 쓴 파일 하나를 모든 rank가 읽는다 (ckpt-init이 manifest의 파일을 Pod마다 다 받아 둔다).
    import torch

    state = torch.load(os.path.join(step_dir, "checkpoint.pt"), map_location=device, weights_only=False)
    model.module.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optim"])
    return int(state["step"]), str(state["config_sha256"])


def load_checkpoint(mode, model, optimizer, step_dir, device, want_step, cfg_hash) -> None:
    """승인본을 불러오고, 안에 적힌 step · 설정 해시가 맞는지 확인한다."""
    if mode == "fsdp":
        step, sha = load_fsdp(model, optimizer, step_dir)
    else:
        step, sha = load_ddp(model, optimizer, step_dir, device)
    if step != want_step:
        raise SystemExit(f"{step_dir} holds step {step}, resume.json says {want_step}")
    if sha and sha != cfg_hash:
        raise SystemExit(f"checkpoint config {sha} != current config {cfg_hash}")


def save_ddp(step, model, optimizer, cfg_hash, step_dir, rank):
    # 모든 rank의 상태가 같으므로 rank 0만 파일 하나를 쓴다.
    import torch

    if rank != 0:
        return
    state = {
        "model": model.module.state_dict(),
        "optim": optimizer.state_dict(),
        "step": step,
        "config_sha256": cfg_hash,
    }
    torch.save(state, os.path.join(step_dir, "checkpoint.pt"))


def save_checkpoint(mode, step, model, optimizer, cfg_hash, rank, ckpt_dir) -> float:
    """CKPT_DIR/step-N/에 쓰고 .done을 마지막에 남긴다. 걸린 초를 돌려준다."""
    import torch.distributed as dist

    start = time.perf_counter()
    step_dir = os.path.join(ckpt_dir, step_name(step))
    os.makedirs(step_dir, exist_ok=True)
    if mode == "fsdp":
        save_fsdp(step, model, optimizer, cfg_hash, step_dir)
    else:
        save_ddp(step, model, optimizer, cfg_hash, step_dir, rank)
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
                 env=None):
        import torch.distributed as dist

        env = os.environ if env is None else env
        if mode not in CHECKPOINT_FORMATS:
            raise SystemExit(f"devoops_ckpt supports {sorted(CHECKPOINT_FORMATS)}, got {mode}")
        self.mode, self.total, self.cfg_hash, self.policy = mode, total_steps, cfg_hash, policy
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
        self.every: int | None = None                           # 저장 간격(step). 측정이 끝나기 전에는 None
        self._measure_start: float | None = None
        self._measure_until = 0
        self._mark: float | None = None
        self._prof = None
        self.metrics.stage("init")
        self.progress.write(status="RUNNING", step=0, total_steps=total_steps, last_checkpoint_step=0)

    # ---------- with ----------
    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop_profiler()
        if exc is not None and not (isinstance(exc, SystemExit) and exc.code == STOP_EXIT):
            self.progress.write(status="FAILED", error=error_line(exc))
        return False                                            # 예외는 그대로 올려 보낸다 (종료 코드 유지)

    # ---------- 재개 ----------
    def load(self, model, optimizer, device=None) -> int:
        """ckpt-init이 받아 둔 승인본을 불러온다. 불러온 step(다음 학습은 step+1부터), 승인본이 없으면 0."""
        import torch

        start, step_dir = read_resume(self.ckpt_dir)
        if start:
            load_checkpoint(self.mode, model, optimizer, step_dir, device or torch.device("cpu"), start, self.cfg_hash)
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
        if stop_requested(self.ckpt_dir):                       # 이행 기간: 사이드카가 펜싱으로 STOP을 쓰면 바로 멈춤
            print(f"rank {self.rank}: STOP at step {step}", flush=True)
            sys.exit(STOP_EXIT)
        self._mark = time.perf_counter()

    def should_save(self, step: int) -> bool:
        """이번 step에 저장할지. 마지막 step은 항상, 그 밖에는 측정한 간격마다 (측정 전에는 안 함)."""
        return step == self.total or (self.every is not None and step % self.every == 0)

    def save(self, model, optimizer, step: int) -> float:
        """CKPT_DIR/step-N/에 저장하고 .done을 남긴다 (모든 rank). 걸린 초."""
        self.metrics.stage("checkpoint")                        # 저장 중 step이 안 오르는 걸 정체로 보지 않게
        took = save_checkpoint(self.mode, step, model, optimizer, self.cfg_hash, self.rank, self.ckpt_dir)
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
        if not wait_approved(self.ckpt_dir, self.last_saved, timeout):
            if stop_requested(self.ckpt_dir):
                sys.exit(STOP_EXIT)
            raise SystemExit(f"{step_name(self.last_saved)} not approved within {timeout:.0f}s")
        dist.barrier()
        self.progress.write(status="COMPLETED")

    def _stop_profiler(self) -> None:
        if self._prof is not None:
            self._prof.stop()
            self._prof = None


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _device_of(dist):
    """all_reduce에 쓸 장치: NCCL이면 현재 GPU, 그 밖(gloo)은 CPU."""
    import torch

    if dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")
