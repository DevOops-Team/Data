"""ckpt-init 다운로드 감시 (GPU-263, GPU-256 후속): 시간 안에 못 끝날 다운로드를 1분 안에 알아챈다.

왜: EKS Kueue waitForPodsReady는 입장 뒤 20분(11/23 GPU-264로 10분 → 20분) 안에 Pod 전부 Ready를 요구한다.
그 안에 ckpt-init 다운로드 · trainer 이미지 pull · 시작이 모두 들어간다. 다운로드가 길면 Kueue가 이유 없이 내린다
→ 감시기 job.evicted(PodsReadyTimeout)만 남고 원인은 안 보인다. 그래서 ckpt-init이 도중에 스스로 판단한다.

판정 (5초마다). 예상 남은 시간 eta = 남은 바이트 ÷ 지금까지 속도, 다운로드 시작 WARMUP_SECONDS(30) 뒤부터
  slow      eta > 여유 뺀 예산(DOWNLOAD_BUDGET_S 1020 − init 시작부터 지난 시간)  → 경고만, 다운로드는 계속
            (1020 = Kueue 대기 1200 − 이미지 pull · 시작 여유 180). 시간 안에 끝날 수도 있으므로 죽이지 않는다
  too_slow  eta > Kueue 한계(KUEUE_READY_TIMEOUT_S 1200 − 지난 시간)                → 실패
            다운로드만으로 Kueue 한계를 넘으므로 실패가 확실. Kueue가 최대 3번 다시 넣으며 GPU를 잡는 것을 막는다
  Kueue 값(Gitops eks/scheduling/base/kueue-values.yaml)을 바꾸면 이 두 기본값도 같이 바꾼다
  stalled   STALL_SECONDS(60) 동안 받은 바이트가 0 — S3 · 네트워크가 멈춤          → 실패

큰 모델을 살리지는 못한다 — 다운로드가 Kueue 한계를 넘는 모델은 어차피 실패하고, 그걸 1분 안에 이유와 함께 알릴 뿐.
큰 모델이 성공하려면 Kueue 대기 시간을 늘린다(GPU-256 → 11/23 20분). 다운로드 130MB/s 기준 20분이면 약 130GB까지.

실패면 init.download_too_slow · init.download_stalled로 job.init_failed를 보내고 init을 바로 끝낸다(종료 코드 4).
boto3 download_file은 끝날 때까지 블로킹이라 다운로드를 끊는 대신 프로세스를 끝낸다 (init 컨테이너라 안전).
경고(slow)는 job.init_slow 사건을 한 번 보내고 계속 받는다.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable

from .log import log


class DownloadWatch:
    """받은 바이트를 세고 too_slow · stalled를 판정한다. add()는 boto3 Callback(여러 스레드)에서 불린다."""

    def __init__(self, *, init_started: float, budget_s: float | None = None, hard_limit_s: float | None = None,
                 warmup_s: float | None = None, stall_s: float | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.init_started = init_started
        self.budget_s = budget_s if budget_s is not None else float(os.environ.get("DOWNLOAD_BUDGET_S", "1020"))
        self.hard_limit_s = (hard_limit_s if hard_limit_s is not None
                             else float(os.environ.get("KUEUE_READY_TIMEOUT_S", "1200")))
        self.warmup_s = warmup_s if warmup_s is not None else float(os.environ.get("DOWNLOAD_WARMUP_SECONDS", "30"))
        self.stall_s = stall_s if stall_s is not None else float(os.environ.get("DOWNLOAD_STALL_SECONDS", "60"))
        self._lock = threading.Lock()
        self.label: str | None = None
        self.total = 0
        self.done = 0
        self.phase_started: float | None = None
        self.last_progress: float | None = None

    def begin(self, label: str, total_bytes: int) -> None:
        """다운로드 단계 시작 (학습 데이터 · 승인본). 단계마다 속도를 새로 잰다."""
        with self._lock:
            now = self.clock()
            self.label, self.total, self.done = label, int(total_bytes), 0
            self.phase_started = self.last_progress = now

    def end(self) -> None:
        with self._lock:
            self.label = None

    def add(self, n: int) -> None:
        if n <= 0:
            return
        with self._lock:
            self.done += n
            self.last_progress = self.clock()

    def check(self) -> tuple[str, dict] | None:
        """판정 결과 (reason, detail) 또는 None."""
        with self._lock:
            if self.label is None or self.phase_started is None:
                return None
            now = self.clock()
            elapsed = now - self.phase_started
            since_init = now - self.init_started
            base = {"phase": self.label, "total_bytes": self.total, "done_bytes": self.done,
                    "elapsed_s": round(elapsed, 1), "since_init_s": round(since_init, 1),
                    "budget_s": self.budget_s, "hard_limit_s": self.hard_limit_s}
            if now - (self.last_progress or now) >= self.stall_s:
                return "stalled", dict(base, stalled_s=round(now - self.last_progress, 1))
            if elapsed < self.warmup_s or self.done <= 0 or self.done >= self.total:
                return None
            rate = self.done / elapsed
            eta = (self.total - self.done) / rate
            info = dict(base, rate_mb_s=round(rate / 1e6, 2), eta_s=round(eta, 1))
            if eta > self.hard_limit_s - since_init:
                return "too_slow", info          # 다운로드만으로 Kueue 한계를 넘음 → 실패 확실
            if eta > self.budget_s - since_init:
                return "slow", info              # 이미지 pull · 시작 여유를 못 남김 → 경고만
            return None


FAIL_REASONS = ("too_slow", "stalled")


class Watchdog:
    """DownloadWatch를 poll_s(5초)마다 확인. 실패(too_slow · stalled)면 on_trigger를 한 번 부르고 멈춤,
    경고(slow)면 on_warn을 단계마다 한 번 부르고 계속 본다."""

    def __init__(self, watch: DownloadWatch, on_trigger: Callable[[str, dict], None],
                 on_warn: Callable[[str, dict], None] | None = None, poll_s: float = 5.0):
        self.watch, self.on_trigger, self.on_warn, self.poll_s = watch, on_trigger, on_warn, poll_s
        self._stop = threading.Event()
        self.fired: tuple[str, dict] | None = None
        self.warned: set[str] = set()   # 경고를 보낸 단계
        self._thread = threading.Thread(target=self._run, name="download-watchdog", daemon=True)

    def start(self) -> "Watchdog":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def poll_once(self) -> bool:
        if self.fired is not None:
            return True
        res = self.watch.check()
        if res is None:
            return False
        reason, detail = res
        if reason not in FAIL_REASONS:
            if detail["phase"] not in self.warned:
                self.warned.add(detail["phase"])
                log("init.download_slow", **detail)
                if self.on_warn is not None:
                    self.on_warn(reason, detail)
            return False
        self.fired = res
        log("init.download_watch", reason=reason, **detail)
        self.on_trigger(reason, detail)
        return True

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            if self.poll_once():
                return
