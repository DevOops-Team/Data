"""ckpt-init 다운로드 감시 (GPU-263): 시간 안에 못 끝날 다운로드 · 멈춘 다운로드를 빨리 알아채는지."""
from __future__ import annotations

import os

from ckpt.download_watch import DownloadWatch, Watchdog

MB = 1_000_000


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def _watch(clock, **kw):
    kw.setdefault("budget_s", 420)
    kw.setdefault("hard_limit_s", 600)
    return DownloadWatch(init_started=0.0, warmup_s=30, stall_s=60, clock=clock, **kw)


def test_fast_download_is_fine():
    c = Clock()
    w = _watch(c)
    w.begin("checkpoint", 1000 * MB)
    c.t = 30
    w.add(300 * MB)                                   # 10MB/s → 남은 700MB는 70초, 여유 충분
    assert w.check() is None


def test_slow_but_can_finish_before_kueue_is_only_a_warning():
    c = Clock()
    w = _watch(c)
    w.begin("checkpoint", 5000 * MB)
    c.t = 30
    w.add(300 * MB)                                   # 10MB/s → 남은 4700MB는 470초: 420 예산은 넘고 600 한계 안
    reason, d = w.check()
    assert reason == "slow" and d["eta_s"] == 470.0 and d["rate_mb_s"] == 10.0


def test_cannot_finish_before_kueue_fails_fast():
    """30GB를 40MB/s로 → 12.5분. Kueue 10분 안에 불가 → 30초 시점에 실패 (지금은 10분 뒤 이유 없이 내림)."""
    c = Clock()
    w = _watch(c)
    w.begin("checkpoint", 30_000 * MB)
    c.t = 30
    w.add(1200 * MB)                                  # 40MB/s
    reason, d = w.check()
    assert reason == "too_slow" and d["eta_s"] == 720.0 and d["since_init_s"] == 30.0
    assert d["total_bytes"] == 30_000 * MB and d["done_bytes"] == 1200 * MB


def test_time_spent_before_download_counts_against_the_limit():
    c = Clock(t=300)                                  # 세대 확인 · 데이터셋에 이미 5분
    w = DownloadWatch(init_started=0.0, budget_s=420, hard_limit_s=600, warmup_s=30, stall_s=60, clock=c)
    w.begin("checkpoint", 4000 * MB)
    c.t = 330
    w.add(300 * MB)                                   # 10MB/s → 370초 > 한계 남은 270초
    assert w.check()[0] == "too_slow"


def test_no_judgement_during_warmup():
    c = Clock()
    w = _watch(c)
    w.begin("checkpoint", 30_000 * MB)
    c.t = 10
    w.add(10 * MB)                                    # 시작 직후 느린 건 판단하지 않음
    assert w.check() is None


def test_stalled_download():
    c = Clock()
    w = _watch(c)
    w.begin("checkpoint", 150 * MB)                   # 남은 양이 적어 속도로는 문제없음 — 멈춤만 본다
    c.t = 20
    w.add(100 * MB)
    c.t = 79
    assert w.check() is None                          # 59초 동안 0바이트 — 아직
    c.t = 80
    reason, d = w.check()
    assert reason == "stalled" and d["stalled_s"] == 60.0


def test_nothing_after_phase_end():
    c = Clock()
    w = _watch(c)
    w.begin("dataset", 1000 * MB)
    w.end()
    c.t = 500
    assert w.check() is None


def test_watchdog_warns_once_per_phase_and_fails_once():
    c = Clock()
    w = _watch(c)
    warns, fails = [], []
    dog = Watchdog(w, on_trigger=lambda r, d: fails.append(r), on_warn=lambda r, d: warns.append(d["phase"]))
    w.begin("checkpoint", 5000 * MB)
    c.t = 30
    w.add(300 * MB)
    assert dog.poll_once() is False and dog.poll_once() is False
    assert warns == ["checkpoint"] and fails == []    # 경고는 한 번, 다운로드는 계속
    c.t = 200                                         # 그 사이 0바이트 170초 → 멈춤
    assert dog.poll_once() is True and dog.poll_once() is True
    assert fails == ["stalled"]                       # 실패는 한 번


def test_store_reports_bytes_to_watch(store, tmp_path):
    """boto3 Callback으로 받은 바이트가 감시에 들어가는지 (moto)."""
    data = os.urandom(3 * MB)
    import hashlib
    store.s3.put_object(Bucket=store.bucket, Key="k/blob", Body=data)
    c = Clock()
    w = _watch(c)
    w.begin("checkpoint", len(data))
    store.progress = w.add
    store.download_verified("k/blob", tmp_path / "blob", hashlib.sha256(data).hexdigest(), len(data))
    assert w.done == len(data)
