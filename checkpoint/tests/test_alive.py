"""사이드카 생존 파일 (ADR-44, GPU-213): liveness probe가 볼 시각."""
import os
import time

from ckpt.sidecar import ALIVE_FILE, touch_alive


def test_default_path_is_what_the_probe_reads():
    assert ALIVE_FILE == "/tmp/sidecar-alive"


def test_touch_creates_and_refreshes(tmp_path):
    p = tmp_path / "alive"
    touch_alive(str(p))
    assert p.exists()
    old = time.time() - 600
    os.utime(p, (old, old))                 # 10분 전으로 돌려 둠 = probe 기준 멈춤
    touch_alive(str(p))
    assert time.time() - p.stat().st_mtime < 5


def test_empty_path_is_off(tmp_path):
    touch_alive("")                           # 아무것도 안 함, 예외 없음
    assert list(tmp_path.iterdir()) == []


def test_unwritable_path_does_not_kill_the_sidecar(tmp_path):
    touch_alive(str(tmp_path / "no-such-dir" / "alive"))   # 로그만 남기고 계속
