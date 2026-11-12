"""ckpt/progress.py — PROGRESS_FILE → Kafka job.progress (GPU-141)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "collector"))      # 수집기 envelope.parse로 형식 호환 확인

from ckpt.progress import ProgressReporter, progress_event_id  # noqa: E402


class Pub:
    def __init__(self):
        self.sent = []

    def publish(self, *, topic, key, value):
        self.sent.append((topic, key, value))

    def flush(self, timeout=5.0):
        return 0


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / "progress.json"
    pub, clock = Pub(), Clock()
    rep = ProgressReporter(path, job_id="3f9a1c2e", generation=2, cluster="eks-a", publisher=pub,
                           interval_seconds=30, clock=clock)
    return path, pub, clock, rep


def write(path, **v):
    path.write_text(json.dumps({"generation": 2, "status": "RUNNING", **v}))


def test_no_file_no_event(setup):
    _, pub, _, rep = setup
    assert rep.tick() is False and pub.sent == []


def test_first_value_sent_with_collector_format(setup):
    path, pub, _, rep = setup
    write(path, step=10, total_steps=2080, loss=3.2, step_seconds=1.21, checkpoint_every_steps=694)
    assert rep.tick() is True
    topic, key, ev = pub.sent[0]
    assert (topic, key, ev["type"], ev["event_id"]) == ("job.lifecycle", "3f9a1c2e", "job.progress",
                                                          "progress-3f9a1c2e-g0002-s00000010")
    assert (ev["job_id"], ev["generation"], ev["cluster"]) == ("3f9a1c2e", 2, "eks-a")
    assert ev["payload"] == {"status": "RUNNING", "step": 10, "total_steps": 2080, "loss": 3.2,
                             "step_seconds": 1.21, "checkpoint_every_steps": 694}
    from collector.envelope import parse
    e = parse(json.dumps(ev).encode(), "job.lifecycle", 0, 0)
    assert e.type == "job.progress" and e.generation == 2


def test_waits_interval_and_sends_latest(setup):
    path, pub, clock, rep = setup
    write(path, step=10, loss=3.2)
    rep.tick()
    for step in (11, 12, 13):
        clock.t += 5
        write(path, step=step, loss=3.0)
        assert rep.tick() is False                         # 30초 안에는 안 보냄
    clock.t += 20
    assert rep.tick() is True
    assert pub.sent[-1][2]["payload"]["step"] == 13         # 그동안 쌓인 것 중 최신 값만


def test_unchanged_file_not_resent(setup):
    path, pub, clock, rep = setup
    write(path, step=10, loss=3.2)
    rep.tick()
    clock.t += 100
    assert rep.tick() is False and len(pub.sent) == 1


def test_status_change_sent_immediately_with_distinct_id(setup):
    path, pub, clock, rep = setup
    write(path, step=2080, loss=1.1)
    rep.tick()
    clock.t += 1
    write(path, step=2080, loss=1.1, status="COMPLETED")
    assert rep.tick() is True                               # 간격 전이라도 상태가 바뀌면 바로
    assert [v["event_id"] for _, _, v in pub.sent] == ["progress-3f9a1c2e-g0002-s00002080",
                                                        "progress-3f9a1c2e-g0002-s00002080-completed"]


def test_force_sends_pending_on_shutdown(setup):
    path, pub, clock, rep = setup
    write(path, step=10, loss=3.2)
    rep.tick()
    clock.t += 3
    write(path, step=12, loss=3.1)
    rep.tick()
    assert len(pub.sent) == 1
    assert rep.tick(force=True) is True and pub.sent[-1][2]["payload"]["step"] == 12


def test_other_generation_or_partial_file_ignored(setup):
    path, pub, _, rep = setup
    path.write_text(json.dumps({"generation": 1, "status": "RUNNING", "step": 5}))
    assert rep.tick() is False
    path.write_text('{"generation": 2, "step": ')           # 쓰는 중
    assert rep.tick() is False and pub.sent == []


def test_publish_failure_keeps_value_for_retry(setup):
    path, pub, clock, rep = setup

    def boom(**_):
        raise BufferError("queue full")
    pub.publish = boom
    write(path, step=10, loss=3.2)
    assert rep.tick() is True                               # 시도는 함, 실패는 삼킴
    assert rep._pending is not None                         # 다음 tick에 다시


def test_event_id_rules():
    assert progress_event_id("a", 1, 7, None) == "progress-a-g0001-s00000007"
    assert progress_event_id("a", 1, 7, "FAILED") == "progress-a-g0001-s00000007-failed"
