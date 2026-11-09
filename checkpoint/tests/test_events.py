"""사이드카 → Kafka checkpoint.approved 발행 (events.py)."""
from datetime import datetime

from ckpt import events, layout
from test_flow import JOB, WORLD, cfg, fake_save, sidecar

# 수집기(collector/collector/store.py _checkpoint_approved)가 요구하는 필드. 바뀌면 둘 다 고친다.
COLLECTOR_REQUIRED = ("step", "format", "world_size", "manifest_key", "file_count", "size_bytes", "config_sha256")
ENVELOPE_REQUIRED = ("event_id", "schema_version", "type", "occurred_at")


class ListPublisher:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def publish(self, *, topic, key, value):
        if self.fail:
            raise RuntimeError("kafka down")
        self.sent.append((topic, key, value))

    def flush(self, timeout=5.0):
        pass


def _approve_step(names, fencing, store, tmp_path, step=100, pub=None):
    fencing.create_job(JOB)
    cs = [cfg(names, tmp_path, gen=1, rank=r, cluster="eks-a", pod=f"r{r}") for r in range(WORLD)]
    from ckpt.init_main import run as init_run
    for c in cs:
        assert init_run(c, fencing, store) == 0
        fake_save(c, step)
    scs = [sidecar(c, fencing, store) for c in cs]
    pubs = [pub or ListPublisher(), ListPublisher()]
    for sc, p in zip(scs, pubs):
        sc.publisher = p
    scs[1].tick()
    scs[0].tick()
    return cs, scs, pubs


def test_checkpoint_approved_matches_collector_contract(names, fencing, store, tmp_path):
    cs, _, pubs = _approve_step(names, fencing, store, tmp_path)
    assert fencing.get(JOB).approved_step == 100
    assert len(pubs[0].sent) == 1 and pubs[1].sent == []          # rank 0만 발행
    topic, key, ev = pubs[0].sent[0]
    assert topic == "job.lifecycle" and key == JOB
    assert all(ev.get(k) not in (None, "") for k in ENVELOPE_REQUIRED)
    assert ev["type"] == "checkpoint.approved" and ev["job_id"] == JOB and ev["generation"] == 1
    assert ev["event_id"] == "ckpt-3f9a1c2e-g0001-s00000100"
    p = ev["payload"]
    assert all(p.get(k) not in (None, "") for k in COLLECTOR_REQUIRED)
    manifest = store.get_json(layout.s3_manifest_key(JOB, 1, 100))
    assert p["manifest_key"] == layout.s3_manifest_key(JOB, 1, 100)
    assert p["file_count"] == len(manifest["files"]) == 5
    assert p["size_bytes"] == sum(f["size_bytes"] for f in manifest["files"])
    assert (p["world_size"], p["format"], p["config_sha256"]) == (WORLD, "fsdp-dcp", "cfg-sha")
    assert isinstance(p["upload_ms"], int)
    for k in ("saved_at", "approved_at"):                         # 시간대 포함 ISO 8601 (수집기 검증)
        assert datetime.fromisoformat(p[k].replace("Z", "+00:00")).tzinfo is not None
    assert p["saved_at"] <= p["approved_at"] == ev["occurred_at"]


def test_republish_same_step_has_same_event_id(names, fencing, store, tmp_path):
    """사이드카 재시작 등으로 같은 step을 다시 승인(ALREADY)해도 event_id가 같아 DB에는 한 번만."""
    cs, scs, pubs = _approve_step(names, fencing, store, tmp_path)
    layout.approved_marker(cs[0].ckpt_dir, 100).unlink()
    scs[0].tick()
    ids = [v["event_id"] for _, _, v in pubs[0].sent]
    assert len(ids) == 2 and len(set(ids)) == 1


def test_kafka_failure_does_not_block_approval(names, fencing, store, tmp_path):
    cs, scs, _ = _approve_step(names, fencing, store, tmp_path, pub=ListPublisher(fail=True))
    assert fencing.get(JOB).approved_step == 100
    assert layout.approved_marker(cs[0].ckpt_dir, 100).exists()
    assert not scs[0].stopped


def test_kafka_publisher_with_fake_producer():
    class Msg:
        def topic(self): return "job.lifecycle"
        def key(self): return b"j"
        def partition(self): return 0
        def offset(self): return 7

    class FakeProducer:
        def __init__(self): self.calls, self.polls = [], 0
        def produce(self, topic, key, value, on_delivery):
            if topic == "full":
                raise BufferError("queue full")
            self.calls.append((topic, key, value))
            on_delivery("broker down", Msg())     # 실패 콜백도 예외 없이 로그만
            on_delivery(None, Msg())
        def poll(self, t): self.polls += 1
        def flush(self, t): return 0

    fp = FakeProducer()
    pub = events.KafkaPublisher("unused", producer=fp)
    pub.publish(topic="job.lifecycle", key="j", value={"a": "한글"})
    pub.publish(topic="full", key="j", value={})               # BufferError → 로그만
    pub.flush()
    assert fp.calls == [("job.lifecycle", b"j", '{"a": "한글"}'.encode())] and fp.polls == 2


def test_from_env_without_bootstrap_is_noop(monkeypatch):
    monkeypatch.delenv("KAFKA_BOOTSTRAP", raising=False)
    assert isinstance(events.from_env(), events.NullPublisher)
