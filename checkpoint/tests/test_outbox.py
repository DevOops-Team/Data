"""S3 대기열(outbox): Kafka로 못 보낸 사건 보관 → 다시 보내기."""
import json

import boto3
import pytest

from ckpt import events, layout
from ckpt.events import KafkaPublisher
from ckpt.outbox import S3Outbox, replay
from conftest import REGION
from test_flow import JOB, WORLD, cfg, fake_save, sidecar


class Msg:
    def __init__(self, topic, key, value):
        self._t, self._k, self._v = topic, key, value
    def topic(self): return self._t
    def key(self): return self._k
    def value(self): return self._v
    def partition(self): return 0
    def offset(self): return 1


class FakeProducer:
    """confluent-kafka Producer 흉내. mode: deliver(성공) · fail(전달 실패) · stuck(안 끝남, purge 때 실패 콜백) · full(BufferError)"""
    def __init__(self, mode="deliver"):
        self.mode, self.pending, self.sent, self.purged = mode, [], [], False

    def produce(self, topic, key=None, value=None, on_delivery=None):
        if self.mode == "full":
            raise BufferError("Local: Queue full")
        self.pending.append((Msg(topic, key, value), on_delivery))

    def poll(self, timeout):
        return 0

    def flush(self, timeout=None):
        if self.mode == "stuck" and not self.purged:
            return len(self.pending)
        for msg, cb in self.pending:
            if self.mode == "deliver":
                self.sent.append(msg)
                cb(None, msg)
            else:
                cb("Local: Purged in queue" if self.purged else "Local: Message timed out", msg)
        self.pending = []
        return 0

    def purge(self, in_queue=True, in_flight=True, blocking=True):
        self.purged = True


@pytest.fixture
def outbox(names):
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=names["bucket"], CreateBucketConfiguration={"LocationConstraint": REGION})
    return S3Outbox(names["bucket"], client=s3)


def _ev(eid="ckpt-j-g0001-s00000100"):
    return events.envelope(type_="checkpoint.approved", event_id=eid, cluster="eks-a", job_id="j",
                           generation=1, payload={"step": 100})


def test_delivered_goes_nowhere(outbox):
    pub = KafkaPublisher("unused", producer=FakeProducer("deliver"), outbox=outbox)
    pub.publish(topic="job.lifecycle", key="j", value=_ev())
    pub.flush()
    assert outbox.keys() == []


def test_delivery_failure_goes_to_outbox(outbox):
    pub = KafkaPublisher("unused", producer=FakeProducer("fail"), outbox=outbox)
    pub.publish(topic="job.lifecycle", key="j", value=_ev())
    pub.flush()
    keys = outbox.keys()
    assert keys == ["events-outbox/ckpt-sidecar/ckpt-j-g0001-s00000100.json"]
    item = outbox.get(keys[0])
    assert item["topic"] == "job.lifecycle" and item["key"] == "j"
    assert item["value"]["event_id"] == "ckpt-j-g0001-s00000100" and item["value"]["payload"] == {"step": 100}
    assert item["reason"].startswith("delivery_failed")


def test_buffer_full_goes_to_outbox(outbox):
    pub = KafkaPublisher("unused", producer=FakeProducer("full"), outbox=outbox)
    pub.publish(topic="job.lifecycle", key="j", value=_ev())
    assert outbox.get(outbox.keys()[0])["reason"].startswith("produce_failed")


def test_unsent_at_shutdown_goes_to_outbox(outbox):
    """VPN이 끊긴 채 Pod가 종료(SIGTERM)되는 경우: flush 제한 시간 안에 못 보낸 사건을 purge → outbox."""
    pub = KafkaPublisher("unused", producer=FakeProducer("stuck"), outbox=outbox)
    pub.publish(topic="job.lifecycle", key="j", value=_ev("a"))
    pub.publish(topic="job.lifecycle", key="j", value=_ev("b"))
    pub.flush(0.01)
    assert sorted(outbox.keys()) == ["events-outbox/ckpt-sidecar/a.json", "events-outbox/ckpt-sidecar/b.json"]


def test_same_event_twice_is_one_object(outbox):
    pub = KafkaPublisher("unused", producer=FakeProducer("fail"), outbox=outbox)
    for _ in range(2):
        pub.publish(topic="job.lifecycle", key="j", value=_ev())
    pub.flush()
    assert len(outbox.keys()) == 1


def test_outbox_store_failure_does_not_raise(names):
    broken = S3Outbox("no-such-bucket-xyz", client=boto3.client("s3", region_name=REGION))
    pub = KafkaPublisher("unused", producer=FakeProducer("fail"), outbox=broken)
    pub.publish(topic="job.lifecycle", key="j", value=_ev())
    pub.flush()          # 예외 없이 끝나야 함 (로그만)


def test_replay_sends_and_deletes(outbox):
    outbox.put("job.lifecycle", "j", _ev("a"), "test")
    outbox.put("job.lifecycle", "j", _ev("b"), "test")
    prod = FakeProducer("deliver")
    stats = replay(outbox, prod)
    assert stats["delivered"] == 2 and outbox.keys() == []
    sent = [json.loads(m.value()) for m in prod.sent]
    assert sorted(e["event_id"] for e in sent) == ["a", "b"]
    assert all(m.topic() == "job.lifecycle" and m.key() == b"j" for m in prod.sent)


def test_replay_keeps_when_kafka_still_down(outbox):
    outbox.put("job.lifecycle", "j", _ev("a"), "test")
    stats = replay(outbox, FakeProducer("fail"))
    assert stats["failed"] == 1 and len(outbox.keys()) == 1


def test_replay_skips_broken_object(outbox):
    outbox.s3.put_object(Bucket=outbox.bucket, Key="events-outbox/ckpt-sidecar/broken.json", Body=b"{not json")
    outbox.put("job.lifecycle", "j", _ev("a"), "test")
    stats = replay(outbox, FakeProducer("deliver"))
    assert stats["delivered"] == 1 and stats["skipped"] == 1
    assert outbox.keys() == ["events-outbox/ckpt-sidecar/broken.json"]   # 깨진 건 남겨서 사람이 확인


def test_sidecar_approval_with_kafka_down_lands_in_outbox(names, fencing, store, tmp_path):
    """VPN이 끊겨 Kafka 전달이 실패해도 승인은 되고, 승인 사건은 S3 대기열에 남는다."""
    fencing.create_job(JOB)
    cs = [cfg(names, tmp_path, gen=1, rank=r, cluster="eks-a", pod=f"r{r}") for r in range(WORLD)]
    from ckpt.init_main import run as init_run
    for c in cs:
        assert init_run(c, fencing, store) == 0
        fake_save(c, 100)
    scs = [sidecar(c, fencing, store) for c in cs]
    ob = S3Outbox(names["bucket"], client=store.s3)
    pub = KafkaPublisher("unused", producer=FakeProducer("fail"), outbox=ob)
    scs[0].publisher = pub
    scs[1].tick()
    scs[0].tick()
    pub.flush()
    assert fencing.get(JOB).approved_step == 100
    assert layout.approved_marker(cs[0].ckpt_dir, 100).exists()
    item = ob.get(ob.keys()[0])
    assert item["value"]["type"] == "checkpoint.approved" and item["value"]["event_id"] == "ckpt-3f9a1c2e-g0001-s00000100"


def test_from_env_outbox_switch(monkeypatch):
    monkeypatch.setenv("KAFKA_BOOTSTRAP", "127.0.0.1:1")
    assert events.from_env("b").outbox is not None
    monkeypatch.setenv("OUTBOX_PREFIX", "")
    assert events.from_env("b").outbox is None
    monkeypatch.delenv("OUTBOX_PREFIX")
    assert events.from_env(None).outbox is None
