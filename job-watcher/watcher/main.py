"""(잡) 감시기 실행 루프.

구조
  [JobSet watch 스레드] ─┐
                         ├─▶ 큐 ─▶ [처리 스레드: Watcher] ─▶ Kafka job.lifecycle
  [Workload watch 스레드] ┘

- 각 watch 스레드: list(현재 전부, SYNC) → 그 resourceVersion부터 watch(변화만). 끊기거나 410(Gone)이면 다시 list
- Watcher: JobSet · Workload를 캐시에 두고, 바뀐 JobSet마다 rules.plan으로 사건을 고른 뒤
  아직 안 보낸 것만 발행한다. 다시 list했을 때 캐시에 있던 JobSet이 없으면 그 사이 삭제된 것으로 본다.
- 감시기가 재시작하면 캐시가 비어 있어 사건을 다시 보낸다 → event_id가 같아 수집기가 걸러냄

환경 변수
  WATCH_NAMESPACE(training) · KAFKA_BOOTSTRAP(없으면 표준 출력) · KAFKA_TOPIC_LIFECYCLE(job.lifecycle)
  JOBSET_API(jobset.x-k8s.io/v1alpha2) · WORKLOAD_API(kueue.x-k8s.io/v1beta1) · WATCH_TIMEOUT_S(300)
"""
from __future__ import annotations

import os
import queue
import signal
import threading
import time
from typing import Any, Callable

from . import events
from .log import log, now_iso
from .model import owner_jobset, snapshot
from .rules import plan

SYNC, ADDED, MODIFIED, DELETED = "SYNC", "ADDED", "MODIFIED", "DELETED"


class Watcher:
    def __init__(self, publisher: events.Publisher, topic: str = "job.lifecycle",
                 clock: Callable[[], str] = now_iso, *, wait_initial_sync: bool = True):
        self.publisher, self.topic, self.clock = publisher, topic, clock
        self.jobsets: dict[str, dict[str, Any]] = {}          # 이름 → JobSet
        self.workloads: dict[str, dict[str, Any]] = {}        # JobSet 이름 → Workload
        self.sent: set[str] = set()                           # 이 프로세스에서 보낸 event_id
        # 시작 직후 JobSet 목록만 먼저 오면 started가 Workload PodsReady 시각 대신 본 시각으로 나가고,
        # event_id가 같아 그 값이 DB에 남는다 → 두 목록을 다 받은 뒤부터 판정한다.
        self._synced: set[str] = set() if wait_initial_sync else {"jobset", "workload"}
        self._pending_deleted: list[dict[str, Any]] = []      # 준비 전에 삭제된 JobSet → 준비되면 처리

    @property
    def ready(self) -> bool:
        return self._synced >= {"jobset", "workload"}

    def _mark_synced(self, kind: str) -> None:
        was_ready = self.ready
        self._synced.add(kind)
        if self.ready and not was_ready:
            for obj in self._pending_deleted:
                self.evaluate(obj["metadata"]["name"], obj=obj, deleted=True)
            self._pending_deleted.clear()
            for name in list(self.jobsets):
                self.evaluate(name)

    # ---------- 입력 ----------
    def on_jobset(self, kind: str, obj: Any) -> None:
        if kind == SYNC:
            listed = {o["metadata"]["name"]: o for o in obj}
            for name in set(self.jobsets) - set(listed):       # 연결이 끊긴 사이 삭제됨
                self._deleted(self.jobsets[name])
            for o in listed.values():
                self._upsert(o)
            self._mark_synced("jobset")
        elif kind in (ADDED, MODIFIED):
            self._upsert(obj)
        elif kind == DELETED:
            self._deleted(obj)

    def on_workload(self, kind: str, obj: Any) -> None:
        if kind == SYNC:
            self.workloads = {}
            for o in obj:
                if (name := owner_jobset(o)):
                    self.workloads[name] = o
            if self.ready:
                for name in list(self.jobsets):
                    self.evaluate(name)
            self._mark_synced("workload")
            return
        name = owner_jobset(obj)
        if not name:
            return
        if kind == DELETED:
            self.workloads.pop(name, None)
        else:
            self.workloads[name] = obj
            self.evaluate(name)

    # ---------- 판정 ----------
    def _upsert(self, obj: dict[str, Any]) -> None:
        name = obj["metadata"]["name"]
        self.jobsets[name] = obj
        self.evaluate(name)

    def _deleted(self, obj: dict[str, Any]) -> None:
        name = obj["metadata"]["name"]
        if self.ready:
            self.evaluate(name, obj=obj, deleted=True)
            self.workloads.pop(name, None)
        else:
            self._pending_deleted.append(obj)
        self.jobsets.pop(name, None)

    def _newer_generation_exists(self, job_id: str, generation: int) -> bool:
        for o in self.jobsets.values():
            s = snapshot(o)
            if s and s.job_id == job_id and s.generation > generation:
                return True
        return False

    def evaluate(self, name: str, *, obj: dict[str, Any] | None = None, deleted: bool = False) -> list[dict]:
        obj = obj or self.jobsets.get(name)
        if obj is None or not self.ready:
            return []
        s = snapshot(obj, self.workloads.get(name), deleted=deleted)
        if s is None:
            return []                                           # 이름 · 라벨 규칙 밖 → 우리 학습 JobSet 아님
        sent = []
        for p in plan(s, newer_generation_exists=self._newer_generation_exists(s.job_id, s.generation),
                      observed_at=self.clock()):
            ev = events.build(s, p)
            if ev["event_id"] in self.sent:
                continue
            self.publisher.publish(topic=self.topic, key=s.job_id, value=ev)
            self.sent.add(ev["event_id"])
            sent.append(ev)
            log("event.sent", event_id=ev["event_id"], type=ev["type"], occurred_at=ev["occurred_at"],
                time_source=p.time_source)
        return sent


# ---------- 쿠버네티스 list · watch ----------
def _stream(api, group: str, version: str, plural: str, namespace: str, kind: str,
            out: "queue.Queue[tuple[str, str, Any]]", stop: threading.Event, timeout_s: int) -> None:
    from kubernetes import watch
    from kubernetes.client.rest import ApiException

    while not stop.is_set():
        try:
            lst = api.list_namespaced_custom_object(group, version, namespace, plural)
            out.put((kind, SYNC, lst.get("items") or []))
            rv = (lst.get("metadata") or {}).get("resourceVersion")
            w = watch.Watch()
            for ev in w.stream(api.list_namespaced_custom_object, group, version, namespace, plural,
                               resource_version=rv, timeout_seconds=timeout_s, allow_watch_bookmarks=True):
                if stop.is_set():
                    w.stop()
                    break
                t = ev.get("type")
                if t == "BOOKMARK":
                    continue
                if t == "ERROR":                                # 410 Gone 등 → 다시 list
                    log("watch.error", kind=kind, object=ev.get("object"))
                    break
                out.put((kind, t, ev["object"]))
        except ApiException as e:
            log("watch.api_error", kind=kind, status=e.status, reason=e.reason)
            stop.wait(5)
        except Exception as e:  # noqa: BLE001 — 연결 끊김 등. 잠시 뒤 다시 list
            log("watch.failed", kind=kind, error=repr(e))
            stop.wait(5)


def _split(api_version: str) -> tuple[str, str]:
    group, version = api_version.split("/", 1)
    return group, version


def main() -> int:
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()                               # 로컬 실행 (kind 등)
    api = client.CustomObjectsApi()

    ns = os.environ.get("WATCH_NAMESPACE", "training")
    topic = os.environ.get("KAFKA_TOPIC_LIFECYCLE", "job.lifecycle")
    timeout_s = int(os.environ.get("WATCH_TIMEOUT_S", "300"))
    js_group, js_ver = _split(os.environ.get("JOBSET_API", "jobset.x-k8s.io/v1alpha2"))
    wl_group, wl_ver = _split(os.environ.get("WORKLOAD_API", "kueue.x-k8s.io/v1beta1"))

    publisher = events.from_env()
    watcher = Watcher(publisher, topic)
    q: "queue.Queue[tuple[str, str, Any]]" = queue.Queue()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    threads = [
        threading.Thread(target=_stream, daemon=True, name="jobsets",
                         args=(api, js_group, js_ver, "jobsets", ns, "jobset", q, stop, timeout_s)),
        threading.Thread(target=_stream, daemon=True, name="workloads",
                         args=(api, wl_group, wl_ver, "workloads", ns, "workload", q, stop, timeout_s)),
    ]
    for t in threads:
        t.start()
    log("watcher.started", namespace=ns, topic=topic, jobset_api=f"{js_group}/{js_ver}",
        workload_api=f"{wl_group}/{wl_ver}")

    while not stop.is_set():
        try:
            kind, etype, obj = q.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            (watcher.on_jobset if kind == "jobset" else watcher.on_workload)(etype, obj)
        except Exception as e:  # noqa: BLE001 — 객체 하나 때문에 감시기 전체가 멈추지 않게
            log("watcher.handle_failed", kind=kind, type=etype, error=repr(e))

    left = publisher.flush(10.0)
    log("watcher.stopped", pending=left)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
