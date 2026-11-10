#!/usr/bin/env python3
"""수집기 실제 경로 시험용 사건 만들기 (표준 라이브러리만, Kafka VM에서 실행).

출력: kafka-console-producer에 그대로 넣을 수 있는 "키|JSON" 한 줄씩.

    python3 e2e_events.py --prefix t1 --topic job.lifecycle   > lc.txt
    python3 e2e_events.py --prefix t1 --topic fault.injection > inj.txt

시나리오 (작업 1개 · 세대 1 → 장애 → 세대 2 재개):
  experiment.started → job.submitted(g1) → job.started(g1) → checkpoint.approved(g1, step 100)
  → fault.injected → job.fenced(g1) → job.resumed(g2, 다른 클러스터) → experiment.ended
  + 같은 checkpoint.approved 한 번 더 (중복 무시 확인) + 필수 필드 없는 사건 1건 (event_rejects 확인)

gpu.fault · recovery.action 토픽은 쓰지 않는다 (복구 컨트롤러가 읽을 수 있어 가짜 장애로 복구가 돌 수 있음).
시각은 지금(UTC) 기준으로 과거로 배치해서 DB 도착 지연(received_at - occurred_at)이 양수로 나온다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json


def build(prefix: str) -> dict[str, list[tuple[str, dict]]]:
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    t0 = now - dt.timedelta(seconds=300)

    def iso(s: int) -> str:
        return (t0 + dt.timedelta(seconds=s)).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    job, exp = f"e2e{prefix}", f"exp-e2e-{prefix}"

    def ev(i: int, type_: str, s: int, payload: dict | None = None, **env) -> dict:
        return {"event_id": f"e2e-{prefix}-{i:02d}", "schema_version": 1, "type": type_, "occurred_at": iso(s),
                "source": "e2e-test", "payload": payload or {}, **env}

    g1 = {"job_id": job, "generation": 1, "cluster": "eks-a"}
    approved = ev(4, "checkpoint.approved", 60, {
        "step": 100, "format": "fsdp-dcp", "world_size": 4,
        "manifest_key": f"jobs/{job}/gen-0001/step-00000100/manifest.json", "file_count": 9,
        "size_bytes": 1500000000, "config_sha256": "e2e", "upload_ms": 8000,
        "saved_at": iso(55), "approved_at": iso(60)}, node="n-e2e", **g1)
    lifecycle = [
        (exp, ev(1, "experiment.started", 0, {"exp_id": exp, "name": f"e2e-{prefix}", "scenario": "e2e",
                                               "policy": "ours", "fault_type": "none", "spare_enabled": True,
                                               "checkpoint_every_steps": 100})),
        (job, ev(2, "job.submitted", 5, {"jobset_name": f"job-{job}-g0001", "gpus": 4, "strategy": "fsdp",
                                         "queue": "queue-heavy", "exp_id": exp, "config_sha256": "e2e"}, **g1)),
        (job, ev(3, "job.started", 30, **g1)),
        (job, approved),
        (job, ev(6, "job.fenced", 100, **g1)),
        (job, ev(7, "job.resumed", 160, job_id=job, generation=2, cluster="eks-b")),
        (exp, ev(8, "experiment.ended", 240, {"exp_id": exp})),
        (job, approved),                                                      # 중복
        (job, {"event_id": f"e2e-{prefix}-bad", "type": "job.started"}),      # 형식 오류
    ]
    injection = [("n-e2e", ev(5, "fault.injected", 90, {"fault": "e2e"}, job_id=job, node="n-e2e"))]
    return {"job.lifecycle": lifecycle, "fault.injection": injection}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="시험마다 다르게 (예: t1). 사건 ID · 작업 ID에 붙음")
    ap.add_argument("--topic", required=True, choices=["job.lifecycle", "fault.injection"])
    a = ap.parse_args()
    for key, event in build(a.prefix)[a.topic]:
        print(f"{key}|{json.dumps(event, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
