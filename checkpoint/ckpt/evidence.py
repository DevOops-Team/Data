"""복구 재개 증거 (GPU-254): 복구로 뜬 새 세대가 승인본을 복원하고 첫 step을 실제로 끝냈다는 기록.

왜: 복구 컨트롤러(한송이, K8s recovery-controller)는 이 증거를 펜싱 표 recovery_training_evidence(Map)에서 읽어
RESUMING → COMPLETED로 넘긴다. Pod Running · init 다운로드 성공 · 하트비트만으로는 완료로 보지 않는다 (규약 8절).

누가 · 언제: rank 0 사이드카가 tick마다 확인한다. 쓰는 조건은 셋 다:
  - 펜싱 표가 이 실행의 복구 중: generation = 내 세대, recovery_id 있음, recovery_phase NEW_RUN_SUBMITTED · RESUMING
  - trainer 진행률(progress.json)의 step이 복원 지점(resume.json approved_step, 처음부터면 0)보다 큼
    = 불러온 뒤 파라미터 업데이트를 실제로 1번 이상 끝냄. trainer 실패(status FAILED)면 실패 증거
  - JOBSET_UID(이 Pod를 만든 JobSet UID)가 있음 — 없으면 쓰지 않고 로그만
컨트롤러는 observed_at이 300초보다 오래된 증거를 거절하므로, 쓴 뒤에도 복구가 끝날 때까지 REFRESH_SECONDS마다
최신 step · 시각으로 다시 쓴다. 쓰기는 펜싱 표 조건부 갱신 (fencing.write_recovery_evidence).

형식 (K8s recovery-controller/recovery/evidence.py progress 계약, producer · received_at은 컨트롤러가 붙임):
  {kind: progress, schema_version: 1, job_id, recovery_id, generation, cluster, jobset_uid, event_id, observed_at,
   outcome: succeeded, completed_step, checkpoint_loaded: true, restored_gen, restored_step}   # 승인본에서 재개
   ... outcome: succeeded, completed_step, started_from_scratch: true                          # 승인본 없이 처음부터
   ... outcome: failed, failure_reason                                                         # trainer 실패
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .config import Config
from .fencing import FencingTable
from .log import log

EVIDENCE_PHASES = ("NEW_RUN_SUBMITTED", "RESUMING")
SCHEMA_VERSION = 1


def read_json(path: Path | str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        with open(path) as f:
            body = json.load(f)
        return body if isinstance(body, dict) else None
    except (OSError, ValueError):
        return None


def build(cfg: Config, recovery_id: str, jobset_uid: str, resume: dict[str, Any] | None,
          progress: dict[str, Any] | None, now: float) -> dict[str, Any] | None:
    """증거 본문. 아직 쓸 때가 아니면 None (진행률 없음 · 첫 step 전 · 다른 세대의 진행률)."""
    if not progress or progress.get("generation") not in (None, cfg.generation):
        return None
    body: dict[str, Any] = {
        "kind": "progress", "schema_version": SCHEMA_VERSION,
        "job_id": cfg.job_id, "recovery_id": recovery_id, "generation": cfg.generation,
        "cluster": cfg.cluster, "jobset_uid": jobset_uid,
        "event_id": f"{cfg.job_id}-g{cfg.generation:04d}-{recovery_id}-progress",
        "observed_at": int(now),
    }
    if str(progress.get("status") or "").upper() == "FAILED":
        return dict(body, outcome="failed", failure_reason=str(progress.get("error") or "trainer failed")[:256])
    resume = resume or {}
    restored = resume.get("approved_step") if resume.get("resume") else None
    step = progress.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step <= (restored or 0):
        return None
    body.update(outcome="succeeded", completed_step=step)
    if restored is not None:
        body.update(checkpoint_loaded=True, restored_gen=int(resume["approved_gen"]), restored_step=int(restored))
    else:
        body.update(started_from_scratch=True)
    return body


class RecoveryEvidence:
    def __init__(self, cfg: Config, fencing: FencingTable, progress_file: str | None, jobset_uid: str | None, *,
                 check_seconds: float | None = None, refresh_seconds: float | None = None, clock=time.time):
        self.cfg, self.fencing, self.progress_file, self.jobset_uid = cfg, fencing, progress_file, jobset_uid
        self.check_every = check_seconds if check_seconds is not None else float(os.environ.get("EVIDENCE_CHECK_SECONDS", "10"))
        self.refresh_every = (refresh_seconds if refresh_seconds is not None
                              else float(os.environ.get("EVIDENCE_REFRESH_SECONDS", "60")))
        self.clock = clock
        self._next_at = 0.0
        self._warned_no_uid = False
        self.written: dict[str, Any] | None = None   # 마지막으로 쓴 증거 (테스트 · 로그용)

    def tick(self) -> bool:
        """쓸 때면 조건부로 쓰고 True. DynamoDB 연결 오류는 호출한 쪽(사이드카 tick)이 다음 사이클로 넘긴다."""
        if self.cfg.rank != 0:
            return False
        now = self.clock()
        if now < self._next_at:
            return False
        self._next_at = now + self.check_every
        item = self.fencing.get(self.cfg.job_id)
        if (item is None or item.generation != self.cfg.generation or not item.recovery_id
                or item.recovery_phase not in EVIDENCE_PHASES):
            return False          # 복구로 뜬 실행이 아니거나 복구가 이미 끝남(COMPLETED) · 다른 단계
        if not self.jobset_uid:
            if not self._warned_no_uid:
                log("evidence.no_jobset_uid", recovery_id=item.recovery_id, generation=self.cfg.generation)
                self._warned_no_uid = True
            return False
        body = build(self.cfg, item.recovery_id, self.jobset_uid, read_json(self.cfg.ckpt_dir / "resume.json"),
                     read_json(self.progress_file), now)
        if body is None:
            return False
        ok = self.fencing.write_recovery_evidence(self.cfg.job_id, self.cfg.generation, self.cfg.cluster,
                                                  item.recovery_id, self.jobset_uid, body)
        log("evidence.write", ok=ok, recovery_id=item.recovery_id, generation=self.cfg.generation,
            outcome=body["outcome"], completed_step=body.get("completed_step"))
        if ok:
            self.written = body
            self._next_at = now + self.refresh_every   # 컨트롤러의 300초 신선도 안에서 다시 씀
        return ok
