"""ckpt-sidecar: 학습 컨테이너 옆에서 계속 도는 보조 컨테이너 (규약 6 · 9 · 10절).

한 번의 tick()에서 하는 일
1. CKPT_DIR의 step-XXXXXXXX.done 중 아직 안 올린 것 처리
   - DynamoDB 세대 · owner 확인 → 다르면 STOP
   - 내 rank 파일 업로드 + ranks/rank-NNNN.json
   - rank 0: rank 목록이 WORLD_SIZE개 모이면 manifest 작성 → 조건부 승인
2. 승인된 step은 로컬에 .approved 생성 (rank 0 외에는 DynamoDB를 보고 확인)
3. 오래된 로컬 step 정리 (최근 KEEP_LOCAL개 + 현재 승인본은 유지)
4. rank 0: 하트비트 (HEARTBEAT_SECONDS 간격)
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from . import layout
from .config import Config
from .fencing import ApproveResult, FencingTable
from .log import log
from .storage import Store

STOP_EXIT_CODE = 75  # trainer가 STOP을 보고 쓰는 종료 코드 (노찬경 합의)


class Sidecar:
    def __init__(self, cfg: Config, fencing: FencingTable, store: Store, *,
                 rank_wait_seconds: float | None = None, keep_local: int | None = None,
                 heartbeat_seconds: float | None = None, image_digest: str | None = None):
        self.cfg, self.fencing, self.store = cfg, fencing, store
        self.rank_wait = rank_wait_seconds if rank_wait_seconds is not None else float(os.environ.get("RANK_WAIT_SECONDS", "300"))
        self.keep_local = keep_local if keep_local is not None else int(os.environ.get("KEEP_LOCAL", "2"))
        self.hb_every = heartbeat_seconds if heartbeat_seconds is not None else float(os.environ.get("HEARTBEAT_SECONDS", "10"))
        self.image_digest = image_digest or os.environ.get("IMAGE_DIGEST")
        self._last_hb = 0.0
        self._waiting_since: dict[int, float] = {}
        self.stopped = False

    # ---------- 공통 ----------
    def _uploaded_marker(self, step: int) -> Path:
        return self.cfg.ckpt_dir / f"{layout.step_name(step)}.uploaded"

    def stop(self, reason: str, **fields) -> None:
        if not self.stopped:
            layout.stop_file(self.cfg.ckpt_dir).write_text(reason + "\n")
            log("sidecar.stop", reason=reason, **fields)
        self.stopped = True

    def _fenced(self) -> bool:
        item = self.fencing.get(self.cfg.job_id)
        if item is None:
            self.stop("no_item")
            return True
        if item.generation != self.cfg.generation or item.owner != self.cfg.cluster:
            self.stop("generation_or_owner_mismatch", ddb_gen=item.generation, my_gen=self.cfg.generation,
                      owner=item.owner, me=self.cfg.cluster)
            return True
        return False

    def done_steps(self) -> list[int]:
        if not self.cfg.ckpt_dir.exists():
            return []
        steps = [s for p in self.cfg.ckpt_dir.iterdir() if (s := layout.parse_done(p.name)) is not None]
        return sorted(steps)

    # ---------- 1. 업로드 · 승인 ----------
    def process_step(self, step: int) -> None:
        cfg = self.cfg
        if not self._uploaded_marker(step).exists():
            if self._fenced():  # 업로드 전 펜싱 (규약 9절)
                return
            step_dir = layout.local_step_dir(cfg.ckpt_dir, step)
            t0 = time.monotonic()
            entries = self.store.upload_rank(cfg.job_id, cfg.generation, step, cfg.rank, step_dir)
            self._uploaded_marker(step).touch()
            log("sidecar.uploaded", step=step, rank=cfg.rank, files=len(entries),
                bytes=sum(e.size_bytes for e in entries), upload_ms=int((time.monotonic() - t0) * 1000))

        if layout.approved_marker(cfg.ckpt_dir, step).exists():
            return
        if cfg.rank == 0:
            self._approve_as_rank0(step)
        else:
            self._mark_if_approved(step)

    def _approve_as_rank0(self, step: int) -> None:
        cfg = self.cfg
        lists = self.store.rank_lists(cfg.job_id, cfg.generation, step, cfg.world_size)
        if lists is None:
            first = self._waiting_since.setdefault(step, time.monotonic())
            if time.monotonic() - first > self.rank_wait:
                log("sidecar.rank_wait_timeout", step=step, waited_s=int(time.monotonic() - first))  # 승인 안 함
                self._uploaded_marker(step).with_suffix(".abandoned").touch()
            return
        self._waiting_since.pop(step, None)
        try:
            self.store.build_and_put_manifest(cfg.job_id, cfg.generation, step, cfg.world_size,
                                              cfg.checkpoint_format, cfg.config_sha256, self.image_digest, lists)
        except ValueError as e:
            log("sidecar.verify_failed", step=step, error=str(e))  # 승인 안 함, 이전 승인본 유지
            return
        res = self.fencing.approve(cfg.job_id, cfg.generation, cfg.cluster, step)
        log("sidecar.approve", step=step, result=res.value)
        if res in (ApproveResult.OK, ApproveResult.ALREADY):
            layout.approved_marker(cfg.ckpt_dir, step).touch()
            # TODO: Kafka job.lifecycle에 checkpoint.approved 발행 (수집기와 함께)
        elif res in (ApproveResult.FENCED, ApproveResult.NO_ITEM):
            self.stop("approve_" + res.value, step=step)
        # STALE: 늦은 요청이라 건너뜀. RECOVERING: 다음 tick에 다시 시도

    def _mark_if_approved(self, step: int) -> None:
        item = self.fencing.get(self.cfg.job_id)
        if item and item.approved_gen == self.cfg.generation and (item.approved_step or -1) >= step:
            layout.approved_marker(self.cfg.ckpt_dir, step).touch()

    # ---------- 2. 정리 ----------
    def cleanup(self) -> None:
        item = self.fencing.get(self.cfg.job_id)
        protected = {item.approved_step} if item and item.approved_step is not None else set()
        approved = sorted(s for s in self.done_steps() if layout.approved_marker(self.cfg.ckpt_dir, s).exists())
        for s in approved[:-self.keep_local] if self.keep_local > 0 else approved:
            if s in protected:
                continue
            shutil.rmtree(layout.local_step_dir(self.cfg.ckpt_dir, s), ignore_errors=True)
            for m in (layout.done_marker, layout.approved_marker):
                m(self.cfg.ckpt_dir, s).unlink(missing_ok=True)
            self._uploaded_marker(s).unlink(missing_ok=True)
            log("sidecar.cleanup", step=s)

    # ---------- 3. 하트비트 ----------
    def heartbeat(self, now: float | None = None) -> None:
        if self.cfg.rank != 0:
            return
        now = time.monotonic() if now is None else now
        if now - self._last_hb < self.hb_every:
            return
        self._last_hb = now
        if not self.fencing.heartbeat(self.cfg.job_id, self.cfg.generation, self.cfg.cluster):
            self.stop("heartbeat_fenced")

    # ---------- 한 사이클 ----------
    def tick(self) -> None:
        if self.stopped:
            return
        self.heartbeat()
        for step in self.done_steps():
            if self.stopped:
                return
            if self._uploaded_marker(step).with_suffix(".abandoned").exists():
                continue
            self.process_step(step)
        if not self.stopped:
            self.cleanup()


def main() -> None:
    cfg = Config.from_env()
    sc = Sidecar(cfg, FencingTable(cfg.table, region=cfg.region), Store(cfg.bucket, region=cfg.region))
    poll = float(os.environ.get("POLL_SECONDS", "2"))
    log("sidecar.start", job_id=cfg.job_id, gen=cfg.generation, rank=cfg.rank, world_size=cfg.world_size)
    once = "--once" in sys.argv
    while True:
        sc.tick()
        if once or sc.stopped:
            break
        time.sleep(poll)
    # STOP 후에도 Pod가 끝날 때까지 남아 있으면 되지만, 여기서는 그대로 종료한다.


if __name__ == "__main__":
    main()
