"""ckpt-sidecar: 학습 컨테이너 옆에서 계속 도는 보조 컨테이너 (규약 6 · 9 · 10절).

한 번의 tick()에서 하는 일
1. CKPT_DIR의 step-XXXXXXXX.done 중 아직 안 올린 것 처리
   - DynamoDB 세대 · owner 확인 → 다르면 STOP
   - 내 rank 파일 업로드 + ranks/rank-NNNN.json
   - rank 0: rank 목록이 WORLD_SIZE개 모이면 manifest 작성 → 조건부 승인
   - 승인되면 Kafka job.lifecycle에 checkpoint.approved 발행 (최선 노력, events.py)
2. 승인된 step은 로컬에 .approved 생성 (rank 0 외에는 DynamoDB를 보고 확인)
3. 오래된 로컬 step 정리 (최근 KEEP_LOCAL개 + 현재 승인본은 유지)
4. rank 0: 하트비트 (HEARTBEAT_SECONDS 간격)
"""
from __future__ import annotations

import os
import signal
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import events, layout
from .config import Config
from .fencing import ApproveResult, FencingTable, now_iso
from .log import log
from .retry import is_transient
from .storage import Store

STOP_EXIT_CODE = 75  # trainer가 STOP을 보고 쓰는 종료 코드 (노찬경 합의)


class Sidecar:
    def __init__(self, cfg: Config, fencing: FencingTable, store: Store, *,
                 rank_wait_seconds: float | None = None, keep_local: int | None = None,
                 heartbeat_seconds: float | None = None, image_digest: str | None = None,
                 publisher: events.Publisher | None = None):
        self.cfg, self.fencing, self.store = cfg, fencing, store
        self.publisher = publisher or events.NullPublisher()
        self.lifecycle_topic = os.environ.get("KAFKA_TOPIC_LIFECYCLE", "job.lifecycle")
        self._upload_ms: dict[int, int] = {}
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
            self.cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
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
        # 이미 승인된 step은 올리지 않음. init이 재개용으로 받은 승인본(.done · .approved)을
        # 새 세대 경로로 다시 올리지 않게 함 (11/10 실제 AWS 1단계에서 rank당 134MB 재업로드 확인)
        if layout.approved_marker(cfg.ckpt_dir, step).exists():
            return
        if not self._uploaded_marker(step).exists():
            if self._fenced():  # 업로드 전 펜싱 (규약 9절)
                return
            step_dir = layout.local_step_dir(cfg.ckpt_dir, step)
            t0 = time.monotonic()
            try:
                entries = self.store.upload_rank(cfg.job_id, cfg.generation, step, cfg.rank, step_dir)
            except Exception as e:  # noqa: BLE001
                if not is_transient(e):
                    raise
                # 파일 단위 재시도(3회)까지 실패 → 이 step은 승인하지 않음. 이전 승인본 유지, 다음 저장 때 다시
                self._uploaded_marker(step).with_suffix(".abandoned").touch()
                log("sidecar.upload_failed", step=step, rank=cfg.rank, error=repr(e))
                return
            self._uploaded_marker(step).touch()
            self._upload_ms[step] = int((time.monotonic() - t0) * 1000)
            log("sidecar.uploaded", step=step, rank=cfg.rank, files=len(entries),
                bytes=sum(e.size_bytes for e in entries), upload_ms=self._upload_ms[step])

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
            manifest = self.store.build_and_put_manifest(cfg.job_id, cfg.generation, step, cfg.world_size,
                                                         cfg.checkpoint_format, cfg.config_sha256, self.image_digest,
                                                         lists)
        except ValueError as e:
            log("sidecar.verify_failed", step=step, error=str(e))  # 승인 안 함, 이전 승인본 유지
            return
        res = self.fencing.approve(cfg.job_id, cfg.generation, cfg.cluster, step)
        log("sidecar.approve", step=step, result=res.value)
        if res in (ApproveResult.OK, ApproveResult.ALREADY):
            layout.approved_marker(cfg.ckpt_dir, step).touch()
            self._publish_approved(step, manifest)
        elif res in (ApproveResult.FENCED, ApproveResult.NO_ITEM):
            self.stop("approve_" + res.value, step=step)
        # STALE: 늦은 요청이라 건너뜀. RECOVERING: 다음 tick에 다시 시도

    def _publish_approved(self, step: int, manifest: dict) -> None:
        """checkpoint.approved → 수집기가 data.checkpoints에 저장. 실패해도 승인 흐름에는 영향 없음."""
        cfg = self.cfg
        try:
            approved_at = now_iso()
            done = layout.done_marker(cfg.ckpt_dir, step)
            saved_at = (datetime.fromtimestamp(done.stat().st_mtime, timezone.utc)
                        .isoformat(timespec="milliseconds").replace("+00:00", "Z")) if done.exists() else approved_at
            files = manifest["files"]
            payload = {
                "step": step, "format": manifest["format"], "world_size": manifest["world_size"],
                "manifest_key": layout.s3_manifest_key(cfg.job_id, cfg.generation, step),
                "file_count": len(files), "size_bytes": sum(int(f["size_bytes"]) for f in files),
                "config_sha256": manifest.get("config_sha256") or "unknown",
                "image_digest": manifest.get("image_digest"),
                "upload_ms": self._upload_ms.get(step),     # rank 0 업로드 시간 (재시작 뒤면 없음)
                "saved_at": saved_at,                         # .done 파일 시각 = trainer 저장 완료
                "approved_at": approved_at,
            }
            ev = events.envelope(type_="checkpoint.approved",
                                 event_id=events.checkpoint_approved_id(cfg.job_id, cfg.generation, step),
                                 cluster=cfg.cluster, job_id=cfg.job_id, generation=cfg.generation,
                                 payload=payload, occurred_at=approved_at)
            self.publisher.publish(topic=self.lifecycle_topic, key=cfg.job_id, value=ev)
        except Exception as e:  # noqa: BLE001
            log("sidecar.publish_failed", step=step, error=repr(e))

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
        """DynamoDB · S3 연결 오류는 STOP이 아니다. 이번 사이클만 건너뛰고 다음 사이클에 다시 한다.
        STOP은 조건부 갱신이 '거부'됐을 때(세대 · owner 불일치)만."""
        if self.stopped:
            return
        try:
            self.heartbeat()
            for step in self.done_steps():
                if self.stopped:
                    return
                if self._uploaded_marker(step).with_suffix(".abandoned").exists():
                    continue
                self.process_step(step)
            if not self.stopped:
                self.cleanup()
        except Exception as e:  # noqa: BLE001
            if not is_transient(e):
                raise
            log("sidecar.transient_skip", error=repr(e))


def main() -> None:
    cfg = Config.from_env()
    publisher = events.from_env(cfg.bucket, cfg.region) if cfg.rank == 0 else events.NullPublisher()   # 발행은 rank 0만
    sc = Sidecar(cfg, FencingTable(cfg.table, region=cfg.region), Store(cfg.bucket, region=cfg.region),
                 publisher=publisher)
    poll = float(os.environ.get("POLL_SECONDS", "2"))
    log("sidecar.start", job_id=cfg.job_id, gen=cfg.generation, rank=cfg.rank, world_size=cfg.world_size)
    once = "--once" in sys.argv
    stopping = False

    def _term(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, _term)
    while not stopping:
        try:
            sc.tick()
        except Exception as e:  # noqa: BLE001 — 예상 못 한 오류도 사이드카를 죽이지 않음 (학습은 계속, 이 step만 미승인)
            log("sidecar.error", error=repr(e))
        if once or sc.stopped:
            break
        time.sleep(poll)
    publisher.flush(5.0)   # 보내지 못한 사건이 있으면 최대 5초 기다림
    # STOP 후에도 Pod가 끝날 때까지 남아 있으면 되지만, 여기서는 그대로 종료한다.


if __name__ == "__main__":
    main()
