"""ckpt-init: 학습 컨테이너보다 먼저 한 번 실행되고 끝나는 init 컨테이너 (규약 7 · 9절).

순서
1. 환경 변수 세대 == DynamoDB 세대 확인 (다르면 시작 안 함)
2. owner 선점 (같은 세대가 두 클러스터에서 뜨는 것 방지)
3. config.json · 학습 데이터 다운로드 + 체크섬 확인
4. 승인본이 있으면 manifest의 파일 전부 다운로드 + 체크섬 확인
5. trainer가 읽을 재개 정보 기록: CKPT_DIR/resume.json, step 폴더, .done · .approved 표시

종료 코드
0 정상 · 2 세대 불일치 · 3 owner 선점 실패 · 4 연결 오류로 다운로드 실패(재시도 · 제한 시간 초과, 다시 띄우면 될 수 있음)
5 승인본이 현재 작업과 맞지 않음 · 6 항목 없음 · 7 승인본 손상(체크섬 · 크기 불일치, 파일 없음 — 같은 승인본으로는 안 됨)

실패 보고 (GPU-120, 사건은 전부 Kafka 기준)
- Kafka job.lifecycle에 job.init_failed 발행 → 수집기(DB · 화면) · 복구 컨트롤러가 받는다. KAFKA_BOOTSTRAP 없으면 안 보냄.
  payload: exit_code · reason · rank · checkpoint_gen · checkpoint_step · detail
  checkpoint_gen/step: 실제로 복원하려던 체크포인트 (코드 7이면 받다가 깨진 것). 복원 전에 실패하면 그때 DynamoDB 승인본
- 보조: /dev/termination-log(env TERMINATION_LOG)에 같은 JSON → Pod 상태 terminated.message (Kafka가 안 닿을 때 kubectl로 확인)

재시도: 일시 오류는 3회(5 → 10 → 20초), 다운로드 전체 제한 DOWNLOAD_DEADLINE_S(기본 600초).
복구 컨트롤러의 '복원 · 학습 진행 확인 30분' 안에 끝나도록 한다.

다운로드 감시 (GPU-263, ckpt/download_watch.py): Kueue 대기(20분) 안에 못 끝날 다운로드를 1분 안에 알아챈다.
  예상 시간이 여유 뺀 예산을 넘으면 job.init_slow 경고(계속 받음), Kueue 한계를 넘으면 실패 download_too_slow,
  60초 동안 0바이트면 실패 download_stalled. 실패는 바로 보고하고 프로세스를 끝낸다(종료 코드 4).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from . import layout
from . import events
from .config import Config
from .download_watch import DownloadWatch, Watchdog
from .fencing import FencingTable
from .log import log
from .retry import CorruptError, RetryPolicy, is_transient
from .storage import Store, sha256_file

EXIT_OK, EXIT_GEN, EXIT_OWNER, EXIT_DOWNLOAD, EXIT_MISMATCH, EXIT_NO_ITEM, EXIT_CORRUPT = 0, 2, 3, 4, 5, 6, 7


# 마지막 실패 (main이 Kafka · TERMINATION_LOG로 보고). run()이 시작할 때 비운다
FAILURE: dict = {}
TERMINATION_LIMIT = 4096  # 쿠버네티스 terminationMessage 한도 (바이트)
EVENT_TIMEOUT_MS = 20_000  # 실패 사건 전달 재시도 상한. init 컨테이너가 오래 붙잡혀 복구가 늦어지지 않게


def _fail(code: int, reason: str, **detail) -> int:
    """실패를 로그와 FAILURE에 남기고 종료 코드를 돌려준다 (GPU-120 termination-log)."""
    log(reason, **detail)
    FAILURE.update(exit_code=code, reason=reason.removeprefix("init."), detail=detail)
    return code


def failure_info(cfg: Config, code: int) -> dict:
    """실패 정보 (job.init_failed payload · termination-log 공통)."""
    return {"exit_code": code, "reason": FAILURE.get("reason", "unknown"), "job_id": cfg.job_id,
            "generation": cfg.generation, "cluster": cfg.cluster, "rank": cfg.rank,
            "checkpoint_gen": FAILURE.get("checkpoint_gen"), "checkpoint_step": FAILURE.get("checkpoint_step"),
            "detail": FAILURE.get("detail", {})}


def termination_message(cfg: Config, code: int) -> str:
    """Pod 상태(terminated.message)에 남는 실패 정보 JSON 한 줄 (4KB 이하). Kafka 사건의 보조."""
    body = failure_info(cfg, code)
    text = json.dumps(body, ensure_ascii=False, default=str)
    if len(text.encode()) > TERMINATION_LIMIT:  # 긴 오류 문구는 detail만 줄인다
        body["detail"] = {k: str(v)[:200] for k, v in body["detail"].items()}
        text = json.dumps(body, ensure_ascii=False, default=str)[:TERMINATION_LIMIT]
    return text


def _s3_key(uri: str) -> tuple[str, str]:
    u = urlparse(uri)
    return u.netloc, u.path.lstrip("/")


def run(cfg: Config, fencing: FencingTable, store: Store, policy: RetryPolicy | None = None, *,
        watch: DownloadWatch | None = None, on_watch_fail=None, on_watch_warn=None) -> int:
    """on_watch_fail(reason, detail): 감시가 실패를 판정했을 때 (기본: 보고 후 프로세스 종료).
    on_watch_warn(reason, detail): 경고일 때 (기본: job.init_slow 발행). 테스트는 둘 다 바꿔 끼운다."""
    policy = policy or RetryPolicy(deadline_s=float(os.environ.get("DOWNLOAD_DEADLINE_S", "600")))
    policy.started()
    FAILURE.clear()
    watch = watch or DownloadWatch(init_started=policy._t0)
    store.progress, store.watch = watch.add, watch
    dog = Watchdog(watch, on_watch_fail or (lambda r, d: _abort_download(cfg, r, d)),
                   on_watch_warn or (lambda r, d: _publish_slow(cfg, d)))
    dog.start()
    try:
        return _run(cfg, fencing, store, policy, watch)
    finally:
        dog.stop()
        store.progress = store.watch = None


def _abort_download(cfg: Config, reason: str, detail: dict) -> None:
    """감시 스레드에서: 실패를 보고하고 init을 바로 끝낸다 (download_file은 블로킹이라 끊을 수 없음)."""
    code = _fail(EXIT_DOWNLOAD, f"init.download_{reason}", transient=(reason == "stalled"), **detail)
    report_failure(cfg, code)
    os._exit(code)


def _publish_slow(cfg: Config, detail: dict) -> None:
    """job.init_slow 경고 (최선 노력). 학습은 그대로 받는다."""
    try:
        pub = events.from_env(timeout_ms=EVENT_TIMEOUT_MS)
        payload = dict(detail, rank=cfg.rank)
        ev = events.envelope(type_="job.init_slow", source="ckpt-init",
                             event_id=f"init-{cfg.job_id}-g{cfg.generation:04d}-r{cfg.rank}-slow-{detail['phase']}",
                             cluster=cfg.cluster, job_id=cfg.job_id, generation=cfg.generation, payload=payload)
        pub.publish(topic=os.environ.get("KAFKA_TOPIC_LIFECYCLE", "job.lifecycle"), key=cfg.job_id, value=ev)
        pub.flush(5.0)
    except Exception as e:  # noqa: BLE001
        log("init.slow_event_failed", error=repr(e))


def _run(cfg: Config, fencing: FencingTable, store: Store, policy: RetryPolicy, watch: DownloadWatch) -> int:
    try:
        item = policy.call(lambda: fencing.get(cfg.job_id))
    except Exception as e:  # noqa: BLE001
        return _fail(EXIT_DOWNLOAD, "init.ddb_unreachable", error=repr(e))
    if item is None:
        return _fail(EXIT_NO_ITEM, "init.no_item", job_id=cfg.job_id)
    FAILURE.update(checkpoint_gen=item.approved_gen, checkpoint_step=item.approved_step, ddb_generation=item.generation)

    # 1. 세대 확인
    if item.generation != cfg.generation:
        return _fail(EXIT_GEN, "init.generation_mismatch", env_gen=cfg.generation, ddb_gen=item.generation)

    # 2. owner 선점 (조건 실패는 재시도하지 않음, 연결 오류만 재시도)
    try:
        claimed = policy.call(lambda: fencing.claim_owner(cfg.job_id, cfg.generation, cfg.cluster))
    except Exception as e:  # noqa: BLE001
        return _fail(EXIT_DOWNLOAD, "init.ddb_unreachable", error=repr(e))
    if not claimed:
        return _fail(EXIT_OWNER, "init.owner_claim_failed", cluster=cfg.cluster,
                     owner=(fencing.get(cfg.job_id) or item).owner)
    log("init.owner_claimed", cluster=cfg.cluster, gen=cfg.generation)

    try:
        # 3-1. config.json
        if cfg.config_uri:
            bucket, key = _s3_key(cfg.config_uri)
            dest = cfg.code_dir / "config.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if bucket != store.bucket:
                raise ValueError(f"CONFIG_URI 버킷이 CKPT_BUCKET과 다름: {bucket}")
            if cfg.config_sha256:
                store.download_with_retry(key, dest, cfg.config_sha256, None, policy)
            else:
                policy.call(lambda: store.s3.download_file(bucket, key, str(dest)))
            log("init.config_ready", path=str(dest))

        # 3-2. 학습 데이터 (manifest 형식은 체크포인트와 같다고 가정 — 백엔드와 합의 필요)
        if cfg.dataset_uri:
            bucket, prefix = _s3_key(cfg.dataset_uri)
            prefix = prefix if prefix.endswith("/") else prefix + "/"
            mpath = cfg.data_dir / "manifest.json"
            mpath.parent.mkdir(parents=True, exist_ok=True)
            policy.call(lambda: store.s3.download_file(bucket, prefix + "manifest.json", str(mpath)))
            if cfg.dataset_manifest_sha256 and sha256_file(mpath) != cfg.dataset_manifest_sha256:
                raise CorruptError("학습 데이터 manifest 체크섬 불일치")
            files = json.loads(mpath.read_text())["files"]
            watch.begin("dataset", sum(int(f.get("size_bytes") or 0) for f in files))
            for f in files:
                store.download_with_retry(prefix + f["name"], cfg.data_dir / f["name"], f["sha256"], f["size_bytes"], policy)
            watch.end()
            log("init.dataset_ready", path=str(cfg.data_dir))

        # 4. 승인본
        if item.approved_step is None:
            if item.recovery_id:
                log("init.warn_recovery_without_checkpoint", recovery_id=item.recovery_id)  # 규약 10절: 경고 후 처음부터
            log("init.fresh_start")
            _write_resume(cfg.ckpt_dir, None, None)
            return EXIT_OK

        gen, step = item.approved_gen, item.approved_step
        FAILURE.update(checkpoint_gen=gen, checkpoint_step=step)  # 실제로 복원하려는 체크포인트 (코드 7이면 이게 깨진 것)
        step_dir = layout.local_step_dir(cfg.ckpt_dir, step)
        manifest = store.download_checkpoint(cfg.job_id, gen, step, step_dir, policy)   # 감시 단계 시작은 store가
        watch.end()
    except CorruptError as e:
        # 같은 승인본으로는 안 됨 → 복구 컨트롤러가 prev_approved 등 판단
        return _fail(EXIT_CORRUPT, "init.corrupt", error=str(e))
    except Exception as e:  # noqa: BLE001 — 그 밖의 실패도 시작하지 않고 보고
        return _fail(EXIT_DOWNLOAD, "init.download_failed", error=repr(e), transient=is_transient(e))

    # 승인본이 지금 작업과 맞는지 (규약 7절 3번)
    if cfg.config_sha256 and manifest.get("config_sha256") and manifest["config_sha256"] != cfg.config_sha256:
        return _fail(EXIT_MISMATCH, "init.config_mismatch", manifest=manifest["config_sha256"], env=cfg.config_sha256)
    if manifest.get("world_size") != cfg.world_size:
        return _fail(EXIT_MISMATCH, "init.world_size_mismatch", manifest=manifest.get("world_size"), env=cfg.world_size)

    layout.done_marker(cfg.ckpt_dir, step).touch()
    layout.approved_marker(cfg.ckpt_dir, step).touch()
    _write_resume(cfg.ckpt_dir, gen, step)
    log("init.resume_ready", approved_gen=gen, approved_step=step, files=len(manifest["files"]))
    return EXIT_OK


def _write_resume(ckpt_dir: Path, gen: int | None, step: int | None) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    body = {"resume": step is not None, "approved_gen": gen, "approved_step": step,
            "step_dir": layout.step_name(step) if step is not None else None,
            "start_step": (step + 1) if step is not None else 1}
    (ckpt_dir / "resume.json").write_text(json.dumps(body, indent=2))


def publish_failure(cfg: Config, code: int, publisher: events.Publisher) -> None:
    """job.lifecycle에 job.init_failed (최선 노력: 못 보내도 종료 코드 · termination-log는 그대로)."""
    info = failure_info(cfg, code)
    payload = {k: v for k, v in info.items() if k not in ("job_id", "generation", "cluster")}
    ev = events.envelope(type_="job.init_failed", source="ckpt-init",
                         event_id=events.init_failed_id(cfg.job_id, cfg.generation, cfg.rank, info["reason"]),
                         cluster=cfg.cluster, job_id=cfg.job_id, generation=cfg.generation, payload=payload)
    publisher.publish(topic=os.environ.get("KAFKA_TOPIC_LIFECYCLE", "job.lifecycle"), key=cfg.job_id, value=ev)
    publisher.flush(EVENT_TIMEOUT_MS / 1000)


def report_failure(cfg: Config, code: int) -> None:
    """실패 이유 · 세대 · 복원하려던 체크포인트 전달 (GPU-120): Kafka job.init_failed + termination-log."""
    try:
        publish_failure(cfg, code, events.from_env(timeout_ms=EVENT_TIMEOUT_MS))
    except Exception as e:  # noqa: BLE001 — 사건 발행 실패로 종료 코드가 바뀌면 안 됨
        log("init.event_failed", error=repr(e))
    try:
        Path(os.environ.get("TERMINATION_LOG", "/dev/termination-log")).write_text(termination_message(cfg, code))
    except OSError as e:
        log("init.termination_log_failed", error=repr(e))


def main() -> None:
    cfg = Config.from_env()
    code = run(cfg, FencingTable(cfg.table, region=cfg.region), Store(cfg.bucket, region=cfg.region))
    if code != EXIT_OK:
        report_failure(cfg, code)
    sys.exit(code)


if __name__ == "__main__":
    main()
