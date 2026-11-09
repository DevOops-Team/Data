"""ckpt-init: 학습 컨테이너보다 먼저 한 번 실행되고 끝나는 init 컨테이너 (규약 7 · 9절).

순서
1. 환경 변수 세대 == DynamoDB 세대 확인 (다르면 시작 안 함)
2. owner 선점 (같은 세대가 두 클러스터에서 뜨는 것 방지)
3. config.json · 학습 데이터 다운로드 + 체크섬 확인
4. 승인본이 있으면 manifest의 파일 전부 다운로드 + 체크섬 확인
5. trainer가 읽을 재개 정보 기록: CKPT_DIR/resume.json, step 폴더, .done · .approved 표시

종료 코드
0 정상 · 2 세대 불일치 · 3 owner 선점 실패 · 4 다운로드 · 체크섬 실패 · 5 승인본이 현재 작업과 맞지 않음 · 6 항목 없음
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import urlparse

from . import layout
from .config import Config
from .fencing import FencingTable
from .log import log
from .storage import Store, sha256_file

EXIT_OK, EXIT_GEN, EXIT_OWNER, EXIT_DOWNLOAD, EXIT_MISMATCH, EXIT_NO_ITEM = 0, 2, 3, 4, 5, 6


def _s3_key(uri: str) -> tuple[str, str]:
    u = urlparse(uri)
    return u.netloc, u.path.lstrip("/")


def run(cfg: Config, fencing: FencingTable, store: Store) -> int:
    item = fencing.get(cfg.job_id)
    if item is None:
        log("init.no_item", job_id=cfg.job_id)
        return EXIT_NO_ITEM

    # 1. 세대 확인
    if item.generation != cfg.generation:
        log("init.generation_mismatch", env_gen=cfg.generation, ddb_gen=item.generation)
        return EXIT_GEN

    # 2. owner 선점
    if not fencing.claim_owner(cfg.job_id, cfg.generation, cfg.cluster):
        log("init.owner_claim_failed", cluster=cfg.cluster, owner=(fencing.get(cfg.job_id) or item).owner)
        return EXIT_OWNER
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
                store.download_verified(key, dest, cfg.config_sha256)
            else:
                store.s3.download_file(bucket, key, str(dest))
            log("init.config_ready", path=str(dest))

        # 3-2. 학습 데이터 (manifest 형식은 체크포인트와 같다고 가정 — 백엔드와 합의 필요)
        if cfg.dataset_uri:
            bucket, prefix = _s3_key(cfg.dataset_uri)
            prefix = prefix if prefix.endswith("/") else prefix + "/"
            mpath = cfg.data_dir / "manifest.json"
            mpath.parent.mkdir(parents=True, exist_ok=True)
            store.s3.download_file(bucket, prefix + "manifest.json", str(mpath))
            if cfg.dataset_manifest_sha256 and sha256_file(mpath) != cfg.dataset_manifest_sha256:
                raise ValueError("학습 데이터 manifest 체크섬 불일치")
            for f in json.loads(mpath.read_text())["files"]:
                store.download_verified(prefix + f["name"], cfg.data_dir / f["name"], f["sha256"], f["size_bytes"])
            log("init.dataset_ready", path=str(cfg.data_dir))

        # 4. 승인본
        if item.approved_step is None:
            if item.recovery_id:
                log("init.warn_recovery_without_checkpoint", recovery_id=item.recovery_id)  # 규약 10절: 경고 후 처음부터
            log("init.fresh_start")
            _write_resume(cfg.ckpt_dir, None, None)
            return EXIT_OK

        gen, step = item.approved_gen, item.approved_step
        step_dir = layout.local_step_dir(cfg.ckpt_dir, step)
        manifest = store.download_checkpoint(cfg.job_id, gen, step, step_dir)
    except Exception as e:  # noqa: BLE001 — 어떤 실패든 시작하지 않고 보고
        log("init.download_failed", error=str(e))
        return EXIT_DOWNLOAD

    # 승인본이 지금 작업과 맞는지 (규약 7절 3번)
    if cfg.config_sha256 and manifest.get("config_sha256") and manifest["config_sha256"] != cfg.config_sha256:
        log("init.config_mismatch", manifest=manifest["config_sha256"], env=cfg.config_sha256)
        return EXIT_MISMATCH
    if manifest.get("world_size") != cfg.world_size:
        log("init.world_size_mismatch", manifest=manifest.get("world_size"), env=cfg.world_size)
        return EXIT_MISMATCH

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


def main() -> None:
    cfg = Config.from_env()
    sys.exit(run(cfg, FencingTable(cfg.table, region=cfg.region), Store(cfg.bucket, region=cfg.region)))


if __name__ == "__main__":
    main()
