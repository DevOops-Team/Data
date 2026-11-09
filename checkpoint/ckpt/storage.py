"""S3 업로드 · 다운로드 · manifest (규약 4 · 6 · 7절)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import boto3

from . import layout
from .fencing import now_iso
from .log import log
from .retry import CorruptError, RetryPolicy, is_not_found

CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(CHUNK):
            h.update(b)
    return h.hexdigest()


@dataclass
class FileEntry:
    name: str          # step 폴더 기준 상대 경로 (예: "__0_0.distcp", ".metadata")
    size_bytes: int
    sha256: str


def list_step_files(step_dir: Path) -> list[FileEntry]:
    out = []
    for p in sorted(step_dir.rglob("*")):
        if p.is_file():
            out.append(FileEntry(str(p.relative_to(step_dir)), p.stat().st_size, sha256_file(p)))
    return out


class Store:
    def __init__(self, bucket: str, client=None, region: str = "us-east-2", retry: RetryPolicy | None = None):
        self.bucket = bucket
        self.s3 = client or boto3.client("s3", region_name=region)
        self.retry = retry or RetryPolicy()

    # ---------- 공통 ----------
    def put_json(self, key: str, obj) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=json.dumps(obj, ensure_ascii=False, indent=2).encode(),
                           ContentType="application/json")

    def get_json(self, key: str):
        return json.loads(self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read())

    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def head_size(self, key: str) -> int | None:
        try:
            return self.s3.head_object(Bucket=self.bucket, Key=key)["ContentLength"]
        except self.s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    def download_verified(self, key: str, dest: Path, sha256: str, size: int | None = None) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        try:
            self.s3.download_file(self.bucket, key, str(tmp))
        except Exception as e:  # noqa: BLE001
            if is_not_found(e):
                raise CorruptError(f"파일 없음: {key}") from e
            raise
        if size is not None and tmp.stat().st_size != size:
            tmp.unlink()
            raise CorruptError(f"크기 불일치: {key}")
        if sha256_file(tmp) != sha256:
            tmp.unlink()
            raise CorruptError(f"체크섬 불일치: {key}")
        tmp.replace(dest)

    def download_with_retry(self, key: str, dest: Path, sha256: str, size: int | None, policy: RetryPolicy) -> None:
        """일시 오류는 policy대로 재시도. 손상은 1번 더 받아 보고 또 틀리면 CorruptError."""
        for attempt in (1, 2):
            try:
                policy.call(lambda: self.download_verified(key, dest, sha256, size),
                            on_retry=lambda n, e: log("storage.download_retry", key=key, attempt=n, error=repr(e)))
                return
            except CorruptError as e:
                if attempt == 2:
                    raise
                log("storage.download_redo_corrupt", key=key, error=str(e))

    # ---------- 사이드카: 내 rank 파일 업로드 ----------
    def upload_rank(self, job_id: str, gen: int, step: int, rank: int, step_dir: Path) -> list[FileEntry]:
        prefix = layout.s3_step_prefix(job_id, gen, step)
        entries = list_step_files(step_dir)
        for e in entries:   # 큰 파일은 자동 멀티파트. 일시 오류는 파일 단위로 재시도
            self.retry.call(lambda e=e: self.s3.upload_file(str(step_dir / e.name), self.bucket, prefix + e.name),
                            on_retry=lambda n, err, e=e: log("storage.upload_retry", file=e.name, attempt=n, error=repr(err)))
        body = {"rank": rank, "files": [asdict(e) for e in entries], "uploaded_at": now_iso()}
        self.retry.call(lambda: self.put_json(layout.s3_rank_list_key(job_id, gen, step, rank), body))
        return entries

    def rank_lists(self, job_id: str, gen: int, step: int, world_size: int) -> list[dict] | None:
        """rank 목록이 WORLD_SIZE개 다 있으면 반환, 아니면 None."""
        out = []
        for r in range(world_size):
            key = layout.s3_rank_list_key(job_id, gen, step, r)
            if not self.exists(key):
                return None
            out.append(self.get_json(key))
        return out

    # ---------- rank 0 사이드카: manifest ----------
    def build_and_put_manifest(self, job_id: str, gen: int, step: int, world_size: int, fmt: str,
                               config_sha256: str | None, image_digest: str | None,
                               rank_lists: Iterable[dict]) -> dict:
        prefix = layout.s3_step_prefix(job_id, gen, step)
        files: dict[str, dict] = {}
        for rl in rank_lists:
            for f in rl["files"]:
                prev = files.get(f["name"])
                if prev and prev["sha256"] != f["sha256"]:
                    raise ValueError(f"rank 간 같은 이름 다른 내용: {f['name']}")
                files[f["name"]] = f
        # 검증: S3에 실제로 올라간 크기가 목록과 같은지
        for f in files.values():
            size = self.head_size(prefix + f["name"])
            if size != f["size_bytes"]:
                raise ValueError(f"S3 크기 불일치: {f['name']} (목록 {f['size_bytes']}, S3 {size})")
        manifest = {
            "job_id": job_id, "generation": gen, "step": step, "format": fmt, "world_size": world_size,
            "config_sha256": config_sha256, "image_digest": image_digest,
            "files": sorted(files.values(), key=lambda x: x["name"]),
            "created_at": now_iso(),
        }
        self.put_json(layout.s3_manifest_key(job_id, gen, step), manifest)  # manifest는 마지막에 쓴다
        return manifest

    # ---------- ckpt-init: 승인본 다운로드 ----------
    def download_checkpoint(self, job_id: str, gen: int, step: int, dest_dir: Path,
                            policy: RetryPolicy | None = None) -> dict:
        policy = policy or self.retry
        mkey = layout.s3_manifest_key(job_id, gen, step)
        try:
            manifest = policy.call(lambda: self.get_json(mkey))
        except Exception as e:  # noqa: BLE001
            if is_not_found(e):
                raise CorruptError(f"승인본 manifest 없음: {mkey}") from e
            raise
        prefix = layout.s3_step_prefix(job_id, gen, step)
        for f in manifest["files"]:
            self.download_with_retry(prefix + f["name"], dest_dir / f["name"], f["sha256"], f["size_bytes"], policy)
        return manifest
