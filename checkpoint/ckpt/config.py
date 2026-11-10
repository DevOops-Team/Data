"""환경 변수 → 설정 (규약 12절).

S3 · DynamoDB 접속 주소는 boto3 표준 환경 변수 AWS_ENDPOINT_URL로 바꾼다.
로컬(moto) 테스트 때만 설정하고, EKS에서는 비워 둔다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _req(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"환경 변수 {name}가 없습니다")
    return v


@dataclass(frozen=True)
class Config:
    job_id: str
    generation: int
    cluster: str
    bucket: str
    table: str
    ckpt_dir: Path
    code_dir: Path
    data_dir: Path
    world_size: int
    rank: int
    checkpoint_format: str
    config_uri: str | None
    config_sha256: str | None
    dataset_uri: str | None
    dataset_manifest_sha256: str | None
    region: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            job_id=_req("JOB_ID"),
            generation=int(_req("GENERATION")),
            cluster=_req("CLUSTER_NAME"),
            bucket=_req("CKPT_BUCKET"),
            table=_req("FENCING_TABLE"),
            ckpt_dir=Path(os.environ.get("CKPT_DIR", "/ckpt")),
            code_dir=Path(os.environ.get("CODE_DIR", "/code")),
            data_dir=Path(os.environ.get("DATA_DIR", "/data")),
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            rank=int(os.environ.get("RANK", "0")),
            checkpoint_format=os.environ.get("CHECKPOINT_FORMAT", "fsdp-dcp"),
            config_uri=os.environ.get("CONFIG_URI") or None,
            config_sha256=os.environ.get("CONFIG_SHA256") or None,
            dataset_uri=os.environ.get("DATASET_URI") or None,
            dataset_manifest_sha256=os.environ.get("DATASET_MANIFEST_SHA256") or None,
            region=os.environ.get("AWS_REGION", "us-east-2"),
        )
