"""경로 규칙 (규약 3 · 4절)."""
from __future__ import annotations

from pathlib import Path


def step_name(step: int) -> str:
    return f"step-{step:08d}"


def gen_name(gen: int) -> str:
    return f"gen-{gen:04d}"


def s3_step_prefix(job_id: str, gen: int, step: int) -> str:
    """jobs/{job_id}/gen-0002/step-00001200/"""
    return f"jobs/{job_id}/{gen_name(gen)}/{step_name(step)}/"


def s3_manifest_key(job_id: str, gen: int, step: int) -> str:
    return s3_step_prefix(job_id, gen, step) + "manifest.json"


def s3_rank_list_key(job_id: str, gen: int, step: int, rank: int) -> str:
    return s3_step_prefix(job_id, gen, step) + f"ranks/rank-{rank:04d}.json"


def local_step_dir(ckpt_dir: Path, step: int) -> Path:
    return ckpt_dir / step_name(step)


def done_marker(ckpt_dir: Path, step: int) -> Path:
    return ckpt_dir / f"{step_name(step)}.done"


def approved_marker(ckpt_dir: Path, step: int) -> Path:
    return ckpt_dir / f"{step_name(step)}.approved"


def stop_file(ckpt_dir: Path) -> Path:
    return ckpt_dir / "STOP"


def parse_done(name: str) -> int | None:
    """'step-00001200.done' -> 1200"""
    if name.startswith("step-") and name.endswith(".done"):
        core = name[len("step-") : -len(".done")]
        if core.isdigit():
            return int(core)
    return None
