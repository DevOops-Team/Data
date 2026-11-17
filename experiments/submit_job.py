"""학습 작업 1개를 Backend API로 제출 — 장애 시험 · 측정 확인용 (GPU-119).

투입기(K8s scheduling/submitter, 김성모)가 시나리오 CSV로 여러 작업을 내는 것과 달리, 이 스크립트는
같은 조건의 작업 하나를 손으로 반복해서 낼 때 쓴다 (E2E 장애 주입, 측정 스크립트 확인).

  python experiments/submit_job.py --model-dir ~/models/gpt2 --name e2e-fault-2 --steps 1500
  python experiments/submit_job.py --model-dir ~/models/gpt2 --name e2e-fault-2 --steps 1500 --exp-id e2e-1117 --dry-run

준비
- 모델 폴더: Hugging Face 모델 그대로 (config.json · model.safetensors · tokenizer 파일). gpt2 ≈ 550MB
    mkdir -p ~/models/gpt2 && cd ~/models/gpt2 && for f in config.json generation_config.json model.safetensors \\
      tokenizer.json tokenizer_config.json vocab.json merges.txt; do \\
      curl -sfLO https://huggingface.co/openai-community/gpt2/resolve/main/$f; done
- train.py: 이 레포 trainer/train.py (재개 · 진행률 · 지표가 들어 있는 최신본). 이미지에는 train.py가 없다

학습 데이터는 매번 같은 내용으로 만든다 (seed 고정). train.py는 토큰을 1024개씩 묶어 샘플 하나로 쓰므로
step 수에 맞춰 토큰 수로 크기를 정한다: 필요 토큰 = steps × gpus × batch × 1024 (1500 step · 2장 · 4 → 약 1230만, ≈95MB).

POST /api/jobs multipart 순서 (Backend internal/handler/job.go):
  spec(JSON: name · strategy · gpus · hyperparameters · exp_id) → model/<폴더>/<파일> → data/<파일>.jsonl → train/train.py

exp_id: Backend가 job.submitted payload와 JobSet 주석 devoops/exp-id에 넣는다. 투입기 없이 내면 experiment.started가
없어 job_runs.exp_id가 비어 있다 → 측정 때 `collector.measure --exp <id> --create-exp <이름>`이 실험 행을 만들고 연결한다.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import ssl
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO / "trainer" / "train.py"
API = os.environ.get("DEVOOPS_API", "https://172.30.0.125/api/jobs")
BLOCK = 1024             # trainer/train.py MAX_BLOCK_SIZE
TOKENS_PER_LINE = 97     # 아래 문장을 gpt2 토크나이저로 잰 평균 (11/17, 3000줄)
MARGIN = 1.3             # max_steps보다 데이터가 모자라 일찍 끝나지 않게

WORDS = ("checkpoint recovery cluster spot node failure generation fencing approved manifest shard rank "
         "gradient optimizer scheduler batch epoch token sequence model training dataset storage bucket "
         "kafka event collector measure downtime restart resume spare evacuate dispatcher queue").split()


def dataset_lines(steps: int, gpus: int, batch: int) -> int:
    return int(steps * gpus * batch * BLOCK / TOKENS_PER_LINE * MARGIN)


def make_dataset(path: Path, lines: int, seed: int = 7) -> None:
    """{"text": ...} 한 줄씩 (train.py text-jsonl). 같은 인자면 같은 파일."""
    rnd = random.Random(seed)
    with path.open("w") as f:
        for i in range(lines):
            sent = " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(40, 120)))
            f.write(json.dumps({"text": f"sample {i}: {sent}."}) + "\n")


def parts(spec: dict, model_dir: Path, data: Path, train_py: Path):
    yield "spec", "spec.json", "application/json", json.dumps(spec).encode()
    for p in sorted(model_dir.rglob("*")):
        if p.is_file():
            yield f"model/{model_dir.name}/{p.relative_to(model_dir)}", p.name, "application/octet-stream", p
    yield f"data/{data.name}", data.name, "application/octet-stream", data
    yield "train/train.py", "train.py", "text/x-python", train_py


def body(boundary: str, items) -> bytes:
    out = []
    for name, filename, ctype, payload in items:
        out.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                    f"Content-Type: {ctype}\r\n\r\n").encode())
        out.append(payload if isinstance(payload, bytes) else payload.read_bytes())
        out.append(b"\r\n")
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out)  # gpt2면 약 650MB. 맥 메모리로 충분


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="학습 작업 1개 제출 (GPU-119)")
    ap.add_argument("--model-dir", type=Path, required=True, help="Hugging Face 모델 폴더 (폴더 이름이 모델 이름)")
    ap.add_argument("--name", required=True, help="작업 이름 (화면 표시)")
    ap.add_argument("--steps", type=int, default=1500, help="max_steps. 1500 ≈ 학습 34분 (gpt2 · FSDP 2장, step당 1.32초)")
    ap.add_argument("--gpus", type=int, default=2, choices=[1, 2, 4])
    ap.add_argument("--strategy", default="fsdp", choices=["fsdp", "ddp"])
    ap.add_argument("--batch", type=int, default=4, help="batch_size_per_gpu")
    ap.add_argument("--exp-id", help="실험 ID (측정 때 --exp로 묶음)")
    ap.add_argument("--train-py", type=Path, default=TRAIN_PY)
    ap.add_argument("--work-dir", type=Path, default=Path.cwd(), help="학습 데이터 파일을 만들 곳")
    ap.add_argument("--dry-run", action="store_true", help="데이터만 만들고 spec 출력, 제출 안 함")
    a = ap.parse_args(argv)

    if not (a.model_dir / "config.json").is_file():
        ap.error(f"{a.model_dir}/config.json 없음 (Hugging Face 모델 폴더가 아님)")
    lines = dataset_lines(a.steps, a.gpus, a.batch)
    data = a.work_dir / f"submit-{a.steps}x{a.gpus}x{a.batch}.jsonl"
    make_dataset(data, lines)
    spec = {"name": a.name, "strategy": a.strategy, "gpus": a.gpus,
            "hyperparameters": {"epochs": 1, "max_steps": a.steps, "batch_size_per_gpu": a.batch}}
    if a.exp_id:
        spec["exp_id"] = a.exp_id
    print("spec:", json.dumps(spec, ensure_ascii=False), f"| data {data.name} {lines}줄 {data.stat().st_size // 2**20}MB")
    if a.dry_run:
        return 0

    boundary = uuid.uuid4().hex
    req = urllib.request.Request(API, data=body(boundary, parts(spec, a.model_dir, data, a.train_py)), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    ctx = ssl._create_unverified_context()  # 온프레 Frontend 자체 서명 인증서
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=900) as r:
            created = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(e.code, e.read().decode(), file=sys.stderr)
        return 1
    policy = created.get("config", {}).get("checkpoint_policy", {})
    print(f"제출됨: job {created['id']} ({created['jobsetName']}, {created['queue']}), "
          f"저장 간격 {policy.get('interval_seconds')}초")
    return 0


if __name__ == "__main__":
    sys.exit(main())
