"""수동 장애 주입 기록 (GPU-239, ADR-33): 기록표의 주입 시각을 data.injections에 넣는다 → v_incidents의 장애 시작.

장애는 사람이 CLI로 직접 주입하므로(trainer kill -9 · dcgmi Xid 등) 주입 사건이 Kafka로 오지 않는다. 누른 시각을
`date '+%T'`로 적어 두었다가 여기로 넣는다. 데이터 DB 표 주인(data_app)으로 실행 (접속은 libpq 환경 변수).

  python -m collector.injection add --at "2026-11-20 14:03:21+09" --node ip-172-28-74-75.us-east-2.compute.internal \\
      --kind trainer_kill --job eb497561 --note "rank 1 워커 kill -9"
  python -m collector.injection list                 # 최근 20건
  python -m collector.injection delete 3             # 잘못 넣은 기록 지우기

시각에 시간대가 없으면 KST로 본다 (기록표는 KST).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
KINDS = ("trainer_kill", "xid", "node_stop", "network", "spot", "other")


def parse_at(s: str) -> datetime:
    t = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=KST)


def add(cur, *, at: datetime, node: str, kind: str, job_id: str | None = None, note: str | None = None) -> int:
    if kind not in KINDS:
        raise ValueError(f"kind는 {', '.join(KINDS)} 중 하나")
    if not node.strip():
        raise ValueError("node가 비었음")
    return cur.execute(
        "INSERT INTO data.injections (injected_at, node, job_id, kind, note) VALUES (%s,%s,%s,%s,%s) "
        "RETURNING injection_id", (at, node.strip(), job_id or None, kind, note)).fetchone()[0]


def recent(cur, limit: int = 20) -> list[dict]:
    rows = cur.execute(
        "SELECT injection_id, injected_at, node, job_id, kind, note FROM data.injections "
        "ORDER BY injected_at DESC LIMIT %s", (limit,)).fetchall()
    keys = ("injection_id", "injected_at", "node", "job_id", "kind", "note")
    return [dict(zip(keys, r)) for r in rows]


def delete(cur, injection_id: int) -> bool:
    return cur.execute("DELETE FROM data.injections WHERE injection_id = %s", (injection_id,)).rowcount == 1


def main(argv=None) -> int:
    import psycopg

    ap = argparse.ArgumentParser(description="수동 장애 주입 기록 → data.injections (GPU-239)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="주입 1건 기록")
    a.add_argument("--at", required=True, help="주입 시각 (ISO, 시간대 없으면 KST)")
    a.add_argument("--node", required=True, help="주입한 노드 이름 (kubectl get pods -o wide의 NODE)")
    a.add_argument("--kind", required=True, choices=KINDS)
    a.add_argument("--job", help="작업 ID (알면)")
    a.add_argument("--note")
    sub.add_parser("list", help="최근 20건")
    d = sub.add_parser("delete", help="기록 지우기")
    d.add_argument("injection_id", type=int)
    args = ap.parse_args(argv)

    with psycopg.connect() as conn, conn.transaction():
        cur = conn.cursor()
        if args.cmd == "add":
            out = {"injection_id": add(cur, at=parse_at(args.at), node=args.node, kind=args.kind,
                                       job_id=args.job, note=args.note)}
        elif args.cmd == "list":
            out = recent(cur)
        else:
            out = {"deleted": delete(cur, args.injection_id)}
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
