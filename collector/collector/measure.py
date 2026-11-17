"""측정 (GPU-187, ADR-15): 실험의 작업마다 MTTD · MTTR · ETTR · 날아간 step을 계산해 data.results에 쓴다.

데이터 DB 표 주인(data_app)으로 실행한다 (data_collector는 results 권한 없음). 같은 입력이면 같은 값 (다시 돌려도 덮어씀).

  python -m collector.measure --exp e2e-1117                       # job_runs.exp_id 또는 제출 사건 exp_id = e2e-1117인 작업
  python -m collector.measure --exp e2e-1117 --create-exp "11/17 E2E"   # 투입기 없이 exp_id만 붙여 제출한 경우 (실험 행 만들고 연결)
  python -m collector.measure --exp e2e-1117 --jobs 0cb3b0c3,3989c020 --create-exp "11/17 E2E"   # exp_id 연결 전
  python -m collector.measure --exp e2e-1117 --since 2026-11-17T09:00+09:00 --until 2026-11-17T18:00+09:00 --create-exp "11/17"
  ... --dry-run                                                    # 계산만, 쓰지 않음

접속은 libpq 환경 변수(PGHOST · PGDATABASE · PGUSER …).

정의 (ADR-15, 데이터 DB 스키마):
  - detect_s · classify_s · recover_s: 그 작업 장애들의 v_incidents 평균. MTTD = detect_s, MTTR = classify_s + recover_s
  - lost_steps: 세대가 바뀔 때마다 (옛 세대가 마지막으로 보고한 step − 재개에 쓴 승인 step)의 합.
    마지막 step은 job.progress 사건(최소 30초 간격)이라 실제보다 조금 작을 수 있다
  - wait_s: 첫 제출 → 첫 학습 시작 (큐 대기)
  - ettr: (할당 − 정지 − 재계산) ÷ 할당. 할당 = 첫 시작 → 마지막 종료, 정지 = 세대가 바뀔 때마다 장애 시작(주입 시각,
    없으면 옛 세대 종료) → 새 세대 시작, 재계산 = 날아간 step × step 시간. 아직 안 끝난 작업은 비움
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime

CALC_VERSION = "measure-v1 (2026-11-17)"


@dataclass
class JobResult:
    job_id: str
    generations: int
    incidents: int
    detect_s: float | None
    classify_s: float | None
    recover_s: float | None
    lost_steps: int | None
    wait_s: float | None
    ettr: float | None
    downtime_s: float | None
    recompute_s: float | None


def _avg(xs):
    xs = [float(x) for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def _secs(a: datetime | None, b: datetime | None) -> float | None:
    return (b - a).total_seconds() if a is not None and b is not None else None


def compute_job(cur, job_id: str) -> JobResult | None:
    runs = cur.execute(
        "SELECT generation, submitted_at, started_at, ended_at FROM data.job_runs WHERE job_id = %s ORDER BY generation",
        (job_id,)).fetchall()
    if not runs:
        return None
    inc = cur.execute(
        "SELECT injected_at, detect_s, classify_s, recover_s FROM data.v_incidents WHERE job_id = %s ORDER BY injected_at",
        (job_id,)).fetchall()
    step_seconds = cur.execute(
        "SELECT (payload->>'step_seconds')::float FROM data.events WHERE type = 'job.progress' AND job_id = %s "
        "AND payload ? 'step_seconds' ORDER BY occurred_at DESC LIMIT 1", (job_id,)).fetchone()
    step_seconds = step_seconds[0] if step_seconds else None

    lost, downtime = 0, 0.0
    lost_known = True
    for prev, nxt in zip(runs, runs[1:]):
        gen, start_next = prev[0], nxt[2]
        last = cur.execute(
            "SELECT max((payload->>'step')::bigint) FROM data.events WHERE type = 'job.progress' AND job_id = %s "
            "AND generation = %s", (job_id, gen)).fetchone()[0]
        resume = cur.execute(
            "SELECT max(step) FROM data.checkpoints WHERE job_id = %s AND generation <= %s "
            "AND (%s::timestamptz IS NULL OR approved_at <= %s)", (job_id, gen, start_next, start_next)).fetchone()[0] or 0
        if last is None:
            lost_known = False
        else:
            lost += max(0, last - resume)
        # 정지: 이 세대 교체를 일으킨 장애의 시작(주입 시각) → 새 세대 시작. 장애 기록이 없으면 옛 세대 종료부터
        fault = [i[0] for i in inc if prev[2] is not None and start_next is not None and prev[2] <= i[0] <= start_next]
        gap = _secs(fault[-1] if fault else prev[3], start_next)
        if gap is not None:
            downtime += max(0.0, gap)

    first, final = runs[0], runs[-1]
    alloc = _secs(first[2], final[3])
    recompute = None
    if lost_known:
        recompute = 0.0 if lost == 0 else (lost * step_seconds if step_seconds is not None else None)
    ettr = None
    if alloc and alloc > 0 and recompute is not None:
        ettr = round(max(0.0, alloc - downtime - recompute) / alloc, 4)
    return JobResult(
        job_id=job_id, generations=len(runs), incidents=len(inc),
        detect_s=_avg(i[1] for i in inc), classify_s=_avg(i[2] for i in inc), recover_s=_avg(i[3] for i in inc),
        lost_steps=lost if lost_known else None, wait_s=_secs(first[1], first[2]), ettr=ettr,
        downtime_s=round(downtime, 3), recompute_s=round(recompute, 3) if recompute is not None else None)


def select_jobs(cur, exp_id: str, jobs: list[str] | None, since: str | None, until: str | None) -> list[str]:
    if jobs:
        return jobs
    if since or until:
        rows = cur.execute(
            "SELECT job_id FROM data.job_runs GROUP BY job_id HAVING min(submitted_at) >= COALESCE(%s::timestamptz, '-infinity') "
            "AND min(submitted_at) < COALESCE(%s::timestamptz, 'infinity') ORDER BY min(submitted_at)", (since, until)).fetchall()
    else:
        # job_runs.exp_id가 아직 비어 있어도(실험 행보다 제출이 먼저) 제출 사건의 exp_id로 찾는다
        rows = cur.execute(
            """SELECT job_id FROM data.job_runs WHERE exp_id = %s
               UNION SELECT job_id FROM data.events WHERE type = 'job.submitted' AND payload->>'exp_id' = %s
               ORDER BY 1""", (exp_id, exp_id)).fetchall()
    return [r[0] for r in rows]


def link_jobs(cur, exp_id: str) -> int:
    """제출 사건에 exp_id가 있는데 job_runs.exp_id가 빈 작업을 실험에 연결한다.

    수집기는 job.submitted를 받을 때 실험 행이 없으면 exp_id를 비워 두고, experiment.started가 오면 채운다
    (collector/store.py). 투입기 없이 제출하고 measure --create-exp로 실험 행을 만든 경우는 experiment.started가
    없어서 여기서 같은 방식으로 채운다. 세대 2 이후 줄도 같은 작업이면 같이 채운다.
    """
    cur.execute(
        """UPDATE data.job_runs jr SET exp_id = %s
           FROM data.events ev
           WHERE ev.type = 'job.submitted' AND ev.payload->>'exp_id' = %s
             AND jr.job_id = ev.job_id AND jr.exp_id IS NULL""", (exp_id, exp_id))
    return cur.rowcount


def summary(results: list[JobResult]) -> dict:
    """실험 단위 발표 지표 (ADR-15): 장애 수로 가중한 MTTD · MTTR, 작업 평균 ETTR, 날아간 step 합."""
    inc = [r for r in results if r.incidents]
    w = sum(r.incidents for r in inc)

    def wavg(f):
        xs = [(f(r), r.incidents) for r in inc if f(r) is not None]
        return round(sum(v * n for v, n in xs) / sum(n for _, n in xs), 3) if xs else None
    return {
        "jobs": len(results), "incidents": w,
        "mttd_s": wavg(lambda r: r.detect_s),
        "mttr_s": wavg(lambda r: (r.classify_s + r.recover_s) if r.classify_s is not None and r.recover_s is not None else None),
        "ettr": _avg(r.ettr for r in results),
        "lost_steps": sum(r.lost_steps for r in results if r.lost_steps is not None),
    }


def write(cur, exp_id: str, results: list[JobResult]) -> None:
    for r in results:
        cur.execute(
            """INSERT INTO data.results (exp_id, job_id, ettr, detect_s, classify_s, recover_s, lost_steps, wait_s,
                                         incidents, calc_version, computed_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
               ON CONFLICT (exp_id, job_id) DO UPDATE SET
                 ettr = EXCLUDED.ettr, detect_s = EXCLUDED.detect_s, classify_s = EXCLUDED.classify_s,
                 recover_s = EXCLUDED.recover_s, lost_steps = EXCLUDED.lost_steps, wait_s = EXCLUDED.wait_s,
                 incidents = EXCLUDED.incidents, calc_version = EXCLUDED.calc_version, computed_at = now()""",
            (exp_id, r.job_id, r.ettr, r.detect_s, r.classify_s, r.recover_s, r.lost_steps, r.wait_s,
             r.incidents, CALC_VERSION))


def run(conn, exp_id: str, *, jobs=None, since=None, until=None, create_exp=None, scenario="adhoc",
        policy="ours", dry_run=False) -> dict:
    with conn.transaction():
        cur = conn.cursor()
        exists = cur.execute("SELECT 1 FROM data.experiments WHERE exp_id = %s", (exp_id,)).fetchone()
        if not exists:
            if not create_exp:
                raise SystemExit(f"실험 {exp_id} 없음: experiment.started 사건이 없으면 --create-exp 이름 으로 만든다")
            if not dry_run:
                cur.execute("INSERT INTO data.experiments (exp_id, name, scenario, policy, spare_enabled, note) "
                            "VALUES (%s,%s,%s,%s,true,%s)", (exp_id, create_exp, scenario, policy, "measure --create-exp"))
        linked = link_jobs(cur, exp_id) if not dry_run else 0
        ids = select_jobs(cur, exp_id, jobs, since, until)
        results = [r for r in (compute_job(cur, j) for j in ids) if r is not None]
        if not dry_run:
            write(cur, exp_id, results)
    return {"exp_id": exp_id, "calc_version": CALC_VERSION, "dry_run": dry_run, "linked_job_runs": linked,
            "summary": summary(results), "jobs": [asdict(r) for r in results]}


def main(argv=None) -> int:
    import psycopg

    ap = argparse.ArgumentParser(description="측정값 계산 → data.results (GPU-187)")
    ap.add_argument("--exp", required=True, help="실험 ID")
    ap.add_argument("--jobs", help="작업 ID 쉼표 목록 (exp_id 연결 전)")
    ap.add_argument("--since", help="첫 제출 시각 하한 (ISO)")
    ap.add_argument("--until", help="첫 제출 시각 상한 (ISO)")
    ap.add_argument("--create-exp", metavar="NAME", help="실험 행이 없으면 이 이름으로 만든다")
    ap.add_argument("--scenario", default="adhoc")
    ap.add_argument("--policy", default="ours", choices=["ours", "naive", "hyperpod", "manual"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    with psycopg.connect() as conn:
        out = run(conn, a.exp, jobs=a.jobs.split(",") if a.jobs else None, since=a.since, until=a.until,
                  create_exp=a.create_exp, scenario=a.scenario, policy=a.policy, dry_run=a.dry_run)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
