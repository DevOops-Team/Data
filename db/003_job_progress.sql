-- 003_job_progress.sql — 학습 진행률 · 화면용 작업 상태 뷰 (GPU-141, 2026-11-12)
-- 실행: data_app으로 (테이블 소유자), 002_roles.sql 다음.  psql -v ON_ERROR_STOP=1 -1 -d gpuf_data -f 003_job_progress.sql
-- 다시 실행해도 안전 (IF NOT EXISTS · CREATE OR REPLACE).
--
-- 배경: 진행률을 DynamoDB devoops-prod-job-progress · progress-reporter 대신
--   ckpt-sidecar(rank 0) → Kafka job.lifecycle `job.progress` → 수집기 → data.job_progress 로 받는다.
--   Backend는 화면 상태 · 진행률을 v_job_status 하나에서 읽는다 (004_backend_role.sql의 backend_ro).

SET search_path = data;

-- 1) 작업당 최신 진행률 1행. 더 새로운 (세대, 시각)일 때만 덮어쓴다 (수집기 upsert)
CREATE TABLE IF NOT EXISTS job_progress (
  job_id                 text             PRIMARY KEY,
  generation             integer          NOT NULL,
  status                 text,                                  -- train.py: RUNNING · COMPLETED · FAILED
  step                   bigint           NOT NULL,
  total_steps            bigint,
  loss                   double precision,
  last_checkpoint_step   bigint,                                -- train.py가 저장한 step (승인 전일 수 있음)
  checkpoint_every_steps integer,
  step_seconds           double precision,
  cluster                text,
  event_id               text             NOT NULL,             -- 마지막으로 반영한 job.progress
  updated_at             timestamptz      NOT NULL              -- 사건 봉투 occurred_at (= progress.json 갱신 시각)
);

-- 2) 화면용 작업 상태: 작업마다 최신 세대 1행 + 진행률 + 마지막 "승인된" 체크포인트
CREATE OR REPLACE VIEW v_job_status AS
SELECT jr.job_id,
       jr.generation,
       jr.exp_id,
       jr.jobset_name,
       jr.cluster,
       jr.queue,
       jr.gpus,
       jr.strategy,
       jr.submitted_at,
       jr.started_at,
       jr.ended_at,
       jr.end_reason,
       CASE
         WHEN jr.end_reason IS NOT NULL  THEN upper(jr.end_reason)     -- COMPLETED · FAILED · FENCED · STOPPED · CANCELLED
         WHEN jr.started_at IS NOT NULL  THEN 'RUNNING'
         ELSE 'PENDING'
       END                                         AS state,
       runs.n - 1                                  AS recovery_count,
       p.status                                    AS progress_status,
       p.step,
       p.total_steps,
       p.loss,
       p.step_seconds,
       p.checkpoint_every_steps,
       p.updated_at                                AS progress_updated_at,
       ck.step                                     AS last_approved_step,
       ck.generation                               AS last_approved_generation,
       ck.approved_at                              AS last_approved_at
FROM (SELECT DISTINCT ON (job_id) * FROM job_runs ORDER BY job_id, generation DESC) jr
JOIN (SELECT job_id, count(*) AS n FROM job_runs GROUP BY job_id) runs ON runs.job_id = jr.job_id
LEFT JOIN job_progress p ON p.job_id = jr.job_id AND p.generation = jr.generation
LEFT JOIN LATERAL (
  SELECT c.step, c.generation, c.approved_at FROM checkpoints c
  WHERE c.job_id = jr.job_id ORDER BY c.generation DESC, c.step DESC LIMIT 1
) ck ON true;

-- 3) 권한: 수집기는 쓰기, 대시보드(grafana_ro)는 읽기 (계정이 있을 때만)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'data_collector') THEN
    GRANT SELECT, INSERT, UPDATE ON data.job_progress TO data_collector;   -- UPDATE: ON CONFLICT DO UPDATE
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'grafana_ro') THEN
    GRANT SELECT ON data.job_progress, data.v_job_status TO grafana_ro;
  END IF;
END $$;
