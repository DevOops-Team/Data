-- 005_progress_error.sql — 학습 실패 이유 한 줄 (GPU-188, 2026-11-17)
-- 실행: data_app으로 (테이블 소유자), 003 · 004 다음.  psql -v ON_ERROR_STOP=1 -1 -d gpuf_data -f 005_progress_error.sql
-- 다시 실행해도 안전. 이 파일을 적용한 뒤에는 003을 다시 돌리지 않는다 (003의 뷰는 칸이 하나 적어 교체가 거부됨).
--
-- train.py가 실패할 때 progress.json에 error(한 줄)를 쓰면 ckpt-sidecar가 job.progress로 보내고,
-- 수집기가 job_progress.error에 넣는다. 화면은 v_job_status.progress_error로 본다.

SET search_path = data;

ALTER TABLE job_progress ADD COLUMN IF NOT EXISTS error text;   -- 실패 이유 한 줄 (최대 500자, 성공 · 진행 중이면 NULL)

-- 003과 같은 뷰 + 끝에 progress_error (CREATE OR REPLACE는 칸을 끝에만 더할 수 있음). 권한은 그대로 유지된다
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
         WHEN jr.end_reason IS NOT NULL  THEN upper(jr.end_reason)
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
       ck.approved_at                              AS last_approved_at,
       p.error                                     AS progress_error
FROM (SELECT DISTINCT ON (job_id) * FROM job_runs ORDER BY job_id, generation DESC) jr
JOIN (SELECT job_id, count(*) AS n FROM job_runs GROUP BY job_id) runs ON runs.job_id = jr.job_id
LEFT JOIN job_progress p ON p.job_id = jr.job_id AND p.generation = jr.generation
LEFT JOIN LATERAL (
  SELECT c.step, c.generation, c.approved_at FROM checkpoints c
  WHERE c.job_id = jr.job_id ORDER BY c.generation DESC, c.step DESC LIMIT 1
) ck ON true;
