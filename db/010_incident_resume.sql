-- 010_incident_resume.sql — v_incidents: 복구 끝 = 새 세대가 실제로 step을 진행한 첫 시각 (GPU-283, 2026-11-25)
-- 실행: data_app으로 (뷰 소유자), 009 다음.  psql -v ON_ERROR_STOP=1 -1 -d gpuf_data -f 010_incident_resume.sql
-- 다시 실행해도 안전. 009의 칸은 그대로(이름 · 순서), 뒤에 pods_ready_at · pods_ready_s 두 칸을 더한다.
--
-- 009까지 resumed_at · recover_s의 끝은 감시기 job.resumed(새 세대 Pod 전부 Ready)였다. 11/25 17:22 장애 주입에서
-- 세대 2 Pod는 Ready였지만 trainer가 옛 세대 주소를 기다리며 학습을 시작하지 못했는데(step 0) recover_s가 65초로
-- 찍혔다. ADR-48 완료 기준(「다음 step 실제 진행」)에 맞춰 끝을 바꾼다.
--   resumed_at = 장애 당시 세대보다 높은 세대의 job.progress(RUNNING) 가운데 step > 장애 당시 마지막 승인 step인 첫 사건
--                (승인본이 없으면 step > 0). 사용자 다시 시작으로 생긴 세대(job.resumed payload restart_reason)는 뺀다
--   pods_ready_at = 지금까지의 끝(job.resumed, 재배치 완료) — 중간 단계로 남긴다
-- 장애 당시 세대: 분류 · 감지 사건의 generation → 장애 시작까지 그 작업 사건의 가장 큰 generation → 장애 뒤 첫 복구 세대 - 1.

SET search_path = data;

CREATE OR REPLACE VIEW v_incidents AS
WITH injected AS (
  SELECT 'manual:' || i.injection_id AS injection_id, i.node, i.injected_at AS started_at, i.job_id,
         'manual' AS start_source
  FROM injections i
  UNION ALL
  SELECT e.event_id, e.node, e.occurred_at, e.job_id, 'event'
  FROM events e WHERE e.type = 'fault.injected'
),
natural_faults AS (
  SELECT DISTINCT ON (COALESCE(e.payload->>'dedup_key', e.event_id))
         e.event_id, e.node, COALESCE((e.payload->>'ts')::timestamptz, e.occurred_at) AS started_at, e.job_id,
         e.occurred_at
  FROM events e
  WHERE e.type IN ('fault.detected','fault.held')
  ORDER BY COALESCE(e.payload->>'dedup_key', e.event_id), e.occurred_at
),
starts AS (
  SELECT injection_id, node, started_at, job_id, start_source FROM injected
  UNION ALL
  SELECT n.event_id, n.node, n.started_at, n.job_id, 'signal'
  FROM natural_faults n
  WHERE NOT EXISTS (SELECT 1 FROM injected j
                    WHERE j.node = n.node AND n.occurred_at >= j.started_at
                      AND n.occurred_at <  j.started_at + interval '15 minutes')
)
SELECT st.injection_id,
       st.node,
       st.started_at                                         AS injected_at,
       jb.job_id                                             AS job_id,
       sig.signal_ts                                         AS raw_signal_at,
       sig.occurred_at                                       AS first_signal_at,
       cls.occurred_at                                       AS classified_at,
       res.occurred_at                                       AS resumed_at,
       EXTRACT(EPOCH FROM sig.occurred_at - st.started_at)   AS detect_s,
       EXTRACT(EPOCH FROM cls.occurred_at - st.started_at)   AS classify_s,
       EXTRACT(EPOCH FROM res.occurred_at - cls.occurred_at) AS recover_s,
       EXTRACT(EPOCH FROM sig.occurred_at - sig.signal_ts)   AS signal_lag_s,
       st.start_source,
       rdy.occurred_at                                       AS pods_ready_at,
       EXTRACT(EPOCH FROM rdy.occurred_at - cls.occurred_at) AS pods_ready_s
FROM starts st
LEFT JOIN LATERAL (
  SELECT e.occurred_at, e.job_id, e.generation, (e.payload->>'ts')::timestamptz AS signal_ts FROM events e
  WHERE e.type IN ('fault.detected','fault.held') AND e.node = st.node AND e.occurred_at >= st.started_at
    AND e.occurred_at <  st.started_at + interval '15 minutes'      -- 다른 장애의 나중 사건을 엮지 않음 (GPU-269)
  ORDER BY e.occurred_at LIMIT 1) sig ON true
LEFT JOIN LATERAL (
  SELECT e.occurred_at, e.job_id, e.generation FROM events e
  WHERE e.type IN ('fault.detected','fault.reclassified') AND e.node = st.node AND e.occurred_at >= st.started_at
    AND e.occurred_at <  st.started_at + interval '15 minutes'
  ORDER BY e.occurred_at LIMIT 1) cls ON true
CROSS JOIN LATERAL (SELECT COALESCE(cls.job_id, sig.job_id, st.job_id) AS job_id) jb
LEFT JOIN LATERAL (
  SELECT COALESCE(cls.generation, sig.generation,
                  (SELECT max(e.generation) FROM events e
                   WHERE e.job_id = jb.job_id AND e.occurred_at <= st.started_at),
                  (SELECT min(e.generation) - 1 FROM events e      -- 그것도 없으면 복구로 생긴 첫 새 세대 바로 앞
                   WHERE e.type = 'job.resumed' AND e.job_id = jb.job_id AND e.occurred_at >= st.started_at
                     AND NOT (e.payload ? 'restart_reason'))) AS generation) fg ON true
LEFT JOIN LATERAL (
  SELECT COALESCE(max(c.step), 0) AS step FROM checkpoints c
  WHERE c.job_id = jb.job_id AND c.generation <= fg.generation) ap ON true
LEFT JOIN LATERAL (
  SELECT e.occurred_at FROM events e
  WHERE e.type = 'job.resumed' AND e.job_id = jb.job_id
    AND e.occurred_at >= cls.occurred_at
    AND NOT (e.payload ? 'restart_reason')                 -- 사용자 다시 시작은 장애 복구가 아님 (GPU-234)
  ORDER BY e.occurred_at LIMIT 1) rdy ON true
LEFT JOIN LATERAL (
  SELECT e.occurred_at FROM events e
  WHERE e.type = 'job.progress' AND e.job_id = jb.job_id
    AND e.occurred_at >= st.started_at
    AND e.generation > fg.generation
    AND e.payload->>'status' = 'RUNNING'
    AND (e.payload->>'step')::bigint > ap.step              -- 승인본 다음 step이 실제로 진행됨 (ADR-48)
    AND NOT EXISTS (SELECT 1 FROM events u                 -- 사용자 다시 시작으로 생긴 세대 제외 (GPU-234)
                    WHERE u.type = 'job.resumed' AND u.job_id = e.job_id AND u.generation = e.generation
                      AND u.payload ? 'restart_reason')
  ORDER BY e.occurred_at LIMIT 1) res ON true;
