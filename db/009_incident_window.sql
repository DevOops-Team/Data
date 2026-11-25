-- 009_incident_window.sql — v_incidents: 감지 · 분류 사건을 장애 시작 뒤 15분 안으로 제한 (GPU-269, 2026-11-25)
-- 실행: data_app으로 (뷰 소유자), 008 다음.  psql -v ON_ERROR_STOP=1 -1 -d gpuf_data -f 009_incident_window.sql
-- 다시 실행해도 안전. 칸은 007과 같고 sig · cls 조건만 하나씩 늘었다 (권한은 그대로 유지된다).
--
-- 007은 장애 시작 뒤 같은 노드의 첫 감지(sig) · 분류(cls) 사건을 시간 상한 없이 찾았다. 그래서 감지가 안 붙는
-- 장애(예: fault.held만 있고 fault.detected가 없음)에 몇 시간 뒤 같은 노드의 다른 장애 사건이 붙어 job_id ·
-- classify_s가 그 장애 것으로 바뀌었다 (11/24 05:14 시험 사건 + 07:44 dd0d1281 → classify_s 9036초).
-- 주입 뒤 신호를 같은 장애로 보는 기준(15분, 007 starts)과 같은 창을 sig · cls에도 쓴다. 창 안에 없으면 비워 둔다.

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
       COALESCE(cls.job_id, sig.job_id, st.job_id)           AS job_id,
       sig.signal_ts                                         AS raw_signal_at,
       sig.occurred_at                                       AS first_signal_at,
       cls.occurred_at                                       AS classified_at,
       res.occurred_at                                       AS resumed_at,
       EXTRACT(EPOCH FROM sig.occurred_at - st.started_at)   AS detect_s,
       EXTRACT(EPOCH FROM cls.occurred_at - st.started_at)   AS classify_s,
       EXTRACT(EPOCH FROM res.occurred_at - cls.occurred_at) AS recover_s,
       EXTRACT(EPOCH FROM sig.occurred_at - sig.signal_ts)   AS signal_lag_s,
       st.start_source
FROM starts st
LEFT JOIN LATERAL (
  SELECT e.occurred_at, e.job_id, (e.payload->>'ts')::timestamptz AS signal_ts FROM events e
  WHERE e.type IN ('fault.detected','fault.held') AND e.node = st.node AND e.occurred_at >= st.started_at
    AND e.occurred_at <  st.started_at + interval '15 minutes'      -- 다른 장애의 나중 사건을 엮지 않음 (GPU-269)
  ORDER BY e.occurred_at LIMIT 1) sig ON true
LEFT JOIN LATERAL (
  SELECT e.occurred_at, e.job_id FROM events e
  WHERE e.type IN ('fault.detected','fault.reclassified') AND e.node = st.node AND e.occurred_at >= st.started_at
    AND e.occurred_at <  st.started_at + interval '15 minutes'
  ORDER BY e.occurred_at LIMIT 1) cls ON true
LEFT JOIN LATERAL (
  SELECT e.occurred_at FROM events e
  WHERE e.type = 'job.resumed' AND e.job_id = COALESCE(cls.job_id, sig.job_id, st.job_id)
    AND e.occurred_at >= cls.occurred_at
    AND NOT (e.payload ? 'restart_reason')                 -- 사용자 다시 시작은 장애 복구가 아님 (GPU-234)
  ORDER BY e.occurred_at LIMIT 1) res ON true;
