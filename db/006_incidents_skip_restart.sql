-- 006_incidents_skip_restart.sql — 사용자 "다시 시작"의 job.resumed를 장애 복구로 세지 않음 (GPU-234, 2026-11-20)
-- 실행: data_app으로 (뷰 소유자), 005 다음.  psql -v ON_ERROR_STOP=1 -1 -d gpuf_data -f 006_incidents_skip_restart.sql
-- 다시 실행해도 안전. 칸은 001과 같고 resumed를 고르는 조건만 하나 늘었다 (권한은 그대로 유지된다).
--
-- 감시기는 사용자가 중단한 작업을 다시 시작한 새 세대도 job.submitted + job.resumed로 보낸다. payload에
-- restart_reason이 있으면 사용자 재시작이므로, 장애 주입 뒤 같은 작업에서 사람이 다시 시작해도 MTTR이
-- 그 시각으로 잡히지 않게 뺀다.

SET search_path = data;

CREATE OR REPLACE VIEW v_incidents AS
SELECT inj.event_id                                          AS injection_id,
       inj.node,
       inj.occurred_at                                       AS injected_at,
       COALESCE(cls.job_id, sig.job_id, inj.job_id)          AS job_id,
       sig.signal_ts                                         AS raw_signal_at,
       sig.occurred_at                                       AS first_signal_at,
       cls.occurred_at                                       AS classified_at,
       res.occurred_at                                       AS resumed_at,
       EXTRACT(EPOCH FROM sig.occurred_at - inj.occurred_at) AS detect_s,
       EXTRACT(EPOCH FROM cls.occurred_at - inj.occurred_at) AS classify_s,
       EXTRACT(EPOCH FROM res.occurred_at - cls.occurred_at) AS recover_s,
       EXTRACT(EPOCH FROM sig.occurred_at - sig.signal_ts)   AS signal_lag_s
FROM events inj
LEFT JOIN LATERAL (
  SELECT e.occurred_at, e.job_id, (e.payload->>'ts')::timestamptz AS signal_ts FROM events e
  WHERE e.type IN ('fault.detected','fault.held') AND e.node = inj.node AND e.occurred_at >= inj.occurred_at
  ORDER BY e.occurred_at LIMIT 1) sig ON true
LEFT JOIN LATERAL (
  SELECT e.occurred_at, e.job_id FROM events e
  WHERE e.type IN ('fault.detected','fault.reclassified') AND e.node = inj.node AND e.occurred_at >= inj.occurred_at
  ORDER BY e.occurred_at LIMIT 1) cls ON true
LEFT JOIN LATERAL (
  SELECT e.occurred_at FROM events e
  WHERE e.type = 'job.resumed' AND e.job_id = COALESCE(cls.job_id, sig.job_id, inj.job_id)
    AND e.occurred_at >= cls.occurred_at
    AND NOT (e.payload ? 'restart_reason')                 -- 사용자 다시 시작은 장애 복구가 아님 (GPU-234)
  ORDER BY e.occurred_at LIMIT 1) res ON true
WHERE inj.type = 'fault.injected';
