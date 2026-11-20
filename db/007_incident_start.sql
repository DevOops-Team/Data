-- 007_incident_start.sql — 장애 시작 시각: 수동 주입 기록 · 분류기 신호 (GPU-239, 2026-11-20)
-- 실행: data_app으로 (표 · 뷰 소유자), 006 다음.  psql -v ON_ERROR_STOP=1 -1 -d gpuf_data -f 007_incident_start.sql
-- 다시 실행해도 안전. v_incidents는 001 · 006과 같은 칸 + 끝에 start_source (권한은 그대로 유지된다).
--
-- 장애 주입은 사람이 CLI로 직접 한다 (ADR-33) → fault.injected 사건을 보내는 곳이 없고 fault.injection 토픽은
-- 쓰지 않는다. 그래서 장애 1건의 시작을 세 곳에서 찾는다 (위가 우선):
--   manual  data.injections — 기록표의 주입 시각을 사람이 넣음 (python -m collector.injection add)
--   event   fault.injected 사건 — 옛 시험 데이터 호환
--   signal  주입 기록이 없는 장애 = 분류기 첫 신호(fault.detected · fault.held)의 payload.ts (원본 신호 시각,
--           ADR-15 자연 장애). dedup_key 하나 = 장애 하나
-- 주입(manual · event) 뒤 같은 노드에서 15분 안에 온 신호는 그 주입의 감지로 보고 따로 세지 않는다.
-- 칸 이름 injected_at은 measure.py와 맞추려고 그대로 둔다 (= 장애 시작 시각).

SET search_path = data;

CREATE TABLE IF NOT EXISTS injections (
  injection_id bigserial   PRIMARY KEY,
  injected_at  timestamptz NOT NULL,          -- 주입 명령을 누른 시각 (기록표, 초 단위)
  node         text        NOT NULL,          -- 주입한 노드 (사건 봉투 node와 같은 이름: ip-…compute.internal)
  job_id       text,                          -- 알면 (없으면 분류기 · 감시기 사건에서 채움)
  kind         text        NOT NULL,          -- trainer_kill · xid · node_stop · network …
  note         text,
  recorded_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS injections_node_at_idx ON injections (node, injected_at);

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
  ORDER BY e.occurred_at LIMIT 1) sig ON true
LEFT JOIN LATERAL (
  SELECT e.occurred_at, e.job_id FROM events e
  WHERE e.type IN ('fault.detected','fault.reclassified') AND e.node = st.node AND e.occurred_at >= st.started_at
  ORDER BY e.occurred_at LIMIT 1) cls ON true
LEFT JOIN LATERAL (
  SELECT e.occurred_at FROM events e
  WHERE e.type = 'job.resumed' AND e.job_id = COALESCE(cls.job_id, sig.job_id, st.job_id)
    AND e.occurred_at >= cls.occurred_at
    AND NOT (e.payload ? 'restart_reason')                 -- 사용자 다시 시작은 장애 복구가 아님 (GPU-234)
  ORDER BY e.occurred_at LIMIT 1) res ON true;

-- 대시보드(grafana_ro)는 읽기 (계정이 있을 때만, 003과 같은 방식)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'grafana_ro') THEN
    GRANT SELECT ON data.injections, data.v_incidents TO grafana_ro;
  END IF;
END $$;
