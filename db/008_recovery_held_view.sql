-- 008_recovery_held_view.sql — Grafana HOLD 목록용 뷰 (GPU-252, 2026-11-23, 이태림 요청)
-- 실행: data_app으로 (표 · 뷰 소유자), 007 다음.  psql -v ON_ERROR_STOP=1 -1 -d gpuf_data -f 008_recovery_held_view.sql
-- 다시 실행해도 안전.
--
-- 「장애 복구」 대시보드의 HOLD 목록 패널이 복구 컨트롤러의 recovery.held 사건을 읽는다. 대시보드 계정(grafana_ro)에
-- data.events 전체를 열지 않고, HOLD 사건만 보이는 뷰에 읽기 권한을 준다. 뷰는 소유자(data_app) 권한으로 events를
-- 읽으므로 grafana_ro에 events 권한은 필요 없다. 칸은 events와 같다 (payload에 HOLD 사유 등, 형식은 복구 컨트롤러 기준).

CREATE OR REPLACE VIEW data.v_recovery_held AS
SELECT e.event_id, e.occurred_at, e.received_at, e.cluster, e.node, e.job_id, e.generation, e.source, e.payload
FROM data.events e
WHERE e.type = 'recovery.held';

COMMENT ON VIEW data.v_recovery_held IS '복구 컨트롤러 HOLD 사건 (recovery.held) — Grafana 장애 복구 대시보드용, GPU-252';

-- 대시보드(grafana_ro)는 읽기 (계정이 있을 때만, 003 · 007과 같은 방식)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'grafana_ro') THEN
    GRANT SELECT ON data.v_recovery_held TO grafana_ro;
  END IF;
END $$;
