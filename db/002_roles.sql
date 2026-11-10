-- 002_roles.sql — 서비스 계정 (2026-11-09)
-- 실행: 슈퍼유저(postgres)로, 대상 DB에 접속해서.  psql -v ON_ERROR_STOP=1 -1 -d gpuf_data -f 002_roles.sql
-- 다시 실행해도 안전 (있으면 건너뜀, GRANT는 중복 허용).
-- 비밀번호는 여기 넣지 않는다. 실행 후 psql에서 따로 지정:
--   \password data_collector
--   \password postgres_exporter
-- pg_hba.conf에는 각 계정을 필요한 주소에서만 허용 (scram-sha-256):
--   host gpuf_data data_collector     <수집기 Pod 대역>      scram-sha-256
--   host postgres  postgres_exporter  <모니터링 노드 주소>   scram-sha-256

-- 1) 수집기: 사건 저장 + 파생 테이블 갱신만. 삭제 · DDL · results 접근 없음
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'data_collector') THEN
    CREATE ROLE data_collector LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
  END IF;
  EXECUTE format('GRANT CONNECT ON DATABASE %I TO data_collector', current_database());
END $$;

ALTER ROLE data_collector SET search_path = data;
ALTER ROLE data_collector SET statement_timeout = '30s';   -- 묶음 하나가 DB를 오래 잡지 않게
GRANT USAGE ON SCHEMA data TO data_collector;

GRANT SELECT, INSERT         ON data.events        TO data_collector;  -- SELECT: experiment.started 때 job.submitted 조회
GRANT INSERT                 ON data.event_rejects TO data_collector;
GRANT SELECT, INSERT, UPDATE ON data.experiments   TO data_collector;  -- UPDATE: ON CONFLICT DO UPDATE, ended_at
GRANT SELECT, INSERT, UPDATE ON data.job_runs      TO data_collector;
GRANT SELECT, INSERT         ON data.checkpoints   TO data_collector;

-- 2) postgres_exporter (모니터링, ADR-18): 통계 읽기만. 데이터 테이블 접근 없음
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'postgres_exporter') THEN
    CREATE ROLE postgres_exporter LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE;
  END IF;
  EXECUTE format('GRANT CONNECT ON DATABASE %I TO postgres_exporter', current_database());
END $$;

GRANT pg_monitor TO postgres_exporter;
ALTER ROLE postgres_exporter SET search_path = pg_catalog;
ALTER ROLE postgres_exporter CONNECTION LIMIT 3;
