-- 004_backend_role.sql — Backend 읽기 전용 계정 (GPU-141, 2026-11-12)
-- 실행: 슈퍼유저(postgres)로, 003_job_progress.sql 다음.  psql -v ON_ERROR_STOP=1 -1 -d gpuf_data -f 004_backend_role.sql
-- 다시 실행해도 안전. 비밀번호는 여기 넣지 않는다 → 실행 후 psql에서 \password backend_ro
-- pg_hba.conf: host gpuf_data backend_ro <관리 클러스터 Pod 대역> scram-sha-256
--
-- Backend(GET /api/jobs · /api/jobs/{id})가 화면 상태 · 진행률을 읽는 용도. 쓰기 · 사건 원본 접근 없음.

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'backend_ro') THEN
    CREATE ROLE backend_ro LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
  END IF;
  EXECUTE format('GRANT CONNECT ON DATABASE %I TO backend_ro', current_database());
END $$;

ALTER ROLE backend_ro SET search_path = data;
ALTER ROLE backend_ro SET statement_timeout = '5s';    -- 화면 폴링이 DB를 오래 잡지 않게
ALTER ROLE backend_ro CONNECTION LIMIT 10;
GRANT USAGE ON SCHEMA data TO backend_ro;

GRANT SELECT ON data.v_job_status TO backend_ro;
GRANT SELECT ON data.job_runs, data.job_progress, data.checkpoints TO backend_ro;   -- 세대별 이력 · 체크포인트 목록 화면용
