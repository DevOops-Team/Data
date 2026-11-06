-- 001_init.sql — 데이터 DB 스키마 (데이터 DB 스키마 초안 v4 기준, 2026-11-05)
-- 실행: data_app으로. psql -v ON_ERROR_STOP=1 -1 -f 001_init.sql
-- DB gpuf_data, schema data, owner data_app
-- 시각은 전부 timestamptz(UTC). 사건 원본은 events, 나머지는 거기서 뽑아낸 것.

SET search_path = data;

-- 1) 사건 원본
CREATE TABLE events (
  event_id        text        PRIMARY KEY,            -- 중복 제거 기준. gpu.fault는 trace ID(32자 hex)
  schema_version  smallint    NOT NULL,
  type            text        NOT NULL,
  topic           text        NOT NULL,
  occurred_at     timestamptz NOT NULL,               -- 봉투 시각 = 생산자가 사건을 만든 시각 (gpu.fault는 분류기 판정 시각). 측정 기준
  received_at     timestamptz NOT NULL DEFAULT now(), -- DB 도착 시각 → 기록 지연
  source          text,
  cluster         text,
  node            text,
  job_id          text,                               -- gpu.fault는 분류기가 Pod 라벨 devoops/job-id에서 채움, 못 찾으면 null
  generation      integer,                            -- gpu.fault는 분류기가 Pod 라벨 devoops/generation에서 채움, 못 찾으면 null
  payload         jsonb       NOT NULL DEFAULT '{}',  -- gpu.fault: fault-event v1 그대로 (dedup_key, ts = 원본 신호 시각)
  kafka_partition integer     NOT NULL,
  kafka_offset    bigint      NOT NULL
);
CREATE INDEX events_occurred_idx ON events (occurred_at);
CREATE INDEX events_job_idx      ON events (job_id, occurred_at) WHERE job_id IS NOT NULL;
CREATE INDEX events_node_idx     ON events (node, occurred_at)   WHERE node IS NOT NULL;
CREATE INDEX events_type_idx     ON events (type, occurred_at);
CREATE INDEX events_dedup_idx    ON events ((payload->>'dedup_key')) WHERE topic = 'gpu.fault';

-- 2) 형식 오류 사건
CREATE TABLE event_rejects (
  topic           text        NOT NULL,
  kafka_partition integer     NOT NULL,
  kafka_offset    bigint      NOT NULL,
  received_at     timestamptz NOT NULL DEFAULT now(),
  error           text        NOT NULL,
  raw             bytea       NOT NULL,
  PRIMARY KEY (topic, kafka_partition, kafka_offset)
);

-- 3) 실험
CREATE TABLE experiments (
  exp_id                 text PRIMARY KEY,
  name                   text NOT NULL,
  scenario               text NOT NULL,
  policy                 text NOT NULL,               -- ours | naive | hyperpod | manual
  fault_type             text,
  spare_enabled          boolean NOT NULL,
  checkpoint_every_steps integer,
  conditions             jsonb NOT NULL DEFAULT '{}',
  started_at             timestamptz,
  ended_at               timestamptz,
  note                   text
);

-- 4) 세대별 실행
CREATE TABLE job_runs (
  job_id        text     NOT NULL,
  generation    integer  NOT NULL,
  exp_id        text     REFERENCES experiments ON DELETE SET NULL,
  jobset_name   text     NOT NULL,
  cluster       text,
  queue         text,
  gpus          smallint NOT NULL,
  strategy      text     NOT NULL,
  config_sha256 text,
  image_digest  text,
  submitted_at  timestamptz,
  started_at    timestamptz,                          -- 첫 step 완료 시각 (ETTR 시작)
  ended_at      timestamptz,
  end_reason    text CHECK (end_reason IN ('completed','failed','fenced','stopped','cancelled')),
  PRIMARY KEY (job_id, generation)
);
CREATE INDEX job_runs_exp_idx ON job_runs (exp_id);

-- 5) 승인된 체크포인트 (job_runs와 FK 없음: 토픽이 달라 먼저 도착할 수 있음)
CREATE TABLE checkpoints (
  job_id        text     NOT NULL,
  generation    integer  NOT NULL,
  step          bigint   NOT NULL,
  format        text     NOT NULL,
  world_size    smallint NOT NULL,
  manifest_key  text     NOT NULL,
  file_count    integer  NOT NULL,
  size_bytes    bigint   NOT NULL,
  config_sha256 text     NOT NULL,
  image_digest  text,
  save_ms       integer,
  upload_ms     integer,
  saved_at      timestamptz NOT NULL,
  approved_at   timestamptz NOT NULL,
  PRIMARY KEY (job_id, generation, step)
);

-- 6) 측정 결과
CREATE TABLE results (
  exp_id       text NOT NULL REFERENCES experiments,
  job_id       text NOT NULL,
  ettr         numeric(5,4),
  detect_s     numeric(9,3),
  classify_s   numeric(9,3),
  recover_s    numeric(9,3),
  lost_steps   integer,
  wait_s       numeric(9,3),
  incidents    smallint NOT NULL DEFAULT 0,
  calc_version text NOT NULL,
  computed_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (exp_id, job_id)
);

-- 7) 장애 1건 단위 측정 뷰 (시간 정의 11/05 이태림 확정)
CREATE VIEW v_incidents AS
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
  ORDER BY e.occurred_at LIMIT 1) res ON true
WHERE inj.type = 'fault.injected';
