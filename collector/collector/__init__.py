"""수집기: Kafka 사건 → 데이터 DB (데이터 파트 설계 3절, 데이터 DB 스키마).

- 모든 사건은 events에 원본 그대로 (event_id 중복은 무시)
- 형식 오류는 event_rejects에 원본 바이트 보관
- 파생: experiment.* → experiments, checkpoint.approved → checkpoints, job.* → job_runs
- DB 커밋 후에 Kafka 오프셋 커밋 (최소 한 번 전달 + event_id 중복 제거)
"""
