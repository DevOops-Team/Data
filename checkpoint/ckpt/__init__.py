"""체크포인트 모듈 (ckpt-init · ckpt-sidecar).

기준: Confluence 「체크포인트 규약」 v11.
- 학습 코드는 로컬 CKPT_DIR에 저장 · 불러오기만 한다.
- S3 · DynamoDB는 이 패키지(init · 사이드카)만 다룬다.
"""
