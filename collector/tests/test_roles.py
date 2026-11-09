"""db/002_roles.sql 권한 범위: 필요한 것만 되고 나머지는 막혀야 한다."""
import psycopg
import pytest


def _denied(dsn, sql):
    with psycopg.connect(dsn, autocommit=True) as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute(sql)


def test_collector_cannot_delete_or_ddl(pg_dsn, conn):
    d = pg_dsn["collector"]
    _denied(d, "DELETE FROM data.events")
    _denied(d, "TRUNCATE data.events")
    _denied(d, "UPDATE data.events SET node = 'x'")          # 원본은 고칠 수 없음
    _denied(d, "SELECT * FROM data.results")
    _denied(d, "INSERT INTO data.results (exp_id, job_id, calc_version) VALUES ('e','j','v')")
    _denied(d, "CREATE TABLE data.x (a int)")
    _denied(d, "DROP TABLE data.events")


def test_collector_session_settings(pg_dsn):
    with psycopg.connect(pg_dsn["collector"]) as c:
        assert c.execute("SHOW search_path").fetchone()[0] == "data"
        assert c.execute("SHOW statement_timeout").fetchone()[0] == "30s"


def test_exporter_reads_stats_not_data(pg_dsn):
    with psycopg.connect(pg_dsn["exporter"], autocommit=True) as c:
        assert c.execute("SELECT count(*) FROM pg_stat_database").fetchone()[0] > 0
        # pg_monitor: 다른 계정의 쿼리 내용까지 볼 수 있어야 함 (pg_stat_activity)
        assert c.execute("SELECT pg_has_role('pg_monitor', 'USAGE')").fetchone()[0] is True
        c.execute("SELECT pg_database_size(current_database())")
    _denied(pg_dsn["exporter"], "SELECT * FROM data.events")
    _denied(pg_dsn["exporter"], "INSERT INTO data.event_rejects (topic,kafka_partition,kafka_offset,error,raw) "
                                "VALUES ('t',0,0,'e','')")
