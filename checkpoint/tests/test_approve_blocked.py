"""복구 중(RECOVERING)으로 승인이 오래 막히면 로그로 알린다 (GPU-228).

복구 컨트롤러가 중간 단계(CLAIMED 등)에서 HOLD로 멈추면 recovery_phase가 승인 불가 값에 남아,
학습은 계속되는데 승인이 늘지 않는다. 사이드카는 그걸 sidecar.approve_blocked 로그로 드러낸다.
"""
import json

from ckpt.init_main import EXIT_OK
from ckpt.init_main import run as init_run
from ckpt.sidecar import Sidecar
from test_flow import JOB, WORLD, cfg, fake_save


def events(capsys, name):
    out = capsys.readouterr().out
    return [e for e in (json.loads(line) for line in out.splitlines() if line.startswith("{")) if e["event"] == name]


def start(names, fencing, store, tmp_path, warn):
    fencing.create_job(JOB)
    cs = [cfg(names, tmp_path, gen=1, rank=r, cluster="eks-a", pod=f"r{r}") for r in range(WORLD)]
    for c in cs:
        assert init_run(c, fencing, store) == EXIT_OK
    return cs, [Sidecar(c, fencing, store, keep_local=2, heartbeat_seconds=0, approve_blocked_warn_seconds=warn)
                for c in cs]


def test_blocked_while_recovery_claimed_then_unblocked(names, fencing, store, tmp_path, capsys):
    cs, scs = start(names, fencing, store, tmp_path, warn=0)
    assert fencing.start_recovery(JOB, 1, "rec-1")          # 복구가 CLAIMED에서 멈춘 상태
    for c in cs:
        fake_save(c, 100)
    scs[1].tick()
    scs[0].tick()
    assert fencing.get(JOB).approved_step is None            # 승인 막힘
    blocked = events(capsys, "sidecar.approve_blocked")
    assert blocked and blocked[-1]["recovery_phase"] == "CLAIMED"
    assert (blocked[-1]["job_id"], blocked[-1]["generation"], blocked[-1]["step"]) == (JOB, 1, 100)
    assert blocked[-1]["recovery_id"] == "rec-1"

    fencing.ddb.update_item(TableName=fencing.table, Key={"job_id": {"S": JOB}},     # 복구 끝
                            UpdateExpression="SET recovery_phase = :c",
                            ExpressionAttributeValues={":c": {"S": "COMPLETED"}})
    scs[0].tick()
    assert fencing.get(JOB).approved_step == 100
    assert events(capsys, "sidecar.approve_unblocked")


def test_warns_only_after_threshold_and_then_every_interval(names, fencing, store, tmp_path, capsys):
    _, scs = start(names, fencing, store, tmp_path, warn=600)
    sc = scs[0]
    capsys.readouterr()
    for now in (0, 300, 599):
        sc._note_recovering(100, now=now)
    assert not events(capsys, "sidecar.approve_blocked")    # 10분 전엔 조용
    sc._note_recovering(100, now=600)
    assert [e["blocked_s"] for e in events(capsys, "sidecar.approve_blocked")] == [600]
    sc._note_recovering(100, now=900)                        # 다음 경고는 600초 뒤
    assert not events(capsys, "sidecar.approve_blocked")
    sc._note_recovering(100, now=1200)
    assert [e["blocked_s"] for e in events(capsys, "sidecar.approve_blocked")] == [1200]


def test_short_recovery_does_not_log_unblocked(names, fencing, store, tmp_path, capsys):
    _, scs = start(names, fencing, store, tmp_path, warn=600)
    sc = scs[0]
    sc._note_recovering(100, now=0)
    sc._clear_recovering(100, now=30)                        # 정상 복구처럼 금방 풀림 → 로그 없음
    assert not events(capsys, "sidecar.approve_unblocked")
    assert sc._recovering_since is None
