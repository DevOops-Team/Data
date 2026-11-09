"""DynamoDB 펜싱 테이블 연산 (규약 5 · 8 · 10절).

모든 쓰기는 조건부 갱신 한 번으로 한다. 조건이 틀리면 DynamoDB가 쓰기를 거부하고,
ReturnValuesOnConditionCheckFailure=ALL_OLD로 받은 현재 값으로 실패 종류를 구분한다.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

APPROVABLE_PHASES = ("RESUMING", "COMPLETED")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ApproveResult(enum.Enum):
    OK = "ok"                      # 승인됨
    ALREADY = "already"            # 같은 저장본이 이미 승인됨 → 성공으로 처리
    STALE = "stale"                # 더 최신 step이 이미 승인됨 → 건너뜀 (STOP 아님)
    FENCED = "fenced"              # 세대 · owner 불일치 → STOP
    RECOVERING = "recovering"      # 복구 진행 중(승인 금지 단계) → 이번 승인만 보류
    NO_ITEM = "no_item"            # 항목 없음 → STOP


@dataclass
class JobItem:
    raw: dict[str, Any]

    def _n(self, key: str) -> int | None:
        v = self.raw.get(key)
        return int(v["N"]) if v else None

    def _s(self, key: str) -> str | None:
        v = self.raw.get(key)
        return v["S"] if v else None

    @property
    def generation(self) -> int:
        return self._n("generation") or 0

    @property
    def owner(self) -> str | None:
        return self._s("owner")

    @property
    def approved_gen(self) -> int | None:
        return self._n("approved_gen")

    @property
    def approved_step(self) -> int | None:
        return self._n("approved_step")

    @property
    def recovery_phase(self) -> str | None:
        return self._s("recovery_phase")

    @property
    def recovery_id(self) -> str | None:
        return self._s("recovery_id")


class FencingTable:
    def __init__(self, table: str, client=None, region: str = "us-east-2"):
        self.table = table
        self.ddb = client or boto3.client("dynamodb", region_name=region)

    # ---------- 읽기 ----------
    def get(self, job_id: str) -> JobItem | None:
        r = self.ddb.get_item(TableName=self.table, Key={"job_id": {"S": job_id}}, ConsistentRead=True)
        return JobItem(r["Item"]) if "Item" in r else None

    # ---------- 백엔드: 첫 항목 ----------
    def create_job(self, job_id: str) -> bool:
        """세대 1로 첫 항목 생성. 이미 있으면 False (규약 13절 attribute_not_exists)."""
        try:
            self.ddb.put_item(
                TableName=self.table,
                Item={"job_id": {"S": job_id}, "generation": {"N": "1"}, "retry_count": {"N": "0"}},
                ConditionExpression="attribute_not_exists(job_id)",
            )
            return True
        except ClientError as e:
            if _is_cond_fail(e):
                return False
            raise

    # ---------- ckpt-init: owner 선점 ----------
    def claim_owner(self, job_id: str, gen: int, cluster: str) -> bool:
        """generation = 내 세대 AND (owner 없음 OR owner = 나) 일 때만 owner 기록."""
        try:
            self.ddb.update_item(
                TableName=self.table,
                Key={"job_id": {"S": job_id}},
                UpdateExpression="SET #o = :c",
                ConditionExpression="generation = :g AND (attribute_not_exists(#o) OR #o = :c)",
                ExpressionAttributeNames={"#o": "owner"},
                ExpressionAttributeValues={":g": {"N": str(gen)}, ":c": {"S": cluster}},
            )
            return True
        except ClientError as e:
            if _is_cond_fail(e):
                return False
            raise

    # ---------- 사이드카: 하트비트 ----------
    def heartbeat(self, job_id: str, gen: int, owner: str) -> bool:
        try:
            self.ddb.update_item(
                TableName=self.table,
                Key={"job_id": {"S": job_id}},
                UpdateExpression="SET heartbeat_at = :t",
                ConditionExpression="generation = :g AND #o = :o",
                ExpressionAttributeNames={"#o": "owner"},
                ExpressionAttributeValues={":g": {"N": str(gen)}, ":o": {"S": owner}, ":t": {"S": now_iso()}},
            )
            return True
        except ClientError as e:
            if _is_cond_fail(e):
                return False
            raise

    # ---------- 사이드카(rank 0): 승인 ----------
    def approve(self, job_id: str, gen: int, owner: str, step: int, max_attempts: int = 3) -> ApproveResult:
        """조건부 승인 (규약 5절).

        기존 approved_*를 prev_approved_*로 옮기려면 현재 값을 알아야 한다. 그래서
        읽은 값을 조건에 넣어(approved_step = 읽은 값) 그 사이 바뀌었으면 다시 읽고 재시도한다.
        """
        for _ in range(max_attempts):
            cur = self.get(job_id)
            if cur is None:
                return ApproveResult.NO_ITEM
            pre = _classify(cur, gen, owner, step)
            if pre is not None:
                return pre

            names = {"#o": "owner", "#rp": "recovery_phase"}
            values: dict[str, Any] = {
                ":g": {"N": str(gen)},
                ":o": {"S": owner},
                ":s": {"N": str(step)},
                ":p1": {"S": APPROVABLE_PHASES[0]},
                ":p2": {"S": APPROVABLE_PHASES[1]},
                ":t": {"S": now_iso()},
            }
            cond = (
                "generation = :g AND #o = :o"
                " AND (attribute_not_exists(#rp) OR #rp IN (:p1, :p2))"
            )
            sets = ["approved_gen = :g", "approved_step = :s", "approved_at = :t"]
            if cur.approved_step is None:
                cond += " AND attribute_not_exists(approved_step)"
            else:
                cond += " AND approved_step = :cur_s"
                values[":cur_s"] = {"N": str(cur.approved_step)}
                values[":cur_g"] = {"N": str(cur.approved_gen)}
                sets += ["prev_approved_gen = :cur_g", "prev_approved_step = :cur_s"]
            try:
                self.ddb.update_item(
                    TableName=self.table,
                    Key={"job_id": {"S": job_id}},
                    UpdateExpression="SET " + ", ".join(sets),
                    ConditionExpression=cond,
                    ExpressionAttributeNames=names,
                    ExpressionAttributeValues=values,
                    ReturnValuesOnConditionCheckFailure="ALL_OLD",
                )
                return ApproveResult.OK
            except ClientError as e:
                if not _is_cond_fail(e):
                    raise
                old = e.response.get("Item")
                if old is None:
                    return ApproveResult.NO_ITEM
                res = _classify(JobItem(old), gen, owner, step)
                if res is not None:
                    return res
                # 그 사이 다른 승인이 끼어들어 approved_step만 바뀐 경우 → 다시 읽고 재시도
        return ApproveResult.STALE

    # ---------- 복구 컨트롤러 동작 (테스트 · 로컬 시연용) ----------
    def start_recovery(self, job_id: str, gen: int, recovery_id: str) -> bool:
        """CLAIMED 기록. 실제로는 복구 컨트롤러(한송이)가 한다."""
        try:
            self.ddb.update_item(
                TableName=self.table,
                Key={"job_id": {"S": job_id}},
                UpdateExpression="SET recovery_id = :r, recovery_phase = :p, recovery_updated_at = :t",
                ConditionExpression="generation = :g AND (attribute_not_exists(recovery_phase) OR recovery_phase = :done)",
                ExpressionAttributeValues={
                    ":g": {"N": str(gen)}, ":r": {"S": recovery_id}, ":p": {"S": "CLAIMED"},
                    ":done": {"S": "COMPLETED"}, ":t": {"S": now_iso()},
                },
            )
            return True
        except ClientError as e:
            if _is_cond_fail(e):
                return False
            raise

    def advance_generation(self, job_id: str, gen: int, recovery_id: str) -> bool:
        """세대 +1 (규약 5절). owner 비움, approved_*는 유지."""
        try:
            self.ddb.update_item(
                TableName=self.table,
                Key={"job_id": {"S": job_id}},
                UpdateExpression=(
                    "SET generation = :ng, recovery_phase = :adv, recovery_from_gen = :g,"
                    " recovery_to_gen = :ng, retry_count = if_not_exists(retry_count, :zero) + :one,"
                    " recovery_updated_at = :t REMOVE #o"
                ),
                ConditionExpression="generation = :g AND recovery_phase = :claimed AND recovery_id = :r",
                ExpressionAttributeNames={"#o": "owner"},
                ExpressionAttributeValues={
                    ":g": {"N": str(gen)}, ":ng": {"N": str(gen + 1)}, ":r": {"S": recovery_id},
                    ":claimed": {"S": "CLAIMED"}, ":adv": {"S": "GENERATION_ADVANCED"},
                    ":zero": {"N": "0"}, ":one": {"N": "1"}, ":t": {"S": now_iso()},
                },
            )
            return True
        except ClientError as e:
            if _is_cond_fail(e):
                return False
            raise

    def set_phase(self, job_id: str, recovery_id: str, phase: str) -> bool:
        """복구 단계 기록 (OLD_RUN_STOPPED · NEW_RUN_SUBMITTED · RESUMING · COMPLETED)."""
        try:
            self.ddb.update_item(
                TableName=self.table,
                Key={"job_id": {"S": job_id}},
                UpdateExpression="SET recovery_phase = :p, recovery_updated_at = :t",
                ConditionExpression="recovery_id = :r",
                ExpressionAttributeValues={":p": {"S": phase}, ":r": {"S": recovery_id}, ":t": {"S": now_iso()}},
            )
            return True
        except ClientError as e:
            if _is_cond_fail(e):
                return False
            raise


def _classify(item: JobItem, gen: int, owner: str, step: int) -> ApproveResult | None:
    """승인 전에/실패 후에 현재 값으로 결과를 정한다. None이면 승인 시도 가능."""
    if item.generation != gen or item.owner != owner:
        return ApproveResult.FENCED
    if item.recovery_phase is not None and item.recovery_phase not in APPROVABLE_PHASES:
        return ApproveResult.RECOVERING
    if item.approved_step is not None:
        if item.approved_gen == gen and item.approved_step == step:
            return ApproveResult.ALREADY
        if item.approved_step >= step:
            return ApproveResult.STALE
    return None


def _is_cond_fail(e: ClientError) -> bool:
    return e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException"


def create_table(client, table: str) -> None:
    """로컬 · 테스트용 테이블 생성 (실제 테이블은 박기현 Terraform)."""
    client.create_table(
        TableName=table,
        KeySchema=[{"AttributeName": "job_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "job_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
