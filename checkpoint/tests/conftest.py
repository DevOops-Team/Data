"""moto 서버(S3 + DynamoDB 흉내)를 테스트 동안 띄운다. AWS 계정 없이 실제 boto3 호출로 검증."""
from __future__ import annotations

import socket
import sys
import uuid
from pathlib import Path

import boto3
import pytest
from moto.server import ThreadedMotoServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ckpt.fencing import FencingTable, create_table  # noqa: E402
from ckpt.storage import Store  # noqa: E402

REGION = "us-east-2"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def endpoint():
    port = _free_port()
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port, verbose=False)
    server.start()
    yield f"http://127.0.0.1:{port}"
    server.stop()


@pytest.fixture(autouse=True)
def aws_env(monkeypatch, endpoint):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)


@pytest.fixture
def names():
    sfx = uuid.uuid4().hex[:8]
    return {"bucket": f"ckpt-{sfx}", "table": f"fencing-{sfx}"}


@pytest.fixture
def fencing(names):
    ddb = boto3.client("dynamodb", region_name=REGION)
    create_table(ddb, names["table"])
    return FencingTable(names["table"], client=ddb)


@pytest.fixture
def store(names):
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=names["bucket"], CreateBucketConfiguration={"LocationConstraint": REGION})
    return Store(names["bucket"], client=s3)
