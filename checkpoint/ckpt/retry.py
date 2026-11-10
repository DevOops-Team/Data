"""재시도 정책 (규약 10절 · 한송이 「장애 복구 규약」 4절과 맞춤, 11/09).

구분
- 일시 오류(연결 끊김 · timeout · 5xx · 스로틀링): 정해진 횟수 · 간격으로 다시 시도
- 손상(체크섬 불일치 · 승인본 파일 없음): 다시 받아도 같으므로 재시도 1회 후 실패 → 종료 코드 7
- 설정 불일치 · 권한 오류: 재시도하지 않음

기본값(환경 변수로 변경): 3회 · 간격 5 → 10 → 20초 · ckpt-init 다운로드 전체 10분.
복구 컨트롤러의 '복원 · 학습 진행 확인 30분' 안에 끝나도록 잡았다.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

from botocore.exceptions import ClientError
from botocore.exceptions import ConnectionError as BotoConnectionError
from botocore.exceptions import ReadTimeoutError

T = TypeVar("T")

TRANSIENT_CODES = {
    "RequestTimeout", "RequestTimeoutException", "SlowDown", "Throttling", "ThrottlingException",
    "ProvisionedThroughputExceededException", "RequestLimitExceeded", "InternalError",
    "InternalServerError", "ServiceUnavailable", "500", "502", "503", "504",
}


class CorruptError(Exception):
    """승인본이 손상됨 (체크섬 · 크기 불일치, 파일 없음). 재시도로 해결되지 않는다."""


def is_transient(e: BaseException) -> bool:
    if isinstance(e, (BotoConnectionError, ReadTimeoutError, ConnectionError, TimeoutError)):
        return True
    if isinstance(e, ClientError):
        code = e.response.get("Error", {}).get("Code", "")
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        return code in TRANSIENT_CODES or status >= 500
    return False


def is_not_found(e: BaseException) -> bool:
    if isinstance(e, ClientError):
        return e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound")
    return False


def _env_backoff() -> list[float]:
    raw = os.environ.get("RETRY_BACKOFF", "5,10,20")
    return [float(x) for x in raw.split(",") if x.strip()]


@dataclass
class RetryPolicy:
    attempts: int = field(default_factory=lambda: int(os.environ.get("RETRY_ATTEMPTS", "3")))
    backoff: list[float] = field(default_factory=_env_backoff)
    deadline_s: float | None = None           # 전체 제한 시간 (None이면 없음)
    sleep: Callable[[float], None] = time.sleep

    def started(self) -> "RetryPolicy":
        """전체 제한 시간을 지금부터 센다."""
        self._t0 = time.monotonic()
        return self

    def _expired(self) -> bool:
        return self.deadline_s is not None and time.monotonic() - getattr(self, "_t0", time.monotonic()) > self.deadline_s

    def call(self, fn: Callable[[], T], on_retry: Callable[[int, BaseException], None] | None = None) -> T:
        last: BaseException | None = None
        for i in range(max(1, self.attempts)):
            try:
                return fn()
            except BaseException as e:  # noqa: BLE001
                if not is_transient(e):
                    raise
                last = e
                if i == self.attempts - 1 or self._expired():
                    break
                if on_retry:
                    on_retry(i + 1, e)
                self.sleep(self.backoff[min(i, len(self.backoff) - 1)] if self.backoff else 0)
        assert last is not None
        raise last


def no_wait(attempts: int = 3, deadline_s: float | None = None) -> RetryPolicy:
    """테스트용: 대기 없이 재시도."""
    return RetryPolicy(attempts=attempts, backoff=[0], deadline_s=deadline_s, sleep=lambda _: None)
