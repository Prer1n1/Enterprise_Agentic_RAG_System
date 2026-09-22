"""Shared retry policies for transient third-party API failures (OpenAI,
Cohere, TypeSafe/Jev).

Uses `tenacity` (a real, well-known retry library) instead of a hand-rolled
retry loop — same "use the named real tool" reasoning as RAGAS/LangSmith
elsewhere in this project.

Retrying is scoped to genuinely TRANSIENT errors per provider (rate
limits, connection drops, timeouts, 5xx server errors) — retrying an
AuthenticationError or a BadRequestError would just burn several seconds
waiting for an error that will never go away no matter how many times
it's retried.
"""

from __future__ import annotations

import logging

import cohere.errors
import httpx
import openai
from tenacity import retry, retry_if_exception, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_TRANSIENT_OPENAI_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)

_TRANSIENT_COHERE_ERRORS = (
    cohere.errors.GatewayTimeoutError,
    cohere.errors.InternalServerError,
    cohere.errors.ServiceUnavailableError,
    cohere.errors.TooManyRequestsError,
)

# jev_classifier.py calls TypeSafe's API directly over httpx (no official
# SDK dependency added for a single-endpoint pilot integration) — network-
# level failures are httpx's own connection/timeout exceptions, and
# server-side transient failures surface as a raised HTTPStatusError whose
# status code we check, since httpx doesn't raise on non-2xx by itself.
_TRANSIENT_JEV_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_transient_jev_error(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_JEV_STATUS_CODES
    return False


def _log_retry(retry_state) -> None:
    logger.warning(
        "api_call_retry",
        extra={
            "function": retry_state.fn.__name__ if retry_state.fn else None,
            "attempt": retry_state.attempt_number,
            "error": str(retry_state.outcome.exception()),
        },
    )


# max_attempts=3, exponential backoff starting at 1s (1s, 2s, then gives up):
# enough to ride out a brief network blip or a rate-limit window without
# making a user-facing request (a /query call in particular) hang for a
# long time waiting on retries that were never going to succeed.
retry_openai_call = retry(
    retry=retry_if_exception_type(_TRANSIENT_OPENAI_ERRORS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
    before_sleep=_log_retry,
)

# Same policy shape as retry_openai_call, just scoped to Cohere's own
# transient exception types — a separate provider means a separate set of
# exception classes, not a separate retry strategy.
retry_cohere_call = retry(
    retry=retry_if_exception_type(_TRANSIENT_COHERE_ERRORS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
    before_sleep=_log_retry,
)

# Same policy shape again, scoped via a predicate (not a type tuple) since
# "transient" for a raw HTTP call means a status CODE, not just an
# exception class.
retry_jev_call = retry(
    retry=retry_if_exception(_is_transient_jev_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
    before_sleep=_log_retry,
)
