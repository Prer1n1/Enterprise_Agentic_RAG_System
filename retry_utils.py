"""Shared retry policy for transient OpenAI API failures.

Uses `tenacity` (a real, well-known retry library) instead of a hand-rolled
retry loop — same "use the named real tool" reasoning as RAGAS/LangSmith
elsewhere in this project.

Retrying is scoped to genuinely TRANSIENT OpenAI errors (rate limits,
connection drops, request timeouts, 5xx server errors) — retrying an
AuthenticationError or a BadRequestError would just burn several seconds
waiting for an error that will never go away no matter how many times
it's retried.
"""

from __future__ import annotations

import logging

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_TRANSIENT_OPENAI_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def _log_retry(retry_state) -> None:
    logger.warning(
        "openai_call_retry",
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
