"""Structured (JSON) logging setup for the whole app.

stdlib `logging` + `python-json-logger`, not a dedicated structured-logging
library (structlog) or a hand-rolled formatter: every dependency this
project already uses — uvicorn, langchain, the openai client, urllib3 —
logs through stdlib `logging` too, so attaching a JSON formatter to the
root logger gets all of it in one consistent format for free, no per-library
integration needed. A real production pattern (JSON logs -> Datadog/Splunk/
ELK), not something worth hand-rolling.

Deliberately NOT applied to main.py's CLI output: a human running the CLI
and watching a terminal wants readable text, not JSON lines — print()
there is the right tool for that job, same "match the tool to who's
actually reading it" reasoning as everywhere else in this project.
"""

from __future__ import annotations

import logging
import os

from pythonjsonlogger.json import JsonFormatter

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_CONFIGURED = False


def configure_logging() -> None:
    """Idempotent — safe to call from multiple entry points (config.py's
    import-time side effect AND api/app.py) without installing duplicate
    handlers."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(LOG_LEVEL)

    _CONFIGURED = True
