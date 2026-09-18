"""PII redaction — detects and masks personally identifiable information in
ingested document content BEFORE it's chunked, embedded, or stored. Raw PII
never enters the vector store or chunk store, and never gets sent to
OpenAI for embeddings either, since redaction happens first. Applied to
every ingested document regardless of category — a Finance or IT document
can contain PII just as easily as an HR one.

Uses an LLM with structured output (same pattern as classify_category /
detect_injection) rather than a purpose-built NER tool like Microsoft
Presidio — reuses the existing OpenAI setup, no new dependency or model
download. Real tradeoff, worth naming honestly: this sends document
content to OpenAI specifically to find the PII in it, which is a bit
backwards for a PII-protection feature, and a purpose-built NER engine
would likely be more precise. Accepted here in exchange for zero new
infrastructure, consistent with this project's OpenAI-centric design —
and it's still a real, meaningful improvement over storing PII unredacted.

Scope, deliberately: only HIGH-SENSITIVITY, structured PII (SSN, credit
card, bank account, date of birth, salary/compensation figures, email,
phone number) — NOT ordinary person names. Redacting every name in a
policy document (e.g. "contact Jane Doe in HR") would make the corpus far
less useful while protecting against a much lower-risk exposure than an
SSN or bank account number. A real DLP system would likely tier this the
same way (basic vs. sensitive PII).

Fail-CLOSED on detector failure, matching prompt_injection.py's policy:
if the LLM call fails even after retries, the caller (ingestion/
pipeline.py) blocks that file from ingestion rather than silently storing
potentially-unredacted PII. There is deliberately NO silent fallback here
(unlike classify_category/detect_injection, which degrade to a keyword
heuristic) — guessing wrong about PII redaction risks a real, permanent
leak, so a failure has to be loud, not quiet.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from retry_utils import retry_openai_call

logger = logging.getLogger(__name__)

_REDACTION_MODEL = "gpt-4o-mini"
_MAX_CHARS_FOR_DETECTION = 6000

PII_TYPES = ["SSN", "CREDIT_CARD", "BANK_ACCOUNT", "EMAIL", "PHONE_NUMBER", "DATE_OF_BIRTH", "SALARY"]


class PIIEntity(BaseModel):
    text: str = Field(
        description="The EXACT substring from the source text that is this PII entity, verbatim — no paraphrasing, no reformatting."
    )
    type: str = Field(description=f"One of: {PII_TYPES}")


class PIIDetectionResult(BaseModel):
    entities: List[PIIEntity] = Field(
        default_factory=list,
        description=(
            "Every PII entity found, limited to the listed types. Empty list if none. "
            "Do NOT flag ordinary person names, job titles, or company names unless "
            "they are directly part of one of the listed entity types."
        ),
    )


class PIIDetectionFailed(Exception):
    """Raised when the LLM detector fails even after retries. Callers
    must treat this as fail-closed — never fall back to storing the
    original, unredacted text."""


@retry_openai_call
def _invoke_pii_detector(structured_llm, prompt: str) -> PIIDetectionResult:
    return structured_llm.invoke(prompt)


def redact_pii(text: str, llm=None) -> Tuple[str, List[str]]:
    """Returns (redacted_text, entity_types_found). Raises
    PIIDetectionFailed if the LLM call fails after retries — see module
    docstring for why there's no silent fallback here."""
    try:
        llm = llm or ChatOpenAI(model=_REDACTION_MODEL, temperature=0)
        structured_llm = llm.with_structured_output(PIIDetectionResult)
        result = _invoke_pii_detector(
            structured_llm,
            f"Find all personally identifiable information (PII) in the text below, "
            f"limited to these types ONLY: {PII_TYPES}. Return the EXACT verbatim "
            f"substring for each entity, not a paraphrase or reformatting.\n\n"
            f"Text:\n{text[:_MAX_CHARS_FOR_DETECTION]}",
        )
    except Exception as e:
        logger.exception("pii_detector_failed")
        raise PIIDetectionFailed(str(e)) from e

    redacted = text
    types_found: List[str] = []
    for entity in result.entities:
        # Exact-substring replacement: if the LLM slightly reformatted the
        # entity text rather than quoting it verbatim, that occurrence
        # won't match and won't be redacted — a known, honest limitation
        # (see docs/design-decisions.md), not a silent failure mode.
        if entity.text and entity.text in redacted:
            redacted = redacted.replace(entity.text, f"[{entity.type}_REDACTED]")
            types_found.append(entity.type)

    if types_found:
        logger.warning("pii_redacted", extra={"types": types_found, "count": len(types_found)})
    return redacted, types_found
