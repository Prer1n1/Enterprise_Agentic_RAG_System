"""Manual sanity check for the metadata extractor.

Part 1 uses use_llm=False (the free keyword-based fallback classifier) so
it runs offline with no API key needed.

Part 2 only runs if OPENAI_API_KEY is set: it re-classifies the real SEC
investor bulletin (the document that originally exposed the keyword
classifier's real-world-vocabulary gap — see docs/design-decisions.md)
with the new LLM classifier, to prove the fix actually works on the exact
document that failed before."""

from pathlib import Path

from config import OPENAI_API_KEY
from ingestion.loaders import load_document
from ingestion.metadata_extractor import enrich_all

SAMPLE_DIR = Path(__file__).parent / "sample_data"
DRIVE_DEMO_DIR = Path(__file__).parent / "demo_docs_for_drive"

FILES = [
    SAMPLE_DIR / "onboarding.pdf",
    SAMPLE_DIR / "security_policy.docx",
    SAMPLE_DIR / "handbook.html",
    SAMPLE_DIR / "policy.csv",
]

print("=== Part 1: keyword-based fallback classifier (offline, free) ===")
for file_path in FILES:
    print(f"\n{'=' * 60}\n{file_path.name}\n{'=' * 60}")
    documents = enrich_all(load_document(file_path), use_llm=False)
    for doc in documents:
        e = doc.metadata.extra
        print(f"section={doc.metadata.section!r:25} category={e['category']:10} "
              f"lang={e['language']:5} words={e['word_count']:3} doc_id={e['doc_id']}")

if OPENAI_API_KEY:
    sec_pdf = DRIVE_DEMO_DIR / "sec_ipo_investor_bulletin.pdf"
    if sec_pdf.exists():
        print(f"\n{'=' * 60}\n=== Part 2: LLM classifier on the real SEC bulletin ===\n{'=' * 60}")
        print("(this is the exact document that scored ZERO Finance-tagged chunks")
        print(" under the old keyword-only classifier — 17 Legal, 6 General, 3 HR instead)")
        documents = enrich_all(load_document(sec_pdf), use_llm=True)
        categories = [doc.metadata.extra["category"] for doc in documents]
        counts = {c: categories.count(c) for c in set(categories)}
        print(f"\n{len(documents)} pages -> category counts: {counts}")
        print("(pre-chunking page-level categories here, not the final post-chunking")
        print(" 26-chunk count from the earlier full-pipeline test — enrich_all() runs")
        print(" before chunking, same as the rest of this file)")
        assert counts.get("Finance", 0) > 0, (
            "LLM classifier still failed to tag ANY page as Finance on a document "
            "that is entirely about investing — the fix did not work"
        )
        print("PASS: LLM classifier correctly tagged real investing content as Finance")
    else:
        print(f"\nSkipping Part 2 — {sec_pdf} not found "
              f"(see demo_docs_for_drive/SOURCES.txt for how to re-fetch it).")
else:
    print("\nSkipping Part 2 (LLM classifier test) — OPENAI_API_KEY not set.")
