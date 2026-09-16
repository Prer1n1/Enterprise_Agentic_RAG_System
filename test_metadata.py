"""Manual sanity check for the metadata extractor."""

from pathlib import Path

from ingestion.loaders import load_document
from ingestion.metadata_extractor import enrich_all

SAMPLE_DIR = Path(__file__).parent / "sample_data"

FILES = [
    SAMPLE_DIR / "onboarding.pdf",
    SAMPLE_DIR / "security_policy.docx",
    SAMPLE_DIR / "handbook.html",
    SAMPLE_DIR / "policy.csv",
]

for file_path in FILES:
    print(f"\n{'=' * 60}\n{file_path.name}\n{'=' * 60}")
    documents = enrich_all(load_document(file_path))
    for doc in documents:
        e = doc.metadata.extra
        print(f"section={doc.metadata.section!r:25} category={e['category']:10} "
              f"lang={e['language']:5} words={e['word_count']:3} doc_id={e['doc_id']}")
