"""Quick manual test — runs every loader against its sample file and
prints what came out. Not a unit test suite, just a sanity check while
building the ingestion pipeline."""

from pathlib import Path

from ingestion.loaders import load_document

SAMPLE_DIR = Path(__file__).parent / "sample_data"

FILES = [
    SAMPLE_DIR / "onboarding.pdf",
    SAMPLE_DIR / "security_policy.docx",
    SAMPLE_DIR / "handbook.html",
    SAMPLE_DIR / "policy.csv",
]

for file_path in FILES:
    print(f"\n{'=' * 60}\n{file_path.name}\n{'=' * 60}")
    documents = load_document(file_path)
    print(f"-> {len(documents)} Document(s) produced\n")
    for i, doc in enumerate(documents, start=1):
        print(f"[{i}] section={doc.metadata.section!r} page={doc.metadata.page_number} "
              f"is_table={doc.metadata.extra.get('is_table', False)}")
        preview = doc.content.replace("\n", " | ")[:100]
        print(f"    content: {preview}")
        print(f"    hash: {doc.content_hash()[:12]}...")
