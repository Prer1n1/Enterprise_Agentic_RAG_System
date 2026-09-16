"""Sanity checks for both storage components: ChunkStore (SQLite) and the
Chroma vector store. Uses a fake embedding model so it runs free/offline —
Chroma needs both embed_documents AND embed_query, unlike the chunking
fakes which only needed embed_documents.
"""

import hashlib
import shutil
import tempfile
from pathlib import Path

from ingestion.loaders import load_document
from ingestion.metadata_extractor import enrich_all
from ingestion.chunking import chunk_documents
from storage.chunk_store import ChunkStore, compute_chunk_id
from storage.vector_store import add_chunks, delete_by_source, get_vector_store, similarity_search

SAMPLE_DIR = Path(__file__).parent / "sample_data"


class HashFakeEmbeddings:
    def _vec(self, text: str):
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255 for b in h[:8]]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


tmp_dir = Path(tempfile.mkdtemp())
fake = HashFakeEmbeddings()

all_docs = enrich_all(load_document(SAMPLE_DIR / "security_policy.docx"))
chunks = chunk_documents(all_docs, embeddings=fake)
print(f"Built {len(chunks)} chunks from security_policy.docx")

# --- ChunkStore ---
print("\n=== ChunkStore ===")
chunk_store = ChunkStore(db_path=tmp_dir / "chunks.db")
ids = chunk_store.save_chunks(chunks)
print(f"Saved {len(ids)} chunks, ids: {ids}")

fetched = chunk_store.get_by_source(chunks[0].metadata.source)
print(f"get_by_source returned {len(fetched)} rows")
assert len(fetched) == len(chunks)

# id determinism: saving the same chunks again must produce the SAME ids (upsert, not duplicate)
ids_again = chunk_store.save_chunks(chunks)
assert ids_again == ids
all_rows = chunk_store.get_all()
assert len(all_rows) == len(chunks), "re-saving identical chunks should UPSERT, not duplicate rows"
print("PASS: re-saving same chunks upserted instead of duplicating")

deleted_count = chunk_store.delete_by_source(chunks[0].metadata.source)
print(f"delete_by_source removed {deleted_count} rows")
assert deleted_count == len(chunks)
assert chunk_store.get_all() == []
print("PASS: delete_by_source cleared all rows for that source")

# --- Vector store ---
print("\n=== Vector store (Chroma) ===")
vs = get_vector_store(embeddings=fake, persist_directory=tmp_dir / "chroma_db")
vector_ids = add_chunks(vs, chunks)
print(f"Added {len(vector_ids)} vectors")
assert vector_ids == [compute_chunk_id(c) for c in chunks], "vector store IDs must match ChunkStore IDs"
print("PASS: vector store uses the same chunk_id scheme as ChunkStore")

results = similarity_search(vs, "incident response security", k=3)
print(f"similarity_search returned {len(results)} result(s)")
for r in results:
    print(f"  section={r.metadata['section']!r} category={r.metadata['category']!r}")
assert len(results) > 0

filtered = similarity_search(vs, "incident response security", k=3, filter={"category": "Security"})
print(f"filtered (category=Security) returned {len(filtered)} result(s)")
assert all(r.metadata["category"] == "Security" for r in filtered)
print("PASS: metadata filtering works")

delete_by_source(vs, chunks[0].metadata.source)
results_after_delete = similarity_search(vs, "incident response security", k=3)
print(f"after delete_by_source: {len(results_after_delete)} result(s)")
assert len(results_after_delete) == 0
print("PASS: delete_by_source cleared the vector store too")

shutil.rmtree(tmp_dir, ignore_errors=True)
print("\nALL STORAGE TESTS PASSED")
