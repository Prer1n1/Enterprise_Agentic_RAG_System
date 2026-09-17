# Design Decisions Log

Quick reference for every real technical choice in this project. Format: what, why this, why not the alternative, advantage. Kept short on purpose — this is for interview recall, not documentation.

---

## Project classification: Agentic RAG

There are 5 levels of RAG maturity:
1. **Naive RAG** — retrieve once, stuff into prompt, generate.
2. **Advanced RAG** — adds query rewriting (pre-retrieval) + re-ranking (post-retrieval), still one linear pipeline.
3. **Modular RAG** — pipeline broken into swappable modules (routing, multiple retrievers), but no runtime autonomy.
4. **Agentic RAG** — an agent decides *which* sources to query, *whether* to query, and orchestrates multi-step retrieval before generating. **← this project**
5. **Adaptive / Self-Reflective RAG** — system critiques its own output at runtime and self-corrects (e.g., auto re-retrieves on a failed grounding check).

**Why Agentic, not Self-Reflective**: hallucination detection here is offline monitoring (measure after the fact), not a live loop that changes what the agent does next. Wiring the hallucination check back into the LangGraph flow (fail → auto re-retrieve) would be the one change needed to call it Self-Reflective RAG — worth mentioning as a future extension.

---

## Ingestion & Processing

**Common document schema — one `Document(content, metadata)` shape for all formats**
- Why: PDF/DOCX/HTML/CSV are structurally nothing alike (pages vs. headings vs. DOM vs. rows). Normalizing all of them into the same shape immediately after loading means every later stage (chunking, embedding, retrieval) is written once, not four times.
- Why not just use LangChain's built-in `Document` type directly: keeping our own schema decouples the pipeline from any one library's data model — swapping a loader library later doesn't ripple through the whole codebase.
- Advantage: format is invisible past the loader layer.

**Loaders — one Document per natural unit** (PDF: per page, DOCX/HTML: per heading-delimited section, CSV: per row), tables always kept as their own separate Document.
- Why: keeps metadata meaningful (page number for citations, section name for context) and keeps tables intact for row-based chunking later instead of being flattened into prose.

**Metadata extraction — rule-based enrichment, not an LLM call**
- What it adds beyond loaders: a stable `doc_id` (hash of source path — ties all pages/sections/rows from one file together for citations/dedup), word count, detected language, and a content category (HR/Finance/Security/IT/Legal/General).
- Category classifier — keyword matching per category, not ML: cheap, deterministic, explainable. Why not an LLM call per document: unnecessary cost/latency for a task this simple; scope discipline matters here the same way it did for GraphRAG (see Retrieval section).
- Why not skip category/language entirely: enterprise search needs these for filtering (e.g. "search only Security docs") — that's a real feature, not decoration.
- **Bug caught in testing**: langdetect misclassified a 12-word English CSV row as French. Root cause: langdetect is unreliable below ~20 words. Fix: below that word-count threshold, skip detection and default to `"en"` instead of trusting a low-confidence guess. Good interview story — shows you test assumptions, not just code paths.

**Chunking — hierarchical hybrid, semantic split implemented manually**
- Strategy: structure-aware split first (headers/sections already done by loaders) → semantic chunking within each prose section → row-based batching for tables/CSV (never semantic — there's no "meaning similarity" between table rows to detect).
- Why not fixed-size chunking: cuts mid-idea, splits a claim from its supporting fact.
- Why not pure semantic chunking on raw text: ignores document structure, expensive at scale (embeds every sentence just to find a cut point).
- Semantic mechanism (hand-rolled, not `langchain_experimental.SemanticChunker`): split into sentences → embed each one (OpenAI `text-embedding-3-small`) → cosine distance between every consecutive pair → cut wherever that distance is in the top 10% (sharpest "jumps" in meaning). Hand-rolled so the mechanism can actually be explained in an interview, not just "I called a library function," and avoids a dependency for one function.
- Why OpenAI embeddings over local sentence-transformers: higher retrieval quality; tradeoff is real cost per call and a required API key (`.env`, never hardcoded/committed).
- **Tested with a fake embedding model** (fixed vectors, zero API cost) to prove the boundary math before ever spending a real call — 2 "cat" sentences + 2 "stock" sentences split exactly at the topic change.
- Tabular batching (char cap, default 1500) verified to actually group multiple rows together, not just isolate each one.

**Incremental ingestion tracker — SQLite manifest, content-hash based**
- What: one row per file (`file_path`, `content_hash`, `last_ingested_at`). `check()` compares current file hash vs. stored hash → "new" / "changed" / "unchanged". `mark_ingested()` only runs after a file is *successfully* chunked+embedded — marking earlier would hide a crashed ingestion as "done."
- Why hash-based, not mtime-based: mtime changes on a simple touch/copy even with identical content — hash only changes when the actual bytes change.
- Why SQLite, not a JSON file: safe concurrent reads/writes, structured queries (`find_deleted` needs a set difference against everything tracked), no new infra dependency yet. Swappable for Postgres later without changing the interface.
- Verified all 4 states with real file edits: new → mark → unchanged → edit → changed (other file stayed unchanged) → removed-from-corpus → detected as deleted.

**Pipeline orchestration — `ingest_directory()` wires all four pieces together**
- Order: tracker decides what needs work → loaders normalize it → metadata extractor enriches it → chunker splits it. Only new/changed files get loaded and chunked at all.
- `mark_ingested()` only runs after chunking succeeds for everything — a mid-pipeline crash leaves the manifest unchanged, so the next run correctly retries instead of silently treating a failed file as done.
- Verified end-to-end: run 1 on 4 sample files → 4 ingested, 10 chunks; run 2 (nothing changed) → 0 ingested, 0 chunks, all 4 skipped.

---

## Agent Orchestration (LangGraph)

**Graph shape: plan -> parallel retrieve (one branch per source) -> synthesize**
- `plan_node`: an LLM (structured output, not keyword rules) decides which knowledge-source categories (HR/Finance/Security/IT/Legal/General) are relevant to the query. This is the actual agentic decision the whole "Agentic RAG" classification rests on.
- Why an LLM here and not the same rule-based classifier used at ingestion time: query *intent* is often ambiguous in ways keyword matching misses ("what happens if I break this rule" could be HR, Security, or Legal depending on context) — routing needs real reasoning, tagging documents doesn't.
- Falls back to querying every category if the LLM call fails — a routing mistake should degrade to "search broadly," never "search nothing."
- `route_to_sources`: fans out to one `retrieve_node` execution **per selected category, in parallel**, via LangGraph's `Send` API — this is the literal implementation of "querying multiple knowledge sources," not a single filtered search with extra steps.
- `retrieved_chunks` uses an `operator.add`-annotated state field so LangGraph concatenates results from every parallel branch instead of one branch overwriting another.
- `synthesize_node`: dedupes chunks (the same chunk can surface from more than one category branch), then one grounded LLM call answers using only the retrieved context, citing `[1]`, `[2]`, etc.
- **Verified with real queries**: a focused HR question routed to exactly `['HR']`; a cross-domain question ("security and expense rules for a new employee") routed to `['HR', 'Finance', 'Security']` and the synthesized answer correctly cited facts from all three sources with accurate numbers (not paraphrased/invented).
- Retrieval layer change this required: `HybridRetriever.retrieve()` gained a `category` filter — dense search filters via Chroma's metadata filter, BM25 filters *after* scoring against the full corpus (not by rebuilding a smaller index) to keep its IDF statistics correct.

**Three real bugs found only by running the CLI end-to-end (none caught by earlier component tests)**
1. *Category classification ignored ground-truth structured data.* `policy.csv` has an explicit `department` column, but a row with `department=IT` about "password rotation" got tagged category **Security** anyway — the keyword classifier saw "password" (a Security keyword) and never consulted the department column. Fix: `classify_category()` now takes an optional `hint` and trusts it outright when it's a known category, before ever falling back to keyword matching. Lesson: trust authoritative structured metadata over inferred classification whenever both are available.
2. *Row-batching silently discarded per-row category info.* Table batching grouped rows only by `source`, so 3 CSV rows with 3 different categories (Finance/HR/IT) merged into ONE chunk tagged with only the first row's category. The IT content was still physically present in the chunk's text, but invisible to any category-filtered search. Fix: batching now groups by `(source, category)`, so a batch is never tagged with a category it doesn't fully represent.
3. *`chunk_index` wasn't unique across the final chunk list.* `chunk_documents()` concatenates tabular chunks + prose chunks, but each helper indexed its own output from 0 independently — so the combined list had duplicate indices. Not yet a visible bug (section text differs enough to avoid a real `chunk_id` collision), but a latent one. Fix: reindex the combined list once, at the end.
- None of these were visible in isolated component tests — all three only surfaced running the full CLI on a real cross-domain question. That's the concrete argument for why "run it yourself end-to-end" matters even after every component passes its own tests.

---

## Storage

**Vector store — Chroma, not FAISS**
- Why: FAISS is a pure similarity index — no native metadata storage, so filtering by category/doc_type/source would need to be hand-built separately. Chroma stores metadata alongside each vector and filters directly in the query, which enterprise search actually needs ("only search Security docs").
- Persisted to local disk (`storage/chroma_db/`), not in-memory — survives a process restart.
- Advantage: one component does similarity search AND metadata filtering, instead of two.

**Chunk store — SQLite, separate from the vector store**
- What: every chunk's full text + metadata, keyed by a deterministic `chunk_id` (hash of source+section+chunk_index — same chunk always gets the same ID, so re-saving UPSERTs instead of duplicating).
- Why a separate store when Chroma already holds the text too: this is the source of truth for rebuilding things later — the BM25 keyword index needs raw text to build an inverted index from, and reconstructing a document's full chunk sequence for citations is a relational query Chroma isn't built for.
- **The vector store reuses the exact same `chunk_id` function** — verified in testing that both stores assign identical IDs to identical chunks. This is what makes `delete_by_source` safe to call on both stores and stay in sync.
- Why not Postgres yet: SQLite is enough for local development; swapping the backing store later doesn't change the `ChunkStore` interface — same seam as the ingestion tracker.

**Raw document store — intentionally not built**
- Why: the loader's `source` field already points at the original file. Duplicating file contents into a blob store would be redundant storage with no new capability at this stage — citations can already resolve back to the exact source path (and page/section) without it.

---

## Retrieval

**Hybrid retrieval — dense (Chroma) + sparse (BM25), fused with Reciprocal Rank Fusion**
- Why hybrid at all: dense embeddings catch paraphrases/synonyms ("computer" ≈ "laptop") but can blur exact codes/IDs together; BM25 catches exact term matches (policy IDs, names) but can't match a query that shares zero words with the source. Neither alone covers both cases — verified this directly: a "P-101" query and a fully-paraphrased zero-keyword query each had to hit a *different* retriever to succeed, and both worked.
- Why RRF, not weighted score averaging: cosine similarity (~0-1) and BM25 (unbounded, corpus-dependent) are on incompatible scales — averaging them needs manual normalization, which is its own bug source. RRF only uses each retriever's RANK POSITION, never the raw score, so scale mismatches can't happen by construction.
- `k=60` is the standard constant from the original RRF paper.
- **Real-run observation**: on this 10-chunk demo corpus, fused scores came out very close together (~0.031-0.033) — `k=60` flattens differentiation on small corpora. Not a bug; at real scale (thousands of chunks) separation would be more meaningful. Good evidence in an interview of testing at realistic-vs-toy scale, not just citing the paper's default.
- Both retrievers key results by the same `chunk_id` shared with the storage layer — required for RRF to know a hit in one ranking is the same chunk as a hit in the other.

**GraphRAG / FalkorDB — not used**
- Why not: needs entity/relationship extraction + graph DB + graph maintenance — too much scope for v1, and this project's queries don't need multi-hop graph reasoning.
- Advantage of skipping it: core platform (ingest → retrieve → agent → answer) stays achievable and demoable.

---

## Evaluation & Observability

**RAGAS — real dependency conflict found, root-caused, and fixed (not swapped for a substitute)**
- What happened: `pip install ragas` pulled the latest release (`0.4.3`), but importing it failed immediately — `ragas/llms/base.py` does an unconditional top-level `from langchain_community.chat_models.vertexai import ChatVertexAI`, and that submodule has been deleted from current `langchain-community` (`0.4.2`) as part of its ongoing deprecation/reorganization. This breaks `ragas` for every user regardless of which LLM provider they actually use, not just Google Vertex AI users — confirmed via multiple open GitHub issues on the ragas repo, not a problem specific to this project's setup.
- First instinct (a web search summary) said "pin `ragas==0.3.9`, it works" — **that turned out to be false when actually tested**: 0.3.9 hits the exact same broken import, because the failure depends on the installed `langchain-community` version, not the `ragas` version. Lesson: verify claims by running the code, don't trust a search summary at face value — including this session's own first attempt.
- Root cause, verified directly: the `chat_models.vertexai` submodule still exists in `langchain-community==0.4.1` but not `0.4.2`. Downgrading just that one package (not `ragas`, not `langchain-core`/`langchain-openai`/`langgraph`) fixed the import cleanly.
- Verified the fix doesn't destabilize anything: `pip install langchain-community==0.4.1` didn't touch `langchain-core`, `langchain-openai`, or `langgraph` at all (all already satisfied 0.4.1's requirements), and the full agent test suite (`test_agent.py`) was re-run afterward and still passed — router decisions and grounded citations unchanged.
- Why this was worth the effort instead of hand-rolling: the user's stated goal was to actually use RAGAS (it's the named tool in the original scope), and the real fix turned out to be a one-line, low-risk, verifiable version pin — not the risky wholesale downgrade or dependency-isolation approach originally feared before actually testing it.

**RAGAS API — verified by testing, not by trusting deprecation warnings**
- The library itself is mid-transition between two internal APIs. `llm_factory`/`embedding_factory` are what its own deprecation warnings recommend — tested directly, and both throw `AttributeError('InstructorLLM' object has no attribute 'agenerate_prompt')` when paired with the `Faithfulness`/`ResponseRelevancy` metric classes in this installed version.
- `LangchainLLMWrapper`/`LangchainEmbeddingsWrapper` — flagged deprecated by the library — are what actually works with these metrics. Verified with a live call, including a deliberately fabricated answer to confirm the metric genuinely discriminates: a faithful answer scored `1.0`, a fabricated one ("45 days and a free car" instead of the real "20 days") scored `0.0`. Lesson: when a library's own deprecation guidance conflicts with what actually runs, trust the test, not the warning text — the library's internal migration isn't finished yet.

**Hallucination detection = RAGAS Faithfulness below a threshold, not a separate mechanism**
- Faithfulness already measures "is the answer grounded in the retrieved context" — that IS what a hallucination is. Building a second, separate hallucination-detection pipeline would just be re-deriving the same signal with extra steps.
- `HALLUCINATION_THRESHOLD = 0.7`: below this, an answer is flagged `likely_hallucination`. Chosen to tolerate minor phrasing looseness while catching answers where a meaningful fraction of claims aren't grounded.
- All 5 real eval-set questions scored faithfulness `1.00` — genuinely well-grounded, not a metric that can't fail (proven separately by the fabricated-answer test above).

**LangSmith tracing — config only, zero code changes elsewhere**
- Setting `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` as environment variables is the entire integration — LangChain/LangGraph auto-instrument every LLM call once they're present. No changes needed to `agent/`, `retrieval/`, or anywhere else.
- Wired to degrade silently: if `LANGSMITH_API_KEY` is unset, tracing is simply off, nothing breaks. Requires a free account at smith.langchain.com — external signup, not something that can be scripted.
- **Verified live**, not just assumed from the docs: once the key was added, `client.list_runs(project_name="enterprise-agentic-rag")` showed real traces for an actual query — the full LangGraph execution (`route_to_sources` → `retrieve_node` → `synthesize_node` → `ChatOpenAI`) captured automatically, confirming the config-only claim actually holds.

---

## API Layer (FastAPI)

**Three endpoints, not a REST wrapper around every internal function**
- `GET /health`, `POST /ingest`, `POST /query` — the actual product surface. `evaluate` deliberately stayed CLI-only: RAGAS evaluation is a batch/ops operation (multiple real LLM calls, ~10-15s+), not something that belongs on the live request path.
- Request/response contract lives in `api/schemas.py`, separate from the internal dataclasses (`Chunk`, `RetrievedChunk`) — an internal refactor shouldn't silently change the public API shape.

**Sync route handlers, not async**
- The whole pipeline (LangGraph, embeddings, OpenAI calls) is synchronous. FastAPI runs sync `def` handlers in a thread pool automatically, so this gets non-blocking behavior for free — rewriting the entire stack as `async`/`await` would be a much bigger change for no real benefit at current request volume.

**Retriever built once at startup, not per-request**
- `HybridRetriever.__init__` loads every chunk from `ChunkStore` into memory to build the BM25 index — cheap once, wasteful if rebuilt on every request. `AppState` is constructed once in FastAPI's `lifespan`, held on `app.state`, and reused.
- **Real bug this surfaces if you don't handle it**: after `/ingest` adds or removes chunks, the BM25 index built at startup is a stale in-memory snapshot — new content becomes invisible to keyword search even though it's correctly persisted in both stores. Fixed by calling `rebuild_retriever()` after any ingestion that actually changed something (`ingested_files` or `deleted_files` non-empty) — cheap at this corpus size, and correctness matters more than optimizing away a rebuild that only happens on actual data changes.
- `threading.Lock` (not `asyncio.Lock`) guards state mutation during `/ingest`, since sync handlers run in real OS threads via FastAPI's thread pool, not as coroutines — the correct primitive follows from the sync-handler decision above, not an arbitrary choice.

**`/health` actually checks both stores, not a static `{"status": "ok"}`**
- Runs a real `chunk_store.get_all()` count and a real `vector_store.similarity_search()` — a broken DB file or an unreachable Chroma directory shows up here instead of surfacing as a confusing 500 on the first real query. A health check that can't fail isn't checking anything.

**Verified with real HTTP requests, not just "it starts without erroring"**
- `GET /health` → 200, correctly reported 12 reachable chunks.
- `POST /query` (real question) → 200, correct grounded answer with accurate citations (password rotation question correctly routed to and cited the IT-tagged CSV row).
- `POST /query` (empty question) → 422, Pydantic's own validation (`min_length=1`), not custom code.
- `POST /ingest` (nonexistent directory) → 400 with a clear message.
- `POST /ingest` (valid, nothing changed) → 200, correctly reported `skipped_unchanged: 4`.

---

## Docker

**Status: verified.** `docker compose up --build` was run for real — full image build (Python 3.11-slim + ~20 packages, chromadb pulling in onnxruntime/grpcio/opentelemetry), container started, `HEALTHCHECK` reported `healthy`, and `/health`, `/query`, `/ingest` were all hit against the actual running container with real answers and correct citations. A rebuild after a code-only change (no `requirements.txt` change) completed in ~1 second — confirmed the dependency-layer caching strategy works as designed, not just in theory.

**Real bug found BY running in Docker specifically — not something any earlier test could have caught**
- What happened: ingesting `sample_data` from the Windows host, then hitting `/ingest` again from inside the Linux container (both pointed at the same mounted `./storage` folder) reported `"ingested_files": 4, "deleted_files": 4"` — every file simultaneously "new" AND "deleted." All 4 files' chunks got needlessly re-embedded.
- Root cause: `DocumentMetadata.source` and the incremental tracker's manifest keys were built from `str(file_path)` — Windows renders this with backslashes (`sample_data\handbook.html`), Linux with forward slashes (`sample_data/handbook.html`). Same logical file, two different identity strings, depending on which OS did the ingesting. The tracker's "is this already ingested?" check is a string comparison, so it silently failed across the OS boundary.
- Why this is a genuinely important bug for THIS project specifically, not an edge case: the entire point of Dockerizing an app built on Windows is to run it on Linux somewhere else (a cloud VM, a Kubernetes node). A cross-platform identity bug isn't a corner case here — it's exactly the seam Docker is designed to cross, so it was always going to get hit the moment Docker actually ran.
- Fix: added `canonical_source()` (`ingestion/schema.py`) — `Path(file_path).as_posix()`, always forward-slash regardless of OS — and used it everywhere a file's identity is constructed: all four loaders' `source=` field, both tracker methods, `find_deleted()`, and the ingestion pipeline's `ingested_files`/`skipped_unchanged` lists (these must match the loaders' output exactly, or `delete_by_source()` calls silently target the wrong key and do nothing). Deliberately kept the *native* `str(file_path)` for actually opening files (`PdfReader(str(file_path))`, `docx.Document(str(file_path))`) — only identity strings needed normalizing, not filesystem calls.
- Verified the fix directly: wiped storage, re-ingested from the Windows host, confirmed stored paths were already POSIX-style (`sample_data/handbook.html`) even on Windows, then hit `/ingest` from the container against that same data — `skipped_unchanged: 4, ingested_files: 0, deleted_files: 0`. Re-ran the full test suite (`test_pipeline.py`, `test_tracker.py`, `test_storage.py`, `test_agent.py`) afterward; one test (`test_tracker.py`) needed updating because it was itself asserting against the old OS-native path format — the implementation was right, the test's expectation was stale.

**Consolidated the SQLite tracker's location before writing any Docker config**
- `ingestion_manifest.db` lived at the project root while `chunk_store.db` and `chroma_db/` lived under `storage/` — three persistence artifacts, two locations. Moved the tracker to `storage/ingestion_manifest.db` so ONE volume mount (`./storage:/app/storage`) covers all of it.
- Why this matters specifically for Docker: mounting a single file as a bind mount is a well-known footgun — if the file doesn't exist yet on the host, Docker silently creates a *directory* with that name instead of a file, and the app then fails in a confusing way trying to open a directory as a SQLite database. Consolidating into one folder sidesteps the problem entirely rather than working around it.

**Dependency layer before source code layer (Dockerfile instruction order)**
- `COPY requirements.txt .` + `pip install` happens BEFORE `COPY . .`. Docker caches each instruction as a layer; changing application code invalidates only the layers after it. If the source code copy came first, every code change would force reinstalling ~20 packages (langchain, chromadb, ragas, etc.) from scratch on every rebuild.

**Secrets and data never baked into the image**
- `.env` is in `.dockerignore` — an image is a static artifact that could be pushed to a registry or reused across environments; a key baked into one layer stays extractable from the image forever, even if a later layer appears to remove it.
- `storage/` (the three persistence artifacts) and `sample_data/` (source documents) are both excluded from the image build and mounted as volumes in `docker-compose.yml` instead. Real document sources and real persisted data shouldn't be frozen into a container image — they get mounted in at runtime, same as a real deployment would need to update/replace them without rebuilding the whole image.

**Bind mount, not a named Docker volume**
- `./storage:/app/storage` maps directly to a host folder, so the actual Chroma/SQLite files can be opened directly in Windows Explorer — chosen deliberately for a learning project where being able to inspect the real files matters. A named Docker volume (Docker manages the storage location itself) would be the more portable choice for a real multi-environment deployment, at the cost of not being directly browsable from the host.

**`HEALTHCHECK` reuses the real `/health` endpoint, not a fake liveness ping**
- Lets `docker ps` and any orchestrator (docker-compose, Kubernetes) know whether the container is actually serving correctly — not just that the process hasn't crashed. Consistent with the same "a health check that can't fail isn't checking anything" reasoning from the FastAPI section.

---

## Source Connectivity (production-readiness pass on Ingestion)

**Fixed: non-recursive directory traversal**
- `ingest_directory()` used `directory.iterdir()`, which only lists a folder's immediate children. Real document corpora are organized into subfolders (`documents/HR/`, `documents/Finance/`, ...) — everything nested was silently skipped, no error. Fixed with `directory.rglob("*")`. Verified with a real nested file before/after.

**`POST /documents/upload` — lets documents in over HTTP**
- Why: `/ingest`'s directory-path approach requires filesystem/SSH access to wherever the app runs — not realistic for most users of a real deployment. Upload accepts a file, validates its extension against `SUPPORTED_EXTENSIONS`, saves to `uploads/`, then re-ingests that whole directory through the exact same incremental-tracker path as `/ingest` — one ingestion code path to trust, not two.
- Refactored `/ingest` and `/documents/upload` to share one `_ingest_and_persist()` helper rather than duplicating the delete-then-insert persistence logic in two places that could drift apart.

**SEVERE bug found and fixed: multi-root ingestion silently deleted unrelated data**
- What happened, concretely: uploaded one new file via `/documents/upload` (which ingests `uploads/`). The response reported `deleted_files: 4` and the chunk count dropped from 12 to 1 — all 4 `sample_data/` files' chunks were wrongly purged from both stores.
- Root cause: `IngestionTracker.find_deleted()` compared *every file the tracker has ever tracked* against the current listing, with no concept of which corpus root that listing belonged to. The moment `/ingest` (root: `sample_data/`) and `/documents/upload` (root: `uploads/`) started sharing one tracker, ingesting either root alone made every file in the *other* root look deleted, since it obviously wasn't in the current listing either.
- This was live, active data loss during testing, not a theoretical edge case — caught immediately because every claim in this project gets tested with real requests, not just reasoned about.
- Fix: `find_deleted()` now takes an optional `root` and scopes the "tracked" set to only paths under that root before diffing. `ingest_directory()` passes its own `directory` as that root. Recovered the deleted demo data with a clean re-ingest (the tracker's manifest rows for those files weren't touched by the bug, only their chunk_store/vector_store entries — but since the manifest still thought them "unchanged," a normal re-ingest wouldn't have regenerated them either, so a full storage wipe + fresh ingest was the correct recovery, not a partial one).
- Added a permanent regression test (`test_pipeline.py`) — two separate roots sharing one tracker, ingest one, assert the other's file is never reported as deleted. This exact scenario must never silently regress again.

**Google Drive connector — in progress, requires the user's own Google Cloud setup**
- Deliberately not building SharePoint/Confluence/S3/etc. as well — the connector "shape" (authenticate, list, fetch, detect changes) is the same for all of them; building one for real and documenting the pattern is worth more than five built on unverifiable assumptions. Other named source-connector types are common; other formats are deferred with reasoning until a real account is available to test against — same discipline as the GraphRAG decision earlier.

---
