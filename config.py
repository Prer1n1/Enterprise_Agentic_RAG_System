"""Central place every component reads config/secrets from. Loads a
local .env (gitignored) so real API keys never get committed."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# API layer auth — a shared secret every request to api/app.py must present
# via the X-API-Key header (except /health). See api/security.py.
API_KEY = os.getenv("API_KEY")

# LangSmith tracing: setting these three env vars is the ENTIRE integration —
# LangChain/LangGraph auto-instrument every LLM call once they're present, no
# code changes needed anywhere else in the project. Get a free key at
# smith.langchain.com. If LANGSMITH_API_KEY is unset, tracing is simply off —
# nothing breaks, it degrades silently to "no tracing."
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "enterprise-agentic-rag")

if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", LANGSMITH_API_KEY)
    os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)

# Google Drive connector — both unset means the connector is simply
# unavailable, same "degrade silently, don't break the rest of the app"
# pattern as LangSmith above.
GOOGLE_DRIVE_CREDENTIALS_PATH = os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
