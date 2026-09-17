"""Central place every component reads config/secrets from. Loads a
local .env (gitignored) so real API keys never get committed."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

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
