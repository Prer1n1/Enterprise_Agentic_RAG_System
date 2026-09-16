"""Central place every component reads config/secrets from. Loads a
local .env (gitignored) so real API keys never get committed."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
