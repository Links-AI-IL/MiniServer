import os
ANTHROPIC_API_KEY = os.environ.get("CLAUDE_API_KEY")
ANTHROPIC_MODEL = "claude-opus-4-20250514"
MAX_TOKENS = 2000
PORT = 5001
DB_URL = "sqlite:///rag.db"

