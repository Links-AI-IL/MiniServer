import os

ANTHROPIC_API_KEY = os.environ.get("CLAUDE_API_KEY")
ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"
MAX_TOKENS = 400
PORT = 5001
DB_URL = "sqlite:///rag.db"

