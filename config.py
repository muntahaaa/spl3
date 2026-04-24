# ─────────────────────────────────────────────
#  config.py  –  Project-wide configuration
# ─────────────────────────────────────────────

import os
from pathlib import Path

from dotenv import load_dotenv


# Load project .env so os.getenv resolves local runtime secrets/config.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


# ── Neo4j ─────────────────────────────────────
Neo4j_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
Neo4j_AUTH = (
	os.getenv("NEO4J_USER", "neo4j"),
	os.getenv("NEO4J_PASSWORD","Muntaha12"  ),
)
Neo4j_DB = os.getenv("NEO4J_DB", "graphdb")

# ── Pinecone ──────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "vectordb")

# ── Feature-extraction service (ResNet50) ─────
# Point this at your CPU-based embedding service.
# Example: "http://localhost:8001"
Feature_URI = os.getenv("FEATURE_URI", "http://localhost:8001")

# ── Screenshot storage ────────────────────────
SCREENSHOT_DIR = os.getenv("SCREENSHOT_DIR", "./log/screenshots")
JSON_STATE_DIR = os.getenv("JSON_STATE_DIR", "./log/json_state")

GEMINI_API_KEY =  os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "")

# ── LangChain tracing ────────────────────────
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "")

# ── LLM settings (OpenAI-compatible endpoint) ─
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-pro")
