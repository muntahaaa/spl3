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
