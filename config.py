# ─────────────────────────────────────────────
#  config.py  –  Project-wide configuration
# ─────────────────────────────────────────────

# ── Neo4j ─────────────────────────────────────
Neo4j_URI  = "neo4j://127.0.0.1:7687"
Neo4j_AUTH = ("neo4j", "Muntaha12")
Neo4j_DB   = "graphdb"

# ── Pinecone ──────────────────────────────────
PINECONE_API_KEY = "pcsk_4T3v5b_LpT88xq1jbe4c2RcckgoPBc9ESdiEWccTBNUasGuK8xDzf4BJYLhTHiGsDEePbp"
PINECONE_INDEX_NAME = "vectordb"

# ── Feature-extraction service (ResNet50) ─────
# Point this at your CPU-based embedding service.
# Example: "http://localhost:8001"
Feature_URI = "http://localhost:8001"

# ── Screenshot storage ────────────────────────
SCREENSHOT_DIR = "./log/screenshots"
JSON_STATE_DIR = "./log/json_state"
