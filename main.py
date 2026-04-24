"""
main.py  –  application entry point
=====================================
Runs a single Uvicorn server that serves:
    /        →  Gradio UI  (Steps 1 & 2)
    /api/…   →  FastAPI REST endpoints  (parsed-result insertion)
    /docs    →  Swagger UI  (auto-generated API docs)

Start with:
    python main.py
or
    uvicorn main:app --host 0.0.0.0 --port 7860 --reload
"""

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.chain_routes import router as chain_router

# Some Windows environments carry stale SSL_CERT_FILE/SSL_CERT_DIR values.
# If they point to missing paths, httpx/gradio import can fail immediately.
ssl_cert_file = os.environ.get("SSL_CERT_FILE")
if ssl_cert_file and not Path(ssl_cert_file).is_file():
    os.environ.pop("SSL_CERT_FILE", None)

ssl_cert_dir = os.environ.get("SSL_CERT_DIR")
if ssl_cert_dir and not Path(ssl_cert_dir).is_dir():
    os.environ.pop("SSL_CERT_DIR", None)

import gradio as gr

from api.api_routes import router as api_router
from ui import build_ui

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Human Explorer API",
    description=(
        "REST API for injecting parsed UI results into an active "
        "human-exploration session."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes ────────────────────────────────────────────────────────────────
app.include_router(api_router, prefix="/api")
app.include_router(chain_router, prefix="/api")

# ── Gradio UI ─────────────────────────────────────────────────────────────────
gradio_app = build_ui()
app = gr.mount_gradio_app(app, gradio_app, path="/")


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )
