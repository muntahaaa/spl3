"""
task_store.py
=============
In-process job registry for chain_service async background tasks.

Design notes
------------
* All mutations go through a threading.Lock so the store is safe when
  Gradio's background threads (gr.Blocks queue=True) and FastAPI's
  background tasks write concurrently.
* ``update_job`` raises ``KeyError`` instead of silently dropping writes
  for unknown job IDs — callers (chain_service) must call ``create_job``
  before ``update_job``.
* ``get_job`` returns a typed sentinel dict with status="not_found" when
  the ID is unknown so callers can distinguish "not found" from "pending".
"""

import threading
import uuid
from typing import Any, Dict, Optional

# ── Internal store + lock ──────────────────────────────────────────────────
_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()

# Sentinel status value — exported so callers can compare without magic strings
STATUS_NOT_FOUND = "not_found"
STATUS_PENDING   = "pending"
STATUS_RUNNING   = "running"
STATUS_DONE      = "done"
STATUS_ERROR     = "error"


def create_job() -> str:
    """
    Allocate a new job entry and return its UUID string.

    Returns:
        str: newly created job_id
    """
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "status": STATUS_PENDING,
            "result": None,
            "error":  None,
        }
    return job_id


def update_job(
    job_id: str,
    status: str,
    result: Optional[Any] = None,
    error:  Optional[str] = None,
) -> None:
    """
    Update an existing job's status, result, and/or error fields.

    Args:
        job_id: UUID returned by ``create_job``.
        status: one of STATUS_* constants (pending/running/done/error).
        result: arbitrary result payload (stored as-is).
        error:  error message string (only meaningful when status=error).

    Raises:
        KeyError: if ``job_id`` was never registered via ``create_job``.
                  This is intentional — a missing ID indicates a programming
                  error rather than a transient condition.
    """
    with _lock:
        if job_id not in _jobs:
            raise KeyError(
                f"update_job: unknown job_id '{job_id}'. "
                "Call create_job() before update_job()."
            )
        _jobs[job_id].update({"status": status, "result": result, "error": error})


def get_job(job_id: str) -> Dict[str, Any]:
    """
    Retrieve a job record by ID.

    Args:
        job_id: UUID to look up.

    Returns:
        dict with keys ``status``, ``result``, ``error``, and ``job_id``.
        When not found: ``{"status": "not_found", "result": None,
                           "error": None, "job_id": job_id}``.
    """
    with _lock:
        record = _jobs.get(job_id)

    if record is None:
        return {
            "status": STATUS_NOT_FOUND,
            "result": None,
            "error":  None,
            "job_id": job_id,
        }

    # Return a shallow copy so callers cannot mutate internal state
    return {**record, "job_id": job_id}