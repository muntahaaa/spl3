"""
chain_models.py
===============
Pydantic request/response models for the chain pipeline.

Compatibility
-------------
* Written to work with both Pydantic v1 and v2.
  - v2: ``Optional[X]`` fields require an explicit ``default=None``
    (v2 no longer infers None as the default for Optional).
  - v1: ``Optional[X]`` without a default was silently treated as
    ``Optional[X] = None``; the explicit default is harmless there too.
* ``Literal`` is used for the ``status`` field so that editors and
  validators can catch invalid status strings at the model layer instead
  of at runtime deep inside chain_service.

Import note
-----------
``Literal`` is available from ``typing`` in Python ≥ 3.8 and from
``typing_extensions`` for older versions; the conditional import below
handles both.
"""

try:
    from typing import Literal          # Python ≥ 3.8
except ImportError:
    from typing_extensions import Literal  # type: ignore[assignment]

from typing import Optional
from pydantic import BaseModel, Field


# ── Request models ────────────────────────────────────────────────────────────

class ChainRunIn(BaseModel):
    """Input payload for both chain_understand and chain_evolve endpoints."""
    start_page_id: str = Field(
        ...,
        description="Page ID that starts the chain in Neo4j",
    )


# ── Response models ───────────────────────────────────────────────────────────

class ChainUnderstandOut(BaseModel):
    """Immediate (synchronous) response after kicking off chain_understand."""
    status: str
    start_page_id: str
    triplets_processed: int
    message: str


class ChainEvolveOut(BaseModel):
    """Immediate (synchronous) response after kicking off chain_evolve."""
    status: str
    start_page_id: str
    action_id: Optional[str] = None     # explicit default — required in pydantic v2
    message: str


class ChainJobStatus(BaseModel):
    """
    Polling response for async job status.

    ``status`` is constrained to the four lifecycle values so that
    invalid strings are caught at serialisation time.
    """
    job_id: str                          # included so the caller never loses track
    status: Literal["pending", "running", "done", "error", "not_found"]
    result: Optional[dict] = None       # explicit default — required in pydantic v2
    error:  Optional[str]  = None       # explicit default — required in pydantic v2