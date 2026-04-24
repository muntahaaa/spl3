"""
api_routes.py  –  FastAPI routes
=================================

BUG-3 FIX
──────────
The /api/insert_parsed_result endpoint now accepts an optional
`parsed_content_json_url` field alongside the existing path fields.

Two usage patterns are supported:

Pattern A – local files (same as before)
    POST /api/insert_parsed_result
    {
        "labeled_image_path": "./log/..../labeled_step_1.png",
        "parsed_content_json_path": "/tmp/human_explorer_cloud/ss1.json",
        "parsed_content_json_url": "https://res.cloudinary.com/.../ss1.json"
    }

Pattern B – both are local paths (original behaviour, url defaults to path)
    POST /api/insert_parsed_result
    {
        "labeled_image_path": "...",
        "parsed_content_json_path": "..."
    }
    → parsed_content_json_url defaults to parsed_content_json_path
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional

from data.data_storage import json2db
from state_manager import session

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
#  Request / response models
# ─────────────────────────────────────────────────────────────────────────────

class ParsedResultIn(BaseModel):
    """
    Body for POST /api/insert_parsed_result

    Fields
    ──────
    labeled_image_path
        Local path to the annotated/labeled PNG produced by your parsing tool
        (or downloaded from the cloud).

    parsed_content_json_path
        Local path to the elements JSON file produced by your parsing tool
        (or downloaded from the cloud).

    parsed_content_json_url   [optional]
        The original cloud URL of the elements JSON (e.g. Cloudinary URL).
        If omitted it defaults to `parsed_content_json_path`.
        This URL is stored as `source_json` in history_steps so that the
        state JSON is portable across machines.

    JSON format expected in the elements file:
        [{"ID": 1, "bbox": [x1, y1, x2, y2], "type": "button", "content": "..."}]
    """
    labeled_image_path:       str            = Field(..., description="Local path to labeled image")
    parsed_content_json_path: str            = Field(..., description="Local path to elements JSON")
    parsed_content_json_url:  Optional[str]  = Field(None,  description="Cloud URL of elements JSON (for portability)")


class ParsedResultOut(BaseModel):
    status:          str
    step:            int
    screenshot:      Optional[str]
    previous_json:   Optional[str]
    new_json:        str
    new_json_url:    str
    labeled_image:   str


class SessionStatusOut(BaseModel):
    has_session:         bool
    step:                Optional[int]
    device:              Optional[str]
    task:                Optional[str]
    parsed_result_ready: bool
    pending_screenshot:  Optional[str]
    history_count:       int


class StoreToDbIn(BaseModel):
    json_path: str = Field(..., description="Path to saved state JSON file")


class StoreToDbOut(BaseModel):
    status:    str
    task_id:   str
    json_path: str


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/session/status",
    response_model=SessionStatusOut,
    summary="Get current session status",
    tags=["Session"],
)
def get_session_status():
    state = session.get_state()
    if state is None:
        return SessionStatusOut(
            has_session=False,
            step=None, device=None, task=None,
            parsed_result_ready=False,
            pending_screenshot=None,
            history_count=0,
        )
    return SessionStatusOut(
        has_session=True,
        step=state.get("step", 0),
        device=state.get("device"),
        task=state.get("tsk"),
        parsed_result_ready=session.is_ready_for_action(),
        pending_screenshot=session.pending_screenshot(),
        history_count=len(state.get("history_steps", [])),
    )


@router.post(
    "/insert_parsed_result",
    response_model=ParsedResultOut,
    summary="Insert parsed result for the current screenshot",
    tags=["Parsing"],
)
def insert_parsed_result(body: ParsedResultIn):
    """
    Register the parsed output (labeled image + elements JSON) for the most
    recent screenshot.

    After this call:
    * `state["current_page_json"]`     is set to `parsed_content_json_path`
    * `state["current_page_json_url"]` is set to `parsed_content_json_url`
      (falls back to `parsed_content_json_path` if url is omitted)
    * The labeled image is appended to the gallery.
    * Element coordinates can now be resolved for the next action.
    """
    try:
        result = session.insert_parsed_result(
            labeled_image_path=body.labeled_image_path,
            parsed_content_json_path=body.parsed_content_json_path,
            # BUG-3 FIX: pass through cloud URL; defaults to local path if absent
            parsed_content_json_url=body.parsed_content_json_url or body.parsed_content_json_path,
        )
        return ParsedResultOut(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@router.post(
    "/store_to_db",
    response_model=StoreToDbOut,
    summary="Store a saved state JSON into Neo4j + Pinecone",
    tags=["Storage"],
)
def store_to_db(body: StoreToDbIn):
    """
    Pushes a previously saved `state_*.json` file to Neo4j and Pinecone.
    """
    json_path = body.json_path.strip()
    if not json_path:
        raise HTTPException(status_code=400, detail="json_path cannot be empty")

    if not Path(json_path).is_file():
        raise HTTPException(status_code=400, detail=f"JSON file not found: {json_path}")

    try:
        task_id = json2db(json_path)
        return StoreToDbOut(status="ok", task_id=task_id, json_path=json_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Storage failed: {exc}")