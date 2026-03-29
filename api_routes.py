"""
api_routes.py  –  FastAPI routes
=================================
Exposes a minimal REST API so the user can inject manually-obtained
parsed-result data into the live session.

Mount this router in main.py:
    app.include_router(router, prefix="/api")
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path

from data.data_storage import json2db
from state_manager import session

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
#  Request / response models
# ─────────────────────────────────────────────────────────────────────────────

class ParsedResultIn(BaseModel):
    """
    Body for POST /api/insert_parsed_result
    ----------------------------------------
    After you run your own parsing tool on a screenshot, call this endpoint
    with the two file paths it produced.

    Fields
    ------
    labeled_image_path
        Absolute or relative path to the annotated/labeled PNG produced by
        your parsing tool.
        Example: "./log/screenshots/human_exploration/processed/
                   labeled_human_exploration_step_1_20240101_120000.png"

    parsed_content_json_path
        Absolute or relative path to the elements JSON file produced by
        your parsing tool.
        Example: "./log/screenshots/human_exploration/processed/
                   human_exploration_step_1_20240101_120000.json"

    The JSON file must be an array of element objects:

        [
          {
            "ID":      1,
            "bbox":    [x1_rel, y1_rel, x2_rel, y2_rel],   // 0-1 relative coords
            "type":    "button",
            "content": "Login"
          },
          ...
        ]

    bbox values are relative to the screen size (0.0 – 1.0 range).
    """
    labeled_image_path:       str = Field(..., description="Path to labeled/annotated image")
    parsed_content_json_path: str = Field(..., description="Path to elements JSON file")


class ParsedResultOut(BaseModel):
    status:         str
    step:           int
    screenshot:     str | None
    previous_json:  str | None
    new_json:       str
    labeled_image:  str


class SessionStatusOut(BaseModel):
    has_session:         bool
    step:                int | None
    device:              str | None
    task:                str | None
    parsed_result_ready: bool
    pending_screenshot:  str | None
    history_count:       int


class StoreToDbIn(BaseModel):
    json_path: str = Field(..., description="Path to saved state JSON file")


class StoreToDbOut(BaseModel):
    status: str
    task_id: str
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
    """
    Returns whether a session is active, which step it is on, and whether
    the current screenshot still needs a parsed result.
    """
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
    ## When to call this

    After each screenshot is taken the session waits for you to run your own
    parsing tool.  Once you have the output files (labeled image + elements
    JSON), POST their paths here.

    ## Parsed JSON format

    Your `parsed_content_json_path` file must contain a JSON array:

    ```json
    [
      {
        "ID":      1,
        "bbox":    [0.05, 0.10, 0.90, 0.18],
        "type":    "text",
        "content": "Welcome Screen"
      },
      {
        "ID":      2,
        "bbox":    [0.15, 0.45, 0.85, 0.55],
        "type":    "button",
        "content": "Sign In"
      }
    ]
    ```

    ### bbox format
    `[x1, y1, x2, y2]` as **relative** coordinates (0.0 – 1.0).
    `x1`/`y1` = top-left corner, `x2`/`y2` = bottom-right corner.

    ### Supported element types (examples)
    `button`, `text`, `input`, `image`, `icon`, `checkbox`, `list_item`

    ## What happens after this call
    * `state["current_page_json"]` is updated.
    * The labeled image is added to the gallery.
    * You can now perform actions that require element coordinates.
    """
    try:
        result = session.insert_parsed_result(
            labeled_image_path=body.labeled_image_path,
            parsed_content_json_path=body.parsed_content_json_path,
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

    The file should be generated by Step 2 (`state2json`) and include
    `history_steps` with valid `source_page` and `source_json` paths.
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
