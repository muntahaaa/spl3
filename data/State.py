from typing import Annotated, Callable, Dict, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import add_messages


class State(TypedDict):
    """
    State machine for human exploration mode.

    Lifecycle
    ---------
    1. Initialised in main.py (initialize_device).
    2. After each ADB action a screenshot is taken  →  current_page_json = None.
    3. User calls POST /api/insert_parsed_result    →  current_page_json is set.
    4. Next action resolves element coords from current_page_json.
    5. On stop, state is serialised to JSON via state2json.
    6. JSON is loaded into Neo4j + Pinecone via json2db.
    """

    # ── task ──────────────────────────────────
    tsk:            str            # task description
    app_name:       str            # app under exploration
    completed:      bool
    step:           int            # monotonically increasing step counter
    history_steps:  List[Dict]     # one record per completed action

    # ── page ──────────────────────────────────
    page_history:           List[str]       # labeled image paths (gallery)
    current_page_screenshot: Optional[str]  # raw screenshot path
    current_page_json:       Optional[str]  # elements JSON path (None = not yet parsed)

    # ── operation ─────────────────────────────
    recommend_action:  str
    clicked_elements:  List[Dict]
    action_reflection: List[Dict]
    tool_results:      List[Dict]

    # ── device ────────────────────────────────
    device:      str
    device_info: Dict   # {"width": int, "height": int}

    # ── context / errors ──────────────────────
    context: Annotated[list, add_messages]
    errors:  List[Dict]

    callback: Optional[Callable[[TypedDict], None]]
