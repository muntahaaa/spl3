"""
data_storage.py  –  with swipe + text element resolution
==========================================================
Two changes from the previous version:

1.  _id2element()  –  new helper that finds an element by its integer ID
    directly from the parsed JSON array.  Used for text actions where there
    are no click coordinates, only an element number in the recommended_action.

2.  _extract_element_info()  –  replaces the single-line conditional that
    only handled tap.  Now dispatches on action type:

        tap / long_press  →  clicked_element coords  → _pos2id()
        swipe             →  tool_result.swipe.start  → _pos2id()
        text              →  element_number in recommended_action → _id2element()
        anything else     →  empty placeholder dict

3.  LEADS_TO fix: action_params is serialised with json.dumps() so Neo4j
    does not reject a Python dict as a relationship property.
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

import config as config
from data.graph_db import Neo4jDatabase
from data.vector_db import NodeType, VectorData, VectorStore
from tool.img_tool import element_img, extract_features


def _md5(text: str, length: int = 8) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:length]


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 – save State → JSON
# ─────────────────────────────────────────────────────────────────────────────

def state2json(state: dict, save_path: str = None) -> str:
    if not isinstance(state, dict):
        raise TypeError("'state' must be a dict.")

    filtered = {
        "tsk":           state.get("tsk", ""),
        "app_name":      state.get("app_name", ""),
        "step":          state.get("step", 0),
        "history_steps": state.get("history_steps", []),
        "final_page": {
            "screenshot": state.get("current_page_screenshot", ""),
            "page_json": (
                state["current_page_json"].get("parsed_content_json_path", "")
                if isinstance(state.get("current_page_json"), dict)
                else state.get("current_page_json", "")
            ),
        },
    }

    if save_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{config.JSON_STATE_DIR}/state_{ts}.json"

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=4)
        return str(out.resolve())
    except Exception as exc:
        return f"Error saving state: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 – JSON → Neo4j + Pinecone
# ─────────────────────────────────────────────────────────────────────────────

def json2db(json_path: str) -> str:
    db = Neo4jDatabase(
        uri=config.Neo4j_URI,
        auth=config.Neo4j_AUTH,
        database=config.Neo4j_DB,
    )
    vs = VectorStore(dimension=2048, batch_size=2)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_id        = _md5(data["tsk"], 12)
    pages_info:    list = []
    elements_info: list = []

    for step in data["history_steps"]:

        # ── page node ────────────────────────────────────────────────────────
        page_props = {
            "page_id":      str(uuid4()),
            "description":  "",
            "raw_page_url": step.get("source_page", "").replace("\\", "/"),
            "timestamp":    step["timestamp"],
            "other_info":   json.dumps({
                "step": step["step"],
                **({"task_info": {"task_id": task_id, "description": data["tsk"]}}
                   if step["step"] == 0 else {}),
            }),
        }

        raw_source_json = step.get("source_json")
        elements_path   = None
        elements_data   = []

        if raw_source_json:
            norm = raw_source_json.replace("\\", "/")
            if Path(norm).is_file():
                elements_path = norm
                with open(norm, "r", encoding="utf-8") as f:
                    elements_data = json.load(f)
                page_props["elements"] = json.dumps(elements_data)
            else:
                print(f"[json2db] Step {step['step']}: JSON not found at {norm}")
        else:
            print(f"[json2db] Step {step['step']}: source_json is null – page node only")

        db.create_page(page_props)
        pages_info.append({"page_id": page_props["page_id"], "step": step["step"]})

        # ── element node ─────────────────────────────────────────────────────
        tool_result = step["tool_result"]
        action_type = tool_result.get("action")

        # ── CHANGE: resolve element for swipe and text, not just tap ─────────
        element_info = _extract_element_info(
            action_type=action_type,
            tool_result=tool_result,
            recommended_action=step.get("recommended_action", ""),
            elements_data=elements_data,
            elements_path=elements_path,
        )

        if element_info:
            params = {k: v for k, v in tool_result.items()
                      if k not in ("action", "device", "status")}

            elem_props = {
                "element_id":          str(uuid4()),
                "element_original_id": element_info.get("ID", ""),
                "description":         "",
                "action_type":         action_type,
                "parameters":          json.dumps(params),
                "bounding_box":        element_info.get("bbox", []),
                "other_info":          json.dumps({
                    "type":    element_info.get("type", ""),
                    "content": element_info.get("content", ""),
                }),
            }

            db.create_element(elem_props)
            elements_info.append({
                "element_id": elem_props["element_id"],
                "step":       step["step"],
                "action":     step["recommended_action"],
                "status":     tool_result["status"],
                "timestamp":  step["timestamp"],
            })
            db.add_element_to_page(page_props["page_id"], elem_props["element_id"])

            if element_info.get("ID") and step.get("source_page"):
                source_page = step["source_page"].replace("\\", "/")
                ok = _element2vector(
                    str(element_info["ID"]),
                    elem_props["element_id"],
                    json.dumps(elements_data),
                    source_page,
                    vs,
                )
                if not ok:
                    print(f"[json2db] Vector storage failed for element {element_info['ID']}")

    # ── final page node ───────────────────────────────────────────────────────
    if data.get("final_page"):
        fp = data["final_page"]
        fp_props = {
            "page_id":      str(uuid4()),
            "description":  "",
            "raw_page_url": (fp.get("screenshot") or "").replace("\\", "/"),
            "timestamp":    fp.get("timestamp", ""),
        }
        raw_fp_json = fp.get("page_json")
        if raw_fp_json:
            norm = raw_fp_json.replace("\\", "/")
            if Path(norm).is_file():
                with open(norm, "r", encoding="utf-8") as f:
                    fp_props["elements"] = json.dumps(json.load(f))
            else:
                fp_props["elements"] = json.dumps([])
        else:
            fp_props["elements"] = json.dumps([])

        db.create_page(fp_props)
        pages_info.append({"page_id": fp_props["page_id"], "step": "final"})

    # ── LEADS_TO edges ────────────────────────────────────────────────────────
    for i, cur in enumerate(elements_info):
        if i == len(elements_info) - 1 and len(pages_info) > len(elements_info):
            next_page = pages_info[-1]
        else:
            following = [p for p in pages_info
                         if isinstance(p["step"], int) and p["step"] > cur["step"]]
            next_page = min(following, key=lambda p: p["step"]) if following else None

        if next_page:
            db.add_element_leads_to(
                cur["element_id"],
                next_page["page_id"],
                action_name=cur["action"],
                # FIX: json.dumps so Neo4j receives a string, not a dict
                action_params=json.dumps({
                    "execution_result": cur["status"],
                    "timestamp":        cur["timestamp"],
                }),
            )

    db.close()
    return task_id


# ─────────────────────────────────────────────────────────────────────────────
#  Element resolution  –  handles tap, long_press, swipe, text
# ─────────────────────────────────────────────────────────────────────────────

_EMPTY = {"ID": "", "bbox": [], "type": "", "content": ""}


def _extract_element_info(
    action_type: str,
    tool_result: dict,
    recommended_action: str,
    elements_data: list,
    elements_path: Optional[str],
) -> dict:
    """
    Return the best element dict we can find for any action type.

    tap / long_press
        The original code: use clicked_element coords → _pos2id.

    swipe
        clicked_element is always null for swipes because the ADB command
        moves FROM a start point, not taps it.  But tool_result["swipe"]["start"]
        holds the pixel coords of the element the user intended to swipe from.
        Pass those to _pos2id exactly like a tap.

    text
        No coordinates exist at all — typing doesn't produce a click event.
        However, the element number that was resolved at record-time is
        embedded in the recommended_action string:
            "Executing text with params {'element_number': 15, ...}"
        Parse that integer and look the element up by ID directly with
        _id2element().

    anything else (back, wait, swipe_precise)
        Return _EMPTY — there is no meaningful element to associate.
    """
    if not elements_path or not elements_data:
        # No parsed JSON available — nothing can be resolved regardless of action.
        return _EMPTY

    if action_type in ("tap", "long_press"):
        clicked = tool_result.get("clicked_element")
        if clicked:
            return _pos2id(clicked["x"], clicked["y"], elements_path) or _EMPTY
        return _EMPTY

    if action_type == "swipe":
        # tool_result["swipe"]["start"] = [x, y] in absolute pixel coords
        swipe_data = tool_result.get("swipe", {})
        start = swipe_data.get("start")
        if start and len(start) == 2:
            return _pos2id(start[0], start[1], elements_path) or _EMPTY
        return _EMPTY

    if action_type == "text":
        # Parse element_number from the recommended_action string.
        # e.g. "Executing text with params {'element_number': 15, ...}"
        elem_num = _parse_element_number(recommended_action)
        if elem_num is not None:
            return _id2element(elem_num, elements_data) or _EMPTY
        return _EMPTY

    return _EMPTY


def _parse_element_number(recommended_action: str) -> Optional[int]:
    """
    Extract the integer after 'element_number':  from the recommended_action
    string.  Returns None if not found or not parseable.

    Example input:
        "Executing text with params {'element_number': 15, 'text_input': 'weather'}"
    Returns:
        15
    """
    match = re.search(r"['\"]?element_number['\"]?\s*:\s*(\d+)", recommended_action)
    if match:
        return int(match.group(1))
    return None


def _id2element(element_id: int, elements_data: list) -> Optional[dict]:
    """
    Find an element by its integer ID field directly from the loaded
    elements list.  Used for text actions where we have the element number
    but no click coordinates.

    Returns the matching element dict, or None.
    """
    return next(
        (e for e in elements_data if e.get("ID") == element_id),
        None,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Private helpers  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _pos2id(x: int, y: int, json_path: str) -> Optional[Dict]:
    """Map absolute pixel coords → nearest element dict."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            elements = json.load(f)

        nx, ny = x / 1080, y / 2340

        hit = next(
            (e for e in elements
             if e["bbox"][0] <= nx <= e["bbox"][2]
             and e["bbox"][1] <= ny <= e["bbox"][3]),
            None,
        )
        if hit:
            return hit

        return min(
            elements,
            key=lambda e: (
                ((nx - (e["bbox"][0] + e["bbox"][2]) / 2) ** 2)
                + ((ny - (e["bbox"][1] + e["bbox"][3]) / 2) ** 2)
            ),
            default=None,
        )
    except Exception as exc:
        print(f"[_pos2id] {exc}")
        return None


def _element2vector(
    ID: str,
    element_id: str,
    elements_json: str,
    page_path: str,
    vs: VectorStore,
) -> bool:
    """Crop element image → ResNet50 features → Pinecone upsert."""
    try:
        img      = element_img(page_path, elements_json, int(ID))
        features = extract_features(img, "resnet50")
        elements = json.loads(elements_json)
        target   = next((e for e in elements if e.get("ID") == int(ID)), None)
        if target is None:
            raise ValueError(f"Element {ID} not found in JSON")

        vd = VectorData(
            id=element_id,
            values=features["features"][0],
            metadata={
                "original_id": str(ID),
                "bbox":    target["bbox"],
                "type":    target.get("type", ""),
                "content": target.get("content", ""),
            },
            node_type=NodeType.ELEMENT,
        )
        return vs.upsert_batch([vd])
    except Exception as exc:
        print(f"[_element2vector] {exc}")
        return False
