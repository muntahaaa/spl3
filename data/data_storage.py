"""
data_storage.py
===============
Step 2 : Persist session state to JSON   (state2json)
Step 3 : Load JSON and push to Neo4j + Pinecone  (json2db)
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


# ── helpers ───────────────────────────────────────────────────────────────────

def _md5(text: str, length: int = 8) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:length]


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 – save State → JSON file
# ═══════════════════════════════════════════════════════════════════════════════

def state2json(state: dict, save_path: str = None) -> str:
    """
    Serialise the exploration State to a JSON file.

    Returns the absolute path string on success, or an error string.
    """
    if not isinstance(state, dict):
        raise TypeError("'state' must be a dict.")

    filtered = {
        "tsk":           state.get("tsk", ""),
        "app_name":      state.get("app_name", ""),
        "step":          state.get("step", 0),
        "history_steps": state.get("history_steps", []),
        "final_page": {
            "screenshot": state.get("current_page_screenshot", ""),
            "page_json":  (
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


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 – JSON file → Neo4j + Pinecone
# ═══════════════════════════════════════════════════════════════════════════════

def json2db(json_path: str) -> str:
    """
    Read a state JSON file and push every page + element into
    Neo4j (graph) and Pinecone (vectors).

    Returns the task_id (short MD5 of the task string).
    """
    db = Neo4jDatabase(
        uri=config.Neo4j_URI,
        auth=config.Neo4j_AUTH,
        database=config.Neo4j_DB,
    )
    vs = VectorStore(dimension=2048, batch_size=2)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_id    = _md5(data["tsk"], 12)
    pages_info: list   = []
    elements_info: list = []
    page_vector_attempted = 0
    page_vector_saved = 0
    page_vector_skipped = 0
    element_vector_attempted = 0
    element_vector_saved = 0
    element_vector_skipped = 0

    all_interactable_actions = {"swipe", "tap", "long_press", "text", "back", "wait"}

    # ── first pass: nodes + page→element edges ────────────────────────────────
    for step in data["history_steps"]:
        # --- page node ---
        page_props = {
            "page_id":     str(uuid4()),
            "description": "",
            "raw_page_url": step["source_page"],
            "timestamp":    step["timestamp"],
            "other_info":  json.dumps({
                "step": step["step"],
                **({"task_info": {"task_id": task_id, "description": data["tsk"]}}
                   if step["step"] == 0 else {}),
            }),
        }

        raw_source_json = step.get("source_json")
        elements_path = None
        elements_data = []
        if raw_source_json and Path(raw_source_json.replace("\\", "/")).is_file():
            elements_path = Path(raw_source_json.replace("\\", "/"))
            with open(elements_path, "r", encoding="utf-8") as f:
                elements_data = json.load(f)
            page_props["elements"] = json.dumps(elements_data)
        else:
            print(f"Step {step['step']}: source_json is null – storing page node only")

        db.create_page(page_props)
        pages_info.append({"page_id": page_props["page_id"], "step": step["step"]})

        # --- element node ---
        tool_result   = step["tool_result"]
        action_type   = tool_result.get("action")
        clicked_elem  = tool_result.get("clicked_element")

        # Save a page-level vector for all interactive actions, even when
        # there is no concrete target element (e.g., back/wait).
        if action_type in all_interactable_actions:
            page_vector_attempted += 1
            ok_page = _page2vector(
                page_id=page_props["page_id"],
                page_path=step.get("source_page", ""),
                action_type=action_type,
                step_no=step.get("step"),
                timestamp=step.get("timestamp", ""),
                vs=vs,
            )
            if ok_page:
                page_vector_saved += 1
            else:
                page_vector_skipped += 1
                print(f"[json2db] Page vector failed/skipped (step {step['step']}, action={action_type})")

        element_info = _resolve_element_for_action(
            action_type=action_type,
            clicked_elem=clicked_elem,
            elements_path=elements_path,
            elements_data=elements_data,
            recommended_action=step.get("recommended_action", ""),
        )

        params = {k: v for k, v in tool_result.items()
                  if k not in ("action", "device", "status")}

        elem_props = {
            "element_id":          str(uuid4()),
            "element_original_id": str(element_info["ID"]) if element_info and element_info.get("ID") is not None else None,
            "description":         "",
            "action_type":         action_type,
            "parameters":          json.dumps(params),
            "bounding_box":        element_info.get("bbox") if element_info else None,
            "other_info":          json.dumps({
                "type":           element_info.get("type") if element_info else None,
                "content":        element_info.get("content") if element_info else None,
                "parsed_element": element_info if element_info else None,
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

        # Vector storage for interactions that target a concrete UI element.
        if action_type in {"tap", "long_press", "text"}:
            element_vector_attempted += 1
            if not elements_data:
                element_vector_skipped += 1
                print(f"[json2db] Skip vector (step {step['step']}, action={action_type}): source_json missing or unreadable")
            elif not element_info:
                element_vector_skipped += 1
                print(f"[json2db] Skip vector (step {step['step']}, action={action_type}): parsed element is null")
            elif element_info.get("ID") is None:
                element_vector_skipped += 1
                print(f"[json2db] Skip vector (step {step['step']}, action={action_type}): parsed element has no ID")
            else:
                ok = _element2vector(
                    str(element_info["ID"]),
                    elem_props["element_id"],
                    json.dumps(elements_data),
                    step["source_page"],
                    vs,
                )
                if ok:
                    element_vector_saved += 1
                else:
                    element_vector_skipped += 1
                    print(f"[json2db] Vector storage failed (step {step['step']}) for element {element_info['ID']}")

    # ── final page node ───────────────────────────────────────────────────────
    if data.get("final_page"):
        fp = data["final_page"]
        fp_props = {
            "page_id":     str(uuid4()),
            "description": "",
            "raw_page_url": fp.get("screenshot", ""),
            "timestamp":    fp.get("timestamp", ""),
        }
        if fp.get("page_json"):
            ep = Path(fp["page_json"].replace("\\", "/"))
            with open(ep, "r", encoding="utf-8") as f:
                fp_props["elements"] = json.dumps(json.load(f))
        else:
            fp_props["elements"] = json.dumps([])

        db.create_page(fp_props)
        pages_info.append({"page_id": fp_props["page_id"], "step": "final"})

    # ── second pass: element→page LEADS_TO edges ──────────────────────────────
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
                action_params={
                    "execution_result": cur["status"],
                    "timestamp":        cur["timestamp"],
                },
            )

    db.close()
    print(
        "[json2db] Vector summary: "
        f"page_attempted={page_vector_attempted}, "
        f"page_saved={page_vector_saved}, "
        f"page_skipped={page_vector_skipped}, "
        f"element_attempted={element_vector_attempted}, "
        f"element_saved={element_vector_saved}, "
        f"element_skipped={element_vector_skipped}"
    )
    return task_id


# ── private helpers ───────────────────────────────────────────────────────────

def _pos2id(x: int, y: int, json_path: str) -> Optional[Dict]:
    """Map absolute pixel coords → nearest element dict."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            elements = json.load(f)

        # normalise (assumes 1080×2340 unless we have device_info)
        nx, ny = x / 1080, y / 2340

        hit = next(
            (e for e in elements
             if e["bbox"][0] <= nx <= e["bbox"][2]
             and e["bbox"][1] <= ny <= e["bbox"][3]),
            None,
        )
        if hit:
            return hit

        # fallback: closest centroid
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


def _resolve_element_for_action(
    action_type: str,
    clicked_elem: Optional[Dict],
    elements_path: Optional[Path],
    elements_data: list,
    recommended_action: str,
) -> Optional[Dict]:
    """Resolve an interacted element for tap/long_press/text actions."""
    if action_type in ("tap", "long_press") and clicked_elem and elements_path:
        return _pos2id(clicked_elem["x"], clicked_elem["y"], str(elements_path))

    if action_type == "text" and elements_data:
        # text actions may not carry clicked_element; recover target via element_number
        # embedded in the recommended_action string.
        elem_id = _parse_element_number(recommended_action)
        if elem_id is None:
            return None
        return next((e for e in elements_data if e.get("ID") == elem_id), None)

    return None


def _parse_element_number(recommended_action: str) -> Optional[int]:
    """Extract element_number from recommended_action text."""
    if not recommended_action:
        return None
    match = re.search(r"['\"]?element_number['\"]?\s*:\s*(\d+)", recommended_action)
    return int(match.group(1)) if match else None


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


def _page2vector(
    page_id: str,
    page_path: str,
    action_type: str,
    step_no: Optional[int],
    timestamp: str,
    vs: VectorStore,
) -> bool:
    """Embed full screenshot and upsert as a page vector."""
    try:
        if not page_path:
            raise ValueError("page_path is empty")

        normalized = page_path.replace("\\", "/")
        path_obj = Path(normalized)
        if not path_obj.is_file():
            path_obj = Path.cwd() / normalized
        if not path_obj.is_file():
            raise FileNotFoundError(f"Screenshot not found: {page_path}")

        features = extract_features(str(path_obj), "resnet50")
        feature_list = features.get("features", []) if isinstance(features, dict) else []
        if not feature_list:
            raise ValueError("No features returned for page image")

        vd = VectorData(
            id=page_id,
            values=feature_list[0],
            metadata={
                "action_type": action_type,
                "step": step_no,
                "timestamp": timestamp,
                "source_page": str(path_obj).replace("\\", "/"),
            },
            node_type=NodeType.PAGE,
        )
        return vs.upsert_batch([vd])
    except Exception as exc:
        print(f"[_page2vector] {exc}")
        return False
