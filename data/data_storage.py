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
from PIL import Image
from data.State import State
import config as config
from data.graph_db import Neo4jDatabase
from data.vector_db import NodeType, VectorData, VectorStore
from tool.img_tool import element_img, extract_features


# ── helpers ───────────────────────────────────────────────────────────────────

def _md5(text: str, length: int = 8) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:length]


def _resolve_existing_path(path_like: str) -> Optional[Path]:
    if not path_like:
        return None

    normalized = path_like.replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_file():
        return candidate

    cwd_candidate = Path.cwd() / normalized
    if cwd_candidate.is_file():
        return cwd_candidate

    return None


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
           # Add final_page field, including the screenshot and JSON information of the last page
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
def pos2id(
    x: int,
    y: int,
    json_path: str,
    source_page: Optional[str] = None,
) -> Optional[Dict]:
    """
    Find matching element information from the JSON file based on coordinates.
    If no direct match is found, return the nearest element.

    Args:
        x: The x-coordinate of the click position.
        y: The y-coordinate of the click position.
        json_path: The path to the element JSON file.

    Returns:
        Optional[Dict]: A dictionary of the matched element information, or None if not found.
    """
    try:
        json_file = _resolve_existing_path(json_path)
        if json_file is None:
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        # Read the element JSON file
        with open(json_file, "r", encoding="utf-8") as f:
            elements_data = json.load(f)

        # Convert absolute coordinates to relative coordinates.
        # Prefer the real screenshot dimensions when available.
        screen_width, screen_height = 1080, 2400
        source_page_file = _resolve_existing_path(source_page or "")
        if source_page_file is not None:
            with Image.open(source_page_file) as img:
                screen_width, screen_height = img.size

        norm_x = x / screen_width
        norm_y = y / screen_height

        # Match specific elements from element data
        element_info = next(
            (
                e
                for e in elements_data
                if e["bbox"][0] <= norm_x <= e["bbox"][2]
                and e["bbox"][1] <= norm_y <= e["bbox"][3]
            ),
            None,
        )

        # If no direct match is found, find the closest element
        if element_info is None and elements_data:
            min_distance = float("inf")
            closest_element = None

            for element in elements_data:
                bbox = element["bbox"]
                # Calculate the center point of the bounding box
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2

                # Calculate the Euclidean distance from the click position to the center of the bounding box
                distance = ((norm_x - center_x) ** 2 + (norm_y - center_y) ** 2) ** 0.5

                # Update the closest element
                if distance < min_distance:
                    min_distance = distance
                    closest_element = element

            element_info = closest_element
            print(
                f"No direct match found. Using closest element with distance {min_distance:.4f}"
            )

        return element_info

    except Exception as e:
        print(f"Error in pos2id: {str(e)}")
        return None

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
    vs = VectorStore(
        api_key=config.PINECONE_API_KEY,
        index_name=config.PINECONE_INDEX_NAME,
        dimension=2048,
        batch_size=2,
    )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Generate task ID
    task_id    = _md5(data["tsk"], 12)
    
    # Store page and element information
    pages_info: list   = []
    elements_info: list = []
    # ── first pass: create nodes and page→element edges ────────────────────────────────
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
        # Read element JSON file
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


        # Create page node
        db.create_page(page_props)
        pages_info.append({"page_id": page_props["page_id"], "step": step["step"]})

        # Modify element node processing logic
        tool_result   = step["tool_result"]
        action_type   = tool_result.get("action")
        clicked_element  = tool_result.get("clicked_element")
        
        
        
        

     
        if action_type == "tap" and clicked_element and step.get("source_json"):
            element_info = pos2id(
                clicked_element["x"],
                clicked_element["y"],
                step["source_json"].replace("\\", "/"),
                source_page=step.get("source_page"),
            )
        else:
            element_info = {"ID": "", "bbox": [], "type": "", "content": ""}

        if element_info:
            parameters = {
                k: v
                for k, v in tool_result.items()
                if k not in ["action", "device", "status"]
            }

            element_properties = {
                "element_id": str(uuid4()),
                "element_original_id": element_info.get("ID", ""),
                "description": "",
                "action_type": action_type,
                "parameters": json.dumps(parameters),
                "bounding_box": element_info.get("bbox", []),
                "other_info": json.dumps(
                    {
                        "type": element_info.get("type", ""),
                        "content": element_info.get("content", ""),
                    }
                ),
            }

            # Create element node
            db.create_element(element_properties)
            elements_info.append(
                {
                    "element_id": element_properties["element_id"],
                    "step": step["step"],
                    "action": step["recommended_action"],
                    "status": tool_result["status"],
                    "timestamp": step["timestamp"],
                }
            )

            # Establish element to page ownership relationship
            db.add_element_to_page(
                page_props["page_id"], element_properties["element_id"]
            )

            # Process the visual features of the element and store them in the vector database
            if element_info.get("ID"):  # Only process valid element ID
                success = _element2vector(
                    str(element_info["ID"]),
                    element_properties["element_id"],
                    json.dumps(elements_data),
                    step["source_page"],
                    vs,
                )
                if not success:
                    print(
                        f"Warning: Vector storage failed for element {element_info['ID']}"
                    )

    # Create final page node (if exists)
    if data.get("final_page"):
        final_page_properties = {
            "page_id": str(uuid4()),
            "description": "",
            "raw_page_url": data["final_page"].get("screenshot", ""),
            "timestamp": data["final_page"].get("timestamp", ""),
        }

        # Read final page element JSON
        if data["final_page"].get("page_json"):
            elements_path = _resolve_existing_path(data["final_page"]["page_json"])
            if elements_path is not None:
                with open(elements_path, "r", encoding="utf-8") as f:
                    elements_data = json.load(f)
                    final_page_properties["elements"] = json.dumps(elements_data)
            else:
                final_page_properties["elements"] = json.dumps([])
        else:
            final_page_properties["elements"] = json.dumps(
                []
            )  # If no element data, set to empty list

        # Create final page node
        db.create_page(final_page_properties)
        pages_info.append(
            {"page_id": final_page_properties["page_id"], "step": "final"}
        )

    # ── second pass:  establish element->page leads_to relationship ──────────────────────────────
    for i in range(len(elements_info)):
        current_element = elements_info[i]
        next_page = None

        # If it's the last element, point to the final page (if exists)
        if i == len(elements_info) - 1 and len(pages_info) > len(elements_info):
            next_page = pages_info[-1]  # Last page (final page)
        else:
            # Otherwise point to the next regular page
            next_page = next(
                (p for p in pages_info if p["step"] == current_element["step"] + 1),
                None,
            )

        if next_page:
            db.add_element_leads_to(
                current_element["element_id"],
                next_page["page_id"],
                action_name=current_element["action"],
                action_params={
                    "execution_result": current_element["status"],
                    "timestamp": current_element["timestamp"],
                },
            )
    return task_id


# ── private helpers ───────────────────────────────────────────────────────────



def _resolve_element_for_action(
    action_type: str,
    clicked_elem: Optional[Dict],
    elements_path: Optional[Path],
    elements_data: list,
    recommended_action: str,
) -> Optional[Dict]:
    """Resolve an interacted element for tap/long_press/text actions."""
    if action_type in ("tap", "long_press") and clicked_elem and elements_path:
        return pos2id(clicked_elem["x"], clicked_elem["y"], str(elements_path))

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
    """Crop element image → ResNet50 features → Pinecone upsert.
     Parameters:
        ID: str, the ID of the element in the JSON
        element_id: str, the unique ID of the element in the graph database
        elements_json: str, the element JSON string
        page_path: str, the page image path
        vector_store: VectorStore, the vector database instance

    Returns:
        bool: Whether the storage was successful"""
    try:
        # 1. Extract element image
        img      = element_img(page_path, elements_json, int(ID))
        
        # 2. Extract visual features
        features = extract_features(img, "resnet50")
        
        # 3. Parse JSON string to get element information
        elements = json.loads(elements_json)
        target   = next((e for e in elements if e.get("ID") == int(ID)), None)
        if target is None:
            raise ValueError(f"Element {ID} not found in JSON")
        
        # 4. Prepare vector data
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
         # 5. Store in vector database
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
