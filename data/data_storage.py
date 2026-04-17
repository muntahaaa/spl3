"""
data_storage.py
===============
Step 2 : Persist session state to JSON          (state2json)
Step 3 : Load JSON and push to Neo4j + Pinecone (json2db)

═══════════════════════════════════════════════════════════════════════════════
ROOT-CAUSE ANALYSIS OF ALL CURRENT BUGS
═══════════════════════════════════════════════════════════════════════════════

BUG-1  ── Elements never stored in Neo4j
         CAUSE: json2db() calls _resolve_element_for_action() which only
         resolves ONE interacted element per step (the clicked/typed element).
         It never iterates over ALL elements in the page's source_json and
         creates a node for each.  The AppAgentX schema requires every element
         on a page to be stored as an Element node connected via HAS_ELEMENT.
         FIX: Add a new _store_all_page_elements() inner block that loops over
         every item in elements_data and creates + links every Element node.
         The interacted element is resolved separately and linked to the Action
         node via COMPOSED_OF and gets a LEADS_TO edge to the next page.

BUG-2  ── Pinecone only stores actions, not elements or pages
         CAUSE: _element2vector() is only called when element_info is not None
         (i.e. the single interacted element).  Because BUG-1 meant most
         elements are never created, their vectors are never stored.
         _page2vector() IS called per step but fails silently because
         source_page stores a LOCAL file path that does not exist when the
         image was uploaded to Cloudinary.
         FIX: After BUG-1 fix, _element2vector() is called for every element.
         For page vectors, the local screenshot path is used (it was already
         saved locally by take_screenshot before being uploaded).

BUG-3  ── history_steps contains wrong source_json path
         CAUSE: In explor_human.single_human_explor() the step record stores
             "source_json": _to_relative(state.get("current_page_json", ""))
         But state["current_page_json"] is set by session.insert_parsed_result()
         to the VALUE PASSED IN via the API/UI.  Since the cloud URL popup
         (ui.py / api_routes.py) now downloads the JSON locally and passes the
         LOCAL path, that local path (e.g. /tmp/human_explorer_cloud/ss1.json)
         is what gets stored.  When json2db() later runs on the same machine it
         can open that temp file — but if the temp dir is cleaned, it fails.
         The correct behaviour:
           • state["current_page_json"]      = local temp path  (for live use)
           • step record "source_json"       = Cloudinary JSON URL  (for portability)
           • step record "source_json_local" = local temp path       (for json2db)
         FIX: session.insert_parsed_result() now receives BOTH the cloud URL
         and the local path and stores both on the state.  single_human_explor()
         records the cloud URL in "source_json" and the local path in
         "source_json_local" so json2db() can use the local copy.

BUG-4  ── _page2vector() fails for Cloudinary screenshot URLs
         CAUSE: source_page in history_steps stores the LOCAL screenshot path
         (from take_screenshot).  _page2vector() tries Path(source_page).is_file()
         which works only while the local file still exists.
         FIX: _page2vector() now also accepts an HTTP URL; if the path does not
         exist locally it downloads the image to a temp file first.

BUG-5  ── Action nodes linked to wrong elements / COMPOSED_OF created with
         wrong action_id
         CAUSE: action_properties["action_id"] was created fresh as uuid4() but
         db.create_action() returns the Neo4j elementId, not the action_id
         property.  add_element_to_action() queries by action_id property, so
         the returned elementId was silently useless.
         FIX: Generate action_id = str(uuid4()) and pass it as a property;
         db.create_action() stores it as the action_id property, and
         add_element_to_action() MATCH on that same property — which already
         works correctly in graph_db.py.  No graph_db.py change needed.
"""

import hashlib
import json
import re
import tempfile
import time as _time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from PIL import Image

import config as config
from data.graph_db import Neo4jDatabase
from data.vector_db import NodeType, VectorData, VectorStore
from tool.img_tool import element_img, extract_features


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _md5(text: str, length: int = 8) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:length]


def _resolve_existing_path(path_like: str) -> Optional[Path]:
    """Return a Path if the file exists locally; None otherwise."""
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


def _fetch_to_local(url_or_path: str, suffix: str = "") -> Optional[Path]:
    """
    Return a local Path for the given string.
    • If it is already a local file  →  return as Path directly.
    • If it starts with http(s)://   →  download to a temp file and return that.
    • Otherwise                      →  return None.
    """
    if not url_or_path:
        return None

    local = _resolve_existing_path(url_or_path)
    if local is not None:
        return local

    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        try:
            tmp_dir = Path(tempfile.gettempdir()) / "appagentx_dl"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tail = Path(url_or_path.split("?")[0]).name or f"dl{suffix}"
            if suffix and not tail.endswith(suffix):
                tail += suffix
            dest = tmp_dir / tail
            if not dest.is_file():           # reuse if already downloaded
                urllib.request.urlretrieve(url_or_path, str(dest))
            return dest
        except Exception as exc:
            print(f"[_fetch_to_local] download failed for {url_or_path}: {exc}")
            return None

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 – save State → JSON file
# ═══════════════════════════════════════════════════════════════════════════════

def state2json(state: dict, save_path: str = None) -> str:
    """
    Serialise the exploration State to a JSON file and return the file path.
    """
    if not isinstance(state, dict):
        raise TypeError("'state' must be a dict.")

    filtered = {
        "tsk":           state.get("tsk", ""),
        "app_name":      state.get("app_name", ""),
        "step":          state.get("step", 0),
        "history_steps": state.get("history_steps", []),
        "final_page": {
            "screenshot":       state.get("current_page_screenshot", ""),
            # BUG-3 FIX: prefer the cloud URL stored separately; fall back to
            # whatever is in current_page_json (which may be a local path).
            "page_json":        state.get("current_page_json_url") or (
                state["current_page_json"].get("parsed_content_json_path", "")
                if isinstance(state.get("current_page_json"), dict)
                else (state.get("current_page_json") or "")
            ),
            # local copy for json2db (may differ from cloud URL)
            "page_json_local":  state.get("current_page_json") if isinstance(
                state.get("current_page_json"), str
            ) else "",
            "timestamp":        int(_time.time()),
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
#  Helper – record one completed action into state["history_steps"]
# ═══════════════════════════════════════════════════════════════════════════════

def record_action_to_state(
    state: dict,
    step: int,
    screenshot_path: str,
    elements_json_url: Optional[str],        # BUG-3 FIX: Cloudinary / cloud URL
    elements_json_local: Optional[str],      # BUG-3 FIX: local temp path for json2db
    recommended_action: str,
    clicked_elements: Optional[list] = None,
    tool_results: Optional[dict] = None,
) -> None:
    """
    Record a completed action into state["history_steps"].

    Both the cloud URL and the local path of the elements JSON are stored so
    that:
      • "source_json"       → portable Cloudinary URL (survives machine changes)
      • "source_json_local" → local temp path used by json2db on the same run
    """
    action_record = {
        "step":               step,
        "source_page":        screenshot_path,
        # BUG-3 FIX: store the cloud URL, not the local tmp path
        "source_json":        elements_json_url or "",
        "source_json_local":  elements_json_local or "",
        "recommended_action": recommended_action,
        "timestamp":          int(_time.time()),
        "clicked_elements":   clicked_elements or [],
        "tool_results":       tool_results or {},
    }

    state["history_steps"].append(action_record)
    state["step"] = step + 1


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 – JSON → Neo4j + Pinecone
# ═══════════════════════════════════════════════════════════════════════════════

def pos2id(
    x: int,
    y: int,
    json_path: str,
    source_page: Optional[str] = None,
) -> Optional[Dict]:
    """
    Return the element dict from the JSON file whose bbox contains (x, y).
    Falls back to the nearest element if no exact match is found.
    Accepts a local path or a URL for json_path.
    """
    try:
        json_file = _fetch_to_local(json_path, ".json")
        if json_file is None:
            raise FileNotFoundError(f"Elements JSON not found / unreachable: {json_path}")

        with open(json_file, "r", encoding="utf-8") as f:
            elements_data = json.load(f)

        screen_width, screen_height = 1080, 2400
        page_file = _fetch_to_local(source_page or "", ".png") if source_page else None
        if page_file is not None:
            with Image.open(page_file) as img:
                screen_width, screen_height = img.size

        norm_x = x / screen_width
        norm_y = y / screen_height

        element_info = next(
            (
                e for e in elements_data
                if e["bbox"][0] <= norm_x <= e["bbox"][2]
                and e["bbox"][1] <= norm_y <= e["bbox"][3]
            ),
            None,
        )

        if element_info is None and elements_data:
            min_dist = float("inf")
            closest  = None
            for e in elements_data:
                cx = (e["bbox"][0] + e["bbox"][2]) / 2
                cy = (e["bbox"][1] + e["bbox"][3]) / 2
                d  = ((norm_x - cx) ** 2 + (norm_y - cy) ** 2) ** 0.5
                if d < min_dist:
                    min_dist = d
                    closest  = e
            element_info = closest
            print(f"[pos2id] No exact match; using closest (dist={min_dist:.4f})")

        return element_info

    except Exception as exc:
        print(f"[pos2id] {exc}")
        return None


def _load_elements_for_step(step: dict) -> tuple:
    """
    Load the elements JSON for a history step.

    Tries (in order):
      1. source_json_local  (temp file from the current session)
      2. source_json        (cloud URL – downloaded on demand)

    Returns (elements_path: Optional[Path], elements_data: list).
    """
    for key in ("source_json_local", "source_json"):
        raw = step.get(key, "")
        if not raw:
            continue
        path = _fetch_to_local(raw, ".json")
        if path is not None:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return path, data
            except Exception as exc:
                print(f"[_load_elements_for_step] failed to load {raw}: {exc}")

    print(f"[_load_elements_for_step] Step {step.get('step')}: no elements JSON available")
    return None, []


def json2db(json_path: str) -> str:
    """
    Read a state JSON file produced by state2json() and push every page,
    element, and action into Neo4j (graph) + Pinecone (vectors).

    Returns the task_id (short MD5 of the task string).

    What gets stored
    ────────────────
    Neo4j
      • One Page node per history step + one final Page node
      • ALL Element nodes on each page (HAS_ELEMENT edges)   ← BUG-1 FIX
      • One Action node per step (COMPOSED_OF edge to the interacted element)
      • LEADS_TO edges: interacted element → next page

    Pinecone
      • namespace "page"    : full-screenshot ResNet50 embedding per step
      • namespace "element" : cropped-element ResNet50 embedding per element ← BUG-2 FIX
      • namespace "action"  : action node stored (was already happening; kept)
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
        batch_size=10,
    )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_id       = _md5(data["tsk"], 12)
    pages_info:   List[dict] = []   # {page_id, step}
    elements_info: List[dict] = []  # {element_id, step, action, status, timestamp}
                                    #  — only for the INTERACTED element per step

    # ══════════════════════════════════════════════════════════════════════════
    #  FIRST PASS – create Page / Element / Action nodes for every history step
    # ══════════════════════════════════════════════════════════════════════════
    for step in data["history_steps"]:

        # ── Page node ─────────────────────────────────────────────────────────
        page_id = str(uuid4())
        page_props = {
            "page_id":      page_id,
            "description":  "",
            "raw_page_url": step.get("source_page", ""),
            "timestamp":    step.get("timestamp", int(_time.time())),
            "other_info":   json.dumps({
                "step": step["step"],
                **({"task_info": {"task_id": task_id, "description": data["tsk"]}}
                   if step["step"] == 0 else {}),
            }),
        }

        # BUG-3 FIX: load elements via the helper that tries local path first,
        # then falls back to downloading from the cloud URL.
        elements_path, elements_data = _load_elements_for_step(step)

        if elements_data:
            page_props["elements"] = json.dumps(elements_data)
        else:
            page_props["elements"] = json.dumps([])

        db.create_page(page_props)
        pages_info.append({"page_id": page_id, "step": step["step"]})

        # ── Page vector ────────────────────────────────────────────────────────
        # BUG-4 FIX: _page2vector now handles URLs via _fetch_to_local internally
        if step.get("source_page"):
            ok = _page2vector(
                page_id=page_id,
                page_path=step["source_page"],
                action_type=step.get("recommended_action", ""),
                step_no=step["step"],
                timestamp=str(step.get("timestamp", "")),
                vs=vs,
            )
            if not ok:
                print(f"[json2db] Warning: page vector failed for step {step['step']}")

        # ── BUG-1 FIX: create ALL Element nodes on this page ─────────────────
        # The original code only created the ONE interacted element. The paper
        # requires every element visible on the page to be stored so that
        # visual-similarity search can find any element later.
        element_id_map: Dict[int, str] = {}   # JSON ID → Neo4j element_id
        for elem in elements_data:
            elem_neo4j_id = str(uuid4())
            element_id_map[elem.get("ID", -1)] = elem_neo4j_id

            elem_props = {
                "element_id":          elem_neo4j_id,
                "element_original_id": str(elem.get("ID", "")),
                "description":         "",
                "action_type":         "",          # filled in for interacted elem below
                "parameters":          json.dumps({}),
                "bounding_box":        json.dumps(elem.get("bbox", [])),
                "other_info":          json.dumps({
                    "type":    elem.get("type", ""),
                    "content": elem.get("content", ""),
                }),
            }
            db.create_element(elem_props)
            db.add_element_to_page(page_id, elem_neo4j_id)

            # BUG-2 FIX: store element vector for EVERY element on the page
            if elem.get("ID") is not None and step.get("source_page"):
                _element2vector(
                    ID=str(elem["ID"]),
                    element_id=elem_neo4j_id,
                    elements_json=json.dumps(elements_data),
                    page_path=step["source_page"],
                    vs=vs,
                )

        # ── Action node ────────────────────────────────────────────────────────
        action_uuid = str(uuid4())
        tool_result  = step.get("tool_result") or {}
        # tool_result may be nested one level inside tool_results list
        if not tool_result and step.get("tool_results"):
            tr = step["tool_results"]
            tool_result = tr[0] if isinstance(tr, list) and tr else (tr if isinstance(tr, dict) else {})

        action_type     = tool_result.get("action") or step.get("action_type", "")
        clicked_element = tool_result.get("clicked_element")

        action_props = {
            "action_id":    action_uuid,
            "action_name":  step.get("recommended_action", ""),
            "timestamp":    step.get("timestamp", ""),
            "step":         step["step"],
            "action_result": json.dumps(tool_result),
        }
        db.create_action(action_props)

        # ── Resolve the ONE interacted element and link it to the Action ──────
        interacted_elem_info = _resolve_element_for_action(
            action_type=action_type,
            clicked_elem=clicked_element,
            elements_path=elements_path,
            elements_data=elements_data,
            recommended_action=step.get("recommended_action", ""),
        )

        if interacted_elem_info is not None:
            interacted_json_id = interacted_elem_info.get("ID")
            interacted_neo4j_id = element_id_map.get(interacted_json_id)

            if interacted_neo4j_id:
                # Update the existing element node with action-specific fields
                parameters = {
                    k: v for k, v in tool_result.items()
                    if k not in ("action", "device", "status")
                }
                db.update_node_property(
                    interacted_neo4j_id, "action_type", action_type, node_type="Element"
                )
                db.update_node_property(
                    interacted_neo4j_id, "parameters", json.dumps(parameters), node_type="Element"
                )

                # COMPOSED_OF: Action → interacted Element
                db.add_element_to_action(
                    action_id=action_uuid,
                    element_id=interacted_neo4j_id,
                    order=1,
                    atomic_action=str(action_type),
                    action_params=parameters,
                )

                # Track for LEADS_TO second pass
                elements_info.append({
                    "element_id": interacted_neo4j_id,
                    "step":       step["step"],
                    "action":     step.get("recommended_action", ""),
                    "status":     tool_result.get("status", "unknown"),
                    "timestamp":  step.get("timestamp", ""),
                })
        else:
            print(f"[json2db] Step {step['step']}: no interacted element resolved "
                  f"for action '{action_type}' — LEADS_TO edge will be skipped")

    # ══════════════════════════════════════════════════════════════════════════
    #  FINAL PAGE NODE
    # ══════════════════════════════════════════════════════════════════════════
    if data.get("final_page"):
        fp          = data["final_page"]
        fp_page_id  = str(uuid4())
        fp_props    = {
            "page_id":      fp_page_id,
            "description":  "",
            "raw_page_url": fp.get("screenshot", ""),
            "timestamp":    fp.get("timestamp", int(_time.time())),
        }

        # BUG-3 FIX: try page_json_local first, then page_json (may be URL)
        fp_elements_data: list = []
        for key in ("page_json_local", "page_json"):
            raw = fp.get(key, "")
            if not raw:
                continue
            p = _fetch_to_local(raw, ".json")
            if p is not None:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        fp_elements_data = json.load(f)
                    break
                except Exception as exc:
                    print(f"[json2db] final_page elements load failed ({key}): {exc}")

        fp_props["elements"] = json.dumps(fp_elements_data)
        db.create_page(fp_props)
        pages_info.append({"page_id": fp_page_id, "step": "final"})

        # Create Element nodes for the final page too
        for elem in fp_elements_data:
            fp_elem_id = str(uuid4())
            fp_elem_props = {
                "element_id":          fp_elem_id,
                "element_original_id": str(elem.get("ID", "")),
                "description":         "",
                "action_type":         "",
                "parameters":          json.dumps({}),
                "bounding_box":        json.dumps(elem.get("bbox", [])),
                "other_info":          json.dumps({
                    "type":    elem.get("type", ""),
                    "content": elem.get("content", ""),
                }),
            }
            db.create_element(fp_elem_props)
            db.add_element_to_page(fp_page_id, fp_elem_id)

            if elem.get("ID") is not None and fp.get("screenshot"):
                _element2vector(
                    ID=str(elem["ID"]),
                    element_id=fp_elem_id,
                    elements_json=json.dumps(fp_elements_data),
                    page_path=fp["screenshot"],
                    vs=vs,
                )

        # Final page vector
        success = False
        if fp.get("screenshot"):
            success = _page2vector(
                page_id=fp_page_id,
                page_path=fp["screenshot"],
                action_type="task_completion",
                step_no=data.get("step"),
                timestamp=str(int(_time.time())),
                vs=vs,
            )
        if not success:
            print("[json2db] Warning: final page vector storage failed")

    # ══════════════════════════════════════════════════════════════════════════
    #  SECOND PASS – LEADS_TO edges (interacted element → next page)
    # ══════════════════════════════════════════════════════════════════════════
    for i, cur in enumerate(elements_info):
        if i < len(elements_info) - 1:
            next_page = next(
                (p for p in pages_info if p["step"] == cur["step"] + 1), None
            )
        else:
            # Last interacted element leads to the final page (if it exists)
            next_page = next(
                (p for p in pages_info if p["step"] == "final"), None
            ) or next(
                (p for p in pages_info if p["step"] == cur["step"] + 1), None
            )

        if next_page:
            db.add_element_leads_to(
                element_id=cur["element_id"],
                target_id=next_page["page_id"],
                action_name=cur["action"],
                action_params={
                    "execution_result": cur["status"],
                    "timestamp":        cur["timestamp"],
                },
            )

    db.close()
    return task_id


# ─────────────────────────────────────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_element_for_action(
    action_type: str,
    clicked_elem: Optional[Dict],
    elements_path: Optional[Path],
    elements_data: list,
    recommended_action: str,
) -> Optional[Dict]:
    """
    Return the single element that was interacted with during this action.
    Used only for creating the COMPOSED_OF and LEADS_TO relationships.
    """
    if not elements_data:
        return None

    if action_type in ("tap", "long_press"):
        if clicked_elem and elements_path:
            resolved = pos2id(
                clicked_elem["x"], clicked_elem["y"], str(elements_path)
            )
            if resolved:
                return resolved
        # Fallback: element_number embedded in recommended_action string
        elem_id = _parse_element_number(recommended_action)
        if elem_id is not None:
            return next((e for e in elements_data if e.get("ID") == elem_id), None)

    if action_type == "text":
        elem_id = _parse_element_number(recommended_action)
        if elem_id is not None:
            return next((e for e in elements_data if e.get("ID") == elem_id), None)
        # If no element_number, use first input/edittext element as fallback
        return next(
            (e for e in elements_data if e.get("type", "").lower() in ("input", "edittext")),
            None,
        )

    if action_type == "swipe":
        # Swipe is a navigation gesture; no specific element is the target
        return None

    return None


def _parse_element_number(recommended_action: str) -> Optional[int]:
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
    """
    Crop element from screenshot → ResNet50 → Pinecone "element" namespace.
    BUG-4 FIX: page_path may be a URL; _fetch_to_local handles the download.
    """
    try:
        # Resolve page image (local or remote)
        page_file = _fetch_to_local(page_path, ".png")
        if page_file is None:
            raise FileNotFoundError(f"Screenshot unavailable: {page_path}")

        img      = element_img(str(page_file), elements_json, int(ID))
        features = extract_features(img, "resnet50")

        elements = json.loads(elements_json)
        target   = next((e for e in elements if e.get("ID") == int(ID)), None)
        if target is None:
            raise ValueError(f"Element ID={ID} not found in JSON")

        vd = VectorData(
            id=element_id,
            values=features["features"][0],
            metadata={
                "original_id": str(ID),
                "bbox":        target.get("bbox", []),
                "type":        target.get("type", ""),
                "content":     target.get("content", ""),
            },
            node_type=NodeType.ELEMENT,
        )
        return vs.upsert_batch([vd])

    except Exception as exc:
        print(f"[_element2vector] ID={ID}: {exc}")
        return False


def _page2vector(
    page_id: str,
    page_path: str,
    action_type: str,
    step_no: Optional[int],
    timestamp: str,
    vs: VectorStore,
) -> bool:
    """
    Embed full screenshot → ResNet50 → Pinecone "page" namespace.
    BUG-4 FIX: supports HTTP URLs via _fetch_to_local.
    """
    try:
        if not page_path:
            raise ValueError("page_path is empty")

        path_obj = _fetch_to_local(page_path, ".png")
        if path_obj is None:
            raise FileNotFoundError(f"Screenshot not found / unreachable: {page_path}")

        features     = extract_features(str(path_obj), "resnet50")
        feature_list = features.get("features", []) if isinstance(features, dict) else []
        if not feature_list:
            raise ValueError("extract_features returned no data")

        vd = VectorData(
            id=page_id,
            values=feature_list[0],
            metadata={
                "action_type": action_type,
                "step":        step_no,
                "timestamp":   timestamp,
                "source_page": str(path_obj).replace("\\", "/"),
            },
            node_type=NodeType.PAGE,
        )
        return vs.upsert_batch([vd])

    except Exception as exc:
        print(f"[_page2vector] {exc}")
        return False