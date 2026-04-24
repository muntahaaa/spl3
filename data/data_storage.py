"""
data_storage.py  —  patched
============================

Seven bugs fixed.  Each fix is marked inline with BUG-N FIX.

Root-cause catalogue
────────────────────
BUG-1  Page description always ""
       CAUSE: page_props["description"] is hard-coded to "".
       FIX  : _make_page_description() builds a readable string from task +
              step + element count.

BUG-2  ()->LEADS_TO->(p:Page) returns more than the first page
       CAUSE: get_chain_start_nodes() in graph_db.py uses
              NOT EXISTS{()-[:LEADS_TO]->(n)}.  Every page that has no
              incoming LEADS_TO qualifies, including intermediate pages
              that simply haven't had their in-edge created yet at query time.
       FIX  : Tighten to also require the page has at least one HAS_ELEMENT
              edge (a genuine start page has elements; final pages do not).
              See graph_db.py.

BUG-3  (p:Page)->LEADS_TO->() returns intermediate pages, not just the last
       CAUSE: get_chain_from_start() uses
              WHERE NOT EXISTS { (end)-[:HAS_ELEMENT]->() } as the
              end-node filter.  But every page has HAS_ELEMENT edges, so
              nothing is ever filtered out.
       FIX  : Change to WHERE NOT EXISTS { (end)-[:HAS_ELEMENT]->()-[:LEADS_TO]->() }
              — the last page is one from which no element leads further.
              See graph_db.py.

BUG-4  Element description and visual_embedding_id null for all elements
       (a) description: hard-coded "".
           FIX: _make_element_description() builds a string from type/content/ID.
       (b) visual_embedding_id: after _element2vector() succeeds, its Pinecone
           vector id (= elem_neo4j_id UUID) is never written back to Neo4j.
           FIX: call db.update_node_property("visual_embedding_id", …) after
           a successful upsert.
data_storage.py  —  patched
============================

Seven bugs fixed.  Each fix is marked inline with BUG-N FIX.

Root-cause catalogue
────────────────────
BUG-1  Page description always ""
       CAUSE: page_props["description"] is hard-coded to "".
       FIX  : _make_page_description() builds a readable string from task +
              step + element count.

BUG-2  ()->LEADS_TO->(p:Page) returns more than the first page
       CAUSE: get_chain_start_nodes() in graph_db.py uses
              NOT EXISTS{()-[:LEADS_TO]->(n)}.  Every page that has no
              incoming LEADS_TO qualifies, including intermediate pages
              that simply haven't had their in-edge created yet at query time.
       FIX  : Tighten to also require the page has at least one HAS_ELEMENT
              edge (a genuine start page has elements; final pages do not).
              See graph_db.py.

BUG-3  (p:Page)->LEADS_TO->() returns intermediate pages, not just the last
       CAUSE: get_chain_from_start() uses
              WHERE NOT EXISTS { (end)-[:HAS_ELEMENT]->() } as the
              end-node filter.  But every page has HAS_ELEMENT edges, so
              nothing is ever filtered out.
       FIX  : Change to WHERE NOT EXISTS { (end)-[:HAS_ELEMENT]->()-[:LEADS_TO]->() }
              — the last page is one from which no element leads further.
              See graph_db.py.

BUG-4  Element description and visual_embedding_id null for all elements
       (a) description: hard-coded "".
           FIX: _make_element_description() builds a string from type/content/ID.
       (b) visual_embedding_id: after _element2vector() succeeds, its Pinecone
           vector id (= elem_neo4j_id UUID) is never written back to Neo4j.
           FIX: call db.update_node_property("visual_embedding_id", …) after
           a successful upsert.

BUG-5  Multiple elements share the same element_original_id
       CAUSE: JSON element IDs restart from 0 on every page, so element_original_id
              "0" appears on every page.
       FIX  : Store f"{page_id}::{json_id}" — globally unique, still readable.
BUG-5  Multiple elements share the same element_original_id
       CAUSE: JSON element IDs restart from 0 on every page, so element_original_id
              "0" appears on every page.
       FIX  : Store f"{page_id}::{json_id}" — globally unique, still readable.

BUG-6  LEADS_TO confidence_score always 0.0
       CAUSE: db.add_element_leads_to() defaults confidence_score to 0.0 and
              the call site never passes a value.
       FIX  : _confidence_from_status() computes 1.0/0.5/0.0 from execution
              status; passed at the call site.

BUG-7  COMPOSED_OF order always 1
       CAUSE: db.add_element_to_action() called with order=1 hardcoded.
       FIX  : Pass order=step["step"] + 1 (1-based execution sequence).
BUG-6  LEADS_TO confidence_score always 0.0
       CAUSE: db.add_element_leads_to() defaults confidence_score to 0.0 and
              the call site never passes a value.
       FIX  : _confidence_from_status() computes 1.0/0.5/0.0 from execution
              status; passed at the call site.

BUG-7  COMPOSED_OF order always 1
       CAUSE: db.add_element_to_action() called with order=1 hardcoded.
       FIX  : Pass order=step["step"] + 1 (1-based execution sequence).
"""

import hashlib
import json
import re
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
    - Local file  : return as-is.
    - http(s) URL : download into the workspace log directory and return that.
    - Otherwise   : return None.
    """
    if not url_or_path:
        return None
    local = _resolve_existing_path(url_or_path)
    if local is not None:
        return local
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        try:
            log_root = Path.cwd() / "log"
            if suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                log_dir = log_root / "images"
            elif suffix.lower() == ".json":
                log_dir = log_root / "json_state"
            else:
                log_dir = log_root / "downloads"
            log_dir.mkdir(parents=True, exist_ok=True)
            tail = Path(url_or_path.split("?")[0]).name or f"dl{suffix}"
            if suffix and not tail.endswith(suffix):
                tail += suffix
            dest = log_dir / tail
            if not dest.is_file():
                urllib.request.urlretrieve(url_or_path, str(dest))
            return dest
        except Exception as exc:
            print(f"[_fetch_to_local] download failed for {url_or_path}: {exc}")
            return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  BUG-1 / BUG-4a FIX: description builders
# ─────────────────────────────────────────────────────────────────────────────

def _make_page_description(task: str, step_no: int,
                            elements_data: list, app_name: str = "") -> str:
    """
    BUG-1 FIX: Build a non-empty page description.
    Format: "[app] — Step N: <task> | K elements visible"
    Replace with an LLM call for richer descriptions.
    """
    prefix = f"{app_name} — " if app_name and app_name != "human_exploration" else ""
    k = len(elements_data)
    return f"{prefix}Step {step_no}: {task} | {k} element{'s' if k != 1 else ''} visible"


def _make_element_description(elem: dict, step_no: int) -> str:
    """
    BUG-4a FIX: Build a non-empty description for an Element node.
    Format: "<type> element (ID <id>) at step N — '<content>'"
    """
    t = elem.get("type", "unknown")
    c = str(elem.get("content", "")).strip()[:60]
    i = elem.get("ID", "?")
    desc = f"{t} element (ID {i}) at step {step_no}"
    if c:
        desc += f" — '{c}'"
    return desc


# ─────────────────────────────────────────────────────────────────────────────
#  BUG-6 FIX: confidence score
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_from_status(status: str) -> float:
    """
    BUG-6 FIX: Heuristic confidence score for LEADS_TO edges.
      "success"        -> 1.0
      other non-empty  -> 0.5
      empty/None       -> 0.0
    """
    if not status:
        return 0.0
    return 1.0 if str(status).lower() == "success" else 0.5


# ─────────────────────────────────────────────────────────────────────────────
#  BUG-1 / BUG-4a FIX: description builders
# ─────────────────────────────────────────────────────────────────────────────

def _make_page_description(task: str, step_no: int,
                            elements_data: list, app_name: str = "") -> str:
    """
    BUG-1 FIX: Build a non-empty page description.
    Format: "[app] — Step N: <task> | K elements visible"
    Replace with an LLM call for richer descriptions.
    """
    prefix = f"{app_name} — " if app_name and app_name != "human_exploration" else ""
    k = len(elements_data)
    return f"{prefix}Step {step_no}: {task} | {k} element{'s' if k != 1 else ''} visible"


def _make_element_description(elem: dict, step_no: int) -> str:
    """
    BUG-4a FIX: Build a non-empty description for an Element node.
    Format: "<type> element (ID <id>) at step N — '<content>'"
    """
    t = elem.get("type", "unknown")
    c = str(elem.get("content", "")).strip()[:60]
    i = elem.get("ID", "?")
    desc = f"{t} element (ID {i}) at step {step_no}"
    if c:
        desc += f" — '{c}'"
    return desc


# ─────────────────────────────────────────────────────────────────────────────
#  BUG-6 FIX: confidence score
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_from_status(status: str) -> float:
    """
    BUG-6 FIX: Heuristic confidence score for LEADS_TO edges.
      "success"        -> 1.0
      other non-empty  -> 0.5
      empty/None       -> 0.0
    """
    if not status:
        return 0.0
    return 1.0 if str(status).lower() == "success" else 0.5


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 – save State → JSON file
# ═══════════════════════════════════════════════════════════════════════════════

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
            "page_json":  state.get("current_page_json", ""),
            "timestamp": int(_time.time()),
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
        return f"Error saving state: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper – record one completed action into state["history_steps"]
# ═══════════════════════════════════════════════════════════════════════════════

def record_action_to_state(
    state: dict,
    step: int,
    screenshot_path: str,
    elements_json_path: Optional[str],
    recommended_action: str,
    clicked_elements: Optional[list] = None,
    tool_results: Optional[dict] = None,
) -> None:
    """
    Record a completed action into state["history_steps"].
    
    Args:
        state: State dict
        step: Step number
        screenshot_path: Local path to screenshot (from OmniParser or similar)
        elements_json_path: Local path to elements JSON (from OmniParser)
        recommended_action: Recommended action description
        clicked_elements: Optional list of clicked elements
        tool_results: Optional tool execution results
    """
    state["history_steps"].append({
        "step":               step,
        "source_page":        screenshot_path or "",
        "source_json":        elements_json_path or "",
        "recommended_action": recommended_action,
        "timestamp":          int(_time.time()),
        "clicked_elements":   clicked_elements or [],
        "tool_results":       tool_results or {},
    })
    })
    state["step"] = step + 1


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 – JSON → Neo4j + Pinecone
# ═══════════════════════════════════════════════════════════════════════════════

def pos2id(x: int, y: int, json_path: str,
           source_page: Optional[str] = None) -> Optional[Dict]:
def pos2id(x: int, y: int, json_path: str,
           source_page: Optional[str] = None) -> Optional[Dict]:
    try:
        json_file = _fetch_to_local(json_path, ".json")
        if json_file is None:
            raise FileNotFoundError(f"Elements JSON not found: {json_path}")
            raise FileNotFoundError(f"Elements JSON not found: {json_path}")
        with open(json_file, "r", encoding="utf-8") as f:
            elements_data = json.load(f)

        screen_width, screen_height = 1080, 2400
        page_file = _fetch_to_local(source_page or "", ".png") if source_page else None
        if page_file is not None:
            with Image.open(page_file) as img:
                screen_width, screen_height = img.size

        norm_x, norm_y = x / screen_width, y / screen_height
        norm_x, norm_y = x / screen_width, y / screen_height
        element_info = next(
            (e for e in elements_data
             if e["bbox"][0] <= norm_x <= e["bbox"][2]
             and e["bbox"][1] <= norm_y <= e["bbox"][3]),
            (e for e in elements_data
             if e["bbox"][0] <= norm_x <= e["bbox"][2]
             and e["bbox"][1] <= norm_y <= e["bbox"][3]),
            None,
        )
        if element_info is None and elements_data:
            min_dist, closest = float("inf"), None
            min_dist, closest = float("inf"), None
            for e in elements_data:
                cx = (e["bbox"][0] + e["bbox"][2]) / 2
                cy = (e["bbox"][1] + e["bbox"][3]) / 2
                d  = ((norm_x - cx) ** 2 + (norm_y - cy) ** 2) ** 0.5
                if d < min_dist:
                    min_dist, closest = d, e
                    min_dist, closest = d, e
            element_info = closest
        return element_info
    except Exception as exc:
        print(f"[pos2id] {exc}")
        return None


def _load_elements_for_step(step: dict) -> tuple:
    """
    Load elements JSON from step record.
    
    Args:
        step: Step record from history_steps
        
    Returns:
        tuple: (path_object, elements_list) or (None, []) if not found
    """
    raw = step.get("source_json", "")
    if not raw:
        print(f"[_load_elements_for_step] Step {step.get('step')}: no source_json")
        return None, []
    
    path = _fetch_to_local(raw, ".json")
    if path is not None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return path, json.load(f)
        except Exception as exc:
            print(f"[_load_elements_for_step] Failed to load {raw}: {exc}")
    
    print(f"[_load_elements_for_step] Step {step.get('step')}: could not load elements JSON")
    return None, []


def json2db(json_path: str) -> str:
    """
    Push state JSON → Neo4j + Pinecone.
    Returns task_id (MD5 of task string).

    BUG-2 and BUG-3 fixes live in graph_db.py (query rewrites).
    All other fixes are applied inline below.
    """
    Push state JSON → Neo4j + Pinecone.
    Returns task_id (MD5 of task string).

    BUG-2 and BUG-3 fixes live in graph_db.py (query rewrites).
    All other fixes are applied inline below.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_id   = _md5(data["tsk"], 12)
    task_name = data.get("tsk", "")
    app_name  = data.get("app_name", "")
    pages_info:          List[dict] = []
    elements_info:       List[dict] = []
    navigated_by_pending: List[dict] = []  # element-less steps awaiting NAVIGATED_BY edge

    try:
        db = Neo4jDatabase(uri=config.Neo4j_URI, auth=config.Neo4j_AUTH,
                           database=config.Neo4j_DB)

        vs = VectorStore(api_key=config.PINECONE_API_KEY,
                         index_name=config.PINECONE_INDEX_NAME,
                         dimension=2048, batch_size=10)
    task_id   = _md5(data["tsk"], 12)
    task_name = data.get("tsk", "")
    app_name  = data.get("app_name", "")
    pages_info:          List[dict] = []
    elements_info:       List[dict] = []
    navigated_by_pending: List[dict] = []  # element-less steps awaiting NAVIGATED_BY edge

    try:
        db = Neo4jDatabase(uri=config.Neo4j_URI, auth=config.Neo4j_AUTH,
                           database=config.Neo4j_DB)

        vs = VectorStore(api_key=config.PINECONE_API_KEY,
                         index_name=config.PINECONE_INDEX_NAME,
                         dimension=2048, batch_size=10)

        # ══════════════════════════════════════════════════════════════════════════
        #  FIRST PASS
        # ══════════════════════════════════════════════════════════════════════════
        for step in data["history_steps"]:
            step_no = step["step"]

            elements_path, elements_data = _load_elements_for_step(step)
        # ══════════════════════════════════════════════════════════════════════════
        #  FIRST PASS
        # ══════════════════════════════════════════════════════════════════════════
        for step in data["history_steps"]:
            step_no = step["step"]

            elements_path, elements_data = _load_elements_for_step(step)

            # ── Page node ─────────────────────────────────────────────────────────
            page_id = str(uuid4())

            page_props = {
                "page_id":      page_id,
                # BUG-1 FIX: meaningful description
                "description":  _make_page_description(task_name, step_no,
                                                        elements_data, app_name),
                "raw_page_url": step.get("source_page", ""),
                "timestamp":    step.get("timestamp", int(_time.time())),
                "elements":     json.dumps(elements_data),
                "other_info":   json.dumps({
                    "step": step_no,
                    **({"task_info": {"task_id": task_id, "description": task_name}}
                       if step_no == 0 else {}),
                }),
            }
            db.create_page(page_props)
            pages_info.append({"page_id": page_id, "step": step_no})
            # ── Page node ─────────────────────────────────────────────────────────
            page_id = str(uuid4())

            page_props = {
                "page_id":      page_id,
                # BUG-1 FIX: meaningful description
                "description":  _make_page_description(task_name, step_no,
                                                        elements_data, app_name),
                "raw_page_url": step.get("source_page", ""),
                "timestamp":    step.get("timestamp", int(_time.time())),
                "elements":     json.dumps(elements_data),
                "other_info":   json.dumps({
                    "step": step_no,
                    **({"task_info": {"task_id": task_id, "description": task_name}}
                       if step_no == 0 else {}),
                }),
            }
            db.create_page(page_props)
            pages_info.append({"page_id": page_id, "step": step_no})

            # ── Page vector ───────────────────────────────────────────────────────
            if step.get("source_page"):
                ok = _page2vector(page_id=page_id, page_path=step["source_page"],
                                  action_type=step.get("recommended_action", ""),
                                  step_no=step_no,
                                  timestamp=str(step.get("timestamp", "")), vs=vs)
                if not ok:
                    print(f"[json2db] page vector failed for step {step_no}")

            # ── ALL Element nodes for this page ───────────────────────────────────
            element_id_map: Dict[int, str] = {}
            # ── Page vector ───────────────────────────────────────────────────────
            if step.get("source_page"):
                ok = _page2vector(page_id=page_id, page_path=step["source_page"],
                                  action_type=step.get("recommended_action", ""),
                                  step_no=step_no,
                                  timestamp=str(step.get("timestamp", "")), vs=vs)
                if not ok:
                    print(f"[json2db] page vector failed for step {step_no}")

            # ── ALL Element nodes for this page ───────────────────────────────────
            element_id_map: Dict[int, str] = {}

            for elem in elements_data:
                elem_json_id  = elem.get("ID")
                elem_neo4j_id = str(uuid4())
                element_id_map[elem_json_id] = elem_neo4j_id
            for elem in elements_data:
                elem_json_id  = elem.get("ID")
                elem_neo4j_id = str(uuid4())
                element_id_map[elem_json_id] = elem_neo4j_id

                elem_props = {
                    "element_id":          elem_neo4j_id,
                    # BUG-5 FIX: scope to page so the id is globally unique
                    "element_original_id": f"{page_id}::{elem_json_id}",
                    # BUG-4a FIX: non-empty description
                    "description":         _make_element_description(elem, step_no),
                    "action_type":         "",
                    "parameters":          json.dumps({}),
                    "bounding_box":        json.dumps(elem.get("bbox", [])),
                    "other_info":          json.dumps({
                        "type":    elem.get("type", ""),
                        "content": elem.get("content", ""),
                    }),
                }
                db.create_element(elem_props)
                db.add_element_to_page(page_id, elem_neo4j_id)
                elem_props = {
                    "element_id":          elem_neo4j_id,
                    # BUG-5 FIX: scope to page so the id is globally unique
                    "element_original_id": f"{page_id}::{elem_json_id}",
                    # BUG-4a FIX: non-empty description
                    "description":         _make_element_description(elem, step_no),
                    "action_type":         "",
                    "parameters":          json.dumps({}),
                    "bounding_box":        json.dumps(elem.get("bbox", [])),
                    "other_info":          json.dumps({
                        "type":    elem.get("type", ""),
                        "content": elem.get("content", ""),
                    }),
                }
                db.create_element(elem_props)
                db.add_element_to_page(page_id, elem_neo4j_id)

                if elem_json_id is not None and step.get("source_page"):
                    vec_ok = _element2vector(
                        ID=str(elem_json_id),
                        element_id=elem_neo4j_id,
                        elements_json=json.dumps(elements_data),
                        page_path=step["source_page"],
                        vs=vs,
                    )
                    # BUG-4b FIX: write Pinecone vector id back to Neo4j
                    if vec_ok:
                        db.update_node_property(
                            node_id=elem_neo4j_id,
                            property_name="visual_embedding_id",
                            property_value=elem_neo4j_id,   # Pinecone id == UUID
                            node_type="Element",
                        )

            # ── Action node ───────────────────────────────────────────────────────
            action_uuid = str(uuid4())
            tool_result = step.get("tool_result") or {}
            if not tool_result and step.get("tool_results"):
                tr = step["tool_results"]
                tool_result = (tr[0] if isinstance(tr, list) and tr
                               else (tr if isinstance(tr, dict) else {}))
                if elem_json_id is not None and step.get("source_page"):
                    vec_ok = _element2vector(
                        ID=str(elem_json_id),
                        element_id=elem_neo4j_id,
                        elements_json=json.dumps(elements_data),
                        page_path=step["source_page"],
                        vs=vs,
                    )
                    # BUG-4b FIX: write Pinecone vector id back to Neo4j
                    if vec_ok:
                        db.update_node_property(
                            node_id=elem_neo4j_id,
                            property_name="visual_embedding_id",
                            property_value=elem_neo4j_id,   # Pinecone id == UUID
                            node_type="Element",
                        )

            # ── Action node ───────────────────────────────────────────────────────
            action_uuid = str(uuid4())
            tool_result = step.get("tool_result") or {}
            if not tool_result and step.get("tool_results"):
                tr = step["tool_results"]
                tool_result = (tr[0] if isinstance(tr, list) and tr
                               else (tr if isinstance(tr, dict) else {}))

            action_type     = tool_result.get("action") or step.get("action_type", "")
            clicked_element = tool_result.get("clicked_element")
            action_status   = tool_result.get("status", "")

            db.create_action({
                "action_id":     action_uuid,
                "action_name":   step.get("recommended_action", ""),
                "timestamp":     step.get("timestamp", ""),
                "step":          step_no,
                "action_result": json.dumps(tool_result),
            })
            action_type     = tool_result.get("action") or step.get("action_type", "")
            clicked_element = tool_result.get("clicked_element")
            action_status   = tool_result.get("status", "")

            db.create_action({
                "action_id":     action_uuid,
                "action_name":   step.get("recommended_action", ""),
                "timestamp":     step.get("timestamp", ""),
                "step":          step_no,
                "action_result": json.dumps(tool_result),
            })

            # ── Interacted element → Action (COMPOSED_OF) ─────────────────────────
            interacted = _resolve_element_for_action(
                action_type=action_type,
                clicked_elem=clicked_element,
                elements_path=elements_path,
                elements_data=elements_data,
                recommended_action=step.get("recommended_action", ""),
            )
            # ── Interacted element → Action (COMPOSED_OF) ─────────────────────────
            interacted = _resolve_element_for_action(
                action_type=action_type,
                clicked_elem=clicked_element,
                elements_path=elements_path,
                elements_data=elements_data,
                recommended_action=step.get("recommended_action", ""),
            )

            if interacted is not None:
                interacted_json_id  = interacted.get("ID")
                interacted_neo4j_id = element_id_map.get(interacted_json_id)
            if interacted is not None:
                interacted_json_id  = interacted.get("ID")
                interacted_neo4j_id = element_id_map.get(interacted_json_id)

                if interacted_neo4j_id:
                    parameters = {k: v for k, v in tool_result.items()
                                  if k not in ("action", "device", "status")}
                    db.update_node_property(interacted_neo4j_id, "action_type",
                                            action_type, node_type="Element")
                    db.update_node_property(interacted_neo4j_id, "parameters",
                                            json.dumps(parameters), node_type="Element")

                    # BUG-7 FIX: use step_no + 1 as execution order (1-based)
                    db.add_element_to_action(
                        action_id=action_uuid,
                        element_id=interacted_neo4j_id,
                        order=step_no + 1,              # BUG-7 FIX (was hardcoded 1)
                        atomic_action=str(action_type),
                        action_params=parameters,
                    )
                if interacted_neo4j_id:
                    parameters = {k: v for k, v in tool_result.items()
                                  if k not in ("action", "device", "status")}
                    db.update_node_property(interacted_neo4j_id, "action_type",
                                            action_type, node_type="Element")
                    db.update_node_property(interacted_neo4j_id, "parameters",
                                            json.dumps(parameters), node_type="Element")

                    # BUG-7 FIX: use step_no + 1 as execution order (1-based)
                    db.add_element_to_action(
                        action_id=action_uuid,
                        element_id=interacted_neo4j_id,
                        order=step_no + 1,              # BUG-7 FIX (was hardcoded 1)
                        atomic_action=str(action_type),
                        action_params=parameters,
                    )

                    elements_info.append({
                        "element_id": interacted_neo4j_id,
                        "step":       step_no,
                        "action":     step.get("recommended_action", ""),
                        "status":     action_status,
                        "timestamp":  step.get("timestamp", ""),
                    })
            else:
                # Only swipe_precise has no element by design (start/end coords,
                # not an element ID).  swipe and back now carry element IDs so
                # they reach this else branch only when no element was provided
                # (back without an element_number), which is still valid —
                # use NAVIGATED_BY for those cases too.
                # wait has been removed from the action set entirely.
                ELEMENT_LESS = ("swipe_precise",)
                if action_type in ELEMENT_LESS or action_type in ("back",):
                    navigated_by_pending.append({
                        "source_page_id":  page_id,
                        "source_step":     step_no,
                        "action_type":     action_type,
                        "action_params":   {
                            "execution_result": action_status,
                            "timestamp":        step.get("timestamp", ""),
                        },
                        "confidence_score": _confidence_from_status(action_status),
                    })
                else:
                    print(f"[json2db] Step {step_no}: could not resolve interacted element "
                          f"for action '{action_type}' — LEADS_TO edge skipped")

        # ── Write NAVIGATED_BY edges (after all page_ids are known) ──────────────
        max_history_step = max(
            (p["step"] for p in pages_info if isinstance(p["step"], int)), default=-1
        )
        for nav in navigated_by_pending:
            # Target is the page created for the next step number
            target = next(
                (p for p in pages_info if p["step"] == nav["source_step"] + 1), None
            )
            # If this was the last history step, the target is the final page
            if target is None and nav["source_step"] == max_history_step:
                target = next(
                    (p for p in pages_info if p["step"] == "final"), None
                )
            if target:
                db.add_page_navigated_by(
                    source_page_id=nav["source_page_id"],
                    target_page_id=target["page_id"],
                    action_type=nav["action_type"],
                    action_params=nav["action_params"],
                    confidence_score=nav["confidence_score"],
                )
            else:
                print(f"[json2db] NAVIGATED_BY: no target page for step "
                      f"{nav['source_step']} action '{nav['action_type']}'")
                    elements_info.append({
                        "element_id": interacted_neo4j_id,
                        "step":       step_no,
                        "action":     step.get("recommended_action", ""),
                        "status":     action_status,
                        "timestamp":  step.get("timestamp", ""),
                    })
            else:
                # Only swipe_precise has no element by design (start/end coords,
                # not an element ID).  swipe and back now carry element IDs so
                # they reach this else branch only when no element was provided
                # (back without an element_number), which is still valid —
                # use NAVIGATED_BY for those cases too.
                # wait has been removed from the action set entirely.
                ELEMENT_LESS = ("swipe_precise",)
                if action_type in ELEMENT_LESS or action_type in ("back",):
                    navigated_by_pending.append({
                        "source_page_id":  page_id,
                        "source_step":     step_no,
                        "action_type":     action_type,
                        "action_params":   {
                            "execution_result": action_status,
                            "timestamp":        step.get("timestamp", ""),
                        },
                        "confidence_score": _confidence_from_status(action_status),
                    })
                else:
                    print(f"[json2db] Step {step_no}: could not resolve interacted element "
                          f"for action '{action_type}' — LEADS_TO edge skipped")

        # ── Write NAVIGATED_BY edges (after all page_ids are known) ──────────────
        max_history_step = max(
            (p["step"] for p in pages_info if isinstance(p["step"], int)), default=-1
        )
        for nav in navigated_by_pending:
            # Target is the page created for the next step number
            target = next(
                (p for p in pages_info if p["step"] == nav["source_step"] + 1), None
            )
            # If this was the last history step, the target is the final page
            if target is None and nav["source_step"] == max_history_step:
                target = next(
                    (p for p in pages_info if p["step"] == "final"), None
                )
            if target:
                db.add_page_navigated_by(
                    source_page_id=nav["source_page_id"],
                    target_page_id=target["page_id"],
                    action_type=nav["action_type"],
                    action_params=nav["action_params"],
                    confidence_score=nav["confidence_score"],
                )
            else:
                print(f"[json2db] NAVIGATED_BY: no target page for step "
                      f"{nav['source_step']} action '{nav['action_type']}'")

        # ══════════════════════════════════════════════════════════════════════════
        #  FINAL PAGE NODE
        # ══════════════════════════════════════════════════════════════════════════
        if data.get("final_page"):
            fp         = data["final_page"]
            fp_page_id = str(uuid4())

            fp_elements_data: list = []
            raw_json = fp.get("page_json", "")
            if raw_json:
                p = _fetch_to_local(raw_json, ".json")
                if p is not None:
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            fp_elements_data = json.load(f)
                    except Exception as exc:
                        print(f"[json2db] final_page load failed: {exc}")

            total_steps = data.get("step", len(data["history_steps"]))

            db.create_page({
                "page_id":      fp_page_id,
                # BUG-1 FIX
                "description":  _make_page_description(task_name, total_steps,
                                                        fp_elements_data, app_name),
                "raw_page_url": fp.get("screenshot", ""),
                "timestamp":    fp.get("timestamp", int(_time.time())),
                "elements":     json.dumps(fp_elements_data),
            })
            pages_info.append({"page_id": fp_page_id, "step": "final"})
            total_steps = data.get("step", len(data["history_steps"]))

            db.create_page({
                "page_id":      fp_page_id,
                # BUG-1 FIX
                "description":  _make_page_description(task_name, total_steps,
                                                        fp_elements_data, app_name),
                "raw_page_url": fp.get("screenshot", ""),
                "timestamp":    fp.get("timestamp", int(_time.time())),
                "elements":     json.dumps(fp_elements_data),
            })
            pages_info.append({"page_id": fp_page_id, "step": "final"})

            for elem in fp_elements_data:
                fp_elem_id   = str(uuid4())
                fp_json_id   = elem.get("ID")

                db.create_element({
                    "element_id":          fp_elem_id,
                    "element_original_id": f"{fp_page_id}::{fp_json_id}",  # BUG-5 FIX
                    "description":         _make_element_description(elem, total_steps),  # BUG-4a FIX
                    "action_type":         "",
                    "parameters":          json.dumps({}),
                    "bounding_box":        json.dumps(elem.get("bbox", [])),
                    "other_info":          json.dumps({
                        "type":    elem.get("type", ""),
                        "content": elem.get("content", ""),
                    }),
                })
                db.add_element_to_page(fp_page_id, fp_elem_id)
            for elem in fp_elements_data:
                fp_elem_id   = str(uuid4())
                fp_json_id   = elem.get("ID")

                db.create_element({
                    "element_id":          fp_elem_id,
                    "element_original_id": f"{fp_page_id}::{fp_json_id}",  # BUG-5 FIX
                    "description":         _make_element_description(elem, total_steps),  # BUG-4a FIX
                    "action_type":         "",
                    "parameters":          json.dumps({}),
                    "bounding_box":        json.dumps(elem.get("bbox", [])),
                    "other_info":          json.dumps({
                        "type":    elem.get("type", ""),
                        "content": elem.get("content", ""),
                    }),
                })
                db.add_element_to_page(fp_page_id, fp_elem_id)

                if fp_json_id is not None and fp.get("screenshot"):
                    vec_ok = _element2vector(
                        ID=str(fp_json_id),
                        element_id=fp_elem_id,
                        elements_json=json.dumps(fp_elements_data),
                        page_path=fp["screenshot"],
                        vs=vs,
                    )
                    # BUG-4b FIX
                    if vec_ok:
                        db.update_node_property(
                            node_id=fp_elem_id,
                            property_name="visual_embedding_id",
                            property_value=fp_elem_id,
                            node_type="Element",
                        )

            success = False
            if fp.get("screenshot"):
                success = _page2vector(
                    page_id=fp_page_id, page_path=fp["screenshot"],
                    action_type="task_completion", step_no=total_steps,
                    timestamp=str(int(_time.time())), vs=vs,
                )
            if not success:
                print("[json2db] Warning: final page vector storage failed")
                if fp_json_id is not None and fp.get("screenshot"):
                    vec_ok = _element2vector(
                        ID=str(fp_json_id),
                        element_id=fp_elem_id,
                        elements_json=json.dumps(fp_elements_data),
                        page_path=fp["screenshot"],
                        vs=vs,
                    )
                    # BUG-4b FIX
                    if vec_ok:
                        db.update_node_property(
                            node_id=fp_elem_id,
                            property_name="visual_embedding_id",
                            property_value=fp_elem_id,
                            node_type="Element",
                        )

            success = False
            if fp.get("screenshot"):
                success = _page2vector(
                    page_id=fp_page_id, page_path=fp["screenshot"],
                    action_type="task_completion", step_no=total_steps,
                    timestamp=str(int(_time.time())), vs=vs,
                )
            if not success:
                print("[json2db] Warning: final page vector storage failed")

        # ══════════════════════════════════════════════════════════════════════════
        #  SECOND PASS – LEADS_TO edges
        # ══════════════════════════════════════════════════════════════════════════
        for i, cur in enumerate(elements_info):
            if i < len(elements_info) - 1:
                next_page = next(
                    (p for p in pages_info if p["step"] == cur["step"] + 1), None)
            else:
                next_page = (
                    next((p for p in pages_info if p["step"] == "final"), None)
                    or next((p for p in pages_info if p["step"] == cur["step"] + 1), None)
                )
        # ══════════════════════════════════════════════════════════════════════════
        #  SECOND PASS – LEADS_TO edges
        # ══════════════════════════════════════════════════════════════════════════
        for i, cur in enumerate(elements_info):
            if i < len(elements_info) - 1:
                next_page = next(
                    (p for p in pages_info if p["step"] == cur["step"] + 1), None)
            else:
                next_page = (
                    next((p for p in pages_info if p["step"] == "final"), None)
                    or next((p for p in pages_info if p["step"] == cur["step"] + 1), None)
                )

            if next_page:
                # BUG-6 FIX: pass a meaningful confidence score
                db.add_element_leads_to(
                    element_id=cur["element_id"],
                    target_id=next_page["page_id"],
                    action_name=cur["action"],
                    action_params={
                        "execution_result": cur["status"],
                        "timestamp":        cur["timestamp"],
                    },
                    confidence_score=_confidence_from_status(cur["status"]),  # BUG-6 FIX
                )
            if next_page:
                # BUG-6 FIX: pass a meaningful confidence score
                db.add_element_leads_to(
                    element_id=cur["element_id"],
                    target_id=next_page["page_id"],
                    action_name=cur["action"],
                    action_params={
                        "execution_result": cur["status"],
                        "timestamp":        cur["timestamp"],
                    },
                    confidence_score=_confidence_from_status(cur["status"]),  # BUG-6 FIX
                )

    finally:
        db.close()
    finally:
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
    if not elements_data:
        return None
    # tap, long_press, swipe: element_number is required and x/y are resolved
    # from it in explor_human, so clicked_elem will always have valid coords.
    # back: element_number is optional; if provided, clicked_elem carries coords.
    if action_type in ("tap", "long_press", "swipe", "back"):
    # tap, long_press, swipe: element_number is required and x/y are resolved
    # from it in explor_human, so clicked_elem will always have valid coords.
    # back: element_number is optional; if provided, clicked_elem carries coords.
    if action_type in ("tap", "long_press", "swipe", "back"):
        if clicked_elem and elements_path:
            resolved = pos2id(clicked_elem["x"], clicked_elem["y"], str(elements_path))
            resolved = pos2id(clicked_elem["x"], clicked_elem["y"], str(elements_path))
            if resolved:
                return resolved
        # Fallback: element_number embedded in recommended_action string
        elem_id = _parse_element_number(recommended_action)
        if elem_id is not None:
            return next((e for e in elements_data if e.get("ID") == elem_id), None)
        # back with no element_number provided — no interacted element
        return None
        # back with no element_number provided — no interacted element
        return None
    if action_type == "text":
        elem_id = _parse_element_number(recommended_action)
        if elem_id is not None:
            return next((e for e in elements_data if e.get("ID") == elem_id), None)
        return next(
            (e for e in elements_data if e.get("type", "").lower() in ("input", "edittext")),
            None,
        )
    return None


def _parse_element_number(recommended_action: str) -> Optional[int]:
    if not recommended_action:
        return None
    match = re.search(r"['\"]?element_number['\"]?\s*:\s*(\d+)", recommended_action)
    return int(match.group(1)) if match else None


def _element2vector(ID: str, element_id: str, elements_json: str,
                    page_path: str, vs: VectorStore) -> bool:
    """Crop element → ResNet50 → Pinecone 'element' namespace."""
def _element2vector(ID: str, element_id: str, elements_json: str,
                    page_path: str, vs: VectorStore) -> bool:
    """Crop element → ResNet50 → Pinecone 'element' namespace."""
    try:
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
            id=element_id,          # Pinecone id == Neo4j element_id UUID
            id=element_id,          # Pinecone id == Neo4j element_id UUID
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


def _page2vector(page_id: str, page_path: str, action_type: str,
                 step_no: Optional[int], timestamp: str, vs: VectorStore) -> bool:
    """Full screenshot → ResNet50 → Pinecone 'page' namespace."""
def _page2vector(page_id: str, page_path: str, action_type: str,
                 step_no: Optional[int], timestamp: str, vs: VectorStore) -> bool:
    """Full screenshot → ResNet50 → Pinecone 'page' namespace."""
    try:
        if not page_path:
            raise ValueError("page_path is empty")
        path_obj = _fetch_to_local(page_path, ".png")
        if path_obj is None:
            raise FileNotFoundError(f"Screenshot not found: {page_path}")
            raise FileNotFoundError(f"Screenshot not found: {page_path}")
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