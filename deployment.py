"""
deployment.py  (NVIDIA NIM edition)
-------------------------------------
Replaces Firebase round-trips with direct NVIDIA NIM API calls
(nvidia/llama-3.1-nemotron-nano-vl-8b-v1) via the OpenAI-compatible client.

All public APIs and the LangGraph workflow structure are preserved.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

# pyrefly: ignore [missing-import]
from langchain_core.messages import HumanMessage, SystemMessage
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, END
# pyrefly: ignore [missing-import]
from langgraph.prebuilt import create_react_agent

import config
from data.State import DeploymentState, ElementMatch
from data.graph_db import Neo4jDatabase
from data.vector_db import VectorStore
from tool.img_tool import *
from tool.adb_tools import *
from OmniParser.client import run as omniparser_run
from nvidia_llm_bridge import NvidiaBridge

# ── LangSmith tracing ────────────────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true" if config.LANGCHAIN_TRACING_V2 else "false"
os.environ["LANGCHAIN_ENDPOINT"]   = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"]    = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"]    = "DeploymentExecution"

# ── NVIDIA NIM bridge (direct — no Firebase worker needed) ────────────────────
bridge = NvidiaBridge(
    max_tokens_text=4096,   # task matching responses include full action JSON (900+ tokens)
    max_tokens_json=4096,
    max_tokens_vision=2048,
)

# ── Database / vector store ──────────────────────────────────────────────────
URI  = config.Neo4j_URI
AUTH = config.Neo4j_AUTH
db   = Neo4jDatabase(URI, AUTH)
vector_db = VectorStore(api_key=config.PINECONE_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
#  Tiny sync wrapper so callers that aren't already in an async context
#  can invoke the bridge without restructuring.
# ─────────────────────────────────────────────────────────────────────────────

def _sync_call_text(system_prompt: str, user_prompt: str, timeout: float = 300.0) -> str:
    """Call NVIDIA NIM synchronously from any thread context."""
    return asyncio.run(
        bridge.call_text(system_prompt=system_prompt, user_prompt=user_prompt)
    )


def _sync_call_json(
    system_prompt: str,
    user_prompt: str,
    images_b64: Optional[List[str]] = None,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    return asyncio.run(
        bridge.call_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images_b64=images_b64,
        )
    )


def _sync_call_vision(
    system_prompt: str,
    user_prompt: str,
    images_b64: List[str],
    timeout: float = 300.0,
) -> str:
    return asyncio.run(
        bridge.call_vision(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images_b64=images_b64,
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _img_to_b64(path: str) -> Optional[str]:
    if path and os.path.exists(path):
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")
    return None


def create_execution_state(device: str) -> Dict[str, Any]:
    from data.State import create_deployment_state
    return create_deployment_state(task="", device=device)


# ─────────────────────────────────────────────────────────────────────────────
#  Task → high-level action matching  (semantic — LLM judges intent, not keywords)
# ─────────────────────────────────────────────────────────────────────────────

_MATCH_SYSTEM = """
You are an AI assistant that matches a user’s natural-language task to the
best-fitting stored high-level action using SEMANTIC understanding.

Rules:
- Do NOT do substring/keyword matching. Understand the INTENT of the task.
- A match is valid when the user’s intent is substantially the same as the
  action’s name or description (confidence ≥ 0.6).
- Copy the full action object VERBATIM — do not truncate element_sequence.

Reply with a JSON object ONLY — no prose, no markdown fences:
  If matched:    {"matched": true,  "confidence": 0.85, "action": <full action object>}
  If not matched: {"matched": false, "reason": "<brief explanation>"}
"""


def get_close_high_level_actions(task: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Return up to top_k high-level actions that are most semantically similar to
    the user task.  Used to populate the no-match popup in the UI.
    """
    all_actions = db.get_all_high_level_actions()
    if not all_actions:
        return []

    _RANK_SYSTEM = (
        "You are an assistant that ranks stored actions by their semantic similarity "
        "to a user task. Return ONLY a JSON array of action_ids ordered from most to "
        "least relevant (most relevant first). No prose, no markdown fences."
    )
    actions_summary = json.dumps(
        [{"action_id": a.get("action_id"), "name": a.get("name"), "description": a.get("description", "")}
         for a in all_actions],
        ensure_ascii=False, indent=2,
    )
    user_prompt = (
        f"User task: {task}\n\n"
        f"Stored actions:\n{actions_summary}\n\n"
        f"Return a JSON array of the {top_k} most relevant action_ids, ordered best-first."
    )
    try:
        ranked_ids = asyncio.run(bridge.call_json(
            system_prompt=_RANK_SYSTEM, user_prompt=user_prompt
        ))
        # bridge.call_json may return a dict or a list
        if isinstance(ranked_ids, list):
            id_order = ranked_ids
        elif isinstance(ranked_ids, dict):
            # try common wrapper keys
            id_order = ranked_ids.get("action_ids", ranked_ids.get("ids", []))
        else:
            id_order = []

        action_map = {a.get("action_id"): a for a in all_actions}
        result = [action_map[aid] for aid in id_order if aid in action_map]
        # pad with remaining actions if LLM returned fewer than top_k
        for a in all_actions:
            if len(result) >= top_k:
                break
            if a not in result:
                result.append(a)
        return result[:top_k]
    except Exception as exc:
        print(f"[get_close_high_level_actions] Error: {exc}")
        return all_actions[:top_k]


def match_task_to_action(
    state: Dict[str, Any], task: str
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Semantic matching: LLM understands the user’s INTENT and picks the best
    action regardless of exact wording.  Does NOT fall back on keyword search.
    """
    _log = state.get("log_callback") or print
    _log(f"Matching task (semantic): {task}")

    high_level_actions = db.get_all_high_level_actions()
    if not high_level_actions:
        _log("❌ No high-level action nodes found in Neo4j")
        return False, None

    _log(f"Found {len(high_level_actions)} high-level action node(s)")
    for i, a in enumerate(high_level_actions):
        seq_len = len(a.get("element_sequence") or [])
        _log(f"  [MATCH] action[{i}]: id={a.get('action_id')}  name={a.get('name')}  steps={seq_len}")

    actions_json = json.dumps(high_level_actions, ensure_ascii=False, indent=2)
    user_prompt = (
        f"User task: {task}\n\n"
        f"Available high-level actions:\n{actions_json}\n\n"
        "Return JSON only. Copy the full action object verbatim — do not truncate element_sequence."
    )

    try:
        result = asyncio.run(
            bridge.call_json(system_prompt=_MATCH_SYSTEM, user_prompt=user_prompt)
        )

        if not isinstance(result, dict):
            _log(f"  [MATCH] ❌ Unexpected LLM response type: {type(result)}")
            return False, None

        if not result.get("matched"):
            _log(f"  [MATCH] ❌ No semantic match: {result.get('reason', 'no reason given')}")
            return False, None

        matched_action = result.get("action")
        if not isinstance(matched_action, dict):
            _log(f"  [MATCH] ❌ 'action' field missing or not a dict")
            return False, None

        if not matched_action.get("action_id") and not matched_action.get("name"):
            _log(f"  [MATCH] ❌ Action missing both action_id and name — discarding")
            return False, None

        seq_len = len(matched_action.get("element_sequence") or [])
        confidence = result.get("confidence", "?")
        _log(f"  [MATCH] ✓ Matched: '{matched_action.get('name')}' "
             f"(ID: {matched_action.get('action_id')})  confidence={confidence}  steps={seq_len}")
        return True, matched_action

    except Exception as e:
        _log(f"❌ Error during semantic task matching: {e}")
        import traceback; traceback.print_exc()
        return False, None


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first valid JSON object from a string that may contain prose.
    Scans for '{' and tries progressively larger substrings.
    Returns the parsed dict, or None if no valid JSON object is found.
    """
    start = text.find("{")
    if start == -1:
        return None
    candidate = text[start:]
    depth = 0
    end_idx = -1
    for i, ch in enumerate(candidate):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx == -1:
        return None
    try:
        return json.loads(candidate[:end_idx + 1])
    except json.JSONDecodeError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Screen capture / OmniParser  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def capture_and_parse_screen(state: DeploymentState) -> DeploymentState:
    try:
        screenshot_path = take_screenshot.invoke({
            "device":    state["device"],
            "app_name":  "deployment",
            "step":      state["current_step"],
        })
        if not screenshot_path or not os.path.exists(screenshot_path):
            print("❌ Screenshot failed")
            return state

        with open(screenshot_path, "rb") as fh:
            image_b64 = base64.b64encode(fh.read()).decode("utf-8")

        json_path = omniparser_run(image_b64)
        if not json_path:
            print("❌ Screen element parsing failed: OmniParser returned no result")
            return state

        state["current_page"]["screenshot"]   = screenshot_path
        state["current_page"]["elements_json"] = json_path

        with open(json_path, "r", encoding="utf-8") as f:
            state["current_page"]["elements_data"] = json.load(f)

        print(f"✓ Parsed screen — {len(state['current_page']['elements_data'])} UI elements")
        return state

    except Exception as e:
        print(f"❌ Error capturing and parsing screen: {e}")
        return state


def _log_both(state: Dict[str, Any], msg: str):
    print(msg)
    _log = state.get("log_callback")
    if _log:
        _log(msg)


_VERIFY_MATCH_SYSTEM = (
    "You are a UI matching assistant. Compare a stored template element description and type "
    "against a candidate live screen element's content and type. "
    "Determine if they represent the same functional UI element. "
    "Return a JSON object containing:\n"
    "  \"similarity\": <float, between 0.0 and 1.0>,\n"
    "  \"reason\": \"<brief reason>\""
)

_SEMANTIC_MATCH_SYSTEM = (
    "You are a UI matching assistant. "
    "Given a target element description, select the single best matching element "
    "from the list of live screen elements based on semantic similarity of content. "
    "Return ONLY a JSON object:\n"
    "  {\"screen_element_id\": <int, 0-based index of the matching element in the list>,\n"
    "   \"reason\": \"<brief reason>\"}\n"
    "If no reasonable match exists, set screen_element_id to -1."
)


def _bbox_center_dist(b1: List[float], b2: List[float]) -> float:
    c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
    c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
    return float(((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5)


def match_element_via_pinecone(
    element_id: str,
    step_info: Dict[str, Any],
    state: DeploymentState,
    threshold: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Element-matching strategy:
      1. Fetch the stored Pinecone vector metadata of the corresponding element ID.
      2. Find the candidate live element with the closest spatial bounding box (spatial matching).
      3. Use LLM to verify if stored vector content and live candidate content matches (similarity > 0.7).
      4. If verified, return this candidate.
      5. Otherwise, check which live element has the closest semantic matching (excluding bounding boxes from metadata).
    """
    _log = lambda msg: _log_both(state, msg)
    screen_elements = state["current_page"]["elements_data"]
    screenshot_path = state["current_page"]["screenshot"]

    if not screen_elements or not screenshot_path:
        _log("  [ACTION MATCHING] ⚠️  No screen elements or screenshot available")
        return []

    # ── 1. Fetch stored details from Neo4j & Pinecone ───────────────────────────
    stored_content = ""
    stored_type = ""
    stored_bbox: Optional[List[float]] = None

    # Retrieve from Neo4j first (we prioritize Neo4j description)
    neo4j_element = db.get_element_by_id(element_id) or {}
    neo4j_desc = neo4j_element.get("description") or ""
    neo4j_reasoning = neo4j_element.get("reasoning") or ""
    stored_type = neo4j_element.get("element_type") or ""
    bbox_raw = neo4j_element.get("bounding_box")
    if isinstance(bbox_raw, str):
        try:
            stored_bbox = json.loads(bbox_raw)
        except Exception:
            stored_bbox = None
    elif isinstance(bbox_raw, list):
        stored_bbox = bbox_raw

    stored_content = neo4j_desc

    # Fallback/merge with Pinecone if needed
    _log(f"[ACTION MATCHING] Fetching stored metadata for element {element_id[:8]}...")
    try:
        fetch_result = vector_db.index.fetch(ids=[element_id], namespace="element")
        vec_data = (fetch_result.get("vectors") or {}).get(element_id)
        if vec_data:
            stored_meta = vec_data.get("metadata", {})
            if not stored_content:
                stored_content = stored_meta.get("content", "")
            if not stored_type:
                stored_type = stored_meta.get("type", "")
            
            # Parse stored bbox if not already resolved from Neo4j
            if not stored_bbox:
                bbox_raw_pc = stored_meta.get("bbox")
                if isinstance(bbox_raw_pc, str):
                    try:
                        stored_bbox = json.loads(bbox_raw_pc)
                    except Exception:
                        stored_bbox = None
                elif isinstance(bbox_raw_pc, list):
                    stored_bbox = bbox_raw_pc
        else:
            _log(f"  [ACTION MATCHING] ⚠️ element_id {element_id[:8]} not found in Pinecone.")
    except Exception as exc:
        _log(f"  [ACTION MATCHING] Pinecone fetch error: {exc}")

    if not stored_content:
        # Final fallback: step info parameters
        params = step_info.get("action_params", {})
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                params = {}
        stored_content = params.get("element") or ""

    _log(f"[ACTION MATCHING] Stored element: content='{stored_content}' type='{stored_type}' bbox={stored_bbox}")

    # ── 2. Spatial match candidate search ───────────────────────────────
    corresponding_live_element = None
    spatial_idx = -1
    if stored_bbox and len(stored_bbox) == 4:
        min_dist = float("inf")
        for idx, el in enumerate(screen_elements):
            el_bbox = el.get("bbox")
            if el_bbox and len(el_bbox) == 4:
                dist = _bbox_center_dist(stored_bbox, el_bbox)
                if dist < min_dist:
                    min_dist = dist
                    corresponding_live_element = el
                    spatial_idx = idx

    # ── 3. Match verification using LLM ──────────────────────────────────
    if corresponding_live_element:
        candidate_content = corresponding_live_element.get("content", "")
        candidate_type = corresponding_live_element.get("type", "")
        _log(f"[ACTION MATCHING] Found spatial candidate at index {spatial_idx}: content='{candidate_content}' type='{candidate_type}' (dist: {min_dist:.4f})")
        
        # If spatial match is extremely close (exact spatial match slot), accept it directly and bypass similarity check
        if min_dist < 0.03:
            _log(f"[ACTION MATCHING] ✓ Spatial match verified dynamically via distance ({min_dist:.4f} < 0.03)")
            return [{
                "element_id":        element_id,
                "match_score":       1.0 - min_dist,
                "screen_element_id": spatial_idx,
                "action_type":       step_info.get("atomic_action", "tap"),
                "parameters":        step_info.get("action_params", {}),
            }]

        user_prompt = (
            f"Stored Template Element:\n"
            f"  Description: {stored_content}\n"
            f"  Type: {stored_type}\n\n"
            f"Candidate Live Screen Element:\n"
            f"  Content: {candidate_content}\n"
            f"  Type: {candidate_type}\n\n"
            f"Do they represent the same UI element? Rate the similarity from 0.0 to 1.0. "
            f"Return JSON only."
        )
        try:
            res = _sync_call_json(_VERIFY_MATCH_SYSTEM, user_prompt, timeout=120)
            similarity = float(res.get("similarity", 0.0))
            reason = res.get("reason", "")
            _log(f"[ACTION MATCHING] LLM verify similarity score (description): {similarity:.2f} (threshold: {threshold}) — Reason: {reason}")
            
            if similarity > threshold:
                _log(f"[ACTION MATCHING] ✓ Spatial match verified via description (score {similarity:.2f} > {threshold})")
                return [{
                    "element_id":        element_id,
                    "match_score":       similarity,
                    "screen_element_id": spatial_idx,
                    "action_type":       step_info.get("atomic_action", "tap"),
                    "parameters":        step_info.get("action_params", {}),
                }]
            else:
                _log(f"[ACTION MATCHING] ⚠️ Spatial verification failed with description (score {similarity:.2f} <= {threshold}). Trying fallback with reasoning...")
                if neo4j_reasoning:
                    user_prompt_reasoning = (
                        f"Stored Template Element Reasoning Details:\n"
                        f"  Reasoning: {neo4j_reasoning}\n"
                        f"  Type: {stored_type}\n\n"
                        f"Candidate Live Screen Element:\n"
                        f"  Content: {candidate_content}\n"
                        f"  Type: {candidate_type}\n\n"
                        f"Do they represent the same UI element? Rate the similarity from 0.0 to 1.0. "
                        f"Return JSON only."
                    )
                    res_reasoning = _sync_call_json(_VERIFY_MATCH_SYSTEM, user_prompt_reasoning, timeout=120)
                    similarity_reasoning = float(res_reasoning.get("similarity", 0.0))
                    reason_reasoning = res_reasoning.get("reason", "")
                    _log(f"[ACTION MATCHING] LLM verify reasoning similarity score: {similarity_reasoning:.2f} (threshold: {threshold}) — Reason: {reason_reasoning}")
                    
                    if similarity_reasoning > threshold:
                        _log(f"[ACTION MATCHING] ✓ Spatial match verified via reasoning details (score {similarity_reasoning:.2f} > {threshold})")
                        return [{
                            "element_id":        element_id,
                            "match_score":       similarity_reasoning,
                            "screen_element_id": spatial_idx,
                            "action_type":       step_info.get("atomic_action", "tap"),
                            "parameters":        step_info.get("action_params", {}),
                        }]
                else:
                    _log("[ACTION MATCHING] No reasoning details available in Neo4j element for fallback verification.")
        except Exception as exc:
            _log(f"  [ACTION MATCHING] LLM verification error: {exc}")

    # ── 4. Fallback to semantic matching across all live elements ─────
    _log("[ACTION MATCHING] Spatial match verification failed or scored <= threshold. Falling back to semantic matching...")
    return llm_bbox_fallback(element_id, step_info, state, stored_content, stored_type)


def llm_bbox_fallback(
    element_id: str,
    step_info: Dict[str, Any],
    state: DeploymentState,
    stored_content: str,
    stored_type: str,
) -> List[Dict[str, Any]]:
    """
    Ask the LLM to choose the best semantic match from all live elements
    without including any bounding boxes in the metadata.
    """
    _log = lambda msg: _log_both(state, msg)
    screen_elements = state["current_page"]["elements_data"]

    # Format list of live elements without bounding boxes (User Comment 1: "Remove bounding box from metadata")
    live_elements_list = ""
    for idx, el in enumerate(screen_elements):
        live_elements_list += f"{idx}: type={el.get('type','?')}  content='{el.get('content','')}'\n"

    user_prompt = (
        f"Target Element Description: {stored_content}\n"
        f"Target Element Type: {stored_type}\n\n"
        f"Live Screen Elements:\n{live_elements_list}\n"
        f"Choose the element that has the closest semantic match to the target description. "
        f"Return JSON only."
    )

    try:
        result = _sync_call_json(_SEMANTIC_MATCH_SYSTEM, user_prompt, timeout=120)
        sid = int(result.get("screen_element_id", -1))
        reason = result.get("reason", "")
        if sid >= 0 and sid < len(screen_elements):
            _log(f"[ACTION MATCHING] ✓ LLM semantic fallback picked screen element {sid}. Reason: {reason}")
            return [{
                "element_id":        element_id,
                "match_score":       0.75,
                "screen_element_id": sid,
                "action_type":       step_info.get("atomic_action", "tap"),
                "parameters":        step_info.get("action_params", {}),
            }]
        else:
            _log(f"[ACTION MATCHING] ❌ LLM could not identify a semantic match (screen_element_id={sid})")
            return []
    except Exception as exc:
        _log(f"  [ACTION MATCHING] LLM semantic matching error: {exc}")
        return []


def _parse_action_result(result: Any) -> bool:
    """
    Normalise every possible return value from screen_action.invoke():
      - dict  → check result["status"] == "success"
      - str   → try JSON parse, then check; fall back to truthy non-empty string
      - bool  → use directly
      - None  → False
      - any other truthy value → True (tool returned something non-error)
    """
    print(f"  [DIAG-ADB] screen_action raw result → type={type(result).__name__}  value={repr(result)[:300]}")
    if result is None:
        print("  [DIAG-ADB] → None → False")
        return False
    if isinstance(result, bool):
        print(f"  [DIAG-ADB] → bool → {result}")
        return result
    if isinstance(result, dict):
        status = result.get("status", "")
        if status:
            ok = str(status).lower() in ("success", "ok", "done", "true", "1")
            print(f"  [DIAG-ADB] → dict with status='{status}' → {ok}")
            return ok
        ok = "error" not in result
        print(f"  [DIAG-ADB] → dict without status key, 'error' in keys={not ok} → {ok}")
        return ok
    if isinstance(result, str):
        stripped = result.strip()
        if not stripped:
            print("  [DIAG-ADB] → empty string → False")
            return False
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                status = parsed.get("status", "")
                if status:
                    ok = str(status).lower() in ("success", "ok", "done", "true", "1")
                    print(f"  [DIAG-ADB] → JSON dict status='{status}' → {ok}")
                    return ok
                ok = "error" not in parsed
                print(f"  [DIAG-ADB] → JSON dict no status, 'error' absent={ok} → {ok}")
                return ok
            ok = bool(parsed)
            print(f"  [DIAG-ADB] → JSON scalar={parsed} → {ok}")
            return ok
        except (json.JSONDecodeError, ValueError):
            lower = stripped.lower()
            ok = not any(kw in lower for kw in ("error", "fail", "false", "exception"))
            print(f"  [DIAG-ADB] → plain string, failure keywords absent={ok} → {ok}")
            return ok
    ok = bool(result)
    print(f"  [DIAG-ADB] → other type, truthy={ok} → {ok}")
    return ok


def execute_element_action(state: DeploymentState, element_match: Dict[str, Any]) -> bool:
    try:
        if not element_match:
            return False

        action_type       = element_match.get("action_type", "tap")
        parameters        = element_match.get("parameters", {})
        screen_element_id = element_match.get("screen_element_id", -1)

        if screen_element_id < 0 or screen_element_id >= len(state["current_page"]["elements_data"]):
            print(f"❌ Invalid screen element ID: {screen_element_id}")
            return False

        element     = state["current_page"]["elements_data"][screen_element_id]
        bbox        = element.get("bbox", [0, 0, 0, 0])
        device_size = get_device_size.invoke(state["device"])
        if isinstance(device_size, str):
            device_size = {"width": 1080, "height": 2400}

        # Calculate coordinates conditionally: scale if relative (<= 1.0), use directly otherwise
        if bbox and len(bbox) == 4 and all(val <= 1.0 for val in bbox):
            center_x = int((bbox[0] + bbox[2]) / 2 * device_size["width"])
            center_y = int((bbox[1] + bbox[3]) / 2 * device_size["height"])
        else:
            center_x = int((bbox[0] + bbox[2]) / 2) if bbox and len(bbox) == 4 else 0
            center_y = int((bbox[1] + bbox[3]) / 2) if bbox and len(bbox) == 4 else 0

        action_params = {"device": state["device"], "action": action_type, "x": center_x, "y": center_y}
        if action_type == "text":
            action_params["input_str"] = parameters.get("text", "")
        elif action_type == "long_press":
            action_params["duration"] = parameters.get("duration", 1000)
        elif action_type in ("swipe", "swipe_short", "swipe_long"):
            action_params["direction"] = parameters.get("direction", "up")
            action_params["dist"]      = parameters.get("distance", "medium")

        print(f"Executing action: {action_type} at ({center_x}, {center_y})")
        result = screen_action.invoke(action_params)

        success = _parse_action_result(result)
        if success:
            print("✓ Action executed successfully")
        else:
            print(f"❌ Action failed — raw result: {result!r}")
        return success

    except Exception as e:
        print(f"❌ Error executing element action: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  React fallback  (uses Qwen/Nemotron for decision, ADB tool for execution)
# ─────────────────────────────────────────────────────────────────────────────

_REACT_SYSTEM = (
    "You are an intelligent smartphone operation assistant. "
    "Observe the current screen and perform one atomic operation "
    "(tap / type text / swipe / long press / back) to progress toward the user's goal. "
    "Reply with a JSON object:\n"
    "  {\"action\": \"<type>\", \"element_id\": <int or str of target element>, "
    "\"input_str\": \"<text if action=text>\", "
    "\"direction\": \"<up/down/left/right if action=swipe>\", "
    "\"duration\": <ms if action=long_press>}\n"
    "For back action omit element_id. Return JSON only."
)


def fallback_to_react(state: DeploymentState) -> DeploymentState:
    print("🔄 Falling back to React mode execution...")
    task = state["task"]

    state = capture_and_parse_screen(state)
    if not state["current_page"]["screenshot"]:
        state["execution_status"] = "error"
        print("Unable to capture or parse screen")
        return state

    screenshot_path    = state["current_page"]["screenshot"]
    elements_json_path = state["current_page"]["elements_json"]
    device             = state["device"]

    # ── Device size ───────────────────────────────────────────────────────────
    raw_size   = get_device_size.invoke(device)
    print(f"  [DIAG-REACT] get_device_size raw: {raw_size!r}")
    if isinstance(raw_size, dict):
        device_w = int(raw_size.get("width", raw_size.get("w", 1080)))
        device_h = int(raw_size.get("height", raw_size.get("h", 2400)))
    elif isinstance(raw_size, str):
        try:
            sz = json.loads(raw_size)
            device_w = int(sz.get("width", sz.get("w", 1080)))
            device_h = int(sz.get("height", sz.get("h", 2400)))
        except Exception:
            device_w, device_h = 1080, 2400
    else:
        device_w, device_h = 1080, 2400
    print(f"  [DIAG-REACT] device size: {device_w}x{device_h}")

    with open(elements_json_path, "r", encoding="utf-8") as f:
        elements_data = json.load(f)

    img_b64 = _img_to_b64(screenshot_path)
    images  = [img_b64] if img_b64 else []

    # ── Build element index for coordinate lookup ─────────────────────────────
    # Tell the LLM to refer to elements by their ID so we can resolve coordinates
    # from the parsed bbox — this avoids the model guessing pixel coordinates.
    element_index = {}
    elements_for_prompt = []
    for el in elements_data:
        eid  = el.get("ID", el.get("id", "?"))
        bbox = el.get("bbox", [])
        element_index[str(eid)] = bbox
        elements_for_prompt.append({
            "id":      eid,
            "type":    el.get("type", ""),
            "content": el.get("content", ""),
        })

    user_prompt = (
        f"Device: {device}  Size: {device_w}x{device_h} pixels\n"
        f"Task: {task}\n\n"
        f"Current screen elements:\n"
        f"{json.dumps(elements_for_prompt, ensure_ascii=False, indent=2)}\n\n"
        "Reply with JSON specifying the next single action.\n"
        "IMPORTANT: You must specify the 'element_id' from the elements list above for the target element.\n"
        "Return JSON only."
    )

    # ── Swipe up after any previous tap (escape immediately on 1st retry) ────
    _last_taps = [h for h in state.get("history", [])[-1:] if h.get("action") == "tap"]
    if _last_taps:
        print(f"  [DIAG-REACT] ⚠️  Previous action was a tap — injecting swipe-up to refresh screen before next LLM call")
        swipe_params = {"device": device, "action": "swipe", "x": 540, "y": 1200, "direction": "up", "dist": "medium"}
        screen_action.invoke(swipe_params)
        time.sleep(1.0)

    try:
        result_json = _sync_call_json(
            system_prompt=_REACT_SYSTEM,
            user_prompt=user_prompt,
            images_b64=images,
            timeout=240,
        )
        print(f"  [DIAG-REACT] LLM action JSON: {result_json}")

        action_type = result_json.get("action", "tap")
        action_params: Dict[str, Any] = {"device": device, "action": action_type}

        if action_type != "back":
            selected_id = str(result_json.get("element_id", ""))
            bbox = element_index.get(selected_id)
            if not bbox or len(bbox) < 4:
                # Fallback to coordinates if model still output x and y
                raw_x = result_json.get("x")
                raw_y = result_json.get("y")
                if raw_x is not None and raw_y is not None:
                    if isinstance(raw_x, float) and 0.0 < raw_x <= 1.0:
                        raw_x = int(raw_x * device_w)
                    if isinstance(raw_y, float) and 0.0 < raw_y <= 1.0:
                        raw_y = int(raw_y * device_h)
                    action_params["x"] = int(raw_x)
                    action_params["y"] = int(raw_y)
                else:
                    print(f"  [DIAG-REACT] ❌ Selected element_id '{selected_id}' not found in screen elements")
                    state["execution_status"] = "error"
                    return state
            else:
                # Calculate coordinates conditionally: scale if relative (<= 1.0), use directly otherwise
                if bbox and len(bbox) == 4 and all(val <= 1.0 for val in bbox):
                    center_x = int((bbox[0] + bbox[2]) / 2 * device_w)
                    center_y = int((bbox[1] + bbox[3]) / 2 * device_h)
                else:
                    center_x = int((bbox[0] + bbox[2]) / 2) if bbox and len(bbox) == 4 else 0
                    center_y = int((bbox[1] + bbox[3]) / 2) if bbox and len(bbox) == 4 else 0
                action_params["x"] = center_x
                action_params["y"] = center_y

        if action_type == "text":
            action_params["input_str"] = result_json.get("input_str", "")
        elif action_type == "long_press":
            action_params["duration"] = int(result_json.get("duration", 1000))
        elif action_type in ("swipe", "swipe_short", "swipe_long"):
            action_params["direction"] = result_json.get("direction", "up")
            action_params["dist"]      = result_json.get("dist", "medium")

        print(f"  [DIAG-REACT] Invoking screen_action with: {action_params}")
        action_result = screen_action.invoke(action_params)
        action_ok = _parse_action_result(action_result)
        state["current_step"] += 1
        state["history"].append({
            "step":       state["current_step"],
            "screenshot": screenshot_path,
            "action":     action_type,
            "params":     action_params,
            "status":     "success" if action_ok else "error",
        })
        state["execution_status"] = "success" if action_ok else "error"
        print(f"{'✓' if action_ok else '❌'} React mode: executed {action_type} — result: {action_result!r}")

    except Exception as e:
        print(f"❌ React mode error: {e}")
        import traceback
        traceback.print_exc()
        state["history"].append({
            "step":   state["current_step"],
            "action": "react_mode",
            "status": "error",
            "error":  str(e),
        })
        state["execution_status"] = "error"

    return state


def execute_task(
    state: DeploymentState, task: str, device: str, neo4j_db: Neo4jDatabase = None
) -> Dict[str, Any]:
    """Thin wrapper kept for backward-compat. Real execution is in run_task()."""
    return run_task(task=task, device=device)


# ─────────────────────────────────────────────────────────────────────────────
#  Task completion check  (vision + text via Qwen)
# ─────────────────────────────────────────────────────────────────────────────

_CRITERIA_SYSTEM = (
    "You are an assistant that generates clear, checkable task-completion criteria. "
    "Describe what must appear on screen for the task to be considered done."
)

_JUDGE_SYSTEM = (
    "You are a page assessment assistant. "
    "Given the completion criteria and recent screenshots, decide if the task is complete. "
    "Reply with only 'yes' or 'no'."
)


def check_task_completion(state: DeploymentState) -> DeploymentState:
    _log = state.get("log_callback") or print

    if state.get("execution_status") == "no_match":
        state["completed"] = True
        return state

    # Fetch matched high-level action and the target page description from Neo4j
    matched_action = state.get("current_action")
    last_page_id = None
    last_page_description = None

    if matched_action:
        element_sequence = matched_action.get("element_sequence", [])
        if isinstance(element_sequence, str):
            try:
                element_sequence = json.loads(element_sequence)
            except Exception:
                element_sequence = []
        if element_sequence:
            last_step = element_sequence[-1]
            last_element_id = last_step.get("element_id")
            if last_element_id:
                try:
                    query = """
                    MATCH (e:Element {element_id: $eid})-[:LEADS_TO]->(p:Page)
                    RETURN p.page_id as page_id, p.description as description
                    """
                    with db.driver.session(database=db.database) as session:
                        res = session.run(query, eid=last_element_id)
                        record = res.single()
                        if record:
                            last_page_id = record["page_id"]
                            last_page_description = record["description"]
                            _log(f"🔍 Found task completion page in Neo4j. Page ID: {last_page_id}, Description: '{last_page_description}'")
                except Exception as exc:
                    _log(f"⚠️ Error querying Neo4j task completion page: {exc}")

    # If already marked completed and we have no Neo4j page description to verify, return early.
    # Otherwise, proceed to match the live page semantics against last_page_description.
    if state.get("completed") and not last_page_description:
        _log("✓ Task already completed successfully by action sequence execution.")
        return state

    history = state.get("history") or []
    has_screenshot = bool(state.get("current_page", {}).get("screenshot"))

    if not history and not has_screenshot:
        _log("🔍 check_task_completion: skipping — no actions taken yet")
        return state

    _log(f"🔍 Evaluating if task is completed... (history_len={len(history)}, step={state.get('current_step')})")
    task = state["task"]

    if last_page_description:
        completion_criteria = f"The current screen is a final page which should match the semantic description: {last_page_description}"
    else:
        try:
            completion_criteria = _sync_call_text(
                system_prompt=_CRITERIA_SYSTEM,
                user_prompt=f"The user's task is: {task}\nDescribe clear, checkable completion criteria.",
                timeout=120,
            )
        except Exception as e:
            _log(f"⚠️ Could not generate criteria: {e}")
            return state

    recent_screenshots: List[str] = [
        step["screenshot"] for step in state["history"][-3:] if step.get("screenshot")
    ]
    if not recent_screenshots and state["current_page"]["screenshot"]:
        recent_screenshots = [state["current_page"]["screenshot"]]
    if not recent_screenshots:
        _log("⚠️ No screenshots available")
        return state

    images_b64: List[str] = []
    for p in recent_screenshots:
        b64 = _img_to_b64(p)
        if b64:
            images_b64.append(b64)

    user_prompt = (
        f"Completion criteria: {completion_criteria}\n\n"
        "Analyse the provided screenshots. "
        "If screenshots are identical the task may be stuck — answer 'yes' to end.\n"
        "Is the task complete? Reply yes or no."
    )

    try:
        answer = _sync_call_vision(
            system_prompt=_JUDGE_SYSTEM,
            user_prompt=user_prompt,
            images_b64=images_b64,
            timeout=180,
        ).strip().lower()
    except Exception as e:
        _log(f"⚠️ Completion check error: {e}")
        return state

    _log(f"  [DIAG-COMPLETION] Full judgement answer: {answer!r}")

    def _is_affirmative(text: str) -> bool:
        negative_phrases = [
            "not complete", "not yet", "not done", "not finished",
            "incomplete", "no,", "no.", "no\n", "task is not", "hasn't been",
            "have not", "has not", "cannot confirm", "not confirmed",
            "not set", "not shown", "not visible", "does not show",
        ]
        for phrase in negative_phrases:
            if phrase in text:
                return False
        positive_phrases = ["yes,", "yes.", "yes\n", "task is complete",
                            "task has been completed", "alarm has been set",
                            "alarm is set", "task complete", "completed successfully"]
        for phrase in positive_phrases:
            if phrase in text:
                return True
        if text.startswith("yes") and len(text) < 15:
            return True
        return False

    is_complete = _is_affirmative(answer)
    if is_complete:
        state["completed"]        = True
        state["execution_status"] = "completed"
        _log(f"✓ Task completed: {answer[:100]}")
    else:
        state["completed"] = False
        _log(f"⚠️ Task not yet complete: {answer[:100]}")
        if matched_action:
            _log("⚠️ Verification failed for high-level action sequence. Routing to React fallback mode.")
            state["should_fallback"] = True

    state["history"].append({
        "step":               state["current_step"],
        "action":             "task_completion_check",
        "completion_criteria": completion_criteria,
        "judgement":          answer,
        "status":             "success",
        "completed":          state["completed"],
    })
    return state


# ─────────────────────────────────────────────────────────────────────────────
#  LangGraph node wrappers  (unchanged structure)
# ─────────────────────────────────────────────────────────────────────────────

def capture_screen_node(state: DeploymentState) -> DeploymentState:
    _log = state.get("log_callback") or print
    _log("📸 Capturing and parsing current screen...")
    state_dict = dict(state)
    updated    = capture_and_parse_screen(state_dict)
    for k, v in updated.items():
        if k in state:
            state[k] = v
    if not state["current_page"]["screenshot"]:
        state["should_fallback"] = True
        _log("❌ Unable to capture screen, marking for fallback")
    else:
        _log(f"✓ Screen captured — {len(state['current_page'].get('elements_data') or [])} elements")
    return state


def match_elements_node(state: DeploymentState) -> DeploymentState:
    """Semantically match the user task to a stored Neo4j high-level action."""
    _log = state.get("log_callback") or print

    if state.get("force_fallback"):
        _log("⚡ Force fallback requested — routing directly to React fallback mode")
        state["should_fallback"] = True
        state["close_actions"]   = []
        return state

    _log(f"🔍 Matching task to high-level action: '{state['task']}'")

    state_dict = dict(state)
    is_matched, matched_action = match_task_to_action(state_dict, state["task"])

    if is_matched and matched_action:
        element_sequence = matched_action.get("element_sequence", [])
        if isinstance(element_sequence, str):
            try:
                element_sequence = json.loads(element_sequence)
            except Exception:
                element_sequence = []

        if not element_sequence:
            _log("  [MATCH] ⚠️  element_sequence is empty — no steps to execute")
            state["should_fallback"] = True
            state["close_actions"]   = get_close_high_level_actions(state["task"])
            return state

        state["current_action"]  = matched_action
        state["current_step"]    = 0
        state["total_steps"]     = len(element_sequence)
        state["should_fallback"] = False
        _log(f"  [MATCH] ✓ Matched '{matched_action.get('name')}' — {len(element_sequence)} step(s)")
    else:
        _log("  [MATCH] ❌ No high-level action found — early termination for popup modal")
        state["execution_status"] = "no_match"
        state["completed"]        = True
        state["should_fallback"]  = False
        state["close_actions"]    = get_close_high_level_actions(state["task"])

    return state


def execute_action_node(state: DeploymentState) -> DeploymentState:
    """
    Pinecone-primary execution:
      1. Walk element_sequence from current_action.
      2. For each step: fetch Pinecone embedding → cosine match → execute.
      3. If cosine fails: LLM picks best element from description/content.
      4. React fallback is NOT triggered here — only when no HL action was found.
    """
    _log = state.get("log_callback") or print

    if state.get("execution_status") == "no_match":
        return state

    _log(f"\n{'='*60}")
    _log("⚙️  execute_action_node (Pinecone-primary)")

    matched_action = state.get("current_action")
    if not matched_action:
        _log("  ❌ No current_action in state")
        return state

    element_sequence = matched_action.get("element_sequence", [])
    if isinstance(element_sequence, str):
        try:
            element_sequence = json.loads(element_sequence)
        except Exception:
            element_sequence = []

    if not element_sequence:
        _log("  ❌ element_sequence empty — cannot execute")
        return state

    action_name = matched_action.get("name", "?")
    _log(f"🚀 Executing '{action_name}' — {len(element_sequence)} step(s) via Pinecone matching")
    state["total_steps"] = len(element_sequence)
    state["execution_status"] = "running"
    all_steps_ok = True

    for step_idx, step_info in enumerate(element_sequence):
        _log(f"  ── Step {step_idx+1}/{len(element_sequence)} ──────────────────")
        state["current_step"] = step_idx
        state_dict = dict(state)
        state_dict["current_step"] = step_idx

        # Capture fresh screen for each step
        updated = capture_and_parse_screen(state_dict)
        for k, v in updated.items():
            if k in state:
                state[k] = v
        state_dict = dict(state)
        state_dict["current_step"] = step_idx

        if not state["current_page"]["screenshot"]:
            _log(f"  ❌ Step {step_idx+1}: screen capture failed")
            all_steps_ok = False
            break

        element_id = step_info.get("element_id")
        if not element_id:
            _log(f"  ❌ Step {step_idx+1}: no element_id in step_info")
            all_steps_ok = False
            break

        # Pinecone cosine match
        matches = match_element_via_pinecone(element_id, step_info, state_dict)
        if not matches:
            _log(f"  ❌ Step {step_idx+1}: could not identify element on screen")
            all_steps_ok = False
            break

        best_match = matches[0]
        _log(f"  [EXEC] best_match screen_el={best_match.get('screen_element_id')} "
             f"action={best_match.get('action_type')} score={best_match.get('match_score', 0):.3f}")

        success = execute_element_action(state_dict, best_match)
        _log(f"  {'✓' if success else '❌'} Step {step_idx+1}/{len(element_sequence)} ADB result: {success}")

        if success:
            state["current_step"] = step_idx + 1
            state["history"].append({
                "step":       step_idx,
                "action":     best_match.get("action_type", "tap"),
                "element_id": element_id,
                "status":     "success",
                "screenshot": state["current_page"]["screenshot"],
            })
            time.sleep(1.5)
        else:
            _log(f"  ❌ Step {step_idx+1}: ADB returned failure")
            all_steps_ok = False
            break

    if all_steps_ok:
        state["execution_status"] = "success"
        state["completed"]         = True
        _log(f"✨ '{action_name}' complete — {len(element_sequence)} step(s) done")

    return state


def fallback_node(state: DeploymentState) -> DeploymentState:
    print("\n⚠️  fallback_node entered")
    print(f"  [DIAG-FALLBACK] execution_status={state.get('execution_status')}")
    print(f"  [DIAG-FALLBACK] current_step={state.get('current_step')}  history_len={len(state.get('history') or [])}")
    state = fallback_to_react(state)
    print(f"  [DIAG-FALLBACK] after fallback_to_react: execution_status={state.get('execution_status')}")

    state["completed"] = False
    print(f"  [DIAG-FALLBACK] completed forced to False — check_task_completion will judge")
    return state


# ─────────────────────────────────────────────────────────────────────────────
#  Routing functions  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def should_fallback(state: DeploymentState) -> str:
    _log = state.get("log_callback") or print
    result = "fallback" if state.get("should_fallback") else "continue"
    _log(f"  [ROUTE] should_fallback → '{result}'")
    return result


def is_task_completed(state: DeploymentState) -> str:
    _log = state.get("log_callback") or print
    if state.get("completed"):
        _log(f"  [ROUTE] is_task_completed → 'end'  (status={state.get('execution_status')})")
        return "end"

    workflow_iter = state.get("_workflow_iterations", 0) + 1
    state["_workflow_iterations"] = workflow_iter
    max_iters = state.get("max_workflow_iterations", 10)
    _log(f"  [ROUTE] is_task_completed → 'continue'  (iter={workflow_iter}/{max_iters})")

    if workflow_iter >= max_iters:
        _log(f"⚠️  Workflow iteration cap ({max_iters}) reached — ending")
        state["completed"] = True
        state["execution_status"] = "timeout"
        return "end"

    return "continue"


# ─────────────────────────────────────────────────────────────────────────────
#  LangGraph workflow
# ─────────────────────────────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    workflow = StateGraph(DeploymentState)

    workflow.add_node("capture_screen",   capture_screen_node)
    workflow.add_node("match_elements",   match_elements_node)
    workflow.add_node("execute_action",   execute_action_node)
    workflow.add_node("fallback",         fallback_node)
    workflow.add_node("check_completion", check_task_completion)

    workflow.set_entry_point("capture_screen")
    # screen capture failure → fallback
    workflow.add_conditional_edges(
        "capture_screen", should_fallback,
        {"fallback": "fallback", "continue": "match_elements"},
    )
    # no HL action found → fallback, else execute
    workflow.add_conditional_edges(
        "match_elements", should_fallback,
        {"fallback": "fallback", "continue": "execute_action"},
    )
    workflow.add_edge("execute_action",    "check_completion")
    workflow.add_edge("fallback",          "check_completion")
    workflow.add_conditional_edges(
        "check_completion", is_task_completed,
        {"end": END, "continue": "capture_screen"},
    )

    return workflow


def run_task(
    task: str,
    device: str = "emulator-5554",
    max_workflow_iterations: int = 10,
    log_callback=None,
    force_fallback: bool = False,
) -> Dict[str, Any]:
    """
    Execute a high-level task on an Android device.

    Args:
        task:                    Natural-language task description.
        device:                  ADB device serial.
        max_workflow_iterations: Hard cap on graph loop iterations.
        log_callback:            Callable(str) for real-time log streaming.
                                 Defaults to print() if not provided.
        force_fallback:          Whether to automatically route to React fallback mode.
    """
    _log = log_callback or print
    _log(f"\n{'#'*60}")
    _log(f"# deployment.py  v3  (Pinecone-primary)")
    _log(f"# task='{task}'  device='{device}'  max_iters={max_workflow_iterations}  force_fallback={force_fallback}")
    _log(f"{'#'*60}\n")
    try:
        from data.State import create_deployment_state
        state = create_deployment_state(task=task, device=device, max_retries=3)
        state["_workflow_iterations"]    = 0
        state["max_workflow_iterations"] = max_workflow_iterations
        state["log_callback"]            = log_callback   # propagated to all nodes
        state["close_actions"]           = []             # populated on no-match
        state["force_fallback"]          = force_fallback

        recursion_limit = max(50, max_workflow_iterations * 6)
        app    = build_workflow().compile()
        result = app.invoke(state, config={"recursion_limit": recursion_limit})

        close_actions = result.get("close_actions", [])

        if result["execution_status"] == "success" and result["current_page"]["screenshot"]:
            try:
                from PIL import Image
                Image.open(result["current_page"]["screenshot"]).show()
            except Exception:
                pass

        message = "Task execution completed"
        if result["execution_status"] == "no_match":
            message = "Add test cases in the exploration tab or navigate to fallback mechanism"

        return {
            "status":          result["execution_status"],
            "message":         message,
            "steps_completed": result["current_step"],
            "total_steps":     result["total_steps"],
            "close_actions":   close_actions,
        }
    except Exception as e:
        _log(f"❌ Error executing task: {e}")
        import traceback; traceback.print_exc()
        return {"status": "error", "message": str(e), "error": str(e), "close_actions": []}