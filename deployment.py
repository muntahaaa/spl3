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
import collections
import difflib
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
# pyrefly: ignore [missing-import]
from PIL import Image

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

# ── Plan-Reuse Constants (v3) ────────────────────────────────────────────────
USE_PLAN_REUSE        = os.getenv("USE_PLAN_REUSE", "1") == "1"
SCREENSHOT_SETTLE_SEC = float(os.getenv("SCREENSHOT_SETTLE_SEC", "2.0"))
PLAN_MIN_CONF         = 0.6     # below: whole task goes to ReAct
JUDGE_MIN_CONF        = 0.7     # below: escalate text judge to vision judge
TEXT_MATCH_MIN        = 0.85    # raw-text similarity for the no-LLM element accept
MAX_REACT_STEPS_SEG   = 6
MAX_RETRY_REACT_STEPS = 4
STUCK_WINDOW          = 3
CATALOG_PREFILTER_OVER = 25
CATALOG_TOP_K         = 12
CATALOG_TTL_SEC       = 60
SAFE_TEXT_RE          = r"^[A-Za-z0-9 :.,@+\-_/]*$"

LLM_CALLS = collections.Counter()

# ── LangSmith tracing ────────────────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true" if config.LANGCHAIN_TRACING_V2 else "false"
os.environ["LANGCHAIN_ENDPOINT"]   = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"]    = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"]    = "DeploymentExecution"

# ── NVIDIA NIM bridge (direct — no Firebase worker needed) ────────────────────
bridge = NvidiaBridge(
    max_tokens_text=1024 if USE_PLAN_REUSE else 4096,
    max_tokens_json=1024 if USE_PLAN_REUSE else 4096,
    max_tokens_vision=1024 if USE_PLAN_REUSE else 2048,
)

# ── Database / vector store ──────────────────────────────────────────────────
URI  = config.Neo4j_URI
AUTH = config.Neo4j_AUTH
db   = Neo4jDatabase(URI, AUTH, database=config.Neo4j_DB)
vector_db = VectorStore(api_key=config.PINECONE_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
#  Tiny sync wrapper so callers that aren't already in an async context
#  can invoke the bridge without restructuring.
# ─────────────────────────────────────────────────────────────────────────────

def _sync_call_text(system_prompt: str, user_prompt: str, timeout: float = 300.0, kind: str = "text") -> str:
    """Call NVIDIA NIM synchronously from any thread context."""
    LLM_CALLS[kind] += 1
    return asyncio.run(
        bridge.call_text(system_prompt=system_prompt, user_prompt=user_prompt)
    )


def _sync_call_json(
    system_prompt: str,
    user_prompt: str,
    images_b64: Optional[List[str]] = None,
    timeout: float = 300.0,
    kind: str = "json",
) -> Dict[str, Any]:
    LLM_CALLS[kind] += 1
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
    kind: str = "vision",
) -> str:
    LLM_CALLS[kind] += 1
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
#  Screen capture / OmniParser & Device helpers
# ─────────────────────────────────────────────────────────────────────────────

def _device_size(state: DeploymentState) -> Dict[str, int]:
    """Retrieve and cache device dimensions in state."""
    if state.get("device_size"):
        return state["device_size"]
    raw_size = get_device_size.invoke(state.get("device", "emulator"))
    if isinstance(raw_size, dict) and "width" in raw_size and "height" in raw_size:
        size = {"width": int(raw_size["width"]), "height": int(raw_size["height"])}
    elif isinstance(raw_size, str):
        try:
            sz = json.loads(raw_size)
            size = {"width": int(sz.get("width", sz.get("w", 1080))), "height": int(sz.get("height", sz.get("h", 2400)))}
        except Exception:
            size = {"width": 1080, "height": 2400}
    else:
        size = {"width": 1080, "height": 2400}
    state["device_size"] = size
    return size


def screen_hash(path: str) -> str:
    """Hash the cropped (top ~6% cropped out to ignore status bar / clock), 64x64 grayscale screenshot."""
    try:
        if not path or not os.path.exists(path):
            return ""
        with Image.open(path) as img:
            w, h = img.size
            crop_box = (0, int(h * 0.06), w, h)
            cropped = img.crop(crop_box).convert("L").resize((64, 64))
            return hashlib.md5(cropped.tobytes()).hexdigest()
    except Exception:
        return ""


def capture_and_parse_screen(state: DeploymentState) -> DeploymentState:
    try:
        screenshot_path = take_screenshot.invoke({
            "device":    state["device"],
            "app_name":  "deployment",
            "step":      state["current_step"],
            "settle":    SCREENSHOT_SETTLE_SEC,
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
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """
    Element-matching strategy:
      1. Fetch the stored Neo4j/Pinecone details (raw text & type from other_info).
      2. Guard: if element missing everywhere, abort immediately (0 LLM calls).
      3. Fast No-LLM accept: if spatial candidate has text ratio >= TEXT_MATCH_MIN, accept directly.
      4. If strict: skip distance shortcut and semantic fallback (entry check).
      5. Otherwise, verify via LLM or rank candidates and fallback to semantic matching.
    """
    _log = lambda msg: _log_both(state, msg)
    screen_elements = state["current_page"]["elements_data"]
    screenshot_path = state["current_page"]["screenshot"]

    if not screen_elements or not screenshot_path:
        _log("  [ACTION MATCHING] ⚠️  No screen elements or screenshot available")
        return []

    # ── 1. Fetch stored details from Neo4j & Pinecone ───────────────────────────
    stored_content = ""
    raw_content = ""
    raw_type = ""
    stored_type = ""
    stored_bbox: Optional[List[float]] = None

    neo4j_element = db.get_element_by_id(element_id) or {}
    neo4j_desc = neo4j_element.get("description") or ""
    neo4j_reasoning = neo4j_element.get("reasoning") or ""
    other_info = neo4j_element.get("other_info") or {}
    if isinstance(other_info, str):
        try:
            other_info = json.loads(other_info)
        except Exception:
            other_info = {}
    raw_content = other_info.get("content", "")
    raw_type = other_info.get("type", "")
    stored_type = raw_type or neo4j_element.get("element_type", "")
    bbox_raw = neo4j_element.get("bounding_box")
    if isinstance(bbox_raw, str):
        try:
            stored_bbox = json.loads(bbox_raw)
        except Exception:
            stored_bbox = None
    elif isinstance(bbox_raw, list):
        stored_bbox = bbox_raw

    # Fallback to Pinecone if needed
    if not raw_content or not stored_type or not stored_bbox:
        _log(f"[ACTION MATCHING] Fetching stored metadata for element {element_id[:8]} from Pinecone...")
        try:
            fetch_result = vector_db.index.fetch(ids=[element_id], namespace="element")
            vec_data = (fetch_result.get("vectors") or {}).get(element_id)
            if vec_data:
                stored_meta = vec_data.get("metadata", {})
                if not raw_content:
                    raw_content = stored_meta.get("content", "")
                if not stored_type:
                    stored_type = stored_meta.get("type", "")
                if not stored_bbox:
                    bbox_raw_pc = stored_meta.get("bbox")
                    if isinstance(bbox_raw_pc, str):
                        try:
                            stored_bbox = json.loads(bbox_raw_pc)
                        except Exception:
                            stored_bbox = None
                    elif isinstance(bbox_raw_pc, list):
                        stored_bbox = bbox_raw_pc
        except Exception as exc:
            _log(f"  [ACTION MATCHING] Pinecone fetch error: {exc}")

    # Guard (Step 10.2 / Fact D6)
    if not neo4j_element and not raw_content and not stored_bbox:
        _log(f"  [ACTION MATCHING] ⚠️ Element details missing from both Neo4j and Pinecone for {element_id[:8]}. Aborting match.")
        return []

    stored_content = neo4j_desc or raw_content
    _log(f"[ACTION MATCHING] Stored element: raw='{raw_content}' type='{stored_type}' bbox={stored_bbox}")

    # ── 2. Spatial match candidate search ───────────────────────────────
    corresponding_live_element = None
    spatial_idx = -1
    min_dist = float("inf")
    if stored_bbox and len(stored_bbox) == 4:
        for idx, el in enumerate(screen_elements):
            el_bbox = el.get("bbox")
            if el_bbox and len(el_bbox) == 4:
                dist = _bbox_center_dist(stored_bbox, el_bbox)
                if dist < min_dist:
                    min_dist = dist
                    corresponding_live_element = el
                    spatial_idx = idx

    # ── 3. Match verification ──────────────────────────────────────────
    if corresponding_live_element:
        candidate_content = corresponding_live_element.get("content", "")
        candidate_type = corresponding_live_element.get("type", "")
        _log(f"[ACTION MATCHING] Found spatial candidate at index {spatial_idx}: content='{candidate_content}' type='{candidate_type}' (dist: {min_dist:.4f})")

        # Step 10.3: Fast No-LLM accept via string ratio
        if raw_content and candidate_content:
            ratio = difflib.SequenceMatcher(None, raw_content.strip().lower(), candidate_content.strip().lower()).ratio()
            if ratio >= TEXT_MATCH_MIN and (not raw_type or not candidate_type or raw_type == candidate_type):
                _log(f"[ACTION MATCHING] ✓ Spatial match verified via raw text match ({ratio:.2f} >= {TEXT_MATCH_MIN}) [0 LLM calls]")
                return [{
                    "element_id":        element_id,
                    "match_score":       ratio,
                    "screen_element_id": spatial_idx,
                    "action_type":       step_info.get("atomic_action", "tap"),
                    "parameters":        step_info.get("action_params", {}),
                }]

        # Distance shortcut (if not strict)
        if not strict and min_dist < 0.03:
            _log(f"[ACTION MATCHING] ✓ Spatial match verified dynamically via distance ({min_dist:.4f} < 0.03)")
            return [{
                "element_id":        element_id,
                "match_score":       1.0 - min_dist,
                "screen_element_id": spatial_idx,
                "action_type":       step_info.get("atomic_action", "tap"),
                "parameters":        step_info.get("action_params", {}),
            }]

        # LLM Verification
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
            res = _sync_call_json(_VERIFY_MATCH_SYSTEM, user_prompt, timeout=120, kind="verify_spatial")
            similarity = float(res.get("similarity", 0.0))
            reason = res.get("reason", "")
            _log(f"[ACTION MATCHING] LLM verify similarity score: {similarity:.2f} (threshold: {threshold}) — Reason: {reason}")

            if similarity > threshold:
                _log(f"[ACTION MATCHING] ✓ Spatial match verified via LLM (score {similarity:.2f} > {threshold})")
                return [{
                    "element_id":        element_id,
                    "match_score":       similarity,
                    "screen_element_id": spatial_idx,
                    "action_type":       step_info.get("atomic_action", "tap"),
                    "parameters":        step_info.get("action_params", {}),
                }]
        except Exception as exc:
            _log(f"  [ACTION MATCHING] LLM verification error: {exc}")

    if strict:
        _log("  [ACTION MATCHING] Strict check failed — aborting without semantic fallback")
        return []

    # ── 4. Fallback to semantic matching across all live elements ─────
    _log("[ACTION MATCHING] Spatial match verification failed. Falling back to semantic matching...")
    return llm_bbox_fallback(element_id, step_info, state, stored_content, stored_type, raw_content=raw_content)


def llm_bbox_fallback(
    element_id: str,
    step_info: Dict[str, Any],
    state: DeploymentState,
    stored_content: str,
    stored_type: str,
    raw_content: str = "",
) -> List[Dict[str, Any]]:
    """
    Ask the LLM to choose the best semantic match from all live elements.
    Ranks candidates and selects top 15 to preserve context window.
    """
    _log = lambda msg: _log_both(state, msg)
    screen_elements = state["current_page"]["elements_data"]
    if not screen_elements:
        return []

    # Pre-rank candidate live elements
    ranked = []
    for idx, el in enumerate(screen_elements):
        cnt = el.get("content", "").strip()
        typ = el.get("type", "")
        score = 0.0
        if raw_content and cnt:
            score += difflib.SequenceMatcher(None, raw_content.lower(), cnt.lower()).ratio()
        if stored_type and typ == stored_type:
            score += 0.2
        ranked.append((score, idx, el))
    ranked.sort(key=lambda x: x[0], reverse=True)
    top_candidates = ranked[:15]

    live_elements_list = ""
    for _, idx, el in top_candidates:
        live_elements_list += f"{idx}: type={el.get('type','?')}  content='{el.get('content','')}'\n"

    user_prompt = (
        f"Target Element Description: {stored_content}\n"
        f"Target Element Type: {stored_type}\n\n"
        f"Live Screen Elements:\n{live_elements_list}\n"
        f"Choose the element that has the closest semantic match to the target description. "
        f"Return JSON only."
    )

    try:
        result = _sync_call_json(_SEMANTIC_MATCH_SYSTEM, user_prompt, timeout=120, kind="match_semantic")
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
        device_size = _device_size(state)

        # Calculate coordinates conditionally: scale if relative (<= 1.0), use directly otherwise
        if bbox and len(bbox) == 4 and all(val <= 1.0 for val in bbox):
            center_x = int((bbox[0] + bbox[2]) / 2 * device_size["width"])
            center_y = int((bbox[1] + bbox[3]) / 2 * device_size["height"])
        else:
            center_x = int((bbox[0] + bbox[2]) / 2) if bbox and len(bbox) == 4 else 0
            center_y = int((bbox[1] + bbox[3]) / 2) if bbox and len(bbox) == 4 else 0

        action_params = {"device": state["device"], "action": action_type, "x": center_x, "y": center_y}
        if action_type == "text":
            action_params["input_str"] = parameters.get("text") or parameters.get("input_str", "")
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
#  Bounded ReAct step & segment runner (Steps 11 & 12)
# ─────────────────────────────────────────────────────────────────────────────

_REACT_STEP_SYSTEM = (
    "You are an intelligent smartphone operation assistant. "
    "Observe the current screen elements and perform one atomic operation "
    "(tap / type text / swipe / long press / back) to progress toward the sub-goal. "
    "If the sub-goal is already satisfied on the current screen, reply {\"action\": \"done\"}.\n"
    "Reply with a JSON object:\n"
    "  {\"action\": \"<tap/text/swipe/long_press/back/done>\", "
    "\"element_id\": <int or str of target element>, "
    "\"input_str\": \"<text if action=text>\", "
    "\"direction\": \"<up/down/left/right if action=swipe>\", "
    "\"duration\": <ms if action=long_press>}\n"
    "Return JSON only."
)


def react_step(state: DeploymentState, goal: str) -> str:
    """Execute one bounded ReAct step towards a goal. Returns 'done' | 'acted' | 'error'."""
    elements_data = state["current_page"].get("elements_data") or []
    if not elements_data:
        return "error"

    elem_lines = []
    element_index = {}
    for el in elements_data[:60]:
        eid = el.get("ID", el.get("id", "?"))
        bbox = el.get("bbox", [])
        element_index[str(eid)] = bbox
        cnt = el.get("content", "").strip()
        typ = el.get("type", "")
        if cnt or typ:
            elem_lines.append(f"{eid}|{typ}|{cnt}")

    recent_history = [
        f"{h.get('action', '?')} on {h.get('element_id', '')} status={h.get('status')}"
        for h in (state.get("history") or [])[-5:]
    ]

    user_prompt = (
        f"Goal: {goal}\n\n"
        f"Recent actions:\n" + ("\n".join(recent_history) if recent_history else "None") + "\n\n"
        f"Current Screen Elements (ID|type|content):\n" + "\n".join(elem_lines) + "\n\n"
        f"What action should be taken next?"
    )

    try:
        res = _sync_call_json(_REACT_STEP_SYSTEM, user_prompt, kind="react")
        action_type = res.get("action", "").lower().strip()
        if action_type == "done":
            _log_both(state, "  [REACT] Goal reached — model reported 'done'")
            return "done"

        device_sz = _device_size(state)
        action_params = {"device": state["device"], "action": action_type}

        if action_type == "back":
            pass
        elif action_type in ("tap", "long_press"):
            target_id = str(res.get("element_id", ""))
            bbox = element_index.get(target_id)
            if not bbox or len(bbox) != 4:
                _log_both(state, f"  [REACT] ❌ Invalid element_id {target_id}")
                return "error"
            if all(v <= 1.0 for v in bbox):
                cx = int((bbox[0] + bbox[2]) / 2 * device_sz["width"])
                cy = int((bbox[1] + bbox[3]) / 2 * device_sz["height"])
            else:
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
            action_params["x"] = cx
            action_params["y"] = cy
            if action_type == "long_press":
                action_params["duration"] = int(res.get("duration", 1000))
        elif action_type == "text":
            input_str = res.get("input_str", "")
            if not re.fullmatch(SAFE_TEXT_RE, input_str):
                _log_both(state, f"  [REACT] ❌ Unsafe text in input_str: {input_str}")
                return "error"
            action_params["input_str"] = input_str
        elif action_type in ("swipe", "swipe_short", "swipe_long"):
            action_params["direction"] = res.get("direction", "up")
            action_params["x"] = device_sz["width"] // 2
            action_params["y"] = device_sz["height"] // 2
        else:
            _log_both(state, f"  [REACT] ❌ Unknown action: {action_type}")
            return "error"

        result = screen_action.invoke(action_params)
        success = _parse_action_result(result)
        state["history"].append({
            "step": state["current_step"],
            "action": action_type,
            "params": action_params,
            "status": "success" if success else "error",
            "screenshot": state["current_page"].get("screenshot"),
        })
        state["react_steps"] = state.get("react_steps", 0) + 1
        return "acted" if success else "error"
    except Exception as e:
        _log_both(state, f"  [REACT] Error in react_step: {e}")
        return "error"


def run_react_segment(state: DeploymentState, goal: str, cap: int = MAX_REACT_STEPS_SEG) -> str:
    """Run a bounded ReAct segment with stuck detection. Returns 'done' | 'stuck' | 'error' | 'cap'."""
    _log_both(state, f"🔄 Running bounded ReAct segment: '{goal}' (cap={cap})")
    for step_num in range(cap):
        state = capture_and_parse_screen(state)
        curr_shot = state["current_page"].get("screenshot")
        if not curr_shot or not os.path.exists(curr_shot):
            return "error"
        h = screen_hash(curr_shot)
        hashes = state.get("screen_hashes", [])
        hashes.append(h)
        state["screen_hashes"] = hashes
        if len(hashes) >= STUCK_WINDOW and all(x == hashes[-1] and x != "" for x in hashes[-STUCK_WINDOW:]):
            _log_both(state, "  [REACT] ⚠️ Screen is stuck (identical visual hashes) — stopping segment")
            return "stuck"

        r = react_step(state, goal)
        if r == "done":
            return "done"
        if r == "error":
            return "error"
    return "cap"


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
#  Task completion check  (Legacy & Two-tier v3)
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


def check_task_completion_legacy(state: DeploymentState) -> DeploymentState:
    _log = state.get("log_callback") or print

    if state.get("execution_status") == "no_match":
        state["completed"] = True
        return state

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


def _normalize_text_for_match(s: str) -> str:
    digits = re.sub(r"\D", "", s)
    if digits:
        return digits
    return s.strip().lower()


def texts_present(expected_texts: List[str], elements: List[Dict[str, Any]]) -> Optional[bool]:
    """
    Checks if expected typed texts appear on screen.
    Returns True if all expected texts found, False if absent, or None if OCR text count < 5 (unknown).
    """
    if not expected_texts:
        return True

    on_screen_texts = []
    for el in elements:
        txt = (el.get("content") or el.get("text") or "").strip()
        if txt:
            on_screen_texts.append(txt)

    if len(on_screen_texts) < 5:
        return None

    for exp in expected_texts:
        norm_exp = _normalize_text_for_match(exp)
        found = False
        for ost in on_screen_texts:
            norm_ost = _normalize_text_for_match(ost)
            if norm_exp in norm_ost or (norm_exp.isdigit() and norm_ost.isdigit() and norm_exp == norm_ost):
                found = True
                break
            if exp.strip().lower() in ost.lower():
                found = True
                break
        if not found:
            return False
    return True


_TWO_TIER_JUDGE_SYSTEM = (
    "You are an accurate, strict UI task completion verifier.\n"
    "Evaluate if the user's task was successfully completed based on the current screen elements and recent actions.\n"
    "Reply with a JSON object strictly following this schema:\n"
    "{\n"
    "  \"complete\": <true/false>,\n"
    "  \"confidence\": <float between 0.0 and 1.0>,\n"
    "  \"evidence\": \"<concise reason or visible text/element confirming status>\",\n"
    "  \"missing\": \"<what is still missing if incomplete, otherwise empty>\"\n"
    "}\n"
    "Return JSON only."
)


def check_task_completion(state: DeploymentState) -> DeploymentState:
    """Two-tier JSON judge with text check as router (Steps 7 & 9)."""
    if not USE_PLAN_REUSE:
        return check_task_completion_legacy(state)

    _log = state.get("log_callback") or print
    _log("\n🔍 [JUDGE] Evaluating task completion (two-tier)...")

    # Step 7: Capture final screen first
    updated = capture_and_parse_screen(dict(state))
    for k, v in updated.items():
        if k in state:
            state[k] = v

    screenshot = state["current_page"].get("screenshot")
    elements = state["current_page"].get("elements_data") or []
    state["final_screenshot"] = screenshot
    state["final_elements"] = elements

    if not screenshot or not os.path.exists(screenshot):
        _log("  ❌ Screen capture failed during completion check")
        state["completed"] = False
        state["finished"] = True
        state["execution_status"] = "error"
        return state

    if state.get("execution_status") == "stuck":
        _log("  ⚠️ Execution marked as stuck — task is not complete")
        state["completed"] = False
        state["finished"] = True
        return state

    # Step 9.1: Collect expected_texts from replayed/acted text steps
    expected_texts = []
    for h in state.get("history") or []:
        if h.get("action") == "text":
            val = (h.get("params") or {}).get("input_str") or h.get("input_str") or h.get("text")
            if val:
                expected_texts.append(val)

    # Step 9.2: Text check router
    tp = texts_present(expected_texts, elements)
    _log(f"  [JUDGE] texts_present check: {tp} (expected_texts={expected_texts})")

    task = state["task"]
    recent_actions = [
        f"{h.get('action', '?')} on {h.get('element_id', '')} status={h.get('status')}"
        for h in (state.get("history") or [])[-3:]
    ]
    recent_actions_str = "\n".join(recent_actions) if recent_actions else "None"

    judge_res = None
    # If a value is absent -> skip text judge and go straight to vision judge
    if tp is not False:
        # Step 9.3: Text judge
        elem_lines = []
        for el in elements[:60]:
            eid = el.get("ID", el.get("id", "?"))
            cnt = (el.get("content") or el.get("text") or "").strip()
            if cnt:
                elem_lines.append(f"{eid}|{cnt}")

        prompt_text = (
            f"Task: {task}\n"
            f"Expected typed values: {expected_texts if expected_texts else 'None'}\n"
            f"Last 3 actions:\n{recent_actions_str}\n\n"
            f"On-screen text elements:\n" + ("\n".join(elem_lines) if elem_lines else "None") + "\n\n"
            "Is the task complete?"
        )
        try:
            judge_res = _sync_call_json(
                system_prompt=_TWO_TIER_JUDGE_SYSTEM,
                user_prompt=prompt_text,
                kind="judge_text",
            )
            _log(f"  [JUDGE-TEXT] result: {judge_res}")
        except Exception as e:
            _log(f"  ⚠️ Text judge error: {e}")
            judge_res = None

    # Step 9.4 & 9.5: Vision judge fallback if text judge was skipped or confidence < JUDGE_MIN_CONF
    conf = float(judge_res.get("confidence", 0.0)) if judge_res else 0.0
    if judge_res is None or conf < JUDGE_MIN_CONF:
        _log(f"  [JUDGE] Escalating to vision judge (conf={conf:.2f} < {JUDGE_MIN_CONF})")
        b64 = _img_to_b64(screenshot)
        if b64:
            prompt_vision = (
                f"Task: {task}\n"
                f"Expected typed values: {expected_texts if expected_texts else 'None'}\n"
                f"Last 3 actions:\n{recent_actions_str}\n\n"
                "Evaluate the screen image. Is the task complete?"
            )
            try:
                judge_res = _sync_call_json(
                    system_prompt=_TWO_TIER_JUDGE_SYSTEM,
                    user_prompt=prompt_vision,
                    images_b64=[b64],
                    kind="judge_vision",
                )
                _log(f"  [JUDGE-VISION] result: {judge_res}")
            except Exception as e:
                _log(f"  ⚠️ Vision judge error: {e}")
                judge_res = {"complete": False, "confidence": 0.0, "evidence": str(e), "missing": "judge error"}

    if not judge_res:
        judge_res = {"complete": False, "confidence": 0.0, "evidence": "no response", "missing": "unknown"}

    complete = bool(judge_res.get("complete", False))
    confidence = float(judge_res.get("confidence", 0.0))
    missing = str(judge_res.get("missing", ""))
    evidence = str(judge_res.get("evidence", ""))

    state["history"].append({
        "step": state.get("current_step", 0),
        "action": "task_completion_check",
        "judgement": judge_res,
        "status": "success" if complete else "incomplete",
        "completed": complete,
    })

    # Step 9.8: Route completion outcome
    if complete and confidence >= JUDGE_MIN_CONF:
        state["completed"] = True
        state["finished"] = True
        state["execution_status"] = "completed"
        _log(f"✨ Task judged complete! Evidence: {evidence}")
    else:
        state["completed"] = False
        if state.get("retries", 0) < 1:
            state["retries"] = state.get("retries", 0) + 1
            retry_goal = f"{task}. Still missing: {missing or 'task unconfirmed'}"
            _log(f"⚠️ Task incomplete. Scheduling ReAct retry segment (retry {state['retries']}/1): '{retry_goal}'")
            state["plan"] = {
                "action_id": None,
                "confidence": 1.0,
                "segments": [{"type": "react", "goal": retry_goal, "cap": MAX_RETRY_REACT_STEPS}],
                "text_overrides": {},
            }
            state["seg_index"] = 0
            state["finished"] = False
        else:
            state["execution_status"] = "failed"
            state["finished"] = True
            _log(f"❌ Task failed after retries. Missing: {missing}")

    return state


# ─────────────────────────────────────────────────────────────────────────────
#  Action catalog for planner (Step 13)
# ─────────────────────────────────────────────────────────────────────────────

_catalog_cache = {"timestamp": 0.0, "catalog_text": "", "actions_by_id": {}}


def build_action_catalog(task: str) -> Tuple[str, Dict[str, Any]]:
    """Build compact Action catalog for the planner with caching (Step 13)."""
    global _catalog_cache
    now = time.time()
    if now - _catalog_cache["timestamp"] < CATALOG_TTL_SEC and _catalog_cache["catalog_text"]:
        return _catalog_cache["catalog_text"], _catalog_cache["actions_by_id"]

    actions = db.get_all_high_level_actions()
    if not actions:
        _catalog_cache = {"timestamp": now, "catalog_text": "No stored actions available.", "actions_by_id": {}}
        return _catalog_cache["catalog_text"], _catalog_cache["actions_by_id"]

    actions_by_id = {act["action_id"]: act for act in actions if act.get("action_id")}

    all_elem_ids = set()
    first_elem_ids = set()
    for act in actions:
        seq = act.get("element_sequence") or []
        if isinstance(seq, str):
            try:
                seq = json.loads(seq)
            except Exception:
                seq = []
        act["element_sequence"] = seq
        for idx, step in enumerate(seq):
            eid = step.get("element_id")
            if eid:
                all_elem_ids.add(eid)
                if idx == 0:
                    first_elem_ids.add(eid)

    elements_info = {}
    if all_elem_ids:
        try:
            with db.driver.session(database=db.database) as session:
                q = """
                MATCH (e:Element) WHERE e.element_id IN $ids
                RETURN e.element_id AS id, e.other_info AS oi, e.description AS d
                """
                res = session.run(q, ids=list(all_elem_ids))
                for rec in res:
                    oi_raw = rec["oi"] or "{}"
                    try:
                        oi = json.loads(oi_raw) if isinstance(oi_raw, str) else oi_raw
                    except Exception:
                        oi = {}
                    elements_info[rec["id"]] = {
                        "content": oi.get("content", ""),
                        "description": rec["d"] or "",
                    }
        except Exception as exc:
            print(f"⚠️ Error querying element catalog info: {exc}")

    first_page_tasks = {}
    if first_elem_ids:
        try:
            with db.driver.session(database=db.database) as session:
                q_p = """
                MATCH (p:Page)-[:HAS_ELEMENT]->(e:Element)
                WHERE e.element_id IN $first_ids
                RETURN e.element_id AS eid, p.other_info AS info
                """
                res = session.run(q_p, first_ids=list(first_elem_ids))
                for rec in res:
                    info_raw = rec["info"] or "{}"
                    try:
                        info = json.loads(info_raw) if isinstance(info_raw, str) else info_raw
                    except Exception:
                        info = {}
                    desc = info.get("task_info", {}).get("description")
                    if desc:
                        first_page_tasks[rec["eid"]] = desc
        except Exception as exc:
            print(f"⚠️ Error querying first-page task info: {exc}")

    action_entries = []
    task_tokens = set(re.findall(r"\w+", task.lower()))

    for act in actions:
        aid = act.get("action_id", "")
        seq = act.get("element_sequence") or []
        first_eid = seq[0].get("element_id") if seq else None

        source_task = act.get("source_task")
        if not source_task and first_eid and first_eid in first_page_tasks:
            source_task = first_page_tasks[first_eid]
        if not source_task:
            source_task = act.get("name", "Unnamed Action")

        act["_resolved_source_task"] = source_task

        step_labels = []
        for idx, step in enumerate(seq, 1):
            atomic = step.get("atomic_action", "tap")
            params = step.get("action_params", {})
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except Exception:
                    params = {}
            eid = step.get("element_id", "")
            e_info = elements_info.get(eid, {})

            if atomic in ("tap", "long_press"):
                cnt = e_info.get("content", "")
                lbl = cnt if cnt else (e_info.get("description", "")[:50] or "element")
                step_labels.append(f"{idx} {atomic} \"{lbl}\"")
            elif atomic == "text":
                txt = params.get("text") or params.get("input_str") or ""
                step_labels.append(f"{idx} text \"{txt}\"")
            elif atomic == "back":
                step_labels.append(f"{idx} back")
            elif atomic.startswith("swipe"):
                direction = params.get("direction", "screen")
                step_labels.append(f"{idx} swipe {direction}")
            else:
                step_labels.append(f"{idx} {atomic}")

        act["_step_labels_str"] = " · ".join(step_labels)
        entry_text = f"[id={aid}] recorded as: \"{source_task}\" | name: {act.get('name', '')}\n  {act['_step_labels_str']}"

        entry_tokens = set(re.findall(r"\w+", f"{source_task} {act.get('name', '')} {act['_step_labels_str']}".lower()))
        overlap = len(task_tokens & entry_tokens)
        action_entries.append((overlap, aid, entry_text))

    if len(action_entries) > CATALOG_PREFILTER_OVER:
        action_entries.sort(key=lambda x: x[0], reverse=True)
        action_entries = action_entries[:CATALOG_TOP_K]

    catalog_text = "\n\n".join(e[2] for e in action_entries)
    _catalog_cache = {
        "timestamp": now,
        "catalog_text": catalog_text,
        "actions_by_id": actions_by_id,
    }
    return catalog_text, actions_by_id


# ─────────────────────────────────────────────────────────────────────────────
#  Planner & Plan Executor (Steps 14, 15, 15a)
# ─────────────────────────────────────────────────────────────────────────────

_PLAN_SYSTEM = (
    "Plan how to do the TASK using stored actions. Reply JSON only:\n"
    "{\n"
    "  \"action_id\": \"<action_id or null>\",\n"
    "  \"confidence\": <float 0.0 to 1.0>,\n"
    "  \"segments\": [\n"
    "    {\"type\": \"replay\", \"from\": <1-based int>, \"to\": <1-based int>},\n"
    "    {\"type\": \"react\", \"goal\": \"<short sub-goal>\"}\n"
    "  ],\n"
    "  \"text_overrides\": {\"<step_number>\": \"<new text>\"}\n"
    "}\n"
    "Rules:\n"
    "- A 'replay' segment uses step numbers 'from'..'to' (1-based, inclusive, increasing, no overlap) of that one action, "
    "only for steps that serve the task unchanged.\n"
    "- Use 'text_overrides' ({step_number: new text}) only for steps shown as 'text'.\n"
    "- Anything the stored steps do not cover becomes a 'react' segment with a short goal.\n"
    "- Same app but a different goal: replay only the shared opening steps, then react.\n"
    "- Nothing related: action_id null and one react segment with the whole task.\n"
    "- Never copy step contents."
)


def plan_task(state: DeploymentState) -> DeploymentState:
    """Generate execution plan from action catalog or degrade to ReAct (Step 14)."""
    _log = state.get("log_callback") or print

    if state.get("plan"):
        return state

    task = state["task"]
    _log(f"\n📋 [PLANNER] Planning execution for: '{task}'")

    if state.get("force_fallback"):
        _log("⚡ Force fallback requested — single ReAct segment")
        state["plan"] = {
            "action_id": None,
            "confidence": 1.0,
            "segments": [{"type": "react", "goal": task}],
            "text_overrides": {},
        }
        state["seg_index"] = 0
        return state

    catalog_text, actions_by_id = build_action_catalog(task)

    # Step 14.1: Exact repeat check (0 LLM calls)
    norm_task = re.sub(r"[^a-z0-9]", "", task.lower())
    for aid, act in actions_by_id.items():
        src = act.get("_resolved_source_task", "")
        if norm_task and re.sub(r"[^a-z0-9]", "", src.lower()) == norm_task:
            seq_len = len(act.get("element_sequence") or [])
            if seq_len > 0:
                _log(f"🎯 [PLANNER] Exact task match found with '{src}' (0 LLM calls)")
                state["plan"] = {
                    "action_id": aid,
                    "confidence": 1.0,
                    "segments": [{"type": "replay", "from": 1, "to": seq_len}],
                    "text_overrides": {},
                }
                state["seg_index"] = 0
                return state

    # Step 14.2: LLM planner
    user_prompt = f"TASK: {task}\n\nStored actions catalog:\n{catalog_text}"
    try:
        raw_plan = _sync_call_json(
            system_prompt=_PLAN_SYSTEM,
            user_prompt=user_prompt,
            kind="plan",
        )
        _log(f"  [PLANNER] Raw plan from LLM: {raw_plan}")
    except Exception as e:
        _log(f"⚠️ Planner error: {e} — falling back to single ReAct segment")
        raw_plan = None

    # Step 14.3: Plan validation in code
    valid_plan = False
    plan = None
    if isinstance(raw_plan, dict):
        aid = raw_plan.get("action_id")
        conf = float(raw_plan.get("confidence", 0.0))
        segments = raw_plan.get("segments") or []
        overrides = raw_plan.get("text_overrides") or {}

        if conf >= PLAN_MIN_CONF and segments:
            if aid is None:
                # Valid pure react plan
                if all(s.get("type") == "react" and s.get("goal") for s in segments):
                    valid_plan = True
                    plan = {"action_id": None, "confidence": conf, "segments": segments, "text_overrides": {}}
            elif aid in actions_by_id:
                act = actions_by_id[aid]
                seq = act.get("element_sequence") or []
                seq_len = len(seq)
                curr_step = 0
                seg_ok = True
                cleaned_segments = []

                for s in segments:
                    stype = s.get("type")
                    if stype == "replay":
                        f_idx = int(s.get("from", 0))
                        t_idx = int(s.get("to", 0))
                        if 1 <= f_idx <= t_idx <= seq_len and f_idx > curr_step:
                            cleaned_segments.append({"type": "replay", "from": f_idx, "to": t_idx})
                            curr_step = t_idx
                        else:
                            seg_ok = False
                            break
                    elif stype == "react":
                        goal = str(s.get("goal", "")).strip()
                        if goal:
                            cleaned_segments.append({"type": "react", "goal": goal})
                        else:
                            seg_ok = False
                            break
                    else:
                        seg_ok = False
                        break

                cleaned_overrides = {}
                for k, v in overrides.items():
                    try:
                        step_num = int(k)
                        if 1 <= step_num <= seq_len:
                            target_step = seq[step_num - 1]
                            if target_step.get("atomic_action") == "text":
                                if re.fullmatch(SAFE_TEXT_RE, str(v)):
                                    cleaned_overrides[str(step_num)] = str(v)
                    except Exception:
                        pass

                if seg_ok and cleaned_segments:
                    valid_plan = True
                    plan = {
                        "action_id": aid,
                        "confidence": conf,
                        "segments": cleaned_segments,
                        "text_overrides": cleaned_overrides,
                    }

    if not valid_plan or not plan:
        _log("⚠️ Plan validation failed or low confidence — using single ReAct segment")
        plan = {
            "action_id": None,
            "confidence": 1.0,
            "segments": [{"type": "react", "goal": task}],
            "text_overrides": {},
        }

    state["plan"] = plan
    state["seg_index"] = 0
    _log(f"✓ [PLANNER] Final plan: {plan}")
    return state


def execute_plan_node(state: DeploymentState) -> DeploymentState:
    """Execute plan segments (Replay + ReAct) with entry check (Steps 15 & 15a)."""
    _log = state.get("log_callback") or print
    plan = state.get("plan")

    if not plan or not plan.get("segments"):
        _log("  ❌ No plan to execute")
        state["execution_status"] = "error"
        state["finished"] = True
        return state

    segments = plan["segments"]
    aid = plan.get("action_id")
    overrides = plan.get("text_overrides") or {}

    catalog_text, actions_by_id = build_action_catalog(state["task"])
    act = actions_by_id.get(aid, {}) if aid else {}
    seq = act.get("element_sequence") or []

    _log(f"\n⚙️ [EXECUTE-PLAN] Starting execution from segment {state.get('seg_index', 0)+1}/{len(segments)}")

    while state.get("seg_index", 0) < len(segments):
        idx = state["seg_index"]
        seg = segments[idx]
        stype = seg.get("type")
        _log(f"\n▶ Segment {idx+1}/{len(segments)}: {seg}")

        if stype == "replay":
            f_idx = seg["from"]
            t_idx = seg["to"]

            # Step 15a: Entry check before first replay step that is tap/long_press
            if not state.get("_entry_check_done"):
                first_tap_step = None
                for s_i in range(f_idx - 1, t_idx):
                    if seq[s_i].get("atomic_action") in ("tap", "long_press"):
                        first_tap_step = seq[s_i]
                        break

                if first_tap_step:
                    first_eid = first_tap_step.get("element_id")
                    state = capture_and_parse_screen(state)
                    m = match_element_via_pinecone(first_eid, first_tap_step, state, strict=True)
                    if not m:
                        _log("  ⚠️ Strict entry check: element not found on current screen. Pressing HOME...")
                        press_home(state["device"])
                        state = capture_and_parse_screen(state)
                        m = match_element_via_pinecone(first_eid, first_tap_step, state, strict=True)

                    if not m:
                        _log("  ❌ Strict entry check failed after HOME. Degrading to ReAct for full task.")
                        state["plan"] = {
                            "action_id": None,
                            "confidence": 1.0,
                            "segments": [{"type": "react", "goal": state["task"]}],
                            "text_overrides": {},
                        }
                        state["seg_index"] = 0
                        state["_entry_check_done"] = True
                        return execute_plan_node(state)

                state["_entry_check_done"] = True

            # Walk steps
            step_failed = False
            for step_num in range(f_idx, t_idx + 1):
                step_idx = step_num - 1
                step = seq[step_idx]
                atomic = step.get("atomic_action", "tap")
                params = step.get("action_params", {})
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except Exception:
                        params = {}
                text = overrides.get(str(step_num)) or params.get("text") or params.get("input_str") or ""
                eid = step.get("element_id", "")

                _log(f"  -- Replay step {step_num} ({atomic}) --")

                if atomic == "back":
                    ok = screen_action.invoke({"device": state["device"], "action": "back"})
                    step_ok = _parse_action_result(ok)
                elif atomic == "text":
                    if not text or not re.fullmatch(SAFE_TEXT_RE, text):
                        _log(f"  ❌ Invalid or unsafe text: '{text}'")
                        step_ok = False
                    else:
                        ok = screen_action.invoke({"device": state["device"], "action": "text", "input_str": text})
                        step_ok = _parse_action_result(ok)
                elif atomic == "swipe_precise" or (atomic.startswith("swipe") and params.get("start") and params.get("end")):
                    ok = screen_action.invoke({
                        "device": state["device"],
                        "action": "swipe_precise",
                        "start": tuple(params["start"]),
                        "end": tuple(params["end"]),
                        "duration": params.get("duration", 400),
                    })
                    step_ok = _parse_action_result(ok)
                elif atomic.startswith("swipe"):
                    sz = _device_size(state)
                    ok = screen_action.invoke({
                        "device": state["device"],
                        "action": atomic,
                        "x": sz["width"] // 2,
                        "y": sz["height"] // 2,
                        "direction": params.get("direction", "up"),
                    })
                    step_ok = _parse_action_result(ok)
                else:  # tap, long_press
                    if not eid:
                        _log("  ❌ Missing element_id")
                        step_ok = False
                    else:
                        state = capture_and_parse_screen(state)
                        matches = match_element_via_pinecone(eid, step, state)
                        if not matches:
                            _log(f"  ❌ Could not match element {eid} on screen")
                            step_ok = False
                        else:
                            step_ok = execute_element_action(state, matches[0])

                if not step_ok:
                    _log(f"  ❌ Step {step_num} execution failed")
                    step_failed = True
                    break

                state["replayed_steps"] = state.get("replayed_steps", 0) + 1
                state["history"].append({
                    "step": state.get("current_step", 0) + 1,
                    "action": atomic,
                    "element_id": eid,
                    "text": text,
                    "status": "success",
                    "screenshot": state["current_page"].get("screenshot"),
                })
                state["current_step"] = state.get("current_step", 0) + 1

            if step_failed:
                _log("⚠️ Replay segment failed — degrading once to ReAct for remaining task")
                state["plan"] = {
                    "action_id": None,
                    "confidence": 1.0,
                    "segments": [{"type": "react", "goal": state["task"]}],
                    "text_overrides": {},
                }
                state["seg_index"] = 0
                return execute_plan_node(state)

        elif stype == "react":
            goal = seg.get("goal") or state["task"]
            cap = seg.get("cap", MAX_REACT_STEPS_SEG)
            res = run_react_segment(state, goal, cap)
            _log(f"  [REACT] Segment result: {res}")
            if res == "stuck":
                state["execution_status"] = "stuck"
                state["finished"] = True
                return state

        state["seg_index"] = state.get("seg_index", 0) + 1

    state["execution_status"] = "steps_done"
    _log("✓ All plan segments executed. Proceeding to completion check.")
    return state


# ─────────────────────────────────────────────────────────────────────────────
#  LangGraph legacy node wrappers
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
    """Semantically match the user task to a stored Neo4j high-level action (legacy)."""
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
            _log("  [MATCH] ⚠️ element_sequence is empty — no steps to execute")
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
    """Pinecone-primary execution (legacy)."""
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
    print("\n⚠️ fallback_node entered")
    state = fallback_to_react(state)
    state["completed"] = False
    return state


def should_fallback(state: DeploymentState) -> str:
    _log = state.get("log_callback") or print
    result = "fallback" if state.get("should_fallback") else "continue"
    _log(f"  [ROUTE] should_fallback → '{result}'")
    return result


def is_task_completed(state: DeploymentState) -> str:
    _log = state.get("log_callback") or print
    if state.get("completed"):
        _log(f"  [ROUTE] is_task_completed → 'end' (status={state.get('execution_status')})")
        return "end"

    workflow_iter = state.get("workflow_iterations", 0) + 1
    state["workflow_iterations"] = workflow_iter
    max_iters = state.get("max_workflow_iterations", 10)
    _log(f"  [ROUTE] is_task_completed → 'continue' (iter={workflow_iter}/{max_iters})")

    if workflow_iter >= max_iters:
        _log(f"⚠️ Workflow iteration cap ({max_iters}) reached — ending")
        state["completed"] = True
        state["execution_status"] = "timeout"
        return "end"

    return "continue"


def after_judge(state: DeploymentState) -> str:
    """Route after completion check in the plan-reuse workflow (Step 16)."""
    _log = state.get("log_callback") or print
    if state.get("finished"):
        _log(f"  [ROUTE] after_judge → 'end' (status={state.get('execution_status')}, completed={state.get('completed')})")
        return "end"
    _log("  [ROUTE] after_judge → 'retry'")
    return "retry"


# ─────────────────────────────────────────────────────────────────────────────
#  LangGraph Workflows (v3 Plan Reuse & Legacy)
# ─────────────────────────────────────────────────────────────────────────────

def build_workflow_plan_reuse() -> StateGraph:
    """New plan-reuse graph (Step 16)."""
    workflow = StateGraph(DeploymentState)

    workflow.add_node("capture_screen",   capture_screen_node)
    workflow.add_node("plan_task",        plan_task)
    workflow.add_node("execute_plan",     execute_plan_node)
    workflow.add_node("check_completion", check_task_completion)

    workflow.set_entry_point("capture_screen")

    def after_initial_capture(state: DeploymentState) -> str:
        if not state.get("current_page", {}).get("screenshot"):
            return "end"
        return "plan"

    workflow.add_conditional_edges(
        "capture_screen",
        after_initial_capture,
        {"end": END, "plan": "plan_task"},
    )
    workflow.add_edge("plan_task", "execute_plan")
    workflow.add_edge("execute_plan", "check_completion")
    workflow.add_conditional_edges(
        "check_completion",
        after_judge,
        {"end": END, "retry": "execute_plan"},
    )
    return workflow


def build_workflow_legacy() -> StateGraph:
    """Legacy 5-node graph with fallback."""
    workflow = StateGraph(DeploymentState)

    workflow.add_node("capture_screen",   capture_screen_node)
    workflow.add_node("match_elements",   match_elements_node)
    workflow.add_node("execute_action",   execute_action_node)
    workflow.add_node("fallback",         fallback_node)
    workflow.add_node("check_completion", check_task_completion_legacy)

    workflow.set_entry_point("capture_screen")
    workflow.add_conditional_edges(
        "capture_screen", should_fallback,
        {"fallback": "fallback", "continue": "match_elements"},
    )
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


def build_workflow() -> StateGraph:
    """Master workflow router controlled by USE_PLAN_REUSE."""
    if USE_PLAN_REUSE:
        return build_workflow_plan_reuse()
    return build_workflow_legacy()


def run_task(
    task: str,
    device: str = "emulator-5554",
    max_workflow_iterations: int = 10,
    log_callback=None,
    force_fallback: bool = False,
) -> Dict[str, Any]:
    """
    Execute a high-level task on an Android device (Step 17).
    """
    if USE_PLAN_REUSE:
        return _run_verified_task(task, device, max_workflow_iterations, log_callback, force_fallback)

    _log = log_callback or print
    _log(f"\n{'#'*60}")
    _log(f"# deployment.py (Plan Reuse: {'ENABLED' if USE_PLAN_REUSE else 'DISABLED'})")
    _log(f"# task='{task}' device='{device}' max_iters={max_workflow_iterations} force_fallback={force_fallback}")
    _log(f"{'#'*60}\n")

    LLM_CALLS.clear()

    try:
        from data.State import create_deployment_state
        state = create_deployment_state(task=task, device=device, max_retries=3)
        state["workflow_iterations"]     = 0
        state["max_workflow_iterations"] = max_workflow_iterations
        state["log_callback"]            = log_callback
        state["close_actions"]           = []
        state["force_fallback"]          = force_fallback

        recursion_limit = max(50, max_workflow_iterations * 6)
        app    = build_workflow().compile()
        result = app.invoke(state, config={"recursion_limit": recursion_limit})

        close_actions = result.get("close_actions", [])

        if result.get("completed") and result.get("final_screenshot"):
            try:
                from PIL import Image
                Image.open(result["final_screenshot"]).show()
            except Exception:
                pass

        message = "Task execution completed"
        if result.get("execution_status") == "no_match":
            message = "Add test cases in the exploration tab or navigate to fallback mechanism"

        steps_done = result.get("replayed_steps", 0) + result.get("react_steps", 0)
        return {
            "status":          result.get("execution_status", "unknown"),
            "completed":       result.get("completed", False),
            "message":         message,
            "steps_completed": steps_done,
            "replayed_steps":  result.get("replayed_steps", 0),
            "react_steps":     result.get("react_steps", 0),
            "plan":            result.get("plan"),
            "llm_calls":       dict(LLM_CALLS),
            "close_actions":   close_actions,
        }
    except Exception as e:
        _log(f"❌ Error executing task: {e}")
        import traceback; traceback.print_exc()
        return {
            "status": "error",
            "completed": False,
            "message": str(e),
            "error": str(e),
            "steps_completed": 0,
            "replayed_steps": 0,
            "react_steps": 0,
            "plan": None,
            "llm_calls": dict(LLM_CALLS),
            "close_actions": [],
        }



def _run_verified_task(task, device, max_workflow_iterations=10, log_callback=None, force_fallback=False):
    """Default deployment path; legacy workflow remains opt-in via USE_PLAN_REUSE=0."""
    from replay_engine import ReplayEngine, hydrate_actions
    from deployment_progress import run_with_progress, timeout_setting
    _log = log_callback or print
    from deployment_artifacts import DeploymentImages
    run_images = DeploymentImages(_log)
    def parser_log(message):
        prefix = "[CLIENT] Image saved ? "
        if message.startswith(prefix):
            run_images.track(message[len(prefix):])
        _log(message)
    state = create_execution_state(device)
    state["task"] = task
    state["log_callback"] = log_callback

    _log(f"[DEPLOYMENT] Starting on device={device}; task={task!r}.")

    def capture():
        # Retain the fresh image for vision fallback even when parsing fails.
        page = {"screenshot": None, "elements_data": [], "elements_json": None, "parser_attempted": False}
        state["current_page"] = page
        _log(f"[CAPTURE] Taking screenshot; settling for {SCREENSHOT_SETTLE_SEC}s...")
        shot = take_screenshot.invoke({"device": device, "app_name": "deployment",
                                       "step": state["current_step"], "settle": SCREENSHOT_SETTLE_SEC})
        if not shot or not os.path.isfile(shot):
            _log(f"[CAPTURE] Screenshot failed: {shot}")
            return page
        page["screenshot"] = shot
        run_images.track(shot)
        _log(f"[CAPTURE] Saved screenshot: {shot}")
        try:
            page["parser_attempted"] = True
            _log("[PARSER] Submitting screenshot to OmniParser; waiting for remote worker...")
            path = run_with_progress("OmniParser", lambda: omniparser_run(_img_to_b64(shot), log_callback=parser_log),
                                     timeout=timeout_setting("DEPLOYMENT_PARSER_TIMEOUT_SEC", 125), log=_log)
            if path and os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    page.update(elements_data=data, elements_json=path)
                    _log(f"[PARSER] Ready: {len(data)} elements; JSON={path}")
                else:
                    _log("[PARSER] Unexpected JSON structure; keeping fresh image for vision fallback.")
            else:
                _log("[PARSER] No parsed JSON returned; keeping fresh image for vision fallback.")
        except Exception as exc:
            (log_callback or print)(f"Parsing unavailable; fresh screenshot retained for vision: {exc}")
        return page

    def action(params):
        params = dict(params)
        bbox = params.pop("bbox", None)
        replace = params.pop("replace", False)
        size = _device_size(state)
        if bbox:
            relative = max(bbox) <= 1
            params["x"] = int((bbox[0] + bbox[2]) / 2 * (size["width"] if relative else 1))
            params["y"] = int((bbox[1] + bbox[3]) / 2 * (size["height"] if relative else 1))
        if "x_relative" in params:
            params["x"] = int(params.pop("x_relative") * size["width"])
            params["y"] = int(params.pop("y_relative") * size["height"])
        params["device"] = device
        if params["action"] == "text":
            if "x" not in params or "y" not in params:
                return False
            params["replace"] = replace
        if params["action"].startswith("swipe") and params["action"] != "swipe_precise":
            params.setdefault("x", size["width"] // 2)
            params.setdefault("y", size["height"] // 2)
        _log(f"[ADB] Executing {params.get('action')} on {device}; coordinates=({params.get('x')}, {params.get('y')}).")
        raw_result = screen_action.invoke(params)
        result = _parse_action_result(raw_result)
        _log(f"[ADB] Result: {'success' if result else 'failure'}" + (f"; details={str(raw_result)[:500]}" if not result else ""))
        state["current_step"] += 1
        return result

    def model(kind, system, prompt, page):
        images = None
        if page is not None:
            b64 = _img_to_b64(page.get("screenshot"))
            if not b64:
                raise RuntimeError("Vision escalation requires a fresh screenshot")
            images = [b64]
        # Keep the exact prompt locally, referencing the original image without duplicating base64.
        from pathlib import Path
        import hashlib
        import time
        trace_dir = Path("log/model_requests")
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{time.time_ns()}_{kind}.json"
        shot_path = page.get("screenshot") if page else None
        trace = {"kind": kind, "system_prompt": system, "user_prompt": prompt,
                 "image_path": str(Path(shot_path).resolve()) if shot_path else None}
        if shot_path:
            raw = Path(shot_path).read_bytes()
            trace.update(image_sha256=hashlib.sha256(raw).hexdigest(), image_bytes=len(raw),
                         image_mime="image/png" if raw.startswith(b"\x89PNG") else "image/jpeg")
        trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(f"[MODEL-TRACE] Exact system/user prompt and image reference saved: {trace_path.resolve()}")
        # Metrics are owned by this engine instance, not the legacy global counter.
        timeout = timeout_setting("NVIDIA_REQUEST_TIMEOUT_SEC", 200)
        _log(f"[MODEL-REQUEST] kind={kind}, image attached={bool(images)}, prompt characters={len(prompt)}, HTTP timeout={timeout}s; SDK retries disabled.")
        return run_with_progress(f"NVIDIA/{kind}",
            lambda: asyncio.run(bridge.call_json(system_prompt=system, user_prompt=prompt, images_b64=images, timeout=timeout)),
            timeout=timeout+5, log=_log)

    def load():
        _log("[DATABASE] Fetching high-level actions from Neo4j...")
        actions = run_with_progress("Neo4j/action catalog", lambda: db.get_all_high_level_actions(log_callback=_log),
                                    timeout=timeout_setting("DEPLOYMENT_DB_TIMEOUT_SEC", 30), log=_log)
        from replay_engine import decoded
        ids = {s.get("element_id") for a in actions for s in decoded(a.get("element_sequence"), []) if s.get("element_id") and not all(s.get(k) for k in ("source", "destination", "target"))}
        _log(f"[DATABASE] Found {len(actions)} stored task(s); {len(ids)} step(s) need legacy screen metadata.")
        metadata = run_with_progress("Neo4j/step metadata", lambda: db.get_replay_metadata(list(ids)),
                                     timeout=timeout_setting("DEPLOYMENT_DB_TIMEOUT_SEC", 30), log=_log) if ids else {}
        _log(f"[DATABASE] Loaded metadata for {len(metadata)} element(s); building replay catalog.")
        return hydrate_actions(actions, metadata)

    engine = ReplayEngine(capture, action, lambda: press_home(device), model, load,
                          max_steps=max(1, max_workflow_iterations * 3), log=log_callback or print)
    result = engine.run(task, force_fallback)
    result["image_cleanup"] = run_images.finish(result.get("completed") is True)
    (log_callback or print)(f"Deployment result: {result['status']}; metrics={result['metrics']}")
    return result
