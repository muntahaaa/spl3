"""
deployment.py  (Firebase/Qwen edition)
---------------------------------------
Replaces all direct Gemini (ChatGoogleGenerativeAI) calls with
Firebase Realtime DB round-trips handled by the Colab Qwen2.5-VL
worker notebook.

All public APIs and the LangGraph workflow structure are preserved.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

import config
from data.State import DeploymentState, ElementMatch
from data.graph_db import Neo4jDatabase
from data.vector_db import VectorStore
from tool.img_tool import *
from tool.adb_tools import *
from OmniParser.client import run as omniparser_run
from firebase_llm_bridge import FirebaseLLMBridge

# ── LangSmith tracing ────────────────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true" if config.LANGCHAIN_TRACING_V2 else "false"
os.environ["LANGCHAIN_ENDPOINT"]   = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"]    = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"]    = "DeploymentExecution"

# ── Firebase bridge (replaces ChatGoogleGenerativeAI) ────────────────────────
bridge = FirebaseLLMBridge(
    firebase_url=config.CHAIN_FIREBASE_URL,
    firebase_secret=config.CHAIN_FIREBASE_SECRET,
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
    """Run an async bridge call from synchronous code."""
    return asyncio.get_event_loop().run_until_complete(
        bridge.call_text(system_prompt=system_prompt, user_prompt=user_prompt, timeout=timeout)
    )


def _sync_call_json(
    system_prompt: str,
    user_prompt: str,
    images_b64: Optional[List[str]] = None,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    return asyncio.get_event_loop().run_until_complete(
        bridge.call_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images_b64=images_b64,
            timeout=timeout,
        )
    )


def _sync_call_vision(
    system_prompt: str,
    user_prompt: str,
    images_b64: List[str],
    timeout: float = 300.0,
) -> str:
    return asyncio.get_event_loop().run_until_complete(
        bridge.call_vision(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images_b64=images_b64,
            timeout=timeout,
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
#  Task → high-level action matching
# ─────────────────────────────────────────────────────────────────────────────

_MATCH_SYSTEM = (
    "You are an AI assistant specialized in matching user tasks with predefined "
    "high-level actions. Analyse the task description and decide if it matches any "
    "predefined high-level action. Only consider a match when the degree is above 0.7.\n\n"
    "If matched, reply with:\n  MATCHED: <complete JSON of the best matching action>\n"
    "If not matched, reply with:\n  NO_MATCH <brief explanation>"
)


def match_task_to_action(
    state: Dict[str, Any], task: str
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Match user task with high-level action nodes via the Qwen bridge."""
    print(f"Matching task: {task}")

    high_level_actions = db.get_all_high_level_actions()
    if not high_level_actions:
        print("❌ No high-level action nodes found")
        return False, None

    print(f"Found {len(high_level_actions)} high-level action nodes")

    actions_json = json.dumps(high_level_actions, ensure_ascii=False, indent=2)
    user_prompt  = (
        f"User task: {task}\n\n"
        f"Available high-level actions:\n{actions_json}\n\n"
        "Determine if the task matches any high-level action and reply as instructed."
    )

    try:
        result = _sync_call_text(_MATCH_SYSTEM, user_prompt, timeout=180)

        if result.startswith("MATCHED:"):
            action_json_str = result[len("MATCHED:"):].strip()
            try:
                matched_action = json.loads(action_json_str)
                print(f"✓ Matched: {matched_action.get('name','?')} (ID: {matched_action.get('action_id','?')})")
                return True, matched_action
            except json.JSONDecodeError:
                print(f"❌ Cannot parse matching result: {action_json_str}")
                return False, None
        elif result.startswith("NO_MATCH"):
            print(f"❌ No matching high-level action found: {result[len('NO_MATCH'):].strip()}")
            return False, None
        else:
            print(f"❌ Unrecognised matching result: {result}")
            return False, None

    except Exception as e:
        print(f"❌ Error during task matching: {e}")
        return False, None


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


# ─────────────────────────────────────────────────────────────────────────────
#  Element matching  (visual then semantic fallback)
# ─────────────────────────────────────────────────────────────────────────────

def match_screen_elements(
    state: DeploymentState, action_sequence: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not state["current_page"]["elements_data"]:
        print("❌ Current screen element data is empty")
        return []

    current_step_idx = state["current_step"]
    if current_step_idx >= len(action_sequence):
        return []

    current_action = action_sequence[current_step_idx]
    element_id     = current_action.get("element_id")
    if not element_id:
        return []

    db_element = db.get_action_by_id(element_id) or db.get_element_by_id(element_id)
    if not db_element:
        return []

    if "action_id" in db_element and not any(
        k in db_element for k in ["visual_embedding", "screenshot_path"]
    ):
        return fallback_to_semantic_match(state, action_sequence)

    template_embedding = None
    if "visual_embedding" in db_element and db_element["visual_embedding"]:
        template_embedding = db_element["visual_embedding"]
    else:
        if "screenshot_path" in db_element and db_element["screenshot_path"]:
            try:
                template_embedding = extract_features(db_element["screenshot_path"], "resnet50")["features"]
            except Exception as e:
                print(f"❌ Cannot extract template element features: {e}")
                return fallback_to_semantic_match(state, action_sequence)
        else:
            return fallback_to_semantic_match(state, action_sequence)

    screen_elements   = state["current_page"]["elements_data"]
    screenshot_path   = state["current_page"]["screenshot"]

    try:
        import numpy as np
        element_embeddings = []
        for idx, element in enumerate(screen_elements):
            try:
                element_img_stream = elements_img(screenshot_path, json.dumps(screen_elements), element.get("ID", idx))
                feat = extract_features(element_img_stream, "resnet50")
                element_embeddings.append((idx, feat["features"]))
            except Exception:
                continue

        if not element_embeddings:
            return fallback_to_semantic_match(state, action_sequence)

        matches = []
        tmpl_vec = np.array(template_embedding).flatten()
        tmpl_norm = np.linalg.norm(tmpl_vec)
        for idx, embedding in element_embeddings:
            elem_vec  = np.array(embedding).flatten()
            elem_norm = np.linalg.norm(elem_vec)
            similarity = 0.0 if (tmpl_norm == 0 or elem_norm == 0) else float(np.dot(tmpl_vec, elem_vec) / (tmpl_norm * elem_norm))
            if similarity >= 0.6:
                matches.append({
                    "element_id":       element_id,
                    "match_score":      similarity,
                    "screen_element_id": idx,
                    "action_type":      current_action.get("atomic_action", "tap"),
                    "parameters":       current_action.get("action_params", {}),
                })

        matches.sort(key=lambda x: x["match_score"], reverse=True)
        if matches:
            print(f"✓ Matched screen element {matches[0]['screen_element_id']} (score={matches[0]['match_score']:.2f})")
        return matches

    except Exception as e:
        print(f"❌ Error during visual matching: {e}")
        return fallback_to_semantic_match(state, action_sequence)


# ─────────────────────────────────────────────────────────────────────────────
#  Semantic fallback element matching via Qwen
# ─────────────────────────────────────────────────────────────────────────────

_ELEM_MATCH_SYSTEM = (
    "You are an AI assistant specialized in matching UI elements. "
    "Analyse the template element description and current screen elements, "
    "then find the best match.\n\n"
    "Return ONLY a JSON object with these keys:\n"
    "  element_id (str), match_score (float 0-1), screen_element_id (int),\n"
    "  action_type (str: tap/text/swipe/long_press/back), parameters (dict)\n"
    "If no element scores above 0.6, set match_score=0 and screen_element_id=-1.\n"
    "No preamble, no markdown fences."
)


def fallback_to_semantic_match(
    state: DeploymentState, action_sequence: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    print("🔄 Falling back to semantic matching...")

    current_step_idx = state["current_step"]
    if current_step_idx >= len(action_sequence):
        return []

    step_info  = action_sequence[current_step_idx]
    element_id = step_info.get("element_id")
    if not element_id:
        return []

    db_element = db.get_action_by_id(element_id) or db.get_element_by_id(element_id)
    if not db_element:
        return []

    # Build template description
    id_field = "element_id" if "element_id" in db_element else "action_id"
    tmpl_desc = f"ID: {db_element.get(id_field, 'unknown')}\n"
    if db_element.get("description"):
        tmpl_desc += f"Description: {db_element['description']}\n"
    elif db_element.get("name"):
        tmpl_desc += f"Name: {db_element['name']}\n"

    for field in ["bounding_box", "bbox", "position"]:
        if db_element.get(field):
            bbox = db_element[field]
            if isinstance(bbox, list) and len(bbox) >= 4:
                tmpl_desc += f"Position: [{bbox[0]:.3f},{bbox[1]:.3f},{bbox[2]:.3f},{bbox[3]:.3f}]\n"
            elif isinstance(bbox, str):
                tmpl_desc += f"Position: {bbox}\n"
            break

    tmpl_desc += f"Action type: {step_info.get('atomic_action','tap')}\n"
    params = step_info.get("action_params", {})
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except Exception:
            pass
    if isinstance(params, dict) and params:
        tmpl_desc += "Action parameters:\n" + "\n".join(f"  {k}: {v}" for k, v in params.items())

    # Build screen description
    screen_desc = ""
    for i, element in enumerate(state["current_page"]["elements_data"]):
        screen_desc += f"Element {i} (ID: {element.get('ID', i)}):\n"
        for key in ("type", "content"):
            if element.get(key):
                screen_desc += f"  {key.capitalize()}: {element[key]}\n"
        if "bbox" in element:
            b = element["bbox"]
            screen_desc += f"  Position: [{b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f}]\n"
        screen_desc += "\n"

    user_prompt = (
        f"Template element:\n{tmpl_desc}\n\n"
        f"Current screen elements:\n{screen_desc}\n\n"
        "Find the best match and return JSON."
    )

    try:
        result = _sync_call_json(_ELEM_MATCH_SYSTEM, user_prompt, timeout=180)

        # result may be a raw dict or wrapped in ElementMatch shape
        match_score       = float(result.get("match_score", 0))
        screen_element_id = int(result.get("screen_element_id", -1))

        if match_score >= 0.6 and screen_element_id >= 0:
            print(f"✓ Semantic match — element {screen_element_id} (score={match_score:.2f})")
            return [{
                "element_id":        result.get("element_id", element_id),
                "match_score":       match_score,
                "screen_element_id": screen_element_id,
                "action_type":       result.get("action_type", "tap"),
                "parameters":        result.get("parameters", {}),
            }]
        else:
            print("❌ No matching screen element found via semantic match")
            return []

    except Exception as e:
        print(f"❌ Error during semantic element matching: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  Execute element action  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

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

        center_x = int((bbox[0] + bbox[2]) / 2 * device_size["width"])
        center_y = int((bbox[1] + bbox[3]) / 2 * device_size["height"])

        action_params = {"device": state["device"], "action": action_type, "x": center_x, "y": center_y}
        if action_type == "text":
            action_params["input_str"] = parameters.get("text", "")
        elif action_type == "long_press":
            action_params["duration"] = parameters.get("duration", 1000)
        elif action_type == "swipe":
            action_params["direction"] = parameters.get("direction", "up")
            action_params["dist"]      = parameters.get("distance", "medium")

        print(f"Executing action: {action_type} at ({center_x}, {center_y})")
        result = screen_action.invoke(action_params)

        if isinstance(result, str):
            try:
                result_json = json.loads(result)
                if result_json.get("status") == "success":
                    print("✓ Action executed successfully")
                    return True
                else:
                    print(f"❌ Action failed: {result_json.get('message','unknown')}")
                    return False
            except Exception:
                print(f"❌ Cannot parse operation result: {result}")
                return False
        return False

    except Exception as e:
        print(f"❌ Error executing element action: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  React fallback  (uses Qwen for decision, ADB tool for execution)
# ─────────────────────────────────────────────────────────────────────────────

_REACT_SYSTEM = (
    "You are an intelligent smartphone operation assistant. "
    "Observe the current screen and perform one atomic operation "
    "(tap / type text / swipe / long press / back) to progress toward the user's goal. "
    "Reply with a JSON object:\n"
    "  {\"action\": \"<type>\", \"x\": <int>, \"y\": <int>, "
    "\"input_str\": \"<text if action=text>\", "
    "\"direction\": \"<up/down/left/right if action=swipe>\", "
    "\"duration\": <ms if action=long_press>}\n"
    "For back action omit x/y. Return JSON only."
)


def fallback_to_react(state: DeploymentState) -> DeploymentState:
    print("🔄 Falling back to React mode execution...")
    task = state["task"]

    state = capture_and_parse_screen(state)
    if not state["current_page"]["screenshot"]:
        state["execution_status"] = "error"
        print("Unable to capture or parse screen")
        return state

    screenshot_path   = state["current_page"]["screenshot"]
    elements_json_path = state["current_page"]["elements_json"]
    device            = state["device"]
    device_size       = get_device_size.invoke(device)

    with open(elements_json_path, "r", encoding="utf-8") as f:
        elements_data = json.load(f)

    img_b64 = _img_to_b64(screenshot_path)
    images  = [img_b64] if img_b64 else []

    user_prompt = (
        f"Device: {device}  Size: {device_size}\n"
        f"Task: {task}\n\n"
        f"Current screen elements (bbox values are relative 0-1):\n"
        f"{json.dumps(elements_data, ensure_ascii=False, indent=2)}\n\n"
        "Determine the next single operation and return JSON."
    )

    try:
        result_json = _sync_call_json(
            system_prompt=_REACT_SYSTEM,
            user_prompt=user_prompt,
            images_b64=images,
            timeout=240,
        )

        action_type = result_json.get("action", "tap")
        action_params: Dict[str, Any] = {"device": device, "action": action_type}
        if action_type != "back":
            action_params["x"] = int(result_json.get("x", 0))
            action_params["y"] = int(result_json.get("y", 0))
        if action_type == "text":
            action_params["input_str"] = result_json.get("input_str", "")
        elif action_type == "long_press":
            action_params["duration"] = int(result_json.get("duration", 1000))
        elif action_type == "swipe":
            action_params["direction"] = result_json.get("direction", "up")
            action_params["dist"]      = result_json.get("dist", "medium")

        screen_action.invoke(action_params)
        state["current_step"] += 1
        state["history"].append({
            "step":      state["current_step"],
            "screenshot": screenshot_path,
            "action":    action_type,
            "params":    action_params,
            "status":    "success",
        })
        state["execution_status"] = "success"
        print(f"✓ React mode: executed {action_type}")

    except Exception as e:
        print(f"❌ React mode error: {e}")
        state["history"].append({
            "step":   state["current_step"],
            "action": "react_mode",
            "status": "error",
            "error":  str(e),
        })
        state["execution_status"] = "error"

    return state


# ─────────────────────────────────────────────────────────────────────────────
#  execute_task  (top-level, unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────

def execute_task(
    state: DeploymentState, task: str, device: str, neo4j_db: Neo4jDatabase = None
) -> Dict[str, Any]:
    from data.State import create_deployment_state
    state    = create_deployment_state(task=task, device=device)
    neo4j_db = neo4j_db or db

    all_elements = neo4j_db.get_all_actions()
    if not all_elements:
        print("⚠️ No element nodes in database, falling back to React mode")
        state = fallback_to_react(state)
        return {"status": state["execution_status"], "state": state}

    high_level_actions = neo4j_db.get_high_level_actions_for_task(task)
    if high_level_actions:
        shortcuts = check_shortcut_associations(state, high_level_actions)
        if shortcuts:
            valid_shortcuts = evaluate_shortcut_execution(state, shortcuts)
            if valid_shortcuts:
                execution_template = generate_execution_template(state, valid_shortcuts)
                if execution_template:
                    prioritized = prioritize_shortcuts(state, valid_shortcuts)
                    result = execute_high_level_action(state, prioritized, execution_template)
                    if result.get("status") == "success":
                        state["execution_status"] = "success"
                        state["completed"]         = True
                        return {"status": "success", "state": state}
                    else:
                        state = fallback_to_react(state)
                        return {"status": state["execution_status"], "state": state}
        else:
            for action in high_level_actions:
                action_sequence = action.get("action_sequence", [])
                if not action_sequence:
                    continue
                state = capture_and_parse_screen(state)
                if not state["current_page"]["screenshot"]:
                    state["retry_count"] += 1
                    if state["retry_count"] >= state["max_retries"]:
                        state = fallback_to_react(state)
                        return {"status": state["execution_status"], "state": state}
                    continue
                state["retry_count"] = 0
                element_matches = match_screen_elements(state, action_sequence)
                if not element_matches:
                    state["retry_count"] += 1
                    if state["retry_count"] >= state["max_retries"]:
                        state = fallback_to_react(state)
                        return {"status": state["execution_status"], "state": state}
                    continue
                state["retry_count"] = 0
                best_match = element_matches[0]
                success = execute_element_action(state, best_match)
                if success:
                    state["current_step"] += 1
                    state["history"].append({
                        "step": state["current_step"],
                        "screenshot": state["current_page"]["screenshot"],
                        "action": best_match.get("action_type", "tap"),
                        "element_id": best_match.get("element_id", ""),
                        "status": "success",
                    })
                    if state["current_step"] >= len(action_sequence):
                        state["execution_status"] = "success"
                        state["completed"]         = True
                        return {"status": "success", "state": state}
                else:
                    state["retry_count"] += 1
                    if state["retry_count"] >= state["max_retries"]:
                        state = fallback_to_react(state)
                        return {"status": state["execution_status"], "state": state}
    else:
        print("❌ No matching high-level actions found, falling back to React mode")
        state = fallback_to_react(state)
        return {"status": state["execution_status"], "state": state}

    state = fallback_to_react(state)
    return {"status": state["execution_status"], "state": state}


# ─────────────────────────────────────────────────────────────────────────────
#  Shortcut helpers  (all Gemini prompts replaced with bridge calls)
# ─────────────────────────────────────────────────────────────────────────────

def check_shortcut_associations(
    state: DeploymentState, high_level_actions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    print("🔍 Checking high-level action shortcut associations...")
    shortcuts: List[Dict[str, Any]] = []
    for action in high_level_actions:
        action_id = action.get("action_id")
        if not action_id:
            continue
        associated = state.get("neo4j_db", db).get_shortcuts_for_action(action_id)
        for shortcut in (associated or []):
            shortcuts.append({
                "shortcut_id":    shortcut.get("shortcut_id"),
                "name":           shortcut.get("name"),
                "description":    shortcut.get("description"),
                "action_id":      action_id,
                "action_name":    action.get("name"),
                "action_sequence": action.get("action_sequence", []),
                "conditions":     shortcut.get("conditions", {}),
                "priority":       shortcut.get("priority", 0),
                "page_flow":      shortcut.get("page_flow", []),
            })
    print(f"{'✓' if shortcuts else '⚠️'} Found {len(shortcuts)} associated shortcut(s)")
    return shortcuts


_SHORTCUT_EVAL_SYSTEM = (
    "You are a smartphone operation assistant that evaluates whether the current "
    "scenario meets the conditions for executing shortcuts.\n\n"
    "Return ONLY a JSON object:\n"
    "  {\"valid_shortcuts\": [{\"shortcut_id\": \"...\", \"reason\": \"...\", \"confidence\": 0.0}]}\n"
    "If no shortcuts match, return an empty list. No preamble, no markdown fences."
)


def evaluate_shortcut_execution(
    state: DeploymentState, shortcuts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    print("🧠 Evaluating shortcut execution conditions...")
    if not shortcuts:
        return []

    screen_desc = ""
    if state["current_page"]["elements_data"]:
        screen_desc = "Current screen elements:\n"
        for i, el in enumerate(state["current_page"]["elements_data"]):
            screen_desc += f"  {i+1}. Type: {el.get('type','?')}  Content: {el.get('content','')}\n"

    shortcuts_info = ""
    for i, sc in enumerate(shortcuts):
        shortcuts_info += f"{i+1}. ID: {sc.get('shortcut_id')}  Name: {sc.get('name')}\n"
        shortcuts_info += f"   Description: {sc.get('description','N/A')}\n"
        cond = sc.get("conditions", {})
        if cond:
            if isinstance(cond, dict):
                shortcuts_info += "   Conditions: " + "; ".join(f"{k}={v}" for k, v in cond.items()) + "\n"
            else:
                shortcuts_info += f"   Conditions: {cond}\n"
        shortcuts_info += "\n"

    user_prompt = (
        f"User task: {state['task']}\n\n"
        f"{screen_desc}\n"
        f"Available shortcuts:\n{shortcuts_info}\n"
        "Evaluate which shortcuts meet execution conditions and return JSON."
    )

    try:
        result        = _sync_call_json(_SHORTCUT_EVAL_SYSTEM, user_prompt, timeout=180)
        valid_list    = result.get("valid_shortcuts", [])
        valid_ids     = {v["shortcut_id"] for v in valid_list}
        valid_objects = []
        for sc in shortcuts:
            if sc.get("shortcut_id") in valid_ids:
                sc_copy = sc.copy()
                for v in valid_list:
                    if v["shortcut_id"] == sc.get("shortcut_id"):
                        sc_copy["evaluation"] = {"reason": v.get("reason",""), "confidence": v.get("confidence", 0.0)}
                        break
                valid_objects.append(sc_copy)
        print(f"✓ {len(valid_objects)} shortcut(s) meet execution conditions")
        return valid_objects
    except Exception as e:
        print(f"❌ Error evaluating shortcuts: {e}")
        return []


_TEMPLATE_GEN_SYSTEM = (
    "You are a smartphone operation assistant that generates detailed execution "
    "templates from shortcuts and the current screen state.\n\n"
    "Return ONLY a JSON object:\n"
    "  {\"steps\": [{\"action_type\": \"tap|text|swipe|long_press|back\", "
    "\"target_element_id\": <int or null>, \"parameters\": {...}}]}\n"
    "No preamble, no markdown fences."
)


def generate_execution_template(
    state: DeploymentState, shortcuts: List[Dict[str, Any]]
) -> Dict[str, Any]:
    print("📝 Generating execution template...")
    if not shortcuts:
        return {}

    selected = max(shortcuts, key=lambda x: x.get("evaluation", {}).get("confidence", 0))

    device_size = get_device_size.invoke(state["device"])
    if isinstance(device_size, str):
        device_size = {"width": 1080, "height": 1920}

    shortcut_info = (
        f"ID: {selected.get('shortcut_id')}\n"
        f"Name: {selected.get('name')}\n"
        f"Description: {selected.get('description','N/A')}\n"
    )
    seq = selected.get("action_sequence", [])
    if seq:
        shortcut_info += "Action sequence:\n" + "\n".join(
            f"  {i+1}. {json.dumps(a, ensure_ascii=False)}" for i, a in enumerate(seq)
        )
    eval_ = selected.get("evaluation", {})
    if eval_:
        shortcut_info += f"\nReason: {eval_.get('reason','')}  Confidence: {eval_.get('confidence',0)}"

    user_prompt = (
        f"Shortcut:\n{shortcut_info}\n\n"
        f"Screen elements:\n{json.dumps(state['current_page']['elements_data'], ensure_ascii=False, indent=2)}\n\n"
        f"Device size: {json.dumps(device_size, ensure_ascii=False)}\n\n"
        "Generate the execution template JSON now."
    )

    try:
        result = _sync_call_json(_TEMPLATE_GEN_SYSTEM, user_prompt, timeout=240)
        if "steps" in result and isinstance(result["steps"], list) and result["steps"]:
            print(f"✓ Generated template with {len(result['steps'])} steps")
            return result
        print("❌ Generated template is invalid")
        return {}
    except Exception as e:
        print(f"❌ Error generating template: {e}")
        return {}


def prioritize_shortcuts(
    state: Dict[str, Any], shortcuts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not shortcuts or len(shortcuts) <= 1:
        return shortcuts
    try:
        page_flow = state.get("neo4j_db", db).get_page_flow()
        if not page_flow:
            return sorted(shortcuts, key=lambda x: x.get("element_match", {}).get("match_score", 0), reverse=True)
        prioritized = []
        for sc in shortcuts:
            pos = next((i for i, n in enumerate(page_flow) if n.get("shortcut_id") == sc.get("shortcut_id")), -1)
            prioritized.append({"shortcut": sc, "pos": pos, "score": sc.get("element_match", {}).get("match_score", 0)})
        prioritized.sort(key=lambda x: (x["pos"] if x["pos"] >= 0 else float("inf"), -x["score"]))
        return [p["shortcut"] for p in prioritized]
    except Exception as e:
        print(f"⚠️ Shortcut prioritisation failed: {e}")
        return shortcuts


def execute_high_level_action(
    state: DeploymentState,
    shortcuts: List[Dict[str, Any]],
    execution_template: Dict[str, Any],
) -> Dict[str, Any]:
    print("🚀 Executing high-level operations...")

    if not execution_template or "steps" not in execution_template:
        return {"status": "error", "message": "Invalid execution template"}

    steps = execution_template["steps"]
    if not steps:
        return {"status": "error", "message": "No steps in execution template"}

    state["current_step"]    = 0
    state["total_steps"]     = len(steps)
    state["execution_status"] = "running"
    state["history"]         = []

    shortcut_names = ", ".join(s.get("name", "Unnamed") for s in shortcuts)
    print(f"Executing shortcuts: {shortcut_names}  ({len(steps)} steps)")

    while state["current_step"] < state["total_steps"]:
        idx  = state["current_step"]
        step = steps[idx]
        print(f"\nStep {idx + 1}/{state['total_steps']}")

        state = capture_and_parse_screen(state)
        if not state["current_page"]["screenshot"]:
            state["retry_count"] += 1
            if state["retry_count"] >= state["max_retries"]:
                return {"status": "error", "message": "Unable to capture screen"}
            time.sleep(1)
            continue
        state["retry_count"] = 0

        action_type       = step.get("action_type", "tap")
        target_element_id = step.get("target_element_id")
        parameters        = step.get("parameters", {})

        if action_type == "back":
            screen_action.invoke({"device": state["device"], "action": "back"})
            state["history"].append({"step": idx, "action": "back", "status": "success"})
            state["current_step"] += 1
            time.sleep(1)
            continue

        if target_element_id is None:
            return {"status": "error", "message": f"Step {idx+1} missing target_element_id"}

        screen_elements = state["current_page"]["elements_data"]
        if target_element_id < 0 or target_element_id >= len(screen_elements):
            return {"status": "error", "message": f"Invalid target_element_id for step {idx+1}"}

        element     = screen_elements[target_element_id]
        bbox        = element.get("bbox", [0, 0, 0, 0])
        device_size = get_device_size.invoke(state["device"])
        if isinstance(device_size, str):
            device_size = {"width": 1080, "height": 1920}

        center_x = int((bbox[0] + bbox[2]) / 2 * device_size["width"])
        center_y = int((bbox[1] + bbox[3]) / 2 * device_size["height"])

        action_params: Dict[str, Any] = {"device": state["device"], "action": action_type, "x": center_x, "y": center_y}
        if action_type == "text":
            action_params["input_str"] = parameters.get("text", "")
        elif action_type == "long_press":
            action_params["duration"] = parameters.get("duration", 1000)
        elif action_type == "swipe":
            action_params["direction"] = parameters.get("direction", "up")
            action_params["dist"]      = parameters.get("distance", "medium")

        result  = screen_action.invoke(action_params)
        success = False
        if isinstance(result, str):
            try:
                success = json.loads(result).get("status") == "success"
            except Exception:
                pass

        if success:
            print(f"✓ Step {idx + 1} OK")
            state["history"].append({"step": idx, "action": action_type, "target": target_element_id, "status": "success"})
            state["current_step"] += 1
            state["retry_count"]   = 0
            time.sleep(1.5)
        else:
            print(f"❌ Step {idx + 1} failed")
            state["history"].append({"step": idx, "action": action_type, "target": target_element_id, "status": "error"})
            state["retry_count"] += 1
            if state["retry_count"] >= state["max_retries"]:
                return {"status": "error", "message": f"Step {idx+1} failed after {state['max_retries']} retries"}
            time.sleep(1)

    state = capture_and_parse_screen(state)
    state["execution_status"] = "success"
    state["completed"]         = True
    print(f"\n✨ High-level execution complete — {state['current_step']} steps done")
    return {
        "status":            "success",
        "message":           "Successfully executed high-level operations",
        "steps_completed":   state["current_step"],
        "total_steps":       state["total_steps"],
        "final_screenshot":  state["current_page"]["screenshot"],
        "execution_history": state["history"],
    }


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
    if state["current_step"] < 2:
        return state

    print("🔍 Evaluating if task is completed...")
    task = state["task"]

    # Generate criteria
    try:
        completion_criteria = _sync_call_text(
            system_prompt=_CRITERIA_SYSTEM,
            user_prompt=f"The user's task is: {task}\nDescribe clear, checkable completion criteria.",
            timeout=120,
        )
    except Exception as e:
        print(f"⚠️ Could not generate criteria: {e}")
        return state

    # Collect recent screenshots
    recent_screenshots: List[str] = [
        step["screenshot"] for step in state["history"][-3:] if step.get("screenshot")
    ]
    if not recent_screenshots and state["current_page"]["screenshot"]:
        recent_screenshots = [state["current_page"]["screenshot"]]
    if not recent_screenshots:
        print("⚠️ No screenshots available")
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
        print(f"⚠️ Completion check error: {e}")
        return state

    if "yes" in answer or "complete" in answer:
        state["completed"]        = True
        state["execution_status"] = "completed"
        print(f"✓ Task completed: {answer}")
    else:
        state["completed"] = False
        print(f"⚠️ Task not completed: {answer}")

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
    print("📸 Capturing and parsing current screen...")
    state_dict   = dict(state)
    updated      = capture_and_parse_screen(state_dict)
    for k, v in updated.items():
        if k in state:
            state[k] = v
    if not state["current_page"]["screenshot"]:
        state["should_fallback"] = True
        print("❌ Unable to capture screen, marking for fallback")
    else:
        print("✓ Screen captured successfully")
    return state


def match_elements_node(state: DeploymentState) -> DeploymentState:
    print("🔍 Matching current screen elements...")
    all_elements = db.get_all_actions()
    if not all_elements:
        state["should_fallback"] = True
        return state

    action_sequence: List[Dict[str, Any]] = []
    for element in all_elements:
        if "element_id" in element:
            action_sequence.append({"element_id": element["element_id"], "atomic_action": "tap", "action_params": {}})
        else:
            element_id = element.get("id") or element.get("node_id") or str(hash(str(element)))
            action_sequence.append({"element_id": element_id, "atomic_action": "tap", "action_params": {}})

    state_dict = dict(state)
    element_matches = match_screen_elements(state_dict, action_sequence)
    state["matched_elements"] = element_matches

    if not state["matched_elements"]:
        is_matched, matched_action = match_task_to_action(state_dict, state["task"])
        if is_matched and matched_action:
            state["current_action"] = matched_action
            element_sequence = matched_action.get("element_sequence", [])
            if isinstance(element_sequence, str):
                try:
                    element_sequence = json.loads(element_sequence)
                except Exception:
                    state["should_fallback"] = True
                    return state
            if not element_sequence:
                state["should_fallback"] = True
                return state
            state["current_step"]  = 0
            state["total_steps"]   = len(element_sequence)
            updated = capture_and_parse_screen(state_dict)
            for k, v in updated.items():
                if k in state:
                    state[k] = v
            state["matched_elements"] = match_screen_elements(state_dict, element_sequence)
            if not state["matched_elements"]:
                state["should_fallback"] = True
        else:
            state["should_fallback"] = True
    else:
        print(f"✓ Found {len(state['matched_elements'])} matching elements")
    return state


def check_shortcuts_node(state: DeploymentState) -> DeploymentState:
    print("🔍 Checking element associations with shortcuts...")
    if not state["matched_elements"]:
        state["should_fallback"] = True
        return state
    state_dict = dict(state)
    associated = check_shortcut_associations(state_dict, state["matched_elements"])
    if associated:
        state["associated_shortcuts"] = prioritize_shortcuts(state_dict, associated)
        print(f"✓ Found {len(state['associated_shortcuts'])} associated shortcut node(s)")
    else:
        print("📝 No associated shortcut nodes found")
        state["associated_shortcuts"] = []
    return state


def shortcut_evaluation_node(state: DeploymentState) -> DeploymentState:
    print("🧠 Evaluating whether to execute shortcut operations...")
    if not state["associated_shortcuts"]:
        return state
    state_dict = dict(state)
    # evaluate_shortcut_execution returns a list, not a dict — handle both shapes
    result = evaluate_shortcut_execution(state_dict, state["associated_shortcuts"])
    if isinstance(result, list) and result:
        state["should_execute_shortcut"] = True
        state["current_shortcut"]        = result[0]
    elif isinstance(result, dict):
        state["should_execute_shortcut"] = result.get("should_execute", False)
        if "shortcut" in result:
            state["current_shortcut"] = result["shortcut"]
    print(f"{'✓' if state.get('should_execute_shortcut') else '⚠️'} Execute shortcut: {state.get('should_execute_shortcut')}")
    return state


def generate_template_node(state: DeploymentState) -> DeploymentState:
    print("📝 Generating execution template...")
    if not state.get("should_execute_shortcut") or "current_shortcut" not in state:
        return state
    state_dict = dict(state)
    shortcuts  = [state["current_shortcut"]] if isinstance(state["current_shortcut"], dict) else state["associated_shortcuts"]
    state["execution_template"] = generate_execution_template(state_dict, shortcuts)
    steps = len(state.get("execution_template", {}).get("steps", []))
    print(f"✓ Template generated with {steps} step(s)")
    return state


def execute_action_node(state: DeploymentState) -> DeploymentState:
    state_dict = dict(state)
    if state.get("should_execute_shortcut") and state.get("execution_template"):
        print("🚀 Executing high-level operation...")
        result = execute_high_level_action(state_dict, state["associated_shortcuts"], state["execution_template"])
        if result["status"] == "success":
            state["execution_status"] = "success"
            state["completed"]         = True
            if "execution_history" in result:
                state["history"] = result["execution_history"]
            if "final_screenshot" in result:
                state["current_page"]["screenshot"] = result["final_screenshot"]
        else:
            state["should_fallback"] = True
    else:
        print("📝 Attempting to match task with high-level actions...")
        is_matched, matched_action = match_task_to_action(state_dict, state["task"])
        if is_matched and matched_action:
            state["current_action"] = matched_action
            element_sequence = matched_action.get("element_sequence", [])
            if isinstance(element_sequence, str):
                try:
                    element_sequence = json.loads(element_sequence)
                except Exception:
                    state["should_fallback"] = True
                    return state
            if not element_sequence:
                state["should_fallback"] = True
                return state
            state["current_step"]    = 0
            state["total_steps"]     = len(element_sequence)
            state["execution_status"] = "running"
            state["execution_template"] = {"steps": element_sequence}
        else:
            state["should_fallback"] = True
    return state


def fallback_node(state: DeploymentState) -> DeploymentState:
    print("⚠️ Falling back to basic operation space")
    state = fallback_to_react(state)
    state["completed"] = True
    return state


# ─────────────────────────────────────────────────────────────────────────────
#  Routing functions  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def should_fallback(state: DeploymentState) -> str:
    return "fallback" if state.get("should_fallback") else "continue"

def should_execute_shortcut(state: DeploymentState) -> str:
    return "execute_shortcut" if state.get("should_execute_shortcut") else "match_task"

def is_task_completed(state: DeploymentState) -> str:
    return "end" if state.get("completed") else "continue"


# ─────────────────────────────────────────────────────────────────────────────
#  LangGraph workflow  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    workflow = StateGraph(DeploymentState)

    workflow.add_node("capture_screen",    capture_screen_node)
    workflow.add_node("match_elements",    match_elements_node)
    workflow.add_node("check_shortcuts",   check_shortcuts_node)
    workflow.add_node("evaluate_shortcut", shortcut_evaluation_node)
    workflow.add_node("generate_template", generate_template_node)
    workflow.add_node("execute_action",    execute_action_node)
    workflow.add_node("fallback",          fallback_node)
    workflow.add_node("check_completion",  check_task_completion)

    workflow.set_entry_point("capture_screen")

    workflow.add_conditional_edges("capture_screen",    should_fallback, {"fallback": "fallback", "continue": "match_elements"})
    workflow.add_conditional_edges("match_elements",    should_fallback, {"fallback": "fallback", "continue": "check_shortcuts"})
    workflow.add_edge("check_shortcuts", "evaluate_shortcut")
    workflow.add_conditional_edges("evaluate_shortcut", should_execute_shortcut, {"execute_shortcut": "generate_template", "match_task": "execute_action"})
    workflow.add_edge("generate_template", "execute_action")
    workflow.add_edge("execute_action",    "check_completion")
    workflow.add_conditional_edges("check_completion",  is_task_completed, {"end": END, "continue": "capture_screen"})
    workflow.add_edge("fallback", "check_completion")

    return workflow


def run_task(task: str, device: str = "emulator-5554") -> Dict[str, Any]:
    print(f"🚀 Starting task execution: {task}")
    try:
        from data.State import create_deployment_state
        state   = create_deployment_state(task=task, device=device, max_retries=3)
        app     = build_workflow().compile()
        result  = app.invoke(state)

        if result["execution_status"] == "success" and result["current_page"]["screenshot"]:
            try:
                from PIL import Image
                Image.open(result["current_page"]["screenshot"]).show()
            except Exception:
                pass

        return {
            "status":          result["execution_status"],
            "message":         "Task execution completed",
            "steps_completed": result["current_step"],
            "total_steps":     result["total_steps"],
        }
    except Exception as e:
        print(f"❌ Error executing task: {e}")
        return {"status": "error", "message": str(e), "error": str(e)}