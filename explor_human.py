"""
explor_human.py
===============
Human-exploration node functions.

BUG-3 FIX (source_json path in history_steps)
──────────────────────────────────────────────
ORIGINAL:
    "source_json": _to_relative(state.get("current_page_json", ""))

PROBLEM:
    state["current_page_json"] is set by session.insert_parsed_result() to the
    LOCAL temp file path (e.g. /tmp/human_explorer_cloud/ss1.json).  Storing
    that in history_steps means it only works on the machine that ran the
    session, and only until the temp file is cleaned up.

FIX:
    SessionManager now keeps TWO separate fields:
        state["current_page_json"]      = local temp path  (for live coord lookup)
        state["current_page_json_url"]  = Cloudinary URL   (for portability)

    single_human_explor() stores BOTH in the step record:
        "source_json"       → cloud URL   (used as the canonical identifier)
        "source_json_local" → local path  (used by json2db on the same machine)

    json2db() (_load_elements_for_step) tries source_json_local first, then
    falls back to downloading from source_json (the cloud URL).
"""

import datetime
import json
import os
from pathlib import Path
from typing import Tuple

from data.State import State
from tool.adb_tools import screen_action, take_screenshot


# ─────────────────────────────────────────────────────────────────────────────
#  Portability helper  (kept for screenshot paths which are always local)
# ─────────────────────────────────────────────────────────────────────────────

def _to_relative(path_str: str) -> str:
    """
    Convert an absolute local path to a project-root-relative forward-slash
    path.  Falls back to the original string if relativisation fails.
    """
    if not path_str:
        return path_str
    try:
        return Path(path_str).resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path_str.replace("\\", "/").strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Screenshot capture
# ─────────────────────────────────────────────────────────────────────────────

def capture_screenshot_only(state: State) -> State:
    device   = state.get("device", "emulator")
    app_name = state.get("app_name", "unknown_app")

    result = take_screenshot.invoke({
        "device":   device,
        "save_dir": "./log/screenshots",
        "app_name": app_name,
        "step":     state["step"] + 1,
    })

    if result.startswith("Screenshot failed"):
        print(f"[capture_screenshot_only] {result}")
        state["errors"].append({
            "step":      state["step"],
            "tool":      "take_screenshot",
            "error_msg": result,
        })
    else:
        print(f"[capture_screenshot_only] saved → {result}")
        state["current_page_screenshot"] = result
        # Clear both JSON fields so the UI knows a new parse is needed
        state["current_page_json"]       = None
        state["current_page_json_url"]   = None

    return state


# ─────────────────────────────────────────────────────────────────────────────
#  Element-ID → pixel coords
# ─────────────────────────────────────────────────────────────────────────────

def element_number_to_coords(state: State, element_id: int) -> Tuple[int, int]:
    json_path = state.get("current_page_json")

    if not json_path:
        msg = (
            "[element_number_to_coords] No parsed result available. "
            "Submit cloud URLs first."
        )
        state["errors"].append({"step": state["step"], "func": "element_number_to_coords", "error_msg": msg})
        raise RuntimeError(msg)

    if not os.path.isfile(json_path):
        msg = f"[element_number_to_coords] JSON not found locally: {json_path}"
        state["errors"].append({"step": state["step"], "func": "element_number_to_coords", "error_msg": msg})
        raise FileNotFoundError(msg)

    device_info   = state.get("device_info", {})
    screen_width  = device_info.get("width")
    screen_height = device_info.get("height")

    if not screen_width or not screen_height:
        msg = "[element_number_to_coords] device_info missing width/height."
        state["errors"].append({"step": state["step"], "func": "element_number_to_coords", "error_msg": msg})
        raise ValueError(msg)

    with open(json_path, "r", encoding="utf-8") as f:
        elements = json.load(f)

    target = next((e for e in elements if e.get("ID") == element_id), None)
    if target is None:
        msg = f"[element_number_to_coords] ID={element_id} not found."
        state["errors"].append({"step": state["step"], "func": "element_number_to_coords", "error_msg": msg})
        raise ValueError(msg)

    bbox = target.get("bbox")
    if not bbox or len(bbox) != 4:
        msg = f"[element_number_to_coords] Invalid bbox for element {element_id}: {bbox}"
        state["errors"].append({"step": state["step"], "func": "element_number_to_coords", "error_msg": msg})
        raise ValueError(msg)

    x1, y1, x2, y2 = bbox
    px = int(((x1 + x2) / 2) * screen_width)
    py = int(((y1 + y2) / 2) * screen_height)
    return px, py


# ─────────────────────────────────────────────────────────────────────────────
#  Single ADB action
# ─────────────────────────────────────────────────────────────────────────────

def single_human_explor(state: State, action: str, **kwargs) -> State:
    print(f"[single_human_explor] action={action}  kwargs={kwargs}")

    action_result = None
    x = y = None

    VALID_ACTIONS = {"tap", "text", "long_press", "swipe", "swipe_precise", "back", "wait"}

    if action not in VALID_ACTIONS:
        msg = f"Unsupported action: {action}"
        state["errors"].append({"step": state["step"], "action": action, "error_msg": msg})
        state = capture_screenshot_only(state)
        state["step"] += 1
        return state

    if action in ("tap", "long_press", "swipe"):
        elem_num = kwargs.get("element_number")
        if elem_num is None:
            msg = f"element_number required for {action}."
            state["errors"].append({"step": state["step"], "tool": "screen_action", "error_msg": msg})
            state = capture_screenshot_only(state)
            state["step"] += 1
            return state
        try:
            x, y = element_number_to_coords(state, elem_num)
        except Exception as exc:
            state["errors"].append({"step": state["step"], "tool": "element_number_to_coords", "error_msg": str(exc)})
            state = capture_screenshot_only(state)
            state["step"] += 1
            return state

    if action == "text" and kwargs.get("element_number") is not None:
        try:
            x, y = element_number_to_coords(state, kwargs.get("element_number"))
        except Exception as exc:
            state["errors"].append({"step": state["step"], "tool": "element_number_to_coords", "error_msg": str(exc)})

    params = {"device": state.get("device", "emulator"), "action": action}

    if action == "wait":
        action_result = json.dumps({"status": "success", "action": "wait", "message": "No-op."})
    elif action == "back":
        action_result = screen_action.invoke(params)
    elif action == "tap":
        params.update({"x": x, "y": y})
        action_result = screen_action.invoke(params)
    elif action == "text":
        txt = kwargs.get("text_input")
        if txt:
            params.update({"input_str": txt})
            action_result = screen_action.invoke(params)
        else:
            state["errors"].append({"step": state["step"], "tool": "screen_action", "error_msg": "text_input missing"})
    elif action == "long_press":
        params.update({"x": x, "y": y, "duration": kwargs.get("duration", 1000)})
        action_result = screen_action.invoke(params)
    elif action == "swipe":
        sd = kwargs.get("swipe_direction")
        if sd:
            params.update({
                "x": x, "y": y, "direction": sd,
                "dist": kwargs.get("dist", "medium"),
                "quick": kwargs.get("quick", False),
            })
            action_result = screen_action.invoke(params)
        else:
            state["errors"].append({"step": state["step"], "tool": "screen_action", "error_msg": "swipe_direction missing"})
    elif action == "swipe_precise":
        sc, ec = kwargs.get("start"), kwargs.get("end")
        if sc and ec:
            params.update({"start": sc, "end": ec, "duration": kwargs.get("duration", 400)})
            action_result = screen_action.invoke(params)
        else:
            state["errors"].append({"step": state["step"], "tool": "screen_action", "error_msg": "start/end missing"})

    if action_result:
        try:
            action_dict = json.loads(action_result)
        except json.JSONDecodeError:
            action_dict = {"raw_result": action_result}

        state["tool_results"].append({"tool_name": "screen_action", "action_result": action_dict})

        # ── BUG-3 FIX ────────────────────────────────────────────────────────
        # Store BOTH the cloud URL (portable) and the local temp path (for
        # json2db on this machine).
        #
        # state["current_page_json"]     → local temp path set by insert_parsed_result
        # state["current_page_json_url"] → Cloudinary URL set by insert_parsed_result
        #
        # We store:
        #   "source_json"       = cloud URL   (canonical, portable)
        #   "source_json_local" = local path  (fast access during the same session)
        #
        # json2db() will use source_json_local first; if missing/stale it will
        # download from source_json.

        source_json_url   = state.get("current_page_json_url") or ""
        source_json_local = state.get("current_page_json") or ""

        # Screenshot path is always local; make it relative for portability
        source_page_rel = _to_relative(state.get("current_page_screenshot", ""))

        state["history_steps"].append({
            "step":               state["step"],
            "recommended_action": f"Executing {action} with params {kwargs}",
            "tool_result": {
                "action":  action,
                "device":  state.get("device", "emulator"),
                "clicked_element": (
                    {"x": x, "y": y}
                    if action in ("tap", "long_press", "text") and x is not None and y is not None
                    else None
                ),
                "status":  "success" if action_result else "failed",
                **action_dict,
            },
            "source_page":        source_page_rel,
            "source_json":        source_json_url,        # BUG-3 FIX: cloud URL
            "source_json_local":  source_json_local,      # BUG-3 FIX: local path
            "timestamp":          datetime.datetime.now().isoformat(),
        })

    state = capture_screenshot_only(state)
    state["step"] += 1
    return state