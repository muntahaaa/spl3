"""
explor_human.py  –  ROOT-CAUSE FIX FOR BUG 2
=============================================
The only change from the previous version is in single_human_explor(),
in the step_record that is appended to history_steps.

ORIGINAL (produces Bug 2):
    "source_page": state.get("current_page_screenshot"),
    "source_json": state.get("current_page_json"),

These values come directly from the State dict, which stores whatever
path string was returned by take_screenshot / insert_parsed_result.
On Windows that is often an absolute path like:
    C:\\Users\\you\\project\\log\\screenshots\\...
    D:/4th year/spl3/labeled_image/json_path/ss1.json

When json2db later runs on a different machine or in a Docker container
those absolute paths do not exist and open() raises FileNotFoundError.

FIX:
    Wrap both values with _to_relative() before storing them in the record.
    _to_relative() tries to express the path relative to the current working
    directory (the project root).  If the path is already relative, or if
    relativisation fails for any reason, the original string is kept unchanged.
    The result uses forward slashes only, so it is portable across platforms.
"""

import datetime
import json
import os
from pathlib import Path
from typing import Tuple

from data.State import State
from tool.adb_tools import screen_action, take_screenshot


# ─────────────────────────────────────────────────────────────────────────────
#  Portability helper
# ─────────────────────────────────────────────────────────────────────────────

def _to_relative(path_str: str) -> str:
    """
    Convert an absolute path to a path relative to the project root (cwd).
    Always uses forward slashes.  Falls back to the original string if
    relativisation raises (e.g. path is on a different Windows drive).
    """
    if not path_str:
        return path_str
    try:
        return Path(path_str).resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        # Different drive on Windows, or path outside cwd – keep original
        # but at least normalise separators
        return path_str.replace("\\", "/").strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Screenshot capture  (unchanged from previous version)
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
        state["current_page_json"]       = None

    return state


# ─────────────────────────────────────────────────────────────────────────────
#  Element-ID → pixel coords  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def element_number_to_coords(state: State, element_id: int) -> Tuple[int, int]:
    json_path = state.get("current_page_json")

    if not json_path:
        msg = (
            "[element_number_to_coords] No parsed result available.  "
            "POST /api/insert_parsed_result first."
        )
        state["errors"].append({"step": state["step"], "func": "element_number_to_coords", "error_msg": msg})
        raise RuntimeError(msg)

    if not os.path.isfile(json_path):
        msg = f"[element_number_to_coords] JSON not found: {json_path}"
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
#  Single action  –  ROOT-CAUSE FIX FOR BUG 2 IS HERE
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

    # For text input, element_number is optional but helps us link the action
    # to a concrete UI element later (e.g., vector storage in json2db).
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
            params.update({"x": x, "y": y, "direction": sd,
                           "dist": kwargs.get("dist", "medium"),
                           "quick": kwargs.get("quick", False)})
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

        # ── ROOT-CAUSE FIX FOR BUG 2 ──────────────────────────────────────────
        #
        # ORIGINAL:
        #     "source_page": state.get("current_page_screenshot"),
        #     "source_json": state.get("current_page_json"),
        #
        # Both values are stored raw from the State dict.  On Windows this is
        # often "D:/absolute/path/to/file.json" which is machine-specific and
        # will cause FileNotFoundError when json2db runs on any other system.
        #
        # FIX:
        #     Wrap both through _to_relative() so the stored value is always a
        #     portable relative path like  "log/screenshots/human_exploration/...png"
        #     with forward slashes.  This way json2db can open the file on any
        #     platform as long as it runs from the same project root directory.

        state["history_steps"].append({
            "step":               state["step"],
            "recommended_action": f"Executing {action} with params {kwargs}",
            "tool_result": {
                "action":  action,
                "device":  state.get("device", "emulator"),
                "clicked_element": ({"x": x, "y": y} if action in ("tap", "long_press", "text") and x is not None and y is not None else None),
                "status":  "success" if action_result else "failed",
                **action_dict,
            },
            # ↓ FIX 2: converted to relative portable path before storing
            "source_page": _to_relative(state.get("current_page_screenshot", "")),
            "source_json": _to_relative(state.get("current_page_json", "")) if state.get("current_page_json") else None,
            "timestamp":   datetime.datetime.now().isoformat(),
        })

    state = capture_screenshot_only(state)
    state["step"] += 1
    return state
