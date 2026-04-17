"""
tool/adb_tools.py
=================
ADB helpers: device listing, size query, screenshot, screen actions.
The parsing tool (screen_element) has been intentionally removed –
parsed results are inserted manually via POST /api/insert_parsed_result.
"""

import datetime
import json
import os
import shutil
import subprocess
import config
from time import sleep

from langchain_core.tools import tool


# ── low-level ADB runner ──────────────────────────────────────────────────────

def _resolve_adb() -> str:
    """Resolve adb executable from PATH or common Android SDK locations."""
    if os.environ.get("ADB_PATH"):
        return os.environ["ADB_PATH"]

    found = shutil.which("adb") or shutil.which("adb.exe")
    if found:
        return found

    for var in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
        root = os.environ.get(var)
        if root:
            candidate = os.path.join(root, "platform-tools", "adb.exe")
            if os.path.isfile(candidate):
                return candidate

    return "adb"

def _adb(cmd: str) -> str:
    adb_bin = _resolve_adb()
    cmd = cmd.replace("adb", f'"{adb_bin}"', 1)
    res = subprocess.run(
        cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if res.returncode == 0:
        return res.stdout.strip()
    print(f"[ADB] command failed: {cmd}\n{res.stderr}")
    return "ERROR"


# ── device discovery ──────────────────────────────────────────────────────────

def list_all_devices() -> list:
    """Return a list of currently connected ADB device IDs."""
    result = _adb("adb devices")
    if result == "ERROR":
        return []
    lines = result.split("\n")[1:]
    devices = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def list_devices_diagnostics() -> str:
    """Return raw adb devices output for troubleshooting UI detection."""
    result = _adb("adb devices -l")
    if result == "ERROR":
        return (
            "ADB command failed. Ensure adb is installed and reachable by this "
            "Python process."
        )
    return result


# ── device info ───────────────────────────────────────────────────────────────

@tool
def get_device_size(device: str = "emulator") -> dict:
    """
    Return the screen resolution of the target device.

    Returns:
        {"width": int, "height": int}  or an error string.
    """
    result = _adb(f"adb -s {device} shell wm size")
    if result == "ERROR":
        return {"error": "Failed to get device size."}
    try:
        size_str = result.split(": ")[1]
        w, h = map(int, size_str.split("x"))
        return {"width": w, "height": h}
    except Exception as exc:
        return {"error": str(exc)}


# ── screenshot ────────────────────────────────────────────────────────────────

@tool
def take_screenshot(
    device:   str = "emulator",
    save_dir: str = "./log/screenshots",
    app_name: str = "unknown_app",
    step:     int = 0,
) -> str:
    """
    Take a screenshot via ADB and pull it to the local filesystem.

    Returns the local file path on success, or an error string beginning
    with "Screenshot failed".
    """
    app_dir = os.path.join(save_dir, app_name)
    os.makedirs(app_dir, exist_ok=True)

    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{app_name}_step{step}_{ts}.png"
    local    = os.path.join(app_dir, filename)
    remote   = f"/sdcard/{filename}"

    sleep(2)  # let the screen settle

    if _adb(f"adb -s {device} shell screencap -p {remote}") == "ERROR":
        return "Screenshot failed: screencap error"
    if _adb(f"adb -s {device} pull {remote} {local}") == "ERROR":
        return "Screenshot failed: pull error"
    _adb(f"adb -s {device} shell rm {remote}")

    return local


# ── screen actions ────────────────────────────────────────────────────────────

@tool
def screen_action(
    device:    str   = "emulator",
    action:    str   = "tap",
    x:         int   = None,
    y:         int   = None,
    input_str: str   = None,
    duration:  int   = 1000,
    direction: str   = None,
    dist:      str   = "medium",
    quick:     bool  = False,
    start:     tuple = None,
    end:       tuple = None,
) -> str:
    """
    Tool name: screen_action

    Tool function:
        Perform screen operations on mobile devices (Android emulator or real device), including tap, back, type text, swipe, long press, and drag.

    Parameters:
        - device (str): Specify the target device ID, default is "emulator".
        - action (str): Specify the type of screen operation to perform. Supports the following operations:
            - "tap": Tap the specified coordinates on the screen.
                Requires parameters: x, y
            - "back": Back key operation.
                No additional parameters required.
            - "text": Enter text on the screen.
                Requires parameter: input_str
            - "long_press": Long press the specified coordinates on the screen.
                Requires parameters: x, y, duration (default 1000 milliseconds)
            - "swipe": Swipe operation, supports four directions ("up", "down", "left", "right").
                Requires parameters: x, y, direction, dist (default "medium"), quick (default False)
            - "swipe_precise": Precise swipe, swipe from the specified start point to the specified end point.
                Requires parameters: start, end, duration (default 400 milliseconds)

    Returns:
        Returns a JSON string including the following fields:
        - "status": "success" or "error"
        - "action": Type of operation performed
        - "device": Device ID
        - Other fields appended according to the operation type (e.g., clicked coordinates, input text, swipe start and end points, etc.).
    """
    result: dict = {"action": action, "device": device}

    try:
        cmd = None

        if action == "back":
            cmd = f"adb -s {device} shell input keyevent KEYCODE_BACK"

        elif action == "tap":
            if x is None or y is None:
                return json.dumps({**result, "status": "error", "message": "x and y required"})
            cmd = f"adb -s {device} shell input tap {x} {y}"
            result["clicked_element"] = {"x": x, "y": y}

        elif action == "text":
            if not input_str:
                return json.dumps({**result, "status": "error", "message": "input_str required"})
            safe = input_str.replace(" ", "%s").replace("'", "")
            cmd  = f"adb -s {device} shell input text {safe}"
            result["input_str"] = input_str

        elif action == "long_press":
            if x is None or y is None:
                return json.dumps({**result, "status": "error", "message": "x and y required"})
            cmd = f"adb -s {device} shell input swipe {x} {y} {x} {y} {duration}"
            result["long_press"] = {"x": x, "y": y, "duration": duration}

        elif action == "swipe":
            if x is None or y is None or direction is None:
                return json.dumps({**result, "status": "error", "message": "x, y, direction required"})
            unit = 160  # Increased swipe base distance
            dx, dy = 0, 0
            dist_key = (dist or "medium").lower()
            factor_map = {"short": 2, "medium": 3, "long": 4}
            factor = factor_map.get(dist_key, 3)
            if direction == "up":    dy = -factor * unit
            elif direction == "down":  dy =  factor * unit
            elif direction == "left":  dx = -factor * unit
            elif direction == "right": dx =  factor * unit
            else:
                return json.dumps({**result, "status": "error", "message": f"Unknown direction: {direction}"})
            dur = 100 if quick else 400
            cmd = f"adb -s {device} shell input swipe {x} {y} {x+dx} {y+dy} {dur}"
            result["swipe"] = {"start": (x, y), "end": (x+dx, y+dy), "duration": dur, "direction": direction}

        elif action == "swipe_precise":
            if not start or not end:
                return json.dumps({**result, "status": "error", "message": "start and end required"})
            sx, sy = start;  ex, ey = end
            cmd = f"adb -s {device} shell input swipe {sx} {sy} {ex} {ey} {duration}"
            result["swipe_precise"] = {"start": start, "end": end, "duration": duration}

        else:
            return json.dumps({**result, "status": "error", "message": f"Unknown action: {action}"})

        ret = _adb(cmd)
        result["status"] = "success" if (ret != "ERROR") else "error"
        if ret == "ERROR":
            result["message"] = "ADB command failed"

        return json.dumps(result, ensure_ascii=False)

    except Exception as exc:
        return json.dumps({**result, "status": "error", "message": str(exc)}, ensure_ascii=False)
