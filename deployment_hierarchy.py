"""Best-effort Android accessibility hierarchy observation through ADB."""
import re
import subprocess
import uuid
from pathlib import Path
from xml.etree import ElementTree


def parse_hierarchy(xml, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Device dimensions must be positive")
    root = ElementTree.fromstring(xml)
    result = []
    for node in root.iter("node"):
        attrs = node.attrib
        bounds = re.fullmatch(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", attrs.get("bounds", ""))
        if not bounds:
            continue
        x1,y1,x2,y2 = map(int,bounds.groups())
        if x2 <= x1 or y2 <= y1:
            continue
        cls = attrs.get("class", "")
        editable = "EditText" in cls
        label = (attrs.get("text") or attrs.get("content-desc") or "").strip()
        resource = attrs.get("resource-id", "")
        if not label and editable:
            label = "Search" if "search" in resource.lower() else "Input field"
        if not label:
            continue
        result.append({"ID": len(result), "content": label,
                       "bbox": [max(0,min(1,x1/width)),max(0,min(1,y1/height)),max(0,min(1,x2/width)),max(0,min(1,y2/height))],
                       "type": "input" if editable else "text", "editable": editable,
                       "resource_id": resource, "package": attrs.get("package", ""), "class": cls,
                       "clickable": attrs.get("clickable") == "true",
                       "enabled": attrs.get("enabled", "true") == "true",
                       "selected": attrs.get("selected") == "true"})
    return result


def _extract_xml(output):
    start = output.find("<hierarchy")
    end = output.rfind("</hierarchy>")
    if start < 0 or end < 0:
        raise ValueError("ADB returned no hierarchy XML")
    return output[start:end+len("</hierarchy>")]


def capture_hierarchy(device, adb, size, track, log):
    """Return parsed elements, or [] so the caller can use OmniParser."""
    remote = "/sdcard/codex_deployment_" + uuid.uuid4().hex + ".xml"
    prefix = [str(adb), "-s", str(device)]
    def run(args):
        proc = subprocess.run(prefix + args, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=15)
        if proc.returncode:
            raise RuntimeError(proc.stderr.strip() or "ADB hierarchy command failed")
        return proc.stdout
    try:
        log("[HIERARCHY] Requesting Android UI hierarchy...")
        try:
            xml = _extract_xml(run(["exec-out", "uiautomator", "dump", "/dev/tty"]))
        except (RuntimeError, ValueError):
            try:
                run(["shell", "uiautomator", "dump", remote])
                xml = _extract_xml(run(["exec-out", "cat", remote]))
            finally:
                try:
                    run(["shell", "rm", remote])
                except Exception:
                    log("[HIERARCHY] Could not remove the temporary device XML file.")
        elements = parse_hierarchy(xml, size["width"], size["height"])
        path = Path("log/screenshots/deployment") / ("hierarchy_" + uuid.uuid4().hex + ".xml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(xml, encoding="utf-8")
        track(str(path))
        log(f"[HIERARCHY] Ready: {len(elements)} labeled elements; XML={path}")
        return elements
    except Exception as exc:
        log(f"[HIERARCHY] Unavailable: {type(exc).__name__}: {exc}; falling back to OmniParser.")
        return []
