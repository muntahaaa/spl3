"""
ui.py  –  Gradio user interface
================================
Steps covered:
    ① Initialization  → device selection, task entry
    ② Exploration     → ADB actions + cloud URL popup after every screenshot
    ③ Save & Export   → state → JSON
    ④ Store to DB     → JSON → Neo4j + Pinecone

BUG-3 FIX
──────────
The cloud-URL popup now passes the *original Cloudinary URL* of the JSON file
to session.insert_parsed_result() as `parsed_content_json_url`.  This URL is
what ends up stored in history_steps["source_json"] (the canonical portable
identifier), while the locally downloaded copy is stored in
history_steps["source_json_local"] for fast access during the same session.
"""

import json
import os
import tempfile
import urllib.request
from pathlib import Path

import gradio as gr

from data.data_storage import json2db, state2json
from explor_human import capture_screenshot_only, single_human_explor
from state_manager import session
from tool.adb_tools import get_device_size, list_all_devices, list_devices_diagnostics
from data.State import State


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_devices():
    devices = list_all_devices()
    return devices if devices else ["No devices found"]


def _action_visibility(action: str):
    show_elem  = action in ("tap", "text", "long_press", "swipe")
    show_text  = action == "text"
    show_swipe = action == "swipe"
    return (
        gr.update(visible=show_elem),
        gr.update(visible=show_text),
        gr.update(visible=show_swipe),
    )


def _download_url(url: str, suffix: str) -> str:
    """
    Download a URL to a local temp file.  Returns the local path.
    The original URL is returned unchanged if it is already a local file.
    """
    url = url.strip()
    if not url:
        raise ValueError("URL is empty")

    # Already a local path?
    if not url.startswith("http://") and not url.startswith("https://"):
        if Path(url).is_file():
            return url
        raise FileNotFoundError(f"Local file not found: {url}")

    tmp_dir = Path(tempfile.gettempdir()) / "human_explorer_cloud"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tail = Path(url.split("?")[0]).name or f"download{suffix}"
    if suffix and not tail.endswith(suffix):
        tail += suffix
    dest = tmp_dir / tail

    if not dest.is_file():
        urllib.request.urlretrieve(url, str(dest))

    return str(dest)


# ─────────────────────────────────────────────────────────────────────────────
#  Initialization callbacks
# ─────────────────────────────────────────────────────────────────────────────

def refresh_devices():
    try:
        devices = _get_devices()
        raw     = list_devices_diagnostics()
        return gr.update(choices=devices), raw
    except Exception as exc:
        return gr.update(choices=["No devices found"]), f"Error: {exc}"


def initialize_device(device: str, task_info: str):
    if not task_info:
        return "Error: task information cannot be empty."
    if not device or device == "No devices found":
        return "Error: select a valid ADB device."

    device_info = get_device_size.invoke({"device": device})
    if "error" in device_info:
        return f"Error reading device size: {device_info['error']}"

    state = State(
        tsk=task_info,
        app_name="human_exploration",
        completed=False,
        step=0,
        history_steps=[],
        page_history=[],
        current_page_screenshot=None,
        current_page_json=None,
        current_page_json_url=None,          # BUG-3 FIX: new field
        recommend_action="",
        clicked_elements=[],
        action_reflection=[],
        tool_results=[],
        device=device,
        device_info=device_info,
        context=[],
        errors=[],
        callback=None,
    )
    session.set_state(state)
    session.user_log_storage  = []
    session.user_page_storage = []
    return f"✅ Initialized device '{device}' — task: {task_info}"


# ─────────────────────────────────────────────────────────────────────────────
#  Exploration callbacks
# ─────────────────────────────────────────────────────────────────────────────

def start_session():
    state = session.get_state()
    if state is None:
        return "Error: initialize a device first.", [], gr.update(visible=False)

    updated = capture_screenshot_only(state)
    session.set_state(updated)

    screenshot = updated.get("current_page_screenshot", "")
    msg = (
        f"Session started.\n"
        f"📷 Screenshot saved → {screenshot}\n"
        f"⬇️  Enter cloud URLs below and click Submit."
    )
    return msg, session.user_page_storage, gr.update(visible=True)


def perform_action(action, element_number, text_input, swipe_direction):
    state = session.get_state()
    if state is None:
        return "Error: initialize a device first.", [], gr.update(visible=False)

    updated = single_human_explor(
        state,
        action,
        element_number=int(element_number) if element_number is not None else None,
        text_input=text_input,
        swipe_direction=swipe_direction,
    )
    session.set_state(updated)

    log_entry = {
        "step":       updated["step"],
        "action":     action,
        "completed":  updated["completed"],
        "screenshot": updated.get("current_page_screenshot"),
        "parsed":     updated.get("current_page_json") is not None,
    }
    session.user_log_storage.append(json.dumps(log_entry, ensure_ascii=False))

    for p in updated.get("page_history", []):
        if p and p not in session.user_page_storage:
            session.user_page_storage.append(p)

    status = (
        f"Step {updated['step']} done — '{action}' executed.\n"
        f"📷 New screenshot → {updated.get('current_page_screenshot')}\n"
        f"⬇️  Enter cloud URLs below and click Submit."
    )
    log_text = "\n".join(session.user_log_storage) + "\n" + status
    return log_text, session.user_page_storage, gr.update(visible=True)


def submit_cloud_urls(image_url: str, json_url: str):
    """
    Download both cloud files locally, then call insert_parsed_result()
    passing the original cloud URL alongside the local path.
    """
    state = session.get_state()
    if state is None:
        return "Error: no active session.", [], gr.update(visible=True)

    if not image_url.strip() and not json_url.strip():
        return skip_cloud_urls()

    errors = []

    # ── Download labeled image ────────────────────────────────────────────────
    try:
        local_image = _download_url(image_url, ".png")
    except Exception as exc:
        errors.append(f"Image download failed: {exc}")
        local_image = None

    # ── Download elements JSON ────────────────────────────────────────────────
    try:
        local_json = _download_url(json_url, ".json")
    except Exception as exc:
        errors.append(f"JSON download failed: {exc}")
        local_json = None

    if errors:
        msg = "⚠️  " + " | ".join(errors) + "\n(Popup kept open — fix URLs or click Skip)"
        return "\n".join(session.user_log_storage) + "\n" + msg, session.user_page_storage, gr.update(visible=True)

    # ── BUG-3 FIX: pass BOTH local path AND original cloud URL ───────────────
    try:
        result = session.insert_parsed_result(
            labeled_image_path=local_image,
            parsed_content_json_path=local_json,
            parsed_content_json_url=json_url.strip(),   # ← the Cloudinary URL
        )
        session.user_log_storage.append(
            f"✅ Parsed result inserted (step {result['step']}) — "
            f"json_url: {result['new_json_url']}"
        )
        for p in session.get_state().get("page_history", []):
            if p and p not in session.user_page_storage:
                session.user_page_storage.append(p)
    except Exception as exc:
        session.user_log_storage.append(f"❌ insert_parsed_result error: {exc}")

    return "\n".join(session.user_log_storage), session.user_page_storage, gr.update(visible=False)


def skip_cloud_urls():
    session.user_log_storage.append(
        "⏭️  Cloud URL step skipped — no parsed result for this screenshot."
    )
    return "\n".join(session.user_log_storage), session.user_page_storage, gr.update(visible=False)


def stop_and_save():
    state = session.get_state()
    if state is None:
        return "Error: no active session."
    state["completed"] = True
    session.set_state(state)
    saved_path = state2json(state)
    session.user_log_storage.append(f"💾 State saved → {saved_path}")
    return "\n".join(session.user_log_storage)


def store_to_db(json_path: str):
    if not json_path:
        return "Error: provide the JSON state file path."
    try:
        task_id = json2db(json_path.strip())
        return f"✅ Stored to Neo4j + Pinecone.  Task ID: {task_id}"
    except Exception as exc:
        return f"Error: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
#  Gradio layout
# ─────────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Human Explorer") as demo:
        gr.Markdown(
            "# 📱 Human Explorer\n"
            "**3-step pipeline:** "
            "① Explore (ADB actions + screenshots) → "
            "② Save session to JSON → "
            "③ Push to Neo4j + Pinecone"
        )

        # ── Tab 1 : Initialization ────────────────────────────────────────────
        with gr.Tab("① Initialization"):
            gr.Markdown("Select your ADB device and describe the task you are exploring.")
            devices_box  = gr.Textbox(label="Connected devices", interactive=False)
            refresh_btn  = gr.Button("🔄 Refresh devices")
            device_radio = gr.Radio(label="Select ADB device", choices=[])
            task_input   = gr.Textbox(
                label="Task description",
                placeholder="e.g. Log in and navigate to Settings",
            )
            init_btn    = gr.Button("✅ Initialize")
            init_status = gr.Textbox(label="Status", interactive=False)

            refresh_btn.click(refresh_devices, outputs=[device_radio, devices_box], queue=False)
            demo.load(refresh_devices, outputs=[device_radio, devices_box], queue=False)
            init_btn.click(initialize_device, inputs=[device_radio, task_input], outputs=[init_status], queue=False)

        # ── Tab 2 : Exploration ───────────────────────────────────────────────
        with gr.Tab("② Exploration"):
            gr.Markdown(
                "### Workflow per step\n"
                "1. Perform an action → screenshot is saved automatically.\n"
                "2. The cloud-URL panel appears automatically.\n"
                "3. Paste your **Cloudinary** (or other CDN) URLs and click **Submit**.\n"
                "4. The files are downloaded locally.  The JSON URL is stored in the session "
                "   as `source_json` for portability.\n"
                "5. Click **Skip** to continue without parsing (LEADS_TO edge will be skipped)."
            )

            start_btn    = gr.Button("▶ Start session (take initial screenshot)")
            action_radio = gr.Radio(
                ["tap", "text", "long_press", "swipe", "back", "wait"], label="Action"
            )
            element_num = gr.Number(label="Element ID (from parsed JSON)", precision=0, visible=False)
            text_in     = gr.Textbox(label="Text input", visible=False)
            swipe_dir   = gr.Radio(["up", "down", "left", "right"], label="Swipe direction", visible=False)
            perform_btn = gr.Button("⚡ Perform action")
            stop_btn    = gr.Button("🛑 Stop & save to JSON")
            logs_box    = gr.TextArea(label="Step log", interactive=False, lines=10)
            gallery     = gr.Gallery(label="Labeled screenshots", height=500)

            # ── Cloud URL popup ───────────────────────────────────────────────
            with gr.Group(visible=False) as cloud_url_panel:
                gr.Markdown(
                    "### ☁️ Provide Cloud URLs for Parsed Result\n"
                    "Paste the **Cloudinary** (or S3 / GCS) pre-signed URLs for the "
                    "two files produced by your parsing tool.\n\n"
                    "The JSON URL will be stored as `source_json` in `history_steps` "
                    "so the state file is portable.  The files are also downloaded "
                    "locally for immediate coordinate lookups."
                )
                cloud_image_url = gr.Textbox(
                    label="Labeled image URL (.png)",
                    placeholder="https://res.cloudinary.com/.../labeled_step_N.png",
                )
                cloud_json_url = gr.Textbox(
                    label="Elements JSON URL (.json)",
                    placeholder="https://res.cloudinary.com/.../elements_step_N.json",
                )
                with gr.Row():
                    submit_urls_btn = gr.Button("✅ Submit", variant="primary")
                    skip_urls_btn   = gr.Button("⏭️ Skip")

            action_radio.change(
                _action_visibility,
                inputs=[action_radio],
                outputs=[element_num, text_in, swipe_dir],
                queue=False,
            )
            start_btn.click(
                start_session,
                outputs=[logs_box, gallery, cloud_url_panel],
                queue=False,
            )
            perform_btn.click(
                perform_action,
                inputs=[action_radio, element_num, text_in, swipe_dir],
                outputs=[logs_box, gallery, cloud_url_panel],
                queue=False,
            )
            submit_urls_btn.click(
                submit_cloud_urls,
                inputs=[cloud_image_url, cloud_json_url],
                outputs=[logs_box, gallery, cloud_url_panel],
                queue=False,
            )
            skip_urls_btn.click(
                skip_cloud_urls,
                outputs=[logs_box, gallery, cloud_url_panel],
                queue=False,
            )
            stop_btn.click(stop_and_save, outputs=[logs_box], queue=False)

        # ── Tab 3 : Store to DB ───────────────────────────────────────────────
        with gr.Tab("③ Store to Neo4j + Pinecone"):
            gr.Markdown(
                "Load a saved JSON state file and push all pages, elements, and "
                "visual embeddings to the graph and vector databases."
            )
            json_path_in = gr.Textbox(
                label="Path to saved JSON state",
                placeholder="./log/json_state/state_20240101_120000.json",
            )
            store_btn    = gr.Button("🚀 Store to databases")
            store_status = gr.Textbox(label="Result", interactive=False)
            store_btn.click(store_to_db, inputs=[json_path_in], outputs=[store_status], queue=False)

    return demo