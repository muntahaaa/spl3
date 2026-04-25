"""
ui.py  –  Gradio user interface
================================
Steps covered:
    ① Initialization  → device selection, task entry
    ② Exploration     → ADB actions + OmniParser parsing via client.run() after every screenshot
    ③ Save & Export   → state → JSON
    ④ Store to DB     → JSON → Neo4j + Pinecone
    ⑤ Chain Processing → chain_understand / chain_evolve
"""

import asyncio
import base64
import json
import os
import threading
import time

import gradio as gr

from data.data_storage import json2db, state2json
from explor_human import capture_screenshot_only, single_human_explor
from state_manager import session
from tool.adb_tools import get_device_size, list_all_devices, list_devices_diagnostics
from data.State import State

# ── OmniParser endpoint ───────────────────────────────────────────────────────
from OmniParser.client import run as omniparser_run

# ── Chain pipeline: service layer, job store, and response models ─────────────
from chain.chain_service import run_understand, run_evolve
from chain.task_store import create_job, get_job
from chain.chain_models import ChainJobStatus


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helper: take a raw screenshot then pipe through client.run()
# ─────────────────────────────────────────────────────────────────────────────

def _screenshot_and_parse(state: State) -> tuple[str, str]:
    """
    1. Call capture_screenshot_only() to get a raw ADB screenshot path.
    2. Read the file, base64-encode it, and send to client.run() (OmniParser).
    3. Store both paths back into state and return (screenshot_path, json_path).

    Args:
        state: current exploration State dict (mutated in-place)

    Returns:
        (screenshot_path, json_path) — json_path is "" if parsing failed.
    """
    # ── 1. Capture raw screenshot via ADB ────────────────────────────────────
    updated = capture_screenshot_only(state)

    screenshot_path: str = updated.get("current_page_screenshot", "")
    if not screenshot_path or not os.path.exists(screenshot_path):
        print(f"[ui] Warning: screenshot not found at '{screenshot_path}'")
        state.update(updated)
        state["current_page_json"] = ""
        return screenshot_path, ""

    # ── 2. Encode screenshot and send to OmniParser via client.run() ─────────
    with open(screenshot_path, "rb") as fh:
        image_b64 = base64.b64encode(fh.read()).decode("utf-8")

    json_path: str = omniparser_run(image_b64)   # "" on failure

    if not json_path:
        print(f"[ui] Warning: OmniParser returned no result for {screenshot_path}")

    # ── 3. Persist both paths in state ───────────────────────────────────────
    updated["current_page_json"] = json_path
    state.update(updated)

    return screenshot_path, json_path


def _labeled_image_from_json(json_path: str) -> str:
    if not json_path:
        return ""
    base = os.path.splitext(os.path.basename(json_path))[0]
    img_path = os.path.join("labeled_image", "img", f"{base}.png")
    return img_path if os.path.exists(img_path) else ""


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_devices():
    devices = list_all_devices()
    return devices if devices else ["No devices found"]


def _action_visibility(action: str):
    show_elem  = action in ("tap", "text", "long_press", "swipe", "back")
    show_text  = action == "text"
    show_swipe = action == "swipe"
    return (
        gr.update(visible=show_elem),
        gr.update(visible=show_text),
        gr.update(visible=show_swipe),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Initialization callbacks
# ─────────────────────────────────────────────────────────────────────────────

def refresh_devices():
    try:
        devices = _get_devices()
        raw = list_devices_diagnostics()
        return gr.update(choices=devices), raw
    except Exception as exc:
        return gr.update(choices=["No devices found"]), f"Error: {exc}"


def initialize_device(device: str, task_info: str, app_name: str):
    if not task_info:
        return "Error: task information cannot be empty."
    if not device or device == "No devices found":
        return "Error: select a valid ADB device."

    app_name = app_name.strip() if app_name and app_name.strip() else "human_exploration"

    device_info = get_device_size.invoke({"device": device})
    if "error" in device_info:
        return f"Error reading device size: {device_info['error']}"

    state = State(
        tsk=task_info,
        app_name=app_name,
        completed=False,
        step=0,
        history_steps=[],
        page_history=[],
        current_page_screenshot=None,
        current_page_json=None,
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
    return f"✅ Initialized device '{device}' — app: {app_name} — task: {task_info}"


# ─────────────────────────────────────────────────────────────────────────────
#  Exploration callbacks
# ─────────────────────────────────────────────────────────────────────────────

def start_session():
    """
    Take an initial screenshot, send it through client.run() for OmniParser
    annotation, and update the session state with both result paths.
    """
    state = session.get_state()
    if state is None:
        return "Error: initialize a device first.", []

    # ── Capture + parse via client.run() ─────────────────────────────────────
    screenshot_path, json_path = _screenshot_and_parse(state)
    session.set_state(state)

    # Track screenshot in gallery list
    gallery_path = _labeled_image_from_json(json_path) or screenshot_path
    if gallery_path and gallery_path not in session.user_page_storage:
        session.user_page_storage.append(gallery_path)

    msg = (
        "Session started.\n"
        f"📷 Screenshot saved → {screenshot_path}\n"
        f"📊 Parsed with OmniParser → {json_path if json_path else '(parsing failed)'}"
    )
    return msg, session.user_page_storage


def perform_action(action, element_number, text_input, swipe_direction):
    """
    Execute a single human-driven action (tap / text / swipe / back / long_press),
    then immediately capture and parse the resulting screen via client.run().
    """
    state = session.get_state()
    if state is None:
        return "Error: initialize a device first.", []

    resolved_elem = int(element_number) if element_number is not None else None

    # ── 1. Execute the action via explor_human ────────────────────────────────
    updated = single_human_explor(
        state,
        action,
        element_number=resolved_elem,
        text_input=text_input,
        swipe_direction=swipe_direction,
    )

    # ── 2. Re-parse the post-action screenshot through client.run() ───────────
    #    single_human_explor may have already saved a new screenshot; we read it
    #    and send it to OmniParser to get the annotated image + JSON.
    post_screenshot: str = updated.get("current_page_screenshot", "")
    json_path = ""

    if post_screenshot and os.path.exists(post_screenshot):
        with open(post_screenshot, "rb") as fh:
            image_b64 = base64.b64encode(fh.read()).decode("utf-8")
        json_path = omniparser_run(image_b64)          # "" on failure
        if not json_path:
            print(f"[ui] Warning: OmniParser returned no result for step {updated.get('step', '?')}")
    else:
        print(f"[ui] Warning: post-action screenshot not found at '{post_screenshot}'")

    updated["current_page_json"] = json_path
    session.set_state(updated)

    # ── 3. Update log and gallery ─────────────────────────────────────────────
    log_entry = {
        "step":       updated["step"],
        "action":     action,
        "completed":  updated["completed"],
        "screenshot": post_screenshot,
        "parsed":     bool(json_path),
    }
    session.user_log_storage.append(json.dumps(log_entry, ensure_ascii=False))

    labeled = _labeled_image_from_json(json_path) if json_path else ""
    gallery_path = labeled or post_screenshot
    if gallery_path and gallery_path not in session.user_page_storage:
        session.user_page_storage.append(gallery_path)

    status = (
        f"Step {updated['step']} done — '{action}' executed.\n"
        f"📷 New screenshot → {post_screenshot}\n"
        f"📊 Parsed with OmniParser → {json_path if json_path else '(parsing failed)'}"
    )
    log_text = "\n".join(session.user_log_storage) + "\n" + status
    return log_text, session.user_page_storage


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
            app_name_input = gr.Textbox(
                label="App name",
                placeholder="e.g. YouTube, Settings, com.example.app",
                info="Name of the app being explored. Used in page descriptions and screenshot paths.",
            )
            task_input   = gr.Textbox(
                label="Task description",
                placeholder="e.g. Log in and navigate to Settings",
            )
            init_btn    = gr.Button("✅ Initialize")
            init_status = gr.Textbox(label="Status", interactive=False)

            refresh_btn.click(refresh_devices, outputs=[device_radio, devices_box], queue=False)
            demo.load(refresh_devices, outputs=[device_radio, devices_box], queue=False)
            init_btn.click(
                initialize_device,
                inputs=[device_radio, task_input, app_name_input],
                outputs=[init_status],
                queue=False,
            )

        # ── Tab 2 : Exploration ───────────────────────────────────────────────
        with gr.Tab("② Exploration"):
            gr.Markdown(
                "### Workflow per step\n"
                "1. Click **Start session** to take the initial screenshot "
                "and send it to OmniParser.\n"
                "2. Select an action (tap, swipe, text, etc.) and click **Perform action**.\n"
                "3. The post-action screenshot is automatically sent to OmniParser.\n"
                "4. Repeat until the task is complete.\n"
                "5. Click **Stop & save to JSON** to finalise the exploration."
            )

            start_btn    = gr.Button("▶ Start session (take initial screenshot)")
            action_radio = gr.Radio(
                ["tap", "text", "long_press", "swipe", "back"], label="Action"
            )
            element_num = gr.Number(
                label="Element ID",
                info="Required for tap / long_press / swipe. Optional for text and back.",
                precision=0, visible=False,
            )
            text_in   = gr.Textbox(label="Text input", visible=False)
            swipe_dir = gr.Radio(["up", "down", "left", "right"], label="Swipe direction", visible=False)
            perform_btn = gr.Button("⚡ Perform action")
            stop_btn    = gr.Button("🛑 Stop & save to JSON")
            logs_box    = gr.TextArea(label="Step log", interactive=False, lines=10)
            gallery     = gr.Gallery(label="Labeled screenshots", height=500)

            action_radio.change(
                _action_visibility,
                inputs=[action_radio],
                outputs=[element_num, text_in, swipe_dir],
                queue=False,
            )
            start_btn.click(start_session, outputs=[logs_box, gallery], queue=False)
            perform_btn.click(
                perform_action,
                inputs=[action_radio, element_num, text_in, swipe_dir],
                outputs=[logs_box, gallery],
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

        # ── Tab 4 : Chain Processing ──────────────────────────────────────────
        with gr.Tab("④ Chain Processing"):
            gr.Markdown(
                "Run understanding and evolution on a stored chain.\n"
                "Requires the chain's data to already be in Neo4j (use Tab ③ first).\n\n"
                "Both operations run as **background jobs** so the UI stays responsive.\n"
                "Click **▶ Start**, then use **🔄 Poll status** to check progress."
            )
            start_page_id_input = gr.Textbox(
                label="Start Page ID",
                placeholder="page_abc123",
            )
            with gr.Row():
                understand_btn = gr.Button("🧠 Start chain_understand")
                evolve_btn     = gr.Button("🚀 Start chain_evolve")

            job_id_box = gr.Textbox(
                label="Job ID (copy this to poll for status)",
                interactive=False,
            )
            poll_btn        = gr.Button("🔄 Poll status")
            chain_status_box = gr.Textbox(label="Result", interactive=False, lines=5)

            # ── Helpers ───────────────────────────────────────────────────────

            def _launch_background(coro_fn, job_id: str, start_page_id: str) -> None:
                """
                Run an async coroutine from chain_service in a daemon thread so
                that Gradio's synchronous callback layer is not blocked.

                ``asyncio.run`` is safe here because each thread gets its own
                event loop — there is no existing loop to conflict with.
                """
                def _worker():
                    asyncio.run(coro_fn(job_id, start_page_id))

                t = threading.Thread(target=_worker, daemon=True)
                t.start()

            def _format_job_status(record: dict) -> str:
                """
                Convert a raw task_store record into a human-readable status
                string for the Gradio textbox, validated through ChainJobStatus.
                """
                model = ChainJobStatus(
                    job_id=record.get("job_id", ""),
                    status=record.get("status", "not_found"),
                    result=record.get("result"),
                    error=record.get("error"),
                )
                if model.status == "not_found":
                    return f"⚠️  Job '{model.job_id}' not found in store."
                if model.status == "pending":
                    return f"⏳ [{model.job_id}] Job is queued — not started yet."
                if model.status == "running":
                    return f"🔄 [{model.job_id}] Running…"
                if model.status == "done":
                    result_str = json.dumps(model.result, indent=2) if model.result else "—"
                    return f"✅ [{model.job_id}] Done.\n{result_str}"
                if model.status == "error":
                    return f"❌ [{model.job_id}] Error: {model.error}"
                return f"[{model.job_id}] status={model.status}"

            # ── Button callbacks ──────────────────────────────────────────────

            def start_chain_understand(page_id: str):
                """
                Create a job, launch run_understand in the background, and
                immediately return the job_id so the user can poll for results.
                """
                page_id = page_id.strip()
                if not page_id:
                    return "Error: provide a start_page_id.", ""
                job_id = create_job()
                _launch_background(run_understand, job_id, page_id)
                return (
                    f"🧠 chain_understand started.\nJob ID: {job_id}\n"
                    "Click '🔄 Poll status' to check progress.",
                    job_id,
                )

            def start_chain_evolve(page_id: str):
                """
                Create a job, launch run_evolve in the background, and
                immediately return the job_id.
                """
                page_id = page_id.strip()
                if not page_id:
                    return "Error: provide a start_page_id.", ""
                job_id = create_job()
                _launch_background(run_evolve, job_id, page_id)
                return (
                    f"🚀 chain_evolve started.\nJob ID: {job_id}\n"
                    "Click '🔄 Poll status' to check progress.",
                    job_id,
                )

            def poll_chain_status(job_id: str):
                """
                Look up the current job record in task_store and format it for
                the status textbox.
                """
                job_id = job_id.strip()
                if not job_id:
                    return "Error: no job ID to poll. Start a job first."
                record = get_job(job_id)
                return _format_job_status(record)

            # ── Wire buttons ──────────────────────────────────────────────────

            understand_btn.click(
                start_chain_understand,
                inputs=[start_page_id_input],
                outputs=[chain_status_box, job_id_box],
            )
            evolve_btn.click(
                start_chain_evolve,
                inputs=[start_page_id_input],
                outputs=[chain_status_box, job_id_box],
            )
            poll_btn.click(
                poll_chain_status,
                inputs=[job_id_box],
                outputs=[chain_status_box],
            )

    return demo