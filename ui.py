"""
ui.py  –  Gradio user interface
================================
Steps covered:
    ① Initialization  → device selection, task entry
    ② Exploration     → ADB actions + automatic OmniParser parsing after every screenshot
    ③ Save & Export   → state → JSON
    ④ Store to DB     → JSON → Neo4j + Pinecone
"""

import json

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
    # back now shows element_num so the user can optionally mark the back icon.
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
        raw     = list_devices_diagnostics()
        return gr.update(choices=devices), raw
    except Exception as exc:
        return gr.update(choices=["No devices found"]), f"Error: {exc}"


def initialize_device(device: str, task_info: str, app_name: str):
    if not task_info:
        return "Error: task information cannot be empty."
    if not device or device == "No devices found":
        return "Error: select a valid ADB device."

    # Normalise: strip whitespace, fall back to a safe default if left blank
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
    state = session.get_state()
    if state is None:
        return "Error: initialize a device first.", []

    updated = capture_screenshot_only(state)
    session.set_state(updated)

    screenshot = updated.get("current_page_screenshot", "")
    parsed_json = updated.get("current_page_json", "")
    msg = (
        f"Session started.\n"
        f"📷 Screenshot saved → {screenshot}\n"
        f"📊 Parsed with OmniParser → {parsed_json if parsed_json else '(parsing failed)'}"
    )
    return msg, session.user_page_storage


def perform_action(action, element_number, text_input, swipe_direction):
    state = session.get_state()
    if state is None:
        return "Error: initialize a device first.", []

    # element_number is shared across tap/long_press/swipe/back/text.
    # For back it is optional; single_human_explor handles None gracefully.
    resolved_elem = int(element_number) if element_number is not None else None

    updated = single_human_explor(
        state,
        action,
        element_number=resolved_elem,
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

    parsed_json = updated.get("current_page_json", "")
    status = (
        f"Step {updated['step']} done — '{action}' executed.\n"
        f"📷 New screenshot → {updated.get('current_page_screenshot')}\n"
        f"📊 Parsed with OmniParser → {parsed_json if parsed_json else '(parsing failed)'}"
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
                "1. Click **Start session** to take the initial screenshot.\n"
                "2. The screenshot is automatically parsed with **OmniParser**.\n"
                "3. Select an action (tap, swipe, text, etc.) and click **Perform action**.\n"
                "4. Repeat steps 1–3 until task is complete.\n"
                "5. Click **Stop & save to JSON** to finalize the exploration."
            )

            start_btn    = gr.Button("▶ Start session (take initial screenshot)")
            action_radio = gr.Radio(
                ["tap", "text", "long_press", "swipe", "back"], label="Action"
            )
            element_num = gr.Number(
                label="Element ID",
                info="Required for tap / long_press / swipe. Optional for text and back (marks the icon in the graph).",
                precision=0, visible=False,
            )
            text_in     = gr.Textbox(label="Text input", visible=False)
            swipe_dir   = gr.Radio(["up", "down", "left", "right"], label="Swipe direction", visible=False)
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
            start_btn.click(
                start_session,
                outputs=[logs_box, gallery],
                queue=False,
            )
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
                "Requires the chain's data to already be in Neo4j (use Tab ③ first)."
            )
            start_page_id_input = gr.Textbox(
                label="Start Page ID",
                placeholder="page_abc123"
            )
            with gr.Row():
                understand_btn = gr.Button("🧠 Run chain_understand")
                evolve_btn     = gr.Button("🚀 Run chain_evolve")
            chain_status_box = gr.Textbox(label="Result", interactive=False, lines=5)

            def run_chain_understand(page_id):
                import asyncio
                from chain_understand import process_and_update_chain
                if not page_id.strip():
                    return "Error: provide a start_page_id."
                try:
                    triplets = asyncio.run(process_and_update_chain(page_id.strip()))
                    return f"✅ chain_understand complete. Triplets processed: {len(triplets)}"
                except Exception as e:
                    return f"❌ Error: {e}"

            def run_chain_evolve(page_id):
                import asyncio
                from chain_evolve import evolve_chain_to_action
                if not page_id.strip():
                    return "Error: provide a start_page_id."
                try:
                    action_id = asyncio.run(evolve_chain_to_action(page_id.strip()))
                    if action_id:
                        return f"✅ chain_evolve complete. Action node ID: {action_id}"
                    return "⚠️ Chain evaluated as non-templateable — no node created."
                except Exception as e:
                    return f"❌ Error: {e}"

            understand_btn.click(run_chain_understand, inputs=[start_page_id_input], outputs=[chain_status_box])
            evolve_btn.click(run_chain_evolve, inputs=[start_page_id_input], outputs=[chain_status_box])

    return demo