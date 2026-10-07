"""
ui.py  –  Gradio user interface
================================
Steps covered:
    ① Initialization  → device selection, task entry
    ② Exploration     → ADB actions + OmniParser parsing via client.run() after every screenshot
    ③ Save & Export   → state → JSON
    ④ Store to DB     → JSON → Neo4j
    ⑤ Chain Processing → chain_understand / chain_evolve
"""

import asyncio
import json
import os
import re
import threading
import time
import io
import contextlib

# pyrefly: ignore [missing-import]
import gradio as gr
from deployment_ui_blocks import DeploymentBlocks

from data.data_storage import json2db, state2json
from explor_human import capture_screenshot_only, single_human_explor
from state_manager import session
from tool.adb_tools import get_device_size, list_all_devices, list_devices_diagnostics
from data.State import State
from deployment import run_task as run_high_level_task

# ── Chain pipeline: service layer, job store, and response models ─────────────
from chain.chain_service import run_understand, run_evolve
from chain.task_store import create_job, get_job
from chain.chain_models import ChainJobStatus


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helper: take a raw screenshot (parsing is done inside
#  capture_screenshot_only via OmniParser — no second call needed)
# ─────────────────────────────────────────────────────────────────────────────

def _screenshot_and_parse(state: State) -> tuple[str, str]:
    """
    Call capture_screenshot_only() which captures an ADB screenshot AND
    automatically parses it with OmniParser (saving the annotated image
    and JSON).  Returns (screenshot_path, json_path).

    Args:
        state: current exploration State dict (mutated in-place)

    Returns:
        (screenshot_path, json_path) — json_path is "" if parsing failed.
    """
    updated = capture_screenshot_only(state)
    state.update(updated)

    screenshot_path: str = state.get("current_page_screenshot", "")
    json_path: str = state.get("current_page_json", "") or ""

    if not screenshot_path or not os.path.exists(screenshot_path):
        print(f"[ui] Warning: screenshot not found at '{screenshot_path}'")

    if not json_path:
        print(f"[ui] Warning: OmniParser returned no result for {screenshot_path}")

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
    show_elem  = action in ("tap", "text", "long_press", "swipe_short", "swipe_long", "back")
    show_text  = action == "text"
    show_swipe = action in ("swipe_short", "swipe_long")
    return (
        gr.update(visible=show_elem),
        gr.update(visible=show_text),
        gr.update(visible=show_swipe),
    )


def _run_high_level(task: str, device: str, force_fallback: bool = False, action_id=None):
    """
    Generator that streams real-time log lines from run_task via log_callback.
    Yields (reasoning_text, outcome_text, popup_col_update, close_actions_df_update) tuples so the UI updates every line.
    """
    import queue
    task   = (task or "").strip()
    device = (device or "").strip()
    if not task:
        yield "Error: provide a task description.", "", gr.update(visible=False), [], "", ""
        return
    if not device or device == "No devices found":
        yield "Error: select a valid ADB device.", "", gr.update(visible=False), [], "", ""
        return

    from deployment_control import Assistance, device_lock
    control = Assistance()
    lock = device_lock(device)
    if not lock.acquire(blocking=False):
        control.close()
        yield "Device already has an active deployment.", "", gr.update(visible=False), [], "", ""
        return
    log_queue: queue.Queue = queue.Queue()
    result_holder: dict = {}
    accumulated: list[str] = []

    def _log(line: str):
        print(line, flush=True)
        log_queue.put(line)

    def _worker():
        try:
            res = run_high_level_task(task=task, device=device, force_fallback=force_fallback, log_callback=_log,
                                      assistance_callback=control.request, action_id=action_id)
            result_holder["result"] = res
        except Exception as exc:
            _log(f"[ERROR] Deployment worker failed: {type(exc).__name__}: {exc}")
            result_holder["result"] = {"status": "error", "message": str(exc), "close_actions": []}
        finally:
            control.close()
            lock.release()
            log_queue.put(None)  # sentinel

    _log(f"[UI] Deployment queued: task={task!r}, device={device}, force fallback={force_fallback}.")
    last_event_at = time.monotonic()
    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    # Stream lines until worker finishes
    while True:
        try:
            line = log_queue.get(timeout=5)
        except queue.Empty:
            idle = time.monotonic() - last_event_at
            waiting = f"[WAIT] Background operation still running; {idle:.0f}s since the last event. Last operation: {accumulated[-1] if accumulated else 'starting worker'}"
            pending = control.pending
            question = str(pending.get("question") or pending.get("missing")) if pending else ""
            yield "\n".join(accumulated + [waiting]), ("Waiting for your input: " + question if pending else waiting), gr.update(visible=bool(pending)), [], control.token, question
            continue
        last_event_at = time.monotonic()
        if line is None:
            break
        accumulated.append(line)
        pending = control.pending
        question = str(pending.get("question") or pending.get("missing")) if pending else ""
        yield "\n".join(accumulated), ("Waiting for your input: " + question if pending else f"Running: {line}"), gr.update(visible=bool(pending)), [], control.token, question

    # Worker done — format final outcome
    result   = result_holder.get("result", {})
    from deployment_report import outcome_with_success_values
    outcome = outcome_with_success_values(result, task)
    close_actions = result.get("close_actions", [])
    show_popup    = bool(close_actions)
    
    formatted_actions = []
    for act in close_actions:
        name = act.get("name", "")
        description = act.get("description", "")
        desc = f"{name}: {description}" if name and description else (name or description)
        score = act.get("similarity_score") or act.get("score") or 0.65
        formatted_actions.append([act.get("action_id", ""), desc, score])

    yield "\n".join(accumulated), outcome, gr.update(visible=show_popup), formatted_actions, "", ""


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
    Execute a single human-driven action (tap / text / swipe / back / long_press).
    single_human_explor already captures and parses the post-action screenshot
    via capture_screenshot_only → OmniParser, so no second parse is needed.
    """
    state = session.get_state()
    if state is None:
        return "Error: initialize a device first.", []

    resolved_elem = int(element_number) if element_number is not None else None

    # ── 1. Execute the action via explor_human ────────────────────────────────
    #    single_human_explor internally calls capture_screenshot_only() which
    #    takes the screenshot AND parses it with OmniParser in one pass.
    updated = single_human_explor(
        state,
        action,
        element_number=resolved_elem,
        text_input=text_input,
        swipe_direction=swipe_direction,
    )

    session.set_state(updated)

    # ── 2. Read paths that were already set by single_human_explor ────────────
    post_screenshot: str = updated.get("current_page_screenshot", "")
    json_path: str = updated.get("current_page_json", "") or ""

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
        return f"✅ Stored to Neo4j. Task ID: {task_id}"
    except Exception as exc:
        return f"Error: {exc}"


def get_high_level_actions_with_app():
    query = """
    MATCH (a:Action)
    WHERE coalesce(a.is_high_level, false) = true
    OPTIONAL MATCH (p:Page)-[:HAS_ELEMENT]->(e:Element)<-[:COMPOSED_OF]-(a)
    RETURN a, collect(properties(p)) as pages_data
    """
    try:
        from data.graph_db import Neo4jDatabase
        import config
        temp_db = Neo4jDatabase(config.Neo4j_URI, config.Neo4j_AUTH, database=config.Neo4j_DB)
        actions = []
        with temp_db.driver.session(database=temp_db.database) as session:
            result = session.run(query)
            for record in result:
                action = dict(record["a"])
                pages = record.get("pages_data") or []

                app_name = None

                # 1. Action node property directly
                if action.get("app_name") and action["app_name"] not in ("unknown_app", "human_exploration"):
                    app_name = action["app_name"]

                # 2. Page app_name property (if already written to node)
                if not app_name:
                    for p in pages:
                        pan = p.get("app_name")
                        if pan and pan not in ("unknown_app", "human_exploration"):
                            app_name = pan
                            break

                # 3. Extract from raw_page_url (e.g. log/screenshots/Clock/...)
                if not app_name:
                    for p in pages:
                        url = p.get("raw_page_url")
                        if url:
                            norm_url = url.replace("\\", "/").strip()
                            parts = norm_url.split("/")
                            if "screenshots" in parts:
                                idx = parts.index("screenshots")
                                if idx + 1 < len(parts):
                                    cand = parts[idx + 1]
                                    if cand and cand not in ("unknown_app", "deployment", "human_exploration"):
                                        app_name = cand
                                        break

                # 4. Check page other_info
                if not app_name:
                    for p in pages:
                        oi_raw = p.get("other_info")
                        if oi_raw:
                            try:
                                oi = json.loads(oi_raw) if isinstance(oi_raw, str) else oi_raw
                                cand = oi.get("app_name")
                                if cand and cand not in ("unknown_app", "human_exploration"):
                                    app_name = cand
                                    break
                            except Exception:
                                pass

                # 5. Extract from page description
                if not app_name:
                    for p in pages:
                        desc = p.get("description")
                        if not desc:
                            continue
                        if " — Step " in desc:
                            cand = desc.split(" — Step ")[0].strip()
                            if cand and cand not in ("unknown_app", "human_exploration"):
                                app_name = cand
                                break
                        m = re.search(r"['\"]([A-Za-z0-9_-]+)['\"]\s+app", desc, re.IGNORECASE)
                        if m:
                            app_name = m.group(1)
                            break

                # 6. Fallback inference from action name or source_task
                if not app_name:
                    text_corpus = f"{action.get('name', '')} {action.get('source_task', '')}".lower()
                    if "clock" in text_corpus or "alarm" in text_corpus:
                        app_name = "Clock"
                    elif "photo" in text_corpus:
                        app_name = "Photos"
                    elif "gallery" in text_corpus:
                        app_name = "Gallery"
                    elif "setting" in text_corpus:
                        app_name = "Settings"
                    elif "weather" in text_corpus:
                        app_name = "Weather"
                    elif "youtube" in text_corpus:
                        app_name = "YouTube"
                    else:
                        app_name = "App"

                if "element_sequence" in action and isinstance(action["element_sequence"], str):
                    try:
                        action["element_sequence"] = json.loads(action["element_sequence"])
                    except json.JSONDecodeError:
                        pass

                if str(action.get("source_task") or "").strip().casefold() in {"unknown task", "unknown", "n/a"}:
                    action["source_task"] = action.get("name") or ""
                action["app_name"] = app_name
                if not action.get("action_id"):
                    action["action_id"] = action.get("id") or (str(record["a"].element_id) if hasattr(record["a"], "element_id") else "")

                actions.append(action)
        temp_db.close()
        return actions
    except Exception as exc:
        print(f"Error fetching high-level actions(test cases) with app: {exc}")
        return []



def load_and_filter_actions(search_query=""):
    actions = get_high_level_actions_with_app()
    search_query = (search_query or "").strip().lower()
    
    rows = []
    for act in actions:
        app = act.get("app_name", "App")
        task = act.get("source_task") or act.get("name", "N/A")
        
        if search_query:
            if search_query not in app.lower() and search_query not in task.lower():
                continue
                
        rows.append([app, task])
        
    return rows


def stored_case_choices(actions):
    """Build stable Gradio choices from stored actions with usable IDs."""
    choices = []
    for action in actions or []:
        action_id = action.get("action_id")
        if not action_id:
            continue
        label = action.get("source_task") or action.get("name") or action_id
        choices.append((str(label), str(action_id)))
    return choices


def select_stored_cases(actions, selected_ids):
    """Resolve an explicit multi-case selection without silently running others."""
    if isinstance(selected_ids, str):
        selected_ids = [selected_ids]
    selected_ids = selected_ids or []
    by_id = {
        str(action.get("action_id")): action
        for action in (actions or [])
        if action.get("action_id")
    }
    selected = []
    seen = set()
    for action_id in selected_ids:
        action_id = str(action_id)
        if action_id in by_id and action_id not in seen:
            selected.append(by_id[action_id])
            seen.add(action_id)
    return selected


# ─────────────────────────────────────────────────────────────────────────────
#  Gradio layout
# ─────────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with DeploymentBlocks(title="Human Explorer", css="#deployment-help {position:fixed; top:15%; left:15%; width:70%; max-height:70vh; overflow:auto; z-index:1000; background:white; padding:24px; border:2px solid #777; box-shadow:0 4px 40px #555;}") as demo:
        gr.Markdown(
            "# 📱 Vision QA - Automate your testing process\n"
            "**3-step pipeline:** "
            "① Explore (ADB actions + screenshots) → "
            "② Save session to JSON → "
            "③ Push to Neo4j"
        )


        assistance_token = gr.State("")
        with gr.Column(visible=False, elem_id="deployment-help") as popup_col:
            gr.Markdown("### Missing app or element: execution paused")
            assistance_question = gr.Markdown("The required question will appear here.")
            assistance_info = gr.Textbox(label="Answer the question above")
            close_actions_df = gr.Dataframe(visible=False)
            assistance_status = gr.Textbox(label="Assistance response", interactive=False)
            with gr.Row():
                provide_info_btn = gr.Button("Provide information and continue")
                skip_task_btn = gr.Button("Skip task - add a new test case")
        from deployment_control import answer_request
        provide_info_btn.click(lambda token,info: answer_request(token,info),
            inputs=[assistance_token,assistance_info], outputs=[assistance_status], queue=False)
        skip_task_btn.click(lambda token: answer_request(token,skip=True),
            inputs=[assistance_token], outputs=[assistance_status], queue=False)


        with gr.Tabs() as tabs_container:
            # ── Tab 1 : Initialization ────────────────────────────────────────────
            with gr.Tab("① Initialization", id=1):
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
            with gr.Tab("② Exploration", id=2):
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
                    ["tap", "text", "long_press", "swipe_short", "swipe_long", "back"], label="Action"
                )
                element_num = gr.Number(
                    label="Element ID",
                    info="Required for tap / long_press / swipe_short / swipe_long. Optional for text and back.",
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
            with gr.Tab("③ Store to Neo4j", id=3):
                gr.Markdown(
                    "Load a saved JSON state file and push its pages, elements, and actions to Neo4j."
                )
                json_path_in = gr.Textbox(
                    label="Path to saved JSON state",
                    placeholder="./log/json_state/state_20240101_120000.json",
                )
                store_btn    = gr.Button("🚀 Store to databases")
                store_status = gr.Textbox(label="Result", interactive=False)
                store_btn.click(store_to_db, inputs=[json_path_in], outputs=[store_status], queue=False)

            # ── Tab 4 : Chain Processing ──────────────────────────────────────────
            with gr.Tab("④ Chain Processing", id=4):
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

            # ── Tab 5 : High-Level Execution ─────────────────────────────────────
            with gr.Tab("⑤ High-Level Execution", id=5):
                gr.Markdown(
                    "Run high-level task execution using stored actions in Neo4j.\n"
                    "Provide the task text and select the device; logs show the reasoning process."
                )
                hl_task_input = gr.Textbox(
                    label="High-level task",
                    placeholder="e.g. Check today's weather in the Weather app",
                )
                hl_device_radio = gr.Radio(
                    label="Select ADB device",
                    choices=_get_devices(),
                )
                hl_refresh_btn = gr.Button("🔄 Refresh devices")
                hl_run_btn = gr.Button("▶ Run high-level task")

                hl_reasoning = gr.TextArea(
                    label="Deployment progress / process logs",
                    interactive=False,
                    lines=22,
                )
                hl_outcome = gr.TextArea(
                    label="Outcome",
                    interactive=False,
                    lines=8,
                )

                hl_refresh_btn.click(
                    lambda: gr.update(choices=_get_devices()),
                    outputs=[hl_device_radio],
                    queue=False,
                )
                hl_run_btn.click(
                    _run_high_level,
                    inputs=[hl_task_input, hl_device_radio],
                    outputs=[hl_reasoning, hl_outcome, popup_col, close_actions_df, assistance_token, assistance_question],
                    concurrency_id="deployment", concurrency_limit=1,
                )

            # ── Tab 6 : High-Level Actions ────────────────────────────────────────
            with gr.Tab("⑥ Stored test cases", id=6) as actions_tab:
                gr.Markdown(
                    "View and search all test cases stored in database."
                )
                with gr.Row():
                    search_input = gr.Textbox(
                        label="Search Action",
                        placeholder="Type app name or task description to filter...",
                    )
                    refresh_actions_btn = gr.Button("🔄 Refresh list")
                
                actions_df = gr.Dataframe(
                    headers=["App name", "Task description"],
                    datatype=["str", "str"],
                    interactive=False,
                )
                
                case_choice = gr.Dropdown(label="Select stored test case", choices=[])
                with gr.Row():
                    run_selected_btn = gr.Button("Run selected test case")
                    delete_case_btn = gr.Button("Delete selected test case")
                case_status = gr.Textbox(label="Case management result", interactive=False)

                with gr.Group():
                    gr.Markdown(
                        "### Run multiple stored test cases\n"
                        "Choose the cases to add to this execution. They will run one by one "
                        "in the order shown, followed by a combined outcome report."
                    )
                    batch_case_choices = gr.CheckboxGroup(
                        label="Test cases to execute",
                        choices=[],
                    )
                    with gr.Row():
                        select_all_cases_btn = gr.Button("Select all stored cases")
                        clear_case_selection_btn = gr.Button("Clear selection")
                        run_batch_btn = gr.Button("Run selected test cases", variant="primary")

                def case_selector_updates():
                    actions = get_high_level_actions_with_app()
                    choices = stored_case_choices(actions)
                    return gr.update(choices=choices), gr.update(choices=choices, value=[])

                def select_all_stored_cases():
                    choices = stored_case_choices(get_high_level_actions_with_app())
                    return gr.update(choices=choices, value=[action_id for _, action_id in choices])

                cases_device = gr.Radio(label="ADB device for stored test cases", choices=_get_devices())
                cases_device_refresh = gr.Button("Refresh execution devices")
                cases_current = gr.Textbox(label="Currently executing", interactive=False)
                cases_results = gr.Dataframe(headers=["Case ID", "Test case", "Status", "Time (s)", "Result"],
                    datatype=["str", "str", "str", "number", "str"], interactive=False)
                cases_report = gr.Markdown("Run a case to see its execution report.")
                cases_logs = gr.TextArea(label="Stored test case execution logs", lines=24, interactive=False)
                cases_outcome = gr.TextArea(label="Current case outcome", lines=8, interactive=False)
                cases_device_refresh.click(lambda: gr.update(choices=_get_devices()),outputs=[cases_device],queue=False)

                from deployment_report import stream_case_execution
                def run_selected(case_id, device):
                    actions = get_high_level_actions_with_app()
                    selected = next((a for a in actions if a.get("action_id") == case_id), None)
                    if selected is None:
                        yield "Select an existing stored case.", "", gr.update(visible=False), [], "", "", "No case selected", [], "Select a stored test case."
                        return
                    yield from stream_case_execution([selected],device,_run_high_level,gr.update)

                def run_selected_cases(case_ids, device):
                    cases = select_stored_cases(get_high_level_actions_with_app(), case_ids)
                    if not cases:
                        yield "Select one or more stored test cases.", "", gr.update(visible=False), [], "", "", "No cases selected", [], "Select at least one test case, then run again."
                        return
                    yield from stream_case_execution(cases,device,_run_high_level,gr.update)

                def delete_selected(case_id):
                    from deployment import db
                    from deployment_cases import delete_case
                    try:
                        from deployment_control import deployment_active
                        if deployment_active():
                            raise RuntimeError("Wait for active deployment to finish before deleting a test case")
                        result = delete_case(db,case_id)
                        single_update, batch_update = case_selector_updates()
                        return json.dumps(result,indent=2),single_update,batch_update,load_and_filter_actions("")
                    except Exception as exc:
                        single_update, batch_update = case_selector_updates()
                        return f"Deletion failed: {exc}",single_update,batch_update,load_and_filter_actions("")

                refresh_actions_btn.click(case_selector_updates, outputs=[case_choice,batch_case_choices],queue=False)
                actions_tab.select(case_selector_updates, outputs=[case_choice,batch_case_choices],queue=False)
                execution_outputs = [cases_logs,cases_outcome,popup_col,close_actions_df,assistance_token,assistance_question,cases_current,cases_results,cases_report]
                run_selected_btn.click(run_selected,inputs=[case_choice,cases_device],outputs=execution_outputs,concurrency_id="deployment",concurrency_limit=1)
                run_batch_btn.click(run_selected_cases,inputs=[batch_case_choices,cases_device],outputs=execution_outputs,concurrency_id="deployment",concurrency_limit=1)
                select_all_cases_btn.click(select_all_stored_cases,outputs=[batch_case_choices],queue=False)
                clear_case_selection_btn.click(lambda: gr.update(value=[]),outputs=[batch_case_choices],queue=False)
                delete_case_btn.click(delete_selected,inputs=[case_choice],outputs=[case_status,case_choice,batch_case_choices,actions_df])

                search_input.change(
                    load_and_filter_actions,
                    inputs=[search_input],
                    outputs=[actions_df],
                    queue=False,
                )
                refresh_actions_btn.click(
                    load_and_filter_actions,
                    inputs=[search_input],
                    outputs=[actions_df],
                    queue=False,
                )
                actions_tab.select(
                    load_and_filter_actions,
                    inputs=[search_input],
                    outputs=[actions_df],
                    queue=False,
                )

    return demo.queue()
