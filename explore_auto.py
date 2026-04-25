import os
import json
import base64
import datetime
import config
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import RetryPolicy
from pydantic import SecretStr
from data.State import State
from tool.screen_content import screen_action  # only screen_action is still used here
from langchain_google_genai import ChatGoogleGenerativeAI
# ── NEW: route all screenshot+parse through client.run() ─────────────────────
from client import run as omniparser_run
from tool.adb_tools import take_adb_screenshot          # raw ADB screenshot only

os.environ["LANGCHAIN_TRACING_V2"] = config.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = config.LANGCHAIN_PROJECT

model = ChatGoogleGenerativeAI(
    model=config.LLM_MODEL,
    google_api_key=config.LLM_API_KEY,
)


# ── Helper: capture screenshot via ADB then send to client.run() ─────────────

def _capture_and_parse(device: str, app_name: str, step: int) -> dict:
    """
    1. Capture a raw screenshot from the ADB device and save it locally.
    2. Read it as base64 and send to client.run() (OmniParser endpoint).
    3. Return {"screenshot_path": str, "json_path": str} on success,
       or {"screenshot_path": str, "json_path": ""} if parsing fails.

    Args:
        device:   ADB device serial
        app_name: current app name (used for screenshot filename)
        step:     current step index (used for screenshot filename)

    Returns:
        dict with keys ``screenshot_path`` and ``json_path``
    """
    # ── Step 1: take raw ADB screenshot ──────────────────────────────────────
    # take_adb_screenshot saves the file locally and returns its path.
    screenshot_path: str = take_adb_screenshot(
        device=device, app_name=app_name, step=step
    )

    if not screenshot_path or not os.path.exists(screenshot_path):
        print(f"[explore_auto] Warning: screenshot not found at {screenshot_path}")
        return {"screenshot_path": screenshot_path or "", "json_path": ""}

    # ── Step 2: encode as base64 and forward to OmniParser via client.run() ──
    with open(screenshot_path, "rb") as fh:
        image_b64 = base64.b64encode(fh.read()).decode("utf-8")

    json_path: str = omniparser_run(image_b64)   # returns "" on failure

    if not json_path:
        print(f"[explore_auto] Warning: OmniParser returned no result for step {step}")

    return {"screenshot_path": screenshot_path, "json_path": json_path}


# ─────────────────────────────────────────────────────────────────────────────
#  Graph nodes
# ─────────────────────────────────────────────────────────────────────────────

def tsk_setting(state: State):
    """Determine target app name and initialise context."""
    message = [
        SystemMessage("Please reply with only the application name"),
        HumanMessage(
            f"The task goal is: {state['tsk']}, please infer the related application name. "
            "(The application name should not contain spaces) and reply with only one"
        ),
    ]
    llm_response = model.invoke(message)
    app_name = llm_response.content
    state["app_name"] = app_name

    from tool.adb_tools import get_device_size          # local import avoids circular deps
    state["device_info"] = get_device_size.invoke({"device": state["device"]})

    state["context"] = [
        HumanMessage(
            f"The task goal is: {state['tsk']}, the inferred application name is: {app_name}"
        )
    ]

    callback_info = {
        "app_name": state["app_name"],
        "device_info": state["device_info"],
        "task": state["tsk"],
    }
    if state.get("callback"):
        state["callback"](state, node_name="tsk_setting", info=callback_info)

    return state


def page_understand(state: State):
    """
    Capture the current screen via ADB, send the screenshot to client.run()
    (OmniParser endpoint), and store both paths in state.
    """
    parsed = _capture_and_parse(
        device=state["device"],
        app_name=state["app_name"],
        step=state["step"],
    )

    state["current_page_screenshot"] = parsed["screenshot_path"]
    state["current_page_json"] = parsed["json_path"]   # "" when parsing failed

    # Keep page_history in sync
    if parsed["screenshot_path"]:
        if not isinstance(state.get("page_history"), list):
            state["page_history"] = []
        state["page_history"].append(parsed["screenshot_path"])

    # Persist tool result for downstream nodes
    if not isinstance(state.get("tool_results"), list):
        state["tool_results"] = []
    state["tool_results"].append(
        {
            "tool_name": "omniparser_client",
            "screenshot": parsed["screenshot_path"],
            "json_path": parsed["json_path"],
        }
    )

    if state.get("callback"):
        state["callback"](state, node_name="page_understand")

    return state


def perform_action(state: State):
    """
    Use the React agent to decide and execute the next UI action.

    Reads the annotated screenshot and parsed JSON produced by page_understand,
    constructs a multimodal prompt, and calls the LLM-backed action agent.
    """
    action_agent = create_react_agent(model, [screen_action])

    labeled_image_path = state.get("current_page_screenshot")
    json_labeled_path  = state.get("current_page_json")
    user_intent  = state.get("tsk", "No specific task")
    device       = state.get("device", "Unknown device")
    device_size  = state.get("device_info", {})

    # ── Screenshot → base64 ───────────────────────────────────────────────────
    if not labeled_image_path or not os.path.exists(labeled_image_path):
        print("[explore_auto] Warning: screenshot missing for perform_action.")
        image_data = ""
    else:
        with open(labeled_image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

    # ── JSON page content ─────────────────────────────────────────────────────
    if not json_labeled_path or not os.path.exists(json_labeled_path):
        print("[explore_auto] Warning: parsed JSON missing for perform_action.")
        page_json = "{}"
    else:
        with open(json_labeled_path, "r", encoding="utf-8") as f:
            page_json = f.read()

    # ── Build prompt messages ─────────────────────────────────────────────────
    messages = [
        SystemMessage(
            content=(
                "Below is the current page information and user intent, please analyse and "
                "recommend a reasonable next step based on this. Please only complete one step. "
                "All tool calls must include device to specify the operation device."
            )
        ),
        HumanMessage(
            content=(
                f"The current device is: {device}, "
                f"the screen size of the device is {device_size}. "
                f"The current task intent is: {user_intent}"
            )
        ),
        HumanMessage(
            content=(
                "Below is the parsed JSON data of the current page "
                "(the bbox is relative, please convert it to actual operation position "
                "based on screen size): \n" + page_json
            )
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Below is the base64 data of the annotated page screenshot:",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        ),
    ]

    state["context"].extend(messages)

    action_result  = action_agent.invoke({"messages": state["context"][-4:]})
    final_messages = action_result.get("messages", [])

    if final_messages:
        state["context"].append(final_messages[-1])
    else:
        state["context"].append(
            SystemMessage(content="No action decided due to an error.")
        )
        state.setdefault("errors", []).append(
            {"step": state["step"], "error": "No messages returned by action_agent"}
        )
        return state

    recommended_action = final_messages[-1].content.strip()
    state["recommend_action"] = recommended_action

    tool_messages = [msg for msg in final_messages if msg.type == "tool"]
    tool_output: dict = {}
    for tool_message in tool_messages:
        tool_output.update(json.loads(tool_message.content))

    if tool_output:
        if not isinstance(state.get("tool_results"), list):
            state["tool_results"] = []
        tool_output["tool_name"] = "screen_action"
        state["tool_results"].append(tool_output)

    step_record = {
        "step": state["step"],
        "recommended_action": recommended_action,
        "tool_result": tool_output,
        "source_page": state["current_page_screenshot"],
        "source_json": state["current_page_json"],
        "timestamp": datetime.datetime.now().isoformat(),
    }
    state.setdefault("history_steps", []).append(step_record)

    state["step"] += 1

    if state.get("callback"):
        state["callback"](state, node_name="perform_action")

    return state


def tsk_completed(state: State):
    """
    After ≥2 steps, use LLM + the last 3 screenshots to judge task completion.
    Also re-captures and re-parses the screen via client.run() when marking done.
    """
    if state["step"] < 2:
        return state["completed"]

    user_task = state.get("tsk", "No task description")

    # ── Step 1: generate completion criteria ──────────────────────────────────
    reflection_messages = [
        SystemMessage(
            content="You are a supportive intelligent assistant, helping to analyse task completion criteria."
        ),
        HumanMessage(
            content=(
                f"The user's task is: {user_task}\n"
                "Please describe clear and checkable task completion criteria. "
                "For example: 'When a certain element or status appears on the page, "
                "it indicates that the task is completed.'"
            )
        ),
    ]
    completion_criteria = model.invoke(reflection_messages).content.strip()
    state["context"].append(
        SystemMessage(content=f"Generated task completion judgment criteria: {completion_criteria}")
    )

    # ── Step 2: collect up to 3 recent screenshots ────────────────────────────
    recent_images = (state.get("page_history") or [])[-3:]

    image_messages = []
    for idx, img_path in enumerate(recent_images, start=1):
        if img_path and os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            image_messages.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": f"Below is the base64 data of the {idx}th screenshot:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}},
                    ]
                )
            )
        else:
            image_messages.append(
                HumanMessage(content=f"Cannot find the {idx}th screenshot path: {img_path}")
            )

    judgement_messages = [
        SystemMessage(
            content=(
                "You are a page judgment assistant, you will judge whether the task is "
                "completed based on the given completion criteria and current page screenshot."
            )
        ),
        HumanMessage(
            content=(
                f"Completion criteria: {completion_criteria}\n"
                "Please judge whether the task is completed based on the following "
                "three page screenshots. Please note that if all three screenshots are "
                "complete, it indicates task failure, please directly reply yes or complete "
                "to end the program."
            )
        ),
    ] + image_messages

    judgement_answer = model.invoke(judgement_messages).content.strip()

    # ── Helper: re-capture & re-parse after completion / step limit ───────────
    def _refresh_state():
        parsed = _capture_and_parse(
            device=state["device"],
            app_name=state["app_name"],
            step=state["step"],
        )
        state["current_page_screenshot"] = parsed["screenshot_path"]
        state["current_page_json"] = parsed["json_path"]
        if parsed["screenshot_path"]:
            state.setdefault("page_history", []).append(parsed["screenshot_path"])

    if "yes" in judgement_answer.lower() or "complete" in judgement_answer.lower():
        state["completed"] = True
        _refresh_state()       # capture final state via client.run()
    else:
        state["completed"] = False

    state["context"].append(
        SystemMessage(content=f"LLM's answer on whether the task is completed: {judgement_answer}")
    )
    state["context"].append(
        SystemMessage(content=f"Final task completion status: {state['completed']}")
    )

    # Debug guard: force-stop after 5 steps and refresh
    if state["step"] > 5:
        _refresh_state()
        return True

    return state["completed"]


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_task(initial_state: State, progress_callback=None):
    """Build and execute the LangGraph task-automation graph."""
    graph_builder = StateGraph(State)

    graph_builder.add_node("tsk_setting",    tsk_setting,    retry=RetryPolicy(max_attempts=5))
    graph_builder.add_node("page_understand", page_understand, retry=RetryPolicy(max_attempts=5))
    graph_builder.add_node("perform_action", perform_action)

    graph_builder.add_edge(START, "tsk_setting")
    graph_builder.add_edge("tsk_setting", "page_understand")
    graph_builder.add_conditional_edges(
        "page_understand", tsk_completed, {True: END, False: "perform_action"}
    )
    graph_builder.add_edge("perform_action", "page_understand")

    graph = graph_builder.compile()

    if progress_callback is not None:
        initial_state["callback"] = progress_callback

    return graph.invoke(initial_state)