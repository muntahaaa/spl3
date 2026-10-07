"""Presentation-only streaming of sequential case execution."""
import json
import time


def outcome_with_success_values(result, task):
    """Keep machine-readable results while making passed-case values prominent."""
    payload = {k:v for k,v in result.items() if k != "close_actions"}
    if result.get("completed") is True:
        metrics = result.get("metrics") or {}
        cleanup = result.get("image_cleanup") or {}
        payload = {"success_summary": {
            "result": "PASSED", "task": task,
            "completion_evidence": result.get("completion_evidence") or result.get("message") or "Not provided",
            "steps_completed": result.get("steps_completed", "Not reported"),
            "stored_steps_replayed": result.get("replayed_steps",metrics.get("replayed_steps", "Not reported")),
            "react_steps": result.get("react_steps",metrics.get("react_steps", "Not reported")),
            "text_model_calls": metrics.get("text_calls", "Not reported"),
            "vision_model_calls": metrics.get("vision_calls", "Not reported"),
            "returned_home": result.get("returned_home", "Not reported"),
            "temporary_files_deleted": cleanup.get("deleted", "Not reported"),
            "cleanup_errors": cleanup.get("errors", "Not reported")}, **payload}
    return json.dumps(payload,ensure_ascii=False,indent=2)


def classify_outcome(outcome):
    try:
        result = json.loads(outcome)
    except (ValueError, TypeError):
        return "failed", str(outcome or "Execution returned no result")
    if not isinstance(result, dict):
        return "failed", "Invalid execution result"
    status = str(result.get("status", "")).lower()
    if result.get("completed") is True:
        return "completed", str(result.get("completion_evidence") or result.get("message") or "Task completed")
    if status.startswith("uncertain"):
        return "uncertain", str(result.get("message") or status)
    if status.startswith("skipped") or "skipped:" in status:
        return "skipped", str(result.get("message") or "Add a new test case")
    return "failed", str(result.get("message") or result.get("status") or "Task failed")


def report_summary(rows, finished=False):
    count = lambda mark: sum(row[2].endswith(mark) for row in rows)
    done, skipped, failed = count("Passed"), count("Skipped"), count("Failed")
    uncertain = count("Uncertain")
    pending = len(rows)-done-skipped-failed-uncertain
    title = "Execution report" if finished else "Execution progress"
    return (f"### {title}\n\nCases: **{len(rows)}** | ✅ Passed: **{done}** | "
            f"⏭ Skipped: **{skipped}** | ❌ Failed: **{failed}** | Uncertain: **{uncertain}** | Remaining: **{pending}**\n\n"
            f"Total execution time: **{sum(row[3] for row in rows):.1f}s**. "
            "See the case results and execution logs below for details.")


def stream_case_execution(cases, device, runner, update, clock=time.monotonic):
    rows = [[case["action_id"], case.get("source_task") or case.get("name") or case["action_id"], "⏳ Pending", 0.0, ""] for case in cases]
    accumulated = ""
    if not rows:
        yield "No stored test cases available.", "", update(visible=False), [], "", "", "No cases to execute", [], report_summary([],True)
        return
    if not device or device == "No devices found":
        yield "Select an ADB device before execution.", "", update(visible=False), [], "", "", "Device required", rows, "Select a device, then run again."
        return
    for index, case in enumerate(cases):
        name = rows[index][1]
        heading = f"Case {index+1}/{len(cases)}: {name} [{case['action_id']}]"
        rows[index][2] = "▶ Running"
        started = clock()
        outcome = ""
        logs = ""
        yield accumulated + heading, "Starting case", update(visible=False), [], "", "", heading, [list(row) for row in rows], report_summary(rows)
        try:
            for logs,outcome,popup,table,token,question in runner(name,device,action_id=case["action_id"]):
                rows[index][3] = round(clock()-started,1)
                paused = isinstance(popup,dict) and popup.get("visible") is True
                rows[index][2] = (chr(0x23f8) + " Waiting for input") if paused else (chr(0x25b6) + " Running")
                current = heading + " - Waiting for your input" if paused else heading
                yield accumulated + heading + "\n" + logs, outcome, popup, table, token, question, current, [list(row) for row in rows], report_summary(rows)
        except Exception as exc:
            outcome = json.dumps({"status":"error","message":f"{type(exc).__name__}: {exc}"})
            logs += f"\n[ERROR] {type(exc).__name__}: {exc}"
        status, message = classify_outcome(outcome)
        rows[index][2] = {"completed":"✅ Passed","skipped":"⏭ Skipped","uncertain":chr(0x2753)+" Uncertain","failed":"❌ Failed"}[status]
        rows[index][3] = round(clock()-started,1)
        rows[index][4] = message
        accumulated += heading + "\n" + logs + f"\n{rows[index][2]}: {message}\n\n"
        yield accumulated, outcome, update(visible=False), [], "", "", heading + " ? " + rows[index][2], [list(row) for row in rows], report_summary(rows)
    yield accumulated, outcome, update(visible=False), [], "", "", "Execution finished", rows, report_summary(rows,True)
