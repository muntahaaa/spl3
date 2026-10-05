"""Deterministic stored-task replay with bounded, text-first ReAct.

No device, database, or model is initialized by importing this module. Dependencies
are supplied by deployment.py, and the same engine is exercised by offline tests.
"""
from __future__ import annotations

import copy
import difflib
import json
import math
import re
import unicodedata
import time
from collections import Counter
from functools import lru_cache


def decoded(value, default=None):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return default
    return value if value is not None else default


def normalize(value):
    # Preserve digits, punctuation and AM/PM; never collapse all numeric strings.
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def elements(page):
    value = decoded((page or {}).get("elements", (page or {}).get("elements_data", [])), [])
    return [e for e in value if isinstance(e, dict)] if isinstance(value, list) else []


def box(element):
    value = decoded(element.get("bbox", element.get("bounding_box", [])), [])
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        result = tuple(float(x) for x in value)
    except (ValueError, TypeError):
        return None
    if not all(math.isfinite(x) and x >= 0 for x in result):
        return None
    return result if result[2] > result[0] and result[3] > result[1] else None


def content(element):
    return normalize(element.get("content", element.get("text", "")))


def meaningful(page):
    result = []
    for e in elements(page):
        b = box(e)
        # Ignore only the Android status/debug strip, not numbers elsewhere.
        if b and max(b) <= 1 and b[3] <= .06:
            continue
        if content(e):
            result.append(e)
    return result


def compatible_apps(a, b):
    for key in ("app_package", "activity"):
        if a.get(key) and b.get(key) and a[key] != b[key]:
            return False
    return True


@lru_cache(maxsize=512)
def _signature(serialized):
    page = json.loads(serialized)
    return tuple((content(e), e.get("type", ""), e.get("selected"), box(e)) for e in meaningful(page))


def screen_score(expected, live):
    if not compatible_apps(expected, live):
        return 0.0
    a = _signature(json.dumps(expected, sort_keys=True))
    b = _signature(json.dumps(live, sort_keys=True))
    if not a or not b:
        return 0.0
    aa, bb = Counter(e[:3] for e in a), Counter(e[:3] for e in b)
    # Task values are not volatile: a wrong time/phone must not be outvoted by labels.
    numeric = lambda sig: Counter(t for t, _, _, _ in sig if re.search(r"\d", t))
    if numeric(a) != numeric(b):
        return 0.0
    weight = lambda key: .5 if key[0] in NAVIGATION else 1.0
    overlap = sum(count * weight(key) for key, count in (aa & bb).items())
    total = sum(count * weight(key) for counter in (aa, bb) for key, count in counter.items())
    dice = 2 * overlap / total
    # Layout is secondary evidence, never a substitute for content.
    distances = []
    for text, typ, selected, bounds in a:
        candidates = [x[3] for x in b if x[:3] == (text, typ, selected) and x[3]]
        if bounds and candidates and max(bounds) <= 1:
            distances.append(min(math.hypot((bounds[0]+bounds[2]-v[0]-v[2])/2,
                                           (bounds[1]+bounds[3]-v[1]-v[3])/2)
                                 for v in candidates if max(v) <= 1) if any(max(v) <= 1 for v in candidates) else 0)
    layout = sum(max(0, 1-d) for d in distances)/len(distances) if distances else 1
    return dice * (.95 + .05 * layout)


def screen_difference(expected, live):
    expected_texts = Counter(content(e) for e in meaningful(expected))
    live_texts = Counter(content(e) for e in meaningful(live))
    return {"missing": list((expected_texts-live_texts).elements())[:8],
            "unexpected": list((live_texts-expected_texts).elements())[:8]}


def target_index(target, live):
    """Content/type first; layout disambiguates, but never proves identity alone."""
    scores = []
    for i, e in enumerate(elements(live)):
        if not box(e) or not content(target) or not content(e):
            continue
        if target.get("type") and e.get("type") and target["type"] != e["type"]:
            continue
        ratio = difflib.SequenceMatcher(None, content(target), content(e)).ratio()
        if ratio < .93:
            continue
        distance = 1.0
        a, b = box(target), box(e)
        if a and b and (max(a) <= 1) == (max(b) <= 1):
            distance = math.hypot((a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2)
        scores.append((ratio + .05 * max(0, 1-distance*10), i))
    scores.sort(reverse=True)
    if not scores or (len(scores) > 1 and scores[0][0] - scores[1][0] < .025):
        return None
    return scores[0][1]


NAVIGATION = {"clock", "gallery", "contacts", "notes", "pictures", "picture", "albums", "album", "world clock", "alarm", "alarms", "settings", "apps", "home", "back"}
COMMIT = {"save", "done", "create", "add contact", "delete", "send", "confirm", "ok"}


def step_role(step):
    action = step.get("atomic_action")
    label = content(step.get("target", {}))
    if action == "text":
        return "input"
    if label in COMMIT:
        return "commit"
    # A create/open button is reusable only when it visibly opens a form/picker.
    source_labels = {content(e) for e in elements(step.get("source", {}))}
    destination_labels = {content(e) for e in elements(step.get("destination", {}))}
    opens_editor = bool(destination_labels & {"hour", "minute", "am", "pm", "set alarm", "first name", "phone", "title"}) and not bool(source_labels & {"hour", "minute", "am", "pm", "set alarm", "first name", "phone", "title"})
    if action == "tap" and label in {"add", "+", "new", "new alarm", "new contact", "new note"} and opens_editor:
        return "navigation"
    if action == "tap" and label in NAVIGATION:
        return "navigation"
    return "interaction"


def app_open_prefix(task, action):
    """Prove a simple app-opening subgoal from task text AND stored targets."""
    request = normalize(task).strip(" .!?")
    request = re.sub(r"^please\s+", "", request)
    match = re.fullmatch(r"(?:go to|navigate to|open|launch)\s+(?:the\s+)?([a-z][a-z0-9 -]*?)(?:\s+app)?", request)
    if not match:
        return 0
    app = match.group(1).strip()
    # Refuse compound goals and parameter-bearing requests.
    if re.search(r"\b(and|then|switch|set|create|with|at|tab|album|albums|picture|pictures)\b", app):
        return 0
    stored = normalize(action.get("source_task") or action.get("name"))
    opening = r"^(?:please\s+)?(?:go to|navigate to|open|launch)\s+(?:the\s+)?" + re.escape(app) + r"(?:\s+app)?(?=\s|[.,!?]|$)"
    if not re.search(opening, stored):
        return 0
    for index, step in enumerate(action.get("element_sequence", [])):
        if step_role(step) != "navigation":
            return 0
        if step.get("atomic_action") == "tap" and content(step.get("target", {})) == app:
            # A recording lacking destination JSON cannot prove this subgoal.
            return index + 1 if meaningful(step.get("destination", {})) else 0
    return 0


def navigation_intent(task):
    """Canonicalize only explicit app/tab navigation; other goals need reasoning."""
    text = re.sub(r"^please\s+", "", normalize(task).strip(" .!?"))
    match = re.fullmatch(r"(?:go to|navigate to|open|launch)\s+(?:the\s+)?([a-z][a-z -]*?)(?:\s+app)?(?:\s+and\s+(?:then\s+)?(?:switch to|navigate to|go to|open)\s+(?:the\s+)?([a-z][a-z -]*?)(?:\s+tab)?)?", text)
    if not match:
        return None
    aliases = {"albums": "album", "pictures": "picture", "alarms": "alarm"}
    labels = tuple(aliases.get(x, x) for x in match.groups() if x)
    if not labels or any(label not in NAVIGATION for label in labels):
        return None
    return labels


def equivalent_navigation(task, action):
    intent = navigation_intent(task)
    if not intent:
        return False
    texts = [action.get("source_task"), action.get("name")]
    if not any(navigation_intent(text) == intent for text in texts if text):
        return False
    # Select by task intent only. Source/target validity belongs to replay alignment,
    # not retrieval: historical parser descriptions can be noisy or incomplete.
    return bool(action.get("element_sequence"))


REPLAY_SOURCE_THRESHOLD = .75


def replay_source_match(step, live):
    """Use screen context plus target identity; navigation tolerates layout drift."""
    source = step.get("source", {})
    score = screen_score(source, live)
    if not compatible_apps(source, live):
        return score, False, "app identity differs"
    if step.get("atomic_action") not in {"tap", "long_press", "text"}:
        return score, score >= REPLAY_SOURCE_THRESHOLD, "screen context"
    if step.get("target_metadata_error"):
        return score, False, step["target_metadata_error"]
    index = target_index(step.get("target", {}), live)
    if index is None:
        return score, False, "target missing or ambiguous"
    exact_navigation = step_role(step) == "navigation" and content(step.get("target", {})) == content(elements(live)[index])
    accepted = score >= REPLAY_SOURCE_THRESHOLD or exact_navigation
    return score, accepted, "unique navigation target" if exact_navigation else "screen context and unique target"


def page_snapshot(page):
    return {k: v for k, v in {
        "page_id": page.get("page_id"), "elements": elements(page),
        "app_name": page.get("app_name"), "app_package": page.get("app_package"),
        "activity": page.get("activity"),
    }.items() if v is not None}


def task_clauses(task):
    return [" ".join(re.sub(r"\b(the|a|an|app)\b", "", clause).split())
            for clause in re.split(r"\s+and\s+|[,;]", normalize(task)) if clause.strip()]


def bounded_task_prefix(task, action):
    requested = task_clauses(task)
    stored = task_clauses(action.get("source_task") or action.get("name", ""))
    if len(requested) != 1 or len(stored) < 2 or requested[0] != stored[0]:
        return 0
    verb = requested[0].split()[0]
    if verb not in {"start", "pause", "stop"}:
        return 0
    for index, step in enumerate(action.get("element_sequence", [])):
        label = content(step.get("target", {}))
        if label == verb and step.get("atomic_action") == "tap":
            return index + 1
        if step.get("atomic_action") == "text" or label in COMMIT or label in {"start", "stop", "pause"}:
            break
    return 0


def unsuitable_target(element, atomic="tap"):
    bounds = box(element)
    return (not bounds or (max(bounds) <= 1 and bounds[3] <= .06)
            or normalize(element.get("type")) == "empty"
            or content(element) == "empty space"
            or element.get("interactable") is False)


def repair_step_target(step):
    target = step.get("target", {})
    source = step.get("source", {})
    bounds = box(target)
    # Correct a stale label only from unique recorded geometry, never from task guesses.
    candidates = [e for e in elements(source) if bounds and box(e) == bounds
                  and not unsuitable_target(e) and content(e)]
    if len(candidates) == 1 and (not content(target) or content(target) != content(candidates[0])):
        step["target"] = copy.deepcopy(candidates[0])
        step["target_repair"] = "unique recorded source bbox"
    if unsuitable_target(step.get("target", {})) and step.get("atomic_action") in {"tap", "long_press", "text"}:
        step["target_metadata_error"] = "recorded target is missing, empty, or in status/debug strip; re-record this step"
    step["role"] = step_role(step)
    return step


def enrich_step(step, source, destination, element):
    result = copy.deepcopy(step)
    oi = decoded(element.get("other_info"), {})
    result.update(source=page_snapshot(source), destination=page_snapshot(destination),
                  target={"content": oi.get("content", element.get("content", "")),
                          "type": oi.get("type", element.get("type", "")),
                          "bbox": decoded(element.get("bounding_box", element.get("bbox")), [])})
    result["role"] = step_role(result)
    return repair_step_target(result)


def hydrate_actions(actions, metadata):
    """Read-only compatibility migration. Ambiguous graph edges aren't guessed."""
    result = copy.deepcopy(actions)
    for action in result:
        seq = decoded(action.get("element_sequence"), [])
        action["element_sequence"] = seq if isinstance(seq, list) else []
        for i, step in enumerate(action["element_sequence"]):
            if step.get("source") and step.get("destination") and step.get("target"):
                repair_step_target(step)
                continue
            records = metadata.get(step.get("element_id"), [])
            if len(records) == 1:
                rec = records[0]
                action["element_sequence"][i] = enrich_step(step, rec["source"], rec["destination"], rec["element"])
        seq = action["element_sequence"]
        if seq:
            action.setdefault("final_screen", seq[-1].get("destination", {}))
            action.setdefault("app_name", next((s.get("destination", {}).get("app_name") for s in seq if s.get("destination", {}).get("app_name")), ""))
            if not action.get("source_task"):
                records = metadata.get(seq[0].get("element_id"), [])
                if len(records) == 1:
                    info = decoded(records[0]["source"].get("other_info"), {})
                    action["source_task"] = info.get("task_info", {}).get("description", "")
    return result


def values_present(values, page):
    texts = [content(e) for e in elements(page)]
    for value in values:
        value = normalize(value)
        # Boundaries prevent 8 matching 18, and preserve AM vs PM.
        pattern = r"(?<!\w)" + re.escape(value) + r"(?!\w)"
        if not value or not any(re.search(pattern, text) for text in texts):
            return False
    return True


def valid_value(field, value):
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        return False
    if any(ord(c) < 32 and c not in "\n\t" for c in value):
        return False
    kind = normalize(field)
    if "email" in kind or "e-mail" in kind:
        return bool(re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value))
    if "phone" in kind or "mobile" in kind:
        return bool(re.fullmatch(r"\+?[0-9 ()-]{7,24}", value)) and 7 <= len(re.sub(r"\D", "", value)) <= 15
    return True


def form_fields(page, known_keys=()):
    fields = []
    for i, e in enumerate(elements(page)):
        label = content(e)
        editable = e.get("editable") or any(x in normalize(e.get("type", "")) for x in ("edittext", "input", "textfield"))
        # OmniParser often represents input placeholders as text.
        placeholder = label in {"name", "first name", "last name", "phone", "phone number", "mobile", "email", "e-mail", "title", "note", "write a note", "notes", "surname"}
        if box(e) and (editable or placeholder):
            fields.append({"key": label or f"field_{i}", "index": i, "label": label, "value": str(e.get("value", ""))})
    # Do not call a navigation tab called 'Notes' a form.
    labels = {content(e) for e in elements(page)}
    single_note = any(f["key"] in {"title", "note", "write a note"} for f in fields) and bool(labels & {"save", "done"})
    if len(fields) < 2 and not any(e.get("editable") for e in elements(page)) and not single_note and not any(f["key"] in known_keys for f in fields):
        return []
    return fields


PLAN_PROMPT = '''Plan a smartphone task using the supplied stored actions. Return JSON:
{"action_id": "id or null", "relation": "equivalent|related|none", "confidence": 0.0,
 "shared_prefix": 0, "reason": "why these steps serve the task unchanged"}.
Equivalent means identical intent AND parameter values, not merely the same app.
Related means same app with different goal or values. shared_prefix is the number
of consecutive opening steps unchanged for the requested task. Stop before any
parameter-dependent or unrelated operation. For alarm -> world clock reuse only
navigation into Clock. For a changed alarm time stop before choosing old values.
Never reuse a save/delete/create operation for a different task. No related action:
action_id null, relation none. Choose only supplied IDs. JSON only.'''
REACT_PROMPT = '''Operate the smartphone toward the task using one atomic action.
Return JSON {"action":"tap|text|swipe|long_press|back|done|need_vision",
"element_id":0,"input_str":"","field":"field label","direction":"up",
"reason":"brief evidence"}. element_id is the supplied zero-based INDEX.
Use done only with visible evidence of the whole goal, including saving forms.
Use need_vision when the parsed screen is insufficient. Do not repeat successful
creation/save actions. For text choose the actual input field and its field label;
use supplied form values exactly. Never invent new values if supplied. On a vision
request with no parsed target, return normalized x,y coordinates in [0,1].'''
FORM_PROMPT = '''Resolve all form values together. Return JSON {"values":{"field label":"value"}}.
Priority: explicit values in CURRENT TASK, then relevant STORED VALUES, then generate
random but plausible synthetic data for missing fields. Use printable ASCII, single-line values for the ADB input transport. Use valid email/phone formats.
Only return the requested field keys. Keep the same person's information consistent.
For a note provide a meaningful title and body. Never replace explicit user values.'''
JUDGE_PROMPT = '''Verify the WHOLE smartphone task from current evidence. Return JSON
{"complete":true,"confidence":0.9,"evidence":"visible proof","missing":""}.
If parsed evidence is insufficient, return need_vision:true with low confidence.
An open form, a successful ADB command, or a prior save attempt alone is not proof
of saving. Check requested parameter values exactly, including AM/PM.'''


class ReplayEngine:
    def __init__(self, capture, act, home, model, load_actions, *, max_steps=30, log=print):
        self.capture_fn, self.act_fn, self.home_fn = capture, act, home
        self.model_fn, self.load_fn = model, load_actions
        self.max_steps = max_steps
        self.started_at = time.monotonic()
        self.log = lambda message: log(f"[+{time.monotonic() - self.started_at:7.1f}s] {message}")
        self.metrics = Counter()
        self.page = None
        self.history = []
        self.form_values = {}
        self.form_resolved = set()
        self.filled = set()
        self.resume_index = None
        self.evidence = ""
        self.failure = ""
        self.actions = []
        self.remaining = max_steps
        self.model_backend_error = None
        self.react_final_screen = None
        self.prefix_goal_screen = None
        self.react_use_vision = False

    def ask(self, kind, system, payload, vision=False):
        if self.model_backend_error:
            raise RuntimeError(f"Model backend unavailable after {self.model_backend_error}; automatic repeated calls disabled")
        self.metrics["vision_calls" if vision else "text_calls"] += 1
        self.metrics[kind] += 1
        self.log(f"[MODEL] Starting {kind}: {'vision (fresh image attached)' if vision else 'text only (no image)'}. Waiting for response...")
        if kind in {"judge_text", "judge_vision"}:
            self.log(f"[JUDGE-SYSTEM-PROMPT] {kind}:\n{system}")
        started = time.monotonic()
        try:
            try:
                result = self.model_fn(kind, system, json.dumps(payload, ensure_ascii=False), self.page if vision else None)
            except ValueError as exc:
                raw = getattr(exc, "response_text", "")
                if not raw or kind.endswith("_format"):
                    raise
                self.log(f"[MODEL-FORMAT] {kind} supplied invalid JSON; one text-only formatting recovery, no image resend.")
                self.metrics["text_calls"] += 1
                self.metrics["format_recoveries"] += 1
                result = self.model_fn(kind + "_format", system,
                    json.dumps({"original_request": payload, "existing_response": raw,
                                "instruction": "Convert the existing response into the required JSON only. Preserve uncertainty; do not invent actions or completion evidence."}, ensure_ascii=False), None)
            if not isinstance(result, dict):
                raise ValueError(f"{kind}: expected a JSON object")
        except Exception as exc:
            if isinstance(exc, TimeoutError) or type(exc).__name__ in {"APITimeoutError", "APIConnectionError", "RateLimitError"}:
                self.model_backend_error = f"{type(exc).__name__}: {exc}"
            self.log(f"[MODEL] {kind} failed after {time.monotonic()-started:.1f}s: {type(exc).__name__}: {exc}")
            raise
        summary = {key: result[key] for key in ("action_id", "relation", "shared_prefix", "action", "element_id", "field", "complete", "confidence", "reason", "evidence", "missing") if key in result}
        if kind == "form":
            summary = {"resolved_fields": list(result.get("values", {})) if isinstance(result.get("values"), dict) else "invalid field mapping"}
        self.log(f"[MODEL] {kind} returned in {time.monotonic()-started:.1f}s: {json.dumps(summary, ensure_ascii=False)[:900]}")
        return result

    def observe(self):
        if self.page is None:
            self.metrics["captures"] += 1
            self.log(f"[SCREEN] Requesting fresh observation #{self.metrics['captures']} after {len(self.history)} action attempt(s).")
            self.page = self.capture_fn()
            if isinstance(self.page, dict):
                self.metrics["parser_calls"] += int(self.page.get("parser_attempted", True))
            if not isinstance(self.page, dict) or not self.page.get("screenshot"):
                self.page = None
                self.log("[ERROR] Fresh screenshot unavailable; stale screen data will not be used.")
                raise RuntimeError("Fresh screen capture failed")
            labels = [content(e) for e in meaningful(self.page)[:10]]
            self.log(f"[SCREEN] Observation ready: {len(elements(self.page))} elements; visible labels={labels}; screenshot={self.page.get('screenshot')}")
        return self.page

    def act(self, params, mode):
        if self.remaining <= 0:
            self.failure = "budget_exhausted"
            self.log("[STOP] Action budget exhausted.")
            return False
        self.remaining -= 1
        details = {key: value for key, value in params.items() if key != "input_str"}
        if "input_str" in params:
            details["input_length"] = len(params["input_str"])
        self.log(f"[ACTION] #{len(self.history)+1} mode={mode}: {details}; remaining budget={self.remaining}. Sending to ADB...")
        success = bool(self.act_fn(params))
        self.log(f"[ACTION] ADB {'succeeded' if success else 'FAILED'}; UI effect still needs screen verification.")
        self.history.append({"action": params.get("action"), "params": params, "status": "success" if success else "error"})
        self.page = None  # Even failed ADB calls may have partially changed the UI.
        self.react_use_vision = False
        self.metrics[f"{mode}_steps"] += int(success)
        if not success:
            self.failure = "action_error"
        return success

    def plan(self, task):
        self.log("[CATALOG] Loading stored tasks and replay metadata...")
        try:
            self.actions = hydrate_actions(self.load_fn(), {})
        except Exception as exc:
            self.log(f"Stored actions unavailable; using ReAct: {exc}")
            self.metrics["retrieval_errors"] += 1
            self.actions = []
        self.log(f"[CATALOG] Loaded {len(self.actions)} stored task(s). Checking exact task intent first.")
        exact = [a for a in self.actions if any(normalize(text) == normalize(task) for text in (a.get("source_task"), a.get("name")) if text) and a.get("element_sequence")]
        exact = [a for a in exact if len(task_clauses(a.get("source_task") or a.get("name", ""))) <= len(task_clauses(task))]
        for action in self.actions:
            for index, step in enumerate(action.get("element_sequence", [])):
                if step.get("target_repair"):
                    self.log(f"[METADATA] Stored step {index+1} target repaired from {step['target_repair']}: {content(step['target'])!r}.")
                if step.get("target_metadata_error"):
                    self.log(f"[METADATA] Stored step {index+1}: {step['target_metadata_error']}.")
        if exact:
            # Choose by catalog order; live alignment happens after task selection.
            self.log(f"[PLAN] Exact match: id={exact[0].get('action_id')}, task={exact[0].get('source_task')}, steps={len(exact[0]['element_sequence'])}. Planner call skipped.")
            return exact[0], "equivalent", len(exact[0]["element_sequence"])
        local_equivalent = [a for a in self.actions if equivalent_navigation(task, a)]
        if local_equivalent:
            chosen = local_equivalent[0]
            self.metrics["local_semantic_matches"] += 1
            self.log(f"[PLAN] Equivalent navigation intent: requested={task!r}, stored={chosen.get('source_task') or chosen.get('name')!r}; replay all {len(chosen['element_sequence'])} steps. Planner skipped; capture follows task matching.")
            return chosen, "equivalent", len(chosen["element_sequence"])
        partials = [(a, bounded_task_prefix(task, a)) for a in self.actions]
        partials = [(a, count) for a, count in partials if count]
        if partials:
            chosen, count = partials[0]
            self.prefix_goal_screen = chosen["element_sequence"][count-1].get("destination", {})
            self.log(f"[PLAN] Requested first subtask only: replay {count}/{len(chosen['element_sequence'])} steps; subsequent recorded operations excluded.")
            return chosen, "prefix", count
        prefixes = [(a, app_open_prefix(task, a)) for a in self.actions]
        prefixes = [(a, count) for a, count in prefixes if count]
        if prefixes:
            # Different recordings must agree on the requested app's destination.
            goal = prefixes[0][0]["element_sequence"][prefixes[0][1]-1]["destination"]
            if all(screen_score(goal, a["element_sequence"][count-1]["destination"]) >= .90 for a, count in prefixes):
                chosen, count = prefixes[0]
                self.prefix_goal_screen = chosen["element_sequence"][count-1]["destination"]
                self.metrics["deterministic_prefix_matches"] += 1
                self.log(f"[PLAN] Deterministic app-opening prefix: requested={task!r}, stored={chosen.get('source_task')!r}, replay {count}/{len(chosen['element_sequence'])} steps. Planner skipped; later recorded actions excluded.")
                return chosen, "prefix", count
            self.log("[PLAN] App-opening prefix candidates have conflicting destination evidence; semantic planner required.")
        if not self.actions:
            self.log("[FALLBACK] No stored tasks available. Skipping planner; use full ReAct.")
            return None, "none", 0
        tokens = set(re.findall(r"\w+", normalize(task)))
        # App-scoped retrieval when the task explicitly identifies a known app.
        app_names = {normalize(a.get("app_name", "")) for a in self.actions} - {""}
        named_apps = {name for name in app_names if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", normalize(task))}
        pool = [a for a in self.actions if not named_apps or normalize(a.get("app_name", "")) in named_apps]
        ranked = sorted(pool, key=lambda a: len(tokens & set(re.findall(r"\w+", normalize(str(a.get("source_task", "")) + " " + str(a.get("app_name", "")) + " " + str(a.get("name", "")))))), reverse=True)[:12]
        self.log(f"[PLAN] No exact match. Comparing {len(ranked)} candidate task(s), app filter={sorted(named_apps) or 'unspecified'}. Requesting one text planner call.")
        catalog = [{"action_id": a.get("action_id"), "task": a.get("source_task", a.get("name")), "app": a.get("app_name"),
                    "steps": [{"action": s.get("atomic_action"), "target": content(s.get("target", {})), "params": s.get("action_params", {})} for s in a.get("element_sequence", [])]} for a in ranked]
        try:
            p = self.ask("plan", PLAN_PROMPT, {"task": task, "actions": catalog})
            action = next((a for a in ranked if a.get("action_id") == p.get("action_id")), None)
            confidence = float(p.get("confidence", 0))
            if not action or not math.isfinite(confidence) or not .8 <= confidence <= 1:
                self.log(f"[FALLBACK] Planner selected no valid stored task or confidence {confidence} is outside [0.8, 1].")
                return None, "none", 0
            relation = p.get("relation")
            seq = action["element_sequence"]
            if relation == "equivalent" and len(task_clauses(action.get("source_task") or action.get("name", ""))) > len(task_clauses(task)):
                self.log("[PLAN] Equivalent replay rejected: recording contains additional task operations. Only shared navigation may replay.")
                relation = "related"
            if relation == "equivalent":
                # Numeric or quoted parameters must not silently change meaning.
                significant = lambda t: re.findall(r"\d+(?::\d+)?|\bam\b|\bpm\b|[\"'][^\"']+[\"']", normalize(t))
                recorded_values = [(decoded(st.get("action_params"), {}) or {}).get("text") or (decoded(st.get("action_params"), {}) or {}).get("input_str") for st in seq if st.get("atomic_action") == "text"]
                changed_text = any(normalize(value) in normalize(action.get("source_task", "")) and normalize(value) not in normalize(task) for value in recorded_values if value)
                if not changed_text and significant(task) == significant(action.get("source_task", "")):
                    self.log(f"[PLAN] Equivalent task: {action.get('action_id')}; replay all {len(seq)} steps.")
                    return action, relation, len(seq)
                self.log("[PLAN] Full replay rejected: requested parameters differ from the recording. Checking shared navigation only.")
                relation = "related"
            if relation == "related":
                requested = max(0, min(int(p.get("shared_prefix", 0)), len(seq)))
                shared = 0
                for step in seq[:requested]:
                    # Conservative code-enforced boundary: no form/time/save replay.
                    if step_role(step) != "navigation":
                        self.log(f"[PLAN] Shared prefix stops before step {shared+1}: target={content(step.get('target', {}))!r}, role={step_role(step)}.")
                        break
                    shared += 1
                self.log(f"[PLAN] Related task {action.get('action_id')}: replay {shared}/{len(seq)} shared steps, then ReAct for the remaining goal.")
                return action, relation, shared
        except Exception as exc:
            self.log(f"Planner unavailable or rejected: {exc}")
        return None, "none", 0

    def align(self, seq, limit):
        page = self.observe()
        self.log(f"[ALIGN] Comparing live parsed JSON against {limit} eligible stored screens; source threshold=0.75 with unique target checks; navigation can match by exact target; ambiguity margin=0.06. No model call.")
        candidates = []
        for i, step in enumerate(seq[:limit]):
            # Navigation can be skipped. A text step can be skipped only when
            # its exact value is already visible on the matched form screen.
            # A screenshot cannot establish that a prior commit happened.
            safe = True
            for prior in seq[:i]:
                role = step_role(prior)
                params = decoded(prior.get("action_params"), {}) or {}
                value = params.get("text") or params.get("input_str")
                if role != "navigation" and not (role == "input" and value and values_present([value], page)):
                    safe = False
                    break
            if not safe:
                self.log(f"[ALIGN] Step {i+1} excluded: earlier input/commit prerequisites are not proven.")
                continue
            score, accepted, basis = replay_source_match(step, page)
            self.log(f"[ALIGN] Stored step {i+1}: similarity={score:.3f}, target={content(step.get('target', {}))!r}, accepted={accepted}, basis={basis}.")
            if not accepted:
                self.log(f"[ALIGN] Step {i+1} differences: {screen_difference(step.get('source', {}), page)}")
            if accepted:
                candidates.append((max(score, REPLAY_SOURCE_THRESHOLD), i))
        # Already at the end of a pure navigation prefix: no need to go Home.
        if 0 < limit <= len(seq) and (limit < len(seq) or self.prefix_goal_screen is not None) and all(step_role(s) == "navigation" for s in seq[:limit]):
            score = screen_score(seq[limit-1].get("destination", {}), page)
            self.log(f"[ALIGN] Navigation destination: similarity={score:.3f}.")
            if score >= .90:
                candidates.append((score, limit))
        candidates.sort(reverse=True)
        if not candidates or (len(candidates) > 1 and candidates[0][0] - candidates[1][0] < .06):
            self.log(f"[ALIGN] No unambiguous match; qualifying candidates={candidates}.")
            return None
        self.log(f"[ALIGN] Accepted stored position {candidates[0][1]+1}, score={candidates[0][0]:.3f}.")
        return candidates[0][1]

    def replay(self, action, limit):
        seq = action["element_sequence"]
        index = self.align(seq, limit)
        if index is None:
            self.log("[RECOVERY] Live screen did not match safely. Sending ADB Home, then checking stored step 1.")
            self.metrics["home_attempts"] += 1
            self.page = None
            if not self.home_fn():
                self.failure = "home_failed"
                self.log("[RECOVERY] ADB Home failed; replay aborted, use ReAct.")
                return False
            # Recovery must restart at step 1, not silently jump after HOME.
            restart_score, accepted, basis = replay_source_match(seq[0], self.observe())
            self.log(f"[RECOVERY] After Home, stored step 1 similarity={restart_score:.3f}, accepted={accepted}, basis={basis}.")
            if not accepted:
                self.failure = "restart_unmatched"
                return False
            index = 0
        self.resume_index = index + 1
        self.log(f"Replay starts at stored step {index+1}")
        for number, step in enumerate(seq[index:limit], start=index+1):
            self.log(f"[REPLAY] Stored step {number}/{limit}: {step.get('atomic_action')} target={content(step.get('target', {}))!r}; remaining task uses stored instructions.")
            atomic = step.get("atomic_action")
            # Initial alignment chooses the start. During replay only resolve the
            # next actionable target; do not judge full screens between steps.
            page = self.observe() if atomic in ("tap", "long_press", "text") else self.page
            self.log("[REPLAY] Stored instruction execution; intermediate screen/final-screen comparison skipped.")
            params = decoded(step.get("action_params"), {}) or {}
            command = {"action": atomic}
            if atomic in ("tap", "long_press", "text"):
                idx = target_index(step.get("target", {}), page)
                if idx is None:
                    self.log("[REPLAY] Target missing or ambiguous in live elements; refusing to guess coordinates.")
                    self.failure = "ambiguous_target"
                    return False
                self.log(f"[REPLAY] Target matched to live element index={idx}, bbox={box(elements(page)[idx])}. No model call.")
                command["bbox"] = box(elements(page)[idx])
            if atomic == "text":
                text = params.get("text") or params.get("input_str")
                if not valid_value(content(step.get("target", {})), text):
                    self.failure = "missing_form_value"
                    return False
                command.update(input_str=text, replace=True)
            elif atomic == "long_press":
                command["duration"] = params.get("duration", 1000)
            elif atomic and atomic.startswith("swipe"):
                command.update(params)
                if params.get("start") and params.get("end"):
                    command["action"] = "swipe_precise"
            elif atomic not in ("tap", "back"):
                self.failure = "unsupported_action"
                return False
            if atomic == "text" and values_present([command["input_str"]], page):
                self.log("[FORM] Recorded value already visible; skipping duplicate text entry.")
                self.metrics["already_filled"] += 1
                continue
            if not self.act(command, "replayed"):
                return False
            self.history[-1]["role"] = step_role(step)
            # Completion is checked once after the full stored sequence. The
            # next target-based action will request a fresh observation as needed.
        self.log(f"[REPLAY] Finished stored step range {index+1}..{limit}; no intermediate completion judges.")
        return True

    def resolve_form(self, task, action):
        # Merely viewing a contact/note must not trigger unsolicited form writes.
        if not re.search(r"\b(create|add|write|edit|update|fill|enter|compose|make|save|set)\b", normalize(task)):
            return
        fields = form_fields(self.observe())
        unresolved = [f for f in fields if f["key"] not in self.form_resolved]
        if not unresolved:
            return
        self.log(f"[FORM] Found unresolved field(s): {[f['key'] for f in unresolved]}. Resolving current input, then stored defaults, then generated data.")
        stored = {}
        for step in (action or {}).get("element_sequence", []):
            if step.get("atomic_action") == "text":
                params = decoded(step.get("action_params"), {}) or {}
                stored[content(step.get("target", {}))] = params.get("text") or params.get("input_str", "")
        # Resolve explicit label:value strings without inference.
        missing = []
        for f in unresolved:
            match = re.search(r"\b" + re.escape(f["label"]) + r"\s*[:=]\s*(?:\"([^\"]+)\"|([^,;\n]+))", task, re.I)
            value = (match.group(1) or match.group(2)).strip() if match else None
            if value and valid_value(f["key"], value):
                self.form_values[f["key"]] = value
                self.log(f"[FORM] {f['key']!r}: using explicit task input ({len(value)} characters).")
            else:
                missing.append(f)
        # An unchanged recorded task uses recorded values without generation.
        if normalize(task) == normalize((action or {}).get("source_task")):
            still_missing = []
            for f in missing:
                value = stored.get(f["key"])
                if valid_value(f["key"], value):
                    self.form_values[f["key"]] = value
                    self.log(f"[FORM] {f['key']!r}: reusing valid stored input; no generation call.")
                else:
                    still_missing.append(f)
            missing = still_missing
        if missing:
            self.log(f"[FORM] Batch-resolving {len(missing)} remaining field(s) in one text call.")
            result = self.ask("form", FORM_PROMPT, {"task": task, "fields": [f["key"] for f in missing], "stored_values": stored, "already_resolved": self.form_values})
            values = result.get("values", {})
            if not isinstance(values, dict):
                raise ValueError("Invalid form response")
            for f in missing:
                value = values.get(f["key"])
                if not valid_value(f["key"], value):
                    raise ValueError(f"Invalid generated value for {f['key']}")
                self.form_values[f["key"]] = value
        self.form_resolved.update(f["key"] for f in unresolved)

    def autofill_one(self):
        """Fill a resolved field deterministically; refresh before the next field."""
        page = self.observe()
        for field in form_fields(page, self.form_values):
            key = field["key"]
            value = self.form_values.get(key)
            if not value or (key, value) in self.filled:
                continue
            if normalize(field["value"]) == normalize(value):
                self.filled.add((key, value))
                continue
            command = {"action": "text", "bbox": box(elements(page)[field["index"]]),
                       "input_str": value, "replace": True}
            self.log(f"[FORM] Filling {key!r} ({len(value)} characters), focus+replace enabled. No per-field model call.")
            if not self.act(command, "react"):
                return "error"
            if not values_present([value], self.observe()):
                self.failure = "input_not_verified"
                return "error"
            self.filled.add((key, value))
            self.log(f"[FORM] {key!r}: entered value verified in fresh screen JSON.")
            return "filled"
        return "none"

    def verify(self, task, prefer_vision=False):
        self.log("[VERIFY] Checking whole-task completion; save attempts alone are not proof.")
        # Completion does not need tap coordinates, parser IDs, or verbose metadata.
        compact = lambda page: [{k: e[k] for k in ("content", "text", "type", "selected", "value", "editable") if k in e} for e in meaningful(page)]
        history = [{"action": h.get("action"), "status": h.get("status"),
                    "input_str": h.get("params", {}).get("input_str")} for h in self.history[-8:]]
        payload = {"task": task, "elements": compact(self.observe()), "history": history, "form_values": self.form_values}
        if self.react_final_screen:
            payload["expected_final_elements"] = compact(self.react_final_screen)
        if prefer_vision:
            self.log("[VERIFY] Stored JSON did not confirm completion; inspect the fresh screenshot before any further action. Skipping an extra text-only judge.")
            self.react_use_vision = True
        for vision in ((True,) if prefer_vision or not elements(self.observe()) else (False, True)):
            try:
                result = self.ask("judge_vision" if vision else "judge_text", JUDGE_PROMPT, payload, vision)
                confidence = float(result.get("confidence", 0))
            except Exception as exc:
                self.log(f"Completion evidence unavailable: {exc}")
                continue
            if not vision and (result.get("need_vision") is True or result.get("action") == "need_vision"):
                self.log("[VERIFY] Text judge reports insufficient parsed evidence; escalating to the fresh screenshot.")
                self.react_use_vision = True
                continue
            if math.isfinite(confidence) and .85 <= confidence <= 1:
                if result.get("complete") is True:
                    self.evidence = str(result.get("evidence", ""))
                    self.log(f"[VERIFY] Complete={bool(self.evidence)}, confidence={confidence:.3f}, evidence={self.evidence}.")
                    return bool(self.evidence)
                self.log(f"[VERIFY] Incomplete: {result.get('missing') or result.get('evidence', 'no proof')}; confidence={confidence:.3f}.")
                return False
            self.log(f"[VERIFY] Confidence {confidence} is insufficient; escalate if another evidence source is available.")
        return False

    def check_react_progress(self, task):
        self.metrics["react_progress_checks"] += 1
        page = self.observe()
        self.log(f"[REACT-PROGRESS] After action #{len(self.history)}: checking the requested final goal.")
        if self.react_final_screen:
            score = screen_score(self.react_final_screen, page)
            self.log(f"[REACT-PROGRESS] Relevant stored final screen similarity={score:.3f}; required=0.90. No model call.")
            if score >= .90:
                self.evidence = "ReAct reached the requested task's stored final screen"
                return True
            return self.verify(task, prefer_vision=True)
        self.log("[REACT-PROGRESS] No relevant stored final screen for this goal; use text-first completion judgement.")
        return self.verify(task)

    def react(self, task, action):
        self.log(f"[REACT] Starting adaptive execution; remaining action budget={self.remaining}, replay handoff reason={self.failure or 'remaining goal not covered by stored steps'}.")
        repeats = Counter()
        while self.remaining > 0:
            page = self.observe()
            self.resolve_form(task, action)
            filled = self.autofill_one()
            if filled == "error":
                return False
            if filled == "filled":
                if self.check_react_progress(task):
                    return True
                continue
            payload = {"task": task, "elements": [{**e, "index": i} for i, e in enumerate(elements(page))], "history": self.history[-8:], "form_values": self.form_values, "replay_failure": self.failure}
            vision = self.react_use_vision or not elements(page)
            try:
                result = self.ask("react_vision" if vision else "react", REACT_PROMPT, payload, vision)
            except Exception:
                if vision:
                    raise
                vision = True
                result = self.ask("react_vision", REACT_PROMPT, payload, True)
            if result.get("action") == "need_vision" and not vision:
                vision = True
                result = self.ask("react_vision", REACT_PROMPT, payload, True)
            idx = result.get("element_id")
            if result.get("action") in {"tap", "text", "long_press"} and isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < len(elements(page)) and unsuitable_target(elements(page)[idx], result["action"]):
                self.log(f"[REACT] Rejecting unsuitable target index={idx}; no ADB action. Requesting one vision correction.")
                payload["rejected_target"] = {"index": idx, "reason": "status/debug strip, empty, or explicitly non-interactable element"}
                result = self.ask("react_vision", REACT_PROMPT, payload, True)
                vision = True
            atomic = result.get("action")
            self.log(f"[REACT] Proposed action={atomic}, element index={result.get('element_id')}, reason={result.get('reason', 'not provided')}.")
            if atomic == "done":
                if self.verify(task):
                    return True
                self.remaining -= 1  # Repeated false 'done' cannot loop forever.
                continue
            command = {"action": atomic}
            if atomic in ("tap", "text", "long_press"):
                idx = result.get("element_id")
                if isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < len(elements(page)) and box(elements(page)[idx]) and not unsuitable_target(elements(page)[idx], atomic):
                    command["bbox"] = box(elements(page)[idx])
                elif vision and atomic != "text" and not isinstance(idx, int) and all(isinstance(result.get(k), (float, int)) and 0 <= result[k] <= 1 for k in ("x", "y")) and result["y"] > .06:
                    command.update(x_relative=result["x"], y_relative=result["y"])
                else:
                    self.failure = "invalid_react_target"
                    return False
            if atomic == "text":
                field = normalize(result.get("field", ""))
                value = self.form_values.get(field, result.get("input_str"))
                if not valid_value(field, value):
                    self.failure = "invalid_input"
                    return False
                if (field, value) in self.filled and values_present([value], page):
                    self.failure = "duplicate_input"
                    return False
                command.update(input_str=value, replace=True)
            elif atomic == "long_press":
                command["duration"] = min(5000, max(100, int(result.get("duration", 1000))))
            elif atomic == "swipe":
                if result.get("direction") not in ("up", "down", "left", "right"):
                    self.failure = "invalid_direction"
                    return False
                command["direction"] = result["direction"]
            elif atomic not in ("tap", "back"):
                self.failure = "invalid_react_action"
                return False
            fingerprint = (json.dumps([content(e) for e in meaningful(page)]), json.dumps(command, sort_keys=True))
            repeats[fingerprint] += 1
            if repeats[fingerprint] >= 3:
                self.failure = "stuck"
                self.log("[STOP] Same action proposed on the same parsed screen three times; stopping as stuck.")
                return False
            if not self.act(command, "react"):
                return False
            if atomic == "text":
                if not values_present([command["input_str"]], self.observe()):
                    self.failure = "input_not_verified"
                    return False
                self.filled.add((field, command["input_str"]))
            if self.check_react_progress(task):
                return True
        # Each action has already been checked; do not repeat a final judge.
        self.failure = "budget_exhausted"
        return False

    def run(self, task, force_fallback=False):
        action, relation, limit = None, "none", 0
        complete = False
        self.log(f"[START] Task={task!r}; max actions={self.max_steps}; force ReAct={force_fallback}.")
        try:
            if not force_fallback:
                action, relation, limit = self.plan(task)
            else:
                self.log("[PLAN] Forced fallback selected; stored-task matching skipped.")
            if relation == "prefix":
                self.react_final_screen = self.prefix_goal_screen
            if relation == "equivalent" and action:
                self.react_final_screen = action.get("final_screen") or action["element_sequence"][-1].get("destination", {})
            if limit:
                replay_ok = self.replay(action, limit)
                if not replay_ok:
                    self.log(f"[FALLBACK] Replay stopped: {self.failure}; preserve successful history for ReAct.")
                if replay_ok and (relation == "prefix" or (relation == "equivalent" and limit == len(action["element_sequence"]))):
                    final = self.prefix_goal_screen if relation == "prefix" else (action.get("final_screen") or action["element_sequence"][-1].get("destination", {}))
                    # Verify only the requested goal boundary after replay.
                    final_step = action["element_sequence"][limit-1]
                    distinguishable = screen_score(final_step.get("source", {}), final) < .90
                    self.metrics["replay_final_checks"] += 1
                    final_score = screen_score(final, self.observe())
                    self.log(f"[VERIFY] Last stored step finished. ONE final-screen comparison: score={final_score:.3f}, required=0.90.")
                    complete = distinguishable and final_score >= .90
                    if complete:
                        self.evidence = "Stored instructions executed and final screen verified once after the last step"
                        self.log("[VERIFY] Stored final postcondition confirmed using parsed JSON; model judge skipped.")
                    else:
                        self.failure = "stored_final_not_confirmed"
                        self.log("[VERIFY] Stored JSON is inconclusive; judge actual completion before handing off to ReAct.")
                        complete = self.verify(task, prefer_vision=True)
            if not complete and self.history and self.history[-1].get("role") == "commit":
                # A save may have succeeded despite a changed result screen. Verify
                # before allowing ReAct to create a duplicate item.
                complete = self.verify(task)
            if not complete:
                if self.model_backend_error:
                    raise RuntimeError(f"Model request failed: {self.model_backend_error}; ReAct needs the same backend, so no duplicate request is dispatched")
                complete = self.react(task, action)
        except Exception as exc:
            self.failure = f"error: {exc}"
            self.log(self.failure)
        self.log(f"[FINISH] status={'completed' if complete else self.failure or 'failed'}, replayed={self.metrics['replayed_steps']}, react={self.metrics['react_steps']}, text calls={self.metrics['text_calls']}, vision calls={self.metrics['vision_calls']}, parser calls={self.metrics['parser_calls']}; evidence={self.evidence or 'none'}.")
        return {"status": "completed" if complete else (self.failure or "failed"), "completed": complete,
                "message": self.evidence if complete else self.failure, "completion_evidence": self.evidence,
                "steps_completed": self.metrics["replayed_steps"] + self.metrics["react_steps"],
                "replayed_steps": self.metrics["replayed_steps"], "react_steps": self.metrics["react_steps"],
                "resumed_step": self.resume_index, "metrics": dict(self.metrics),
                "llm_calls": {k: v for k, v in self.metrics.items() if k in {"plan", "form", "react", "react_vision", "judge_text", "judge_vision"}},
                "plan": {"action_id": (action or {}).get("action_id"), "relation": relation, "replay_prefix": limit},
                "history": self.history, "close_actions": []}
