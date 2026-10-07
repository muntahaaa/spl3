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


class ParameterResolutionError(RuntimeError):
    """A stored plan was matched, but one of its input values is unresolved."""


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


def page_layout_score(expected, live):
    """Final-page identity from structure and stable controls, not their values."""
    if not compatible_apps(expected, live):
        return 0.0
    tabs = {"alarm", "world clock", "stopwatch", "timer", "pictures", "albums"}
    selected = lambda page: {content(e) for e in meaningful(page) if e.get("selected") and content(e) in tabs}
    a_selected, b_selected = selected(expected), selected(live)
    if a_selected and b_selected and a_selected.isdisjoint(b_selected):
        return 0.0
    def structural(page):
        result = copy.deepcopy(page)
        entries = []
        for e in meaningful(page):
            if e.get("type") == "empty" or content(e) == "empty space":
                continue
            item = copy.deepcopy(e)
            label = re.sub(r"\d+(?:[.:]\d+)*", "value", content(e))
            label = re.sub(r"\b(?:am|pm)\b", "period", label)
            item["content"] = label
            item["type"] = "input" if e.get("editable") or e.get("type") in {"input","edittext"} else "control"
            item["selected"] = bool(e.get("selected"))
            entries.append(item)
        result["elements"] = entries
        result.pop("elements_data",None)
        return result
    # Stable anchors and relative positions identify the page. Numeric tokens
    # become structural placeholders, preserving clock/readout locations.
    return screen_score(structural(expected),structural(live))


def screen_difference(expected, live):
    expected_texts = Counter(content(e) for e in meaningful(expected))
    live_texts = Counter(content(e) for e in meaningful(live))
    return {"missing": list((expected_texts-live_texts).elements())[:8],
            "unexpected": list((live_texts-expected_texts).elements())[:8]}


def expanded_label_match(wanted, candidate):
    """Whole-phrase containment for expanded result labels, never an input or command."""
    if candidate.get("editable") or candidate.get("type") in {"input", "edittext"}:
        return False
    if candidate.get("enabled") is False or (candidate.get("package") and candidate.get("clickable") is False):
        return False
    if len(wanted) < 3 or not any(c.isalpha() for c in wanted):
        return False
    if wanted in NAVIGATION | COMMIT | {"start", "stop", "pause", "reset", "resume", "add", "search"}:
        return False
    return bool(re.search(r"(?<!\w)" + re.escape(wanted) + r"(?!\w)", content(candidate)))


def control_role(target):
    """Use a stored role or known descriptive intent without rewriting its label."""
    role = normalize(target.get("role", ""))
    label = content(target).rstrip(" .")
    words = set(re.findall(r"[a-z]+", label))
    add_or_create = bool(words & {"add", "adding", "create", "creating", "new"})
    generic_created_object = bool(words & {
        "item", "alarm", "contact", "note", "event", "task", "folder", "album", "city"
    })
    resource_id = normalize(target.get("resource_id") or target.get("resource-id") or "")
    if (role in {"add", "create", "add/create"}
            or (add_or_create and generic_created_object)
            or bool(re.search(r"(?:^|[_:/-])(?:add|create|new)(?:$|[_:/-])", resource_id))):
        return "add"
    return None


def role_candidate_valid(role, candidate):
    if role != "add" or candidate.get("enabled") is False or unsuitable_target(candidate):
        return False
    if candidate.get("editable") or candidate.get("type") in {"input", "edittext"}:
        return False
    # Hierarchy provides authoritative clickability; parser-only observations do not.
    if candidate.get("clickable") is False:
        return False
    label = content(candidate)
    if label in {"add", "create", "new", "+"}:
        return True
    return bool(re.fullmatch(r"(?:add|create|new)\s+(?:a\s+)?(?:city|alarm|contact|note|item|event|task|folder|album|timer)",label))


def target_index(target, live):
    """Stable identity first, then exact text, roles and finally fuzzy text."""
    role = control_role(target)
    scores = []
    expanded = []
    for i, e in enumerate(elements(live)):
        if target.get("query_dependent") and (e.get("editable") or e.get("type") in {"input","edittext"} or e.get("enabled") is False or (e.get("package") and e.get("clickable") is False)):
            continue
        if not box(e) or not content(target) or not content(e):
            continue
        target_id = target.get("resource_id") or target.get("resource-id")
        live_id = e.get("resource_id") or e.get("resource-id")
        if target_id and live_id and target_id == live_id:
            scores.append((2.0, i))
            continue
        # Hierarchy and image-parser type names differ; label identity is primary.
        if target.get("type") == "input" and e.get("type") not in {"input", "edittext"}:
            continue
        # Exact stored evidence outranks semantic aliases. Parser descriptions
        # are valid exact labels even when they are more verbose than the UI.
        if content(target) == content(e):
            scores.append((1.8,i))
            continue
        if role and not role_candidate_valid(role,e):
            continue
        if role:
            scores.append((1.0,i))
            continue
        target_label = {"search function.": "search", "search function": "search"}.get(content(target), content(target))
        ratio = difflib.SequenceMatcher(None, target_label, content(e)).ratio()
        if ratio < .93:
            if target.get("type") not in {"input", "edittext"} and expanded_label_match(target_label,e):
                expanded.append((.90,i))
            continue
        distance = 1.0
        a, b = box(target), box(e)
        if a and b and (max(a) <= 1) == (max(b) <= 1):
            distance = math.hypot((a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2)
        scores.append((ratio + .05 * max(0, 1-distance*10), i))
    # Accessibility trees may expose both a clickable tab and its child label.
    # Collapse only nodes proven to share the same clickable ancestor.
    using_expanded = not scores
    if using_expanded:
        scores = expanded
    grouped = {}
    for score, i in scores:
        e = elements(live)[i]
        key = (e.get("package"), e["control_id"]) if e.get("control_id") is not None else ("index", i)
        if key not in grouped or score > grouped[key][0]:
            grouped[key] = (score, i)
    scores = sorted(grouped.values(), reverse=True)
    if using_expanded and len(scores) > 1:
        return None
    if not scores or (len(scores) > 1 and scores[0][0] - scores[1][0] < .025):
        return None
    return scores[0][1]


NAVIGATION = {"clock", "gallery", "contacts", "notes", "pictures", "picture", "albums", "album", "world clock", "timer", "alarm", "alarms", "settings", "apps", "home", "back"}
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
    if action == "tap" and (label in NAVIGATION or any(n in label for n in NAVIGATION)):
        return "navigation"
    source_app = step.get("source", {}).get("app_name")
    dest_app = step.get("destination", {}).get("app_name")
    if dest_app and (not source_app or source_app in ("home", "launcher", "android") or source_app != dest_app):
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
            or element.get("interactable") is False or element.get("enabled") is False)


def recovered_target_valid(target, candidate, atomic):
    if unsuitable_target(candidate,atomic):
        return False
    if atomic == "text":
        return candidate.get("type") in {"input","edittext"} or candidate.get("editable") is True
    if candidate.get("package") and candidate.get("clickable") is False:
        return False
    wanted,actual=content(target),content(candidate)
    if wanted == actual:
        return True
    role = control_role(target)
    if role:
        return role_candidate_valid(role,candidate)
    aliases={"search function.":"search","search function":"search",
             "adding a new item or creating something new.":"add", "kity/country/region":"search", "city/country/region":"search"}
    wanted=aliases.get(wanted,wanted)
    if wanted == "add":
        return actual in {"add","add city","+","new","add alarm"}
    if wanted == "search":
        return "search" in actual or candidate.get("type") == "input"
    if wanted == "stopwatch" and actual == "elapsed time":
        return True
    return wanted == actual or expanded_label_match(wanted,candidate) or difflib.SequenceMatcher(None,wanted,actual).ratio() >= .85


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


def input_value_present(value, page, target=None, index=None):
    """Only the edited control can prove entry; a matching list row cannot."""
    live = elements(page)
    candidates = [e for e in live if e.get("editable") or e.get("type") in {"input","edittext"}]
    if target:
        resource = target.get("resource_id") or target.get("resource-id")
        matched = [e for e in candidates if resource and resource == (e.get("resource_id") or e.get("resource-id"))]
        if matched:
            candidates = matched
        elif len(candidates) > 1:
            bounds = box(target)
            candidates = [e for e in candidates if bounds and box(e) == bounds]
    if not candidates and not any(e.get("editable") or e.get("type") in {"input","edittext"} for e in live):
        # Legacy image-parser records may omit input roles. Require field geometry.
        bounds = box(target or {})
        candidates = [e for e in live if bounds and box(e) == bounds]
    if index is not None and 0 <= index < len(live) and live[index] in candidates:
        candidates = [live[index]]
    return any(normalize(e.get("value",e.get("text",e.get("content","")))) == normalize(value) for e in candidates)


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
Return a JSON object with action (tap, text, swipe, long_press, back, done or need_vision),
element_id (supplied zero-based INDEX), and reason (concrete task-specific visible evidence).
For text include input_str and field; for swipe include direction. Do not return placeholder reasons.
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
JUDGE_PROMPT = '''Verify the WHOLE smartphone task from current evidence. Return a JSON object with complete (boolean), confidence (number from 0 to 1),
evidence (concrete visible proof) and missing (unfinished requirements). Assess the evidence; do not copy a template.
Provide concrete visible task-specific evidence. Never copy example values or use generic visible proof.
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
        self.matched_plan = None
        self.exact_task_match = False
        self.exact_refresh_steps = set()
        self.assistance_attempted_steps = set()
        self.input_extraction_cache = {}
        self.input_extraction_calls = 0
        self.workflow_parameter_cache = {}

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
            if getattr(self,"selected_case_id",None):
                raise
            self.log(f"Stored actions unavailable; using ReAct: {exc}")
            self.metrics["retrieval_errors"] += 1
            self.actions = []
        self.log(f"[CATALOG] Loaded {len(self.actions)} stored task(s). Checking exact task intent first.")
        for stored_action in self.actions:
            if normalize(stored_action.get("source_task")) in {"unknown task","unknown","n/a"}:
                stored_action["source_task"] = stored_action.get("name") or ""
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
            self.exact_task_match = True
            self.matched_plan = (exact[0], "equivalent", len(exact[0]["element_sequence"]))
            self.log(f"[PLAN] Exact match: id={exact[0].get('action_id')}, task={task!r}, steps={len(exact[0]['element_sequence'])}. Planner call skipped.")
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
            # Preserve retrieval independently from later parameter binding. If
            # binding needs assistance, the selected recording must not vanish
            # into the planner's ordinary "no match" fallback.
            self.matched_plan = (action, relation, len(seq))
            if relation == "equivalent" and len(task_clauses(action.get("source_task") or action.get("name", ""))) > len(task_clauses(task)):
                self.log("[PLAN] Equivalent replay rejected: recording contains additional task operations. Only shared navigation may replay.")
                relation = "related"
            if relation == "equivalent":
                # Numeric or quoted parameters must not silently change meaning.
                significant = lambda t: re.findall(r"\d+(?::\d+)?|\bam\b|\bpm\b|[\"'][^\"']+[\"']", normalize(t))
                adapted = self.bind_replay_text(action,task)
                if adapted != action:
                    action = adapted
                    seq = action["element_sequence"]
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
        except ParameterResolutionError:
            raise
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
                if role == "commit":
                    safe = False
                    break
                if role == "input" and not (value and input_value_present(value,page,prior.get("target",{}))):
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
            if score > .70 and not self.pending_operations() and not self.navigation_goal_contradicted(getattr(self,"current_task","")):
                candidates.append((score, limit))
        candidates.sort(reverse=True)
        if not candidates or (len(candidates) > 1 and candidates[0][0] - candidates[1][0] < .06):
            self.log(f"[ALIGN] No unambiguous match; qualifying candidates={candidates}.")
            return None
        self.log(f"[ALIGN] Accepted stored position {candidates[0][1]+1}, score={candidates[0][0]:.3f}.")
        return candidates[0][1]

    def explicit_text_value(self, field, task=None):
        cache_key = (normalize(field),getattr(self,"user_guidance",""),task or getattr(self,"current_task",""))
        if cache_key in self.input_extraction_cache:
            return self.input_extraction_cache[cache_key]
        for instruction in (getattr(self,"user_guidance",""), task or getattr(self,"current_task","")):
            if not instruction:
                continue
            labeled = re.search(re.escape(field) + r"\s*[:=]\s*(?:[\"]([^\"]+)[\"]|([^,;\n]+))",instruction,re.I) if field else None
            if labeled:
                return (labeled.group(1) or labeled.group(2)).strip()
            if any(word in normalize(field) for word in ("search","city","country","region","kity")):
                match = re.search(r"\b(?:search\s+(?:and|then)\s+add|search(?:\s+for)?|write|type|enter)\s+(?:[\"]([^\"]+)[\"]|(.+?))(?=\s+(?:and|then)\b|[,;\n]|$)",instruction,re.I)
                if match:
                    return (match.group(1) or match.group(2)).strip().strip(" .")
        return None

    def resolve_input_parameter(self, field, task, stored_value=None, stored_task=""):
        """Fast parsing, then bounded text extraction grounded in user instructions."""
        value = self.explicit_text_value(field,task)
        guidance = getattr(self,"user_guidance","")
        key = (normalize(field),guidance,task)
        if key in self.input_extraction_cache:
            return value
        suspicious = bool(value and re.match(r"^(?:and|then|please|search|type|enter|add)\b",normalize(value)))
        instructions = guidance or task
        input_requested = bool(re.search(r"\b(search|find|look for|type|enter|write|named|called|query)\b",normalize(instructions)))
        changed_request = bool(guidance) or normalize(task) != normalize(stored_task)
        if not suspicious and (value or not input_requested or not changed_request):
            return value
        if self.input_extraction_calls >= 3:
            self.log("[INPUT-EXTRACT] Model extraction budget exhausted; no repeated request.")
            self.input_extraction_cache[key] = None
            return None
        self.input_extraction_calls += 1
        self.log(f"[INPUT-EXTRACT] Deterministic value missing or suspicious for {field!r}; one text-only extraction, cached for this instruction/field.")
        prompt = (
            "Extract the exact user-provided input value for the requested field in ANY application. "
            "Return JSON with found:boolean, value:string or null, confidence:number, source_quote:string. "
            "User guidance overrides task text when it explicitly supplies this field's value. "
            "Separate command verbs and navigation clauses from the value; preserve multiword names, punctuation, "
            "numbers and the actual text of notes/messages. Do not invent values or copy stored defaults. "
            "If no value is supplied or the field is ambiguous, return found:false. "
            "source_quote must be an exact excerpt of task or guidance containing the extracted value."
        )
        extracted = None
        try:
            result = self.ask("extract_input",prompt,{"task":task,"user_guidance":guidance,"field":field,
                "stored_value":stored_value,"deterministic_candidate":value})
            candidate = result.get("value")
            quote = result.get("source_quote")
            confidence = float(result.get("confidence",0))
            sources = (task,guidance)
            if (result.get("found") is True and isinstance(candidate,str) and candidate.strip()
                and isinstance(quote,str) and quote and any(quote in source for source in sources)
                and candidate in quote and math.isfinite(confidence) and .85 <= confidence <= 1
                and valid_value(field,candidate)):
                extracted = candidate.strip()
                self.log(f"[INPUT-EXTRACT] Accepted instruction-grounded value for {field!r}; cached, no image needed.")
            else:
                self.log("[INPUT-EXTRACT] No trustworthy explicit value; rejected invented/ambiguous extraction.")
        except Exception as exc:
            self.log(f"[INPUT-EXTRACT] Extraction unavailable: {type(exc).__name__}: {exc}; no automatic retry.")
        self.input_extraction_cache[key] = extracted
        return extracted

    def extract_workflow_parameters(self, action, task):
        """Resolve all recorded input fields together, independent of task verbs."""
        guidance = getattr(self,"user_guidance","")
        fields = []
        for index, step in enumerate(action.get("element_sequence",[])):
            if step.get("atomic_action") != "text":
                continue
            params = decoded(step.get("action_params"),{}) or {}
            fields.append({"id":str(index),"field":content(step.get("target",{})),
                           "description":step.get("description",""),
                           "stored_value":params.get("text") or params.get("input_str")})
        if not fields:
            return {}
        key = (action.get("action_id"),task,guidance,tuple((f["id"],f["field"]) for f in fields))
        if key in self.workflow_parameter_cache:
            cached = self.workflow_parameter_cache[key]
            if "error" in cached:
                raise RuntimeError("Input parameter unresolved: " + cached["error"])
            return cached
        mapping = {}
        unresolved = []
        changed = bool(guidance) or normalize(task) != normalize(action.get("source_task") or action.get("name"))
        for field in fields:
            value = self.explicit_text_value(field["field"],task)
            if value and valid_value(field["field"],value) and not re.match(r"^(and|then|please|search|type|enter|add)\b",normalize(value)):
                mapping[field["id"]] = value
            elif changed:
                unresolved.append(field)
            else:
                mapping[field["id"]] = None
        if unresolved:
            self.log(f"[INPUT-EXTRACT] Resolve {len(unresolved)} stored input field(s) in one text-only request; no verb whitelist.")
            prompt = (
                "Resolve user-provided parameters for ALL supplied stored input steps, for any application. "
                "Return JSON {parameters:[{id:string,status:explicit|absent|ambiguous,value:string or null,"
                "source_quote:string,confidence:number}]}. Return exactly one entry per supplied id. "
                "Use field labels, descriptions and stored values to identify the field's role. "
                "User guidance takes priority. Separate command/navigation words from actual input; preserve "
                "multiword names and note/message contents. Explicit means a supplied replacement; absent means "
                "no replacement for that field and its stored default may be retained; ambiguous means clarification "
                "is needed. Do not guess or generate data. For explicit values source_quote must be an exact "
                "excerpt of task or guidance containing the exact value."
            )
            payload = {"task":task,"user_guidance":guidance,
                "stored_task":action.get("source_task") or action.get("name"),"fields":unresolved}
            result = self.ask("extract_parameters",prompt,payload)
            entries = result.get("parameters",[])
            expected_ids = {field["id"] for field in unresolved}
            returned_ids = {entry.get("id") for entry in entries if isinstance(entry,dict)} if isinstance(entries,list) else set()
            if not isinstance(entries,list) or not expected_ids.issubset(returned_ids):
                self.log("[INPUT-EXTRACT] Structured response omitted required field(s); one bounded text-only recovery.")
                recovery_prompt = (
                    "Repair and complete the parameter extraction. Return only a JSON object with a parameters array. "
                    "Include exactly one entry for every supplied field id. For each entry use status explicit, absent, "
                    "or ambiguous; value or null; confidence; and source_quote. Extract values from task/user_guidance, "
                    "not from stored_value. A phrase such as 'add Sri Lanka' supplies the value 'Sri Lanka' when the "
                    "stored input field is a city/country search field. Do not omit an id even when ambiguous."
                )
                result = self.ask("extract_parameters_recovery",recovery_prompt,
                    {**payload,"invalid_response":result})
                entries = result.get("parameters",[])
            for field in unresolved:
                matches = [e for e in entries if isinstance(e,dict) and e.get("id") == field["id"]] if isinstance(entries,list) else []
                entry = matches[0] if len(matches) == 1 else {}
                status = entry.get("status")
                value,quote = entry.get("value"),entry.get("source_quote")
                try:
                    confidence = float(entry.get("confidence",0))
                except (ValueError,TypeError):
                    confidence = 0
                trustworthy = math.isfinite(confidence) and .85 <= confidence <= 1
                if status == "explicit" and trustworthy and isinstance(value,str) and isinstance(quote,str) and quote and any(quote in src for src in (task,guidance)) and value in quote and valid_value(field["field"],value):
                    mapping[field["id"]] = value.strip()
                elif status == "absent" and trustworthy:
                    mapping[field["id"]] = None
                else:
                    # Cache failure too, avoiding automatic calls on unchanged instructions.
                    self.workflow_parameter_cache[key] = {"error":field["field"]}
                    self.request_assistance(
                        task,
                        "Ambiguous input parameter for stored field: " + field["field"],
                        f"What exact value should be entered in the stored field {field['field']!r}?",
                    )
                    if getattr(self,"user_guidance","") != guidance and not self.failure.startswith("skipped"):
                        return self.extract_workflow_parameters(action,task)
                    raise ParameterResolutionError("Input parameter unresolved: " + field["field"])
        self.workflow_parameter_cache[key] = mapping
        for field in fields:
            self.input_extraction_cache[(normalize(field["field"]),guidance,task)] = mapping.get(field["id"])
        return mapping

    def bind_replay_text(self, action, task):
        if not action:
            return action
        bound = copy.deepcopy(action)
        parameters = self.extract_workflow_parameters(action,task)
        replacements = []
        for index, step in enumerate(bound.get("element_sequence",[])):
            if step.get("atomic_action") != "text":
                continue
            params = decoded(step.get("action_params"),{}) or {}
            old = params.get("text") or params.get("input_str")
            value = parameters.get(str(index))
            if value and old and normalize(value) != normalize(old):
                replacements.append((old,value))
                # A result selected immediately after text entry is query-dependent
                # when its recorded label contains the old query, unlike a fixed
                # control such as Save or Add.
                seq = bound.get("element_sequence",[])
                if index + 1 < len(seq):
                    result_step = seq[index + 1]
                    label = content(result_step.get("target",{}))
                    fixed = NAVIGATION | COMMIT | {
                        "add", "search", "start", "stop", "pause", "reset",
                        "resume", "navigate up",
                    }
                    if result_step.get("atomic_action") == "tap" and label and label not in fixed and not result_step.get("target",{}).get("editable"):
                        field_label = content(step.get("target",{}))
                        search_field = any(word in field_label for word in ("search","city","country","region","query")) or "search" in str(step.get("target",{}).get("resource_id","")).lower()
                        if normalize(old) in label or search_field:
                            result_name = label.split("/", 1)[0].strip()
                            if result_name and normalize(result_name) != normalize(old):
                                replacements.insert(0, (result_name, value))
                            result_step["target"] = {
                                **result_step.get("target",{}),
                                "content": value,
                                "query_dependent": True,
                            }
                            result_step["target"].pop("resource_id",None)
                            result_step["target"].pop("resource-id",None)
                            self.log(f"[TEXT-BIND] Query-dependent result {label!r} now selects requested value {value!r}; fixed controls retained.")
                self.log(f"[TEXT-BIND] Explicit task/guidance replaces stored text {old!r} with {value!r}; dependent targets follow the replacement.")
        def rewrite(value):
            if isinstance(value,str):
                if replacements:
                    lookup = {normalize(old):new for old,new in replacements}
                    alternatives = "|".join(re.escape(old) for old in sorted(lookup,key=len,reverse=True))
                    value = re.sub(r"(?<!\w)(?:" + alternatives + r")(?!\w)",
                                   lambda match:lookup[normalize(match.group(0))],value,flags=re.I)
                return value
            if isinstance(value,list): return [rewrite(v) for v in value]
            if isinstance(value,dict): return {k:(v if k in {"element_id","page_id","action_id","resource_id","screenshot","elements_json"} else rewrite(v)) for k,v in value.items()}
            return value
        for key in ("element_sequence","final_screen"):
            if key in bound: bound[key] = rewrite(bound[key])
        return bound

    def request_assistance(self, task, missing, question=None):
        callback = getattr(self, "assistance_callback", None)
        if callback is None:
            return True  # Noninteractive callers retain ordinary ReAct fallback.
        question = question or f"What exact information is required to continue: {missing}?"
        self.log(f"[ASSISTANCE] Waiting for user: {question} No further device actions until answered.")
        answer = callback({"task": task, "missing": missing, "question": question,
                           "history": self.history[-8:]})
        if not answer or answer.get("skip"):
            self.failure = "skipped: add a new test case for this workflow"
            return False
        self.user_guidance = answer.get("info", "")
        self.form_resolved.clear()
        for field in list(self.form_values):
            value = self.explicit_text_value(field,task)
            if value:
                self.form_values[field] = value
        if getattr(self,"react_rejoin",None):
            stored,limit,index = self.react_rejoin
            self.react_rejoin = (self.bind_replay_text(stored,task),limit,index)
        self.react_use_vision = True
        return True

    def open_app(self, app, task):
        if not app:
            return True
        label = normalize(app)
        for attempt in range(3):
            page = self.observe()
            targets = [i for i,e in enumerate(elements(page)) if content(e) == label and not unsuitable_target(e)]
            if len(targets) == 1:
                return self.act({"action":"tap", "bbox":box(elements(page)[targets[0]])}, "launcher")
            searches = [e for e in elements(page) if ("search" in content(e) or "search" in normalize(e.get("resource_id"))) and not unsuitable_target(e)]
            if searches:
                field = searches[0]
                if not self.act({"action":"tap","bbox":box(field)}, "launcher"):
                    break
                current = self.observe()
                fields = [e for e in elements(current) if e.get("type") == "input" and not unsuitable_target(e)]
                target = fields[0] if fields else field
                if not self.act({"action":"text","bbox":box(target),"input_str":app,"replace":True}, "launcher"):
                    break
            elif attempt == 0:
                if not self.act({"action":"swipe","direction":"up"}, "launcher"):
                    break
            else:
                break
        self.app_needs_react = True
        return self.request_assistance(task, f"App {app!r} was not found on Home or launcher search")

    def recover_missing_tab(self, target, app):
        """Bounded tab recovery: Back, Home, launcher swipe, reopen app."""
        self.log("[RECOVERY] Tab missing: try Back first.")
        if self.act({"action":"back"},"recovery"):
            idx = target_index(target,self.observe())
            if idx is not None:
                return idx
        self.log("[RECOVERY] Tab still missing: return Home and try the app icon first.")
        if not app or self.remaining <= 0 or not self.home_fn():
            return None
        self.metrics["home_attempts"] += 1
        self.page = None
        home = self.observe()
        icons = [e for e in elements(home) if content(e) == normalize(app) and not unsuitable_target(e)]
        if len(icons) == 1:
            if not self.act({"action":"tap","bbox":box(icons[0])},"launcher"):
                return None
            return target_index(target,self.observe())
        self.log("[RECOVERY] Last resort: swipe up, tap launcher search, type app name, open matching result.")
        if not self.act({"action":"swipe","direction":"up"},"recovery"):
            return None
        drawer = self.observe()
        searches = [e for e in elements(drawer) if ("search" in content(e) or "search" in normalize(e.get("resource_id"))) and not unsuitable_target(e)]
        if not searches or not self.act({"action":"tap","bbox":box(searches[0])},"launcher"):
            return None
        current = self.observe()
        fields = [e for e in elements(current) if e.get("type") == "input" and not unsuitable_target(e)]
        if not fields or not self.act({"action":"text","bbox":box(fields[0]),"input_str":app,"replace":True},"launcher"):
            return None
        results = [e for e in elements(self.observe()) if content(e) == normalize(app) and e.get("type") != "input" and not unsuitable_target(e)]
        if len(results) != 1 or not self.act({"action":"tap","bbox":box(results[0])},"launcher"):
            return None
        return target_index(target,self.observe())

    def replay(self, action, limit):
        seq = action["element_sequence"]
        index = self.align(seq, limit)
        if index is None and getattr(self, "start_from_home", False):
            index = 0
            self.log("[RECOVERY] Already started from Home; locate pending stored target in the current app instead of restarting.")
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
        # Alignment may legitimately resume after an input already present on screen.
        self.verified_existing_text = getattr(self,"verified_existing_text",set())
        for prior in seq[:index]:
            params = decoded(prior.get("action_params"),{}) or {}
            value = params.get("text") or params.get("input_str")
            if prior.get("atomic_action") == "text" and value and input_value_present(value,self.observe(),prior.get("target",{})):
                self.verified_existing_text.add(normalize(value))
        self.resume_index = index + 1
        self.log(f"Replay starts at stored step {index+1}")
        cur_idx = index
        while cur_idx < limit:
            step = seq[cur_idx]
            number = cur_idx + 1
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
                # An exact stored case gets one fresh deterministic observation
                # before model recovery. This is bounded and runs only on a miss.
                if idx is None and self.exact_task_match and number not in self.exact_refresh_steps:
                    self.exact_refresh_steps.add(number)
                    self.log(f"[RECOVERY] Exact stored step {number}: refresh live UI once before model recovery.")
                    self.page = None
                    page = self.observe()
                    idx = target_index(step.get("target", {}), page)
                    if idx is not None:
                        self.log(f"[RECOVERY] Exact stored target found after refresh at index={idx}; model call skipped.")
                # If target is missing or screen differs from expected source, check if the screen is already at a further stored step
                if idx is None or (step_role(step) == "navigation" and step.get("source") and screen_score(step.get("source", {}), page) < 0.60):
                    fast_forward_idx = None
                    for f_idx in range(cur_idx + 1, limit):
                        f_step = seq[f_idx]
                        intervening_safe = True
                        for prior in seq[cur_idx:f_idx]:
                            p_role = step_role(prior)
                            p_params = decoded(prior.get("action_params"), {}) or {}
                            p_val = p_params.get("text") or p_params.get("input_str")
                            if p_role not in {"navigation", "input"}:
                                intervening_safe = False
                                break
                            if p_role == "input" and not (p_val and input_value_present(p_val,page,prior.get("target",{}))):
                                intervening_safe = False
                                break
                        if not intervening_safe:
                            break
                        f_score, f_accepted, f_basis = replay_source_match(f_step, page)
                        if f_accepted:
                            fast_forward_idx = f_idx
                            self.log(f"[REPLAY] Live screen matches further stored step {f_idx+1} ({f_basis}, similarity={f_score:.3f}). Fast-forwarding replay from step {number} to {f_idx+1}.")
                            break
                    if fast_forward_idx is not None:
                        cur_idx = fast_forward_idx
                        step = seq[cur_idx]
                        number = cur_idx + 1
                        atomic = step.get("atomic_action")
                        params = decoded(step.get("action_params"), {}) or {}
                        command = {"action": atomic}
                        idx = target_index(step.get("target", {}), page)

                visual_box = None
                tab_target = content(step.get("target", {})) in {"alarm","world clock","stopwatch","timer","pictures","albums"}
                if idx is None and tab_target:
                    idx = self.recover_missing_tab(step.get("target", {}),action.get("app_name", ""))
                    page = self.observe()
                    if self.failure.startswith("skipped"):
                        return False
                if idx is None:
                    self.log(f"[RECOVERY] Locate stored target {content(step.get('target', {}))!r} on the current screenshot; no Home or blind swipe.")
                    try:
                        located = self.ask("recover_vision", "Locate only the intended stored element on the current smartphone screenshot. "
                            "Use medium-depth analysis: compare its label, description, role and nearby controls, including tabs. "
                            "UI position changes do not mean missing. Do not complete the task or choose another control. "
                            "Return JSON with found:boolean, element_id:zero-based live index or null, x:normalized number or null, "
                            "y:normalized number or null, confidence:number, reason:concrete visible evidence. If absent return found:false.",
                            {"task":getattr(self,"current_task", ""), "stored_target":step.get("target", {}),
                             "user_guidance":getattr(self,"user_guidance", ""),
                             "stored_step_description":step.get("description") or step.get("reasoning"),
                             "stored_page_labels":[content(e) for e in meaningful(step.get("source",{}))],
                             "live_elements":[{"index":i,"content":content(e),"type":e.get("type")} for i,e in enumerate(elements(page))]}, True)
                        proof = normalize(located.get("reason", ""))
                        confidence = float(located.get("confidence",0))
                        valid = located.get("found") is True and .85 <= confidence <= 1 and proof not in {"", "visible proof", "visual evidence", "brief evidence"}
                        candidate = located.get("element_id")
                        if tab_target:
                            expected_tab = content(step.get("target", {}))
                            if isinstance(candidate,int) and not isinstance(candidate,bool) and 0 <= candidate < len(elements(page)):
                                candidate_label = content(elements(page)[candidate])
                                valid = valid and (candidate_label == expected_tab or (expected_tab in proof and candidate_label not in {"alarm","world clock","stopwatch","timer","pictures","albums"} and not re.search(r"\d",candidate_label)))
                            else:
                                valid = valid and expected_tab in proof
                            if not valid:
                                self.log("[RECOVERY] Rejecting VLM target: it does not identify the requested tab.")
                        if isinstance(candidate,int) and not isinstance(candidate,bool) and 0 <= candidate < len(elements(page)):
                            valid = valid and recovered_target_valid(step.get("target",{}),elements(page)[candidate],atomic)
                            if not valid:
                                self.log(f"[RECOVERY] Rejected index {candidate}: label/role {content(elements(page)[candidate])!r} does not match stored target {content(step.get('target',{}))!r}.")
                        if valid and isinstance(candidate,int) and not isinstance(candidate,bool) and 0 <= candidate < len(elements(page)) and not unsuitable_target(elements(page)[candidate],atomic):
                            idx = candidate
                        elif valid and not elements(page) and atomic != "text" and all(isinstance(located.get(k),(int,float)) and not isinstance(located[k],bool) and math.isfinite(located[k]) and 0 <= located[k] <= 1 for k in ("x","y")) and located["y"] > .06:
                            x,y=located["x"],located["y"]
                            visual_box=(max(0,x-.001),max(0,y-.001),min(1,x+.001),min(1,y+.001))
                        self.log(f"[RECOVERY] Visual target accepted={idx is not None or visual_box is not None}; evidence={located.get('reason','none')}.")
                    except Exception as exc:
                        self.log(f"[RECOVERY] Visual location failed: {type(exc).__name__}: {exc}")
                if idx is None and visual_box is None:
                    self.failure = "ambiguous_target"
                    self.react_rejoin = (action, limit, number-1)
                    target_label = content(step.get("target", {})) or "unlabeled control"
                    question = (
                        f"Stored step {number} requires the control {target_label!r}, but it was not found after "
                        "a fresh UI parse and one visual check. What exact visible label or control should be tapped?"
                    )
                    first_request = number not in self.assistance_attempted_steps
                    self.assistance_attempted_steps.add(number)
                    previous_guidance = getattr(self,"user_guidance", "")
                    answered = self.request_assistance(
                        getattr(self,"current_task", ""),
                        f"Stored step {number}: {target_label}",
                        question,
                    ) if first_request else False
                    if (answered and self.exact_task_match and not self.failure.startswith("skipped")
                            and getattr(self,"user_guidance", "") != previous_guidance):
                        self.log(f"[REPLAY-RESUME] Guidance received for exact stored step {number}; rebind and resume the same stored plan.")
                        rebound = self.bind_replay_text(action,getattr(self,"current_task", ""))
                        self.page = None
                        return self.replay(rebound,limit)
                    return False
                command["bbox"] = visual_box or box(elements(page)[idx])
                self.log(f"[REPLAY] Executing resolved stored target: index={idx}, bbox={command.get('bbox')}.")
            if atomic == "text":
                text = self.explicit_text_value(content(step.get("target",{}))) or params.get("text") or params.get("input_str")
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
            if atomic == "text" and input_value_present(command["input_str"],page,step.get("target",{}),idx):
                self.log("[FORM] Recorded value already visible; skipping duplicate text entry.")
                self.verified_existing_text.add(normalize(command["input_str"]))
                self.metrics["already_filled"] += 1
                cur_idx += 1
                continue
            if not self.act(command, "replayed"):
                return False
            self.history[-1]["role"] = step_role(step)
            self.history[-1]["target"] = content(step.get("target", {}))
            after = self.observe()
            signature = lambda p: [(content(e),e.get("selected"),e.get("checked")) for e in meaningful(p)]
            input_failed = atomic == "text" and not input_value_present(command["input_str"],after,step.get("target",{}),idx)
            changed_expected = page_layout_score(step.get("source",{}),step.get("destination",{})) < .95
            # A focus tap need not alter labels or the recorded page layout.
            # Exempt only a proven live input followed by its stored text step.
            focus_tap = (atomic == "tap" and idx is not None
                         and (elements(page)[idx].get("editable") or elements(page)[idx].get("type") in {"input", "edittext"})
                         and cur_idx + 1 < limit and seq[cur_idx + 1].get("atomic_action") == "text"
                         and target_index(seq[cur_idx + 1].get("target", {}), after) is not None)
            if focus_tap:
                self.log("[REPLAY-EFFECT] Input focus tap accepted; next stored text entry must verify its value on screen.")
            no_effect = not focus_tap and atomic in {"tap","text","long_press"} and signature(page) == signature(after) and changed_expected and page_layout_score(step.get("destination",{}),after) <= .70
            if input_failed or no_effect:
                self.history[-1]["status"] = "unverified"
                self.replay_unverified = True
                self.failure = f"uncertain: stored step {number} target={content(step.get('target',{}))!r} had no verified UI effect; text_visible={not input_failed}"
                self.log("[REPLAY-EFFECT] " + self.failure)
                self.react_rejoin = (action,limit,cur_idx)
                if getattr(self,"assistance_callback",None):
                    self.request_assistance(getattr(self,"current_task",""),self.failure)
                return False
            cur_idx += 1
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
            value = self.resolve_input_parameter(f["key"],task) or ((match.group(1) or match.group(2)).strip() if match else None)
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
            result = self.ask("form", FORM_PROMPT, {"task": task, "user_guidance":getattr(self,"user_guidance",""), "fields": [f["key"] for f in missing], "stored_values": stored, "already_resolved": self.form_values})
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
            if not input_value_present(value,self.observe(),elements(page)[field["index"]]):
                self.failure = "input_not_verified"
                return "error"
            self.filled.add((key, value))
            self.log(f"[FORM] {key!r}: entered value verified in fresh screen JSON.")
            return "filled"
        return "none"

    def pending_operations(self):
        required = getattr(self, "required_operations", [])
        position = 0
        for entry in self.history:
            if entry.get("status") == "success" and position < len(required) and entry.get("target") == required[position]:
                position += 1
        pending = required[position:]
        typed = [normalize(h.get("params",{}).get("input_str")) for h in self.history
                 if h.get("status") == "success" and h.get("action") == "text"]
        for value in getattr(self, "required_text_values", []):
            if normalize(value) not in typed and normalize(value) not in getattr(self,"verified_existing_text",set()):
                pending.append("enter text: " + str(value))
        city = getattr(self,"required_world_clock_city",None)
        if city:
            live = elements(self.observe())
            # A visible search result or editor is not proof that the city was added.
            search_open = any(e.get("editable") or e.get("type") in {"input","edittext"} for e in live)
            city_visible = any(not e.get("editable") and re.search(r"(?<!\w)" + re.escape(normalize(city)) + r"(?!\w)",content(e)) for e in live)
            labels = {content(e) for e in live}
            world_page = "world clock" in labels and not (labels & {"add city","clear search field"})
            if search_open or not city_visible or not world_page:
                pending.append("verify city added to World Clock: " + str(city))
        return pending

    def navigation_goal_contradicted(self, task):
        if "world clock" not in normalize(task):
            return False
        labels = [content(e) for e in meaningful(self.observe())]
        selected = {content(e) for e in meaningful(self.observe()) if e.get("selected")}
        contradicted = (any("hour" in label for label in labels) and any("minute" in label for label in labels)) or (bool(selected & {"alarm", "stopwatch", "timer"}) and "world clock" not in selected)
        if contradicted:
            self.log("[VERIFY] Visible alarm editor or selected tab contradicts the World clock goal.")
        return bool(contradicted)

    def world_clock_navigation_complete(self, task):
        # Only the destination-only goal; city/time modification needs its own proof.
        intent = navigation_intent(task)
        if intent != navigation_intent("Go to world clock"):
            return False
        page = self.observe()
        if self.pending_operations() or self.navigation_goal_contradicted(task):
            return False
        labels = [content(e) for e in meaningful(page)]
        selected = {content(e) for e in meaningful(page) if e.get("selected")}
        distinctive = ("world clock" in labels and any("local time zone" in label for label in labels)
                       and any("hours behind" in label or "hours ahead" in label for label in labels)
                       and any(re.search(r"\d+[.:]\d+",label) for label in labels))
        if "world clock" in selected or distinctive:
            self.evidence = "World Clock destination confirmed by selected tab or local time-zone and time-offset content; current clock time is dynamic."
            self.log("[VERIFY] " + self.evidence + " No VLM call required.")
            return True
        return False

    def stopwatch_reset_complete(self, task):
        goal = normalize(task)
        if "stopwatch" not in goal or not re.search(r"\breset\b",goal):
            return False
        page = self.observe()
        labels = [content(e) for e in meaningful(page)]
        selected = {content(e) for e in meaningful(page) if e.get("selected")}
        achieved = [entry.get("target") for entry in self.history if entry.get("status") == "success"]
        start_pos = next((i for i,label in enumerate(achieved) if label == "start"),None)
        reset_after_start = start_pos is not None and "reset" in achieved[start_pos+1:]
        zero = any(re.fullmatch(r"0+(?:[.:]0+)+(?:\s*seconds?)?",label) for label in labels)
        if "stopwatch" in selected and zero and "start" in labels and not ({"stop","pause","resume"} & set(labels)) and reset_after_start and not self.pending_operations():
            self.evidence = "Stopwatch selected, elapsed time is zero and Start is visible; successful Start then Reset actions recorded."
            self.log("[VERIFY] " + self.evidence + " No VLM judgment required.")
            return True
        return False

    def verify(self, task, prefer_vision=False):
        if getattr(self,"replay_unverified",False):
            self.log("[VERIFY] Completion blocked: a replay action had no verified UI effect.")
            return False
        if self.world_clock_navigation_complete(task) or self.stopwatch_reset_complete(task):
            return True
        if self.pending_operations() or self.navigation_goal_contradicted(task):
            self.log(f"[VERIFY] Completion blocked; outstanding stored operations={self.pending_operations()}.")
            return False
        self.log("[VERIFY] Checking whole-task completion; save attempts alone are not proof.")
        # Completion does not need tap coordinates, parser IDs, or verbose metadata.
        compact = lambda page: [{k: e[k] for k in ("content", "text", "type", "selected", "value", "editable") if k in e} for e in meaningful(page)]
        history = [{"action": h.get("action"), "target": h.get("target"), "status": h.get("status"),
                    "input_str": h.get("params", {}).get("input_str")} for h in self.history[-8:]]
        payload = {"task": task, "elements": compact(self.observe()), "history": history, "form_values": self.form_values}
        judge_prompt = JUDGE_PROMPT
        if self.react_final_screen:
            payload["expected_final_elements"] = compact(self.react_final_screen)
            payload["parsed_final_similarity"] = page_layout_score(self.react_final_screen,self.observe())
            if prefer_vision:
                judge_prompt += ("\nCompare the attached live screenshot with expected_final_elements semantically. "
                    "The parsed similarity is inconclusive, not proof of failure. Ignore layout and harmless wording differences. "
                    "Compare active tab and page layout only; ignore numeric values, current times, AM/PM and notification-bar values. Return semantic_similarity (number 0 to 1) "
                    "alongside complete, confidence, evidence and missing. Complete means the requested goal and relevant stored "
                    "final state agree; common navigation labels alone are insufficient. Explain concrete matching content or differences.")
                self.log("[VERIFY] Parsed final similarity is at or below the threshold; requesting VLM semantic comparison of live image and stored final content.")
        if prefer_vision:
            self.log("[VERIFY] Stored JSON did not confirm completion; inspect the fresh screenshot before any further action. Skipping an extra text-only judge.")
            self.react_use_vision = True
        for vision in ((True,) if prefer_vision or not elements(self.observe()) else (False, True)):
            try:
                result = self.ask("judge_vision" if vision else "judge_text", judge_prompt, payload, vision)
                confidence = float(result.get("confidence", 0))
                if "semantic_similarity" in result:
                    self.log(f"[VERIFY] VLM semantic similarity={result.get('semantic_similarity')}; evidence={result.get('evidence','')}; differences={result.get('missing','')}.")
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
                    if normalize(self.evidence) in {"", "visible proof", "visual evidence", "brief evidence", "task completed", "completed"}:
                        self.evidence = ""
                        self.log("[VERIFY] Rejecting generic/template completion evidence.")
                        return False
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
        if self.world_clock_navigation_complete(task) or self.stopwatch_reset_complete(task):
            return True
        if self.react_final_screen:
            score = page_layout_score(self.react_final_screen, page)
            self.log(f"[REACT-PROGRESS] Relevant stored final page layout similarity={score:.3f}; required > 0.70. No model call.")
            if score > .70 and not self.pending_operations() and not self.navigation_goal_contradicted(task):
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
            rejoin = getattr(self, "react_rejoin", None)
            if rejoin:
                stored, limit, index = rejoin
                pending = stored["element_sequence"][index]
                if target_index(pending.get("target", {}), page) is not None:
                    self.react_rejoin = None
                    remainder = copy.deepcopy(stored)
                    remainder["element_sequence"] = remainder["element_sequence"][index:limit]
                    self.log("[REJOIN] Missing target found; resume remaining stored steps without model calls.")
                    if self.replay(remainder, len(remainder["element_sequence"])):
                        return self.verify(task, prefer_vision=True)
            payload = {"task": task, "user_guidance": getattr(self,"user_guidance", ""), "elements": [{**e, "index": i} for i, e in enumerate(elements(page))], "history": self.history[-8:], "form_values": self.form_values, "replay_failure": self.failure}
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
            if normalize(result.get("reason", "")) in {"brief evidence", "visible proof", "visual evidence", "field label"}:
                if self.world_clock_navigation_complete(task) or self.stopwatch_reset_complete(task):
                    return True
                self.failure = "uncertain: model returned placeholder action reasoning; no further device action dispatched"
                self.log("[STOP] " + self.failure)
                if getattr(self,"assistance_callback",None):
                    self.request_assistance(task,"Model returned a template response instead of identifying the intended action; provide guidance or skip")
                return False
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
                    if getattr(self,"assistance_callback",None):
                        if not self.request_assistance(task,"ReAct could not locate a usable app/element target"):
                            return False
                        self.remaining -= 1
                        self.react_use_vision = True
                        continue
                    return False
            if atomic == "text":
                field = normalize(result.get("field", ""))
                value = self.resolve_input_parameter(field,task) or self.form_values.get(field, result.get("input_str"))
                if not valid_value(field, value):
                    self.failure = "invalid_input"
                    return False
                if (field, value) in self.filled and input_value_present(value,page,elements(page)[idx],idx):
                    self.log("[TEXT] Requested value is already visible; skip repeated input and require a different next action.")
                    self.remaining -= 1
                    repeat_key = ("filled-input",field,normalize(value),tuple(content(e) for e in elements(page)))
                    repeats[repeat_key] += 1
                    if repeats[repeat_key] >= 2:
                        self.failure = "uncertain: repeated input proposed instead of selecting the next result"
                        self.log("[STOP] " + self.failure)
                        return False
                    self.failure = "uncertain: repeated input requested despite value already present"
                    self.log("[REJOIN] Check the pending stored result before requesting another model action.")
                    continue
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
            if isinstance(result.get("element_id"),int) and not isinstance(result["element_id"],bool) and 0 <= result["element_id"] < len(elements(page)):
                self.history[-1]["target"] = content(elements(page)[result["element_id"]])
            if atomic == "text":
                if not input_value_present(command["input_str"],self.observe(),elements(page)[idx]):
                    self.failure = "input_not_verified"
                    return False
                self.filled.add((field, command["input_str"]))
            if self.check_react_progress(task):
                return True
        # Each action has already been checked; do not repeat a final judge.
        self.failure = "budget_exhausted"
        return False

    def _skipped_result(self, action, relation, limit):
        return {"status":"skipped", "completed":False, "message":"Add a new test case for this workflow", "history":self.history,
                "metrics":dict(self.metrics), "close_actions":[], "plan":{"action_id":(action or {}).get("action_id"),"relation":relation,"replay_prefix":limit}}

    def run(self, task, force_fallback=False):
        action, relation, limit = None, "none", 0
        complete = False
        self.current_task = task
        self.log(f"[START] Task={task!r}; max actions={self.max_steps}; force ReAct={force_fallback}.")
        try:
            if not force_fallback:
                action, relation, limit = self.plan(task)
            else:
                self.log("[PLAN] Forced fallback selected; stored-task matching skipped.")
            action = self.bind_replay_text(action,task)
            self.required_text_values = [p.get("text") or p.get("input_str")
                for s in (action or {}).get("element_sequence",[])[:limit]
                if s.get("atomic_action") == "text"
                for p in [decoded(s.get("action_params"), {}) or {}]
                if p.get("text") or p.get("input_str")]
            self.required_world_clock_city = None
            if "world clock" in normalize(task) and re.search(r"\badd\b",normalize(task)):
                self.required_world_clock_city = self.explicit_text_value("City/country/region",task)
                if not self.required_world_clock_city and self.required_text_values:
                    self.required_world_clock_city = self.required_text_values[-1]
            self.required_operations = [content(s.get("target",{})) for s in (action or {}).get("element_sequence",[])[:limit]
                                        if content(s.get("target",{})) in {"start","stop","pause","reset","delete","save","resume"}]
            if "stopwatch" in normalize(task) and re.search(r"\bstart\b",normalize(task)) and re.search(r"\breset\b",normalize(task)):
                if "start" not in self.required_operations:
                    self.required_operations.insert(0,"start")
                if "reset" not in self.required_operations:
                    self.required_operations.append("reset")
            if getattr(self, "start_from_home", False):
                self.log("[HOME] Starting task from Home after task matching.")
                if not self.home_fn():
                    raise RuntimeError("Could not start task from Home")
                self.page = None
                app = (action or {}).get("app_name", "")
                if not self.open_app(app, task):
                    raise RuntimeError(self.failure)
                if getattr(self,"app_needs_react",False) and action and limit:
                    self.react_rejoin = (action,limit,0)
                    limit = 0
                if app and action and limit:
                    current = self.observe()
                    first = action["element_sequence"][0]
                    launched_step = (first.get("atomic_action") == "tap" and
                        (content(first.get("target", {})) == normalize(app) or
                         normalize(first.get("destination", {}).get("app_name")) == normalize(app)))
                    source_labels = {content(e) for e in meaningful(first.get("source",{}))}
                    if first.get("atomic_action") == "tap" and len(source_labels & {"play store","galaxy store","google","tap for weather info"}) >= 2:
                        launched_step = True
                    if launched_step and limit > 1 and (len(source_labels & {"play store","galaxy store","google","tap for weather info"}) >= 2 or target_index(action["element_sequence"][1].get("target", {}), current) is not None):
                        action = copy.deepcopy(action)
                        action["element_sequence"] = action["element_sequence"][1:]
                        limit -= 1
                        self.log("[LAUNCHER] App entry confirmed by the next stored target; stored app-opening step already satisfied.")
                    elif launched_step and limit == 1 and page_layout_score(first.get("destination", {}),current) > .70 and not self.navigation_goal_contradicted(task):
                        complete = True
                        limit = 0
                        self.evidence = "App opened from Home and its stored final screen verified"
            if relation == "prefix":
                self.react_final_screen = self.prefix_goal_screen
            if relation == "equivalent" and action:
                self.react_final_screen = action.get("final_screen") or action["element_sequence"][-1].get("destination", {})
            if limit:
                replay_ok = self.replay(action, limit)
                if not replay_ok:
                    self.log(f"[FALLBACK] Replay stopped: {self.failure}; preserve successful history for ReAct.")
                    if self.failure.startswith("skipped"):
                        return self._skipped_result(action,relation,limit)
                if replay_ok and (relation == "prefix" or (relation == "equivalent" and limit == len(action["element_sequence"]))):
                    final = self.prefix_goal_screen if relation == "prefix" else (action.get("final_screen") or action["element_sequence"][-1].get("destination", {}))
                    # Verify only the requested goal boundary after replay.
                    self.metrics["replay_final_checks"] += 1
                    final_score = page_layout_score(final, self.observe())
                    self.log(f"[VERIFY] Last stored step finished. ONE final-page layout comparison: score={final_score:.3f}, required > 0.70.")
                    complete = self.world_clock_navigation_complete(task) or (final_score > .70 and not self.pending_operations() and not self.navigation_goal_contradicted(task))
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
            if (not complete and self.exact_task_match and self.failure == "ambiguous_target"
                    and not self.failure.startswith("skipped")):
                self.failure = "uncertain: exact stored step remains unresolved after bounded recovery"
                self.log("[FALLBACK] Exact stored plan is retained; open-ended ReAct is disabled after bounded target recovery.")
            if not complete and not self.failure.startswith("skipped") and not getattr(self,"replay_unverified",False) and not self.failure.startswith("uncertain:"):
                if self.failure == "restart_unmatched":
                    self.react_rejoin = (action,limit,0) if action and limit else None
                    self.request_assistance(task, "Stored starting screen or target cannot be located")
                if self.failure.startswith("skipped"):
                    return self._skipped_result(action, relation, limit)
                if self.model_backend_error:
                    raise RuntimeError(f"Model request failed: {self.model_backend_error}; ReAct needs the same backend, so no duplicate request is dispatched")
                complete = self.react(task, action)
        except ParameterResolutionError as exc:
            if self.failure.startswith("skipped"):
                retained = self.matched_plan or (action,relation,limit)
                return self._skipped_result(*retained)
            if self.matched_plan:
                action,relation,limit = self.matched_plan
            self.failure = "uncertain: " + str(exc)
            self.log(f"[PLAN-RECOVERY] Matched stored plan retained: id={(action or {}).get('action_id')}, "
                     f"relation={relation}, steps={limit}. Full ReAct is not started without the required input.")
        except Exception as exc:
            if self.failure.startswith("skipped"):
                return self._skipped_result(action,relation,limit)
            self.failure = f"error: {exc}"
            self.log(self.failure)
        if self.failure.startswith("skipped"):
            return self._skipped_result(action,relation,limit)
        self.log(f"[FINISH] status={'completed' if complete else self.failure or 'failed'}, replayed={self.metrics['replayed_steps']}, react={self.metrics['react_steps']}, text calls={self.metrics['text_calls']}, vision calls={self.metrics['vision_calls']}, parser calls={self.metrics['parser_calls']}; evidence={self.evidence or 'none'}.")
        return {"status": "completed" if complete else (self.failure or "failed"), "completed": complete,
                "message": self.evidence if complete else self.failure, "completion_evidence": self.evidence,
                "steps_completed": self.metrics["replayed_steps"] + self.metrics["react_steps"],
                "replayed_steps": self.metrics["replayed_steps"], "react_steps": self.metrics["react_steps"],
                "resumed_step": self.resume_index, "metrics": dict(self.metrics),
                "llm_calls": {k: v for k, v in self.metrics.items() if k in {"plan", "form", "react", "react_vision", "judge_text", "judge_vision"}},
                "plan": {"action_id": (action or {}).get("action_id"), "relation": relation, "replay_prefix": limit},
                "history": self.history, "stalled_state_report": getattr(self,"stalled_state_report",None), "close_actions": []}
