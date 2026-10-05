# Task-Level Reuse + Completion Fix: Implementation Guide (v3)

Files read: `deployment_old.py`, `State.py`, `adb_tools.py`, `chain_evolve.py`, `chain_understand.py`, `graph_db.py`, `data_storage.py`.
**Not uploaded (anything about them is marked [UNVERIFIED])**: `config.py`, `vector_db.py`, `nvidia_llm_bridge.py`, `llm_rate_limit.py`, `img_tool.py`, OmniParser client.
`deployment.py` (the newer file) is not used.

Goal: if the new task's app or intent is related to a stored task ("set alarm at 10" → "set alarm at 4.30"), replay the stored steps and use ReAct only for what is new. No HOME anchor. Few LLM calls.

---

## 0. What the new files changed in v1

| v1 assumption | What the code says | Change in v2 |
|---|---|---|
| `element_sequence` is a reliable recording | `chain_evolve.generate_action_node` asks an LLM to write it; nothing checks it against the chain | Rebuild it from the Element nodes at write time (Step 3), audit existing data first (Step 2) |
| Every step has an `element_id` | **Corrected in v3:** `get_chain_from_start` only returns element hops, so every recorded step has an element; an empty id can only come from an LLM-written sequence. See 0.2 for the real reason `back`/swipe/text are replayed without matching | Step 15 |
| Element text-similarity shortcut compares stored vs live text | `chain_understand` overwrites `Element.description` with LLM prose for interacted elements. **Refined in v3:** raw text is in Neo4j `Element.other_info` (`{"type","content"}`) for every element | Compare against `other_info.content`; Pinecone is only a fallback (Step 10) |
| New state keys "may" need declaring | `DeploymentState` is a `TypedDict` | Declare them (Step 4); existing `_workflow_iterations` is undeclared (see fact S3) |
| `STEP_SETTLE_SEC` sleep after each step | `take_screenshot` already does `sleep(2)` | No extra sleep (Step 6, 15) |
| Original recorded task text unavailable | It is stored at `Page.other_info.task_info.description` of the first page (`chain_evolve.extract_task_description`) | Planner sees the recorded task text, derived at runtime (Step 13) |
| Negative text check ends the run | Typed "430" may display as "4:30", 24h clocks differ | Absent text only routes the judge to vision (Step 9) |

### 0.2 Changes from reading `graph_db.py` and `data_storage.py`

| v2 assumption or open question | What the code says | Change |
|---|---|---|
| Typed text is `action_params.text`, or sits in `triplet["action"]` | `json2db` stores the recorded `screen_action` result minus `action/device/status` as `Element.parameters`, so typed text is `{"input_str": "..."}`. LEADS_TO params are only `{execution_result, timestamp}`. `extract_element_details` never shows `parameters` to the LLM, so an LLM-written `action_params` for a text step cannot know the typed text. `execute_element_action` reads `parameters["text"]` | Step 3 builds steps from Element nodes; Step 6 reads both keys |
| Swipe params are `direction`/`distance` | Recorded swipe params are `{"swipe": {"start","end","duration","direction"}}` (`swipe_precise` and `long_press` are nested the same way). No distance is recorded | Normalise at write time (Step 3); replay with `swipe_precise` when start/end exist (Step 15) |
| Steps with no element exist ("legacy direct hop") | Never from `get_chain_from_start`. But `back` and swipe steps are bound to the element nearest the screen centre (`_fallback_element_by_center`) and `text` to the first input element, so matching them against the live screen is wrong | `back`/swipe/text replay without matching (Step 15) |
| `db.create_action` may merge on id | It does: `MERGE (n:Action {action_id})` + `SET n += props`. A repeated LLM-chosen id overwrites the earlier Action's name/description/`element_sequence` while its old COMPOSED_OF relationships stay | Code-generated id (Step 3); the audit detects past collisions (Step 2) |
| bbox units unknown | OmniParser bboxes are normalised 0–1 (`pos2id` divides click coordinates by screen size; the centre fallback uses 0.5). The `< 0.03` shortcut is 3% of the screen | Keep it, but not for the entry check (Step 10) |
| Raw element text only via Pinecone | `Element.other_info` holds `{"type","content"}` for every element; `get_element_by_id` already parses it. `element_type` is **not** a property that `json2db` sets | Use Neo4j for raw text and type (Step 10, 13) |
| DB name mismatch is "possible" | `Neo4jDatabase` defaults to `database="graphdb"` and creates it if missing; `json2db` and the chain files use `config.Neo4j_DB` | Step 1 |
| Recorded task text via runtime query for every Action | `task_info` is written only on the step-0 Page (`if step_no == 0`) | Store `source_task` on the Action at write time; runtime query only for old Actions (Step 13) |
| Chain order is reliable | `get_chain_from_start` sorts by target-page `timestamp` (whole seconds) | Sort by the source page's `other_info.step` instead (Step 3) |

---

## 1. Verified facts

**deployment_old.py**
- D1. `match_task_to_action` sends every stored action with full `element_sequence`; asks the LLM to copy one back verbatim. No match → `no_match`, `completed=True`.
- D2. `match_elements_node` re-runs the LLM match on every graph loop.
- D3. `execute_action_node`: replays from step 0; a step with empty `element_id` sets `all_steps_ok=False` and breaks; after the last step sets `completed=True`; sleeps 1.5 s after every step.
- D4. `check_task_completion`: early return when `completed` and no `last_page_description`; judge images = `history[-3:]` (pre-action screenshots); criteria = stored final-page description; free-text `_is_affirmative`; prompt says identical screenshots → "yes".
- D5. `fallback_to_react` does one action per call; injects a swipe-up when the previous history entry is a `tap`; each ReAct action goes back through `check_completion` (up to 2 more LLM calls).
- D6. `match_element_via_pinecone`: if the element is missing from Neo4j and Pinecone, `stored_content` is `""`, `stored_bbox` is `None`, and it still calls `llm_bbox_fallback` with an empty target description, so the LLM picks an arbitrary live element.
- D7. `execute_element_action` calls `get_device_size.invoke` (one ADB subprocess) on every action.
- D8. `run_task`: the `except` branch has no `return`.
- D9. Module-level `db = Neo4jDatabase(URI, AUTH)` (no `database=`), while `chain_evolve.py` and `chain_understand.py` use `Neo4jDatabase(..., database=config.Neo4j_DB)`.
- D10. `bridge = NvidiaBridge(max_tokens_text=4096, max_tokens_json=4096, max_tokens_vision=2048)`; the comment says 4096 was needed for full action JSON.

**State.py**
- S1. `DeploymentState(TypedDict, total=False)`; `create_deployment_state` sets every default.
- S2. LangGraph builds channels from the TypedDict annotations. **[Likely]** keys not declared there are dropped between nodes. Test once: write an undeclared key in node A and read it in node B.
- S3. `_workflow_iterations` and `max_workflow_iterations` are written in `run_task` and read/mutated in the router `is_task_completed`, but are not declared in `DeploymentState`. **[Likely]** the iteration cap in the legacy graph never triggers (every read sees 0), and a router's mutations are not persisted anyway; a non-completing run only ends at the recursion limit (`max(50, 10*6)=60`) and then `run_task` returns `None` (D8).

**adb_tools.py**
- A1. `take_screenshot` always `sleep(2)`, then screencap + pull + rm (3 subprocesses).
- A2. `_adb(cmd)` returns `"ERROR"` on failure. `_adb` starts with an underscore, so `from tool.adb_tools import *` does not import it; the file has no `__all__`, so any new public function is imported.
- A3. `screen_action` with `action="text"` ignores `x`/`y`; it only runs `input text`. The field must already be focused by the previous tap.
- A4. Commands run through `subprocess.run(..., shell=True)`; only spaces and `'` are sanitized in `input_str`. Text containing `& | ; < > ( ) $ \ "` breaks or injects into the host command. Today `input_str` comes from the LLM (ReAct) and would come from the LLM (plan overrides).
- A5. `screen_action` supports tap, back, text, long_press, swipe/swipe_short/swipe_long, swipe_precise. No `home`.
- A6. `take_screenshot` returns the string `"Screenshot failed: ..."` on error (not `None`); `capture_and_parse_screen` already guards with `os.path.exists`.

**chain_evolve.py**
- E1. `element_sequence`, `action_id`, `name`, `description`, `template_pattern` are all LLM output (`_GEN_SYSTEM`); `create_action_node_in_db` stores them without validating. `action_id` is LLM-chosen ("high_level_action_xxx"), so collisions are possible **[Likely]**, and `db.create_action` merges on id (see G2), so a collision silently overwrites.
- E2. `preconditions`, `element_sequence`, `template_pattern` are stored with `json.dumps` (strings in Neo4j). `template_pattern` has `criteria` and `parameter_fields` (format LLM-defined, unreliable).
- E3. Only chains judged `is_templateable` become Actions. A recording that was rejected has nothing to replay.
- E4. `extract_task_description(chain)` reads the task text from `chain[0]["source_page"]["other_info"]["task_info"]["description"]`; `other_info` may be a JSON string. The Action node does not store it.

**chain_understand.py**
- C1. `process_triplet` replaces `Element.description` with `element_enhanced_desc` (LLM prose) and writes it to Neo4j.
- C2. `_persist_triplet` does `triplet["element"]` directly (KeyError when a triplet failed before `element` was set).

**graph_db.py (verified)**
- G1. `Neo4jDatabase(uri, auth, database="graphdb")`; `_ensure_database_exists` runs `CREATE DATABASE ... IF NOT EXISTS` against `system`.
- G2. `create_action` / `create_page` / `create_element` = `MERGE` on the business id + `SET n += props`; arbitrary extra properties (e.g. `source_task`) are stored. `update_node_property(..., node_type="Action")` works (`id_field = "action_id"`).
- G3. `add_element_to_action` does `MATCH` on both nodes before `MERGE (a)-[:COMPOSED_OF {order}]->(e)`: an invented `element_id` creates no relationship (and `create_action_element_relations` only logs it).
- G4. `get_all_high_level_actions` filters `coalesce(is_high_level,false)=true` and `json.loads` the `element_sequence`; `template_pattern` and `preconditions` stay JSON strings.
- G5. `get_chain_from_start` returns only element hops (`hop_type="element_hop"`); each triplet has `source_page`, `target_page`, `element` (always a dict), `action` (= LEADS_TO properties: `action_name`, `action_params` JSON string, `confidence_score`, plus `action_type` alias). It dedups on (src page, tgt page) and sorts by target `timestamp`.
- G6. `get_element_by_id` returns the node dict with `other_info` / `possible_actions` parsed from JSON when they are strings.

**data_storage.py / json2db (verified)**
- J1. Every recorded step creates new Page/Element nodes (uuid4); `Page.other_info = {"step": n}` plus `task_info{task_id, description}` **only when `step_no == 0`**; `raw_page_url` is the step screenshot.
- J2. Element properties: `element_id`, `element_original_id`, `description` (placeholder `"<type> element (ID n) at step N — '<content>'"`), `action_type`, `parameters` (JSON string), `bounding_box` (JSON string), `other_info = {"type","content"}`, `visual_embedding_id`. There is no `element_type` property.
- J3. For the interacted element: `Element.action_type = action_type`, `Element.parameters = json.dumps(tool_result minus action/device/status)`. For text: `{"input_str": ...}`; tap: `{"clicked_element": {x,y}}`; swipe: `{"swipe": {...}}`; long_press: `{"long_press": {...}}`; swipe_precise: `{"swipe_precise": {...}}`; back: `{}`.
- J4. Interacted-element resolution: tap/long_press/swipe*/back use the clicked coordinates when present (`pos2id`), else the `element_number` parsed from `recommended_action`, else the element nearest the screen centre. Swipe and back results contain no `clicked_element`, so they normally fall to the centre fallback. `text` uses `element_number`, else the first `input`/`edittext` element. If no element resolves, the step gets no LEADS_TO edge and the chain is cut there.
- J5. `LEADS_TO` params are `{"execution_result", "timestamp"}` only; `action_name` = `action_type or recommended_action`.
- J6. Per-step `Action` nodes (uuid, no `is_high_level`) are created too and are excluded by G4.
- J7. Page `timestamp` is `int(time.time())` from `record_action_to_state` (whole seconds).
- J8. Element vectors: Pinecone id = `element_id`, metadata `original_id`, `bbox`, `type`, `content` (same raw text as `other_info`).

---

## 2. Rules for Copilot

- Add new functions; keep old ones. New flow behind `USE_PLAN_REUSE` (env, default `"1"`). Rename the old `build_workflow` to `build_workflow_legacy`; `build_workflow()` returns it when the flag is `"0"`.
- Reuse: `capture_and_parse_screen`, `match_element_via_pinecone`, `execute_element_action`, `_sync_call_json`, `_sync_call_text`, `_sync_call_vision`, `_log_both`, `_img_to_b64`, `_parse_action_result`.
- The LLM never copies action objects. It returns IDs, indices, short strings.
- Everything the LLM returns is validated in code. Invalid output degrades to ReAct, never crashes.
- Constants at the top of `deployment_old.py`:

```python
USE_PLAN_REUSE        = os.getenv("USE_PLAN_REUSE", "1") == "1"
SCREENSHOT_SETTLE_SEC = float(os.getenv("SCREENSHOT_SETTLE_SEC", "2.0"))  # same as today
PLAN_MIN_CONF         = 0.6     # below: whole task goes to ReAct
JUDGE_MIN_CONF        = 0.7     # below: escalate text judge to vision judge
TEXT_MATCH_MIN        = 0.85    # raw-text similarity for the no-LLM element accept
MAX_REACT_STEPS_SEG   = 6
MAX_RETRY_REACT_STEPS = 4
STUCK_WINDOW          = 3
CATALOG_PREFILTER_OVER = 25
CATALOG_TOP_K         = 12
CATALOG_TTL_SEC       = 60
SAFE_TEXT_RE          = r"[A-Za-z0-9 :.,@+\-_/]*"   # allowed typed text from LLM/plan
```

---

# PART 0: Data checks (before any runtime change)

### Step 1: Database name (confirmed risk, pass the name explicitly)
`Neo4jDatabase.__init__` defaults to `database="graphdb"` (G1) and creates that database when it does not exist. `json2db`, `chain_evolve` and `chain_understand` pass `database=config.Neo4j_DB`; `deployment_old.py` calls `Neo4jDatabase(URI, AUTH)` with no database. `config.py` was not uploaded, so the value of `Neo4j_DB` is **[UNVERIFIED]**. If it is not `"graphdb"`, deployment silently creates and reads an empty `graphdb`, `get_all_high_level_actions()` returns `[]`, and every task ends `no_match`.
Fix regardless of the value: `db = Neo4jDatabase(URI, AUTH, database=config.Neo4j_DB)`.
**Done when:** `len(db.get_all_high_level_actions())` equals the number of high-level Actions in the Neo4j browser for that database.

### Step 2: `audit_actions.py` (read-only)
For every Action from `db.get_all_high_level_actions()` (`element_sequence` is already a list, G4), report:
- high-level Actions vs `len(db.get_chain_start_nodes())` (E3: recordings that never became Actions);
- **invented ids:** `rels = db.get_action_sequence(action_id)`. `len(rels) < len(element_sequence)` means some `element_id`s do not exist (G3);
- **id collisions:** `len(rels) > len(element_sequence)` means a later Action reused the `action_id` and overwrote the properties while the old COMPOSED_OF relationships stayed (G2, G3);
- **LLM damage:** one query `MATCH (e:Element) WHERE e.element_id IN $ids RETURN e.element_id AS id, e.action_type AS at, e.parameters AS p` (via `db.driver.session(database=db.database)`). Compare per step: `atomic_action` vs `Element.action_type`, and typed text (`action_params.text` or `.input_str`) vs the recorded `Element.parameters.input_str`. Count mismatches and empty text params;
- `atomic_action` histogram (allowed: `tap, text, long_press, swipe, swipe_short, swipe_long, swipe_precise, back`); swipe steps without a direction;
- how many Actions have a derivable task text (the page owning step 1's element has `other_info.task_info.description`, J1).
**Done when:** you have the counts. Any invented id, collision, or text/swipe mismatch means the stored `element_sequence` cannot be replayed faithfully; do Step 3 and the migration before trusting replay.

### Step 3: `chain_evolve.py`: build the sequence from the Element nodes (not from the LLM)
The recording already holds everything needed (J3): `Element.action_type` and `Element.parameters` on each interacted element. Use the LLM only for `name`, `description`, `preconditions`, `template_pattern`.

```python
import uuid
ALLOWED_ATOMIC = {"tap","text","long_press","swipe","swipe_short","swipe_long","swipe_precise","back"}

def _json(v):
    if isinstance(v, dict): return v
    try: return json.loads(v) if v else {}
    except Exception: return {}

def _source_step(t):                       # other_info.step of the source page (J1)
    return _json(t["source_page"].get("other_info")).get("step", 10**9)

def normalise_params(atomic, raw):         # raw = Element.parameters (J3)
    if atomic == "text":
        txt = raw.get("text") or raw.get("input_str") or ""
        return {"text": txt} if txt else None
    if atomic in ("swipe", "swipe_short", "swipe_long"):
        sw = raw.get("swipe") or {}
        d = raw.get("direction") or sw.get("direction")
        return {"direction": d, "start": sw.get("start"), "end": sw.get("end")} if d else None
    if atomic == "swipe_precise":
        sp = raw.get("swipe_precise") or {}
        return ({"start": sp["start"], "end": sp["end"], "duration": sp.get("duration", 400)}
                if sp.get("start") and sp.get("end") else None)
    if atomic == "long_press":
        lp = raw.get("long_press") or {}
        return {"duration": raw.get("duration") or lp.get("duration", 1000)}
    return {}                              # tap, back

def build_sequence_from_chain(chain):
    seq = []
    for i, t in enumerate(sorted(chain, key=_source_step)):     # stable sort
        e = t["element"]
        atomic = e.get("action_type") or t["action"].get("action_name", "")
        if atomic not in ALLOWED_ATOMIC:
            return None, f"step {i+1}: unsupported action '{str(atomic)[:30]}'"
        params = normalise_params(atomic, _json(e.get("parameters")))
        if params is None:
            return None, f"step {i+1}: {atomic} has no usable parameters"
        seq.append({"element_id": e["element_id"], "order": i + 1,
                    "atomic_action": atomic, "action_params": params})
    return seq, "ok"
```
In `evolve_chain_to_action`, after `generate_action_node` succeeds:
```python
seq, why = build_sequence_from_chain(chain)
if seq is None:
    print(f"Chain not storable: {why}"); return None       # refuse; never store a guessed step
if seq != action_data.get("element_sequence"):
    print("[evolve] LLM element_sequence differed; using the chain-derived one")
action_data["element_sequence"] = seq
action_data["action_id"]   = f"high_level_action_{uuid.uuid4().hex[:8]}"   # G2: no collisions
action_data["source_task"] = extract_task_description(chain)
```
and in `create_action_node_in_db` add `"source_task": action_data.get("source_task", "")` to `properties` (accepted, G2).

**Optional, saves tokens:** remove `action_id` and `element_sequence` from `_GEN_SYSTEM`'s required keys and from the `"action_id" in result` check (use `"name" in result`). The deployment comment says the sequence alone was ~900 tokens, and the LLM output for it is now discarded.

**Migration (`rebuild_actions.py`):** record the old high-level `action_id`s; for each `start in db.get_chain_start_nodes()` run `await evolve_chain_to_action(start["page_id"])`; once a recording has a new Action, retire the old ones non-destructively with `db.update_node_property(old_id, "is_high_level", False, node_type="Action")` (G4 then hides them). This repeats the templateability LLM call per recording.
**Done when:** a recording containing a text step, a back step and a swipe yields a sequence whose length equals the chain length, whose text step has `{"text": "<recorded value>"}`, and whose Action has `source_task` set.

---

# PART 1: Foundations

### Step 4: `State.py`: declare new keys
Add to `DeploymentState` and give defaults in `create_deployment_state`:

```python
plan: Optional[Dict]            # None
seg_index: int                  # 0
react_steps: int                # 0
replayed_steps: int             # 0
retries: int                    # 0
final_screenshot: Optional[str] # None
final_elements: List[Dict]      # []
screen_hashes: List[str]        # []
device_size: Optional[Dict]     # None
finished: bool                  # False
workflow_iterations: int        # 0   (replaces undeclared _workflow_iterations)
max_workflow_iterations: int    # 10
```
**Done when:** a node sets `state["plan"]` and the next node reads it (S2 test passes).

### Step 5: `adb_tools.py`: three small changes
1. `press_home`:
```python
def press_home(device: str) -> bool:
    return _adb(f"adb -s {device} shell input keyevent KEYCODE_HOME") != "ERROR"
```
(public name, so `import *` in `deployment_old.py` picks it up, A2).
2. `take_screenshot(..., settle: float = 2.0)`; replace `sleep(2)` with `sleep(settle)`. Default keeps today's behavior.
3. In the `text` branch of `screen_action`: `if not re.fullmatch(SAFE_TEXT_RE, input_str): return json.dumps({**result, "status": "error", "message": "unsafe characters"})` (A4). Add `import re`.
**Done when:** typing `4:30` works and `4:30 & calc` returns an error without running anything.

### Step 6: `deployment_old.py`: small speed fixes
- `capture_and_parse_screen`: pass `"settle": SCREENSHOT_SETTLE_SEC` to `take_screenshot.invoke`.
- `_device_size(state)`: call `get_device_size.invoke(state["device"])` once, store in `state["device_size"]`, fall back to 1080x2400 on an error dict. Use it in `execute_element_action` and `react_step` (D7).
- `execute_element_action`: change `action_params["input_str"] = parameters.get("text", "")` to `parameters.get("text") or parameters.get("input_str", "")` (J3: raw recordings use `input_str`; today a raw text step would type an empty string and `screen_action` would return an error). Harmless for both paths.
- In the new executor do not add any `time.sleep` after an action; the next capture already waits `SCREENSHOT_SETTLE_SEC` (A1). Today the legacy loop waits 1.5 s + 2 s per step.
- `NvidiaBridge(max_tokens_text=..., max_tokens_json=..., max_tokens_vision=...)` accepts these kwargs (D10). Use `1024/1024/1024` when `USE_PLAN_REUSE`, otherwise keep `4096/4096/2048`. New outputs are IDs and short JSON.
- LLM counter: `LLM_CALLS = collections.Counter()`; add an optional `kind` argument to `_sync_call_text/json/vision` and increment `LLM_CALLS[kind]`. Count the direct `asyncio.run(bridge.call_json(...))` calls too. Reset in `run_task`.
**Done when:** a run prints e.g. `{"plan": 1, "react": 4, "judge_text": 1}`.

---

# PART 2: Completion fix (independent; do first)

### Step 7: Judge sees the screen after the last action; only the judge sets `completed`
- New `check_task_completion`: capture first (`final_screenshot`, `final_elements`). Keep the old function as `check_task_completion_legacy`.
- Executor ends with `execution_status="steps_done"`, never `completed=True`. Remove the early return in D4 and the `no_match → completed=True` branch (not used in the new flow).
- `state["completed"]` = judged complete; `state["finished"]` ends the graph.
**Done when:** a run whose last tap silently failed ends `failed`.

### Step 8: Stuck detection in code
`screen_hash(path)`: PIL (already used in `run_task`), crop the top ~6%, resize to 64×64 grayscale, return bytes/hash. Append after every capture. `STUCK_WINDOW` equal hashes → `execution_status="stuck"`, stop the segment. Stuck is never completed.
**Done when:** three identical captures give `stuck`; two screens differing only by the clock count as identical.

### Step 9: Two-tier JSON judge, with the text check as a router
1. `expected_texts` = typed values in replayed `text` steps (after overrides). May be empty.
2. `texts_present(expected_texts, elements)` normalizes by digits-only for numbers (`4.30`, `4:30`, `04:30`, `430` → `430`) and lowercase for words; True when every expected value appears in some on-screen text. If OmniParser returned fewer than ~5 text elements, return `None` (unknown).
3. If present or unknown or no expected values → **text judge**: `_sync_call_json(kind="judge_text")` with task, expected values, up to 60 on-screen `ID|content` lines (skip empty), last 3 actions as one-liners. No image.
4. If a value is **absent** → skip the text judge and go straight to the **vision judge** (one image). Reason: formatting differences (24-hour clocks, "430" shown as "4:30") make "absent" a hint, not proof.
5. Text judge `confidence < JUDGE_MIN_CONF` → vision judge.
6. Schema: `{"complete": bool, "confidence": 0-1, "evidence": "...", "missing": "..."}`. Invalid JSON or missing keys → `complete=False`, logged.
7. Delete `_is_affirmative`, `_CRITERIA_SYSTEM`, and the "identical → yes" sentence.
8. `complete and confidence >= JUDGE_MIN_CONF` → `completed=True`, `finished=True`. Otherwise, if `retries < 1`: set a single react segment `goal = f"{task}. Still missing: {missing}"`, cap `MAX_RETRY_REACT_STEPS`, `retries += 1`, route to the executor. Else `execution_status="failed"`, `finished=True`.
**Done when:** a screen showing an alarm at "10:00" for a "4.30" task is judged not complete.

---

# PART 3: Element matching: fewer LLM calls

### Step 10: Changes to `match_element_via_pinecone` / `llm_bbox_fallback`
1. **Raw text and type from Neo4j.** `neo4j_element = db.get_element_by_id(...)` already returns `other_info` parsed (G6). Set `raw_content = other_info.get("content","")` and `raw_type = other_info.get("type","")`; fall back to the Pinecone metadata `content`/`type` (J8) if Neo4j has none. Today `stored_type` is read from `neo4j_element["element_type"]`, which `json2db` never sets (J2), so it always comes from Pinecone; the new `raw_type` fixes that. Keep using the Neo4j `description` for the LLM verify prompt as today. For interacted elements it is LLM prose (C1), so never compare it character-wise with live OCR text.
2. **Guard (fixes D6):** if there is no Neo4j element, no Pinecone vector, no `stored_bbox`, and no `params["element"]`, log and `return []` before any LLM call.
3. **No-LLM accept:** spatial candidate found and `raw_content` non-empty and same type and `difflib.SequenceMatcher(None, norm(raw_content), norm(candidate_content)).ratio() >= TEXT_MATCH_MIN` → accept with `match_score=ratio`.
4. New parameter `strict: bool = False` (entry check, Step 15a). Strict mode: skip the `min_dist < 0.03` shortcut, never call `llm_bbox_fallback`; accept only through item 3, or through the existing LLM verify when `raw_content` is empty (icon-only element). The shortcut itself is meaningful (bboxes are normalised, 0.03 = 3% of the screen) but too lenient for deciding "am I on the right screen".
5. `llm_bbox_fallback(..., raw_content="")`: rank live elements by text ratio against `raw_content` (or token overlap with the description if empty), small bonus for equal type; send only the top 15 with their original indices so `screen_element_id` keeps its meaning.
6. Do not call this function for `back`, `swipe*` or `text` steps (Step 15): their bound element is not meaningful (J4).
**Done when:** replaying a known tap on the matching screen makes 0 `verify` LLM calls, and a fake `element_id` returns `[]` with 0 LLM calls.

---

# PART 4: ReAct refactor (small, bounded, self-terminating)

### Step 11: `react_step(state, goal) -> "done" | "acted" | "error"`
New function; the old `fallback_to_react` stays for the legacy path.
- Uses `goal`, not `state["task"]`.
- No swipe-up injection (D5).
- Prompt: goal, last ≤5 actions as one-liners (`tap id=12 'Save'`), element list as `ID|type|content` lines (skip empty content, max ~60), no `indent=2`.
- System prompt adds: "If the goal is already satisfied on the current screen, reply `{\"action\":\"done\"}`."
- Before `screen_action`: for `action == "text"`, reject `input_str` that does not match `SAFE_TEXT_RE` (A4); count it as an error.
- Reuse the coordinate logic from `fallback_to_react`; use `_device_size(state)`.
- Image on the first step of a segment and after a failed/no-change action; text-only otherwise. **[Guessing]** that text-only is enough; keep `REACT_ALWAYS_IMAGE` (env) to turn it back on.

### Step 12: `run_react_segment(state, goal, cap)`
```
for _ in range(cap):
    capture_and_parse_screen(state); hash → stuck → return "stuck"
    r = react_step(state, goal)
    if r == "done": return "done"
    if r == "error": return "error"
return "cap"
```
No judge calls inside the loop; no sleeps (capture waits).
**Done when:** N ReAct actions cost N+1 LLM calls.

---

# PART 5: Planner and executor

A plan is an ordered list of segments: `replay` (stored steps `from..to`, 1-based inclusive) or `react` (a short goal).

```json
{"action_id": "high_level_action_ab12cd34", "confidence": 0.9,
 "segments": [{"type":"replay","from":1,"to":2},
              {"type":"react","goal":"set the alarm time to 4:30"},
              {"type":"replay","from":6,"to":6}],
 "text_overrides": {"3": "4:30"}}
```

| Situation | Plan |
|---|---|
| Same task, new value, value was a typed `text` step | one replay + `text_overrides`, no ReAct |
| Same task, new value, value was set by taps | replay prefix → react gap → replay suffix |
| Same app, different goal | replay shared opening steps → react |
| Unrelated | `action_id: null`, one react segment with the whole task |

### Step 13: Catalog for the planner (compact, built in code, cached)
`build_action_catalog(task) -> (catalog_text, actions_by_id)`, cached for `CATALOG_TTL_SEC`:
1. `actions = db.get_all_high_level_actions()` (`element_sequence` is already a list, G4). `template_pattern` is a JSON string; use `parameter_fields` only as an optional hint when it parses to a list/dict of names, never to validate.
2. **Step labels, one Cypher query for all step elements** (no Pinecone): `MATCH (e:Element) WHERE e.element_id IN $ids RETURN e.element_id AS id, e.other_info AS oi, e.description AS d` via `db.driver.session(database=db.database)`; `json.loads(oi)` → `content`. Label by `atomic_action`:
   - `tap` / `long_press` → the element `content` (fallback: first 50 chars of `d`);
   - `text` → `text "<stored text>"`;
   - `back` → `back`; `swipe*` → `swipe <direction>`.
   Never label `back`/swipe steps by element content: their bound element is the one nearest the screen centre (J4), which would mislead the planner.
3. **Recorded task text:** `action["source_task"]` (written by Step 3). For Actions that lack it, one Cypher query for the first step's element: `MATCH (p:Page)-[:HAS_ELEMENT]->(e:Element) WHERE e.element_id IN $first_ids RETURN e.element_id AS eid, p.other_info AS info`; `json.loads` `info`, read `task_info.description` (only the step-0 Page has it, J1). Fall back to the Action `name`.
4. If `len(actions) > CATALOG_PREFILTER_OVER`: keep the top `CATALOG_TOP_K` by lowercase token overlap between the task and `source_task + name + step labels`. **[Guessing]** that it rarely drops the right action; log the kept IDs.
5. One entry per action; step numbers are 1-based list positions (not the `order` field):
   ```
   [id=ab12cd34] recorded as: "Set alarm at 10" | name: <name>
     1 tap "Clock" · 2 tap "+" · 3 text "10:00" · 4 tap "Save" · 5 back
   ```

### Step 14: `plan_task(state)`
1. Exact repeat: if the normalized task equals a normalized recorded task of one action → plan = `replay 1..N`, no overrides, **0 LLM calls**.
2. Otherwise one `_sync_call_json(kind="plan")` call. System prompt:
   > Plan how to do the TASK using stored actions. Reply JSON only: `action_id`, `confidence`, `segments`, `text_overrides`. A `replay` segment uses step numbers `from`..`to` (1-based, inclusive, increasing, no overlap) of that one action, only for steps that serve the task unchanged. Use `text_overrides` ({step number: new text}) only for steps shown as `text`. Anything the stored steps do not cover becomes a `react` segment with a short goal. Same app but a different goal: replay only the shared opening steps, then react. Nothing related: `action_id` null and one react segment with the whole task. Never copy step contents.
3. `validate_plan` in code:
   - `action_id` exists; every `from/to` is an int inside the sequence; `from <= to`; segments increasing, no overlap; at least one segment; every react goal non-empty;
   - `text_overrides` keys point to steps whose `atomic_action == "text"`; **every override value matches `SAFE_TEXT_RE`** (A4);
   - salvage: drop invalid segments if a valid prefix remains, else single react plan.
4. `confidence < PLAN_MIN_CONF` or `force_fallback` → `[{"type":"react","goal": task}]`.
5. Store in `state["plan"]`, `seg_index = 0`. Skip if `state["plan"]` already exists.
**Done when:** a dry run (no device) of the example tasks in section 7 gives the shown plans; a broken LLM reply degrades to one react segment.

### Step 15: `execute_plan_node`
For each segment from `state["seg_index"]`:

**Replay segment** (per step `i` in `from-1..to-1`):
```
step   = seq[i]; atomic = step["atomic_action"]; params = step["action_params"]  (dict; json.loads if str)
text   = override.get(str(i+1)) or params.get("text") or params.get("input_str")   # J3: both key names
eid    = step.get("element_id") or ""

if atomic == "back":                                       # bound element is arbitrary (J4)
    ok = screen_action(action="back")
elif atomic == "text":                                     # A3: x/y ignored; needs no matching
    ok = screen_action(action="text", input_str=text)      # SAFE_TEXT_RE check first; empty text → "match_failed"
elif atomic == "swipe_precise" or (atomic.startswith("swipe") and params.get("start") and params.get("end")):
    ok = screen_action(action="swipe_precise", start=tuple(params["start"]), end=tuple(params["end"]), duration=400)
elif atomic.startswith("swipe"):                           # direction only
    x, y = centre of _device_size(state)                   # [Guessing]; J4 shows the recording used the centre element
    ok = screen_action(action=atomic, x=x, y=y, direction=params["direction"])
else:                                                      # tap, long_press: need the live element
    if not eid: return "match_failed"
    capture if the screen changed since the last capture
    matches = match_element_via_pinecone(eid, step, state)   # [] → return "match_failed"
    ok = execute_element_action(state, matches[0])
not ok → return "adb_failed"
append history; state["replayed_steps"] += 1; mark the screen as changed
```
Recorded swipe coordinates are absolute pixels of the recording device **[UNVERIFIED]** that it equals the deployment device; if sizes differ, scale by `live_size / recording_size` (the recording size can be read from the step screenshot with PIL). `text` steps use the override when present, otherwise the stored value. **No `time.sleep`**; the next capture waits (A1). Capture only before steps that need element matching, and once more for the judge.

**React segment:** `run_react_segment(state, goal, MAX_REACT_STEPS_SEG)`.

**On any non-ok/non-done result:** degrade once to a single react segment with `goal = task` (log the failed step). If already degraded, stop with `execution_status` = the failure reason. A failed gap in the middle never continues into the next replay (wrong screen).

After the last segment: `execution_status="steps_done"`, then the judge (Steps 7–9).

### Step 15a: Entry check (no HOME anchor)
Before the first replay step that is a `tap` or `long_press`:
1. Reuse the capture needed for that step anyway. Call `match_element_via_pinecone(..., strict=True)` (Step 10). Match → replay starts here (covers "already in the right place" and "recording started from a different screen").
2. No match → `press_home(device)` (Step 5), wait for the next capture, try strict once more.
3. Still no match → degrade to the single react plan (ReAct then does swipe → search bar → app name itself).
HOME is pressed only in 2, only when the first element cannot be found. A react-first plan never presses HOME. If the first replay step is `text`, `back` or a swipe, there is nothing to verify; replay directly.
**Done when:** replay starts correctly from (a) HOME, (b) inside another app after one HOME press, (c) an unrelated screen (falls to ReAct).

---

# PART 6: Graph and `run_task`

### Step 16: New workflow (flag on)
```
capture_screen ─(capture failed → end, status "error")─▶ plan_task ─▶ execute_plan ─▶ check_completion ─┬─ end (finished)
                                                                         ▲                               │
                                                                         └─────── retry (≤1) ────────────┘
```
- Router `after_judge` reads `state["finished"]`. Routers cannot persist state (S3); any counter change happens inside a node.
- Keep `fallback_node`, `should_fallback`, `match_elements_node`, `is_task_completed`, `execute_action_node` untouched for the legacy graph.

### Step 17: `run_task`
- Fix the header (it says "deployment.py v3").
- Add `return {"status": "error", "message": str(e), ...}` in the `except` branch (D8).
- Return `llm_calls`, `replayed_steps`, `react_steps`, `plan`.
- `Image.open(...).show()` block: test `"completed"` instead of `"success"`.
- `no_match` and `close_actions` exist only in the legacy path.
- Statuses: `completed | failed | stuck | match_failed | adb_failed | error`.

---

# PART 7: Validation

### Step 18: `eval_plan_reuse.py`
Run each task with `USE_PLAN_REUSE=0` and `1` against the same Actions. Log per run: LLM calls by kind, replayed steps, ReAct steps, wall time, your manual verdict vs the judge's.

| Stored | New task | Expected plan |
|---|---|---|
| "Set alarm at 10" | "Set alarm at 4.30" | replay + override, or a react gap if set by taps |
| "Set alarm at 10" | "Open the world clock" | replay opening step(s) → react |
| "Set alarm at 10" | "Turn on Wi-Fi" | react only |
| "Set alarm at 10" | "Set alarm at 10" | exact repeat: 0 plan calls |
| "Set alarm at 10" | "Set alarm at 4.30" starting inside another app | strict entry check fails, HOME once, then row 1 |

**Done when:** with the flag on, row 1 (typed value) makes about 2 LLM calls (plan + judge) and no run ends `completed` while the screen shows the wrong value.

---

# PART 8: Optional, later
- Skip the app launch when the app is already foreground: needs the package name stored per action at recording time (recorder not uploaded). If you read it from the device, do not parse `dumpsys` with a host-side `| grep`; `_adb` runs through the host shell and Windows has no `grep`. Fetch the output and filter in Python.
- Write back completed ReAct segments through the existing recording pipeline so the next similar task replays them. Check what that pipeline expects first.

---

## 7. Example scenarios (illustrative; step contents invented)

Stored action **S1**, recorded as "Set alarm at 10": `1 tap "Clock" · 2 tap "+" · 3 text "10:00" · 4 tap "Save"`.

**A. "Set alarm at 4.30" (value was typed).** Plan: replay 1–4, `text_overrides {"3":"4:30"}`. Capture → strict entry check matches "Clock" on the launcher → tap (match) → tap (match) → type `4:30` (no capture, no matching) → capture → tap Save → capture. Judge: "4:30" present → text judge → `completed`. LLM calls: plan 1 + judge 1 = **2**, plus verify calls only if a match is ambiguous.

**B. Same, but the time was set by tapping picker digits.** Plan: replay 1–2 → react "set the alarm time to 4:30" → replay 6. ReAct ≈ 3 actions + `done`; judge 1; plan 1 ≈ **6** **[Guessing]**.

**C. "Open the world clock".** Plan: replay 1 → react "open the World Clock tab" (~1 action + `done`). Judge text-only (no expected values). ≈ **4** calls; today this is `no_match`.

**D. Phone starts inside Settings.** Strict check does not find "Clock" → `press_home` → found → continue as A.

**E. "Turn on Wi-Fi".** `action_id: null` → one react segment (cap 6) → one text judge. Same ReAct behavior as before, without per-action judging or the swipe-up injection.

**F. Wrong result.** Replay ended, screen shows only "10:00". "4:30" absent → vision judge → not complete → retry: react "set alarm at 4.30. Still missing: 4:30 alarm" (≤4 actions) → judge again → second failure → `failed`.

**G. A recording with a `back` step and a swipe.** At recording time both were bound to the element nearest the screen centre (J4), which may not exist on the live screen. Old code tries to match that element (LLM calls, possible abort at step failure). New code runs `screen_action(back)` and replays the swipe from its recorded start/end, with no matching.

**H. A recorded text step.** `Element.parameters` holds `{"input_str": "10:00"}` (J3). Step 3 stores it as `{"text": "10:00"}`; old code with raw params would type an empty string and fail.

---

## Implementation order
1. Step 1 (DB name), Step 2 (audit). Decide whether stored data is trustworthy.
2. Step 3 (`chain_evolve` rebuild from Element nodes) and the migration. Do it even if the audit looks clean, because new recordings need the code-generated id and `source_task`.
3. Steps 4–6 (state, adb_tools, small speed fixes, counter).
4. Steps 7–9 (completion). Verify on the legacy replay path before touching planning.
5. Step 10 (matching).
6. Steps 11–12 (ReAct); test ReAct alone on one task.
7. Steps 13–15a (planner, executor); dry-run the planner first.
8. Steps 16–17 (graph, `run_task`).
9. Step 18 (eval); then decide on Part 8.

---

## 8. Verification status

**Resolved by reading `graph_db.py` and `data_storage.py` [Certain unless noted]**
1. Typed text is recorded as `Element.parameters = {"input_str": ...}`; it is not in LEADS_TO and the generator LLM never sees it.
2. `create_action` merges on `action_id`; collisions overwrite.
3. OmniParser bboxes are normalised 0–1.
4. Empty `element_id` cannot come from `get_chain_from_start`; `back`/swipe steps are bound to an arbitrary centre element.
5. `Neo4jDatabase` default database is `"graphdb"` and it is auto-created.
6. Raw element text and type are in `Element.other_info`; `element_type` is never set.

**Still open**
- Value of `config.Neo4j_DB` (`config.py` not uploaded). Step 1 removes the dependency either way.
- Whether recording step numbers start at 0. `source_task` (and the existing `extract_task_description`) rely on `task_info` being written at `step_no == 0`. The audit reports how many Actions have it.
- Whether the recording device and the deployment device have the same resolution (raw swipe/`swipe_precise` coordinates are absolute pixels).
- Whether `record_action_to_state` receives raw or labeled screenshots (`img_tool.py`/UI code not uploaded).
- `nvidia_llm_bridge.py`, `vector_db.py`, `llm_rate_limit.py` internals.
- **[Guessing]** the LLM call counts in the examples; text-only ReAct accuracy; the screen-centre swipe fallback (now used only when a swipe has no recorded start/end).
