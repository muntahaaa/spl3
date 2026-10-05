# Deployment replay implementation and verification

## Implemented behavior

The default `run_task` path (`USE_PLAN_REUSE=1`, the existing default) now calls
`ReplayEngine` through `_run_verified_task` in `deployment.py`. The old workflow
remains available when explicitly selecting `USE_PLAN_REUSE=0`.

| Requirement | Implementation | Automated evidence |
|---|---|---|
| 1. Stored task and matching start | Exact task lookup, deterministic screen/target matching, replay without intermediate screen comparisons, one final recorded-screen postcondition after the last action. Confirmed replay makes no reasoning calls. | `test_case1_exact_replay_zero_model_calls`; actual Gallery recording; real adapter contract test |
| 2. Start on a later stored screen | Scan eligible source screens using cached parsed-JSON signatures. Require a strong score and separation from competing matches. Resume at the action leaving that screen. If unmatched/ambiguous, press Home once and verify step 1 before restarting; otherwise use ReAct. | Picture-tab resume, unmatched-screen Home recovery, ambiguity rejection, safe form resume |
| 3. Related task | One text-only semantic plan selects an action and common prefix. Code limits related-task replay to recognized navigation/editor-opening steps. Changed numeric, AM/PM, quoted, and recorded text parameters cannot silently reuse the full sequence. Rest uses ReAct. | Alarm to World Clock; 10:30 PM to 8 AM, including reused picker opening and different time selection |
| 4. No related task | Empty catalog skips the planner. No/invalid match uses full-task ReAct automatically. Fresh JSON is preferred; missing/insufficient parsing escalates to vision. Explicit step budget, stuck detection, and completion evidence prevent unbounded loops or assumed success. | Empty/unrelated catalogs; malformed plan; 12-action fallback; exhausted budget; vision escalation |
| 5. Form filling | Current explicit label:value inputs take priority. Exact recorded tasks replay stored input. During recovery, sufficient stored values are reused. Otherwise one structured text request resolves current-task values, relevant stored defaults, and missing synthetic values together. Validate formats, focus/replace fields, and verify entered values. Known fields fill deterministically without one reasoning call per field. Save completion is separately verified. | Recorded contact; explicit contact fields; generated contact; generated note; invalid email; stored-value recovery |

## Matching and recovery details

- Screens use normalized content, types, optional selected state, weighted overlap,
  and secondary layout evidence. The top 6% normalized status/debug strip is ignored.
- Task-significant numeric content is retained. `10:30 AM` and `10:30 PM` differ.
- Target matching requires content/type agreement. Position alone never approves a tap.
- Duplicate/ambiguous matches fall back rather than guessing.
- Earlier navigation steps may be skipped. Earlier text steps may only be skipped
  when their exact values are visible on the matched form screen. Earlier commits
  cannot be inferred from screen similarity.
- Full stored sequences align against source screens. An unmatched start, including a final screen that matches no source, uses Home recovery before replay.
- Every action invalidates the live observation, even if ADB reports failure.
  Capture failures cannot reuse stale JSON. A fresh image is retained if parsing
  fails so ReAct can use vision.
- A save with unexpected output is checked before attempting further actions,
  reducing the risk of duplicate creation.
- The final saved screen must be distinguishable from the last source screen for
  deterministic completion. Otherwise a model judge is required.

## Persistence and compatibility

`chain_evolve.py` now enriches recorded steps with source/destination JSON snapshots,
target descriptors, and action roles. Newly evolved actions carry replay schema
version 1. `data_storage.py` retains the app name on stored pages and resolves text
field coordinates from recorded actions. Human exploration preserves those coordinates.

Existing actions are hydrated in memory through one batch Neo4j query using
Page/HAS_ELEMENT/Element/LEADS_TO relationships. No destructive migration is performed.
Ambiguous or missing graph evidence is not guessed. Enriched actions skip the
metadata query. Tasks are ranked per run, so a prior task's shortlist cannot leak.

## Model-call and parsing budget

- Exact matched replay, including successful resume: 0 text/vision reasoning calls when the final JSON is sufficiently distinctive. An ambiguous final state needs a completion judge.
- Semantically equivalent tasks need a text planning call, then the same replay policy.
- Stored replay checks live target identity and ADB results but does not compare intermediate screens to destination/final screens. It compares the final screen once after the last stored action.
- Full and partial ReAct check completion after each successful action: relevant stored final JSON first, otherwise a text-first completion judge. Related tasks never use an unrelated stored final as their goal.
- Related task: normally 1 text planning call; 0 reasoning calls for the replay prefix.
- Empty catalog: 0 planning calls; ReAct handles the task.
- Forms: no generation when explicit/recorded data suffices; one batched resolution
  call for unresolved visible fields. Newly revealed fields can require another call.
- Vision is reserved for insufficient parsed observations, text reasoning failures,
  or low-confidence/error completion judgment.
- Observations are reused until an action changes the screen. Cached screen
  signatures avoid rebuilding identical comparison data.
- Each run returns `metrics`, `llm_calls`, `resumed_step`, `plan`, `history`, and
  `completion_evidence`. Parser attempts are counted separately from reasoning calls.
  OmniParser itself remains the existing external inference service.

## Verification performed

Command:

```powershell
python -B -m unittest test_replay_engine test_replay_integration test_deployment_logging test_replay_policy_timing test_deployment_artifacts -q
```

Result: **87 tests passed**, including two start positions using the existing
Gallery recording's parsed JSON. These tests perform no real device actions or
model/API requests. Model responses are controlled fixtures; the tests verify
execution decisions and call budgets, not the accuracy of the hosted model itself.

Integration-contract tests compile the actual adapter/recording/ADB functions from
the source with external I/O mocked. This verifies their wiring without importing
the application's eager service initialization. It is not a full application boot test.

Syntax checks passed for all changed implementation files. `git diff --check` passed.

## Remaining environment validation and limits

- `adb devices -l` returned no connected devices. Live Gallery/Clock/Contacts/Notes
  execution, actual ADB key-combination support, and accuracy under live OCR variation
  have NOT been verified.
- The available Python 3.14 environment lacks `langgraph`, `langchain_core`, `PIL`,
  and `pinecone`. The full application has NOT been started in this environment.
- Existing database contents and hosted model/parser responses were not queried.
- Similarity thresholds are conservative initial settings, not calibrated accuracy
  statistics. Dynamic numeric content outside the status strip can force fallback.
- Related-prefix validation recognizes common navigation and editor-opening controls.
  Unrecognized controls are handled by ReAct rather than blindly replayed. This can
  reuse fewer steps for unfamiliar apps.
- App identity is used when recorded metadata is available; historical records may
  lack it. Ambiguous page/target matches still cause conservative recovery.
- Form discovery supports editable metadata and common English contact/note labels.
  Other labels/layouts may need ReAct or schema extensions.
- ADB text input now safely handles printable ASCII punctuation and spaces through
  argument-list transport and Android-shell quoting. Unicode/newlines are explicitly
  rejected rather than silently corrupted; those require a compatible input method.
  Random generated form values are requested in the supported format.
- Old records with missing page JSON or field identity may require re-recording to
  obtain model-free replay. Fallback remains available.

For live acceptance, connect the target device and run the five scenarios through
normal deployment. Check visible results (including saved contacts/notes and the
alarm's AM/PM), and compare returned metrics to the budgets above. Start the Gallery
case from Home, Picture, and an unrelated screen separately.

## Waiting diagnostics and deadlines

The deployment callback prints stage start/end, decisions, executed instructions, model call kinds, and final evidence to the deployment UI. Read/inference waits emit elapsed-time messages every five seconds.

| Operation | Default deadline | Environment setting |
|---|---|---|
| NVIDIA request | 200 seconds (outer guard: 205) | `NVIDIA_REQUEST_TIMEOUT_SEC` |
| Neo4j read | 30 seconds | `DEPLOYMENT_DB_TIMEOUT_SEC` |
| OmniParser | 125 seconds | `DEPLOYMENT_PARSER_TIMEOUT_SEC` |
| ADB command | 30 seconds | `ADB_COMMAND_TIMEOUT_SEC` |

The NVIDIA bridge previously accepted but ignored request timeout arguments. It now forwards them and disables SDK automatic retries. A timed-out model backend is not repeatedly called during that run. Read/inference deadlines stop deployment waiting; underlying remote work may still finish, and late results are ignored. Device actions are never dispatched by the read/inference timeout worker. Parser worker failure now returns immediately. ADB text batches ordinary words/spaces into one input command, preserving literal percent sequences separately.

These changes and timeout forwarding are verified with controlled tests. The precise cause of the user's live stall still requires the new stage logs from a restarted deployment process; no hosted API or device run was performed for this update.

## App-opening prefix correction (2026-10-05)

Simple requests such as `Go to Gallery app`, `Open Gallery`, and `Launch Gallery` now use a local prefix proof before semantic planning. The recorded task must begin with opening the same app, its consecutive steps must be navigation, and a stored tap target must name that app with destination JSON available. Replay stops at that tap; the prefix destination is the requested final screen. It never executes the later Album action for a Gallery-only request. Conflicting destination evidence or compound goals still require reasoning. An already-matched prefix destination needs no actions.

The logged live JSON `20261005_125018_794.json` scores 0.914 against the recorded Gallery source, exceeding 0.90. An offline regression with these actual JSON fixtures completes the Gallery-only task with one stored action, one final check, no Home recovery, and zero model calls. This verifies routing against the supplied screen data, not a live device execution.

The model timeout defaults to 200 seconds, as requested by the user. It remains configurable with `NVIDIA_REQUEST_TIMEOUT_SEC`. Other services retain independent deadlines; five-second progress reporting continues. Automatic SDK retries remain disabled to prevent duplicate long waits. A planner transport failure is now reported before entering ReAct, since ReAct requires the same unavailable backend. A longer deadline allows slower responses but cannot repair a server/network outage.

## Completion judgement after JSON mismatch

A relevant stored final (or app-opening prefix destination) is checked using parsed JSON first. Matching evidence completes without a model call. A mismatch is inconclusive, so a fresh screenshot judge runs before further ReAct actions; the same policy applies after ReAct actions with a relevant stored goal. If visual judgement says incomplete, the next action planner uses vision on that same observation rather than relying solely on unconfirmed OCR. Intermediate stored replay steps still do not perform completion judgements.

Without a relevant stored final, completion uses a text judgement over parsed JSON first, escalating to vision for absent elements, low confidence, invalid responses, or an explicit `need_vision` result. Related tasks never compare against an unrelated task's final screen. The model deadline remains 200 seconds. Tests cover garbled final JSON, visual completion without extra actions, visual ReAct handoff, and text-to-vision judgement escalation. No live device/API execution was performed for this update.

## Judge request diagnostics and latency correction

The timeout log shows no returned judgement, not a model decision that the task failed. The local bridge mislabeled raw PNG bytes as JPEG; it now detects PNG and labels its data URI correctly. Completion prompts omit parser coordinates, IDs, and other irrelevant metadata while retaining all meaningful labels, values, selection state, form values, and recent action results. GLM-5.3-Flash requests now explicitly use low reasoning effort (configurable via NVIDIA_REASONING_EFFORT), since NVIDIA documents maximum effort as its default. Other models receive no GLM-specific reasoning parameter.

Every deployment model request now saves the exact system/user prompt and source image path/hash/MIME under log/model_requests and reports that path in the UI. Images/base64 and API credentials are not duplicated into the trace. The model timeout remains 200 seconds. Request construction tests verify PNG MIME, GLM reasoning settings, compatibility with other models, and compact judgement evidence. No provider request was performed, so provider latency and availability remain unverified.

## UI judge prompts and successful-run image cleanup

Judge calls emit the exact system prompt through the UI callback with [JUDGE-SYSTEM-PROMPT]. Each run tracks its raw screenshots and annotated parser images; after confirmed completion, only those tracked images inside the deployment screenshot/parser image output directories are deleted. Failed runs retain their images. Stored recordings, older runs, parsed JSON, and request prompt traces remain intact. Late parser images are also removed if that run has already completed successfully. Per-image cleanup errors are logged without changing the task's completion result.

## Replay source tolerance

The source-screen threshold is now 0.75, a project tuning choice rather than a universal similarity standard. Targeted actions require a unique content/type target match. Navigation can proceed by a unique exact target label even when dynamic layout/content yields a lower screen score, provided app identity is compatible and earlier prerequisites are safe. These checks apply both before and after Home recovery. Ambiguous candidate positions remain rejected. Completion/destination evidence retains its 0.90 threshold and visual judgement fallback.

## Task-first matching

Deployment now loads tasks and selects exact, conservative locally equivalent navigation, app-opening prefix, or general semantic planner results before capturing a screen. Exact matching checks both source_task and title/name. Local navigation equivalence recognizes go-to/open/launch wording and Album/Albums synonyms only when recorded navigation targets prove the complete same app/tab sequence. Parameterized or unsupported goals still use text semantic planning. Screenshots are taken afterwards for replay alignment or ReAct. Task matching itself no longer captures images to rank candidates. Tests verify catalog/planner ordering and the user's Gallery/Albums wording with zero model calls.

## Separate task selection from target validation

Equivalent explicit navigation wording now selects stored tasks based on canonical task intent alone, without requiring parser target descriptions to reproduce the task name. `Go to world clock` selects `Open world clock` even when its target description is generic. The planner is skipped; subsequent screen alignment still refuses missing or ambiguous recorded targets. Selection does not repair historical target metadata or guarantee successful replay.
