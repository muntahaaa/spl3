# Acceptance Testing Report

## Scope and testing components

This report covers the automated acceptance and regression tests for the project as executed on 8 October 2026. The suite contains 232 tests in 24 modules.

| Component | Acceptance coverage | Test modules |
|---|---|---|
| Stored-task replay | Exact replay, partial-task reuse, fast-forwarding, safe target selection, action budget and final verification | `test_replay_engine.py`, `test_replay_policy_timing.py`, `test_replay_integration.py` |
| Input and form adaptation | Recorded values, changed parameters, dependent result selection, one bounded model extraction and input verification | `test_replay_text_binding.py`, `test_recovery_validation.py` |
| UI-change recovery | Changed labels/positions, visual target recovery, missing tabs, Back/Home/launcher search sequence and assistance | `test_deployment_visual_recovery.py`, `test_tab_recovery.py`, `test_deployment_fixes.py` |
| Completion judgement | Final-screen threshold, layout comparison, dynamic values, world-clock and stopwatch postconditions | `test_final_similarity_threshold.py`, `test_page_layout_completion.py`, `test_world_clock_completion.py`, `test_stopwatch_completion.py` |
| Android observation | UI hierarchy parsing, clickable/control identity, hierarchy failure fallback and artifact handling | `test_deployment_hierarchy.py`, `test_deployment_artifacts.py` |
| Deployment UI and reporting | Streaming logs, multi-case selection, sequential status updates, combined reports and assistance controls | `test_deployment_logging.py`, `test_deployment_report.py`, `test_deployment_ui_blocks.py`, `test_stored_case_selection.py`, `test_deployment_control.py`, `test_deployment_strategy.py` |
| Chain processing | Understanding/evolution model configuration, JSON recovery, persistence safety and task metadata | `test_chain_consistency.py`, `test_chain_fastpath.py`, `test_evolve_task_metadata.py` |
| Feature extraction | In-memory upload, batching, cache behavior, invalid input and per-crop recovery | `test_feature_optimizations.py` |

## Testing approach and strategy

The project uses acceptance testing backed by deterministic unit, integration, contract and regression tests.

1. **User-goal acceptance tests** model tasks such as opening Gallery, setting an alarm, using World Clock, completing forms and resetting Stopwatch. A test passes only when the externally visible task outcome and required action history agree.
2. **Stored-replay tests** verify the preferred deployment path: match the stored task before capture, reuse safe stored steps, adapt explicit input values and avoid unnecessary VLM calls.
3. **Recovery tests** introduce layout drift, renamed controls, screens ahead of or behind the recording, missing hierarchy data and model failures. Recovery must remain bounded and must not tap an unrelated element.
4. **Completion-safety tests** reject success based only on an ADB success response, a visible form, a search result or generic model evidence. Required operations and the final page must both be proven.
5. **Offline isolation** replaces ADB, Neo4j, OmniParser and model calls with controlled fixtures for most tests. This keeps the suite fast and reproducible.
6. **Integration tests** verify adapters, saved JSON hydration, shell-safe text entry and end-to-end replay orchestration without requiring every external service.
7. **Live-device acceptance** remains a separate final gate because Android accessibility timing, installed app versions and remote inference cannot be fully represented by offline fixtures.

## Item pass/fail criteria

| Acceptance item | Pass criteria | Fail criteria | Current result |
|---|---|---|---|
| Exact stored task | Replays stored actions with zero planning calls and verifies completion after the last step | Falls into full ReAct, skips required actions or judges every intermediate screen | Pass |
| Partial/similar task | Reuses only the proven common prefix and hands the remainder to bounded ReAct | Replays an incompatible suffix or loses the matched plan | Pass |
| Screen alignment | Finds the same or a later safe stored step using parsed UI structure and unique targets | Uses position alone, skips a required write/commit or accepts ambiguity | Pass, with one policy regression listed below |
| Changed control layout | Locates a uniquely valid control by label, role, resource ID or one bounded visual recovery | Taps an unrelated element, loops, or returns Home before bounded in-app recovery | Pass for the current Add-control regression |
| Input adaptation | Uses explicit task/guidance values, otherwise recorded values, with one bounded extraction call | Types stale stored input after a changed request or repeatedly proposes the same input | Pass |
| Completion judgement | Confirms required operations and task-specific final evidence; similarity over the configured threshold can pass | Generic evidence, initial final-looking screen or ADB success alone passes | Fail: completion-loop/final-settlement regressions |
| Multi-case UI | Runs only selected cases sequentially and atomically marks final rows before producing the report | Runs unselected cases or leaves the completed final row Pending | Pass |
| Case deletion | Removes the selected case and exclusively owned Neo4j records while protecting shared records | Deletes shared graph data or requires removed vector infrastructure | Pass |
| Chain processing | Produces validated structured results, bounds recovery and does not persist failed enrichment | Timeout/invalid output is silently treated as success | Pass |
| Full automated acceptance gate | Zero failures and zero errors; skips require an approved environmental reason | Any failure/error, or an unexplained skip | **Fail: 4 failures, 4 errors, 2 explained skips** |

## Detailed automated outcomes

Command executed:

```powershell
python -B -m unittest discover -s test_case -t . -p "test_*.py"
```

Measured result: **232 total; 222 passed; 4 failed; 4 errors; 2 skipped**. The full automated acceptance gate therefore currently fails.

| Test module | Total | Passed | Failed | Errors | Skipped | Outcome |
|---|---:|---:|---:|---:|---:|---|
| `test_chain_consistency.py` | 12 | 12 | 0 | 0 | 0 | Pass |
| `test_chain_fastpath.py` | 12 | 12 | 0 | 0 | 0 | Pass |
| `test_deployment_artifacts.py` | 6 | 6 | 0 | 0 | 0 | Pass |
| `test_deployment_control.py` | 3 | 3 | 0 | 0 | 0 | Pass |
| `test_deployment_fixes.py` | 20 | 14 | 2 | 4 | 0 | **Fail** |
| `test_deployment_hierarchy.py` | 6 | 6 | 0 | 0 | 0 | Pass |
| `test_deployment_logging.py` | 6 | 6 | 0 | 0 | 0 | Pass |
| `test_deployment_report.py` | 8 | 8 | 0 | 0 | 0 | Pass |
| `test_deployment_strategy.py` | 7 | 7 | 0 | 0 | 0 | Pass |
| `test_deployment_ui_blocks.py` | 2 | 2 | 0 | 0 | 0 | Pass |
| `test_deployment_visual_recovery.py` | 4 | 4 | 0 | 0 | 0 | Pass |
| `test_evolve_task_metadata.py` | 3 | 3 | 0 | 0 | 0 | Pass |
| `test_feature_optimizations.py` | 5 | 5 | 0 | 0 | 0 | Pass |
| `test_final_similarity_threshold.py` | 6 | 6 | 0 | 0 | 0 | Pass |
| `test_page_layout_completion.py` | 4 | 4 | 0 | 0 | 0 | Pass |
| `test_recovery_validation.py` | 16 | 16 | 0 | 0 | 0 | Pass |
| `test_replay_engine.py` | 37 | 37 | 0 | 0 | 0 | Pass |
| `test_replay_integration.py` | 6 | 5 | 0 | 0 | 1 | Pass with fixture skip |
| `test_replay_policy_timing.py` | 36 | 33 | 2 | 0 | 1 | **Fail** |
| `test_replay_text_binding.py` | 18 | 18 | 0 | 0 | 0 | Pass |
| `test_stopwatch_completion.py` | 4 | 4 | 0 | 0 | 0 | Pass |
| `test_stored_case_selection.py` | 3 | 3 | 0 | 0 | 0 | Pass |
| `test_tab_recovery.py` | 3 | 3 | 0 | 0 | 0 | Pass |
| `test_world_clock_completion.py` | 5 | 5 | 0 | 0 | 0 | Pass |

## Test module results in test-case format

| **Test Case Id** | T1 |
|---|---|
| **Title** | Chain Consistency |
| **Test Scenario** | Validate model configuration, visual recovery, reasoning persistence, JSON formatting and database-write failure handling across chain understanding and evolution. |
| **Expected Outcome** | All invalid or failed reasoning results are rejected, successful reasoning is preserved, and model-specific settings do not leak between runs. |
| **pass/fail** | **pass** — 12/12 passed. |

| **Test Case Id** | T2 |
|---|---|
| **Title** | Chain Fast Path |
| **Test Scenario** | Exercise low-reasoning model calls, description merging, triplet timeout recovery, stream parsing and page/element description generation. |
| **Expected Outcome** | Deterministic cases skip unnecessary model work; bounded recovery handles timeouts; generated descriptions preserve meaningful layout and content. |
| **pass/fail** | **pass** — 12/12 passed. |

| **Test Case Id** | T3 |
|---|---|
| **Title** | Deployment Artifact Management |
| **Test Scenario** | Track screenshots, parser JSON, labeled images, hierarchy XML and model prompt traces throughout deployment. |
| **Expected Outcome** | Successful runs delete only their generated artifacts, failed runs retain evidence, and files outside managed directories remain untouched. |
| **pass/fail** | **pass** — 6/6 passed. |

| **Test Case Id** | T4 |
|---|---|
| **Title** | Deployment Assistance Control |
| **Test Scenario** | Pause a deployment for user guidance, resume it with supplied information, skip it, close pending requests and enforce device locking. |
| **Expected Outcome** | Only a valid pending token resumes its worker; skip and close terminate safely; concurrent use of the same device is blocked. |
| **pass/fail** | **pass** — 3/3 passed. |

| **Test Case Id** | T5 |
|---|---|
| **Title** | Deployment Regression Fixes |
| **Test Scenario** | Validate partial-task boundaries, stored target repair, safe ReAct targets, completion loops, Back recovery and unchanged-final-page settlement. |
| **Expected Outcome** | Required operations cannot be skipped, incomplete evidence is judged once, unchanged final pages settle correctly, and recovery never loops or guesses unsafe targets. |
| **pass/fail** | **fail** — 14 passed, 2 failed and 4 errored. Completion-loop expectations failed and `ReplayEngine.settle_unchanged_screen` is missing. |

| **Test Case Id** | T6 |
|---|---|
| **Title** | Android UI Hierarchy |
| **Test Scenario** | Parse Android XML labels, bounds, inputs, focus, clickability and shared parent/child control identity, including ADB hierarchy failures. |
| **Expected Outcome** | Hierarchy elements are normalized correctly and an idle-state failure immediately falls back without unsafe or repeated XML operations. |
| **pass/fail** | **pass** — 6/6 passed. |

| **Test Case Id** | T7 |
|---|---|
| **Title** | Deployment Logging |
| **Test Scenario** | Stream replay actions, model requests, parser progress, wait heartbeats, failures and final results into the UI. |
| **Expected Outcome** | Logs remain ordered and visible, callbacks receive parser progress, and the final result does not erase earlier execution evidence. |
| **pass/fail** | **pass** — 6/6 passed. |

| **Test Case Id** | T8 |
|---|---|
| **Title** | Multi-Case Execution Report |
| **Test Scenario** | Run stored cases sequentially and classify completed, failed, skipped and uncertain outcomes while streaming a combined report. |
| **Expected Outcome** | Every case receives its final status and time; the last row updates atomically; totals and success evidence match the individual outcomes. |
| **pass/fail** | **pass** — 8/8 passed. |

| **Test Case Id** | T9 |
|---|---|
| **Title** | Deployment Strategy |
| **Test Scenario** | Validate launcher search, assistance, ReAct-to-replay rejoining, cleanup behavior and Neo4j case deletion with shared-data protection. |
| **Expected Outcome** | Recovery follows the configured order, resumes stored steps when safe, retains failure evidence and never deletes shared graph records. |
| **pass/fail** | **pass** — 7/7 passed. |

| **Test Case Id** | T10 |
|---|---|
| **Title** | Deployment UI Streaming Blocks |
| **Test Scenario** | Stream background deployment messages and table snapshots through Gradio components. |
| **Expected Outcome** | Root output replacement is correct, reports remain strings and previously emitted table snapshots are not mutated. |
| **pass/fail** | **pass** — 2/2 passed. |

| **Test Case Id** | T11 |
|---|---|
| **Title** | Visual Target Recovery |
| **Test Scenario** | Locate changed stored targets from fresh screenshots, handle failed visual lookup and protect pending required operations from premature completion. |
| **Expected Outcome** | A valid visual target continues without Home; missing targets request precise guidance; generic visual proof cannot complete a task. |
| **pass/fail** | **pass** — 4/4 passed. |

| **Test Case Id** | T12 |
|---|---|
| **Title** | Evolved Task Metadata |
| **Test Scenario** | Extract task metadata when the first recorded step is reordered, when target-page metadata is present and when metadata is absent. |
| **Expected Outcome** | Explicit metadata is recovered from supported locations and no task name is invented from unrelated page descriptions. |
| **pass/fail** | **pass** — 3/3 passed. |

| **Test Case Id** | T13 |
|---|---|
| **Title** | Feature Extraction Optimizations |
| **Test Scenario** | Exercise in-memory uploads, batched crops, cache ordering/eviction, failed-batch recovery and invalid inputs. |
| **Expected Outcome** | Batching and caching improve execution without changing output order, corrupting caller streams or suppressing valid per-item recovery. |
| **pass/fail** | **pass** — 5/5 passed. |

| **Test Case Id** | T14 |
|---|---|
| **Title** | Final Similarity Threshold |
| **Test Scenario** | Compare stored and live final pages above, below and exactly at the 0.70 deterministic threshold, including pending-action safeguards. |
| **Expected Outcome** | Scores above 0.70 can pass; scores at or below it require visual judgement; pending required actions always block success. |
| **pass/fail** | **pass** — 6/6 passed. |

| **Test Case Id** | T15 |
|---|---|
| **Title** | Page Layout Completion |
| **Test Scenario** | Compare Clock-family pages while times, AM/PM values, status-bar values, selected tabs and form input values change. |
| **Expected Outcome** | Dynamic numeric text does not destroy layout identity, selected-tab differences remain significant, and form values remain verifiable. |
| **pass/fail** | **pass** — 4/4 passed. |

| **Test Case Id** | T16 |
|---|---|
| **Title** | Recovery Target Validation |
| **Test Scenario** | Validate input, search, result, Add and tab targets; reject duplicate/unsafe controls; preserve matched plans through bounded assistance. |
| **Expected Outcome** | Only a unique control with the required role is executed, exact evidence outranks aliases, and failed recovery cannot produce a false success. |
| **pass/fail** | **pass** — 16/16 passed. |

| **Test Case Id** | T17 |
|---|---|
| **Title** | Replay Engine Acceptance Cases |
| **Test Scenario** | Cover exact stored replay, later-step alignment, shared-prefix reuse, full ReAct fallback, form filling, model escalation and completion evidence. |
| **Expected Outcome** | The five principal deployment cases choose the cheapest safe path, execute required operations and avoid unnecessary text or vision calls. |
| **pass/fail** | **pass** — 37/37 passed. |

| **Test Case Id** | T18 |
|---|---|
| **Title** | Replay Integration |
| **Test Scenario** | Verify the deployment adapter, stored JSON hydration, ADB text handling, Unicode rejection and replay of recorded Gallery data. |
| **Expected Outcome** | Adapter and text-entry contracts work end to end, and available recorded fixtures replay successfully. |
| **pass/fail** | **pass with environmental skip** — 5 passed; 1 skipped because the recorded Gallery JSON fixtures are absent. |

| **Test Case Id** | T19 |
|---|---|
| **Title** | Replay Policy and Timing |
| **Test Scenario** | Validate task matching before capture, partial-task boundaries, screen drift policy, final verification frequency, timeouts, model requests and ADB limits. |
| **Expected Outcome** | Replay ignores permitted intermediate drift, performs exactly one required final check, rejects missing targets and bounds all external waits. |
| **pass/fail** | **fail** — 33 passed, 2 failed and 1 skipped. A missing target was not blocked and one expected final-screen check was not performed; one JSON-fixture case was unavailable. |

| **Test Case Id** | T20 |
|---|---|
| **Title** | Replay Text and Parameter Binding |
| **Test Scenario** | Change stored task inputs, bind dependent search results, use guidance overrides, batch field extraction and recover from unresolved parameters. |
| **Expected Outcome** | Current explicit values replace stored values, dependent targets follow the new query, one bounded model extraction handles unresolved fields and the stored plan is retained. |
| **pass/fail** | **pass** — 18/18 passed. |

| **Test Case Id** | T21 |
|---|---|
| **Title** | Stopwatch Completion |
| **Test Scenario** | Distinguish a stopwatch that was started and reset from an untouched zero display, nonzero state or placeholder action. |
| **Expected Outcome** | Completion requires proven Start then Reset actions and a zero final state; an initial zero or placeholder proposal cannot pass. |
| **pass/fail** | **pass** — 4/4 passed. |

| **Test Case Id** | T22 |
|---|---|
| **Title** | Stored Case Selection |
| **Test Scenario** | Build stored-case UI choices and resolve an explicit multi-case selection containing reordered, duplicate or invalid IDs. |
| **Expected Outcome** | Only valid selected cases run, their selected order is preserved, duplicates are removed and an empty selection runs nothing. |
| **pass/fail** | **pass** — 3/3 passed. |

| **Test Case Id** | T23 |
|---|---|
| **Title** | Missing Tab Recovery |
| **Test Scenario** | Recover a missing in-app tab through Back, Home/app icon and the final launcher swipe/search/type/open sequence. |
| **Expected Outcome** | The least disruptive recovery succeeds first; launcher search is used only when the app is not visible after Home. |
| **pass/fail** | **pass** — 3/3 passed. |

| **Test Case Id** | T24 |
|---|---|
| **Title** | World Clock Completion |
| **Test Scenario** | Judge World Clock navigation with dynamic city/time content, common Clock tabs and selected-tab state. |
| **Expected Outcome** | Dynamic clock values are tolerated, but only World Clock-specific or selected-tab evidence confirms navigation; changed-city tasks still require their requested operation. |
| **pass/fail** | **pass** — 5/5 passed. |

### Failed and errored cases

| Test | Outcome and observed cause |
|---|---|
| `CompletionLoopTests.test_identical_incomplete_evidence_is_judged_once` | Failed: two model calls were made where one bounded judgement was expected. |
| `CompletionLoopTests.test_done_loop_stops_without_repeated_capture_or_judge` | Failed: execution ended with `error: pop from an empty deque` rather than `completion_unconfirmed`. |
| Four `StalledFinalPageTests` cases | Error: tests call `ReplayEngine.settle_unchanged_screen`, but that method is absent. This blocks deterministic and semantic settlement tests for unchanged final pages. |
| `ReplayPolicyTests.test_missing_replay_target_still_blocks_wrong_tap` | Failed: replay returned success when the test expected the missing target to block execution. This is a safety regression. |
| `ReplayPolicyTests.test_replay_ignores_intermediate_screen_drift_and_checks_final_once` | Failed: expected one final-screen check; observed zero. |

### Skipped cases

| Test | Reason | Contingency |
|---|---|---|
| `IntegrationTests.test_actual_recorded_gallery_json_replays_and_resumes` | Recorded Gallery JSON fixtures are absent | Restore the recorded JSON fixture set and rerun this case. |
| `ReplayPolicyTests.test_user_logged_live_home_matches_recorded_gallery_prefix` | User live/recorded JSON fixtures are unavailable | Export representative live-home and stored Gallery JSON files into the expected fixture location. |

## Risks and contingencies

| Risk | Impact | Contingency |
|---|---|---|
| Android layout/accessibility changes | Stored labels or hierarchy nodes may move, duplicate or disappear | Prefer resource ID, role and unique parsed labels; use one bounded visual recovery; request a precise visible label only after those checks fail. |
| UIAutomator idle-state failures | XML hierarchy is unavailable | Log the idle-state cause and immediately use OmniParser rather than retrying XML indefinitely. |
| VLM latency, timeout or malformed JSON | Recovery or judgement may stall | Use local JSON/hierarchy matching first, 200-second bounded requests, disabled SDK retries and one formatting recovery. |
| False-positive completion | A task may be marked passed without required actions | Require operation history plus task-specific final evidence; repair the currently failing completion-settlement tests before release. |
| Missing replay target | Wrong control may be tapped | Reject ambiguity and unrelated roles; treat the failing wrong-tap regression as release-blocking. |
| Dynamic numeric content | Clocks and timers can invalidate raw text comparison | Compare structural layout for page identity while verifying requested task values separately. |
| Missing fixtures | Integration behavior is not exercised | Keep skips visible in the report and restore versioned, sanitized fixtures before formal acceptance. |
| External Neo4j/ADB/model availability | Offline tests may pass while live execution fails | Run a connected-device smoke matrix after the automated suite: exact replay, changed layout, changed input, missing element and multi-case execution. |
| Test data cleanup | Successful tasks may leave screenshots/JSON or delete shared graph records | Track artifacts per run; delete them only on success; preserve shared Neo4j nodes during case deletion. |

## How to run tests

Run commands from the project root.

### Entire suite

```powershell
python -B -m unittest discover -s test_case -t . -p "test_*.py"
```

With the project Conda environment explicitly selected:

```powershell
& "C:\Users\HP\anaconda3\envs\spl3\python.exe" -B -m unittest discover -s test_case -t . -p "test_*.py"
```

### One test file

```powershell
python -B -m unittest test_case.test_replay_engine
```

### One test class

```powershell
python -B -m unittest test_case.test_replay_engine.ReplayTests
```

### One individual test case

```powershell
python -B -m unittest test_case.test_replay_engine.ReplayTests.test_case1_exact_replay_zero_model_calls
```

### Verbose output

```powershell
python -B -m unittest test_case.test_recovery_validation -v
```

The module path uses dots and omits `.py`. The class and method names are case-sensitive. A successful individual run ends with `OK`; any `FAIL`, `ERROR` or unexplained `skipped` result does not meet the acceptance gate.
