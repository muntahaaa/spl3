# Chain understanding integration verification

Understanding now uses low reasoning effort, streaming, concise validated output, a 200-second HTTP timeout and a 205-second total request guard with five-second wait logs. At most two triplets are processed concurrently, and the existing four-requests-per-minute rate limit remains. A primary vision timeout gets one smaller-image recovery attempt. Text-only processing is explicitly marked when image evidence is unavailable; it never claims visual verification.

Task descriptions, recorded input parameters, and screenshot order are included in prompts. All seven description fields must be nonempty strings. Invalid/truncated responses are rejected. A run containing failed triplets does not persist enriched results or report successful understanding. Database write failures are surfaced, although already completed database writes are not rolled back.

Access-denial state is isolated with ContextVar per run. Corrupt images produce warnings and leave recorded text available. Identical/empty descriptions merge locally; failed optional model merges preserve both factual descriptions and expose a warning. Graph retrieval hydrates saved Element reasoning into top-level triplet reasoning for chain evolution. The verification script calls the current one-argument process_triplet API.

Offline regression command:

```powershell
python -B -m unittest test_chain_consistency test_chain_fastpath test_replay_engine test_replay_integration test_deployment_logging test_replay_policy_timing test_deployment_artifacts -q
```

Result: 108 passed, 2 recording-fixture tests skipped. Tests include visual timeout recovery, response validation, per-run isolation, failure job status, reasoning persistence/hydration, corrupt image handling, and failed database writes. Provider requests and rate-limit waits can still fail under a network or provider outage; these changes do not guarantee hosted availability.

## Live model probes

Read-only probes used the existing Home and World Clock screenshots from the Clock recording; no Neo4j writes occurred. The provider's /models endpoint returned HTTP 200 in approximately 0.1 seconds. The configured z-ai/glm-5.3-flash vision request timed out after approximately 200.6 seconds despite low reasoning effort and streaming. A listed google/gemma-3-12b-it route returned HTTP 404 (not available for the account). A google/gemma-4-31b-it vision request also timed out at approximately 200 seconds. No successful live understanding result was obtained, so hosted inference availability remains unresolved. Catalog reachability does not prove inference availability or isolate congestion from inference-path network issues.

## Current Llama configuration and successful verification

NVIDIA_MODEL, CHAIN_UNDERSTAND_MODEL and CHAIN_EVOLVE_MODEL select meta/llama-3.2-11b-vision-instruct in .env. Config defaults use the same model and each stage can override it. NVIDIA_BASE_URL is https://integrate.api.nvidia.com/v1; including /chat/completions in this SDK base URL caused observed HTTP 404 responses and has been corrected. The HTTP timeout remains 200 seconds.

The Llama endpoint rejects more than one input image. Understanding combines source and target into one labeled image. The shared bridge also packages other multi-image requests as numbered panels. Vision output instructions are included in the user message. Llama receives no unsupported GLM reasoning_effort option; concise prompts and focused sampling limit verbose output. JSON parsing accepts surrounding prose and identical repeated objects, but rejects malformed or conflicting objects. A vision response that lacks valid JSON gets at most one text-only formatting recovery using the existing visual observations, without resending screenshots. Seven nonempty understanding fields remain required. Evolution validates evaluation types and generated metadata before returning it for persistence.

With the corrected endpoint, a read-only live test on the recorded Clock screenshots returned validated visual understanding in 5.8 seconds, templateability evaluation in 1.6 seconds, and high-level metadata generation in 14.2 seconds. No formatting recovery was needed in that run. No Neo4j writes were made by these live probes; database persistence behavior was verified with mocks. These timings are observations, not latency guarantees. Restart the application to reload the environment and bridge instances.
