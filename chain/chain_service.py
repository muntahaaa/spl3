"""
chain_service.py
================
Async background runners for chain_understand and chain_evolve.

These coroutines are meant to be launched from ui.py's Tab ④ via
``asyncio.create_task`` (or Gradio's background queue), *not* called
with ``asyncio.run`` from inside an already-running event loop.

Each runner follows the same contract:
    1. Mark the job as "running".
    2. Await the chain function.
    3. Mark the job as "done" with a result payload, or "error" with the
       exception message.

Import fix
----------
The original file imported ``from chain.task_store import update_job``.
``task_store.py`` lives at the project root (same level as chain_service.py),
not inside a ``chain/`` sub-package, so the correct import is simply
``from task_store import update_job``.
"""

from chain.task_store import create_job, update_job, get_job
from chain_understand import process_and_update_chain
from chain_evolve import evolve_chain_to_action


async def run_understand(job_id: str, start_page_id: str) -> None:
    """
    Background coroutine: run chain_understand for *start_page_id*.

    Stores triplet count in the job result on success, or the exception
    message on failure.

    Args:
        job_id:        UUID previously created with task_store.create_job().
        start_page_id: Neo4j page-node ID that anchors the chain.
    """
    update_job(job_id, "running")
    try:
        triplets = await process_and_update_chain(start_page_id)
        update_job(
            job_id,
            "done",
            result={
                "triplets_processed": len(triplets),
                "start_page_id": start_page_id,
            },
        )
    except Exception as exc:
        update_job(job_id, "error", error=str(exc))


async def run_evolve(job_id: str, start_page_id: str) -> None:
    """
    Background coroutine: run chain_evolve for *start_page_id*.

    Stores the resulting action_id (or None when the chain is
    non-templateable) in the job result on success.

    Args:
        job_id:        UUID previously created with task_store.create_job().
        start_page_id: Neo4j page-node ID that anchors the chain.
    """
    update_job(job_id, "running")
    try:
        action_id = await evolve_chain_to_action(start_page_id)
        update_job(
            job_id,
            "done",
            result={
                "action_id": action_id,        # may be None — caller must check
                "start_page_id": start_page_id,
            },
        )
    except Exception as exc:
        update_job(job_id, "error", error=str(exc))