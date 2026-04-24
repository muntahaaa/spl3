import asyncio
from chain.task_store import update_job
from chain_understand import process_and_update_chain
from chain_evolve import evolve_chain_to_action

async def run_understand(job_id: str, start_page_id: str):
    update_job(job_id, "running")
    try:
        triplets = await process_and_update_chain(start_page_id)
        update_job(job_id, "done", result={
            "triplets_processed": len(triplets),
            "start_page_id": start_page_id
        })
    except Exception as e:
        update_job(job_id, "error", error=str(e))

async def run_evolve(job_id: str, start_page_id: str):
    update_job(job_id, "running")
    try:
        action_id = await evolve_chain_to_action(start_page_id)
        update_job(job_id, "done", result={
            "action_id": action_id,
            "start_page_id": start_page_id
        })
    except Exception as e:
        update_job(job_id, "error", error=str(e))