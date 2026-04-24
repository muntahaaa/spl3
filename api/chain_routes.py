import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks
from chain.chain_models import ChainRunIn, ChainUnderstandOut, ChainEvolveOut, ChainJobStatus
from chain.task_store import create_job, get_job
from chain.chain_service import run_understand, run_evolve

router = APIRouter(prefix="/chain", tags=["Chain"])

@router.post("/understand", summary="Run triplet reasoning on a stored chain")
def trigger_understand(body: ChainRunIn, background_tasks: BackgroundTasks):
    job_id = create_job()
    background_tasks.add_task(
        asyncio.run, run_understand(job_id, body.start_page_id)
    )
    return {"job_id": job_id, "status": "queued", "start_page_id": body.start_page_id}

@router.post("/evolve", summary="Evolve a chain into a high-level action node")
def trigger_evolve(body: ChainRunIn, background_tasks: BackgroundTasks):
    job_id = create_job()
    background_tasks.add_task(
        asyncio.run, run_evolve(job_id, body.start_page_id)
    )
    return {"job_id": job_id, "status": "queued", "start_page_id": body.start_page_id}

@router.get("/status/{job_id}", response_model=ChainJobStatus)
def get_chain_status(job_id: str):
    job = get_job(job_id)
    if job["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    return ChainJobStatus(job_id=job_id, **job)