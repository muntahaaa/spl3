import uuid
from typing import Dict, Any

_jobs: Dict[str, Dict[str, Any]] = {}

def create_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "result": None, "error": None}
    return job_id

def update_job(job_id: str, status: str, result=None, error=None):
    if job_id in _jobs:
        _jobs[job_id].update({"status": status, "result": result, "error": error})

def get_job(job_id: str) -> dict:
    return _jobs.get(job_id, {"status": "not_found"})