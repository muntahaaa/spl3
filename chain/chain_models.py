from pydantic import BaseModel, Field
from typing import Optional

class ChainRunIn(BaseModel):
    start_page_id: str = Field(..., description="Page ID that starts the chain in Neo4j")

class ChainUnderstandOut(BaseModel):
    status: str
    start_page_id: str
    triplets_processed: int
    message: str

class ChainEvolveOut(BaseModel):
    status: str
    start_page_id: str
    action_id: Optional[str]
    message: str

class ChainJobStatus(BaseModel):
    job_id: str
    status: str          # "pending" | "running" | "done" | "error"
    result: Optional[dict]
    error: Optional[str]