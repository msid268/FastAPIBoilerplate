from typing import Optional

import uuid
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.request_log import JobLog
from app.services.logging.logging_service import LoggingService
from app.core.decorators import log_action

router = APIRouter(prefix="/jobs", tags=["jobs"])


# --- DB dependency -----------------------------------------------------------

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Schemas -----------------------------------------------------------------

class JobCreate(BaseModel):
    """Payload for creating a new job."""
    prompt: str
    # add any other fields your job needs
    # e.g. user_id: int | None = None


class JobSubmitResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None


# --- Background worker entry point ------------------------------------------
# This is where the real work happens (could be Celery/RQ/etc. instead).

@log_action(action_type="job", action_name="process_job")
def process_job(job_id: str, request_id: Optional[str], payload: dict) -> None:
    """
    Example worker function.

    - Updates JobLog status to running/succeeded/failed.
    - Does the expensive work in the background.
    """
    from app.core.request_context import current_request_id, current_job_id
    import traceback

    # Attach IDs to context vars so nested @log_action calls can correlate
    req_token = current_request_id.set(request_id)
    job_token = current_job_id.set(job_id)

    try:
        # mark job as started
        LoggingService.update_job_log(
            job_id,
            status="running",
            mark_started=True,
        )

        # --- your heavy logic here -----------------------------------------
        # For demo purposes we just create a fake result.
        result = {
            "message": f"Job {job_id} processed prompt: {payload.get('prompt')}"
        }
        # --------------------------------------------------------------------

        # mark as succeeded + store result
        LoggingService.update_job_log(
            job_id,
            status="succeeded",
            result_payload=result,
            mark_finished=True,
        )

    except Exception as exc:
        LoggingService.update_job_log(
            job_id,
            status="failed",
            error_message=str(exc),
            error_traceback=traceback.format_exc(),
            mark_finished=True,
        )
        # Optional: re-raise if your worker framework wants to see the error
        # raise
    finally:
        # Always reset context vars
        current_request_id.reset(req_token)
        current_job_id.reset(job_token)


# --- Routes ------------------------------------------------------------------


@router.post("/job/submit", response_model=JobSubmitResponse)
@log_action(action_type="http", action_name="submit_job")
async def submit_job(
    request: Request,
    payload: JobCreate,
    background_tasks: BackgroundTasks,
) -> JobSubmitResponse:
    """
    Submit a new job.

    - Uses the request's `request_id` (from middleware) for correlation.
    - Creates a JobLog row with status `queued`.
    - Enqueues a background task to process the job.
    """
    # Correlation ID set by RequestLoggingMiddleware
    request_id: Optional[str] = getattr(request.state, "request_id", None)

    # Public job identifier (could also be shorter / different)
    job_id = str(uuid.uuid4())

    # Persist initial job metadata
    LoggingService.create_job_log(
        job_id=job_id,
        request_id=request_id,
        input_payload=payload.dict(),
        status="queued",
    )

    # Enqueue background work (FastAPI BackgroundTasks example)
    background_tasks.add_task(
        process_job,
        job_id=job_id,
        request_id=request_id,
        payload=payload.dict(),
    )

    # Optionally expose job_id on request.state for other decorators/services
    request.state.job_id = job_id

    return JobSubmitResponse(job_id=job_id)


@router.get("/job/{job_id}", response_model=JobStatusResponse)
@log_action(action_type="http", action_name="get_job_status")
async def get_job_status(
    request: Request,
    job_id: str,
    db: Session = Depends(get_db),
) -> JobStatusResponse:
    """
    Retrieve job status / result by job_id.

    - Fast path read from JobLog.
    - Keeps normal request logging (RequestLoggingMiddleware).
    """
    job: JobLog | None = (
        db.query(JobLog)
        .filter(JobLog.job_id == job_id)
        .first()
    )

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Only expose result when succeeded; you can adjust this logic
    result_value: Optional[str]
    if job.status == "succeeded":
        result_value = job.result_payload
    else:
        result_value = None

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        result=result_value,
        error=job.error_message,
    )
