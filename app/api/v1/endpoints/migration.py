from typing import Optional

import uuid
import time

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


# --- Decorated helpers to generate multiple ActionLogs ----------------------

@log_action(action_type="job_wait", action_name="wait_one_second")
def wait_one_second(
    job_id: Optional[str],
    second_index: int,
    total_seconds: int,
) -> dict:
    """
    Wait for 1 second and return timing info.

    This function is decorated, so each call creates its own ActionLog row
    (action_type="job_wait", action_name="wait_one_second").
    """
    start = time.perf_counter()
    time.sleep(1.0)
    duration_ms = (time.perf_counter() - start) * 1000.0

    return {
        "job_id": job_id,
        "second_index": second_index,
        "total_seconds": total_seconds,
        "duration_ms": round(duration_ms, 2),
        "message": f"Waited second {second_index}/{total_seconds}",
    }


@log_action(action_type="job_wait", action_name="wait_time")
def wait_for(seconds: int, job_id: Optional[str]) -> dict:
    """
    Wait for `seconds` seconds.

    - This call itself is logged as one ActionLog (wait_time).
    - Inside, we call `wait_one_second` `seconds` times, so you get
      one additional ActionLog per second.
    """
    ticks: list[dict] = []

    for i in range(seconds):
        tick_result = wait_one_second(
            job_id=job_id,
            second_index=i + 1,
            total_seconds=seconds,
        )
        ticks.append(tick_result)

    return {
        "job_id": job_id,
        "waited_seconds": seconds,
        "ticks": ticks,
    }


@log_action(action_type="job_step", action_name="pipeline_step")
def run_pipeline_step(
    job_id: str,
    request_id: Optional[str],
    step_name: str,
    sleep_seconds: int,
    prompt: str,
) -> dict:
    """
    One logical step in the job pipeline.

    Each call is a separate ActionLog (action_type="job_step").
    """
    start = time.perf_counter()
    time.sleep(sleep_seconds)
    duration_ms = (time.perf_counter() - start) * 1000.0

    return {
        "job_id": job_id,
        "request_id": request_id,
        "step": step_name,
        "simulated_sleep_s": sleep_seconds,
        "duration_ms": round(duration_ms, 2),
        "prompt_preview": prompt[:100],
    }


# --- Background worker entry point ------------------------------------------

@log_action(action_type="job", action_name="process_job")
def process_job(job_id: str, request_id: Optional[str], payload: dict) -> dict:
    """
    Background worker for a job.

    - Updates JobLog status to running/succeeded/failed.
    - Uses only decorated helpers, so *all* action logs are created via the
      `@log_action` decorator.
    """
    from app.core.request_context import current_request_id, current_job_id
    import traceback

    # Attach IDs to context vars so nested @log_action calls can correlate
    req_token = current_request_id.set(request_id)
    job_token = current_job_id.set(job_id)

    try:
        # Mark job as started
        LoggingService.update_job_log(
            job_id,
            status="running",
            mark_started=True,
        )

        prompt: str = payload.get("prompt", "")

        job_pipeline_start = time.perf_counter()

        # --------------------------------------------------------------------
        # 1) Wait actions (multiple ActionLogs via decorator):
        #    - one ActionLog for wait_for
        #    - one ActionLog per second for wait_one_second
        # --------------------------------------------------------------------
        wait_summary = wait_for(seconds=10, job_id=job_id)

        # --------------------------------------------------------------------
        # 2) Pipeline steps (multiple ActionLogs via decorator):
        #    each call to run_pipeline_step produces an ActionLog
        # --------------------------------------------------------------------
        steps = [
            ("validate_input", 3),
            ("fetch_context", 4),
            ("call_model", 6),
            ("post_process_result", 3),
        ]

        actions: list[dict] = []
        for step_name, seconds in steps:
            step_result = run_pipeline_step(
                job_id=job_id,
                request_id=request_id,
                step_name=step_name,
                sleep_seconds=seconds,
                prompt=prompt,
            )
            actions.append(step_result)

        total_duration_s = time.perf_counter() - job_pipeline_start

        # --------------------------------------------------------------------
        # 3) Fake "model" output & token usage (captured by decorator as part
        #    of the process_job result)
        # --------------------------------------------------------------------
        processed_text = f"Processed prompt for job {job_id}: {prompt!r}"

        input_tokens = len(prompt.split()) if prompt else 0
        output_tokens = len(processed_text.split())
        token_details = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

        result = {
            "message": processed_text,
            "job_id": job_id,
            "request_id": request_id,
            "wait_summary": wait_summary,
            "actions": actions,
            "total_duration_s": round(total_duration_s, 2),
            "token_details": token_details,
        }

        # Mark as succeeded + store result (JobLog, not ActionLog)
        LoggingService.update_job_log(
            job_id,
            status="succeeded",
            result_payload=result,
            mark_finished=True,
        )

        # Let @log_action see the result (for ActionLog + LLM token fields)
        return result

    except Exception as exc:
        LoggingService.update_job_log(
            job_id,
            status="failed",
            error_message=str(exc),
            error_traceback=traceback.format_exc(),
            mark_finished=True,
        )
        return {
            "message": "Job failed",
            "job_id": job_id,
            "request_id": request_id,
            "error": str(exc),
        }

    finally:
        # Always reset context vars
        current_request_id.reset(req_token)
        current_job_id.reset(job_token)


# --- Routes ------------------------------------------------------------------


@router.post("/submit", response_model=JobSubmitResponse)
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

    # Persist initial job metadata (JobLog)
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


@router.get("/{job_id}", response_model=JobStatusResponse)
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
    if job.status == "succeeded":
        result_value: Optional[str] = job.result_payload
    else:
        result_value = None

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        result=result_value,
        error=job.error_message,
    )
