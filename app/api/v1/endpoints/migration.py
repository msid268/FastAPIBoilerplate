import uuid
import asyncio
from fastapi import APIRouter, Request, BackgroundTasks
from app.services.logging.logging_service import LoggingService

router = APIRouter()


# ---------------------------------------------------------------------
# 1. START A LONG-POLLING JOB (returns immediately)
# ---------------------------------------------------------------------
@router.post("/long-poll/start")
async def start_long_poll(request: Request, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    # Log request and create job
    await LoggingService.create_request_log(
        request_id=request_id,
        method="POST",
        url="/long-poll/start",
        body=(await request.body()).decode("utf-8"),
    )

    await LoggingService.create_job_log(
        job_id=job_id,
        status="running",
        message="Job started",
    )

    print(f"[TEST LOG] Started long-poll job: {job_id}")

    # Launch background task
    background_tasks.add_task(run_long_poll_job, job_id)

    # Return immediately
    return {"request_id": request_id, "job_id": job_id, "status": "running"}


# ---------------------------------------------------------------------
# Background task: simulate long polling
# ---------------------------------------------------------------------
async def run_long_poll_job(job_id: str):
    for i in range(10):
        await asyncio.sleep(1)
        print(f"[TEST LOG] Job {job_id} tick {i+1}/10")

    # Mark job complete
    LoggingService.update_job_log(
        job_id=job_id,
        status="completed",
        message="Long polling completed successfully",
    )

    print(f"[TEST LOG] Job completed: {job_id}")


# ---------------------------------------------------------------------
# 2. CLIENT POLLS JOB STATUS
# ---------------------------------------------------------------------
@router.get("/long-poll/check/{job_id}")
async def check_job_status(job_id: str):
    job = await LoggingService.get_job(job_id)

    if not job:
        return {"job_id": job_id, "status": "not_found"}

    print(f"[TEST LOG] Checking job: {job_id} → {job.status}")

    return {
        "job_id": job_id,
        "status": job.status,
        "message": job.message,
    }
