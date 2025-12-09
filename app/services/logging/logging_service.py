"""
Helper service for structured request, action, and job logging.

This module centralizes all DB writes related to logging:

- request-level logs (HTTP request/response lifecycle),
- action-level logs (service calls, LLM calls, etc.),
- job-level logs (long-running or background tasks).

The idea is to keep the rest of your codebase free of direct logging
persistence logic while providing a small, focused API to record
what happened during a request or job.
"""

from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, Dict, Any
import json
import logging

from app.models.request_log import RequestLog, ActionLog, JobLog
from app.db.session import get_db_context
from app.core.config import settings

logger = logging.getLogger(__name__)


class LoggingService:
    """
    Thin service layer around `RequestLog`, `ActionLog`, and `JobLog` models.

    Responsibilities:
    - Sanitize data for safe DB storage,
    - Create and update request logs,
    - Create and update action logs,
    - Create and update job logs.
    """

    @staticmethod
    async def sanitize_data(data: Any, max_length: int = 500_000) -> Optional[str]:
        if data is None:
            return None
        try:
            if isinstance(data, (dict, list)):
                str_data = json.dumps(data, default=str)
            else:
                str_data = str(data)
            return str_data[:max_length] if len(str_data) > max_length else str_data
        except Exception as exc:
            logger.warning("Error sanitizing data: %s", str(exc))
            return "[ERROR SANITIZING DATA]"

    # ----------------------------- Request Logs ----------------------------- #
    @staticmethod
    async def create_request_log(
        request_id: str,
        method: str,
        url: str,
        query_params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        body: Optional[str] = None,
    ) -> Optional[int]:
        server_name = getattr(settings, "SERVER_NAME", None)
        api_version = getattr(settings, "API_VERSION", "0.0.0")
        try:
            with get_db_context() as db:
                request_log = RequestLog(
                    request_id=request_id,
                    method=method,
                    url=url,
                    query_params=LoggingService.sanitize_data(query_params),
                    headers=LoggingService.sanitize_data(headers),
                    body=LoggingService.sanitize_data(body),
                    server_name=server_name,
                    api_version=api_version,
                    start_time=datetime.utcnow(),
                )
                db.add(request_log)
                db.flush()
                return request_log.id
        except Exception as exc:
            logger.error("Error creating request log: %s", str(exc), exc_info=True)
            return None

    @staticmethod
    async def update_request_log(
        request_id: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        response_headers: Optional[Dict] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
    ) -> bool:
        try:
            with get_db_context() as db:
                request_log = db.query(RequestLog).filter(RequestLog.request_id == request_id).first()
                if not request_log:
                    logger.warning("Request log not found: %s", request_id)
                    return False

                end_time = datetime.utcnow()
                request_log.end_time = end_time
                request_log.duration_ms = (end_time - request_log.start_time).total_seconds() * 1000

                if status_code is not None:
                    request_log.status_code = status_code
                    request_log.is_error = 1 if status_code >= 400 else 0
                if response_body is not None:
                    request_log.response_body = LoggingService.sanitize_data(response_body)
                if error_message:
                    request_log.error_message = error_message
                    request_log.is_error = 1
                if error_traceback:
                    request_log.error_traceback = error_traceback

                logger.debug("Updated request log: %s", request_id)
                return True
        except Exception as exc:
            logger.error("Error updating request log %s: %s", request_id, str(exc), exc_info=True)
            return False

    # ----------------------------- Action Logs ----------------------------- #
    @staticmethod
    async def create_action_log(
        request_id: str,
        action_type: str,
        action_name: str,
        module_name: Optional[str] = None,
        function_name: Optional[str] = None,
        line_number: Optional[int] = None,
        input_params: Optional[Dict] = None,
    ) -> Optional[int]:
        try:
            with get_db_context() as db:
                request_log = db.query(RequestLog).filter(RequestLog.request_id == request_id).first()
                if not request_log:
                    logger.warning("Request log not found for action; request_id=%s", request_id)
                    return None

                action_log = ActionLog(
                    request_id=request_log.id,
                    action_type=action_type,
                    action_name=action_name,
                    module_name=module_name,
                    function_name=function_name,
                    line_number=line_number,
                    input_params=LoggingService.sanitize_data(input_params),
                    start_time=datetime.utcnow(),
                )
                db.add(action_log)
                db.flush()
                return action_log.id
        except Exception as exc:
            logger.error("Error creating action log for request %s: %s", request_id, str(exc), exc_info=True)
            return None

    @staticmethod
    async def update_action_log(
        action_log_id: int,
        output_result: Optional[Any] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
    ) -> bool:
        try:
            with get_db_context() as db:
                action_log = db.query(ActionLog).filter(ActionLog.id == action_log_id).first()
                if not action_log:
                    logger.warning("Action log not found: %s", action_log_id)
                    return False

                end_time = datetime.utcnow()
                action_log.end_time = end_time
                action_log.duration_ms = (end_time - action_log.start_time).total_seconds() * 1000

                if output_result is not None:
                    action_log.output_results = LoggingService.sanitize_data(output_result)
                if error_message:
                    action_log.error_message = error_message
                    action_log.is_error = 1
                if error_traceback:
                    action_log.error_traceback = error_traceback

                if output_result and isinstance(output_result, dict) and output_result.get("token_details"):
                    token_details = output_result.get("token_details")
                    action_log.llm_provider = getattr(settings, "LLM_PROVIDER", None)
                    provider = action_log.llm_provider
                    if provider == "bedrock":
                        action_log.llm_model = getattr(settings, "BEDROCK_MODEL_ID", None)
                    elif provider == "openai":
                        action_log.llm_model = getattr(settings, "OPENAI_MODEL", None)
                    else:
                        action_log.llm_model = None

                    action_log.llm_prompt_tokens = token_details.get("input_tokens")
                    action_log.llm_completion_tokens = token_details.get("output_tokens")
                    action_log.llm_total_tokens = token_details.get("total_tokens")

                logger.debug("Updated action log: %s", action_log_id)
                return True
        except Exception as exc:
            logger.error("Error updating action log %s: %s", action_log_id, str(exc), exc_info=True)
            return False

    # ------------------------------- Job Logs ------------------------------- #
    @staticmethod
    async def create_job_log(job_id: str, status: str = "pending", message: Optional[str] = None) -> Optional[int]:
        try:
            with get_db_context() as db:
                job = JobLog(
                    job_id=job_id,
                    status=status,
                    message=message,
                    start_time=datetime.utcnow(),
                )
                db.add(job)
                db.flush()
                return job.id
        except Exception as exc:
            logger.error("Error creating job log %s: %s", job_id, str(exc), exc_info=True)
            return None

    @staticmethod
    async def update_job_log(job_id: str, status: Optional[str] = None, message: Optional[str] = None) -> bool:
        try:
            with get_db_context() as db:
                job = db.query(JobLog).filter(JobLog.job_id == job_id).first()
                if not job:
                    logger.warning("Job log not found: %s", job_id)
                    return False

                now = datetime.utcnow()
                job.end_time = now
                if job.start_time:
                    job.duration_ms = (now - job.start_time).total_seconds() * 1000
                if status:
                    job.status = status
                if message:
                    job.message = message

                logger.debug("Updated job log: %s", job_id)
                return True
        except Exception as exc:
            logger.error("Error updating job log %s: %s", job_id, str(exc), exc_info=True)
            return False

    @staticmethod
    async def get_job(job_id: str) -> Optional[JobLog]:
        try:
            with get_db_context() as db:
                return db.query(JobLog).filter(JobLog.job_id == job_id).first()
        except Exception as exc:
            logger.error("Error getting job log %s: %s", job_id, str(exc), exc_info=True)
            return None
