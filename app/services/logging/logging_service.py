"""
Helper service for structured request and action logging.

This module centralizes all DB writes related to logging:

- request-level logs (HTTP request/response lifecycle),
- action-level logs (service calls, LLM calls, etc.).

The idea is to keep the rest of your codebase free of direct logging
persistence logic while providing a small, focused API to record
what happened during a request.
"""

from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, Dict, Any
import json
import logging

from app.models.request_log import RequestLog, ActionLog
from app.db.session import get_db_context
from app.core.config import settings

logger = logging.getLogger(__name__)


class LoggingService:
    """
    Thin service layer around `RequestLog` and `ActionLog` models.

    This class only knows how to:
    - sanitize data so it is safe to store,
    - create/update request logs,
    - create/update action logs.

    It does not know anything about FastAPI, middleware, or decorators.
    Those higher-level pieces call into this service.
    """

    @staticmethod
    def sanitize_data(data: Any, max_length: int = 500_000) -> Optional[str]:
        """
        Convert arbitrary data into a safe, reasonably sized string.

        Parameters
        ----------
        data : Any
            Value to sanitize (dict, list, string, etc.).
        max_length : int, optional
            Maximum number of characters to keep, by default 500_000.

        Returns
        -------
        Optional[str]
            The sanitized string representation, or ``None`` if `data` is ``None``.
        """
        if data is None:
            return None

        try:
            # Try to keep structured shapes as JSON when possible.
            if isinstance(data, (dict, list)):
                str_data = json.dumps(data, default=str)
            else:
                str_data = str(data)

            # Hard cap to avoid blowing up the DB with huge payloads.
            if len(str_data) > max_length:
                str_data = str_data[:max_length]
            return str_data
        except Exception as exc:
            # Sanitizing is best-effort – if it fails, we log a hint and
            # store a generic error marker instead.
            logger.warning("Error sanitizing data: %s", str(exc))
            return "[ERROR SANITIZING DATA]"

    @staticmethod
    def create_request_log(
        request_id: str,
        method: str,
        url: str,
        query_params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        body: Optional[str] = None,
    ) -> Optional[int]:
        """
        Insert a new `RequestLog` row at the start of a request.

        Parameters
        ----------
        request_id : str
            Correlation ID used throughout the request (usually a UUID).
        method : str
            HTTP method (GET, POST, etc.).
        url : str
            Full request URL.
        query_params : Optional[Dict], optional
            Raw query parameters, by default None.
        headers : Optional[Dict], optional
            Raw request headers, by default None.
        body : Optional[str], optional
            Raw request body, by default None.

        Returns
        -------
        Optional[int]
            The primary key of the created `RequestLog`, or ``None`` if creation failed.
        """
        server_name = getattr(settings, "SERVER_NAME", None)
        api_version = getattr(settings, "API_VERSION", "0.0.0")

        try:
            # Use the app's DB context manager so transaction handling is consistent.
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
                db.flush()  # Flush so we get an ID without needing a full commit here.
                request_log_id = request_log.id

                logger.debug("Created request log: %s", request_log_id)
                return request_log_id
        except Exception as exc:
            logger.error("Error creating request log: %s", str(exc), exc_info=True)
            return None

    @staticmethod
    def update_request_log(
        request_id: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        response_headers: Optional[Dict] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
    ) -> bool:
        """
        Update an existing `RequestLog` entry at the end of a request.

        Parameters
        ----------
        request_id : str
            Correlation ID used to find the `RequestLog` row.
        status_code : Optional[int], optional
            HTTP status code of the response, by default None.
        response_body : Optional[str], optional
            Raw response body, by default None.
        response_headers : Optional[Dict], optional
            Response headers (currently unused but reserved), by default None.
        error_message : Optional[str], optional
            Short error message, if any, by default None.
        error_traceback : Optional[str], optional
            Full traceback string, if any, by default None.

        Returns
        -------
        bool
            ``True`` if update succeeded, ``False`` otherwise.
        """
        try:
            with get_db_context() as db:
                request_log: RequestLog | None = (
                    db.query(RequestLog)
                    .filter(RequestLog.request_id == request_id)
                    .first()
                )

                if not request_log:
                    # This usually means logging was not set up early enough,
                    # or the row was deleted.
                    logger.warning("Request log not found: %s", request_id)
                    return False

                end_time = datetime.utcnow()
                duration = (end_time - request_log.start_time).total_seconds() * 1000

                request_log.end_time = end_time
                request_log.duration_ms = duration

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
            logger.error(
                "Error updating request log %s: %s",
                request_id,
                str(exc),
                exc_info=True,
            )
            return False

    @staticmethod
    def create_action_log(
        request_id: str,
        action_type: str,
        action_name: str,
        module_name: Optional[str] = None,
        function_name: Optional[str] = None,
        line_number: Optional[int] = None,
        input_params: Optional[Dict] = None,
    ) -> Optional[int]:
        """
        Create an `ActionLog` row linked to an existing `RequestLog`.

        Parameters
        ----------
        request_id : str
            Correlation ID used to find the owning `RequestLog`.
        action_type : str
            Category of the action (e.g. ``"service_call"``, ``"llm_call"``).
        action_name : str
            Human-readable action name, usually the function's logical name.
        module_name : Optional[str], optional
            Python module where the action lives, by default None.
        function_name : Optional[str], optional
            Function name, by default None.
        line_number : Optional[int], optional
            Source line number, by default None.
        input_params : Optional[Dict], optional
            Sanitized input parameters, by default None.

        Returns
        -------
        Optional[int]
            ID of the created `ActionLog`, or ``None`` if creation failed.
        """
        try:
            with get_db_context() as db:
                # We first look up the parent RequestLog so we can store the FK ID.
                request_log: RequestLog | None = (
                    db.query(RequestLog)
                    .filter(RequestLog.request_id == request_id)
                    .first()
                )

                if not request_log:
                    logger.warning(
                        "Request log not found for action; request_id=%s",
                        request_id,
                    )
                    return None

                action_log = ActionLog(
                    request_id=request_log.id,  # DB FK (PK of RequestLog)
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
                action_log_id = action_log.id

                logger.debug(
                    "Created action log: %s for request_id=%s",
                    action_log_id,
                    request_id,
                )
                return action_log_id
        except Exception as exc:
            logger.error(
                "Error creating action log for request %s: %s",
                request_id,
                str(exc),
                exc_info=True,
            )
            return None

    @staticmethod
    def update_action_log(
        action_log_id: int,
        output_result: Optional[Any] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
    ) -> bool:
        """
        Update an existing `ActionLog` with result, error, or LLM metadata.

        Parameters
        ----------
        action_log_id : int
            Primary key of the `ActionLog` to update.
        output_result : Optional[Any], optional
            Raw result of the action (will be sanitized), by default None.
        error_message : Optional[str], optional
            Error message, if the action failed, by default None.
        error_traceback : Optional[str], optional
            Full traceback string, by default None.

        Returns
        -------
        bool
            ``True`` if the update succeeded, ``False`` otherwise.
        """
        try:
            with get_db_context() as db:
                action_log: ActionLog | None = (
                    db.query(ActionLog)
                    .filter(ActionLog.id == action_log_id)
                    .first()
                )

                if not action_log:
                    logger.warning("Action log not found: %s", action_log_id)
                    return False

                end_time = datetime.utcnow()
                duration = (end_time - action_log.start_time).total_seconds() * 1000

                action_log.end_time = end_time
                action_log.duration_ms = duration

                if output_result is not None:
                    # We store a sanitized representation only – not the raw value.
                    action_log.output_results = LoggingService.sanitize_data(
                        output_result
                    )

                if error_message:
                    action_log.error_message = error_message
                    action_log.is_error = 1

                if error_traceback:
                    action_log.error_traceback = error_traceback

                # LLM-related metadata is optional; we only touch fields
                # when values are provided.
                
                if output_result.get("token_details"):
                    token_details = output_result.get("token_details")
                    action_log.llm_provider = getattr(settings, 'LLM_PROVIDER')
                    if getattr(settings, 'LLM_PROVIDER') == 'bedrock':
                        action_log.llm_model = getattr(settings, 'BEDROCK_MODEL_ID')
                    elif getattr(settings, 'LLM_PROVIDER') == 'openai':
                        action_log.llm_model = getattr(settings, 'OPENAI_MODEL')
                    else:
                        action_log.llm_model = None

                    if token_details.get("input_tokens") is not None:
                        action_log.llm_prompt_tokens = token_details.get("input_tokens")

                    if token_details.get("output_tokens") is not None:
                        action_log.llm_completion_tokens = token_details.get("output_tokens")

                    if token_details.get("total_tokens") is not None:
                        action_log.llm_total_tokens = token_details.get("total_tokens")

                logger.debug("Updated action log: %s", action_log_id)
                return True

        except Exception as exc:
            logger.error(
                "Error updating action log %s: %s",
                action_log_id,
                str(exc),
                exc_info=True,
            )
            return False
