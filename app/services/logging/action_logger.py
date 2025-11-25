# app/services/action_logger.py

import json
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import Request
from sqlalchemy.orm import Session

from app.models.request_log import ActionLog  # <-- adjust import path


def log_action(
    db: Session,
    request: Request,
    *,
    action_name: str,
    action_type: Optional[str] = None,
    module_name: Optional[str] = None,
    function_name: Optional[str] = None,
    line_number: Optional[int] = None,
    input_params: Optional[Dict[str, Any]] = None,
    output_results: Optional[Dict[str, Any]] = None,
    error: Optional[BaseException] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_prompt_tokens: Optional[int] = None,
    llm_completion_tokens: Optional[int] = None,
    llm_total_tokens: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    duration_ms: Optional[float] = None,
) -> None:
    """
    Simple 'fire and forget' action logger.
    You can precompute start/end/duration and pass them in, or just log metadata.
    """

    request_db_id = getattr(request.state, "request_db_id", None)

    # Convert dicts to JSON strings for input/output
    def to_json_str(d: Optional[Dict[str, Any]]) -> Optional[str]:
        if d is None:
            return None
        try:
            return json.dumps(d, ensure_ascii=False, default=str)
        except Exception:
            return str(d)

    error_message = None
    error_traceback = None
    is_error = 0

    if error is not None:
        is_error = 1
        error_message = str(error)
        error_traceback = traceback.format_exc()

    action = ActionLog(
        request_id=request_db_id,
        action_type=action_type,
        action_name=action_name,
        module_name=module_name,
        function_name=function_name,
        line_number=line_number,
        input_params=to_json_str(input_params),
        output_results=to_json_str(output_results),
        start_time=start_time or datetime.utcnow(),
        end_time=end_time,
        duration_ms=duration_ms,
        error_message=error_message,
        error_traceback=error_traceback,
        is_error=is_error,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_prompt_tokens=llm_prompt_tokens,
        llm_completion_tokens=llm_completion_tokens,
        llm_total_tokens=llm_total_tokens,
    )

    db.add(action)
    db.commit()


def log_timed_action(
    db: Session,
    request: Request,
    *,
    action_name: str,
    action_type: Optional[str] = None,
    module_name: Optional[str] = None,
    function_name: Optional[str] = None,
    line_number: Optional[int] = None,
    input_params: Optional[Dict[str, Any]] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_prompt_tokens: Optional[int] = None,
    llm_completion_tokens: Optional[int] = None,
    llm_total_tokens: Optional[int] = None,
    run: callable,
) -> Any:
    """
    Convenience helper that:
      - records start_time,
      - calls `run()`,
      - records end_time, duration_ms,
      - captures exceptions and marks is_error,
      - returns whatever `run()` returns.
    """
    py_start = time.perf_counter()
    start_time = datetime.utcnow()

    try:
        result = run()
        py_end = time.perf_counter()
        end_time = datetime.utcnow()
        duration_ms = (py_end - py_start) * 1000.0

        log_action(
            db,
            request,
            action_name=action_name,
            action_type=action_type,
            module_name=module_name,
            function_name=function_name,
            line_number=line_number,
            input_params=input_params,
            output_results={"result": result},
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_prompt_tokens=llm_prompt_tokens,
            llm_completion_tokens=llm_completion_tokens,
            llm_total_tokens=llm_total_tokens,
        )
        return result

    except Exception as e:
        py_end = time.perf_counter()
        end_time = datetime.utcnow()
        duration_ms = (py_end - py_start) * 1000.0

        log_action(
            db,
            request,
            action_name=action_name,
            action_type=action_type,
            module_name=module_name,
            function_name=function_name,
            line_number=line_number,
            input_params=input_params,
            output_results=None,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            error=e,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_prompt_tokens=llm_prompt_tokens,
            llm_completion_tokens=llm_completion_tokens,
            llm_total_tokens=llm_total_tokens,
        )
        raise
