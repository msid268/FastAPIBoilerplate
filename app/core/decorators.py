"""
Action-level structured logging decorator with full support for both
request_id and job_id correlation.

Why this exists
---------------
Service functions often perform multiple nested “actions”:

- DB queries
- External API calls
- Business logic chunks
- Calls to other services

When debugging issues, it is extremely valuable to know:

- which action was executed
- what parameters were passed in
- what result came back
- if any error occurred
- and crucially — which request and which background job it belongs to

This module provides a single decorator: `log_action`, which wraps a function
(sync or async) and creates a corresponding ActionLog row in the database.

The decorator is safe:
- If logging fails, your function still executes normally.
- No impact to business logic.
"""

from functools import wraps
from typing import Callable, Any, Optional, Mapping
import inspect
import logging
import traceback
import json

from app.services.logging.logging_service import LoggingService
from app.core.request_context import current_request_id, current_job_id

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
#  Helpers
# --------------------------------------------------------------------------------------

def _is_fastapi_request(obj: Any) -> bool:
    """
    Determine whether an object looks like a FastAPI/Starlette Request object.

    Returns
    -------
    bool
        True if the object is a Request instance or at least exposes a `.state`
        with a `request_id` attribute.
    """
    try:
        from starlette.requests import Request
        return isinstance(obj, Request)
    except Exception:
        return hasattr(obj, "state") and hasattr(getattr(obj, "state", None), "request_id")


def _find_request_id(args: tuple[Any], kwargs: dict[str, Any]) -> Optional[str]:
    """
    Locate the request_id using the hierarchy:

    1. request= keyword argument (if FastAPI Request)
    2. any positional argument that looks like a Request
    3. value stored in current_request_id ContextVar
    """
    req = kwargs.get("request")
    if req and _is_fastapi_request(req):
        rid = getattr(getattr(req, "state", None), "request_id", None)
        if rid:
            return rid

    for obj in args:
        if _is_fastapi_request(obj):
            rid = getattr(getattr(obj, "state", None), "request_id", None)
            if rid:
                return rid

    return current_request_id.get()


def _find_job_id(args: tuple[Any], kwargs: dict[str, Any]) -> Optional[str]:
    """
    Resolve job_id using the hierarchy:

    1. Explicit job_id argument: `job_id="abc"`
    2. From request.state (if FastAPI Request)
    3. From current_job_id ContextVar
    """
    # case 1: direct explicit argument
    if "job_id" in kwargs:
        return kwargs["job_id"]

    # case 2: look for job_id in FastAPI Request
    req = kwargs.get("request")
    if req and _is_fastapi_request(req):
        jid = getattr(getattr(req, "state", None), "job_id", None)
        if jid:
            return jid

    for obj in args:
        if _is_fastapi_request(obj):
            jid = getattr(getattr(obj, "state", None), "job_id", None)
            if jid:
                return jid

    # case 3: context var fallback
    return current_job_id.get()


def _safe_jsonable(obj: Any, max_len: int = 10_000) -> Any:
    """
    Convert an arbitrary object into a logging-safe representation.

    Strategies:
    - Attempt json.dumps(obj, default=str)
    - If it's small enough, return original obj (let DB/service serialize it)
    - If too large, return placeholder
    - If JSON fails, attempt __dict__ summary
    - Fallback: truncated repr()
    """
    try:
        s = json.dumps(obj, default=str)
        if len(s) <= max_len:
            return obj
        return f"<JSON too large: {len(s)} bytes>"
    except Exception:
        pass

    if hasattr(obj, "__dict__"):
        return {"type": type(obj).__name__, "id": getattr(obj, "id", None)}

    rep = repr(obj)
    return rep if len(rep) <= max_len else rep[:max_len] + "...<truncated>"


def _filter_params(bound: inspect.BoundArguments) -> Mapping[str, Any]:
    """
    Filter function arguments before logging.

    - Removes `self` / `cls`
    - Removes request objects
    - Converts values to safe JSONable form
    """
    filtered: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        if name in ("self", "cls"):
            continue
        if _is_fastapi_request(value):
            continue
        filtered[name] = _safe_jsonable(value)
    return filtered


# --------------------------------------------------------------------------------------
# Main decorator
# --------------------------------------------------------------------------------------

def log_action(
    action_type: str,
    action_name: Optional[str] = None,
    log_result: bool = True,
    log_params: bool = True,
) -> Callable:
    """
    Decorator that records action-level logs around a function call.

    Parameters
    ----------
    action_type : str
        High-level category (e.g. "service_call", "db_query")
    action_name : Optional[str]
        Logical name; defaults to the function name
    log_result : bool
        Whether to log sanitized output
    log_params : bool
        Whether to log sanitized input parameters

    Returns
    -------
    Callable
        Wrapped function, async or sync.
    """

    def decorator(func: Callable) -> Callable:
        is_coro = inspect.iscoroutinefunction(func)
        module_name = func.__module__
        function_name = func.__name__
        line_number = func.__code__.co_firstlineno
        resolved_action_name = action_name or function_name

        # ---------------------- internal helpers ------------------------------

        def create_action_log(request_id, job_id, input_params):
            """Create an ActionLog record if we have at least one correlation ID."""
            if not request_id and not job_id:
                logger.debug("No request_id or job_id; skipping action log creation")
                return None

            try:
                return LoggingService.create_action_log(
                    request_id=request_id,
                    job_id=job_id,
                    action_type=action_type,
                    action_name=resolved_action_name,
                    module_name=module_name,
                    function_name=function_name,
                    line_number=line_number,
                    input_params=input_params,
                )
            except Exception as exc:
                logger.error("Failed to create action log: %s", exc, exc_info=True)
                return None

        def update_result(action_log_id, result):
            if not (action_log_id and log_result):
                return
            try:
                LoggingService.update_action_log(
                    action_log_id=action_log_id,
                    output_result=_safe_jsonable(result),
                )
            except Exception:
                logger.debug("Could not log result", exc_info=True)

        def update_error(action_log_id, err):
            if not action_log_id:
                return
            try:
                LoggingService.update_action_log(
                    action_log_id=action_log_id,
                    error_message=str(err),
                    error_traceback=traceback.format_exc(),
                )
            except Exception:
                logger.error("Failed to update action log", exc_info=True)

        # ---------------------- sync wrapper ------------------------------

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            request_id = _find_request_id(args, kwargs)
            job_id = _find_job_id(args, kwargs)

            input_params = None
            if log_params:
                try:
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    input_params = _filter_params(bound)
                except Exception:
                    logger.debug("Failed to capture params", exc_info=True)

            action_log_id = create_action_log(request_id, job_id, input_params)

            try:
                result = func(*args, **kwargs)
                update_result(action_log_id, result)
                return result
            except Exception as exc:
                update_error(action_log_id, exc)
                raise

        # ---------------------- async wrapper ------------------------------

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            request_id = _find_request_id(args, kwargs)
            job_id = _find_job_id(args, kwargs)

            input_params = None
            if log_params:
                try:
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    input_params = _filter_params(bound)
                except Exception:
                    logger.debug("Failed to capture params", exc_info=True)

            action_log_id = create_action_log(request_id, job_id, input_params)

            try:
                result = await func(*args, **kwargs)
                update_result(action_log_id, result)
                return result
            except Exception as exc:
                update_error(action_log_id, exc)
                raise

        return async_wrapper if is_coro else sync_wrapper

    return decorator
