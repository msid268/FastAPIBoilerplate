# app/core/decorators.py
"""
Infrastructure for action-level logging.

`log_action` is a reusable decorator that records:
- which logical action was executed (type/name/module/function/line),
- its input parameters (optional),
- its result or any raised exception (optional),
and ties all of this back to a request-level identifier when available.
"""

from functools import wraps
from typing import Callable, Any, Optional, Mapping
import inspect
import logging
import traceback
import json

from app.services.logging.logging_service import LoggingService
from app.core.request_context import current_request_id

logger = logging.getLogger(__name__)


def _is_fastapi_request(obj: Any) -> bool:
    """
    Return True if the given object looks like a FastAPI/Starlette Request.

    The import is done lazily so this module can be imported outside of a
    FastAPI context (e.g. in scripts or tests) without pulling in Starlette.
    """
    try:
        from starlette.requests import Request
        return isinstance(obj, Request)
    except Exception:
        # Fallback heuristic for environments where Starlette may not be importable.
        return hasattr(obj, "state") and hasattr(getattr(obj, "state", None), "request_id")


def _find_request_id(args, kwargs) -> Optional[str]:
    """
    Try to resolve the current request id from function arguments or the context var.

    Resolution order:
    1. Look for a `request` kwarg that is a Request and has `state.request_id`.
    2. Scan all positional arguments for a Request instance with `state.request_id`.
    3. Fallback to the `current_request_id` context variable.
    """
    # Look for an explicit `request` keyword argument first.
    for k in ("request",):
        req = kwargs.get(k)
        if req and _is_fastapi_request(req):
            rid = getattr(getattr(req, "state", None), "request_id", None)
            if rid:
                return rid

    # Then scan positional arguments.
    for obj in args:
        if _is_fastapi_request(obj):
            rid = getattr(getattr(obj, "state", None), "request_id", None)
            if rid:
                return rid

    # Finally, fall back to the context variable (e.g. set by middleware).
    rid = current_request_id.get()
    return rid


def _safe_jsonable(obj: Any, max_len: int = 10_000) -> Any:
    """
    Best-effort conversion of arbitrary objects into something JSON/log friendly.

    - Tries json.dumps with a `default=str` hook.
    - If the JSON representation is too large, returns a placeholder string.
    - Falls back to a small dictionary (type/id) for objects with a __dict__.
    - Otherwise uses `repr`, truncated to `max_len` characters.
    """
    try:
        s = json.dumps(obj, default=str)
        if len(s) <= max_len:
            # Small enough; return the original object so the logger/service
            # can decide how to persist it.
            return obj
        return f"<JSON too large: {len(s)} bytes, truncated>"
    except Exception:
        pass

    if hasattr(obj, "__dict__"):
        return {"type": type(obj).__name__, "id": getattr(obj, "id", None)}

    rep = repr(obj)
    return rep if len(rep) <= max_len else rep[:max_len] + "...<truncated>"


def _filter_params(bound: inspect.BoundArguments) -> Mapping[str, Any]:
    """
    Filter and sanitize function arguments for logging.

    - Strips out `self` / `cls`.
    - Excludes any Request objects.
    - Runs the remaining values through `_safe_jsonable`.
    """
    filtered: dict[str, Any] = {}
    for k, v in bound.arguments.items():
        if k in ("self", "cls"):
            continue
        if _is_fastapi_request(v):
            continue
        filtered[k] = _safe_jsonable(v)
    return filtered


def log_action(
    action_type: str,
    action_name: Optional[str] = None,
    log_result: bool = True,
    log_params: bool = True,
):
    """
    Decorator for recording structured action logs around a function call.

    Typical usage:

        @log_action(action_type="service_call", log_result=False)
        def do_something(arg1, arg2):
            ...

    Behaviour:
    - At function entry, a new action log row is created (if a request id can be resolved).
    - If `log_params` is True, arguments (minus self/cls/Request) are stored.
    - On successful completion, the result is stored if `log_result` is True.
    - On exception, the error message and traceback are recorded.
    - If no request id can be found, the decorator becomes a no-op for persistence
      but still calls the underlying function.

    The decorator works for both sync and async callables.
    """

    def decorator(func: Callable) -> Callable:
        is_coro = inspect.iscoroutinefunction(func)
        module_name = func.__module__
        function_name = func.__name__
        line_number = func.__code__.co_firstlineno
        resolved_action_name = action_name if action_name else function_name

        def create_action_log(
            request_id: Optional[str],
            input_params: Optional[Mapping[str, Any]],
        ) -> Optional[int]:
            """
            Create a new action log entry if a request id is present.

            Returns the action_log_id from the LoggingService or None if logging
            is skipped or fails.
            """
            if not request_id:
                logger.debug("No request_id found, skipping action log creation")
                return None
            try:
                return LoggingService.create_action_log(
                    request_id=request_id,
                    action_type=action_type,
                    action_name=resolved_action_name,
                    module_name=module_name,
                    function_name=function_name,
                    line_number=line_number,
                    input_params=input_params,
                )
            except Exception as e:
                logger.error("Failed to create action log: %s", e, exc_info=True)
                return None

        def update_with_result(action_log_id: Optional[int], result: Any) -> None:
            """
            Update an existing action log with the function result, if requested.
            """
            if not (action_log_id and log_result):
                return
            try:
                serializable = _safe_jsonable(result)
                LoggingService.update_action_log(
                    action_log_id=action_log_id,
                    output_result=serializable,
                )
            except Exception as e:
                # Result logging should never break the main code path.
                logger.debug("Could not log result: %s", e)

        def update_with_error(action_log_id: Optional[int], err: BaseException) -> None:
            """
            Update an existing action log with error details.
            """
            if not action_log_id:
                return
            try:
                LoggingService.update_action_log(
                    action_log_id=action_log_id,
                    error_message=str(err),
                    error_traceback=traceback.format_exc(),
                )
            except Exception as log_error:
                logger.error("Failed to update action log: %s", log_error)

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            """
            Wrapper for synchronous functions.
            """
            request_id = _find_request_id(args, kwargs)

            input_params: Optional[Mapping[str, Any]] = None
            if log_params:
                try:
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    input_params = _filter_params(bound)
                except Exception as e:
                    logger.debug("Could not capture parameters: %s", e, exc_info=True)

            action_log_id = create_action_log(request_id, input_params)

            try:
                result = func(*args, **kwargs)
                update_with_result(action_log_id, result)
                return result
            except Exception as e:
                update_with_error(action_log_id, e)
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            """
            Wrapper for coroutine functions.
            """
            request_id = _find_request_id(args, kwargs)

            input_params: Optional[Mapping[str, Any]] = None
            if log_params:
                try:
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    input_params = _filter_params(bound)
                except Exception as e:
                    logger.debug("Could not capture parameters: %s", e, exc_info=True)

            action_log_id = create_action_log(request_id, input_params)

            try:
                result = await func(*args, **kwargs)
                update_with_result(action_log_id, result)
                return result
            except Exception as e:
                update_with_error(action_log_id, e)
                raise

        return async_wrapper if is_coro else sync_wrapper

    return decorator
