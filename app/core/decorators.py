# app/core/decorators.py
from functools import wraps
from typing import Callable, Any, Optional, Mapping
import inspect
import logging
import traceback
import json

from app.services.logging.logging_service import LoggingService
from app.core.request_context import current_request_id  # NEW

logger = logging.getLogger(__name__)

def _is_fastapi_request(obj: Any) -> bool:
    try:
        from starlette.requests import Request
        return isinstance(obj, Request)
    except Exception:
        return hasattr(obj, "state") and hasattr(getattr(obj, "state", None), "request_id")

def _find_request_id(args, kwargs) -> Optional[str]:
    # 1) look in kwargs/args
    for k in ("request",):
        req = kwargs.get(k)
        if req and _is_fastapi_request(req):
            rid = getattr(getattr(req, "state", None), "request_id", None)
            if rid:
                return rid
    for obj in args:
        if _is_fastapi_request(obj):
            rid = getattr(getattr(obj, "state", None), "request_id", None)
            if rid:
                return rid
    # 2) fallback to context var
    rid = current_request_id.get()
    return rid

def _safe_jsonable(obj: Any, max_len: int = 10_000) -> Any:
    try:
        s = json.dumps(obj, default=str)
        if len(s) <= max_len:
            return obj  # small enough, return original
        return f"<JSON too large: {len(s)} bytes, truncated>"
    except Exception:
        pass
    if hasattr(obj, "__dict__"):
        return {"type": type(obj).__name__, "id": getattr(obj, "id", None)}
    rep = repr(obj)
    return rep if len(rep) <= max_len else rep[:max_len] + "...<truncated>"

def _filter_params(bound: inspect.BoundArguments) -> Mapping[str, Any]:
    filtered = {}
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
    log_params: bool = True
):
    def decorator(func: Callable) -> Callable:
        is_coro = inspect.iscoroutinefunction(func)
        module_name = func.__module__
        function_name = func.__name__
        line_number = func.__code__.co_firstlineno
        resolved_action_name = action_name if action_name else function_name

        def create_action_log(request_id: Optional[str], input_params: Optional[Mapping[str, Any]]) -> Optional[int]:
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
                logger.error(f"Failed to create action log: {e}", exc_info=True)
                return None

        def update_with_result(action_log_id: Optional[int], result: Any) -> None:
            if not (action_log_id and log_result):
                return
            try:
                serializable = _safe_jsonable(result)
                LoggingService.update_action_log(
                    action_log_id=action_log_id,
                    output_result=serializable,
                )
            except Exception as e:
                logger.debug(f"Could not log result: {e}")

        def update_with_error(action_log_id: Optional[int], err: BaseException) -> None:
            if not action_log_id:
                return
            try:
                LoggingService.update_action_log(
                    action_log_id=action_log_id,
                    error_message=str(err),
                    error_traceback=traceback.format_exc(),
                )
            except Exception as log_error:
                logger.error(f"Failed to update action log: {log_error}")

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            request_id = _find_request_id(args, kwargs)
            input_params = None
            if log_params:
                try:
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    input_params = _filter_params(bound)
                except Exception as e:
                    logger.debug(f"Could not capture parameters: {e}", exc_info=True)

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
            request_id = _find_request_id(args, kwargs)
            input_params = None
            if log_params:
                try:
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    input_params = _filter_params(bound)
                except Exception as e:
                    logger.debug(f"Could not capture parameters: {e}", exc_info=True)

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
