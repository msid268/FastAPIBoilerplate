"""
Infrastructure for action-level logging using a decorator.

Why this exists
---------------
For a single request we often trigger multiple "actions" deep in our service layer:

- calling external APIs,
- talking to the database,
- running domain logic etc.

When something goes wrong, it's very helpful to know:

- *which* action ran,
- *with what* input,
- *what* it returned, or
- *which* error it raised,

and to be able to tie all of that back to a single request-level ID.

What this module provides
-------------------------
This module exposes a single decorator:

- :func:`log_action` — wraps a function (sync or async) and:
  - records a new "action log" row on function entry,
  - optionally logs the sanitized input parameters,
  - optionally logs the sanitized output result,
  - logs any exception message + traceback,
  - links everything to a request-level identifier when available.

The decorator is **non-invasive**: if logging fails for any reason, the
original function is still executed normally.

Request ID correlation
----------------------
The decorator tries to find a `request_id` so logs can be grouped per
incoming HTTP request. The resolution order is:

1. A `request` keyword argument that looks like a FastAPI/Starlette Request
   and has `request.state.request_id`.
2. Any positional argument that looks like a Request with `state.request_id`.
3. The `current_request_id` context variable set by your middleware.

Typical FastAPI usage
---------------------
Example middleware that sets `current_request_id`:

    from fastapi import FastAPI, Request
    import uuid

    from app.core.request_context import set_request_id

    app = FastAPI()

    @app.middleware("http")
    async def add_request_id_middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        set_request_id(rid)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

Example of using the decorator in a service function:

    from app.services.logging.action_logging import log_action

    @log_action(action_type="user_service", action_name="create_user")
    async def create_user(email: str, name: str) -> User:
        ...
        return user

The `LoggingService` handles persistence (e.g. writing into a database),
while this module focuses on collecting and structuring the data.
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
    Check whether ``obj`` looks like a FastAPI/Starlette ``Request``.

    Parameters
    ----------
    obj : Any
        Object to inspect.

    Returns
    -------
    bool
        ``True`` if the object behaves like a Request instance, otherwise
        ``False``.

    Notes
    -----
    - The import of ``Request`` is done lazily so this module can be used
      in environments where Starlette is not installed (tests, scripts, etc.).
    - If Starlette cannot be imported, a lightweight heuristic is used
      (looking for a ``state`` attribute with a ``request_id``).
    """
    try:
        from starlette.requests import Request
        return isinstance(obj, Request)
    except Exception:
        # Fallback heuristic for environments where Starlette may not be importable.
        # This is intentionally loose: we only care that it "looks enough" like
        # a Request for the purpose of extracting a request_id.
        return hasattr(obj, "state") and hasattr(getattr(obj, "state", None), "request_id")


def _find_request_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[str]:
    """
    Resolve the current request ID from function arguments or the context var.

    Resolution order
    ----------------
    1. A ``request`` keyword argument that is a Request and has ``state.request_id``.
    2. Any positional argument that is a Request with ``state.request_id``.
    3. The :data:`current_request_id` context variable.

    Parameters
    ----------
    args : tuple
        Positional arguments passed to the wrapped function.
    kwargs : dict
        Keyword arguments passed to the wrapped function.

    Returns
    -------
    Optional[str]
        The resolved request ID, or ``None`` if no ID could be found.
    """
    # Look for an explicit `request` keyword argument first – this is the most common
    # pattern in FastAPI route handlers and dependencies.
    req = kwargs.get("request")
    if req and _is_fastapi_request(req):
        rid = getattr(getattr(req, "state", None), "request_id", None)
        if rid:
            return rid

    # Then scan positional arguments for a Request-like object.
    for obj in args:
        if _is_fastapi_request(obj):
            rid = getattr(getattr(obj, "state", None), "request_id", None)
            if rid:
                return rid

    # Finally, fall back to the context variable (e.g. set by middleware).
    return current_request_id.get()


def _safe_jsonable(obj: Any, max_len: int = 10_000) -> Any:
    """
    Convert arbitrary objects into something JSON/log friendly (best effort).

    Parameters
    ----------
    obj : Any
        The value that should be logged.
    max_len : int, optional
        Maximum allowed length for the serialized representation, by default 10_000.

    Returns
    -------
    Any
        A value that should be safe to pass to JSON/logging systems.

    Strategy
    --------
    - Try ``json.dumps`` with ``default=str`` to see if it's reasonably sized.
      If the JSON string is small enough, we return the *original* object so
      the logging backend can decide how to persist it.
    - If the JSON string is too large, return a short placeholder message.
    - If JSON serialization fails, and the object has a ``__dict__``, return
      a small dict with its type and an ``id`` attribute if available.
    - As a last resort, return a truncated ``repr(obj)``.
    """
    try:
        s = json.dumps(obj, default=str)
        if len(s) <= max_len:
            # Small enough; return the original object so the logger/service
            # can decide how to persist it (e.g. store raw JSON, string, etc.).
            return obj
        return f"<JSON too large: {len(s)} bytes, truncated>"
    except Exception:
        # If even JSON with default=str fails, we fall through to other strategies.
        pass

    if hasattr(obj, "__dict__"):
        # Provide a tiny, stable summary instead of dumping the whole object.
        return {"type": type(obj).__name__, "id": getattr(obj, "id", None)}

    rep = repr(obj)
    return rep if len(rep) <= max_len else rep[:max_len] + "...<truncated>"


def _filter_params(bound: inspect.BoundArguments) -> Mapping[str, Any]:
    """
    Filter and sanitize function arguments for logging.

    Parameters
    ----------
    bound : inspect.BoundArguments
        Bound arguments object produced by ``inspect.Signature.bind``.

    Returns
    -------
    Mapping[str, Any]
        A mapping of parameter names to sanitized, log-safe values.

    Notes
    -----
    - ``self`` / ``cls`` are stripped out to keep logs focused.
    - Request objects are excluded entirely so we don't dump framework internals.
    - Remaining values are passed through :func:`_safe_jsonable`.
    """
    filtered: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        if name in ("self", "cls"):
            # Instance/class references are not interesting for structured logs.
            continue
        if _is_fastapi_request(value):
            # Avoid logging the entire Request object; we only care about its ID.
            continue
        filtered[name] = _safe_jsonable(value)
    return filtered


def log_action(
    action_type: str,
    action_name: Optional[str] = None,
    log_result: bool = True,
    log_params: bool = True,
) -> Callable:
    """
    Decorator for recording structured action logs around a function call.

    Parameters
    ----------
    action_type : str
        High-level category of the action (e.g. ``"service_call"``,
        ``"db_query"``, ``"external_api"``).
    action_name : Optional[str], optional
        Logical name of the action. If not provided, the wrapped function's
        name is used.
    log_result : bool, optional
        Whether to store the function's return value, by default True.
    log_params : bool, optional
        Whether to store sanitized input parameters, by default True.

    Returns
    -------
    Callable
        A decorator that wraps the target function.

    Behaviour
    ---------
    - On function entry:
      - Resolves a ``request_id``.
      - Optionally captures a sanitized snapshot of input parameters.
      - Creates an action log row via :class:`LoggingService`.
    - On successful completion:
      - Optionally logs a sanitized representation of the result.
    - On exception:
      - Logs the error message and full traceback.
      - Re-raises the original exception.

    Notes
    -----
    - If no ``request_id`` can be found, the decorator becomes effectively
      a no-op for persistence (no DB writes), but the function is still
      executed as normal.
    - The decorator supports both synchronous and asynchronous callables.
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
            Create a new action log entry if a request ID is present.

            Parameters
            ----------
            request_id : Optional[str]
                The request identifier to attach this action to.
            input_params : Optional[Mapping[str, Any]]
                Sanitized input parameters, or ``None`` if capturing was skipped.

            Returns
            -------
            Optional[int]
                The ``action_log_id`` from :class:`LoggingService`, or ``None``
                if logging is skipped or fails.
            """
            if not request_id:
                # No request context available (e.g. background task,
                # CLI script, or logging not wired yet).
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
            except Exception as exc:
                # Logging must never break the main flow, so we just log the issue.
                logger.error("Failed to create action log: %s", exc, exc_info=True)
                return None

        def update_with_result(action_log_id: Optional[int], result: Any) -> None:
            """
            Update an existing action log with the function result.

            Parameters
            ----------
            action_log_id : Optional[int]
                Identifier of the action log row to update.
            result : Any
                The result returned by the wrapped function.

            Notes
            -----
            If ``log_result`` is ``False`` or ``action_log_id`` is falsy,
            this function is a no-op.
            """
            if not (action_log_id and log_result):
                return
            try:
                serializable = _safe_jsonable(result)
                LoggingService.update_action_log(
                    action_log_id=action_log_id,
                    output_result=serializable,
                )
            except Exception as exc:
                # Result logging is "nice to have"; silently swallow errors here.
                logger.debug("Could not log result: %s", exc, exc_info=True)

        def update_with_error(action_log_id: Optional[int], err: BaseException) -> None:
            """
            Update an existing action log with error details.

            Parameters
            ----------
            action_log_id : Optional[int]
                Identifier of the action log row to update.
            err : BaseException
                Exception raised by the wrapped function.
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
                # If even error logging fails, we still don't interfere with
                # the original exception propagation.
                logger.error("Failed to update action log: %s", log_error, exc_info=True)

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            """
            Wrapper for synchronous functions.

            This is the code path used when the wrapped function is not
            declared with ``async def``.
            """
            request_id = _find_request_id(args, kwargs)

            input_params: Optional[Mapping[str, Any]] = None
            if log_params:
                try:
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    input_params = _filter_params(bound)
                except Exception as exc:
                    # Parameter capturing is non-critical; if something is odd
                    # (e.g. weird *args/**kwargs combination), we just log it.
                    logger.debug("Could not capture parameters: %s", exc, exc_info=True)

            action_log_id = create_action_log(request_id, input_params)

            try:
                result = func(*args, **kwargs)
                update_with_result(action_log_id, result)
                return result
            except Exception as exc:
                update_with_error(action_log_id, exc)
                # Always re-raise the original exception so behaviour is unchanged.
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            """
            Wrapper for coroutine functions.

            This is the code path used when the wrapped function is declared
            as ``async def``.
            """
            request_id = _find_request_id(args, kwargs)

            input_params: Optional[Mapping[str, Any]] = None
            if log_params:
                try:
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    input_params = _filter_params(bound)
                except Exception as exc:
                    logger.debug("Could not capture parameters: %s", exc, exc_info=True)

            action_log_id = create_action_log(request_id, input_params)

            try:
                result = await func(*args, **kwargs)
                update_with_result(action_log_id, result)
                return result
            except Exception as exc:
                update_with_error(action_log_id, exc)
                raise

        # Choose the correct wrapper based on whether the function is async.
        return async_wrapper if is_coro else sync_wrapper

    return decorator
