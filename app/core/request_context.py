"""
Utilities for storing a per-request ID using `contextvars`.

Why this exists
---------------
In a FastAPI (or any async) application, a single OS thread can be serving
many requests *at the same time* using `async`/`await`. Because of that:

- **Global variables** are shared between all requests (bad for request-scoped data).
- **Thread-locals** don't work reliably either, because multiple requests
  can be interleaved on the same thread.

`contextvars.ContextVar` solves this by giving you **request-scoped storage**
that is safe in async code. Each request can have its own `request_id`, and
any code running inside that request (middleware, dependencies, route
handlers, logging, DB calls, etc.) can read it without passing it around
explicitly.

Typical FastAPI usage
---------------------
Example wiring in FastAPI:

    from fastapi import FastAPI, Request
    from .request_context import set_request_id

    app = FastAPI()

    @app.middleware("http")
    async def add_request_id_middleware(request: Request, call_next):
        # Grab or generate a unique ID for this request
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Store it in the ContextVar so the whole call stack can see it
        set_request_id(rid)

        response = await call_next(request)
        # (Optional) you can also add it to the response headers
        response.headers["X-Request-ID"] = rid
        return response

Then anywhere in your code:

    from .request_context import get_request_id

    def log_something():
        rid = get_request_id()
        logger.info("Something happened", extra={"request_id": rid})

This module keeps all the context-var related logic in one place.
"""

from contextvars import ContextVar
from typing import Optional

# ContextVar that holds the *current* request ID for whatever request
# is being processed in this execution context.
#
# - Default is None so code can safely handle "no request" situations
#   (e.g. background tasks, scripts, startup hooks).
current_request_id: ContextVar[Optional[str]] = ContextVar(
    "current_request_id",
    default=None,
)


current_job_id = ContextVar("current_job_id", default=None)

def get_request_id() -> Optional[str]:
    """
    Get the current request ID stored in the context.

    Returns
    -------
    Optional[str]
        The request ID for the current execution context, or ``None`` if
        no request ID has been set yet.

    Notes
    -----
    This function is read-only and safe to call from anywhere in your
    FastAPI app (routes, dependencies, logging helpers, etc.).
    """
    return current_request_id.get()


def set_request_id(rid: str) -> None:
    """
    Set the request ID for the current execution context.

    Parameters
    ----------
    rid : str
        A unique identifier for the current request. This is often taken
        from a header (e.g. ``X-Request-ID``) or generated (e.g. UUID).

    Notes
    -----
    - You typically call this once at the beginning of request handling,
      e.g. in a FastAPI middleware.
    - After calling this, any code running *within the same request*
      can retrieve the ID via :func:`get_request_id`.
    """
    # We don't store the token returned here because we rarely need to
    # reset to the previous value manually in typical FastAPI middleware
    # setups. If you ever run nested request contexts, you might want
    # to keep that token and restore it later.
    current_request_id.set(rid)
