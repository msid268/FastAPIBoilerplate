"""
Request/response logging middleware for FastAPI.

Why this exists
---------------
In a typical FastAPI application, each incoming HTTP request can:

- hit multiple endpoints and dependencies,
- call several services and external APIs,
- raise exceptions at various layers.

When you're debugging a production issue, it helps a lot to have a single
place where you can see:

- what the request looked like (method, URL, headers, body),
- how long it took,
- what the response was,
- whether it failed and with which traceback,
- a stable `request_id` that ties everything together.

This middleware creates a `RequestLog` database row for every request and
populates a **request-scoped identifier** that other parts of the system
can use (for example the :func:`log_action` decorator).

What this module provides
-------------------------
- :class:`RequestLoggingMiddleware`
  A Starlette-compatible middleware that:
  - generates a UUID-based `request_id`,
  - stores that ID in the `current_request_id` ContextVar,
  - persists a `RequestLog` entry at the beginning of the request,
  - updates it with response data, timing, and errors.

- :func:`get_request_id`
  Convenience helper to get the database primary key of the `RequestLog`
  row from a FastAPI `Request` object.

How it interacts with action-level logging
------------------------------------------
The `RequestLoggingMiddleware` sets the request ID in two places:

- `current_request_id` (a ContextVar) — this allows the
  :func:`log_action` decorator to link service-level actions to the
  current request even if the `Request` object is not passed around.
- `request.state.request_id` — this makes the ID available to any route,
  dependency, or service that receives the `Request` object.

The general flow is:

1. Middleware starts → generates UUID → sets `current_request_id`.
2. Middleware writes an initial `RequestLog` row.
3. Your routes and services run (possibly using `@log_action`).
4. Middleware captures the response (or error) and updates the `RequestLog`.
5. Middleware resets `current_request_id` so it doesn’t leak between requests.
"""


import json
import traceback
import uuid
import time
from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.request_log import RequestLog
from app.core.config import settings
from app.core.request_context import current_request_id  # <-- ContextVar used by decorators


def get_request_id(request: Request):
    """
    Return the database primary key of the current request's `RequestLog` row.

    Parameters
    ----------
    request : fastapi.Request
        The active request object.

    Returns
    -------
    Optional[int]
        The integer primary key of the `RequestLog` entry for this request,
        or ``None`` if it cannot be found.

    Notes
    -----
    - This is just a convenience helper; the more common identifier is the
      UUID-style `request_id` that lives in `request.state.request_id`.
    - This helper relies on the middleware having run and attached
      ``request.state.request_db_id``.
    """
    if not hasattr(request.state, "request_db_id"):
        # This usually means the middleware is not installed or ran into
        # an error before it could attach state. We log a hint for debugging.
        print("MISSING request.state.request_db_id")
        return None
    return request.state.request_db_id


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs incoming HTTP requests and outgoing responses.

    Responsibilities
    ----------------
    - Generate a per-request UUID (`request_id`).
    - Store it in the :data:`current_request_id` ContextVar so it can be
      picked up by logging decorators.
    - Persist a `RequestLog` row at the start of the request.
    - Capture response body, status code, duration, and error details (if any).
    - Reset the ContextVar at the end to avoid cross-request leakage.

    Typical registration
    --------------------
    In your FastAPI setup:

        from fastapi import FastAPI
        from app.middleware.request_logging import RequestLoggingMiddleware

        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

    Once installed, you’ll have a complete audit trail for each request in
    your `RequestLog` table, plus a request-level ID that other parts of the
    app (like `@log_action`) can use for correlation.
    """

    async def dispatch(self, request: Request, call_next):
        """
        Core middleware entry point.

        This method wraps the request/response lifecycle:

        1. Opens a DB session.
        2. Generates a request UUID and sets `current_request_id`.
        3. Reads and stores the incoming request body, headers, and query params.
        4. Creates a `RequestLog` DB row.
        5. Calls the downstream handler (endpoint + other middleware).
        6. Captures the response body and updates the `RequestLog` row.
        7. Handles any exception by logging it and re-raising.
        8. Closes the DB session and resets the ContextVar token.
        """
        db: Session = SessionLocal()

        py_start_time = time.perf_counter()
        utc_start = datetime.utcnow()

        # Generate a unique request_id (string, 36 chars, standard UUID4).
        # This ID is used everywhere to correlate logs.
        request_uuid = str(uuid.uuid4())
        request_id= request.headers.get("X-Request-ID")
        request_uuid = await self._get_or_generate_request_id(
            db=db,
            client_request_id=request_id
        )
        # Propagate to ContextVar so decorators (e.g. `log_action`) can read
        # it, even if they don't receive the Request object explicitly.
        token = current_request_id.set(request_uuid)

        # Read request body once; ASGI only gives us the stream one time.
        body_bytes = await request.body()
        try:
            body_text = body_bytes.decode("utf-8")
        except Exception:
            # Don't crash on weird encodings; just mark it as binary.
            body_text = "<binary>"

        # Re-inject the body so downstream handlers can still read it normally.
        # We replace the underlying ASGI receive call with one that returns
        # our buffered body.
        async def receive():
            return {"type": "http.request", "body": body_bytes}

        request._receive = receive  # type: ignore[attr-defined]

        # Serialize headers & query params as JSON for structured storage.
        headers_dict = dict(request.headers)
        query_dict = dict(request.query_params)

        headers_str = json.dumps(headers_dict, ensure_ascii=False)
        query_str = json.dumps(query_dict, ensure_ascii=False)

        req_log = None
        try:
            # 1) Create RequestLog row with all the request-side information.
            req_log = RequestLog(
                request_id=request_uuid,
                method=request.method,
                url=str(request.url),
                query_params=query_str,
                headers=headers_str,
                body=body_text,
                server_name=getattr(settings, "SERVER_NAME", None)
                if "settings" in globals()
                else None,
                api_version=getattr(settings, "API_VERSION", None)
                if "settings" in globals()
                else None,
                start_time=utc_start,
                is_error=0,
            )
            db.add(req_log)
            db.commit()
            db.refresh(req_log)

            # Attach identifiers to request.state so actions/decorators can use them.
            # These are available throughout the request lifecycle.
            request.state.request_id = request_uuid            # what the decorator looks for
            request.state.request_db_id = req_log.id           # PK in DB
            request.state.request_public_id = request_uuid     # e.g. for external correlation

            # 2) Call the actual endpoint stack (other middleware + route handler).
            response = await call_next(request)

            # 3) Capture the response body. We need to consume the body iterator
            # so we can both log it and send it back to the client.
            resp_body_bytes = b""
            async for chunk in response.body_iterator:
                resp_body_bytes += chunk

            try:
                resp_text = resp_body_bytes.decode("utf-8")
            except Exception:
                resp_text = "<binary>"

            # 4) Update RequestLog with response-side info and duration.
            py_end_time = time.perf_counter()
            utc_end = datetime.utcnow()
            duration_ms = (py_end_time - py_start_time) * 1000.0

            req_log.status_code = response.status_code
            req_log.response_body = resp_text
            req_log.end_time = utc_end
            req_log.duration_ms = duration_ms
            req_log.is_error = 1 if response.status_code >= 400 else 0

            db.add(req_log)
            db.commit()

            # 5) Rebuild the Response with the captured body so FastAPI can
            # still return it to the client as expected.
            return Response(
                content=resp_body_bytes,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        except Exception as exc:
            # If something explodes anywhere in the pipeline, we still try to
            # log it in `RequestLog` before letting FastAPI handle the error.
            db.rollback()  # important: clear failed transaction before reusing the session

            py_end_time = time.perf_counter()
            utc_end = datetime.utcnow()
            duration_ms = (py_end_time - py_start_time) * 1000.0

            tb = traceback.format_exc()

            if req_log is None:
                # If we failed before creating the row, try to create a minimal one now.
                req_log = RequestLog(
                    request_id=request_uuid,
                    method=request.method,
                    url=str(request.url),
                    query_params=query_str,
                    headers=headers_str,
                    body=body_text,
                    server_name=getattr(settings, "SERVER_NAME", None)
                    if "settings" in globals()
                    else None,
                    api_version=getattr(settings, "API_VERSION", None)
                    if "settings" in globals()
                    else None,
                    start_time=utc_start,
                )

            req_log.status_code = 500
            req_log.response_body = f"Internal Server Error: {exc}"
            req_log.end_time = utc_end
            req_log.duration_ms = duration_ms
            req_log.is_error = 1
            req_log.error_message = str(exc)
            req_log.error_traceback = tb

            db.add(req_log)
            db.commit()

            # Re-raise so FastAPI's normal exception handling still kicks in
            # (e.g. HTTPException handlers, global error handlers, etc.).
            raise
        finally:
            # Always clean up the DB session and reset the ContextVar token,
            # even if an exception was raised.
            db.close()
            try:
                current_request_id.reset(token)
            except Exception:
                # If something odd happens with the token, we swallow it:
                # request cleanup should never crash the process.
                pass
    
    
    async def _get_or_generate_request_id(
        self,
        db: Session,
        client_request_id: str | None,
    ) -> str:
        """
        If client_request_id is provided and unique, use it.
        Otherwise generate a new UUID4 string.
        """
        # No client ID → generate
        if not client_request_id:
            return str(uuid.uuid4())

        # OPTIONAL: validate format (e.g. must be a UUID)
        try:
            # this will raise ValueError if not valid UUID
            uuid_obj = uuid.UUID(client_request_id)
            normalized_id = str(uuid_obj)
        except ValueError:
            # invalid UUID format → ignore and generate our own
            return str(uuid.uuid4())

        # Check uniqueness in DB
        exists = (
            db.query(RequestLog)
            .filter(RequestLog.request_id == normalized_id)
            .first()
        )
        if exists:
            # Not unique → generate our own
            return str(uuid.uuid4())

        # Client ID is valid and unique → use it
        return normalized_id