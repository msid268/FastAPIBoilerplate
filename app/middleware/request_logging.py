# app/middleware/request_logging.py
import json
import logging
import uuid
from typing import Dict, Any, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.request_context import current_request_id
from app.services.logging.logging_service import LoggingService

logger = logging.getLogger(__name__)

SENSITIVE_HEADERS = {"authorization", "cookie", "set-cookie"}

def _scrub_headers(h: Dict[str, Any]) -> Dict[str, Any]:
    return {k: ("<redacted>" if k.lower() in SENSITIVE_HEADERS else v) for k, v in h.items()}

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 1) Make / attach correlation id
        rid = str(uuid.uuid4())
        request.state.request_id = rid
        token = current_request_id.set(rid)

        # 2) Best-effort capture of incoming request data (non-blocking)
        method = request.method
        url = str(request.url)
        query_params = dict(request.query_params)
        headers = _scrub_headers(dict(request.headers))

        body_text: Optional[str] = None
        try:
            # reading body: make a copy then replace the stream so downstream can still read
            body_bytes = await request.body()
            if body_bytes:
                body_text = body_bytes.decode(errors="replace")
            async def receive():
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            request._receive = receive  # type: ignore[attr-defined]
        except Exception as e:
            logger.debug(f"Could not read request body for {rid}: {e}")

        # 3) Create the RequestLog row
        LoggingService.create_request_log(
            request_id=rid,
            method=method,
            url=url,
            query_params=query_params or None,
            headers=headers or None,
            body=body_text,
        )

        # 4) Call downstream and capture outcome
        status_code = None
        error_message = None
        error_traceback = None

        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            # Ensure we mark the request as error
            status_code = 500
            error_message = str(e)
            import traceback as _tb
            error_traceback = _tb.format_exc()
            raise
        finally:
            try:
                LoggingService.update_request_log(
                    request_id=rid,
                    status_code=status_code,
                    error_message=error_message,
                    error_traceback=error_traceback,
                )
            except Exception as log_err:
                logger.error(f"Failed updating request log for {rid}: {log_err}", exc_info=True)
            finally:
                # Always reset the context var
                current_request_id.reset(token)
