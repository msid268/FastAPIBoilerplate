# app/middleware/request_logging.py

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
from app.core.request_context import current_request_id  # <-- make sure this exists


def get_request_id(request: Request):
    """
    Helper to get the DB PK of the RequestLog row (if you need it elsewhere).
    """
    if not hasattr(request.state, "request_db_id"):
        print("⚠️ MISSING request.state.request_db_id")
        return None
    return request.state.request_db_id


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        db: Session = SessionLocal()

        py_start_time = time.perf_counter()
        utc_start = datetime.utcnow()

        # Generate a unique request_id (string, 36 chars)
        request_uuid = str(uuid.uuid4())

        # Propagate to ContextVar so decorators can read it even without Request
        token = current_request_id.set(request_uuid)

        # Read request body once
        body_bytes = await request.body()
        try:
            body_text = body_bytes.decode("utf-8")
        except Exception:
            body_text = "<binary>"

        # Re-inject body so downstream handlers can still read it
        async def receive():
            return {"type": "http.request", "body": body_bytes}

        request._receive = receive

        # Serialize headers & query params
        headers_dict = dict(request.headers)
        query_dict = dict(request.query_params)

        headers_str = json.dumps(headers_dict, ensure_ascii=False)
        query_str = json.dumps(query_dict, ensure_ascii=False)

        req_log = None
        try:
            # 1) Create RequestLog row
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

            # Attach identifiers to request.state so actions/decorators can use them
            request.state.request_id = request_uuid            # <-- what decorator looks for
            request.state.request_db_id = req_log.id           # PK in DB
            request.state.request_public_id = request_uuid     # NVARCHAR(36)

            # 2) Call the actual endpoint
            response = await call_next(request)

            # 3) Capture the response body
            resp_body_bytes = b""
            async for chunk in response.body_iterator:
                resp_body_bytes += chunk

            try:
                resp_text = resp_body_bytes.decode("utf-8")
            except Exception:
                resp_text = "<binary>"

            # 4) Update RequestLog with response info
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

            # 5) Rebuild the Response with the captured body
            return Response(
                content=resp_body_bytes,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        except Exception as exc:
            # If something explodes, we still try to log it
            db.rollback()  # important: clear failed transaction

            py_end_time = time.perf_counter()
            utc_end = datetime.utcnow()
            duration_ms = (py_end_time - py_start_time) * 1000.0

            tb = traceback.format_exc()

            if req_log is None:
                # Even if creation failed before, try again
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

            # Re-raise so FastAPI's exception handling still runs
            raise
        finally:
            db.close()
            # Reset the ContextVar so it doesn't leak to other requests
            try:
                current_request_id.reset(token)
            except Exception:
                # If something odd happens with token, don't crash the request
                pass
