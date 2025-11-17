import time 
import uuid 
import json 
import traceback 
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request 
from starlette.responses import Response, JSONResponse
from typing import Callable
import logging 

from app.services.logging.logging_service import LoggingService
from app.core.config import settings 

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        body = None 
        if request.method in ['POST', 'PUT', 'PATCH']:
            try:
                body_bytes = await request.body()
                if body_bytes:
                    body = body_bytes.decode('utf-8')
                async def recieve():
                    return {"type": "http.request", "body":body_bytes}
            except Exception as e:
                logger.warning(f"Could not read request_body: {str(e)}")
                
            query_params = dict(request.query_params) if request.query_params else None 
            headers = dict(request.headers)
            
            logger.info(
                f"Request started",
                extra = {
                    "extra_data":{
                        "request_id": request_id, 
                        "method":request.method,
                        "path": request.url.path,
                    }
                }
            )
            
            
            # create request log in database
            try:
                LoggingService.create_request_log(
                    request_id=request_id,
                    method=request.method,
                    url = str(request.url),
                    query_params=query_params,
                    headers=headers,
                    body=body
                )
            except Exception as e:
                logger.error(f"Failed to create request log: {str(3)}", exc_info=True)
            
            # process request
            response = None 
            error_occurred = None 
            error_message = None 
            error_trace = None 
            status_code = 200
            
            try:
                response = await call_next(request)
                status_code = response.status_code 
            except Exception as e:
                error_occured = True
                error_message = str(e)
                error_trace = traceback.format_exc()
                status_code = 200
                
                logger.error(
                "Request failed with exception",
                extra={"extra_data": {"request_id": request_id, "error": error_message}},
                exc_info=True
                )
                
                response = JSONResponse(
                    status_code=500,
                    content = {
                        "success": False,
                        "request_id": request_id,
                        "error": "Internal Server Error",
                        "message": error_message if settings.DEBUG else ""
                    }
                )
                
            process_time = (time.time() - start_time ) * 1000
            
            # Extract response body
            response_body = None 
            try:
                if hasattr(response, "body"):
                    response_body = response.body.decode('utf-8') if response.body else None
            except Exception:
                pass
            
            logger.info(
                "Request Completed",
                extra = {
                   "extra_data": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round(process_time, 2),
                    "error": error_occurred,
                }
                }
            )
            
            # Update Request Log in database
            
            try:
                LoggingService.update_request_log(
                    request_id=request_id,
                    status_code=status_code,
                    response_body=response_body,
                    response_headers=dict(response.headers),
                    error_message=error_message,
                    error_traceback=error_trace,
                )
            except Exception as e:
                logger.error(f"Failed to update request log: {str(e)}", exc_info=True)
            
            return response