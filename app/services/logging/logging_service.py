from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, Dict, Any 
import json 
import logging 

from app.models.request_log import RequestLog, ActionLog
from app.db.session import get_db_context
from app.core.config import settings

logger = logging.getLogger(__name__)


class LoggingService:
    
    @staticmethod
    def sanitize_data(data: Any, max_length: int = 500000) -> Optional[str]:
        if data is None:
            return None 
        
        try:
            if isinstance(data, (dict, list)):
                str_data = json.dumps(data, default=str)
            else:
                str_data = str(data)
            if len(str_data) > max_length:
                str_data = str_data[:max_length]
            return str_data
        except Exception as e:
            logger.warning(f"Error sanitizing data: {str(e)}")
            return "[ERROR SANITIZING DATA]"
        
    @staticmethod
    def create_request_log(
        request_id: str,
        method: str,
        url: str, 
        query_params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        body: Optional[str] = None 
    ) -> Optional[int]:
        server_name = getattr(settings, 'SERVER_NAME', None)
        api_version = getattr(settings, 'API_VERSION', '0.0.0')
        
        try:
            with get_db_context() as db:
                request_log = RequestLog(
                    request_id=request_id,
                    method=method,
                    url=url,
                    query_params=LoggingService.sanitize_data(query_params),
                    headers=LoggingService.sanitize_data(headers),
                    body=LoggingService.sanitize_data(body),
                    server_name=server_name,
                    api_version=api_version,
                    start_time=datetime.utcnow()
                )
                
                db.add(request_log)
                db.flush()
                request_log_id = request_log.id
                
                logger.debug(f"Created request log: {request_log_id}")
                return request_log_id
        except Exception as e:
            logger.error(f"Error creating request log: {str(e)}", exc_info=True)
            return None
        
    @staticmethod
    def update_request_log(
        request_id: str,
        status_code: Optional[int] = None, 
        response_body: Optional[str] = None,
        response_headers: Optional[Dict] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None
    ) -> bool:
        try:
            with get_db_context() as db:
                request_log = db.query(RequestLog).filter(
                    RequestLog.request_id == request_id
                ).first()
                
                if not request_log:
                    logger.warning(f"Request log not found: {request_id}")
                    return False 
                
                end_time = datetime.utcnow()
                duration = (end_time - request_log.start_time).total_seconds() * 1000
                
                request_log.end_time = end_time
                request_log.duration_ms = duration
                
                if status_code is not None:
                    request_log.status_code = status_code
                    request_log.is_error = 1 if status_code >= 400 else 0
                
                if response_body is not None:
                    request_log.response_body = LoggingService.sanitize_data(response_body)
                
                if error_message:
                    request_log.error_message = error_message
                    request_log.is_error = 1
                
                if error_traceback:
                    request_log.error_traceback = error_traceback
                
                logger.debug(f"Updated request log: {request_id}")
                return True
            
        except Exception as e:
            logger.error(f"Error updating request log {request_id}: {str(e)}", exc_info=True)
            return False 
        
    @staticmethod
    def create_action_log(
        request_id: str, 
        action_type: str, 
        action_name: str,
        module_name: Optional[str] = None,
        function_name: Optional[str] = None,
        line_number: Optional[int] = None,
        input_params: Optional[Dict] = None,
    ) -> Optional[int]:
        try:
            with get_db_context() as db:
                request_log = db.query(RequestLog).filter(
                    RequestLog.request_id == request_id
                ).first()
                
                if not request_log:
                    logger.warning(f"Request log not found for action: {request_id}")
                    return None
                
                action_log = ActionLog(
                    request_id=request_log.id,  # ✅ FIXED: Changed from request_log_id
                    action_type=action_type,
                    action_name=action_name,
                    module_name=module_name,
                    function_name=function_name,
                    line_number=line_number,
                    input_params=LoggingService.sanitize_data(input_params),
                    start_time=datetime.utcnow()
                )
                
                db.add(action_log)
                db.flush()
                action_log_id = action_log.id
                
                logger.debug(f"Created action log: {action_log_id} for request: {request_id}")
                return action_log_id
        except Exception as e:
            logger.error(f"Error creating action log for request {request_id}: {str(e)}", exc_info=True)
            return None 
        
    @staticmethod
    def update_action_log(
        action_log_id: int,
        output_result: Optional[Any] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_prompt_tokens: Optional[int] = None,
        llm_completion_tokens: Optional[int] = None,
        llm_total_tokens: Optional[int] = None 
    ) -> bool:
        try:
            with get_db_context() as db:
                action_log = db.query(ActionLog).filter(
                    ActionLog.id == action_log_id
                ).first()
                
                if not action_log:
                    logger.warning(f"Action log not found: {action_log_id}")
                    return False
                
                end_time = datetime.utcnow()
                duration = (end_time - action_log.start_time).total_seconds() * 1000
                
                action_log.end_time = end_time
                action_log.duration_ms = duration
                
                if output_result is not None:
                    action_log.output_results = LoggingService.sanitize_data(output_result)  # ✅ FIXED: Changed to output_results (plural)
                
                if error_message:
                    action_log.error_message = error_message
                    action_log.is_error = 1
                
                if error_traceback:
                    action_log.error_traceback = error_traceback
                
                if llm_provider:
                    action_log.llm_provider = llm_provider
                
                if llm_model:
                    action_log.llm_model = llm_model
                
                if llm_prompt_tokens is not None:
                    action_log.llm_prompt_tokens = llm_prompt_tokens
                
                if llm_completion_tokens is not None:
                    action_log.llm_completion_tokens = llm_completion_tokens
                
                if llm_total_tokens is not None:
                    action_log.llm_total_tokens = llm_total_tokens
                
                logger.debug(f"Updated action log: {action_log_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating action log {action_log_id}: {str(e)}", exc_info=True)
            return False