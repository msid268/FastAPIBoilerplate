from fastapi import APIRouter, Depends, HTTPException, Query 
from sqlalchemy.orm import Session 
from sqlalchemy import desc
from typing import Optional, List 
from datetime import datetime, timedelta 
import logging 

from app.db.session import get_db 
from app.models.request_log import ActionLog, RequestLog
from pydantic import BaseModel 


logger = logging.getLogger(__name__)
router = APIRouter()

# Response Schemas
class ActionLogResponse(BaseModel):
    id: int
    action_type:str
    action_name: str 
    module_name: Optional[str] 
    function_name: Optional[str]
    start_time: datetime
    end_time:  Optional[datetime] 
    duration_ms: Optional[float]
    is_error: int
    error_message: Optional[str]
    input_params: Optional[dict]
    output_result: Optional[dict]
    
    class Config:
        from_attributes = True 
    
class RequestLogResponse(BaseModel):
    id: int
    request_id: str
    method: str
    url: str
    status_code: Optional[int]
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[float]
    is_error: int
    client_host: Optional[str]
    user_id: Optional[int]
    
    class Config:
        from_attributes = True

class RequestLogDetailResponse(RequestLogResponse):
    url: str
    query_params: Optional[dict]
    headers: Optional[dict]
    body: Optional[str]
    response_body: Optional[str]
    error_message: Optional[str]
    action_logs: List[ActionLogResponse]
    
    class Config:
        from_attributes = True


@router.get("/requests", response_model=List[RequestLogResponse])
def get_request_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    method: Optional[str] = Query(None),
    url: Optional[str] = Query(None),
    status_code: Optional[int] = Query(None),
    is_error: Optional[int] = Query(None),
    hours_ago: Optional[int] = Query(24, ge=1, le=720),
    db: Session = Depends(get_db)
):
    logger.info("Fetching request logs")
    try:
        query = db.query(RequestLog)
        
        # Apply filters
        if method:
            query = query.filter(RequestLog.method == method.upper())
        
        if url:
            query = query.filter(RequestLog.url.like(f"%{url}%"))
        
        if status_code:
            query = query.filter(RequestLog.status_code == status_code)
        
        if is_error is not None:
            query = query.filter(RequestLog.is_error == (1 if is_error else 0))
        
        # Filter by time range
        if hours_ago:
            time_threshold = datetime.utcnow() - timedelta(hours=hours_ago)
            query = query.filter(RequestLog.start_time >= time_threshold)
        
        # Order by most recent first
        query = query.order_by(desc(RequestLog.start_time))
        
        # Pagination
        total = query.count()
        logs = query.offset(skip).limit(limit).all()
        
        logger.info(f"Retrieved {len(logs)} request logs")
        return logs
        
    except Exception as e:
        logger.error(f"Error fetching request logs: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.get("/requests/{request_id}", response_model=RequestLogDetailResponse)
def get_request_log_detail(
    request_id: str,
    db: Session = Depends(get_db)
):
    """Get detailed request log including all action logs."""
    logger.info(f"Fetching request log detail: {request_id}")
    
    try:
        request_log = db.query(RequestLog).filter(
            RequestLog.request_id == request_id
        ).first()
        
        if not request_log:
            raise HTTPException(
                status_code=404,
                detail="Request log not found"
            )
        
        logger.info(f"Retrieved request log with {len(request_log.action_logs)} actions")
        return request_log
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching request log detail: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@router.get("/actions", response_model=List[ActionLogResponse])
def get_action_logs(
    request_id: Optional[str] = Query(None),
    action_type: Optional[str] = Query(None),
    is_error: Optional[bool] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get list of action logs with filters."""
    logger.info("Fetching action logs")
    
    try:
        query = db.query(ActionLog)
        
        # Apply filters
        if request_id:
            request_log = db.query(RequestLog).filter(
                RequestLog.request_id == request_id
            ).first()
            if request_log:
                query = query.filter(ActionLog.request_log_id == request_log.id)
        
        if action_type:
            query = query.filter(ActionLog.action_type == action_type)
        
        if is_error is not None:
            query = query.filter(ActionLog.is_error == (1 if is_error else 0))
        
        # Order by most recent first
        query = query.order_by(desc(ActionLog.start_time))
        
        # Pagination
        logs = query.offset(skip).limit(limit).all()
        
        logger.info(f"Retrieved {len(logs)} action logs")
        return logs
        
    except Exception as e:
        logger.error(f"Error fetching action logs: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@router.get("/stats")
def get_log_stats(
    hours_ago: int = Query(24, ge=1, le=720),
    db: Session = Depends(get_db)
):
    """Get statistics about logged requests."""
    logger.info("Fetching log statistics")
    
    try:
        time_threshold = datetime.utcnow() - timedelta(hours=hours_ago)
        
        # Total requests
        total_requests = db.query(RequestLog).filter(
            RequestLog.start_time >= time_threshold
        ).count()
        
        # Error requests
        error_requests = db.query(RequestLog).filter(
            RequestLog.start_time >= time_threshold,
            RequestLog.is_error == 1
        ).count()
        
        # Average response time
        avg_duration = db.query(
            func.avg(RequestLog.duration_ms)
        ).filter(
            RequestLog.start_time >= time_threshold,
            RequestLog.duration_ms.isnot(None)
        ).scalar() or 0
        
        # Requests by method
        from sqlalchemy import func
        methods = db.query(
            RequestLog.method,
            func.count(RequestLog.id).label('count')
        ).filter(
            RequestLog.start_time >= time_threshold
        ).group_by(RequestLog.method).all()
        
        # Requests by status code
        status_codes = db.query(
            RequestLog.status_code,
            func.count(RequestLog.id).label('count')
        ).filter(
            RequestLog.start_time >= time_threshold,
            RequestLog.status_code.isnot(None)
        ).group_by(RequestLog.status_code).all()
        
        # Slowest endpoints
        slowest = db.query(
            RequestLog.url,
            func.avg(RequestLog.duration_ms).label('avg_duration')
        ).filter(
            RequestLog.start_time >= time_threshold,
            RequestLog.duration_ms.isnot(None)
        ).group_by(RequestLog.url).order_by(
            desc('avg_duration')
        ).limit(10).all()
        
        stats = {
            "time_range_hours": hours_ago,
            "total_requests": total_requests,
            "error_requests": error_requests,
            "error_rate": (error_requests / total_requests * 100) if total_requests > 0 else 0,
            "average_duration_ms": round(avg_duration, 2),
            "methods": {method: count for method, count in methods},
            "status_codes": {str(code): count for code, count in status_codes if code},
            "slowest_endpoints": [
                {"url": url, "avg_duration_ms": round(duration, 2)}
                for url, duration in slowest
            ]
        }
        
        logger.info("Log statistics retrieved successfully")
        return stats
        
    except Exception as e:
        logger.error(f"Error fetching log stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )