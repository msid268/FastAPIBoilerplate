# from fastapi import APIRouter, Depends 
# from sqlalchemy.orm import Session 
# from sqlalchemy import text 
# import logging 

# from app.db.session import get_db 
# from app.core.config import settings 

# logger = logging.getLogger(__name__)
# router = APIRouter()


# @router.get("/health")
# def health_check():
#     """Basic health check endpoint"""
#     logging.debug("Health Check Called.")
#     return {
#         "status": "healthy",
#         "service": settings.APP_NAME,
#         "version": settings.APP_VERSION
#     }
    
# @router.get("/health/db")
# def health_check_db(db: Session = Depends(get_db))