from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.v1.router import api_router
from app.middleware.logging_middleware import RequestLoggingMiddleware

# setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    yield 
    
    # shutdown
    logger.info(f"Shutting down {settings.APP_NAME}")
    
# create FastAPI Application
app = FastAPI(
    title = settings.APP_NAME,
    version = settings.APP_VERSION,
    debug = settings.DEBUG,
    lifespan=lifespan,
    docs_url = "/docs" if settings.DEBUG else None,
    redoc_url = "/redoc" if settings.DEBUG else None 
)

# add cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS
)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)

@app.get("/")
def root():
    """Root endpoint."""
    logger.debug("Root endpoint accessed")
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": f"{settings.API_V1_PREFIX}/docs" if settings.DEBUG else "disabled"
    }
    
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=6710,
        reload=not settings.DEBUG,
        log_config=None,  # Use our custom logging
    )