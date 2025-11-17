
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
from functools import lru_cache
import os 

class Settings(BaseSettings):
    # -------------------------------- Application ------------------------------- #
    APP_NAME:str ="FASTAPI Boilerplate"
    APP_VERSION:str = "1.0.0"
    DEBUG:bool =False
    ENVIRONMENT:str = "development"
    SERVER_NAME:str = os.getenv("server_instance")
    
    # ------------------------------------ API ----------------------------------- #
    API_V1_PREFIX:str = "/api/v1 "
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # --------------------------------- Database --------------------------------- #
    DATABASE_SERVER:str =os.getenv("db_server_name")
    print(DATABASE_SERVER)
    DATABASE_USER:str = ""
    DATABASE_PASSWORD:str = ""
    DATABASE_NAME:str = ""
    DB_POOL_SIZE:int = 20
    DB_MAX_OVERFLOW:int = 10
    DB_POOL_TIMEOUT:int = 30
    DB_POOL_RECYCLE:int = 3600
    DB_ECHO:bool = False
    ODBC_DRIVER_VERSION:str = ""
    

    # ----------------------------------- CORS ----------------------------------- #
    CORS_ORIGINS: List[str] = ["http://localhost:5173"]
    CORS_ALLOW_CREDENTIALS:bool = True
    CORS_ALLOW_METHODS:List[str] = ["*"]
    CORS_ALLOW_HEADERS:List[str] = ["*"]
    
    # ---------------------------------- LOGGING --------------------------------- #
    LOG_LEVEL:str = "INFO"
    LOG_FORMAT:str = "json"
    LOG_FILE_PATH:str = "logs/app.log"
    LOG_MAX_SIZE:int =10
    LOG_BACKUP_COUNT:int =10
    
    # ---------------------------------- OPENAI ---------------------------------- #
    OPENAI_API_KEY:str=""
    OPENAI_MODEL:str=""
    OPENAI_MAX_TOKENS:int=8192
    OPENAI_TEMPERATURE:float=0.7
    
    # ---------------------------------- BEDROCK --------------------------------- #
    AWS_ACCESS_KEY_ID:str =""
    AWS_SECRET_ACCESS_KEY:str =""
    AWS_REGION:str ="us-east-1"
    BEDROCK_MODEL_ID:str =""
    BEDROCK_MAX_TOKENS:int=8192
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()