from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Hunyuan3D Serverless API"
    
    # Model Settings
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/runpod-volume/models")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "/runpod-volume/outputs")
    
    # Redis Settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    # R2/S3 Storage Settings
    R2_ACCOUNT_ID: Optional[str] = os.getenv("R2_ACCOUNT_ID")
    R2_ACCESS_KEY_ID: Optional[str] = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET_ACCESS_KEY: Optional[str] = os.getenv("R2_SECRET_ACCESS_KEY")
    R2_BUCKET_NAME: Optional[str] = os.getenv("R2_BUCKET_NAME")
    R2_PUBLIC_URL: Optional[AnyHttpUrl] = os.getenv("R2_PUBLIC_URL")
    
    # Cleanup Settings
    CLEANUP_FILES_AFTER: str = os.getenv("CLEANUP_FILES_AFTER", "1h")
    MAX_OUTPUT_FILES: int = int(os.getenv("MAX_OUTPUT_FILES", "1000"))
    
    class Config:
        case_sensitive = True
        env_file = ".env"

# Global settings instance
settings = Settings()
