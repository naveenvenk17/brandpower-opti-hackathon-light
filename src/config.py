"""
Simplified production configuration with environment variables.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Server configuration
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8010)
    WORKERS: int = Field(default=1)
    
    # Azure OpenAI (for agent)
    AZURE_OPENAI_API_KEY: Optional[str] = Field(default=None)
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(default=None)
    AZURE_OPENAI_DEPLOYMENT: Optional[str] = Field(default="gpt-4o")
    AZURE_OPENAI_API_VERSION: str = Field(default="2024-02-15-preview")
    
    # Paths
    DATA_DIR: str = Field(default="data")
    UPLOADS_DIR: str = Field(default="uploads")
    MODELS_DIR: str = Field(default="models")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    
    # Retry configuration
    RETRY_MAX_ATTEMPTS: int = Field(default=3)
    RETRY_WAIT_MIN: float = Field(default=1.0)
    RETRY_WAIT_MAX: float = Field(default=10.0)
    RETRY_MULTIPLIER: float = Field(default=2.0)
    
    class Config:
        env_file = ".env"
        env_prefix = "APP_"
        extra = "ignore"  # Ignore extra fields in .env file


# Singleton
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
