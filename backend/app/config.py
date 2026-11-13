import os
from pydantic import BaseModel

class Settings(BaseModel):
    APP_NAME: str = "SemanticSearch"
    APP_VERSION: str = "1.0.0"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./semanticsearch.db")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1")
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 80
    DEFAULT_TOP_K: int = 5
    CORS_ORIGINS: list[str] = ["*"]

settings = Settings()
