"""
Configuration management for the RAG system.
"""
from typing import Literal, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # App Information
    app_title: str = "Chat With Your Docs"
    app_version: str = "0.1.0"

    # Local LLM Configuration
    use_local_llm: bool = True
    local_llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_llm_device: str = "auto"
    local_llm_max_length: int = 4096
    local_llm_temperature: float = 0.1
    local_llm_top_p: float = 0.9
    local_llm_top_k: int = 25
    local_llm_repetition_penalty: float = 1.1

    # Prompt Settings
    system_prompt: str = "You are an enterprise AI assistant designed to answer questions using retrieved document content. Prioritize factual accuracy, reference relevant context when possible, and maintain a clear, structured response. If the answer cannot be derived from the documents, explicitly state the limitation."

    # Embedding Model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    embedding_dimension: int = 384

    # Vector Database - Milvus
    milvus_db_path: str = "./data/milvus_lite.db"
    milvus_collection_name: str = "documents"

    # File Storage
    file_storage_dir: str = "./data/files"

    # Document Processing
    upload_dir: str = "./data/uploads"
    max_file_size_mb: int = 50
    allowed_extensions: Union[list[str], str] = [".pdf", ".txt", ".md"]

    # Chunking Strategy
    chunking_strategy: Literal["recursive",
                               "semantic", "paragraph"] = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k_retrieval: int = 3
    similarity_threshold: float = 0.7
    use_reranking: bool = True

    # Conversation
    max_conversation_history: int = 10
    session_timeout_minutes: int = 30

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: Union[list[str], str] = [
        "http://localhost:3000", "http://localhost:8000"]

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    @field_validator('allowed_extensions', mode='before')
    @classmethod
    def parse_allowed_extensions(cls, v):
        """Parse comma-separated string into list."""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(',')]
        return v

    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse comma-separated string into list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v

    @property
    def local_llm_device_detected(self) -> str:
        """Detect best available device if set to 'auto' or 'cpu' fallback."""
        if self.local_llm_device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.local_llm_device

    @property
    def upload_path(self) -> Path:
        """Get upload directory as Path object."""
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def file_storage_path(self) -> Path:
        """Get file storage directory as Path object."""
        path = Path(self.file_storage_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global settings instance
settings = Settings()
