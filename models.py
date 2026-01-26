"""
Pydantic models for API requests and responses.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# Document Ingestion Models
class DocumentUploadResponse(BaseModel):
    """Response from document upload."""
    job_id: str
    file_name: str
    status: str
    message: str


class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    doc_id: str
    file_name: str
    file_type: str
    pages: Optional[int] = None
    total_chunks: int
    indexed_at: Optional[str] = None


# Query Models
class QueryRequest(BaseModel):
    """Request for single-turn query."""
    question: str = Field(..., min_length=1, description="User question")
    top_k: Optional[int] = Field(
        None, ge=1, le=20, description="Number of sources to retrieve")
    use_multi_query: bool = Field(
        False, description="Use multi-query retrieval")


class SourceReference(BaseModel):
    """Source reference with citation."""
    source_id: int
    file_name: str
    page: Any  # Can be int or "N/A"
    chunk_index: int
    score: float
    citation: str


class QueryResponse(BaseModel):
    """Response from query."""
    answer: str
    sources: List[SourceReference]
    confidence: float
    metadata: Dict[str, Any] = {}


# Conversation Models
class ConversationRequest(BaseModel):
    """Request for multi-turn conversation."""
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    top_k: Optional[int] = Field(None, ge=1, le=20)


class ConversationResponse(BaseModel):
    """Response from conversation."""
    answer: str
    sources: List[SourceReference]
    confidence: float
    session_id: str
    metadata: Dict[str, Any] = {}


class ConversationHistory(BaseModel):
    """Conversation history."""
    session_id: str
    created_at: str
    last_updated: str
    messages: List[Dict[str, Any]]


# Pipeline Status Models
class PipelineStatus(BaseModel):
    """Pipeline stage status."""
    stage: str
    status: str  # 'started', 'progress', 'completed', 'error'
    message: str
    timestamp: str
    metadata: Dict[str, Any] = {}


# Health Check
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    vector_store_status: Dict[str, Any]
    timestamp: str
