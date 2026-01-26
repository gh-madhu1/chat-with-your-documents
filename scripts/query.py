"""
Query and conversation API routes.
"""
from fastapi import APIRouter, HTTPException
from src.api.models import (
    QueryRequest,
    QueryResponse,
    ConversationRequest,
    ConversationResponse,
    ConversationHistory,
    SourceReference
)
from src.core.rag_pipeline import get_rag_pipeline
from src.core.conversation_manager import get_conversation_manager
from src.core.observability import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Ask a question over indexed documents (single-turn).
    
    This endpoint performs:
    1. Retrieval of relevant context
    2. LLM inference with grounding
    3. Citation extraction
    """
    logger.info("query_request_received", question=request.question[:100])
    
    try:
        rag_pipeline = get_rag_pipeline()
        
        response = await rag_pipeline.query(
            question=request.question,
            session_id=None  # Single-turn, no session
        )
        
        # Convert sources to SourceReference models
        sources = [
            SourceReference(
                source_id=src["source_id"],
                file_name=src["file_name"],
                page=src["page"],
                chunk_index=src["chunk_index"],
                score=src["score"],
                citation=src["citation"]
            )
            for src in response.sources
        ]
        
        return QueryResponse(
            answer=response.answer,
            sources=sources,
            confidence=response.confidence,
            metadata=response.metadata
        )
    
    except Exception as e:
        logger.error("query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/conversation", response_model=ConversationResponse)
async def conversation_query(request: ConversationRequest):
    """
    Ask a question in a multi-turn conversation.
    
    Maintains conversation history and context across multiple turns.
    If no session_id is provided, a new session will be created.
    """
    logger.info(
        "conversation_request_received",
        question=request.question[:100],
        session_id=request.session_id
    )
    
    try:
        conversation_manager = get_conversation_manager()
        rag_pipeline = get_rag_pipeline()
        
        # Create or get session
        session_id = request.session_id
        if not session_id:
            session_id = conversation_manager.create_session()
            logger.info("new_conversation_session_created", session_id=session_id)
        else:
            # Verify session exists
            if not conversation_manager.get_session(session_id):
                raise HTTPException(status_code=404, detail="Session not found or expired")
        
        # Process query with conversation context
        response = await rag_pipeline.query(
            question=request.question,
            session_id=session_id
        )
        
        # Convert sources
        sources = [
            SourceReference(
                source_id=src["source_id"],
                file_name=src["file_name"],
                page=src["page"],
                chunk_index=src["chunk_index"],
                score=src["score"],
                citation=src["citation"]
            )
            for src in response.sources
        ]
        
        return ConversationResponse(
            answer=response.answer,
            sources=sources,
            confidence=response.confidence,
            session_id=session_id,
            metadata=response.metadata
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("conversation_query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Conversation query failed: {str(e)}")


@router.get("/conversation/{session_id}", response_model=ConversationHistory)
async def get_conversation_history(session_id: str):
    """Get the history of a conversation session."""
    conversation_manager = get_conversation_manager()
    
    conversation = conversation_manager.get_session(session_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    messages = [
        {
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
            "metadata": msg.metadata
        }
        for msg in conversation.messages
    ]
    
    return ConversationHistory(
        session_id=session_id,
        created_at=conversation.created_at.isoformat(),
        last_updated=conversation.last_updated.isoformat(),
        messages=messages
    )


@router.delete("/conversation/{session_id}")
async def delete_conversation(session_id: str):
    """Delete a conversation session."""
    conversation_manager = get_conversation_manager()
    conversation_manager.delete_session(session_id)
    
    logger.info("conversation_session_deleted_via_api", session_id=session_id)
    
    return {"message": "Conversation session deleted", "session_id": session_id}


@router.get("/conversations")
async def list_conversations():
    """List all active conversation sessions."""
    conversation_manager = get_conversation_manager()
    return conversation_manager.list_sessions()
