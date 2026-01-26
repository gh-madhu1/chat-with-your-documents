"""
Server-Sent Events streaming routes.
"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import json
import asyncio
from typing import Optional
from src.core.rag_pipeline import get_rag_pipeline
from src.core.callbacks import CallbackManager
from src.core.observability import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/stream", tags=["streaming"])


@router.get("/query")
async def stream_query(question: str, session_id: str = None, top_k: int = None):
    """
    Stream query processing with real-time pipeline updates.

    Returns Server-Sent Events with:
    - Pipeline stage updates (retrieval, inference, etc.)
    - Streaming answer tokens

    Args:
        question: User query question
        session_id: Optional session ID for conversation context
        top_k: Number of results to retrieve (default from settings)
    """
    logger.info("stream_query_started",
                question=question[:100], session_id=session_id, top_k=top_k)

    async def event_generator():
        """Generate SSE events in proper format."""
        try:
            # Create callback manager
            callback_manager = CallbackManager()

            # Get RAG pipeline
            rag_pipeline = get_rag_pipeline()

            # Stream response chunks with sources
            chunk_count = 0
            sources_sent = False

            async for chunk, sources in rag_pipeline.stream_query(
                question=question,
                session_id=session_id,
                callback_manager=callback_manager,
                top_k=top_k
            ):
                # Send sources once at the start
                if sources is not None and not sources_sent:
                    sources_data = [
                        {
                            "source_id": s.get("source_id", ""),
                            "file_name": s.get("file_name", "Unknown"),
                            "page": s.get("page", "N/A"),
                            "chunk_index": s.get("chunk_index", 0),
                            "score": s.get("score", 0.0),
                            "citation": s.get("citation", "")
                        }
                        for s in sources
                    ]
                    event_data = json.dumps({"sources": sources_data})
                    yield f"event: sources\ndata: {event_data}\n\n"
                    sources_sent = True

                # Send answer chunks
                if chunk and chunk != "__SOURCES__":
                    event_data = json.dumps({"chunk": chunk})
                    yield f"event: answer\ndata: {event_data}\n\n"
                    chunk_count += 1
                    logger.debug("chunk_sent", chunk_len=len(
                        chunk), total_chunks=chunk_count)

            # Send completion event
            logger.info("stream_query_completed", total_chunks=chunk_count)
            yield "event: complete\ndata: {\"message\": \"Query completed\"}\n\n"

        except Exception as e:
            logger.error("stream_query_error", error=str(e),
                         error_type=type(e).__name__)
            error_data = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
