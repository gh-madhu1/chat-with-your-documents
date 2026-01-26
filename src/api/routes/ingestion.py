"""
Document ingestion API routes.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pathlib import Path
import uuid
import shutil
from typing import List
from src.api.models import DocumentUploadResponse, DocumentInfo
from src.config import settings
from src.core.document_processor import DocumentProcessor
from src.core.chunking_strategies import ChunkingStrategyFactory
from src.core.embeddings import get_embedding_model
from src.core.vector_store import get_vector_store
from src.core.callbacks import CallbackManager, PipelineStage
from src.core.observability import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["ingestion"])

# Global state for tracking jobs
ingestion_jobs = {}


async def process_document_background(
    file_path: Path,
    doc_id: str,
    job_id: str,
    original_filename: str = None
):
    """Background task to process document."""
    callback_manager = CallbackManager()

    try:
        # Stage 1: Document processing
        await callback_manager.on_stage_start(
            PipelineStage.INGESTION,
            {"file": str(file_path)}
        )

        processor = DocumentProcessor()
        document = await processor.process_file(file_path, doc_id, original_filename)

        await callback_manager.on_stage_complete(
            PipelineStage.INGESTION,
            {"content_length": len(document.content)}
        )

        # Stage 2: Chunking
        await callback_manager.on_stage_start(
            PipelineStage.CHUNKING,
            {"strategy": settings.chunking_strategy}
        )

        chunker = ChunkingStrategyFactory.create(
            settings.chunking_strategy,
            settings.chunk_size,
            settings.chunk_overlap
        )

        chunks = chunker.chunk_text(
            document.content, document.metadata, document.page_mapping)

        await callback_manager.on_stage_complete(
            PipelineStage.CHUNKING,
            {"num_chunks": len(chunks)}
        )

        # Stage 3: Embedding
        await callback_manager.on_stage_start(
            PipelineStage.EMBEDDING,
            {"num_chunks": len(chunks)}
        )

        embedding_model = get_embedding_model()
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = embedding_model.embed_texts(chunk_texts)

        await callback_manager.on_stage_complete(
            PipelineStage.EMBEDDING,
            {"num_embeddings": len(embeddings)}
        )

        # Stage 4: Store in vector database
        vector_store = get_vector_store()

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        await vector_store.add_documents(
            chunks=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=chunk_ids
        )

        ingestion_jobs[job_id] = {
            "status": "completed",
            "doc_id": doc_id,
            "num_chunks": len(chunks),
            "message": "Document processed successfully"
        }

        logger.info(
            "document_ingestion_completed",
            job_id=job_id,
            doc_id=doc_id,
            num_chunks=len(chunks)
        )

    except Exception as e:
        ingestion_jobs[job_id] = {
            "status": "failed",
            "error": str(e),
            "message": f"Failed to process document: {str(e)}"
        }
        logger.error("document_ingestion_failed", job_id=job_id, error=str(e))
        await callback_manager.on_stage_error(PipelineStage.ERROR, e)


@router.post("/ingest", response_model=DocumentUploadResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload and process a document.

    The document will be processed in the background:
    1. Extract text and metadata
    2. Chunk the content
    3. Generate embeddings
    4. Store in vector database
    """
    # Validate file
    file_suffix = Path(file.filename).suffix.lower()
    if file_suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {settings.allowed_extensions}"
        )

    # Generate IDs
    job_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    # Save uploaded file
    file_path = settings.upload_path / f"{doc_id}{file_suffix}"

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save file: {str(e)}")

    # Start background processing
    ingestion_jobs[job_id] = {
        "status": "processing",
        "doc_id": doc_id,
        "file_name": file.filename,
        "message": "Processing document..."
    }

    background_tasks.add_task(
        process_document_background,
        file_path,
        doc_id,
        job_id,
        file.filename
    )

    logger.info("document_upload_started",
                job_id=job_id, file_name=file.filename)

    return DocumentUploadResponse(
        job_id=job_id,
        file_name=file.filename,
        status="processing",
        message="Document upload started. Processing in background."
    )


@router.get("/ingest/status/{job_id}")
async def get_ingestion_status(job_id: str):
    """Get the status of a document ingestion job."""
    if job_id not in ingestion_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return ingestion_jobs[job_id]


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all indexed documents."""
    vector_store = get_vector_store()
    documents = await vector_store.list_documents()

    return [
        DocumentInfo(
            doc_id=doc["doc_id"],
            file_name=doc["file_name"],
            file_type=doc["file_type"],
            pages=doc.get("pages"),
            total_chunks=doc.get("total_chunks", 0)
        )
        for doc in documents
    ]


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the vector store and file storage."""
    from src.core.file_storage import get_file_storage

    # Delete from vector store
    vector_store = get_vector_store()
    await vector_store.delete_document(doc_id)

    # Delete stored files
    file_storage = get_file_storage()
    file_storage.delete_file(doc_id)

    logger.info("document_deleted_via_api", doc_id=doc_id)

    return {"message": "Document deleted successfully", "doc_id": doc_id}
