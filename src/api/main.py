"""
FastAPI application setup.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from src.config import settings
from src.core.observability import setup_logging, get_logger
from src.core.vector_store import get_vector_store
from src.core.embeddings import get_embedding_model
from src.api.routes import ingestion, query, streaming
from src.api.models import HealthResponse

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("application_starting")

    # Initialize vector store
    vector_store = get_vector_store()
    logger.info("vector_store_initialized")

    # Pre-load Embedding Model
    get_embedding_model()
    logger.info("embedding_model_preloaded")

    # Pre-load Local LLM
    if settings.use_local_llm:
        from src.core.local_llm import get_local_llm
        get_local_llm()
        logger.info("local_llm_preloaded")

    yield

    # Shutdown
    logger.info("application_shutting_down")


# Create FastAPI app
app = FastAPI(
    title=settings.app_title,
    description="Production-grade RAG system for document Q&A",
    version=settings.app_version,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(ingestion.router)
app.include_router(query.router)
app.include_router(streaming.router)

# Mount static files
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serve the web UI."""
    static_file = static_dir / "ui" / "index.html"
    if static_file.exists():
        return FileResponse(static_file)
    return {
        "message": f"{settings.app_title} API",
        "version": settings.app_version,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_collection_stats()
    except Exception as e:
        logger.error("health_check_vector_store_error", error=str(e))
        stats = {
            "collection_name": settings.milvus_collection_name,
            "total_chunks": 0,
            "db_path": settings.milvus_db_path,
            "status": "error"
        }

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        vector_store_status=stats,
        timestamp=datetime.now().isoformat()
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
