"""
Embedding generation and management.
"""
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from src.config import settings
from src.core.observability import get_logger, track_metrics

logger = get_logger(__name__)


class EmbeddingModel:
    """Manages embedding model and generation."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize embedding model."""
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        logger.info(
            "loading_embedding_model",
            model=settings.embedding_model,
            device=settings.embedding_device
        )
        
        self._model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device
        )
        
        # Get embedding dimension
        self.embedding_dim = self._model.get_sentence_embedding_dimension()
        
        logger.info(
            "embedding_model_loaded",
            model=settings.embedding_model,
            dimension=self.embedding_dim,
            device=settings.embedding_device
        )
    
    @track_metrics("embedding_generation")
    def embed_texts(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing (default from settings)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        batch_size = batch_size or settings.embedding_batch_size
        
        logger.info(
            "generating_embeddings",
            num_texts=len(texts),
            batch_size=batch_size
        )
        
        # Generate embeddings
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        # Convert to list of lists
        embeddings_list = embeddings.tolist()
        
        logger.info(
            "embeddings_generated",
            num_embeddings=len(embeddings_list),
            dimension=len(embeddings_list[0]) if embeddings_list else 0
        )
        
        return embeddings_list
    
    @track_metrics("single_embedding")
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        logger.debug("generating_query_embedding", query_length=len(query))
        
        embedding = self._model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embedding.tolist()
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim


# Global embedding model instance
def get_embedding_model() -> EmbeddingModel:
    """Get the global embedding model instance."""
    return EmbeddingModel()
