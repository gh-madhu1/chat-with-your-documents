"""
Vector store management using Milvus.
"""
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from dataclasses import dataclass
from src.config import settings
from src.core.observability import get_logger, track_metrics

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Represents a search result with metadata."""
    content: str
    metadata: dict
    score: float
    chunk_id: str

    def get_citation(self) -> str:
        """Format citation for this result."""
        file_name = self.metadata.get("file_name", "Unknown")
        page = self.metadata.get("page", "N/A")
        file_path = self.metadata.get("file_path", "")

        if file_path:
            return f"[{file_name} (Page {page})](file://{file_path})"
        return f"{file_name} (Page {page})"


class VectorStore:
    """Manages vector database operations with Milvus."""

    def __init__(self, collection_name: str = None):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the collection (default from settings)
        """
        self.collection_name = collection_name or settings.milvus_collection_name
        self._collection = None
        self._initialize()

    def _initialize(self):
        """Initialize Milvus connection and collection."""
        logger.info(
            "initializing_vector_store",
            mode="in-memory",
            collection=self.collection_name
        )

        # Use Milvus Lite for in-memory vector database
        try:
            # Connect using URI from settings
            connections.connect(
                alias="default",
                uri=settings.milvus_db_path
            )
            logger.info("connected_to_milvus", uri=settings.milvus_db_path)
        except Exception as e:
            logger.error("milvus_connection_failed", error=str(e))
            raise

        # Create collection if it doesn't exist
        if utility.has_collection(self.collection_name):
            self._collection = Collection(self.collection_name)
            logger.info(
                "vector_store_loaded",
                collection=self.collection_name,
                count=self._collection.num_entities
            )
        else:
            self._create_collection()

    def _create_collection(self):
        """Create a new Milvus collection with schema."""
        logger.info("creating_new_collection", collection=self.collection_name)

        # Define schema
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR,
                        is_primary=True, max_length=100),
            FieldSchema(name="content", dtype=DataType.VARCHAR,
                        max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,
                        dim=settings.embedding_dimension),
            # Metadata fields
            FieldSchema(name="file_name",
                        dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="file_path",
                        dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="file_type",
                        dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
        ]

        schema = CollectionSchema(
            fields=fields, description="Document chunks for RAG")

        # Create collection
        self._collection = Collection(
            name=self.collection_name,
            schema=schema
        )

        # Create index for vector field
        index_params = {
            "metric_type": "L2",  # Euclidean distance
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }

        self._collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        logger.info("collection_created", collection=self.collection_name)

    @track_metrics("vector_store_add")
    async def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ):
        """
        Add documents to the vector store.

        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
            ids: List of unique IDs for each chunk
        """
        if not chunks:
            logger.warning("no_chunks_to_add")
            return

        logger.info(
            "adding_documents_to_vector_store",
            num_documents=len(chunks),
            collection=self.collection_name
        )

        # Prepare data for insertion
        entities = [
            ids,  # chunk_id
            chunks,  # content
            embeddings,  # embedding
            [m.get("file_name", "unknown") for m in metadatas],
            [m.get("file_path", "") for m in metadatas],
            [m.get("file_type", "") for m in metadatas],
            [int(m.get("page", 1)) if m.get("page")
             is not None else 1 for m in metadatas],  # Ensure INT64
            [int(m.get("chunk_index", 0)) if m.get("chunk_index")
             is not None else 0 for m in metadatas],  # Ensure INT64
            [m.get("doc_id", "unknown") for m in metadatas],
        ]

        # Insert into Milvus
        self._collection.insert(entities)
        self._collection.flush()

        logger.info(
            "documents_added",
            num_documents=len(chunks),
            total_count=self._collection.num_entities
        )

    @track_metrics("vector_store_search")
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of SearchResult objects
        """
        top_k = top_k or settings.top_k_retrieval

        logger.info(
            "searching_vector_store",
            top_k=top_k,
            has_filter=filter_metadata is not None
        )

        # Load collection into memory
        self._collection.load()

        # Define search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }

        # Build filter expression if provided
        expr = None
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                if isinstance(value, str):
                    conditions.append(f'{key} == "{value}"')
                else:
                    conditions.append(f'{key} == {value}')
            expr = " && ".join(conditions)

        # Search
        results = self._collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["content", "file_name", "file_path",
                           "file_type", "page", "chunk_index", "doc_id"]
        )

        # Parse results
        search_results = []

        if results and len(results) > 0:
            for hit in results[0]:
                # Convert L2 distance squared to an intuitive similarity score
                # For normalized vectors, d^2 is in [0, 4]. 
                # We use a mapping that feels "generous" for good matches.
                score = max(0.0, 1.0 - float(hit.distance) / 6.0)

                metadata = {
                    "file_name": hit.entity.get("file_name"),
                    "file_path": hit.entity.get("file_path"),
                    "file_type": hit.entity.get("file_type"),
                    "page": hit.entity.get("page"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "doc_id": hit.entity.get("doc_id"),
                }

                search_results.append(SearchResult(
                    content=hit.entity.get("content"),
                    metadata=metadata,
                    score=score,
                    chunk_id=hit.id
                ))

        logger.info(
            "search_completed",
            num_results=len(search_results),
            top_score=search_results[0].score if search_results else 0
        )

        return search_results

    async def get_all_documents(self) -> List[SearchResult]:
        """
        Retrieve all documents from the collection for BM25 indexing.

        Returns:
            List of all SearchResult objects in the collection
        """
        logger.info("retrieving_all_documents")

        try:
            # Load collection into memory
            self._collection.load()

            # Query all documents - use expression that always matches
            query_results = self._collection.query(
                # Match all documents (chunk_index is always >= 0)
                expr="chunk_index >= 0",
                output_fields=["content", "file_name", "file_path",
                               "file_type", "page", "chunk_index", "doc_id"]
            )

            # Parse results into SearchResult objects
            all_documents = []

            if query_results:
                for entity in query_results:
                    metadata = {
                        "file_name": entity.get("file_name"),
                        "file_path": entity.get("file_path"),
                        "file_type": entity.get("file_type"),
                        "page": entity.get("page"),
                        "chunk_index": entity.get("chunk_index"),
                        "doc_id": entity.get("doc_id"),
                    }

                    all_documents.append(SearchResult(
                        content=entity.get("content"),
                        metadata=metadata,
                        score=1.0,  # Default score for retrieved documents
                        chunk_id=str(entity.get("pk", ""))
                    ))

            logger.info(
                "all_documents_retrieved",
                total_count=len(all_documents)
            )

            return all_documents
        except Exception as e:
            logger.warning(
                "get_all_documents_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            # Return empty list on failure - BM25 will be skipped gracefully
            return []

    async def delete_document(self, doc_id: str):
        """
        Delete all chunks belonging to a document.

        Args:
            doc_id: Document ID to delete
        """
        logger.info("deleting_document", doc_id=doc_id)

        # Load collection to ensure it's ready for deletion
        self._collection.load()

        # Delete by expression
        expr = f'doc_id == "{doc_id}"'
        self._collection.delete(expr)
        self._collection.flush()

        logger.info("document_deleted", doc_id=doc_id)

    async def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all unique documents in the store.

        Returns:
            List of document metadata
        """
        # Query all documents
        self._collection.load()

        results = self._collection.query(
            expr="chunk_index >= 0",
            output_fields=["doc_id", "file_name", "file_type", "page"],
            limit=10000
        )

        # Extract unique documents
        doc_map = {}
        for result in results:
            doc_id = result.get("doc_id")
            if doc_id and doc_id not in doc_map:
                doc_map[doc_id] = {
                    "doc_id": doc_id,
                    "file_name": result.get("file_name"),
                    "file_type": result.get("file_type"),
                    "total_chunks": 0  # Will be counted
                }
            if doc_id:
                doc_map[doc_id]["total_chunks"] += 1

        return list(doc_map.values())

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self._collection.num_entities if self._collection else 0

            return {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "db_path": settings.milvus_db_path
            }
        except Exception as e:
            logger.error("get_collection_stats_error", error=str(e))
            return {
                "collection_name": self.collection_name,
                "total_chunks": 0,
                "db_path": settings.milvus_db_path,
                "error": str(e)
            }

    def reset_collection(self):
        """Delete all documents in the collection."""
        logger.warning("resetting_collection", collection=self.collection_name)

        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

        self._create_collection()
        logger.info("collection_reset", collection=self.collection_name)


# Global vector store instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """Get the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
