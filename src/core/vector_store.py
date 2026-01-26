"""
Vector store management supporting multiple providers (Milvus, FAISS).
"""
from typing import List, Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import shutil
from pathlib import Path

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import chromadb
from chromadb.config import Settings as ChromaSettings

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


class VectorStore(ABC):
    """Abstract base class for vector store providers."""

    @abstractmethod
    async def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ):
        """Add documents to the vector store."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    async def get_all_documents(self) -> List[SearchResult]:
        """Retrieve all documents from the collection."""
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str):
        """Delete all chunks belonging to a document."""
        pass

    @abstractmethod
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all unique documents in the store."""
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        pass

    @abstractmethod
    def reset_collection(self):
        """Delete all documents in the collection."""
        pass


class MilvusProvider(VectorStore):
    """Milvus implementation of the VectorStore."""

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or settings.milvus_collection_name
        self._collection = None
        self._initialize()

    def _initialize(self):
        """Initialize Milvus connection and collection."""
        logger.info(
            "initializing_milvus_provider",
            uri=settings.milvus_db_path,
            collection=self.collection_name
        )

        try:
            connections.connect(
                alias="default",
                uri=settings.milvus_db_path
            )
            logger.info("connected_to_milvus", uri=settings.milvus_db_path)
        except Exception as e:
            logger.error("milvus_connection_failed", error=str(e))
            raise

        if utility.has_collection(self.collection_name):
            self._collection = Collection(self.collection_name)
        else:
            self._create_collection()

    def _create_collection(self):
        """Create a new Milvus collection with schema."""
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR,
                        is_primary=True, max_length=100),
            FieldSchema(name="content", dtype=DataType.VARCHAR,
                        max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,
                        dim=settings.embedding_dimension),
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

        self._collection = Collection(
            name=self.collection_name,
            schema=schema
        )

        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }

        self._collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

    @track_metrics("vector_store_add")
    async def add_documents(self, chunks, embeddings, metadatas, ids):
        if not chunks:
            return

        entities = [
            ids,
            chunks,
            embeddings,
            [m.get("file_name", "unknown") for m in metadatas],
            [m.get("file_path", "") for m in metadatas],
            [m.get("file_type", "") for m in metadatas],
            [int(m.get("page", 1)) if m.get("page") is not None else 1 for m in metadatas],
            [int(m.get("chunk_index", 0)) if m.get("chunk_index") is not None else 0 for m in metadatas],
            [m.get("doc_id", "unknown") for m in metadatas],
        ]

        self._collection.insert(entities)
        self._collection.flush()

    @track_metrics("vector_store_search")
    async def search(self, query_embedding, top_k=None, filter_metadata=None):
        top_k = top_k or settings.top_k_retrieval
        self._collection.load()

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        expr = None
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                if isinstance(value, str):
                    conditions.append(f'{key} == "{value}"')
                else:
                    conditions.append(f'{key} == {value}')
            expr = " && ".join(conditions)

        results = self._collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["content", "file_name", "file_path",
                           "file_type", "page", "chunk_index", "doc_id"]
        )

        search_results = []
        if results and len(results) > 0:
            for hit in results[0]:
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
        return search_results

    async def get_all_documents(self):
        try:
            self._collection.load()
            query_results = self._collection.query(
                expr="chunk_index >= 0",
                output_fields=["content", "file_name", "file_path",
                               "file_type", "page", "chunk_index", "doc_id"]
            )
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
                        score=1.0,
                        chunk_id=str(entity.get("pk", ""))
                    ))
            return all_documents
        except Exception as e:
            logger.warning("get_all_documents_failed", error=str(e))
            return []

    async def delete_document(self, doc_id: str):
        self._collection.load()
        expr = f'doc_id == "{doc_id}"'
        self._collection.delete(expr)
        self._collection.flush()

    async def list_documents(self):
        self._collection.load()
        results = self._collection.query(
            expr="chunk_index >= 0",
            output_fields=["doc_id", "file_name", "file_type", "page"],
            limit=10000
        )
        doc_map = {}
        for result in results:
            doc_id = result.get("doc_id")
            if doc_id and doc_id not in doc_map:
                doc_map[doc_id] = {
                    "doc_id": doc_id,
                    "file_name": result.get("file_name"),
                    "file_type": result.get("file_type"),
                    "total_chunks": 0
                }
            if doc_id:
                doc_map[doc_id]["total_chunks"] += 1
        return list(doc_map.values())

    def get_collection_stats(self):
        count = self._collection.num_entities if self._collection else 0
        return {
            "provider": "milvus",
            "collection_name": self.collection_name,
            "total_chunks": count,
            "db_path": settings.milvus_db_path
        }

    def reset_collection(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
        self._create_collection()


class ChromaProvider(VectorStore):
    """ChromaDB implementation of the VectorStore."""

    def __init__(self, collection_name: str = "document_chunks"):
        self.db_path = settings.chroma_db_path
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and collection."""
        logger.info("initializing_chroma_provider", path=self.db_path)
        os.makedirs(self.db_path, exist_ok=True)
        
        self._client = chromadb.PersistentClient(path=self.db_path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "l2"}  # L2 to match Milvus/FAISS logic
        )
        logger.info("chroma_collection_initialized", name=self.collection_name)

    @track_metrics("vector_store_add")
    async def add_documents(self, chunks, embeddings, metadatas, ids):
        if not chunks:
            return

        # Prepare metadatas: ensure all values are simple types (str, int, float, bool)
        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            clean_metadatas.append(clean_meta)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=clean_metadatas
        )
        logger.info("chroma_documents_added", count=len(chunks))

    @track_metrics("vector_store_search")
    async def search(self, query_embedding, top_k=None, filter_metadata=None):
        top_k = top_k or settings.top_k_retrieval
        
        # Chroma where filter
        where = None
        if filter_metadata:
            if len(filter_metadata) > 1:
                where = {"$and": [{k: v} for k, v in filter_metadata.items()]}
            else:
                where = filter_metadata

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                # L2 distance conversion to score (simplified)
                dist = results["distances"][0][i]
                score = max(0.0, 1.0 - float(dist) / 6.0)
                
                search_results.append(SearchResult(
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    score=score,
                    chunk_id=results["ids"][0][i]
                ))
        
        return search_results

    async def get_all_documents(self):
        results = self._collection.get(include=["documents", "metadatas"])
        all_docs = []
        if results and results["ids"]:
            for i in range(len(results["ids"])):
                all_docs.append(SearchResult(
                    content=results["documents"][i],
                    metadata=results["metadatas"][i],
                    score=1.0,
                    chunk_id=results["ids"][i]
                ))
        return all_docs

    async def delete_document(self, doc_id: str):
        """Delete all chunks belonging to a document."""
        self._collection.delete(where={"doc_id": doc_id})
        logger.info("chroma_document_deleted", doc_id=doc_id)

    async def list_documents(self):
        results = self._collection.get(include=["metadatas"])
        doc_map = {}
        if results and results["metadatas"]:
            for meta in results["metadatas"]:
                d_id = meta.get("doc_id")
                if d_id and d_id not in doc_map:
                    doc_map[d_id] = {
                        "doc_id": d_id,
                        "file_name": meta.get("file_name"),
                        "file_type": meta.get("file_type"),
                        "total_chunks": 0
                    }
                if d_id:
                    doc_map[d_id]["total_chunks"] += 1
        return list(doc_map.values())

    def get_collection_stats(self):
        count = self._collection.count()
        return {
            "provider": "chroma",
            "db_path": self.db_path,
            "collection_name": self.collection_name,
            "total_chunks": count
        }

    def reset_collection(self):
        self._client.delete_collection(self.collection_name)
        self._initialize()


# Global vector store instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """Get the global vector store instance based on configuration."""
    global _vector_store
    if _vector_store is None:
        provider = settings.vector_store_provider
        if provider == "milvus":
            _vector_store = MilvusProvider()
        elif provider == "chroma":
            _vector_store = ChromaProvider()
        else:
            raise ValueError(f"Unsupported vector store provider: {provider}")
    return _vector_store
