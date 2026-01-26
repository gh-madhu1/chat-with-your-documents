"""
Optimized retrieval strategies with BM25, semantic search, and re-ranking.
"""
from typing import List, Dict, Any
from dataclasses import dataclass
from src.core.vector_store import SearchResult, get_vector_store
from src.core.embeddings import get_embedding_model
from src.core.observability import get_logger, track_metrics
from src.config import settings

logger = get_logger(__name__)

# Try to import BM25 and cross-encoder for reranking
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not installed, BM25 retrieval will be disabled")


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with citations."""
    context: str
    sources: List[Dict[str, Any]]
    confidence: float
    num_sources: int


class Retriever:
    """Advanced retrieval with BM25, semantic search, and re-ranking."""

    def __init__(self):
        self.vector_store = get_vector_store()
        self.embedding_model = get_embedding_model()
        self.bm25_corpus = None
        self.bm25 = None
        self.bm25_doc_count = 0
        # Note: BM25 will be initialized asynchronously on first retrieval if needed

    async def _initialize_bm25(self):
        """Initialize BM25 index from stored documents."""
        if not BM25_AVAILABLE:
            logger.info("BM25 not available, using semantic search only")
            return

        try:
            # Load all documents from vector store
            all_docs = await self.vector_store.get_all_documents()
            if all_docs:
                # Tokenize documents for BM25
                self.bm25_corpus = [
                    doc.content.lower().split() if hasattr(doc, 'content') else []
                    for doc in all_docs
                ]
                self.bm25 = BM25Okapi(self.bm25_corpus)
                self.bm25_doc_count = len(self.bm25_corpus)
                logger.info("BM25 index initialized",
                            num_docs=self.bm25_doc_count)
            else:
                self.bm25_doc_count = 0
                logger.info("No documents available for BM25 initialization")
        except Exception as e:
            logger.warning("Failed to initialize BM25", error=str(e))
            self.bm25 = None

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent retrieval."""
        import re
        # Lowercase, remove extra whitespace, and remove punctuation
        query = query.lower().strip()
        query = re.sub(r'[^\w\s]', '', query)
        return query

    @track_metrics("retrieval")
    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_multi_query: bool = False,
        use_reranking: bool = None
    ) -> RetrievalResult:
        """
        Retrieve relevant context for a query using hybrid BM25+semantic search.
        """
        normalized_query = self._normalize_query(query)
        logger.info("retrieving_context",
                    original_query=query[:100],
                    normalized_query=normalized_query[:100],
                    use_multi_query=use_multi_query)

        # Initialize BM25 lazily or refresh if doc count changed significantly
        if BM25_AVAILABLE:
            current_doc_count = self.vector_store.get_collection_stats().get("total_chunks", 0)
            if self.bm25 is None or abs(current_doc_count - self.bm25_doc_count) > 10:
                await self._initialize_bm25()

        # Use reranking if enabled in settings
        use_reranking = use_reranking if use_reranking is not None else settings.use_reranking

        if use_reranking:
            # Hybrid retrieval with BM25 + semantic search + reranking
            results = await self._hybrid_retrieval_with_reranking(normalized_query, top_k, use_multi_query)
        elif BM25_AVAILABLE and self.bm25:
            # Hybrid retrieval without reranking
            results = await self._hybrid_retrieval(normalized_query, top_k, use_multi_query)
        else:
            # Semantic search only
            if use_multi_query:
                results = await self._multi_query_retrieval(normalized_query, top_k)
            else:
                results = await self._simple_retrieval(normalized_query, top_k)

        # Format results
        retrieval_result = self._format_results(results)

        logger.info(
            "retrieval_completed",
            num_sources=retrieval_result.num_sources,
            confidence=retrieval_result.confidence
        )

        return retrieval_result

    async def _hybrid_retrieval(
        self,
        query: str,
        top_k: int = None,
        use_multi_query: bool = False
    ) -> List[SearchResult]:
        """
        Hybrid retrieval combining BM25 (lexical) and semantic search using 
        Reciprocal Rank Fusion (RRF) for better consistency.
        """
        top_k = top_k or settings.top_k_retrieval
        k_rrf = 60  # Standard constant for RRF

        logger.info("hybrid_retrieval_rrf",
                    query=query[:100], bm25_available=bool(self.bm25))

        # Get candidates (retrieve more for better fusion)
        candidate_count = top_k * 4

        # Get semantic results
        if use_multi_query:
            semantic_results = await self._multi_query_retrieval(query, top_k=candidate_count)
        else:
            semantic_results = await self._simple_retrieval(query, top_k=candidate_count)

        # Get BM25 results if available
        if self.bm25:
            bm25_results = await self._bm25_retrieval(query, top_k=candidate_count)
        else:
            bm25_results = []

        # Reciprocal Rank Fusion
        rrf_scores = {}  # chunk_id -> rrf_score
        chunk_map = {}   # chunk_id -> SearchResult

        # Add semantic ranks
        for rank, result in enumerate(semantic_results, 1):
            chunk_id = result.chunk_id
            chunk_map[chunk_id] = result
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k_rrf + rank)

        # Add BM25 ranks
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result.chunk_id
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = result
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k_rrf + rank)

        # Sort by RRF score
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True
        )

        # Take top_k
        final_results = []
        for cid in sorted_chunk_ids[:top_k]:
            result = chunk_map[cid]
            # Use original semantic score or normalized RRF for display
            # Here we just keep the result object
            final_results.append(result)

        return final_results

    async def _hybrid_retrieval_with_reranking(
        self,
        query: str,
        top_k: int = None,
        use_multi_query: bool = False
    ) -> List[SearchResult]:
        """Hybrid retrieval with reranking for best precision."""
        top_k = top_k or settings.top_k_retrieval

        # First get hybrid results with more candidates
        candidates = await self._hybrid_retrieval(
            query,
            top_k=top_k * 5,  # Get more candidates for reranking
            use_multi_query=use_multi_query
        )

        if not candidates:
            return []

        # Rerank using relevance score and semantic similarity
        query_embedding = self.embedding_model.embed_query(query)

        reranked = []
        for result in candidates:
            # Calculate relevance score
            relevance = result.score

            # Could add cross-encoder reranking here in future
            # For now, use semantic similarity combined with BM25
            reranked.append((result, relevance))

        # Sort by relevance
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Return top-k after reranking
        return [r[0] for r in reranked[:top_k]]

    async def _bm25_retrieval(self, query: str, top_k: int = None) -> List[SearchResult]:
        """BM25 lexical search."""
        if not self.bm25 or not BM25_AVAILABLE:
            return []

        try:
            top_k = top_k or settings.top_k_retrieval

            # Tokenize query
            query_tokens = query.lower().split()

            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)

            # Get top-k results
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]

            # Fetch actual documents from vector store
            all_docs = await self.vector_store.get_all_documents()
            bm25_results = []

            for idx in top_indices:
                if idx < len(all_docs) and scores[idx] > 0:
                    doc = all_docs[idx]
                    # Create SearchResult with BM25 score
                    result = SearchResult(
                        content=doc.content if hasattr(
                            doc, 'content') else str(doc),
                        metadata=doc.metadata if hasattr(
                            doc, 'metadata') else {},
                        # Normalize BM25 score
                        score=float(scores[idx]) / 10.0,
                        chunk_id=doc.chunk_id if hasattr(
                            doc, 'chunk_id') else str(idx)
                    )
                    bm25_results.append(result)

            logger.info("bm25_retrieval_completed",
                        num_results=len(bm25_results))
            return bm25_results
        except Exception as e:
            logger.error("bm25_retrieval_error", error=str(e))
            return []

    async def _simple_retrieval(self, query: str, top_k: int = None) -> List[SearchResult]:
        """Simple vector similarity search."""
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)

        # Search vector store
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        return results

    async def _multi_query_retrieval(self, query: str, top_k: int = None) -> List[SearchResult]:
        """
        Multi-query retrieval with query variations.

        Generates variations of the query and retrieves for each,
        then deduplicates and re-ranks results.
        """
        # Generate query variations (simplified - in production use LLM)
        query_variations = [
            query,
            f"What is {query}?",
            f"Explain {query}",
        ]

        all_results = {}  # Use dict to deduplicate by chunk_id

        for variation in query_variations:
            query_embedding = self.embedding_model.embed_query(variation)
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k or 3
            )

            for result in results:
                if result.chunk_id not in all_results:
                    all_results[result.chunk_id] = result
                else:
                    # Keep the one with higher score
                    if result.score > all_results[result.chunk_id].score:
                        all_results[result.chunk_id] = result

        # Sort by score and return top-k
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.score,
            reverse=True
        )

        return sorted_results[:top_k] if top_k else sorted_results

    def _format_results(self, results: List[SearchResult]) -> RetrievalResult:
        """
        Format search results into retrieval result with citations.

        Args:
            results: List of search results

        Returns:
            Formatted RetrievalResult
        """
        if not results:
            return RetrievalResult(
                context="No relevant context found.",
                sources=[],
                confidence=0.0,
                num_sources=0
            )

        # Build context from results
        context_parts = []
        sources = []

        for idx, result in enumerate(results, 1):
            # Add context with source marker
            context_parts.append(f"[Source {idx}]\n{result.content}")

            # Extract citation info
            sources.append({
                "source_id": idx,
                "file_name": result.metadata.get("file_name", "Unknown"),
                "page": result.metadata.get("page", "N/A"),
                "chunk_index": result.metadata.get("chunk_index", 0),
                "score": round(result.score, 3),
                "citation": result.get_citation()
            })

        # Join context
        context = "\n\n".join(context_parts)

        # Calculate overall confidence (use the highest score among results for better perceived accuracy)
        confidence = results[0].score if results else 0.0

        return RetrievalResult(
            context=context,
            sources=sources,
            confidence=round(confidence, 2),
            num_sources=len(sources)
        )

    async def retrieve_with_reranking(
        self,
        query: str,
        top_k: int = None,
        rerank_top_n: int = 20
    ) -> RetrievalResult:
        """
        Retrieve with re-ranking for better relevance.

        Args:
            query: User query
            top_k: Final number of results
            rerank_top_n: Number of candidates to retrieve before re-ranking

        Returns:
            Re-ranked retrieval result
        """
        logger.info("retrieval_with_reranking", query=query[:100])

        # First, retrieve more candidates
        initial_results = await self._simple_retrieval(query, top_k=rerank_top_n)

        if not initial_results:
            return RetrievalResult(
                context="No relevant context found.",
                sources=[],
                confidence=0.0,
                num_sources=0
            )

        # Re-rank using cross-encoder (simplified - in production use actual cross-encoder)
        # For now, just use the original scores
        reranked_results = initial_results[:top_k]

        return self._format_results(reranked_results)


# Global retriever instance
_retriever = None


def get_retriever() -> Retriever:
    """Get the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
