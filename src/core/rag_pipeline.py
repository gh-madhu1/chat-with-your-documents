"""
Complete RAG pipeline orchestration.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from src.config import settings
from src.core.retrieval import get_retriever
from src.core.conversation_manager import get_conversation_manager
from src.core.callbacks import CallbackManager, PipelineStage
from src.core.observability import get_logger, track_metrics
from src.core.local_llm import get_local_llm

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    answer: str
    sources: list[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]


class RAGPipeline:
    """Complete RAG pipeline with LLM integration."""

    def __init__(self):
        self.retriever = get_retriever()
        self.conversation_manager = get_conversation_manager()

        # Use local LLM only
        self.llm = get_local_llm()
        self.use_local = True
        logger.info("using_local_llm", model=settings.local_llm_model)

    @track_metrics("rag_query")
    async def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline.

        Args:
            question: User question
            session_id: Optional conversation session ID
            callback_manager: Optional callback manager for progress tracking

        Returns:
            RAG response with answer and sources
        """
        logger.info("rag_query_started",
                    question=question[:100], session_id=session_id)

        callbacks = callback_manager or CallbackManager()

        try:
            # Stage 1: Retrieval
            await callbacks.on_stage_start(PipelineStage.RETRIEVAL, {"question": question})

            retrieval_result = await self.retriever.retrieve(
                query=question,
                use_multi_query=False
            )

            await callbacks.on_stage_complete(
                PipelineStage.RETRIEVAL,
                {"num_sources": retrieval_result.num_sources}
            )

            # Stage 2: Build context with conversation history
            context = retrieval_result.context

            if session_id:
                conversation_context = self.conversation_manager.get_context(
                    session_id)
                if conversation_context:
                    context = f"Previous conversation:\n{conversation_context}\n\n{context}"

            # Stage 3: LLM Inference
            await callbacks.on_stage_start(PipelineStage.INFERENCE, {"question": question})

            answer = await self._generate_answer(question, context, callbacks)

            await callbacks.on_stage_complete(
                PipelineStage.INFERENCE,
                {"answer_length": len(answer)}
            )

            # Stage 4: Complete
            await callbacks.on_stage_complete(
                PipelineStage.COMPLETE,
                {"confidence": retrieval_result.confidence}
            )

            # Save to conversation if session exists
            if session_id:
                self.conversation_manager.add_message(
                    session_id, "user", question)
                self.conversation_manager.add_message(
                    session_id,
                    "assistant",
                    answer,
                    {"sources": retrieval_result.sources}
                )

            return RAGResponse(
                answer=answer,
                sources=retrieval_result.sources,
                confidence=retrieval_result.confidence,
                metadata={
                    "session_id": session_id,
                    "num_sources": retrieval_result.num_sources
                }
            )

        except Exception as e:
            await callbacks.on_stage_error(PipelineStage.ERROR, e)
            logger.error("rag_query_failed", error=str(e),
                         error_type=type(e).__name__)
            raise

    async def _generate_answer(
        self,
        question: str,
        context: str,
        callbacks: CallbackManager
    ) -> str:
        """
        Generate answer using LLM.

        Args:
            question: User question
            context: Retrieved context with sources
            callbacks: Callback manager

        Returns:
            Generated answer
        """
        if not self.llm:
            # Mock response if LLM not configured
            return f"Mock answer for: {question}\n\nBased on the provided context."

        # Build prompt
        prompt = self._build_prompt(question, context)

        # Generate response based on LLM type
        if self.use_local:
            response = await self.llm.agenerate(prompt)
        else:
            response = await self.llm.ainvoke(prompt)
            response = response.content

        return response

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build prompt for LLM.

        Args:
            question: User question
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        # Check if context is empty or insufficient
        if not context or context.strip() == "" or len(context.strip()) < 10:
            context = "[No relevant context found in the documents]"

        prompt = f"""{settings.system_prompt}

IMPORTANT INSTRUCTIONS:
1. Answer the question using ONLY the information from the provided context below
2. If the context says "[No relevant context found]" or doesn't contain enough information to answer the question, politely explain that you don't have enough information in the uploaded documents to answer this question
3. When you have relevant context, cite your sources by referencing the source information
4. Be concise, accurate, and helpful
5. If you're uncertain, express your level of confidence

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION:
{question}

YOUR ANSWER:"""

        return prompt

    async def stream_query(
        self,
        question: str,
        session_id: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        top_k: Optional[int] = None
    ):
        """
        Stream query response token by token.

        Args:
            question: User question
            session_id: Optional conversation session ID
            callback_manager: Optional callback manager
            top_k: Number of results to retrieve

        Yields:
            Tuple of (chunk, sources) - chunk is answer token, sources are only sent once at start
        """
        logger.info("rag_stream_query_started", question=question[:100])

        callbacks = callback_manager or CallbackManager()

        # Retrieval stage
        await callbacks.on_stage_start(PipelineStage.RETRIEVAL)
        retrieval_result = await self.retriever.retrieve(query=question, top_k=top_k)
        await callbacks.on_stage_complete(PipelineStage.RETRIEVAL)

        # Yield sources first
        yield ("__SOURCES__", retrieval_result.sources)

        # Build context
        context = retrieval_result.context
        if session_id:
            conversation_context = self.conversation_manager.get_context(
                session_id)
            if conversation_context:
                context = f"Previous conversation:\n{conversation_context}\n\n{context}"

        # Inference stage
        await callbacks.on_stage_start(PipelineStage.INFERENCE)

        if not self.llm:
            yield ("Mock streaming response for: " + question, None)
            return

        prompt = self._build_prompt(question, context)

        # Stream response based on LLM type
        full_answer = ""

        if self.use_local:
            async for chunk in self.llm.astream(prompt):
                full_answer += chunk
                yield (chunk, None)
        else:
            async for chunk in self.llm.astream(prompt):
                if chunk.content:
                    full_answer += chunk.content
                    yield (chunk.content, None)

        await callbacks.on_stage_complete(PipelineStage.INFERENCE)

        # Save to conversation
        if session_id:
            self.conversation_manager.add_message(session_id, "user", question)
            self.conversation_manager.add_message(
                session_id,
                "assistant",
                full_answer,
                {"sources": retrieval_result.sources}
            )


# Global RAG pipeline instance
_rag_pipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """Get the global RAG pipeline instance."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
