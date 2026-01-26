"""
Configurable chunking strategies for document splitting.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from src.core.observability import get_logger, track_metrics

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    content: str
    metadata: dict
    chunk_id: str

    def __len__(self):
        return len(self.content)


def _find_page_for_chunk(chunk_content: str, chunk_start_pos: int, page_mapping: Optional[Dict]) -> int:
    """
    Find which page a chunk belongs to based on page mapping.

    Args:
        chunk_content: The chunk text content
        chunk_start_pos: The starting position of the chunk in the original text
        page_mapping: Dict mapping position ranges to page numbers

    Returns:
        Page number (1-indexed), defaults to 1 if not found
    """
    if not page_mapping:
        return 1

    chunk_end_pos = chunk_start_pos + len(chunk_content)

    # Find the page that contains this chunk's start position
    for pos_range, page_num in page_mapping.items():
        try:
            start, end = pos_range.split(':')
            start_pos = int(start)
            end_pos = int(end) if end else float('inf')

            # If chunk starts within this range, it belongs to this page
            if start_pos <= chunk_start_pos < end_pos:
                return page_num
        except (ValueError, AttributeError):
            continue

    # Default to first page if no mapping found
    return 1


class ChunkingStrategy(ABC):
    """Base class for chunking strategies."""

    @abstractmethod
    def chunk_text(self, text: str, metadata: dict, page_mapping: Optional[Dict] = None) -> List[Chunk]:
        """Split text into chunks."""
        pass


class RecursiveChunker(ChunkingStrategy):
    """
    Recursive character-based chunking.
    Best for general-purpose text splitting with token awareness.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False,
        )

    @track_metrics("recursive_chunking")
    def chunk_text(self, text: str, metadata: dict, page_mapping: Optional[Dict] = None) -> List[Chunk]:
        """Split text using recursive character splitting with page tracking."""
        logger.info(
            "chunking_text",
            strategy="recursive",
            text_length=len(text),
            chunk_size=self.chunk_size
        )

        chunks = []
        splits = self.splitter.split_text(text)

        # Track position in original text
        current_pos = 0

        for idx, split in enumerate(splits):
            chunk_metadata = metadata.copy()

            # Find which page this chunk belongs to
            page_num = _find_page_for_chunk(split, current_pos, page_mapping)

            chunk_metadata.update({
                "chunk_index": idx,
                "total_chunks": len(splits),
                "chunk_size": len(split),
                "chunking_strategy": "recursive",
                "page": page_num  # Add page number
            })

            chunk_id = f"{metadata.get('doc_id', 'unknown')}_{idx}"

            chunks.append(Chunk(
                content=split,
                metadata=chunk_metadata,
                chunk_id=chunk_id
            ))

            current_pos += len(split)

        logger.info("chunking_completed", num_chunks=len(chunks))
        return chunks


class SemanticChunker(ChunkingStrategy):
    """
    Semantic-based chunking using embeddings.
    Groups text by semantic similarity to create coherent chunks.
    """

    def __init__(self, chunk_size: int = 1000, similarity_threshold: float = 0.7):
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        # Use character splitter for initial splits
        self.splitter = CharacterTextSplitter(
            chunk_size=chunk_size // 2,  # Smaller initial chunks
            chunk_overlap=50,
            separator="\n\n"
        )

    @track_metrics("semantic_chunking")
    def chunk_text(self, text: str, metadata: dict, page_mapping: Optional[Dict] = None) -> List[Chunk]:
        """
        Split text using semantic boundaries with page tracking.

        Note: This is a simplified version. For production, you'd want to:
        1. Generate embeddings for sentences
        2. Calculate similarity between consecutive sentences
        3. Create boundaries where similarity drops below threshold
        """
        logger.info(
            "chunking_text",
            strategy="semantic",
            text_length=len(text),
            chunk_size=self.chunk_size
        )

        # For now, use paragraph-based splitting as a proxy for semantic chunks
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_idx = 0
        current_pos = 0  # Track position for page mapping

        for para in paragraphs:
            para = para.strip()
            if not para:
                current_pos += len(para) + 2
                continue

            para_length = len(para)

            # If adding this paragraph exceeds chunk size, finalize current chunk
            if current_length + para_length > self.chunk_size and current_chunk:
                chunk_content = "\n\n".join(current_chunk)
                chunk_metadata = metadata.copy()

                # Find which page this chunk belongs to
                page_num = _find_page_for_chunk(
                    chunk_content, current_pos - len(chunk_content), page_mapping)

                chunk_metadata.update({
                    "chunk_index": chunk_idx,
                    "chunk_size": len(chunk_content),
                    "chunking_strategy": "semantic",
                    "page": page_num  # Add page number
                })

                chunk_id = f"{metadata.get('doc_id', 'unknown')}_{chunk_idx}"
                chunks.append(Chunk(
                    content=chunk_content,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id
                ))

                current_chunk = []
                current_length = 0
                chunk_idx += 1

            current_chunk.append(para)
            current_length += para_length
            current_pos += para_length + 2

        # Add remaining chunk
        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            chunk_metadata = metadata.copy()

            # Find which page this chunk belongs to
            page_num = _find_page_for_chunk(
                chunk_content, current_pos - len(chunk_content), page_mapping)

            chunk_metadata.update({
                "chunk_index": chunk_idx,
                "chunk_size": len(chunk_content),
                "chunking_strategy": "semantic",
                "page": page_num  # Add page number
            })

            chunk_id = f"{metadata.get('doc_id', 'unknown')}_{chunk_idx}"
            chunks.append(Chunk(
                content=chunk_content,
                metadata=chunk_metadata,
                chunk_id=chunk_id
            ))

        # Update total_chunks in all metadata
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        logger.info("chunking_completed", num_chunks=len(chunks))
        return chunks


class ParagraphChunker(ChunkingStrategy):
    """
    Paragraph-based chunking.
    Preserves natural paragraph boundaries for better context.
    """

    def __init__(self, max_paragraphs_per_chunk: int = 5):
        self.max_paragraphs_per_chunk = max_paragraphs_per_chunk

    @track_metrics("paragraph_chunking")
    def chunk_text(self, text: str, metadata: dict, page_mapping: Optional[Dict] = None) -> List[Chunk]:
        """Split text by paragraphs with page tracking."""
        logger.info(
            "chunking_text",
            strategy="paragraph",
            text_length=len(text),
            max_paragraphs=self.max_paragraphs_per_chunk
        )

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_pos = 0

        for idx in range(0, len(paragraphs), self.max_paragraphs_per_chunk):
            chunk_paragraphs = paragraphs[idx:idx +
                                          self.max_paragraphs_per_chunk]
            chunk_content = "\n\n".join(chunk_paragraphs)

            chunk_metadata = metadata.copy()

            # Find which page this chunk belongs to
            page_num = _find_page_for_chunk(
                chunk_content, current_pos, page_mapping)

            chunk_metadata.update({
                "chunk_index": idx // self.max_paragraphs_per_chunk,
                "total_chunks": (len(paragraphs) + self.max_paragraphs_per_chunk - 1)
                // self.max_paragraphs_per_chunk,
                "page": page_num,  # Add page number
                "chunk_size": len(chunk_content),
                "num_paragraphs": len(chunk_paragraphs),
                "chunking_strategy": "paragraph"
            })

            chunk_id = f"{metadata.get('doc_id', 'unknown')}_{chunk_metadata['chunk_index']}"

            chunks.append(Chunk(
                content=chunk_content,
                metadata=chunk_metadata,
                chunk_id=chunk_id
            ))

        logger.info("chunking_completed", num_chunks=len(chunks))
        return chunks


class ChunkingStrategyFactory:
    """Factory for creating chunking strategies."""

    @staticmethod
    def create(
        strategy_name: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
    ) -> ChunkingStrategy:
        """
        Create a chunking strategy instance.

        Args:
            strategy_name: One of 'recursive', 'semantic', 'paragraph'
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks (for recursive)
            **kwargs: Additional strategy-specific parameters
        """
        strategies = {
            "recursive": lambda: RecursiveChunker(chunk_size, chunk_overlap),
            "semantic": lambda: SemanticChunker(
                chunk_size,
                kwargs.get("similarity_threshold", 0.7)
            ),
            "paragraph": lambda: ParagraphChunker(
                kwargs.get("max_paragraphs_per_chunk", 5)
            ),
        }

        if strategy_name not in strategies:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available: {list(strategies.keys())}"
            )

        logger.info("creating_chunking_strategy", strategy=strategy_name)
        return strategies[strategy_name]()
