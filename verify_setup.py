"""
Quick verification script to test core components.
"""
import asyncio
from pathlib import Path
from src.config import settings
from src.core.document_processor import DocumentProcessor
from src.core.chunking_strategies import ChunkingStrategyFactory
from src.core.embeddings import get_embedding_model
from src.core.vector_store import get_vector_store
from src.core.observability import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def verify_components():
    """Verify all core components work."""
    print("=" * 80)
    print("Component Verification")
    print("=" * 80)
    
    # 1. Test document processor
    print("\n1. Testing Document Processor...")
    processor = DocumentProcessor()
    
    # Create a test document
    test_doc = Path("test_doc.txt")
    test_doc.write_text("This is a test document for verification. It contains sample text.")
    
    try:
        document = await processor.process_file(test_doc, "test-doc-1")
        print(f"   ✓ Document processed: {len(document.content)} chars")
        print(f"   ✓ Metadata: {document.metadata}")
    finally:
        test_doc.unlink()
    
    # 2. Test chunking
    print("\n2. Testing Chunking Strategies...")
    chunker = ChunkingStrategyFactory.create("recursive", chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_text(document.content, document.metadata)
    print(f"   ✓ Created {len(chunks)} chunks")
    print(f"   ✓ First chunk: {chunks[0].content[:50]}...")
    
    # 3. Test embeddings
    print("\n3. Testing Embedding Model...")
    embedding_model = get_embedding_model()
    test_texts = ["Hello world", "This is a test"]
    embeddings = embedding_model.embed_texts(test_texts)
    print(f"   ✓ Generated {len(embeddings)} embeddings")
    print(f"   ✓ Embedding dimension: {len(embeddings[0])}")
    
    # 4. Test vector store
    print("\n4. Testing Vector Store...")
    vector_store = get_vector_store()
    
    # Use only first 2 chunks and their embeddings
    test_chunks = chunks[:min(2, len(chunks))]
    chunk_texts = [chunk.content for chunk in test_chunks]
    chunk_embeddings = embedding_model.embed_texts(chunk_texts)
    
    await vector_store.add_documents(
        chunks=chunk_texts,
        embeddings=chunk_embeddings,
        metadatas=[chunk.metadata for chunk in test_chunks],
        ids=[chunk.chunk_id for chunk in test_chunks]
    )
    
    stats = vector_store.get_collection_stats()
    print(f"   ✓ Vector store initialized")
    print(f"   ✓ Total chunks in store: {stats['total_chunks']}")
    
    # 5. Test retrieval
    print("\n5. Testing Retrieval...")
    query_embedding = embedding_model.embed_query("test document")
    results = await vector_store.search(query_embedding, top_k=2)
    print(f"   ✓ Retrieved {len(results)} results")
    if results:
        print(f"   ✓ Top result score: {results[0].score:.3f}")
    
    print("\n" + "=" * 80)
    print("All components verified successfully! ✓")
    print("=" * 80)
    
    # Cleanup
    vector_store.reset_collection()


if __name__ == "__main__":
    asyncio.run(verify_components())
