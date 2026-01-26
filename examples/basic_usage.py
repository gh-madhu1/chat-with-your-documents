"""
Basic usage example for the Chat With Your Docs API.
"""
import httpx
import asyncio
from pathlib import Path


API_BASE_URL = "http://localhost:8000"


async def main():
    """Demonstrate basic API usage."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("=" * 80)
        print("Chat With Your Docs - Basic Usage Example")
        print("=" * 80)
        
        # 1. Check health
        print("\n1. Checking API health...")
        response = await client.get(f"{API_BASE_URL}/health")
        health = response.json()
        print(f"   Status: {health['status']}")
        print(f"   Vector Store: {health['vector_store_status']['total_chunks']} chunks")
        
        # 2. Upload a document
        print("\n2. Uploading a document...")
        
        # For demo, create a sample text file
        sample_doc = Path("sample_document.txt")
        sample_doc.write_text("""
# Sample Document

This is a sample document for testing the RAG system.

## Introduction
The Chat With Your Docs system is a production-grade RAG (Retrieval-Augmented Generation) 
application that allows users to ask questions about their documents.

## Key Features
- Document ingestion with PDF and text support
- Configurable chunking strategies
- Vector-based similarity search
- Multi-turn conversations
- Real-time streaming responses
- Comprehensive evaluation metrics

## How It Works
1. Documents are uploaded and processed
2. Text is chunked using configurable strategies
3. Chunks are embedded using sentence transformers
4. Embeddings are stored in a vector database
5. User queries are matched against stored embeddings
6. Relevant context is retrieved and passed to an LLM
7. The LLM generates grounded answers with citations

## Benefits
- Fast and accurate question answering
- Source attribution for transparency
- Scalable architecture
- Production-ready observability
        """)
        
        with open(sample_doc, "rb") as f:
            files = {"file": ("sample_document.txt", f, "text/plain")}
            response = await client.post(f"{API_BASE_URL}/api/ingest", files=files)
        
        upload_result = response.json()
        job_id = upload_result["job_id"]
        print(f"   Job ID: {job_id}")
        print(f"   Status: {upload_result['status']}")
        
        # Wait for processing to complete
        print("\n3. Waiting for document processing...")
        while True:
            response = await client.get(f"{API_BASE_URL}/api/ingest/status/{job_id}")
            status = response.json()
            print(f"   Status: {status['status']}")
            
            if status["status"] in ["completed", "failed"]:
                break
            
            await asyncio.sleep(2)
        
        if status["status"] == "failed":
            print(f"   Error: {status.get('error')}")
            return
        
        print(f"   Processed {status['num_chunks']} chunks")
        
        # 4. Ask a question
        print("\n4. Asking a question...")
        question = "What are the key features of the system?"
        
        response = await client.post(
            f"{API_BASE_URL}/api/query",
            json={"question": question}
        )
        
        result = response.json()
        print(f"\n   Question: {question}")
        print(f"\n   Answer: {result['answer']}")
        print(f"\n   Confidence: {result['confidence']:.3f}")
        print(f"\n   Sources ({len(result['sources'])}):")
        for source in result['sources']:
            print(f"     - {source['citation']} (score: {source['score']:.3f})")
        
        # 5. Start a conversation
        print("\n5. Starting a multi-turn conversation...")
        
        # First turn
        response = await client.post(
            f"{API_BASE_URL}/api/conversation",
            json={"question": "How does the system work?"}
        )
        
        conv_result = response.json()
        session_id = conv_result["session_id"]
        print(f"\n   Session ID: {session_id}")
        print(f"   Q: How does the system work?")
        print(f"   A: {conv_result['answer'][:200]}...")
        
        # Follow-up question
        response = await client.post(
            f"{API_BASE_URL}/api/conversation",
            json={
                "question": "What are the benefits?",
                "session_id": session_id
            }
        )
        
        conv_result = response.json()
        print(f"\n   Q: What are the benefits?")
        print(f"   A: {conv_result['answer'][:200]}...")
        
        # 6. Get conversation history
        print("\n6. Retrieving conversation history...")
        response = await client.get(f"{API_BASE_URL}/api/conversation/{session_id}")
        history = response.json()
        print(f"   Total messages: {len(history['messages'])}")
        
        # 7. List documents
        print("\n7. Listing indexed documents...")
        response = await client.get(f"{API_BASE_URL}/api/documents")
        documents = response.json()
        print(f"   Total documents: {len(documents)}")
        for doc in documents:
            print(f"     - {doc['file_name']} ({doc['total_chunks']} chunks)")
        
        print("\n" + "=" * 80)
        print("Example completed successfully!")
        print("=" * 80)
        
        # Cleanup
        sample_doc.unlink()


if __name__ == "__main__":
    asyncio.run(main())
