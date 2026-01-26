"""
Example SSE streaming client.
"""
import httpx
import asyncio
import json


API_BASE_URL = "http://localhost:8000"


async def stream_query_example():
    """Demonstrate SSE streaming for real-time pipeline updates."""
    print("=" * 80)
    print("SSE Streaming Example")
    print("=" * 80)
    
    question = "What are the main topics covered in the document?"
    
    print(f"\nQuestion: {question}\n")
    print("Streaming response:\n")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "GET",
            f"{API_BASE_URL}/api/stream/query",
            params={"question": question}
        ) as response:
            answer_parts = []
            
            async for line in response.aiter_lines():
                if not line:
                    continue
                
                # Parse SSE format
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    
                    try:
                        data = json.loads(data_str)
                        
                        # Handle different event types
                        if "stage" in data:
                            stage = data["stage"]
                            status = data["status"]
                            message = data["message"]
                            print(f"[{stage.upper()}] {status}: {message}")
                        
                        elif "chunk" in data:
                            chunk = data["chunk"]
                            answer_parts.append(chunk)
                            print(chunk, end="", flush=True)
                        
                        elif "error" in data:
                            print(f"\nError: {data['error']}")
                            break
                        
                        elif "message" in data and data["message"] == "Query completed":
                            print("\n\n[COMPLETE]")
                            break
                    
                    except json.JSONDecodeError:
                        continue
            
            full_answer = "".join(answer_parts)
            print(f"\n\nFull answer ({len(full_answer)} chars):")
            print(full_answer)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(stream_query_example())
