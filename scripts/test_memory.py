import asyncio
import httpx
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

async def test_multi_turn_memory():
    async with httpx.AsyncClient(timeout=120.0) as client:
        print("\n" + "="*50)
        print("TESTING MULTI-TURN CONVERSATION MEMORY")
        print("="*50)

        # 1. Upload a document with specific info
        print("\n1. Uploading test document...")
        test_file = Path("memory_test.txt")
        test_file.write_text("""
        The Project X is a hidden research facility located in the mountains of Switzerland.
        It was founded in 1995 by a group of scientists led by Dr. Elena Vance.
        The primary goal of Project X is to study quantum entanglement at macro scales.
        """)
        
        try:
            with open(test_file, "rb") as f:
                files = {"file": ("memory_test.txt", f, "text/plain")}
                response = await client.post(f"{API_BASE_URL}/api/ingest", files=files)
            
            job_id = response.json()["job_id"]
            print(f"   Upload job started: {job_id}")

            # Wait for processing
            while True:
                response = await client.get(f"{API_BASE_URL}/api/ingest/status/{job_id}")
                status = response.json()
                if status["status"] == "completed":
                    print("   Processing completed.")
                    break
                elif status["status"] == "failed":
                    print(f"   Processing failed: {status.get('error')}")
                    return
                await asyncio.sleep(2)

            # 2. Ask first question
            print("\n2. First question: 'What is Project X?'")
            response = await client.post(
                f"{API_BASE_URL}/api/conversation",
                json={"question": "What is Project X?"}
            )
            data = response.json()
            session_id = data["session_id"]
            print(f"   Answer: {data['answer']}")
            print(f"   Session ID: {session_id}")

            # 3. Ask follow-up with ambiguous pronoun
            print("\n3. Follow-up: 'When was it founded and by whom?'")
            print("   (Verifying if 'it' is correctly identified as 'Project X' for retrieval)")
            
            response = await client.post(
                f"{API_BASE_URL}/api/conversation",
                json={
                    "question": "When was it founded and by whom?",
                    "session_id": session_id
                }
            )
            data = response.json()
            print(f"   Answer: {data['answer']}")
            
            if "1995" in data["answer"] and "Elena Vance" in data["answer"]:
                print("\n✅ SUCCESS: Context memory used successfully!")
            else:
                print("\n❌ FAILURE: Context memory might be broken.")

        finally:
            if test_file.exists():
                test_file.unlink()

if __name__ == "__main__":
    asyncio.run(test_multi_turn_memory())
