import os
import uuid
from groq import Groq
from dotenv import load_dotenv
from embeddings import search

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

from embeddings import load_index
from rag_pipeline import ask, reset_conversation

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load the index we built in previous step
index_path = os.path.join(
    os.path.dirname(__file__),
    '..', 'data', 'indexes', 'infosys.index'
)
chunks_path = os.path.join(
    os.path.dirname(__file__),
    '..', 'data', 'indexes', 'infosys_chunks.json'
)

print("Loading index...")
index, chunks = load_index(index_path, chunks_path)

# Test 1: Single question
print("\n=== Test 1: Single Question ===")
result = ask(
    question="What was the revenue in Q4 FY25?",
    index=index,
    chunks=chunks,
    groq_client=groq_client
)

debug_chunks = search("What was the revenue in Q4 FY25?", index, chunks, top_k=3)
for c in debug_chunks:
    print(f"\nScore {c['similarity_score']:.3f} | Page {c['page']} | {c['type']}")
    print(f"Content: {c['content']}")

print(f"Answer: {result['answer']}")
print(f"Sources used: {result['chunks_used']}")
for source in result['sources'][:2]:
    print(f"  - {source['source']} Page {source['page']} "
          f"(similarity: {source['similarity']})")

# Test 2: Multi-turn memory
print("\n=== Test 2: Multi-turn Memory ===")
result2 = ask(
    question="How does that compare to Q3?",  # refers to previous answer
    index=index,
    chunks=chunks,
    groq_client=groq_client
)
print(f"Answer: {result2['answer']}")

# Test 3: Question not in document
print("\n=== Test 3: Out Of Scope Question ===")
result3 = ask(
    question="What is the weather in Bangalore today?",
    index=index,
    chunks=chunks,
    groq_client=groq_client
)
print(f"Answer: {result3['answer']}")

# Test 4: Reset and verify memory cleared
print("\n=== Test 4: Reset Conversation ===")
reset_conversation()
result4 = ask(
    question="How does that compare to Q3?",  # without memory this should fail gracefully
    index=index,
    chunks=chunks,
    groq_client=groq_client
)
print(f"Answer after reset: {result4['answer']}")