import os
import sys
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

from document_processor import load_chunks
from embeddings import (
    generate_embeddings, 
    build_faiss_index,
    save_index,
    load_index,
    search
)

# Paths
chunks_path = os.path.join(
    os.path.dirname(__file__), 
    '..', 'data', 'infosys_chunks.json'
)
index_path = os.path.join(
    os.path.dirname(__file__), 
    '..', 'data', 'indexes', 'infosys.index'
)
index_chunks_path = os.path.join(
    os.path.dirname(__file__), 
    '..', 'data', 'indexes', 'infosys_chunks.json'
)

# Create indexes folder if not exists
os.makedirs(os.path.dirname(index_path), exist_ok=True)

# Step 1: Load chunks from previous step
print("Loading chunks...")
chunks = load_chunks(chunks_path)

# Step 2: Generate embeddings
embeddings = generate_embeddings(chunks)

# Step 3: Build FAISS index
index = build_faiss_index(embeddings)

# Step 4: Save everything
save_index(index, chunks, index_path, index_chunks_path)

# Step 5: Test search with financial questions
print("\n--- Testing Search ---")
test_questions = [
    "What was the revenue in Q4?",
    "What is the operating margin?",
    "How many employees does Infosys have?",
    "What was the year on year growth?"
]

for question in test_questions:
    print(f"\nQuestion: {question}")
    results = search(question, index, chunks, top_k=3)
    
    for result in results:
        print(f"  Rank {result['rank']} "
              f"(score: {result['similarity_score']:.3f}) "
              f"[{result['type']}] "
              f"Page {result['page']}: "
              f"{result['content'][:100]}...")