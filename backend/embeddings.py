import os
import json
import faiss
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Load embedding model once at module level
# This means model loads when file is imported
# Not every time you call a function
# Saves 3-4 seconds per request

print("Loading embedding model...")
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded!")

def generate_embeddings(chunks: List[Dict]) -> np.ndarray:
    """
    Converts list of chunks into embedding vectors.
    
    Input:  [{"content": "Revenue was...", "type": "table"}, ...]
    Output: numpy array of shape (num_chunks, 384)
            each row = one chunk's vector
    
    Example for 87 chunks:
    Output shape = (87, 384)
    meaning 87 vectors, each 384 numbers long
    """
    
    # Extract just the text content from each chunk
    texts = [chunk["content"] for chunk in chunks]
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    print("This takes 1-3 minutes on CPU, please wait...")
    
    # batch_size=32 means process 32 chunks at once
    # show_progress_bar=True shows a progress bar
    embeddings = EMBEDDING_MODEL.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True  # FAISS needs numpy arrays
    )
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    # prints something like: Generated embeddings shape: (87, 384)
    
    return embeddings

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a FAISS index from embeddings.
    
    FAISS index is like a search engine for vectors.
    You give it 87 vectors, it organizes them so it can
    find the most similar one to any query in milliseconds.
    
    Why IndexFlatIP (Inner Product)?
    
    Two common similarity measures:
    1. Cosine similarity  → angle between vectors
    2. Inner product      → dot product of vectors
    
    When vectors are normalized (length=1),
    inner product == cosine similarity.
    We normalize below so IndexFlatIP gives cosine similarity.
    """
    
    # Get dimension from embeddings (384 for our model)
    dimension = embeddings.shape[1]
    print(f"Building FAISS index with dimension {dimension}...")
    
    # Normalize vectors to length 1
    # This makes inner product equal to cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create the index
    index = faiss.IndexFlatIP(dimension)
    
    # Add all embeddings to index
    index.add(embeddings)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    return index

def save_index(
    index: faiss.Index, 
    chunks: List[Dict], 
    index_path: str, 
    chunks_path: str
):
    """
    Saves FAISS index and chunks to disk.
    
    Why save both?
    FAISS index stores vectors (numbers).
    But when you retrieve, you need the original text too.
    
    So you save them separately and load together:
    index  → tells you WHICH chunks are most similar
    chunks → gives you the ACTUAL TEXT of those chunks
    
    They stay in sync because chunk[i] corresponds to 
    the i-th vector in the index. Order must never change.
    """
    
    # Save FAISS index
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")
    
    # Save chunks as JSON
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunks saved to {chunks_path}")


def load_index(
    index_path: str, 
    chunks_path: str
) -> Tuple[faiss.Index, List[Dict]]:
    """
    Loads FAISS index and chunks from disk.
    Call this at startup instead of rebuilding every time.
    Loading takes 1 second vs building which takes 3 minutes.
    """
    
    index = faiss.read_index(index_path)
    print(f"Loaded FAISS index with {index.ntotal} vectors")
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    
    return index, chunks

def search(
    query: str,
    index: faiss.Index,
    chunks: List[Dict],
    top_k: int = 5
) -> List[Dict]:
    """
    Searches for most relevant chunks for a given query.
    
    Steps:
    1. Convert query text to embedding vector
    2. Normalize that vector
    3. FAISS finds top_k most similar vectors
    4. Return those chunks with their similarity scores
    
    top_k=5 means return 5 most relevant chunks.
    These 5 chunks become the context for Groq to answer from.
    
    Why 5?
    Too few (1-2) → might miss important context
    Too many (10+) → overwhelms the LLM, slower response
    5 → sweet spot for financial documents
    """
    
    # Step 1: Embed the query
    query_embedding = EMBEDDING_MODEL.encode(
        [query],                # list with one item
        convert_to_numpy=True
    )
    
    # Step 2: Normalize
    faiss.normalize_L2(query_embedding)
    
    # Step 3: Search — returns distances and indices
    distances, indices = index.search(query_embedding, top_k)
    
    # distances shape: (1, top_k) — similarity scores
    # indices shape:  (1, top_k) — which chunks matched
    
    # Step 4: Build results with metadata
    results = []
    for i, (distance, idx) in enumerate(
        zip(distances[0], indices[0])
    ):
        if idx == -1:  # FAISS returns -1 if not enough vectors
            continue
            
        chunk = chunks[idx].copy()
        chunk["similarity_score"] = float(distance)
        chunk["rank"] = i + 1
        results.append(chunk)
    
    return results

