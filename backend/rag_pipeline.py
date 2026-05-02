import os
from typing import List, Dict, Tuple
from groq import Groq
from dotenv import load_dotenv

from embeddings import search, load_index
from document_processor import process_pdf, save_chunks

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Conversation memory stored as simple list
# Each item is {"role": "user/assistant", "content": "..."}
# This is exactly what Groq API expects — no extra library needed
conversation_history = []

def build_prompt(
    question: str,
    retrieved_chunks: List[Dict],
    conversation_history: List[Dict]
) -> List[Dict]:

    # Format context
    context_parts = []
    for chunk in retrieved_chunks:
        source_info = f"[Source {chunk['rank']}: {chunk['source']}, Page {chunk.get('page', 'N/A')}]"
        context_parts.append(f"{source_info}\n{chunk['content']}")
    
    context = "\n\n".join(context_parts)

    # Simple, direct system message
    system_message = """You are a financial analyst assistant.
Use the provided context to answer questions.
Always cite sources as (Source 1, Page 2).
Give complete, detailed answers — include all relevant 
numbers and context from the source, not just one figure.
Be factual and concise."""
    
    # Direct user message
    user_message = f"""Context:
{context}

Question: {question}
Answer:"""

    messages = [{"role": "system", "content": system_message}]
    messages.extend(conversation_history[-6:])
    messages.append({"role": "user", "content": user_message})

    return messages

def ask(
    question: str,
    index,
    chunks: List[Dict],
    groq_client: Groq,
    top_k: int = 5
) -> Dict:
    """
    Main function that handles a complete Q&A interaction.
    
    Flow:
    1. Search FAISS for relevant chunks
    2. Build prompt with those chunks
    3. Send to Groq
    4. Save to conversation history
    5. Return answer with sources
    """
    
    global conversation_history
    
    # Step 1: Retrieve relevant chunks
    retrieved_chunks = search(question, index, chunks, top_k=top_k)
    
    if not retrieved_chunks:
        return {
            "answer": "No relevant information found in the document.",
            "sources": [],
            "question": question
        }
    
    # Step 2: Build the prompt
    messages = build_prompt(question, retrieved_chunks, conversation_history)
    
    # Step 3: Send to Groq
    print(f"Sending to Groq with {len(retrieved_chunks)} context chunks...")
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1024,
            temperature=0.1  # slightly above 0 for natural language
                             # but still very factual
        )
        
        answer = response.choices[0].message.content.strip()
        
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": [],
            "question": question,
            "chunks_used": 0
        }
    
    # Step 4: Update conversation history
    # Save question WITHOUT context (context is retrieved fresh each time)
    conversation_history.append({
        "role": "user",
        "content": question  # just the question, not the full prompt
    })
    conversation_history.append({
        "role": "assistant", 
        "content": answer
    })
    
    # Step 5: Build sources list for frontend to display
    sources = []
    for chunk in retrieved_chunks:
        sources.append({
            "rank": chunk["rank"],
            "source": chunk["source"],
            "page": chunk.get("page"),
            "type": chunk["type"],
            "similarity": round(chunk["similarity_score"], 3),
            "preview": chunk["content"][:150] + "..."
        })
    
    return {
        "answer": answer,
        "sources": sources,
        "question": question,
        "chunks_used": len(retrieved_chunks)
    }


def reset_conversation():
    """Clears conversation history — called when user uploads new document."""
    global conversation_history
    conversation_history = []
    print("Conversation history cleared")

# Stores each user's index and chunks in memory
# Key: session_id, Value: {index, chunks}
sessions = {}


def create_session(
    session_id: str,
    pdf_path: str,
    groq_client: Groq
) -> Dict:
    """
    Processes a user uploaded PDF and creates a session.
    
    Why sessions?
    User A uploads Infosys report → gets session "abc123"
    User B uploads TCS report    → gets session "def456"
    
    Each user gets their own index in memory.
    Questions from User A only search User A's document.
    """
    
    print(f"Creating session {session_id} for {pdf_path}")
    
    # Process PDF into chunks
    chunks = process_pdf(pdf_path, groq_client)
    
    if not chunks:
        return {"error": "Could not extract content from PDF"}
    
    # Generate embeddings
    from embeddings import generate_embeddings, build_faiss_index
    embeddings = generate_embeddings(chunks)
    index = build_faiss_index(embeddings)
    
    # Store in sessions dict
    sessions[session_id] = {
        "index": index,
        "chunks": chunks,
        "history": [],
        "pdf_name": os.path.basename(pdf_path)
    }
    
    return {
        "session_id": session_id,
        "chunks_created": len(chunks),
        "pdf_name": os.path.basename(pdf_path)
    }


def ask_session(
    session_id: str,
    question: str,
    groq_client: Groq
) -> Dict:
    """
    Answers a question for a specific session.
    Uses that session's index and maintains that session's history.
    """
    
    if session_id not in sessions:
        return {"error": "Session not found. Please upload a document first."}
    
    session = sessions[session_id]
    
    # Temporarily set this session's history as active
    global conversation_history
    conversation_history = session["history"]
    
    # Get answer
    result = ask(
        question=question,
        index=session["index"],
        chunks=session["chunks"],
        groq_client=groq_client
    )
    
    # Save updated history back to session
    session["history"] = conversation_history
    
    return result


def delete_session(session_id: str):
    """Removes session from memory when user is done."""
    if session_id in sessions:
        del sessions[session_id]
        print(f"Session {session_id} deleted")

        