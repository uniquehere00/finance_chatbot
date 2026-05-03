from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os, uuid, shutil
from pathlib import Path

from rag_pipeline import ask_session, create_session, delete_session
from embeddings import load_index

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

app = FastAPI(title="Financial Document RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "pdfs"
UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Track processing status separately
# Key: session_id
# Value: "processing" | "ready" | "error"
processing_status = {}


class QuestionRequest(BaseModel):
    session_id: str
    question: str

class SampleDocRequest(BaseModel):
    filename: str


# ===== BACKGROUND TASK =====
def process_pdf_background(session_id: str, pdf_path: str):
    """
    Runs in background after upload returns.
    Updates processing_status when done.
    """
    try:
        processing_status[session_id] = "processing"
        print(f"Background processing started for session {session_id}")
        
        result = create_session(
            session_id=session_id,
            pdf_path=pdf_path,
            groq_client=groq_client
        )
        
        if "error" in result:
            processing_status[session_id] = f"error:{result['error']}"
            print(f"Processing error: {result['error']}")
        else:
            processing_status[session_id] = "ready"
            print(f"Session {session_id} ready — {result['chunks_created']} chunks")
    
    except Exception as e:
        processing_status[session_id] = f"error:{str(e)}"
        print(f"Background task failed: {e}")


# ===== ENDPOINTS =====
@app.get("/")
async def root():
    return {"status": "running", "message": "Financial RAG API is live"}

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    1. Validates and saves PDF
    2. Returns session_id IMMEDIATELY
    3. Processes PDF in background
    4. Frontend polls /status/{session_id} until ready
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    
    if size_mb > 50:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB. Maximum is 50MB."
        )
    
    # Generate session ID and save file
    session_id = str(uuid.uuid4())[:8]
    pdf_filename = f"{session_id}_{file.filename}"
    pdf_path = UPLOAD_DIR / pdf_filename
    
    with open(pdf_path, 'wb') as f:
        f.write(contents)
    
    print(f"PDF saved: {pdf_path} ({size_mb:.1f}MB)")
    
    # Mark as processing immediately
    processing_status[session_id] = "processing"
    
    # Add background task — runs AFTER this function returns
    background_tasks.add_task(
        process_pdf_background,
        session_id,
        str(pdf_path)
    )
    
    # Return immediately — don't wait for processing
    return {
        "session_id": session_id,
        "message": "Upload received, processing started",
        "pdf_name": file.filename,
        "status": "processing"
    }


@app.get("/status/{session_id}")
async def get_status(session_id: str):
    """
    Frontend polls this every 3 seconds until status = ready.
    
    Returns:
    {"status": "processing", "message": "..."}
    {"status": "ready", "chunks_created": 87, "pdf_name": "..."}
    {"status": "error", "message": "..."}
    """
    from rag_pipeline import sessions
    
    status = processing_status.get(session_id, "not_found")
    
    if status == "not_found":
        return {"status": "not_found"}
    
    elif status == "processing":
        return {
            "status": "processing",
            "message": "Extracting content and building search index..."
        }
    
    elif status == "ready":
        session = sessions.get(session_id, {})
        return {
            "status": "ready",
            "chunks_created": len(session.get("chunks", [])),
            "pdf_name": session.get("pdf_name", "document.pdf")
        }
    
    elif status.startswith("error:"):
        return {
            "status": "error",
            "message": status.replace("error:", "")
        }
    
    return {"status": status}


@app.post("/load-sample")
async def load_sample_document(
    background_tasks: BackgroundTasks,
    request: SampleDocRequest
):
    pdf_path = SAMPLE_DIR / request.filename
    
    if not pdf_path.exists():
        available = [f.name for f in SAMPLE_DIR.glob("*.pdf")]
        raise HTTPException(
            status_code=404,
            detail=f"Sample not found. Available: {available}"
        )
    
    session_id = str(uuid.uuid4())[:8]
    processing_status[session_id] = "processing"
    
    background_tasks.add_task(
        process_pdf_background,
        session_id,
        str(pdf_path)
    )
    
    return {
        "session_id": session_id,
        "message": "Sample loading started",
        "pdf_name": request.filename,
        "status": "processing"
    }


@app.get("/samples")
async def list_samples():
    samples = []
    for pdf in SAMPLE_DIR.glob("*.pdf"):
        size_mb = pdf.stat().st_size / (1024 * 1024)
        samples.append({
            "filename": pdf.name,
            "display_name": pdf.stem.replace("_", " ").title(),
            "size_mb": round(size_mb, 1)
        })
    return {"samples": samples}


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Check session is ready before answering
    status = processing_status.get(request.session_id)
    if status == "processing":
        raise HTTPException(
            status_code=202,
            detail="Document still processing. Please wait."
        )
    if status != "ready":
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a document first."
        )
    
    result = ask_session(
        session_id=request.session_id,
        question=request.question,
        groq_client=groq_client
    )
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "question": result["question"],
        "chunks_used": result["chunks_used"]
    }


@app.post("/reset/{session_id}")
async def reset_session_history(session_id: str):
    from rag_pipeline import sessions
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id]["history"] = []
    return {"message": "Conversation cleared", "session_id": session_id}


@app.delete("/session/{session_id}")
async def close_session(session_id: str):
    for pdf in UPLOAD_DIR.glob(f"{session_id}_*.pdf"):
        pdf.unlink()
    delete_session(session_id)
    if session_id in processing_status:
        del processing_status[session_id]
    return {"message": "Session closed"}


@app.get("/session/{session_id}/status")
async def session_status(session_id: str):
    from rag_pipeline import sessions
    if session_id not in sessions:
        return {"active": False}
    session = sessions[session_id]
    return {
        "active": True,
        "pdf_name": session["pdf_name"],
        "chunks": len(session["chunks"]),
        "conversation_turns": len(session["history"]) // 2
    }


@app.get("/debug/sessions")
async def debug_sessions():
    from rag_pipeline import sessions
    return {
        "active_sessions": list(sessions.keys()),
        "processing_status": processing_status,
        "count": len(sessions)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)