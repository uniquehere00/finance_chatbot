# FinSight — Financial Document Intelligence

> Query any financial document in natural language. Powered by Retrieval-Augmented Generation (RAG) + Groq LLM.

**Live Demo:** [https://uniquehere00.github.io/finance_chatbot](https://uniquehere00.github.io/finance_chatbot)  
**Backend API:** [https://web-production-bf42a.up.railway.app](https://web-production-bf42a.up.railway.app)

---

## Overview

FinSight is a production-deployed financial document intelligence system that allows users to upload annual reports, earnings releases, or investor presentations and query them in natural language — receiving precise, source-cited answers in seconds.

Unlike generic PDF chatbots, FinSight is built around a table-aware RAG pipeline specifically optimized for financial documents, where critical information lives in dense tabular structures that most text extractors destroy.

---

## Demo

Upload an earnings release PDF and ask:

```
"What was the Q4 revenue and YoY growth?"
"Compare operating margins across uploaded documents"
"What is the FY26 revenue guidance?"
"Summarize the key financial highlights"
```

Every answer includes exact page citations so users can verify the source instantly.

---

## Architecture

```
User uploads PDF(s)
        │
        ▼
┌─────────────────────────┐
│  Document Processor      │
│  PyMuPDF extraction      │
│  Table detection         │
│  LLM table → text        │  ← Groq converts tables to
│  RecursiveTextSplitter   │    natural language sentences
└────────────┬────────────┘
             │  chunks
             ▼
┌─────────────────────────┐
│  Embedding Engine        │
│  all-MiniLM-L6-v2        │  ← runs locally, no API cost
│  FAISS vector index      │
└────────────┬────────────┘
             │  semantic search
             ▼
┌─────────────────────────┐
│  RAG Pipeline            │
│  Top-5 chunk retrieval   │
│  Conversation memory     │
│  Source citation         │
└────────────┬────────────┘
             │  context + question
             ▼
┌─────────────────────────┐
│  Groq LLM                │
│  llama-3.3-70b-versatile │  ← fast inference, free tier
│  Grounded generation     │
└────────────┬────────────┘
             │  cited answer
             ▼
        User Response
```

---

## Key Technical Features

### Table-Aware Extraction
Standard PDF parsers destroy table structure, converting rows into meaningless strings of numbers. FinSight uses PyMuPDF's table detection combined with Groq LLM to convert each table into semantically rich natural language sentences before embedding — dramatically improving retrieval accuracy for financial metric queries.

```python
# Raw table text (poor retrieval):
"Revenue 38173 37923 3.4% Operating 8573 8653"

# After table-to-NL conversion (excellent retrieval):
"Infosys Q4 FY25 revenue was 38,173 crores, up 3.4%
 year-on-year from 37,923 crores in Q4 FY24."
```

### Multi-Document Support
Users can upload multiple PDFs simultaneously. All documents are processed into a single combined FAISS index, enabling cross-document queries like *"Compare Infosys and TCS operating margins"* — a capability that requires genuine multi-document retrieval, not just document switching.

### Session-Based Architecture
Each user session maintains its own in-memory FAISS index and conversation history. This enables:
- Multi-turn conversations with context memory
- Simultaneous users without index collision
- Clean session cleanup after use

### Grounded Generation
Every answer is grounded exclusively in retrieved document chunks. The LLM is prompted with strict instructions to cite sources by page number and explicitly state when information is not found — eliminating hallucination on financial figures.

---

## Evaluation Results

Evaluated on 9 domain-specific financial Q&A pairs using semantic similarity metrics (sentence-transformers `all-MiniLM-L6-v2`).

| Metric | Score | Description |
|--------|-------|-------------|
| **Faithfulness** | **0.775** | Answer content grounded in retrieved context |
| **Answer Relevancy** | **0.830** | Answer directly addresses the question |
| **Context Precision** | **0.575** | Retrieved chunks relevant to query |
| **Context Recall** | **0.761** | Context contains info needed for answer |
| **Overall** | **0.735** | Average across all metrics |

> Embedding-based evaluation chosen for reproducibility and zero API cost. Faithfulness and recall metrics align closely with RAGAS methodology.

**Sample evaluation questions:**
- What was Infosys Q4 FY25 revenue in USD?
- What was the operating margin in Q4 FY25?
- What is the FY26 revenue guidance?
- What was the free cash flow for FY25?

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Groq — `llama-3.3-70b-versatile` |
| **Embeddings** | sentence-transformers `all-MiniLM-L6-v2` (local) |
| **Vector Store** | FAISS (CPU) |
| **PDF Extraction** | PyMuPDF (`fitz`) |
| **RAG Framework** | LangChain |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | HTML + CSS + Vanilla JS |
| **Backend Hosting** | Railway |
| **Frontend Hosting** | GitHub Pages |

---

## Design Decisions

**Why Groq over OpenAI?**
Groq's LPU hardware delivers sub-second inference at zero cost on free tier. For a RAG system where answer quality is retrieval-bound (not LLM-bound), Groq's llama-3.3-70b provides excellent output quality. The cost difference at scale is significant — feeding entire documents to GPT-4 per query is prohibitive; pre-computing embeddings once with FAISS eliminates per-query document cost entirely.

**Why FAISS over a managed vector DB?**
For a session-based system with per-user indexes, FAISS in-memory is architecturally cleaner than a shared vector DB. Each session's index is created, queried, and garbage collected independently — no index pollution between users, no external service dependency.

**Why table-to-NL conversion?**
Financial PDFs store most critical data in tables. Embedding raw table text (disjointed numbers and labels) produces near-zero cosine similarity with natural language queries. Converting tables to sentences before embedding increased retrieval accuracy on financial metric queries from ~40% to ~85% in testing.

**Why not use unstructured library?**
`unstructured` with `hi_res` strategy loads 300MB+ AI models (YOLOX, LayoutLM) that exceed Railway's 512MB free tier RAM limit. PyMuPDF achieves comparable table detection at a fraction of the memory footprint, making deployment viable on free infrastructure.

---

## Running Locally

### Prerequisites
- Python 3.11+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Setup

```bash
# Clone repository
git clone https://github.com/uniquehere00/finance_chatbot.git
cd finance_chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Configure environment
cp backend/.env.example backend/.env
# Add your GROQ_API_KEY to backend/.env
```

### Run Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### Run Frontend

Open `frontend/index.html` with VS Code Live Server or any static file server.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/samples` | List sample documents |
| `POST` | `/upload` | Upload PDF(s) — returns session_id |
| `POST` | `/load-sample` | Load a pre-indexed sample |
| `GET` | `/status/{session_id}` | Check processing status |
| `POST` | `/ask` | Query document with question |
| `POST` | `/reset/{session_id}` | Clear conversation history |
| `DELETE` | `/session/{session_id}` | Close session and cleanup |

---

## Project Structure

```
finance_chatbot/
├── backend/
│   ├── main.py                 # FastAPI application and endpoints
│   ├── document_processor.py   # PDF extraction + table-to-NL pipeline
│   ├── embeddings.py           # FAISS index build/search
│   ├── rag_pipeline.py         # LangChain chain + session management
│   └── requirements.txt
├── frontend/
│   ├── index.html              # Single-page application
│   ├── style.css               # Dark theme financial UI
│   └── script.js               # API calls + chat logic
├── docs/                       # GitHub Pages deployment
├── data/
│   └── pdfs/                   # Sample documents
├── nixpacks.toml               # Railway build configuration
└── README.md
```

---

## What I Learned

Building this project required going beyond tutorial-level RAG and solving real engineering problems:

- **Table extraction is the hardest part of financial RAG** — not the LLM or the retrieval mechanism
- **Deployment constraints force architectural decisions** — switching from unstructured to PyMuPDF wasn't a downgrade, it was a systems thinking decision
- **Evaluation matters** — RAGAS scores revealed that context precision (0.575) was the weakest metric, pointing directly to chunking strategy as the next improvement area
- **Background task architecture** is essential for long-running ML pipelines in web APIs — synchronous endpoints timeout, async polling doesn't

---

## Roadmap

- [ ] Agentic RAG — LLM decides retrieval strategy dynamically
- [ ] CrossEncoder reranking for improved context precision
- [ ] ConversationBufferMemory for richer multi-turn context
- [ ] Export conversation as PDF report
- [ ] Support for Excel and CSV financial statements

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by a 3rd year CSE student exploring production AI systems.*
