import os
import platform
import json
from pathlib import Path
from typing import List, Dict
from collections import Counter

from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from dotenv import load_dotenv

# Only import pytesseract if on Linux (Render server)
# On Windows it's handled differently
if platform.system() == "Linux":
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
        os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata"
    except ImportError:
        pass

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """
    Splits long text into smaller overlapping chunks.
    
    Why chunking is needed:
    - Embedding models have token limits
    - Smaller chunks = more precise retrieval
    - Overlap ensures context isn't lost at boundaries
    
    Example:
    "Hello world this is a long text..."
    chunk 1: "Hello world this is"
    chunk 2: "this is a long text"  <- overlap keeps context
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
        # tries to split at paragraphs first,
        # then newlines, then sentences, then words
    )
    return splitter.split_text(text)

def convert_table_to_natural_language(table_text: str, groq_client: Groq) -> str:
    """
    Converts raw table text into natural language sentences.
    
    Why this matters:
    Raw table:  "Revenue 38173 37923 3.4% Operating 8573 8653"
    → embedding model sees random numbers, poor retrieval
    
    Natural language: "Infosys Q4 FY25 revenue was 38,173 crores,
    up 3.4% year on year from 37,923 crores in Q4 FY24"
    → embedding model understands context, excellent retrieval
    """
    
    # If table is too short or just numbers, return as is
    if len(table_text.strip()) < 20:
        return table_text
    
    prompt = f"""Convert this financial table data into clear natural language sentences.
Rules:
- Each row should become one complete sentence
- Include all numbers exactly as they appear
- Mention what each number represents
- Keep it factual, no analysis
- Output sentences only, no headings or bullet points

Table data:
{table_text}

Natural language sentences:"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0  # temperature 0 = consistent, factual output
                           # higher temperature = more creative but less accurate
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        # If Groq call fails for any reason, return original table text
        # Your pipeline never crashes because of one bad table
        print(f"Warning: Table conversion failed: {e}")
        return table_text

def group_uncategorized_elements(elements) -> list:
    """
    Groups consecutive UncategorizedText elements into 
    larger meaningful chunks.
    
    Why needed:
    Financial fact sheets break data into tiny fragments.
    "Revenue" + "38,173" + "crores" are 3 separate elements
    but meaningless individually.
    
    Grouping them: "Revenue 38,173 crores" → meaningful chunk.
    """
    grouped = []
    buffer = []
    buffer_page = None
    
    for element in elements:
        if element.category == "UncategorizedText":
            try:
                page = element.metadata.page_number
            except:
                page = None
                
            # If same page, keep buffering
            if buffer_page is None or buffer_page == page:
                buffer.append(element.text.strip())
                buffer_page = page
            else:
                # Different page — flush buffer as one element
                if buffer:
                    grouped.append({
                        "category": "UncategorizedText",
                        "text": " ".join(buffer),
                        "page": buffer_page
                    })
                buffer = [element.text.strip()]
                buffer_page = page
        else:
            # Non-uncategorized element — flush buffer first
            if buffer:
                grouped.append({
                    "category": "UncategorizedText",
                    "text": " ".join(buffer),
                    "page": buffer_page
                })
                buffer = []
                buffer_page = None
            
            # Add current element normally
            try:
                page = element.metadata.page_number
            except:
                page = None
            grouped.append({
                "category": element.category,
                "text": element.text,
                "page": page
            })
    
    # Flush any remaining buffer
    if buffer:
        grouped.append({
            "category": "UncategorizedText",
            "text": " ".join(buffer),
            "page": buffer_page
        })
    
    return grouped

    
def process_pdf(pdf_path: str, groq_client: Groq) -> List[Dict]:
    
    print(f"\nProcessing: {os.path.basename(pdf_path)}")
    print("Extracting elements from PDF...")
    
    elements = partition_pdf(
        filename=pdf_path,
        strategy="auto",
        infer_table_structure=True
    )
    
    types = Counter([el.category for el in elements])
    print(f"Found: {dict(types)}")
    
    # Group consecutive UncategorizedText elements
    print("Grouping fragmented elements...")
    grouped_elements = group_uncategorized_elements(elements)
    print(f"After grouping: {len(grouped_elements)} elements")
    
    chunks = []
    table_count = 0
    text_count = 0
    
    for element in grouped_elements:
        
        page_num = element.get("page")
        category = element["category"]
        text = element["text"].strip()
        
        # Skip empty
        if not text:
            continue
        
        # --- TABLES ---
        if category == "Table":
            table_count += 1
            print(f"  Converting table {table_count} to natural language...")
            natural_text = convert_table_to_natural_language(
                text, groq_client
            )
            chunks.append({
                "content": natural_text,
                "type": "table",
                "original_table": text,
                "source": os.path.basename(pdf_path),
                "page": page_num
            })
        
        # --- NARRATIVE TEXT AND TITLES ---
        elif category in ["NarrativeText", "Title", "ListItem"]:
            if len(text) < 30:
                continue
            text_chunks = chunk_text(text)
            for chunk in text_chunks:
                text_count += 1
                chunks.append({
                    "content": chunk,
                    "type": "text",
                    "source": os.path.basename(pdf_path),
                    "page": page_num
                })
        
        # --- UNCATEGORIZED (now grouped, much richer) ---
        elif category == "UncategorizedText":
            if len(text) < 20:
                continue
            # Chunk if very long, otherwise keep as is
            if len(text) > 500:
                text_chunks = chunk_text(text)
                for chunk in text_chunks:
                    text_count += 1
                    chunks.append({
                        "content": chunk,
                        "type": "text",
                        "source": os.path.basename(pdf_path),
                        "page": page_num
                    })
            else:
                text_count += 1
                chunks.append({
                    "content": text,
                    "type": "text",
                    "source": os.path.basename(pdf_path),
                    "page": page_num
                })
    
    print(f"Created {len(chunks)} chunks ({table_count} tables, {text_count} text)")
    return chunks

def save_chunks(chunks: List[Dict], output_path: str):
    """
    Saves processed chunks to disk as JSON.
    
    Why save to disk?
    Processing PDFs with hi_res takes 3-5 minutes.
    You only want to do this ONCE.
    Save the result, load it instantly next time.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks(input_path: str) -> List[Dict]:
    """Loads previously processed chunks from disk."""
    with open(input_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {input_path}")
    return chunks


def process_multiple_pdfs(pdf_folder: str, groq_client: Groq) -> List[Dict]:
    """
    Processes all PDFs in a folder.
    Useful when user uploads multiple documents.
    """
    all_chunks = []
    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in {pdf_folder}")
        return []
    
    print(f"Found {len(pdf_files)} PDFs to process")
    
    for pdf_path in pdf_files:
        chunks = process_pdf(str(pdf_path), groq_client)
        all_chunks.extend(chunks)
    
    print(f"\nTotal chunks from all PDFs: {len(all_chunks)}")
    return all_chunks