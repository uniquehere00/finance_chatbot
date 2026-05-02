import os
import json
import fitz
from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


def convert_table_to_natural_language(table_text: str, groq_client: Groq) -> str:
    if len(table_text.strip()) < 20:
        return table_text
    prompt = f"""Convert this financial table into clear natural language sentences.
Each row becomes one sentence. Include all numbers exactly. Output sentences only.

Table:
{table_text}

Sentences:"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Table conversion failed: {e}")
        return table_text


def process_pdf(pdf_path: str, groq_client: Groq) -> List[Dict]:
    print(f"\nProcessing: {os.path.basename(pdf_path)}")
    doc = fitz.open(pdf_path)
    chunks = []
    table_count = 0

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Extract tables
        try:
            tables = page.find_tables()
            for table in tables:
                table_count += 1
                try:
                    df = table.to_pandas()
                    table_text = df.to_string()
                except:
                    table_text = str(table.extract())

                if len(table_text.strip()) > 20:
                    natural_text = convert_table_to_natural_language(
                        table_text, groq_client
                    )
                    chunks.append({
                        "content": natural_text,
                        "type": "table",
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1
                    })
        except Exception as e:
            print(f"Table extraction error page {page_num + 1}: {e}")

        # Extract text
        text = page.get_text("text")
        if text.strip():
            text_chunks = chunk_text(text)
            for chunk in text_chunks:
                if len(chunk.strip()) < 30:
                    continue
                chunks.append({
                    "content": chunk,
                    "type": "text",
                    "source": os.path.basename(pdf_path),
                    "page": page_num + 1
                })

    doc.close()
    print(f"Created {len(chunks)} chunks ({table_count} tables)")
    return chunks


def save_chunks(chunks: List[Dict], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks(input_path: str) -> List[Dict]:
    with open(input_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {input_path}")
    return chunks


def process_multiple_pdfs(pdf_folder: str, groq_client: Groq) -> List[Dict]:
    all_chunks = []
    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    if not pdf_files:
        return []
    for pdf_path in pdf_files:
        chunks = process_pdf(str(pdf_path), groq_client)
        all_chunks.extend(chunks)
    return all_chunks