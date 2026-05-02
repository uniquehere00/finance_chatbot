import os
import sys
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Import your new functions
from document_processor import process_pdf, save_chunks

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

pdf_path = os.path.join(
    os.path.dirname(__file__), 
    '..', 'data', 'pdfs', 'infosysy_press_release.pdf'
)

# Process the PDF
chunks = process_pdf(pdf_path, groq_client)

# Show first 3 chunks
print("\n--- First 3 Chunks ---")
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1}:")
    print(f"  Type: {chunk['type']}")
    print(f"  Page: {chunk['page']}")
    print(f"  Content: {chunk['content'][:200]}...")

# Save to disk
output_path = os.path.join(
    os.path.dirname(__file__),
    '..', 'data', 'infosys_chunks.json'
)
save_chunks(chunks, output_path)