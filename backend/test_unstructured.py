import os
import sys
from collections import Counter

# Checks PDF exists first
pdf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'pdfs', 'infosys_q4_fy25.pdf')

if not os.path.exists(pdf_path):
    print(f"ERROR: PDF not found at {pdf_path}")
    print("Make sure you downloaded the PDF into data/pdfs/ folder")
    sys.exit(1)

print(f"PDF found: {pdf_path}")
print("Processing... this may take 30-60 seconds on first run\n")

from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename=pdf_path,
    strategy="hi_res",          
    infer_table_structure=True
)

# Counts what element types were found
types = Counter([el.category for el in elements])
print("Element types found in PDF:")
for typ, count in types.items():
    print(f"  {typ}: {count}")

# Shows first table found
print("\n--- First Table Found ---")
table_found = False
for el in elements:
    if el.category == "Table":
        print(el.text[:600])
        table_found = True
        break

if not table_found:
    print("No tables found with 'fast' strategy")
    print("Try changing strategy to 'hi_res' (slower but more thorough)")

# Shows total elements
print(f"\nTotal elements extracted: {len(elements)}")
print("\nUnstructured is working correctly!")