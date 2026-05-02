import os
import json
import sys
from pathlib import Path

# Fix dotenv path — go up from backend/ to project root
env_path = Path(__file__).parent / '.env'
print(f"Loading .env from: {env_path}")
print(f"File exists: {env_path.exists()}")

from dotenv import load_dotenv
load_dotenv(dotenv_path=env_path)

gemini_key = os.getenv("GEMINI_API_KEY")
print(f"Gemini key loaded: {bool(gemini_key)}")
print(f"Key preview: {gemini_key[:8] if gemini_key else 'NONE'}...")

if not gemini_key:
    print("\nERROR: GEMINI_API_KEY not found in .env")
    print("Make sure backend/.env contains:")
    print("GEMINI_API_KEY=your_key_here")
    sys.exit(1)

# Load cached answers
answers_path = Path(__file__).parent.parent / 'data' / 'generated_answers.json'
print(f"\nLoading answers from: {answers_path}")

with open(answers_path, 'r') as f:
    generated = json.load(f)

print(f"Loaded {len(generated)} answers")

# Check for error answers
valid = [g for g in generated if not g['answer'].startswith('Error')]
print(f"Valid answers: {len(valid)}/{len(generated)}")

if len(valid) < 5:
    print("ERROR: Too few valid answers. Check generated_answers.json")
    sys.exit(1)

questions = [g['question'] for g in valid]
answers = [g['answer'] for g in valid]
contexts = [g['contexts'] for g in valid]
ground_truths = [g['ground_truth'] for g in valid]

# RAGAS setup
from datasets import Dataset
from ragas import evaluate
# Replace collections imports with:
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

eval_dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

print(f"\nDataset: {len(questions)} questions")
print("Initializing Gemini...")

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Create base LLM with longer timeout
base_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=gemini_key,
    temperature=0,
    request_timeout=180,    # 3 minutes per call
    max_retries=3
)

ragas_llm = LangchainLLMWrapper(
    base_llm,
    run_config={"timeout": 180}  # also set on wrapper
)

ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
))

print("Running RAGAS evaluation with Gemini...")
print("No Groq calls — purely Gemini + local embeddings\n")

# Replace with lowercase no-parentheses style:
metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
]

results = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    raise_exceptions=False
)

scores_df = results.to_pandas()
print("Columns found:", scores_df.columns.tolist())

def get_score(df, col):
    if col in df.columns:
        val = df[col].mean()
        import math
        return None if math.isnan(val) else val
    return None

faithfulness_score = get_score(scores_df, "faithfulness")
relevancy_score    = get_score(scores_df, "answer_relevancy")
precision_score    = get_score(scores_df, "context_precision")
recall_score       = get_score(scores_df, "context_recall")

print("\n" + "="*50)
print("RAGAS EVALUATION RESULTS")
print("="*50)
print(f"\nFaithfulness:      {faithfulness_score:.3f}" if faithfulness_score else "Faithfulness:      N/A")
print(f"Answer Relevancy:  {relevancy_score:.3f}"    if relevancy_score    else "Answer Relevancy:  N/A")
print(f"Context Precision: {precision_score:.3f}"    if precision_score    else "Context Precision: N/A")
print(f"Context Recall:    {recall_score:.3f}"       if recall_score       else "Context Recall:    N/A")

available = [s for s in [faithfulness_score, relevancy_score, precision_score, recall_score] if s]
if available:
    overall = sum(available) / len(available)
    print(f"\nOverall Score:     {overall:.3f}")

print("\n--- Interpretation ---")
print("0.8+    = Excellent")
print("0.6-0.8 = Good")
print("Below 0.6 = Needs improvement")

# Save
output = {
    "faithfulness":       round(float(faithfulness_score), 3) if faithfulness_score else None,
    "answer_relevancy":   round(float(relevancy_score), 3)    if relevancy_score    else None,
    "context_precision":  round(float(precision_score), 3)    if precision_score    else None,
    "context_recall":     round(float(recall_score), 3)       if recall_score       else None,
    "overall":            round(float(overall), 3)             if available          else None,
    "num_questions": len(questions)
}

out_path = Path(__file__).parent.parent / 'data' / 'ragas_results.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to {out_path}")
print("Add these to your README!")