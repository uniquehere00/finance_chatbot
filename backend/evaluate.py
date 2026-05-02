import os
import json
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

from embeddings import load_index, search
from rag_pipeline import ask, reset_conversation
from test_questions import TEST_QA_PAIRS

# ===== STEP 1: LOAD INDEX =====
print("Loading index...")
index_path = os.path.join(
    os.path.dirname(__file__),
    '..', 'data', 'indexes', 'infosys.index'
)
chunks_path = os.path.join(
    os.path.dirname(__file__),
    '..', 'data', 'indexes', 'infosys_chunks.json'
)

index, chunks = load_index(index_path, chunks_path)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print(f"Loaded index with {len(chunks)} chunks")
print(f"Running evaluation on {len(TEST_QA_PAIRS)} questions\n")


# ===== STEP 2: GENERATE ANSWERS =====
print("="*50)
print("Generating answers from RAG pipeline...")
print("="*50)

questions = []
answers = []
contexts = []
ground_truths = []

for i, qa in enumerate(TEST_QA_PAIRS):
    print(f"\nQ{i+1}: {qa['question']}")
    reset_conversation()
    if i > 0:
        time.sleep(7)

    result = ask(
        question=qa['question'],
        index=index,
        chunks=chunks,
        groq_client=groq_client,
        top_k=5
    )

    retrieved = search(qa['question'], index, chunks, top_k=5)
    context_texts = [r['content'] for r in retrieved]

    print(f"A: {result['answer'][:100]}...")

    questions.append(qa['question'])
    answers.append(result['answer'])
    contexts.append(context_texts)
    ground_truths.append(qa['ground_truth'])

print("\n" + "="*50)
print("All answers generated. Running RAGAS evaluation...")
print("="*50 + "\n")


# ===== STEP 3: RAGAS SETUP =====
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# Build dataset
eval_dataset = Dataset.from_dict({
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": contexts,
    "reference": ground_truths
})

print("Dataset built:")
print(eval_dataset)

# Configure LLM and embeddings for RAGAS

ragas_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

ragas_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ===== STEP 4: RUN RAGAS =====
print("\nRunning RAGAS metrics (this takes 3-5 minutes)...")

metrics = [
    Faithfulness(llm=ragas_llm),
    ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    LLMContextPrecisionWithReference(llm=ragas_llm),
    LLMContextRecall(llm=ragas_llm)
]

max_retries = 2
results = None

for attempt in range(max_retries):
    try:
        results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            raise_exceptions=False
        )
        break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}")
        if attempt < max_retries - 1:
            print("Retrying in 30 seconds...")
            time.sleep(30)

if results is None:
    print("Evaluation failed after all retries.")
    exit(1)


# ===== STEP 5: DISPLAY RESULTS =====
print("\n" + "="*50)
print("RAGAS EVALUATION RESULTS")
print("="*50)

scores_df = results.to_pandas()
print("\nColumn names available:", scores_df.columns.tolist())

# Map new column names
col_map = {
    "faithfulness": "faithfulness",
    "response_relevancy": "response_relevancy",
    "llm_context_precision_with_reference": "llm_context_precision_with_reference",
    "context_recall": "context_recall"
}

# Get scores safely
def get_score(df, col_name):
    if col_name in df.columns:
        return df[col_name].mean()
    return None

faithfulness_score = get_score(scores_df, "faithfulness")
relevancy_score = get_score(scores_df, "answer_relevancy")
precision_score = get_score(scores_df, "llm_context_precision_with_reference")
recall_score = get_score(scores_df, "context_recall")

print(f"\nFaithfulness:       {faithfulness_score:.3f}" if faithfulness_score else "Faithfulness: N/A")
print(f"Response Relevancy: {relevancy_score:.3f}" if relevancy_score else "Response Relevancy: N/A")
print(f"Context Precision:  {precision_score:.3f}" if precision_score else "Context Precision: N/A")
print(f"Context Recall:     {recall_score:.3f}" if recall_score else "Context Recall: N/A")

available_scores = [s for s in [
    faithfulness_score, relevancy_score,
    precision_score, recall_score
] if s is not None]

if available_scores:
    overall = sum(available_scores) / len(available_scores)
    print(f"\nOverall Score:      {overall:.3f}")

print("\n--- Score Interpretation ---")
print("0.8+    = Excellent")
print("0.6-0.8 = Good")
print("0.4-0.6 = Needs improvement")
print("Below 0.4 = Poor")


# ===== STEP 6: SAVE RESULTS =====
output_path = os.path.join(
    os.path.dirname(__file__),
    '..', 'data', 'ragas_results.json'
)

results_dict = {
    "faithfulness": round(float(faithfulness_score), 3) if faithfulness_score else None,
    "response_relevancy": round(float(relevancy_score), 3) if relevancy_score else None,
    "context_precision": round(float(precision_score), 3) if precision_score else None,
    "context_recall": round(float(recall_score), 3) if recall_score else None,
    "overall": round(float(overall), 3) if available_scores else None,
    "num_questions": len(questions),
    "per_question": []
}

for i in range(len(questions)):
    row = {
        "question": questions[i],
        "answer": answers[i],
        "ground_truth": ground_truths[i],
    }
    for col in scores_df.columns:
        if col not in ["user_input", "response", "retrieved_contexts", "reference"]:
            try:
                row[col] = round(float(scores_df[col][i]), 3)
            except:
                row[col] = None
    results_dict["per_question"].append(row)

with open(output_path, 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\nResults saved to: {output_path}")
print("Add these scores to your README!")