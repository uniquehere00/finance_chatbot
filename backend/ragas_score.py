import os
import json
import math
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load answers
answers_path = Path(__file__).parent.parent / 'data' / 'generated_answers.json'
with open(answers_path, 'r') as f:
    generated = json.load(f)

print(f"Evaluating {len(generated)} Q&A pairs\n")

faithfulness_scores = []
relevancy_scores = []
precision_scores = []
recall_scores = []

for i, item in enumerate(generated):
    question    = item['question']
    answer      = item['answer']
    contexts    = item['contexts']
    ground_truth = item['ground_truth']

    # Skip error answers
    if answer.startswith("Error"):
        continue

    # ===== FAITHFULNESS =====
    # Does answer content appear in retrieved contexts?
    # Measure: semantic similarity between answer and best context
    answer_emb   = model.encode(answer, convert_to_tensor=True)
    context_embs = model.encode(contexts, convert_to_tensor=True)
    sims = util.cos_sim(answer_emb, context_embs)[0]
    faithfulness = float(sims.max())
    faithfulness_scores.append(faithfulness)

    # ===== ANSWER RELEVANCY =====
    # Is answer relevant to the question?
    # Measure: semantic similarity between question and answer
    question_emb = model.encode(question, convert_to_tensor=True)
    relevancy = float(util.cos_sim(question_emb, answer_emb))
    relevancy_scores.append(relevancy)

    # ===== CONTEXT PRECISION =====
    # Are retrieved contexts relevant to question?
    # Measure: avg similarity of top 3 contexts to question
    top_sims = sorted(
        [float(util.cos_sim(question_emb, model.encode(c, convert_to_tensor=True))) 
         for c in contexts[:3]], 
        reverse=True
    )
    precision = sum(top_sims) / len(top_sims)
    precision_scores.append(precision)

    # ===== CONTEXT RECALL =====
    # Does context contain info needed to answer?
    # Measure: similarity between ground truth and best context
    gt_emb = model.encode(ground_truth, convert_to_tensor=True)
    gt_sims = util.cos_sim(gt_emb, context_embs)[0]
    recall = float(gt_sims.max())
    recall_scores.append(recall)

    print(f"Q{i+1}: F={faithfulness:.2f} R={relevancy:.2f} P={precision:.2f} Rc={recall:.2f}")
    print(f"      {question[:60]}")

# Calculate averages
f  = sum(faithfulness_scores) / len(faithfulness_scores)
ar = sum(relevancy_scores)    / len(relevancy_scores)
cp = sum(precision_scores)    / len(precision_scores)
cr = sum(recall_scores)       / len(recall_scores)
overall = (f + ar + cp + cr) / 4

print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"\nFaithfulness:      {f:.3f}")
print(f"Answer Relevancy:  {ar:.3f}")
print(f"Context Precision: {cp:.3f}")
print(f"Context Recall:    {cr:.3f}")
print(f"\nOverall Score:     {overall:.3f}")

print("\n--- Interpretation ---")
print("0.8+    = Excellent")
print("0.6-0.8 = Good")
print("Below 0.6 = Needs improvement")

# Save
out = {
    "faithfulness":      round(f,  3),
    "answer_relevancy":  round(ar, 3),
    "context_precision": round(cp, 3),
    "context_recall":    round(cr, 3),
    "overall":           round(overall, 3),
    "method": "sentence-transformers semantic similarity",
    "model": "all-MiniLM-L6-v2",
    "num_questions": len(faithfulness_scores)
}

out_path = Path(__file__).parent.parent / 'data' / 'ragas_results.json'
with open(out_path, 'w') as f_out:
    json.dump(out, f_out, indent=2)

print(f"\nSaved to {out_path}")
print("\nAdd these scores to your README!")