"""
Evaluation harness: SBERT vs TF-IDF on labeled resume ranking dataset.
Metrics: Precision@1, Precision@2, MRR, NDCG@3

Run from project root:
    python -m evaluation.evaluate
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from evaluation.labeled_dataset import LABELED_DATA
from src.preprocessing import TextPreprocessor
from src.embedding import SemanticMatcher
from src.tfidf_matcher import TFIDFMatcher


# ── Metrics ───────────────────────────────────────────────────────────────────

def precision_at_k(predicted: list, best: str, k: int) -> float:
    return 1.0 if best in predicted[:k] else 0.0


def reciprocal_rank(predicted: list, best: str) -> float:
    if best in predicted:
        return 1.0 / (predicted.index(best) + 1)
    return 0.0


def ndcg_at_k(predicted: list, relevance_map: dict, k: int) -> float:
    n = len(relevance_map)

    def rel_score(rid):
        rank = relevance_map.get(rid, n + 1)
        return n - rank + 1  # rank 1 -> n, rank n -> 1

    dcg = sum(rel_score(rid) / np.log2(i + 2) for i, rid in enumerate(predicted[:k]))
    ideal = sorted(relevance_map.keys(), key=lambda r: relevance_map[r])
    idcg = sum(rel_score(rid) / np.log2(i + 2) for i, rid in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def rank_by_scores(ids: list, scores: list) -> list:
    return [r for r, _ in sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)]


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("Loading models (this may take 30 seconds first time)...")
    preprocessor = TextPreprocessor()
    sbert = SemanticMatcher()
    tfidf = TFIDFMatcher()

    results = []

    for entry in LABELED_DATA:
        jd_id = entry["jd_id"]
        jd_raw = entry["job_description"]
        resumes = entry["resumes"]

        ids = [r["id"] for r in resumes]
        texts = [r["text"] for r in resumes]
        relevance_map = {r["id"]: r["relevance_rank"] for r in resumes}
        best = min(relevance_map, key=relevance_map.get)

        # Preprocess
        jd_sbert = preprocessor.preprocess(jd_raw, mode="clean_only")
        res_sbert = [preprocessor.preprocess(t, mode="clean_only") for t in texts]

        jd_tfidf = preprocessor.preprocess(jd_raw, mode="full")
        res_tfidf = [preprocessor.preprocess(t, mode="full") for t in texts]

        # SBERT ranking
        jd_emb = sbert.embed(jd_sbert)
        res_embs = [sbert.embed(t) for t in res_sbert]
        sbert_scores = [float(cosine_similarity([jd_emb], [e])[0][0]) for e in res_embs]
        sbert_rank = rank_by_scores(ids, sbert_scores)

        # TF-IDF ranking
        jd_vec, res_vecs = tfidf.fit_and_embed_all(jd_tfidf, res_tfidf)
        tfidf_scores = [float(cosine_similarity([jd_vec], [v])[0][0]) for v in res_vecs]
        tfidf_rank = rank_by_scores(ids, tfidf_scores)

        results.append({
            "JD": jd_id,
            "SBERT P@1":    precision_at_k(sbert_rank, best, 1),
            "TFIDF P@1":    precision_at_k(tfidf_rank, best, 1),
            "SBERT P@2":    precision_at_k(sbert_rank, best, 2),
            "TFIDF P@2":    precision_at_k(tfidf_rank, best, 2),
            "SBERT MRR":    reciprocal_rank(sbert_rank, best),
            "TFIDF MRR":    reciprocal_rank(tfidf_rank, best),
            "SBERT NDCG@3": ndcg_at_k(sbert_rank, relevance_map, 3),
            "TFIDF NDCG@3": ndcg_at_k(tfidf_rank, relevance_map, 3),
            "SBERT Top-1":  sbert_rank[0],
            "TFIDF Top-1":  tfidf_rank[0],
            "Ground Truth": best,
        })

    df = pd.DataFrame(results)

    # Average row
    avg = {col: df[col].mean() if df[col].dtype != object else "-" for col in df.columns}
    avg["JD"] = "AVERAGE"
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)

    # Print
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS: SBERT vs TF-IDF")
    print("=" * 60)
    print(df[[
        "JD", "SBERT P@1", "TFIDF P@1",
        "SBERT MRR", "TFIDF MRR",
        "SBERT NDCG@3", "TFIDF NDCG@3"
    ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nTop-1 Predictions vs Ground Truth:")
    print(df[["JD", "SBERT Top-1", "TFIDF Top-1", "Ground Truth"]].to_string(index=False))

    avg_row = df[df["JD"] == "AVERAGE"].iloc[0]
    print(f"\nSummary:")
    print(f"  SBERT avg P@1:    {float(avg_row['SBERT P@1']):.1%}")
    print(f"  TF-IDF avg P@1:   {float(avg_row['TFIDF P@1']):.1%}")
    print(f"  SBERT avg MRR:    {float(avg_row['SBERT MRR']):.3f}")
    print(f"  TF-IDF avg MRR:   {float(avg_row['TFIDF MRR']):.3f}")
    print(f"  SBERT avg NDCG@3: {float(avg_row['SBERT NDCG@3']):.3f}")
    print(f"  TF-IDF avg NDCG@3:{float(avg_row['TFIDF NDCG@3']):.3f}")

    os.makedirs("evaluation/results", exist_ok=True)
    df.to_json("evaluation/results/eval_results.json", orient="records", indent=2)
    print("\nResults saved to evaluation/results/eval_results.json")

    return df


if __name__ == "__main__":
    run_evaluation()