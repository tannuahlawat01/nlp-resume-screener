import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def rank_resumes(
    job_embedding,
    resume_embeddings,
    resume_names,
    job_skills=None,
    resume_skills_list=None,
    tfidf_scores=None,
    tfidf_weight=0.2
):
    """
    Rank resumes by similarity to a job description.

    Primary score: SBERT cosine similarity (semantic matching)
    Optional hybrid: weighted combination with TF-IDF score
    Optional explainability: matched and missing skills per resume

    Args:
        job_embedding:       SBERT embedding of the job description
        resume_embeddings:   list of SBERT embeddings, one per resume
        resume_names:        list of resume file names
        job_skills:          list of skills extracted from the JD (for explainability)
        resume_skills_list:  list of skill lists, one per resume (for explainability)
        tfidf_scores:        list of TF-IDF cosine scores, one per resume (optional)
        tfidf_weight:        weight given to TF-IDF in hybrid score (default 0.2)

    Returns:
        pd.DataFrame with columns:
            Resume, SBERT Score, TF-IDF Score (if provided),
            Final Score, Matched Skills, Missing Skills
    """
    job_embedding = np.array(job_embedding).reshape(1, -1)
    resume_embeddings = np.array(resume_embeddings)

    # SBERT cosine similarities
    sbert_scores = cosine_similarity(job_embedding, resume_embeddings)[0]

    rows = []
    for i, name in enumerate(resume_names):
        sbert_score = float(sbert_scores[i])

        # Hybrid score: blend SBERT + TF-IDF if TF-IDF scores are provided
        if tfidf_scores is not None:
            tfidf_score = float(tfidf_scores[i])
            final_score = (1 - tfidf_weight) * sbert_score + tfidf_weight * tfidf_score
        else:
            tfidf_score = None
            final_score = sbert_score

        # Explainability: matched and missing skills
        if job_skills is not None and resume_skills_list is not None:
            resume_skill_set = set(resume_skills_list[i])
            job_skill_set = set(job_skills)
            matched = sorted(job_skill_set & resume_skill_set)
            missing = sorted(job_skill_set - resume_skill_set)
        else:
            matched = []
            missing = []

        rows.append({
            "Resume":        name,
            "SBERT Score":   round(sbert_score, 4),
            "TF-IDF Score":  round(tfidf_score, 4) if tfidf_score is not None else None,
            "Final Score":   round(final_score, 4),
            "Matched Skills": matched,
            "Missing Skills": missing,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by="Final Score", ascending=False).reset_index(drop=True)
    df.index += 1  # rank starts at 1
    return df