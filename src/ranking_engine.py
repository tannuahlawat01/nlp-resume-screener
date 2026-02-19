import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
def rank_resumes(job_embedding, resume_embeddings, resume_names):
    """
    Rank resumes based on cosine similarity with job embedding
    """
    job_embedding = np.array(job_embedding).reshape(1, -1)
    resume_embeddings = np.array(resume_embeddings)
    similarities = cosine_similarity(
        job_embedding,
        resume_embeddings
    )[0]
    ranking_df = pd.DataFrame({
        "Resume": resume_names,
        "Similarity Score": similarities
    })
    ranking_df = ranking_df.sort_values(
        by="Similarity Score",
        ascending=False
    )
    return ranking_df
