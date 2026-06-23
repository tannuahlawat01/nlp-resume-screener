import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFMatcher:
    """
    TF-IDF + cosine similarity baseline matcher.

    Used as a comparison against SBERT to demonstrate that semantic
    matching captures meaning that keyword overlap misses.

    Example: a resume saying "I build cloud infrastructure on GCP"
    scores low against a JD asking for "AWS experience" under TF-IDF
    (different words), but SBERT recognises the semantic overlap.

    Usage:
        matcher = TFIDFMatcher()
        jd_vec, resume_vecs = matcher.fit_and_embed_all(jd_text, resume_texts)
        scores = [matcher.similarity(jd_vec, r) for r in resume_vecs]
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # unigrams + bigrams catches "machine learning", "deep learning"
            min_df=1,
            max_df=0.95,
            sublinear_tf=True    # log normalization — reduces dominance of high-freq terms
        )
        self._fitted = False

    def fit_and_embed_all(
        self,
        job_description: str,
        resume_texts: list
    ) -> tuple:
        """
        Fit the vectorizer on the full corpus (JD + all resumes), then
        return TF-IDF vectors for each.

        Why fit on the full corpus: the vectorizer must see all documents
        to build a shared vocabulary. Fitting only on the JD and then
        transforming resumes would produce misaligned vectors.

        Args:
            job_description: preprocessed JD text (mode='full')
            resume_texts: list of preprocessed resume texts (mode='full')

        Returns:
            (jd_vector, [resume_vector_1, resume_vector_2, ...])
        """
        corpus = [job_description] + resume_texts
        vectors = self.vectorizer.fit_transform(corpus).toarray()
        self._fitted = True

        jd_vec = vectors[0]
        resume_vecs = list(vectors[1:])
        return jd_vec, resume_vecs

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Cosine similarity between two TF-IDF vectors.

        Returns:
            Float in [0, 1]. Higher = more similar.
        """
        return float(cosine_similarity([vec_a], [vec_b])[0][0])