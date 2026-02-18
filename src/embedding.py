from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
class SemanticMatcher:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text):
        return self.model.encode(text)

    def similarity(self, resume_text, job_text):
        resume_vec = self.embed(resume_text)
        job_vec = self.embed(job_text)
        score = cosine_similarity([resume_vec], [job_vec])[0][0]
        return score
