from src.extractor import process_folder
if __name__ == "__main__":
    resume_folder = "data/resumes"
    process_folder(resume_folder)

from src.preprocessing import TextPreprocessor
if __name__ == "__main__":
    sample = """
    Contact me at test@email.com
    Visit https://example.com
    I am developing machine learning applications.
    """
    processor = TextPreprocessor()
    result = processor.preprocess(sample)
    print("Processed text:")
    print(result)

from src.ner_extractor import SkillExtractor
if __name__ == "__main__":
    text = """
    Experienced Software Engineer skilled in Python, AWS, and Docker.
    """

    extractor = SkillExtractor()
    result = extractor.extract_entities(text)

    print(result)

from src.embedding import SemanticMatcher
if __name__ == "__main__":
    resume = """
    Python developer with AWS experience building ML systems.
    """
    job = """
    Looking for software engineer skilled in Python and cloud computing.
    """
    matcher = SemanticMatcher()
    score = matcher.similarity(resume, job)
    print("\nSemantic similarity score:")
    print(score)


from src.embedding import SemanticMatcher
from src.ranking_engine import rank_resumes
if __name__ == "__main__":
    matcher = SemanticMatcher()
    job_description = "Python cloud engineer required"
    resumes = [
        "Python developer with AWS",
        "Java backend engineer",
        "Cloud engineer using Docker and AWS"
    ]
    resume_names = ["Resume_A", "Resume_B", "Resume_C"]
    job_embedding = matcher.model.encode(job_description)
    resume_embeddings = [matcher.model.encode(r) for r in resumes]
    ranking = rank_resumes(
        job_embedding,
        resume_embeddings,
        resume_names
    )
    print("\n=== Resume Ranking ===")
    print(ranking)

