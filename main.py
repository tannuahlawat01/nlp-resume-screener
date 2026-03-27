from src.extractor import process_folder
from src.preprocessing import TextPreprocessor
from src.ner_extractor import SkillExtractor
from src.embedding import SemanticMatcher
from src.ranking_engine import rank_resumes

if __name__ == "__main__":

    # ── Step 1: Extract text from all resumes in the data folder 
    print("=== Step 1: Extracting Resume Texts ===")
    resume_folder = "data/resumes"
    process_folder(resume_folder)

    # ── Step 2: Preprocessing demo 
    print("\n=== Step 2: Preprocessing Demo ===")
    sample = """
    Contact me at test@email.com
    Visit https://example.com
    I am developing machine learning applications.
    """
    processor = TextPreprocessor()
    result = processor.preprocess(sample)
    print("Processed text:")
    print(result)

    # ── Step 3: NER / Skill extraction demo 
    print("\n=== Step 3: Skill & Role Extraction Demo ===")
    ner_text = """
    Experienced Software Engineer skilled in Python, AWS, and Docker.
    """
    extractor = SkillExtractor()
    ner_result = extractor.extract_entities(ner_text)
    print(ner_result)

    # ── Step 4: Semantic similarity demo 
    print("\n=== Step 4: Semantic Similarity Demo ===")
    resume = """
    Python developer with AWS experience building ML systems.
    """
    job = """
    Looking for software engineer skilled in Python and cloud computing.
    """
    matcher = SemanticMatcher()
    score = matcher.similarity(resume, job)
    print("Semantic similarity score:")
    print(score)

    # ── Step 5: Resume ranking demo 
    job_description = "Python cloud engineer required"
    resumes = [
        "Python developer with AWS",
        "Java backend engineer",
        "Cloud engineer using Docker and AWS",
    ]
    resume_names = ["Resume_A", "Resume_B", "Resume_C"]

    job_embedding = matcher.model.encode(job_description)
    resume_embeddings = [matcher.model.encode(r) for r in resumes]

    ranking = rank_resumes(job_embedding, resume_embeddings, resume_names)
    print("\n=== Resume Ranking ===")
    print(ranking)


