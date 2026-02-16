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
